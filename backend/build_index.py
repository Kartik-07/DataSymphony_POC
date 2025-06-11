# RAG_Project/Backend/build_index.py

import logging
import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path # Import Path
from datetime import datetime, timezone # Ensure timezone is imported

# Import necessary components using absolute paths
from config import settings
from utils import setup_logging
from data_processing import DataLoader, TextProcessor
from indexing import Indexer, ensure_pgvector_setup
from summarization import DataSummarizer
from sql_processing import get_rag_db_utility, SQLDatabase # SQLDatabase for type hint
from langchain_core.documents import Document
import pandas as pd
from sqlalchemy import create_engine

# --- Logging Setup ---
setup_logging(level=logging.INFO) # Set desired log level
logger = logging.getLogger(__name__)

# --- Define Project Root and Metadata Store Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_STORE_DIR = PROJECT_ROOT / "Metadata_Store"


# --- Helper function to clean filename for SQL table name ---
def clean_table_name(filename: str) -> str:
    """Cleans a filename to create a valid SQL table name."""
    name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'[^\w]+', '_', name_without_ext)
    if not re.match(r'^[a-zA-Z_]', cleaned_name):
        cleaned_name = '_' + cleaned_name
    cleaned_name = cleaned_name.lower()
    max_len = 63 # PostgreSQL default identifier limit
    return cleaned_name[:max_len]

# --- Main Indexing Function ---
def main():
    logger.info("--- Starting Data Loading, Metadata Generation, Chunking, and Indexing Process ---")

    try:
        METADATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Metadata store directory ensured at: {METADATA_STORE_DIR}")
    except Exception as e:
        logger.critical(f"Could not create metadata store directory {METADATA_STORE_DIR}: {e}. Aborting.", exc_info=True)
        return

    try:
        logger.info("Ensuring database (PGVector for RAG_DB) is set up...")
        ensure_pgvector_setup()
        logger.info("RAG_DB setup check complete.")
    except Exception as db_err:
        logger.critical(f"RAG_DB setup failed: {db_err}. Aborting.", exc_info=True)
        return

    uploads_engine = None
    try:
        logger.info(f"Creating SQLAlchemy engine for Uploads DB: {settings.postgres_uploads_url}")
        uploads_engine = create_engine(settings.postgres_uploads_url)
        with uploads_engine.connect() as connection:
            logger.info("Successfully connected to Uploads DB (RAG_DB_UPLOADS).")
    except Exception as engine_err:
        logger.error(f"Failed to create engine or connect to Uploads DB: {engine_err}. CSV/XLSX files will not be loaded into DB.", exc_info=True)

    db_utility: Optional[SQLDatabase] = None
    summarizer: Optional[DataSummarizer] = None
    indexer: Optional[Indexer] = None
    processor: Optional[TextProcessor] = None

    try:
        logger.info("Initializing DB Utility for RAG_DB...")
        db_utility = get_rag_db_utility()
        if not db_utility: logger.critical("Failed to initialize DB Utility for RAG_DB. Table processing will be skipped.")
        else: logger.info("RAG_DB utility obtained successfully.")

        logger.info("Initializing Indexer (for RAG_DB)...")
        indexer = Indexer()
        logger.info("Indexer initialized.")

        logger.info("Initializing DataSummarizer (for metadata)...")
        summarizer = DataSummarizer()
        logger.info(f"DataSummarizer initialized with model '{settings.light_llm_model_name}'.")

        logger.info("Initializing TextProcessor...")
        processor = TextProcessor(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        logger.info("TextProcessor initialized.")

    except Exception as init_err:
        logger.critical(f"Failed to initialize components: {init_err}. Aborting.", exc_info=True)
        return

    if not indexer or not summarizer or not processor:
        logger.critical("Essential components (Indexer, Summarizer, Processor) failed to initialize. Aborting.")
        return

    target_folder = os.getenv("INDEXING_TARGET_FOLDER", str(PROJECT_ROOT / "Document"))
    logger.info(f"Target document base folder: {target_folder}")

    if not os.path.isdir(target_folder):
        logger.error(f"Target folder not found: {target_folder}")
        return

    all_final_chunks: List[Document] = []
    metadata_docs_to_index: List[Document] = []

    structured_metadata_list: List[Dict[str, Any]] = []
    unstructured_metadata_list: List[Dict[str, Any]] = []
    all_unstructured_docs_content_for_summary: List[str] = []

    processed_files = 0; skipped_files = 0; unsupported_files = 0; db_uploads = 0; db_upload_failures = 0

    logger.info(f"Scanning folder and its subfolders for documents: {target_folder}")
    for root, _, files in os.walk(target_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.startswith('.') or filename.lower() in ["metadata.json", "structured_metadata.json", "unstructured_metadata.json", "unstructured_summary.json"]:
                logger.debug(f"Skipping item: {filename} in {root}"); continue

            logger.info(f"--- Processing file: {file_path} ---")
            file_ext = filename.lower().split('.')[-1]
            file_individual_metadata: Optional[Dict[str, Any]] = None

            try:
                if file_ext in ["pdf", "txt", "docx"]:
                    logger.info(f"Processing unstructured file type: .{file_ext}")
                    loaded_docs: List[Document] = []
                    if file_ext == "pdf": loaded_docs = DataLoader.load_pdf(file_path)
                    elif file_ext == "txt": loaded_docs = DataLoader.load_text(file_path)
                    elif file_ext == "docx": loaded_docs = DataLoader.load_docx(file_path)

                    if not loaded_docs:
                        logger.warning(f"No content loaded from: {filename}"); skipped_files += 1
                        unstructured_metadata_list.append({"id": f"error-{filename}-load_failed", "document": "Load failure or empty file.", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "error": "Load failure or empty file", "source_type": "file"}})
                        continue
                    
                    processed_files += 1
                    full_doc_content = "\n\n".join([doc.page_content for doc in loaded_docs if doc.page_content])
                    if not full_doc_content.strip():
                        logger.warning(f"No text content extracted for: {filename}"); skipped_files +=1; processed_files -=1
                        unstructured_metadata_list.append({"id": f"error-{filename}-no_content", "document": "No text content extracted.", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "error": "No text content extracted", "source_type": "file"}})
                        continue
                    
                    all_unstructured_docs_content_for_summary.append(full_doc_content)

                    file_individual_metadata = summarizer.summarize(preloaded_content=full_doc_content, summary_method='unstructured', file_name_override=filename, original_file_path=file_path)
                    doc_summary_for_chunks = "Summary generation failed or skipped."
                    if file_individual_metadata and file_individual_metadata.get("metadata", {}).get("data_type") != "error":
                        unstructured_metadata_list.append(file_individual_metadata)
                        doc_summary_for_chunks = file_individual_metadata.get("document", "Summary could not be extracted.")
                        logger.info(f"Generated individual summary for unstructured: {filename}")
                    else:
                        err_msg = file_individual_metadata.get('metadata', {}).get('error', 'Unknown error') if file_individual_metadata else 'Summarizer returned None'
                        logger.warning(f"Failed to generate individual summary for {filename}. Error: {err_msg}")
                        unstructured_metadata_list.append({"id": f"error-{filename}-unstruc_summ_failed", "document": f"Summarization failed: {err_msg}", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "error": f"Summarization failed: {err_msg}", "source_type": "file"}})

                    chunks: List[Document] = processor.split_documents(loaded_docs)
                    if chunks:
                        for chunk_index, chunk in enumerate(chunks):
                            if not hasattr(chunk, 'metadata') or chunk.metadata is None: chunk.metadata = {}
                            chunk.metadata['original_doc_summary'] = doc_summary_for_chunks
                            if 'source' not in chunk.metadata: chunk.metadata['source'] = filename
                            chunk.metadata['original_file_path'] = file_path
                            chunk.metadata['chunk_index_in_doc'] = chunk_index
                            all_final_chunks.append(chunk)
                        logger.info(f"Created {len(chunks)} chunks for: {filename}")
                    else: logger.warning(f"No chunks created for file {filename}.")

                elif file_ext in ["csv", "xlsx"]:
                    logger.info(f"Processing structured file type: .{file_ext}")
                    df: Optional[pd.DataFrame] = None
                    try:
                        df = pd.read_csv(file_path) if file_ext == "csv" else pd.read_excel(file_path, sheet_name=0) # Added encoding for CSV
                        logger.info(f"Loaded DataFrame from {filename}. Shape: {df.shape if df is not None else 'None'}")
                        processed_files += 1
                    except Exception as load_err:
                        logger.error(f"Failed to load structured file {filename}: {load_err}", exc_info=True); skipped_files += 1
                        structured_metadata_list.append({"id": f"error-{filename}-struct_load_failed", "document": f"Failed to load: {load_err}", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "source_type": "file", "error": f"Failed to load structured file: {load_err}"}})
                        continue

                    if df is None or df.empty:
                        logger.warning(f"Structured file is empty or loading failed: {filename}")
                        meta_entry = {"id": f"info-{filename}-empty_structured", "document": "Empty structured file.", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "structured", "source_type": "file", "row_count": 0, "column_count": 0, "columns": [], "enrichment_status": "not_applicable"}}
                        structured_metadata_list.append(meta_entry)
                        continue

                    table_name_cleaned = clean_table_name(filename)
                    if uploads_engine:
                        try:
                            df.to_sql(name=table_name_cleaned, con=uploads_engine, if_exists='replace', index=False, chunksize=1000)
                            logger.info(f"Successfully wrote '{filename}' to DB table '{table_name_cleaned}' in RAG_DB_UPLOADS.")
                            db_uploads += 1
                        except Exception as db_write_err:
                            logger.error(f"Failed to write DataFrame from '{filename}' to DB table '{table_name_cleaned}': {db_write_err}", exc_info=True); db_upload_failures += 1
                    else:
                        logger.warning(f"Skipping DB upload for '{filename}' as Uploads DB engine is not available.")

                    file_individual_metadata = summarizer.summarize(data_input=df, summary_method='llm', file_name_override=filename, original_file_path=file_path, target_db="RAG_DB_UPLOADS", target_table=table_name_cleaned)
                    if file_individual_metadata and file_individual_metadata.get("metadata", {}).get("data_type") != "error":
                        structured_metadata_list.append(file_individual_metadata)
                        logger.info(f"Generated metadata for structured file: {filename}")
                        metadata_doc = Document(
                            page_content=file_individual_metadata.get("document", f"Summary for {filename}"),
                            metadata=file_individual_metadata.get("metadata", {"identifier": filename, "full_path": file_path, "error": "Metadata missing"})
                        )
                        metadata_docs_to_index.append(metadata_doc)
                    else:
                        err_msg = file_individual_metadata.get('metadata',{}).get('error', 'Unknown error') if file_individual_metadata else 'Summarizer returned None'
                        logger.warning(f"Failed to generate metadata for structured file {filename}. Error: {err_msg}")
                        structured_metadata_list.append({"id": f"error-{filename}-struct_summ_failed", "document": f"Summarization failed: {err_msg}", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "source_type": "file", "error": f"Summarization failed: {err_msg}"}})
                else:
                    logger.warning(f"Skipping unsupported file type: {filename}"); unsupported_files += 1
                    unstructured_metadata_list.append({"id": f"error-{filename}-unsupported", "document": f"Unsupported file type '.{file_ext}'.", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "error": "Unsupported file type", "source_type": "file"}})
                    continue
            except Exception as e:
                logger.error(f"Failed processing {filename}: {e}", exc_info=True); skipped_files += 1
                error_meta = {"id": f"error-{filename}-processing_exception", "document": f"Processing exception: {e}", "metadata": {"identifier": filename, "full_path": file_path, "data_type": "error", "error": f"Processing exception: {e}", "source_type": "file"}}
                if file_ext in ["pdf", "txt", "docx"]: unstructured_metadata_list.append(error_meta)
                else: structured_metadata_list.append(error_meta)

    logger.info(f"--- File Processing Complete ---")
    logger.info(f"Processed: {processed_files}, DB Uploads: {db_uploads}, Skipped: {skipped_files}, Unsupported: {unsupported_files}, DB Upload Failures: {db_upload_failures}")

    if db_utility and summarizer.db_utility:
        logger.info("--- Processing RAG_DB Database Tables for Metadata ---")
        tables_to_exclude = {"langchain_pg_collection", "langchain_pg_embedding"}
        try:
            table_names = db_utility.get_usable_table_names()
            tables_to_process = [name for name in table_names if name not in tables_to_exclude and not name.startswith('pg_')]
            logger.info(f"Found {len(tables_to_process)} user tables in RAG_DB to process: {tables_to_process}")
            for table_name in tables_to_process:
                logger.info(f"--- Generating metadata for RAG_DB table: {table_name} ---")
                try:
                    table_metadata_dict = summarizer.summarize(table_name=table_name, summary_method='llm', target_db="RAG_DB")
                    if table_metadata_dict and table_metadata_dict.get("metadata", {}).get("data_type") != "error":
                        structured_metadata_list.append(table_metadata_dict)
                        logger.info(f"Successfully generated metadata for RAG_DB table: {table_name}")
                        metadata_doc = Document(
                            page_content=table_metadata_dict.get("document", f"Summary for RAG_DB table {table_name}"),
                            metadata=table_metadata_dict.get("metadata", {"identifier": table_name, "error": "Metadata missing"})
                        )
                        metadata_docs_to_index.append(metadata_doc)
                    else:
                        err_msg = table_metadata_dict.get('metadata',{}).get('error', 'Unknown error') if table_metadata_dict else 'Summarizer returned None'
                        logger.error(f"Failed to generate metadata for RAG_DB table {table_name}. Error: {err_msg}")
                        structured_metadata_list.append(table_metadata_dict or {"id": f"error-table-{table_name}-summ_failed", "document": "Table summarization failed.", "metadata": {"identifier": table_name, "data_type": "error", "source_type": "database_table", "error": err_msg, "target_database": "RAG_DB"}})
                except Exception as table_err:
                    logger.error(f"Exception while summarizing RAG_DB table {table_name}: {table_err}", exc_info=True)
                    structured_metadata_list.append({"id": f"error-table-{table_name}-exception", "document": f"Table summarization exception: {table_err}", "metadata": {"identifier": table_name, "data_type": "error", "source_type": "database_table", "error": str(table_err), "target_database": "RAG_DB"}})
        except Exception as db_list_err:
            logger.error(f"Failed to get table names from RAG_DB: {db_list_err}", exc_info=True)
    else:
        logger.warning("Skipping RAG_DB table metadata generation as DB utility or summarizer's DB utility is not available.")

    # --- Generate and Save Overall Unstructured Summary ---
    # This part is now adjusted to use the flat dictionary structure from summarizer.summarize_all_unstructured_text
    unstructured_summary_output_for_json = {
        "overall_summary": "No unstructured documents found or processed to generate an overall summary.",
        "domain_themes": "N/A",
        "keywords": [],
        "contributing_sources_count": 0,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_status": "not_attempted",
        "error_message": None
    }
    if all_unstructured_docs_content_for_summary:
        logger.info(f"--- Generating Overall Summary for {len(all_unstructured_docs_content_for_summary)} Unstructured Documents ---")
        concatenated_unstructured_text = "\n\n--- End of Document ---\n\n".join(all_unstructured_docs_content_for_summary)
        logger.info(f"Total length of concatenated unstructured text: {len(concatenated_unstructured_text)} characters.")
        try:
            overall_summary_result_dict = summarizer.summarize_all_unstructured_text(
                all_text_content=concatenated_unstructured_text,
                collection_identifier="all_unstructured_documents" # This identifier is for the summary process itself
            )
            # Populate unstructured_summary_output_for_json with relevant fields from overall_summary_result_dict
            if overall_summary_result_dict.get("llm_status") == "success":
                unstructured_summary_output_for_json["overall_summary"] = overall_summary_result_dict.get("summary", "Summary generation failed.")
                unstructured_summary_output_for_json["domain_themes"] = overall_summary_result_dict.get("domain", "N/A")
                unstructured_summary_output_for_json["keywords"] = overall_summary_result_dict.get("keywords", [])
                unstructured_summary_output_for_json["llm_status"] = "success"
                logger.info("Successfully generated overall summary for unstructured documents.")
            else:
                error_detail = overall_summary_result_dict.get('error', 'Unknown LLM summarization error')
                logger.error(f"Failed to generate overall unstructured summary: {error_detail}")
                unstructured_summary_output_for_json["error_message"] = error_detail
                unstructured_summary_output_for_json["llm_status"] = "failed"
            unstructured_summary_output_for_json["contributing_sources_count"] = len(all_unstructured_docs_content_for_summary)
            unstructured_summary_output_for_json["generation_timestamp"] = overall_summary_result_dict.get("timestamp", datetime.now(timezone.utc).isoformat())

        except Exception as e:
            logger.error(f"Exception during overall unstructured summary generation: {e}", exc_info=True)
            unstructured_summary_output_for_json["error_message"] = str(e)
            unstructured_summary_output_for_json["llm_status"] = "exception"
    
    unstructured_summary_file_path = METADATA_STORE_DIR / "unstructured_summary.json"
    try:
        with open(unstructured_summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(unstructured_summary_output_for_json, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully wrote overall unstructured summary to: {unstructured_summary_file_path}")
    except IOError as io_err:
        logger.error(f"Failed to write overall unstructured summary file: {io_err}", exc_info=True)


    structured_metadata_file_path = METADATA_STORE_DIR / "structured_metadata.json"
    unstructured_metadata_file_path = METADATA_STORE_DIR / "unstructured_metadata.json"
    try:
        with open(structured_metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(structured_metadata_list, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully wrote structured metadata for {len(structured_metadata_list)} items to: {structured_metadata_file_path}")
    except Exception as e:
        logger.error(f"Failed to write structured metadata file: {e}", exc_info=True)
    try:
        with open(unstructured_metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(unstructured_metadata_list, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully wrote unstructured metadata for {len(unstructured_metadata_list)} items to: {unstructured_metadata_file_path}")
    except Exception as e:
        logger.error(f"Failed to write unstructured metadata file: {e}", exc_info=True)

    combined_docs_for_pgvector_indexing = all_final_chunks + metadata_docs_to_index
    if not combined_docs_for_pgvector_indexing:
        logger.warning("No document chunks or metadata summaries prepared for PGVector indexing. Check logs.");
        logger.info("--- PGVector Indexing Process Skipped ---")
        return

    logger.info("Cleaning NUL bytes from document content before PGVector indexing...")
    cleaned_count = 0; nul_byte_sources = []
    for i, doc in enumerate(combined_docs_for_pgvector_indexing):
        try:
            if isinstance(doc.page_content, str) and '\x00' in doc.page_content:
                doc.page_content = doc.page_content.replace('\x00', '')
                cleaned_count += 1; source_info = doc.metadata.get('original_file_path', doc.metadata.get('source', f'doc_idx_{i}'))
                if source_info not in nul_byte_sources: nul_byte_sources.append(source_info)
            if isinstance(doc.metadata, dict):
                for key, value in doc.metadata.items():
                    if isinstance(value, str) and '\x00' in value:
                        doc.metadata[key] = value.replace('\x00', '')
        except Exception as clean_err: logger.error(f"Error cleaning NUL bytes in doc (Source: {doc.metadata.get('source', 'N/A')}): {clean_err}")
    if cleaned_count > 0: logger.info(f"Removed NUL bytes from {cleaned_count} documents. Sources affected (sample): {nul_byte_sources[:5]}")

    logger.info(f"Attempting to index {len(combined_docs_for_pgvector_indexing)} total documents into RAG_DB PGVector...")
    try:
        indexer.index_documents(combined_docs_for_pgvector_indexing)
        logger.info(f"--- PGVector Indexing Process Completed Successfully for {len(combined_docs_for_pgvector_indexing)} documents into RAG_DB ---")
    except Exception as e:
        logger.critical(f"--- PGVector Indexing Process Failed: {e} ---", exc_info=True)

if __name__ == "__main__":
    main()