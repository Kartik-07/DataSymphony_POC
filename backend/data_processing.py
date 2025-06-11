import logging
from typing import Union, Dict, Any, Optional, List
import bs4
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings
import os

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads data from various sources."""

    @staticmethod
    def load_web_page(url: str) -> List[Document]:
        """Loads content from a single web page."""
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header", "content", "main") # Add common tags
                    )
                ),
            )
            loader.requests_per_second = 1 # Respectful scraping
            docs = loader.load()
            # Add URL as metadata if not present
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = url
            logger.info(f"Loaded content from {url}")
            return docs
        except Exception as e:
            logger.error(f"Error loading web page {url}: {e}", exc_info=True)
            return []

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]: # Renamed from load_pdf_plumber
        """Loads content from a PDF file using PDFPlumberLoader."""
        logger.info(f"Attempting to load PDF: {file_path} using PDFPlumber")
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return []
        try:
            # Using PDFPlumberLoader as per your original code
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            if not docs:
                 logger.warning(f"PDFPlumberLoader returned no documents for: {file_path}")

            # Add source metadata (using basename is fine for local files)
            file_name = os.path.basename(file_path)
            for doc in docs:
                # Ensure metadata dictionary exists before access
                if not hasattr(doc, 'metadata'):
                     doc.metadata = {}
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_name
                # Optional: Add full path?
                # doc.metadata['full_path'] = file_path

            logger.info(f"Loaded {len(docs)} document(s) from PDF: {file_name} using PDFPlumber")
            return docs
        except Exception as e:
            # Log the error with traceback for debugging
            logger.error(f"Error loading PDF {file_path} with PDFPlumber: {e}", exc_info=True)
            return [] # Return empty list to allow script to continue

    @staticmethod
    # In data_processing.py

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Loads content from a text file, trying UTF-8 then UTF-16."""
        logger.info(f"Attempting to load TXT: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Text file not found: {file_path}")
            return []

        docs = []
        try:
            # Try UTF-8 first
            logger.debug(f"Trying UTF-8 encoding for {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            logger.debug(f"Successfully loaded {file_path} with UTF-8")

        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try UTF-16
                logger.warning(f"UTF-8 decoding failed for {file_path}. Trying UTF-16 encoding.")
                loader = TextLoader(file_path, encoding='utf-16')
                docs = loader.load()
                logger.debug(f"Successfully loaded {file_path} with UTF-16")
            except Exception as e_utf16:
                # Catch errors during UTF-16 loading
                logger.error(f"Error loading text file {file_path} with UTF-16 fallback: {e_utf16}", exc_info=True)
                return [] # Failed to load with both encodings

        except Exception as e_utf8:
            # Catch other potential errors during UTF-8 loading (e.g., file permissions)
            logger.error(f"Error loading text file {file_path} with UTF-8: {e_utf8}", exc_info=True)
            return [] # Failed to load with UTF-8 for other reasons

        # --- Common processing after successful load ---
        if not docs:
            logger.warning(f"TextLoader returned no documents (file might be empty): {file_path}")
            # Return empty list if file loaded but was empty, allows script to continue normally
            return []

        file_name = os.path.basename(file_path)
        for doc in docs:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            if 'source' not in doc.metadata:
                doc.metadata['source'] = file_name

        logger.info(f"Loaded {len(docs)} document(s) from text file: {file_name}")
        return docs
    
    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """Loads content from a DOCX file using Docx2txtLoader."""
        logger.info(f"Attempting to load DOCX: {file_path} using Docx2txtLoader")
        if not os.path.exists(file_path):
            logger.error(f"DOCX file not found: {file_path}")
            return []
        try:
            # Using Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            if not docs:
                 logger.warning(f"Docx2txtLoader returned no documents for: {file_path}")

            # Add source metadata
            file_name = os.path.basename(file_path)
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                     doc.metadata = {}
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_name
                # doc.metadata['full_path'] = file_path # Optional

            logger.info(f"Loaded {len(docs)} document(s) from DOCX: {file_name} using Docx2txtLoader")
            return docs
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path} with Docx2txtLoader: {e}", exc_info=True)
            return []
        
    @staticmethod
    def load_csv(file_path: str, encoding: str = 'utf-8') -> List[Document]:
        """Loads data from a CSV file into Documents (one per row)."""
        logger.info(f"Attempting to load CSV: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return []
        docs = []
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            # Convert NaN/NaT to None for cleaner metadata
            df = df.where(pd.notnull(df), None)
            for index, row in df.iterrows():
                # Convert row to dictionary, ensuring keys are strings
                row_data = {str(k): v for k, v in row.to_dict().items()}
                docs.append(DataLoader._create_document_from_row(row_data, source=file_name, row_num=index + 1))
            logger.info(f"Loaded {len(docs)} rows as documents from CSV: {file_name}")
            return docs
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file is empty: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}", exc_info=True)
            return []

    @staticmethod
    def load_xlsx(file_path: str, sheet_name: Optional[Union[str, int]] = 0) -> List[Document]:
        """Loads data from an XLSX file sheet into Documents (one per row)."""
        logger.info(f"Attempting to load XLSX: {file_path} (Sheet: {sheet_name})")
        if not os.path.exists(file_path):
            logger.error(f"XLSX file not found: {file_path}")
            return []
        docs = []
        file_name = os.path.basename(file_path)
        source_id = f"{file_name} (Sheet: {sheet_name})" if sheet_name is not None else file_name
        try:
            # sheet_name=None loads all sheets as a Dict[sheet_name, DataFrame]
            # sheet_name=0 loads the first sheet
            # sheet_name='Sheet1' loads by name
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # If multiple sheets are loaded (sheet_name=None), process the first one for now
            # You could modify this to process all sheets or specify which one.
            if isinstance(df, dict):
                if not df:
                    logger.warning(f"No sheets found in XLSX file: {file_path}")
                    return []
                first_sheet_name = list(df.keys())[0]
                logger.warning(f"Multiple sheets found. Loading only the first sheet: '{first_sheet_name}'")
                df = df[first_sheet_name]
                source_id = f"{file_name} (Sheet: {first_sheet_name})"


            if df.empty:
                 logger.warning(f"Sheet '{sheet_name}' in {file_path} is empty.")
                 return []

            # Convert NaN/NaT to None for cleaner metadata
            df = df.where(pd.notnull(df), None)
            for index, row in df.iterrows():
                 # Convert row to dictionary, ensuring keys are strings
                row_data = {str(k): v for k, v in row.to_dict().items()}
                docs.append(DataLoader._create_document_from_row(row_data, source=source_id, row_num=index + 1))

            logger.info(f"Loaded {len(docs)} rows as documents from XLSX: {source_id}")
            return docs
        except FileNotFoundError: # Should be caught above, but as safety
             logger.error(f"XLSX file not found during pandas read: {file_path}")
             return []
        except Exception as e: # Catches errors like invalid sheet name, file corruption etc.
            logger.error(f"Error loading XLSX {file_path} (Sheet: {sheet_name}): {e}", exc_info=True)
            return []
    
        # Add methods for other data sources (CSV, JSON, Databases etc.)

# --- TextProcessor class remains the same ---
class TextProcessor:
    """Splits documents into chunks."""

    def __init__(self, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(f"Text splitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Splits a list of documents."""
        if not docs:
            logger.warning("No documents provided to split.")
            return []
        splits = self.text_splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(splits)} chunks.")
        # Add chunk ID for better traceability
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
        return splits