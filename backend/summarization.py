# RAG_Project/Backend/summarization.py

import json
import logging
import uuid
import os
import re
import warnings
from typing import Union, Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy # Ensure sqlalchemy is imported for inspect

# Attempt to import necessary LangChain and utility components
try:
    from config import settings
    from langchain_core.documents import Document
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from sql_processing import get_rag_db_utility, SQLDatabase # For DB table summarization
    from data_processing import DataLoader
except ImportError as e:
    print(f"Warning: Summarization - Required LangChain/Project components not found: {e}. Using dummy classes.")
    # Define dummy classes if imports fail (for basic structure)
    class DummySettings:
        light_llm_model_name = "gemini-1.5-flash"
        google_api_key = os.getenv("GOOGLE_API_KEY")
    settings = DummySettings()
    class Document: pass
    class ChatGoogleGenerativeAI: pass
    class HumanMessage: pass
    class SystemMessage: pass
    class StrOutputParser: pass
    class SQLDatabase: pass # Dummy SQLDatabase
    def get_rag_db_utility() -> Optional[SQLDatabase]: return None # Dummy getter
    class DataLoader: # Dummy DataLoader
        @staticmethod
        def load_pdf(path: str) -> List: return []
        @staticmethod
        def load_text(path: str) -> List: return []
        @staticmethod
        def load_docx(path: str) -> List: return []

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---
def clean_code_snippet(snippet: str) -> str:
    if not isinstance(snippet, str): return ""
    snippet = re.sub(r"```(?:[a-zA-Z0-9]*)?\s*(.*)\s*```", r"\1", snippet, flags=re.DOTALL)
    return snippet.strip()

def read_dataframe(path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    logger.debug(f"Reading DataFrame from: {path}")
    try:
        if path.lower().endswith(('.xlsx', '.xls')): return pd.read_excel(path)
        elif path.lower().endswith('.csv'): return pd.read_csv(path, encoding=encoding)
        else: logger.warning(f"Unsupported type for DataFrame reading: {path}"); return None
    except FileNotFoundError: logger.error(f"File not found: {path}"); return None
    except ImportError as ie: logger.error(f"Missing dependency for {path}: {ie}"); raise ie
    except Exception as e: logger.error(f"Read failed {path}: {e}"); return None

# --- LLM Prompts ---
STRUCTURED_ENRICHMENT_SYSTEM_PROMPT = """
You are an experienced data analyst that can annotate datasets based on their structure and sample values. Your instructions are as follows:
i) Review the dataset name (which corresponds to a database table name or filename) and column properties provided (including PK/FK info if available for DB tables at the table level).
ii) Generate a concise, accurate `dataset_description` (1-2 sentences) based on the dataset name and columns.
iii) For each field (`column`) in the `fields` list:
    - Generate an accurate `semantic_type` (a single common noun or concept, e.g., company, city, count, product_id, location, gender, latitude, url, category, timestamp, identifier, measurement, description, etc.) based on the column name, data type (`dtype`), and sample values (`samples`).
    - Generate a brief `description` explaining the field's likely content or purpose within the table/file.
iv) Respond ONLY with the updated JSON dictionary, ensuring all original fields are present. Do not include any preamble, explanation, or markdown formatting.
"""
UNSTRUCTURED_SUMMARY_SYSTEM_PROMPT = """
You are an expert text analyst. Your task is to analyze the provided text content and extract key information.
Instructions:
1. Read the text content carefully.
2. Generate a concise, natural language `summary` (3-5 sentences) of the main topics and purpose of the text.
3. Identify the main `domain` or `theme` of the document (e.g., 'Finance', 'Healthcare', 'Technology', 'Legal', 'News Report', 'Scientific Article').
4. Extract relevant `keywords` (provide a list of 5-10 important terms or concepts).
5. If discernible from the text, identify the `author` and relevant `timestamp` (like creation or publication date). If not found, use "N/A".
6. If the document has a clear hierarchical structure (e.g., chapters, sections, headings), list the main `headings`. If not applicable, use an empty list [].
7. Respond ONLY with a JSON object containing the following keys: "summary", "domain", "keywords", "author", "timestamp", "headings". Do not include any preamble, explanation, or markdown formatting.
"""

# === Summarizer Class ===
class DataSummarizer():
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.llm_model = settings.light_llm_model_name
        effective_api_key = api_key or settings.google_api_key
        self.text_gen: Optional[ChatGoogleGenerativeAI] = None
        if not effective_api_key: logger.warning(f"Google API Key missing. LLM calls ({self.llm_model}) may fail.")
        else:
            try: self.text_gen = ChatGoogleGenerativeAI(model=self.llm_model, temperature=0.15)
            except Exception as e: logger.error(f"Failed to init Summarizer LLM: {e}"); self.text_gen = None
        self.db_utility: Optional[SQLDatabase] = get_rag_db_utility()
        logger.info(f"Summarizer Initialized (LLM: {'OK' if self.text_gen else 'Failed'}, DB Util for RAG_DB: {'OK' if self.db_utility else 'Failed'})")


    def _check_type(self, dtype: Any, value: Any) -> Any:
        if pd.isna(value): return None
        try:
            if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
            elif isinstance(value, (np.integer)): return int(value)
            elif isinstance(value, (np.floating)): return float(value) if np.isfinite(value) else None
            elif isinstance(value, (np.bool_)): return bool(value)
            elif "float" in str(dtype): return float(value) if pd.notna(value) and np.isfinite(value) else None
            elif "int" in str(dtype): return int(value) if pd.notna(value) else None
            elif "bool" in str(dtype): return bool(value) if pd.notna(value) else None
            elif isinstance(value, (list, dict)): return value
            else: return str(value)
        except (ValueError, TypeError): return str(value)

    def _get_common_properties(self, series: pd.Series, n_samples: int) -> dict:
        properties = {}
        try:
            non_null_series = series.dropna(); properties["num_unique_values"] = int(non_null_series.nunique())
            non_null_values = non_null_series.unique(); actual_n_samples = min(n_samples, len(non_null_values))
            if actual_n_samples > 0:
                sampled_indices = np.random.choice(len(non_null_values), actual_n_samples, replace=False)
                samples = [non_null_values[i] for i in sampled_indices]
                properties["samples"] = [self._check_type(series.dtype, s) for s in samples]
            else: properties["samples"] = []
        except Exception as e: logger.warning(f"Error common props '{series.name}': {e}"); properties["num_unique_values"], properties["samples"] = 0, []
        properties["semantic_type"], properties["description"] = "", ""
        return properties

    def _get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
        properties_list = []
        if df.empty: return []
        for column in df.columns:
            col_name_str = str(column); logger.debug(f"Analyzing column: '{col_name_str}'")
            series = df[col_name_str]; dtype_orig = series.dtype
            properties = self._get_common_properties(series, n_samples)
            # --- REMOVED schema_info initialization per column ---
            # properties["schema_info"] = {"is_pk": False, "is_fk": False, "fk_references": None}

            numeric_stats = {"std": None, "min": None, "max": None, "mean": None, "median": None, "p25": None, "p75": None}
            dt_stats = {"min": None, "max": None}
            try:
                if pd.api.types.is_numeric_dtype(dtype_orig) and not pd.api.types.is_bool_dtype(dtype_orig):
                    properties["dtype"] = "number"; numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                    if not numeric_series.empty:
                        desc = numeric_series.describe(); numeric_stats.update({"std": self._check_type(dtype_orig, desc.get('std')), "min": self._check_type(dtype_orig, desc.get('min')), "max": self._check_type(dtype_orig, desc.get('max')), "mean": self._check_type(dtype_orig, desc.get('mean')), "median": self._check_type(dtype_orig, numeric_series.median()), "p25": self._check_type(dtype_orig, desc.get('25%')), "p75": self._check_type(dtype_orig, desc.get('75%'))})
                    properties.update(numeric_stats)
                elif pd.api.types.is_bool_dtype(dtype_orig): properties["dtype"] = "boolean"
                elif pd.api.types.is_datetime64_any_dtype(dtype_orig) or pd.api.types.is_timedelta64_dtype(dtype_orig):
                    properties["dtype"] = "datetime"; datetime_series = pd.to_datetime(series, errors='coerce').dropna()
                    if not datetime_series.empty: dt_stats.update({"min": self._check_type(dtype_orig, datetime_series.min()), "max": self._check_type(dtype_orig, datetime_series.max())})
                    properties.update(dt_stats)
                elif pd.api.types.is_categorical_dtype(series): properties["dtype"] = "category"; properties["samples"] = [str(s) for s in properties.get("samples", [])]
                elif dtype_orig == object or pd.api.types.is_string_dtype(dtype_orig):
                    try:
                        with warnings.catch_warnings(): warnings.simplefilter("ignore"); non_null_sample = series.dropna().sample(min(50, series.count()), random_state=42) if series.count() > 0 else pd.Series(dtype=object)
                        if not non_null_sample.empty and pd.to_datetime(non_null_sample, errors='coerce').notna().all():
                            properties["dtype"] = "datetime"; datetime_series = pd.to_datetime(series, errors='coerce').dropna()
                            if not datetime_series.empty: dt_stats.update({"min": self._check_type(dtype_orig, datetime_series.min()), "max": self._check_type(dtype_orig, datetime_series.max())})
                            properties.update(dt_stats)
                        else: raise ValueError("Not datetime")
                    except:
                        non_null_count = series.count(); unique_ratio = series.nunique() / non_null_count if non_null_count > 0 else 0
                        properties["dtype"] = "category" if unique_ratio < 0.6 else "string"
                    properties["samples"] = [str(s) for s in properties.get("samples", [])]
                else: properties["dtype"] = str(dtype_orig); properties["samples"] = [str(s) for s in properties.get("samples", [])]
            except Exception as stat_err: logger.warning(f"Stats/type error '{col_name_str}': {stat_err}"); properties.setdefault("dtype", str(dtype_orig))
            try: properties["missing_values_count"] = int(series.isnull().sum()); properties["missing_values_proportion"] = float(series.isnull().mean()) if len(series) > 0 else 0.0
            except Exception as common_stat_err: logger.warning(f"Common stats error '{col_name_str}': {common_stat_err}"); properties["missing_values_count"], properties["missing_values_proportion"] = -1, -1.0
            properties_list.append({"column": col_name_str, **properties}); logger.debug(f"Col '{col_name_str}' props: {properties}")
        return properties_list

    def _enrich_structured_summary(self, base_summary: dict) -> dict:
        if not self.text_gen: logger.warning("LLM unavailable. Skip enrichment."); base_summary.setdefault("metadata", {})["enrichment_status"] = "skipped_no_llm"; return base_summary
        identifier = base_summary.get('identifier', 'unknown_structured'); logger.info(f"Enriching structured: '{identifier}'")
        prompt_fields = []
        for f in base_summary.get("fields", []): # fields from _get_column_properties or columns_info_raw
            if f.get("column"):
                field_prompt_data = {
                    "column": f.get("column"),
                    "dtype": f.get("dtype"),
                    "samples": f.get("samples", [])[:3] # Samples are still useful for LLM
                }
                # schema_info (PK/FK per column) is no longer added here for the prompt,
                # as it's now at the table_schema_details level.
                # The LLM prompt for enrichment refers to "PK/FK info if available for DB tables at the table level".
                prompt_fields.append(field_prompt_data)

        prompt_input_summary = {
            "name": base_summary.get("name"),
            "identifier": identifier,
            "dataset_description": base_summary.get("dataset_description", ""),
            "fields": prompt_fields, # List of columns with their dtype and samples
            "table_schema_details": base_summary.get("table_schema_details") # Table-level PK, FK, comment
        }
        prompt_content = json.dumps(prompt_input_summary, indent=2, default=str)
        lc_messages = [SystemMessage(content=STRUCTURED_ENRICHMENT_SYSTEM_PROMPT), HumanMessage(content=f"Annotate:\n{prompt_content}")]
        response_text = ""
        try:
            response = self.text_gen.invoke(lc_messages); response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"LLM enrich response:\n{response_text}"); json_string = clean_code_snippet(response_text); enriched_data = json.loads(json_string)
            if not isinstance(enriched_data, dict): raise ValueError("LLM response not dict.")
            final_summary = base_summary.copy() # Start with base_summary which includes table_schema_details
            if enriched_data.get("dataset_description"): final_summary["dataset_description"] = enriched_data["dataset_description"]
            
            enriched_fields_map = {str(f.get("column")): f for f in enriched_data.get("fields", []) if f.get("column")}
            updated_fields = [] # This will be the new list for final_summary["fields"]
            
            for field_in_base in final_summary.get("fields", []): # Iterate over fields from base_summary
                # field_in_base already contains all original properties including dtype, samples, stats etc.
                # and importantly, the raw schema_info from inspector if it was a DB table.
                current_field_data = field_in_base.copy() # Work on a copy
                enriched_props_from_llm = enriched_fields_map.get(str(current_field_data["column"]))
                
                if enriched_props_from_llm:
                    if enriched_props_from_llm.get("semantic_type"):
                        current_field_data["semantic_type"] = enriched_props_from_llm["semantic_type"]
                    if enriched_props_from_llm.get("description"):
                        current_field_data["description"] = enriched_props_from_llm["description"]
                updated_fields.append(current_field_data)
            
            final_summary["fields"] = updated_fields
            final_summary.setdefault("metadata", {})["enrichment_status"] = "success"
            logger.info(f"Enriched structured: '{identifier}'.")
            return final_summary
        except Exception as e: error_msg = f"LLM enrich fail for {identifier}: {e}. Raw: '{response_text}'"; logger.error(error_msg, exc_info=True)
        base_summary.setdefault("metadata", {}); base_summary["metadata"]["enrichment_status"] = "failed"; base_summary["metadata"]["enrichment_error"] = error_msg
        return base_summary

    def _generate_llm_summary_from_content(self, content: str, identifier: str, doc_type: str) -> Dict[str, Any]:
        if not self.text_gen: logger.warning("LLM unavailable. Cannot gen summary."); return {"identifier": identifier, "document_type": doc_type, "summary": "LLM unavailable.", "domain": "N/A", "keywords": [], "author": "N/A", "timestamp": "N/A", "headings": [], "metadata": {"llm_status": "skipped_no_llm"}}
        logger.info(f"Generating LLM summary: {identifier} (len: {len(content)})")
        MAX_CONTENT_LENGTH = 750000
        content_to_send = (content[:MAX_CONTENT_LENGTH] + "\n[Truncated]") if len(content) > MAX_CONTENT_LENGTH else content
        if len(content) > MAX_CONTENT_LENGTH: logger.warning(f"Content truncated: {identifier}")
        lc_messages = [ SystemMessage(content=UNSTRUCTURED_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=f"Analyze:\n\n{content_to_send}") ]
        response_text = ""
        try:
            response = self.text_gen.invoke(lc_messages); response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"LLM unstructured summary response:\n{response_text}"); json_string = clean_code_snippet(response_text); llm_extracted_data = json.loads(json_string)
            required_keys = ["summary", "domain", "keywords", "author", "timestamp", "headings"]
            for key in required_keys: llm_extracted_data.setdefault(key, "N/A" if key in ["author", "timestamp", "domain", "summary"] else [])
            result = {"identifier": identifier, "document_type": doc_type, **llm_extracted_data, "metadata": {"llm_status": "success"}}
            logger.info(f"Generated unstructured summary: {identifier}.")
            return result
        except Exception as e: error_msg = f"LLM unstructured summary fail: {e}. Raw: '{response_text}'"; logger.error(error_msg)
        return {"identifier": identifier, "document_type": doc_type, "error": error_msg, "metadata": {"llm_status": "failed"}}

    def _format_output_json(self, summary_data: Dict[str, Any], data_type: str, source_type: str,
                            target_db: Optional[str] = None, target_table: Optional[str] = None,
                            original_file_path: Optional[str] = None) -> Dict[str, Any]:
        identifier = summary_data.get("identifier", "unknown_source")
        safe_identifier = re.sub(r'[^\w.\-]+', '_', str(identifier)); unique_suffix = str(uuid.uuid4())[:8]
        output_id = f"{data_type}-{safe_identifier}-{unique_suffix}"
        output = { "id": output_id, "document": "", "metadata": {} } # "document" is the main summary text
        current_time_utc = datetime.now(timezone.utc).isoformat()

        output["metadata"].update({
            "identifier": identifier, # Original name/ID (e.g., filename, table name)
            "data_type": data_type, # structured, unstructured, error
            "source_type": source_type, # file, database_table, dataframe
            "collection_time": current_time_utc
        })

        if original_file_path and source_type == "file":
            output["metadata"]["original_file_path"] = original_file_path
            # Ensure 'name' field in metadata (often used for display) reflects the original filename
            if 'name' not in output["metadata"]:
                output["metadata"]['name'] = os.path.basename(original_file_path)


        if data_type == "error":
            output["document"] = summary_data.get("error_message", f"Error processing: {identifier}.")
            output["metadata"]["error"] = summary_data.get("error", "Unknown error during processing")
        elif data_type == "structured":
            output["document"] = summary_data.get("dataset_description", f"Structured data summary for: {identifier}.")
            if not output["document"]: output["document"] = f"Data: {identifier}. Rows: {summary_data.get('row_count')}, Cols: {summary_data.get('column_count')}."
            output["metadata"].update({
                "row_count": summary_data.get("row_count"),
                "column_count": summary_data.get("column_count"),
                "columns": summary_data.get("fields", []), # List of column dictionaries
                "table_schema_details": summary_data.get("table_schema_details"), # Table-level PK/FK, comments
                "enrichment_status": summary_data.get("metadata", {}).get("enrichment_status", "not_applicable"),
                "enrichment_error": summary_data.get("metadata", {}).get("enrichment_error"),
                "target_database": target_db,
                "target_table_name": target_table
            })
            if output["metadata"]["enrichment_status"] in ["success", "not_applicable"]:
                output["metadata"].pop("enrichment_error", None)
            if not target_db: output["metadata"].pop("target_database", None)
            if not target_table: output["metadata"].pop("target_table_name", None)
            if not output["metadata"].get("table_schema_details"): output["metadata"].pop("table_schema_details", None)
        elif data_type == "unstructured":
            output["document"] = summary_data.get("summary", f"Unstructured content summary for: {identifier}.")
            if not output["document"] or output["document"] == "LLM unavailable.":
                 output["document"] = f"Content from: {identifier}. Type: {summary_data.get('document_type')}."
            output["metadata"].update({
                "document_type": summary_data.get("document_type"),
                "domain_themes": summary_data.get("domain"),
                "keywords": summary_data.get("keywords", []),
                "author": summary_data.get("author"),
                "timestamp": summary_data.get("timestamp"),
                "headings": summary_data.get("headings", []),
                "llm_status": summary_data.get("metadata", {}).get("llm_status")
            })
            if "error" in summary_data and output["metadata"].get("llm_status","").startswith("failed"):
                output["metadata"]["llm_error"] = summary_data.get("error")
        else:
            output["document"] = f"General summary for: {identifier}."
            output["metadata"]["error"] = "Unknown data type for summarization"
            output["metadata"]["data_type"] = "unknown"

        def make_serializable(obj):
            if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
            elif isinstance(obj, Path): return str(obj)
            elif pd.isna(obj) or obj is pd.NaT: return None # Ensure NaT is also None
            try: json.dumps(obj); return obj
            except TypeError: return str(obj)

        try:
            # Clean metadata before final output
            cleaned_metadata = {}
            for k, v_meta in output["metadata"].items():
                if isinstance(v_meta, list):
                    cleaned_metadata[k] = [make_serializable(item) for item in v_meta]
                elif isinstance(v_meta, dict):
                    cleaned_metadata[k] = {key_inner: make_serializable(val_inner) for key_inner, val_inner in v_meta.items()}
                else:
                    cleaned_metadata[k] = make_serializable(v_meta)
            output["metadata"] = {k_final: v_final for k_final, v_final in cleaned_metadata.items() if v_final is not None}
        except Exception as json_err:
            logger.error(f"JSON serialization error for metadata of '{identifier}': {json_err}")
            output["metadata"]["json_serialization_error"] = str(json_err)

        return output

    def summarize(
            self, data_input: Union[pd.DataFrame, str, None] = None, table_name: Optional[str] = None,
            file_name_override: Optional[str] = None, n_samples: int = 3, summary_method: str = "auto",
            encoding: str = 'utf-8', table_row_limit: Optional[int] = 1000, preloaded_content: Optional[str] = None,
            target_db: Optional[str] = None, target_table: Optional[str] = None,
            original_file_path: Optional[str] = None) -> dict:
        data_type = "unknown"; summary_result = {}; identifier = None; df_to_process = None
        effective_method = summary_method; source_type = "unknown"

        if table_name: # Processing a database table
            identifier = table_name; source_type = "database_table"; data_type = "structured"
            effective_method = 'llm' if summary_method in ['auto', 'llm'] else 'default'
            # target_db for DB tables is the DB they reside in.
            # If not passed, assume it's the default RAG_DB via self.db_utility
            db_being_queried = target_db if target_db else (self.db_utility._engine.url.database if self.db_utility and self.db_utility._engine else "RAG_DB")
            logger.info(f"Processing DB table: '{identifier}' (from DB '{db_being_queried}') method '{effective_method}'.")

            current_db_utility = self.db_utility # Assumes self.db_utility points to RAG_DB
            if target_db and target_db != "RAG_DB": # Logic for other DBs would be needed if self.db_utility is not generic
                logger.warning(f"Attempting to summarize table from DB '{target_db}'. Ensure self.db_utility is configured for it or this will fail if table not in RAG_DB.")
                # If you have multiple DB connections, you'd select the correct one here.

            if not current_db_utility or not current_db_utility._engine:
                summary_result = {"error": f"DB utility for '{db_being_queried}' missing or engine not initialized."}; data_type = "error"
            else:
                try:
                    engine = current_db_utility._engine
                    inspector = sqlalchemy.inspect(engine)
                    schema = None; actual_table_name_for_db = table_name
                    if '.' in table_name: schema, actual_table_name_for_db = table_name.split('.', 1)
                    
                    if not inspector.has_table(actual_table_name_for_db, schema=schema):
                        raise ValueError(f"Table '{table_name}' not found in DB '{db_being_queried}' with schema '{schema}'.")

                    columns_info_raw = inspector.get_columns(actual_table_name_for_db, schema=schema)
                    # Initialize column properties from schema, schema_info per column is removed here
                    # It will be part of table_schema_details
                    db_columns_properties = []
                    for c in columns_info_raw:
                        db_columns_properties.append({
                            "column": str(c['name']), "dtype": str(c['type']),
                            "nullable": c.get('nullable', True), "default": c.get('default'),
                            "autoincrement": c.get('autoincrement', 'auto'), "comment": c.get('comment')
                            # REMOVED: "schema_info": {"is_pk": False, "is_fk": False, "fk_references": None}
                        })
                    
                    table_schema_details = {"primary_keys": [], "foreign_keys": [], "table_comment": None}
                    try:
                        pk_constraint = inspector.get_pk_constraint(actual_table_name_for_db, schema=schema)
                        if pk_constraint and 'constrained_columns' in pk_constraint:
                            table_schema_details["primary_keys"] = pk_constraint['constrained_columns']
                        
                        fks = inspector.get_foreign_keys(actual_table_name_for_db, schema=schema)
                        for fk in fks:
                            table_schema_details["foreign_keys"].append({
                                "constrained_columns": fk['constrained_columns'], "referred_schema": fk.get('referred_schema'),
                                "referred_table": fk['referred_table'], "referred_columns": fk['referred_columns'],
                                "name": fk.get('name') })
                        if hasattr(inspector, 'get_table_comment'):
                             table_comment_result = inspector.get_table_comment(actual_table_name_for_db, schema=schema)
                             table_schema_details["table_comment"] = table_comment_result.get('text') if table_comment_result else None
                    except Exception as schema_detail_err:
                        logger.warning(f"Could not retrieve some schema details (PK/FK/Comment) for table '{table_name}': {schema_detail_err}")

                    row_count_query = sqlalchemy.text(f"SELECT COUNT(*) FROM {current_db_utility.dialect_specific_quote(table_name)}")
                    with engine.connect() as connection:
                        table_row_count = connection.execute(row_count_query).scalar_one_or_none() or 0

                    final_columns_list_for_summary = db_columns_properties # Base properties from schema
                    if table_row_count > 0:
                        quoted_table_name_for_data = getattr(current_db_utility, 'dialect_specific_quote', lambda x: f'"{x}"')(table_name)
                        sql_query = f'SELECT * FROM {quoted_table_name_for_data}'
                        if table_row_limit: sql_query += f" LIMIT {table_row_limit}"
                        with engine.connect() as connection:
                            df_to_process = pd.read_sql_query(sql=sqlalchemy.text(sql_query), con=connection)
                        if df_to_process is not None and not df_to_process.empty:
                            df_to_process.columns = [str(col) for col in df_to_process.columns]
                            data_driven_column_properties = self._get_column_properties(df_to_process, n_samples) # This gets samples, stats but no schema_info
                            # Merge: db_columns_properties has basic types, data_driven has samples/stats.
                            merged_cols = []
                            for db_col_prop in db_columns_properties:
                                merged_col_data = db_col_prop.copy()
                                for data_col_prop in data_driven_column_properties:
                                    if db_col_prop["column"] == data_col_prop["column"]:
                                        merged_col_data.update(data_col_prop) # Overwrite dtype with more specific one from data, add samples etc.
                                        break
                                merged_cols.append(merged_col_data)
                            final_columns_list_for_summary = merged_cols
                        else: # df_to_process is None or empty after query
                             logger.info(f"No data fetched for table '{table_name}' (limit {table_row_limit}), or table is effectively empty for sampling. Using schema-based column properties.")
                    else: # table_row_count is 0
                        logger.info(f"Table '{table_name}' is empty. Using schema-based column properties.")
                    
                    base_summary = {
                        "identifier": identifier, "name": identifier, "dataset_description": table_schema_details.get("table_comment",""),
                        "fields": final_columns_list_for_summary, 
                        "row_count": table_row_count, "column_count": len(db_columns_properties),
                        "table_schema_details": table_schema_details # Store PK/FK at table level
                    }
                    summary_result = self._enrich_structured_summary(base_summary) if effective_method == 'llm' and self.text_gen else base_summary
                    if not summary_result.get("dataset_description") and table_schema_details.get("table_comment"):
                         summary_result["dataset_description"] = table_schema_details["table_comment"]


                except AttributeError as ae: # Error with db_utility or inspector
                    logger.error(f"Attribute error accessing DB engine/inspector for table '{table_name}': {ae}", exc_info=True)
                    summary_result = {"error": f"DB attribute error: {ae}"}; data_type = "error"
                except Exception as db_err: # Other DB errors (table not found, query error)
                    logger.error(f"Table processing error for '{table_name}': {db_err}", exc_info=True)
                    summary_result = {"error": f"DB processing error: {db_err}"}; data_type = "error"
        
        elif preloaded_content is not None:
            source_type = "file"; identifier = file_name_override or f"preloaded_{str(uuid.uuid4())[:4]}"
            logger.info(f"Processing preloaded: '{identifier}'. Associated file path: {original_file_path}")
            if summary_method == 'auto': effective_method = 'unstructured'
            if effective_method == 'unstructured':
                data_type = "unstructured"
            elif effective_method in ['llm', 'default']: 
                data_type = "structured"; logger.warning(f"Attempting structured summary on preloaded: {identifier}.")
                try:
                    from io import StringIO
                    if '\n' in preloaded_content and (',' in preloaded_content.split('\n', 1)[0] or '\t' in preloaded_content.split('\n',1)[0]):
                        first_line = preloaded_content.split('\n', 1)[0]; sep = ',' if ',' in first_line else '\t'
                        df_to_process = pd.read_csv(StringIO(preloaded_content), sep=sep)
                        if df_to_process.empty: summary_result = {"identifier": identifier, "name": identifier, "dataset_description": "Empty preloaded (structured).", "fields": [], "row_count": 0, "column_count": 0}
                        else: df_to_process.columns = [str(col) for col in df_to_process.columns]
                    else: raise ValueError("Preloaded content not reliably detected as CSV/TSV-like for structured processing.")
                except Exception as parse_err: logger.error(f"Parse fail preloaded: {parse_err}"); summary_result = {"error": f"Parse fail preloaded: {parse_err}"}; data_type = "error"
            else: summary_result = {"error": f"Unsupported method '{effective_method}' for preloaded."}; data_type = "error"

        elif isinstance(data_input, str): # File path
            file_path_input = data_input
            source_type = "file"; identifier = file_name_override or os.path.basename(file_path_input)
            logger.info(f"Processing file path: '{identifier}' ({file_path_input}).")
            if not os.path.exists(file_path_input): summary_result = {"error": "File not found"}; data_type = "error"
            else:
                file_ext = os.path.splitext(identifier)[1].lower()
                if summary_method == 'auto': effective_method = 'llm' if file_ext in ['.csv', '.xlsx', '.xls', '.parquet'] else 'unstructured'
                if effective_method in ['llm', 'default'] and data_type != "error": 
                    data_type = "structured"
                    try:
                        df = read_dataframe(file_path_input, encoding=encoding)
                        if df is None: raise ValueError("read_dataframe returned None.")
                        elif df.empty: summary_result = {"identifier": identifier, "name": identifier, "dataset_description": "Empty file.", "fields": [], "row_count": 0, "column_count": 0}
                        else: df_to_process = df; df_to_process.columns = [str(col) for col in df_to_process.columns]
                    except Exception as load_err: logger.error(f"Load error file: {load_err}"); summary_result = {"error": f"Load fail file: {load_err}"}; data_type = "error"
                elif effective_method == 'unstructured' and data_type != "error":
                    data_type = "unstructured"
                elif data_type != "error":
                    summary_result = {"error": f"Invalid method '{effective_method}' for file."}; data_type = "error"

        elif isinstance(data_input, pd.DataFrame):
            identifier = file_name_override or "dataframe_input"; source_type = "dataframe"; data_type = "structured"
            logger.info(f"Processing DataFrame: '{identifier}'. Associated file path: {original_file_path}")
            if summary_method == 'unstructured': summary_result = {"error": "Unstructured summarization requires file path or preloaded text."}; data_type = "error"
            elif data_input.empty: summary_result = {"identifier": identifier, "name": identifier, "dataset_description": "Empty DataFrame.", "fields": [], "row_count": 0, "column_count": 0}
            else: df_to_process = data_input.copy(); df_to_process.columns = [str(col) for col in df_to_process.columns]; effective_method = 'llm' if summary_method in ['auto', 'llm'] else 'default'
        else:
            summary_result = {"error": "Invalid input type for summarization."}; data_type = "error"; identifier = file_name_override or "invalid_input"

        if data_type != "error" and summary_result.get("row_count", -1) != 0 :
            if data_type == "structured" and df_to_process is not None:
                logger.debug(f"Running structured summarization for: {identifier}")
                try:
                    column_properties = self._get_column_properties(df_to_process, n_samples)
                    base_summary = {"identifier": identifier, "name": identifier, "dataset_description": "",
                                    "fields": column_properties, "row_count": len(df_to_process),
                                    "column_count": len(df_to_process.columns)}
                    # For CSV/XLSX, table_schema_details will be None unless explicitly constructed from df.
                    # It's primarily for DB tables.
                    if effective_method == "llm" and self.text_gen:
                        summary_result = self._enrich_structured_summary(base_summary)
                    else:
                        summary_result = base_summary; summary_result.setdefault("metadata", {})["enrichment_status"] = "not_applicable"
                        if not summary_result.get("dataset_description"): summary_result["dataset_description"] = f"Basic statistical overview for {identifier}."
                except Exception as e: logger.error(f"Structured summarization failed for {identifier}: {e}", exc_info=True); summary_result = {"identifier": identifier, "error": f"Structured processing failed: {e}"}; data_type = "error"

            elif data_type == "unstructured":
                logger.debug(f"Running unstructured summarization for: {identifier}")
                doc_content_unstructured = None; doc_type_unstructured = 'unknown_unstructured'
                if preloaded_content is not None:
                    doc_content_unstructured = preloaded_content
                    doc_type_unstructured = os.path.splitext(identifier)[1].lower().strip('.') if identifier and '.' in identifier else 'preloaded_text'
                elif isinstance(data_input, str) and os.path.exists(data_input):
                    file_path_for_load = data_input
                    try:
                        file_ext_unstructured = os.path.splitext(identifier)[1].lower()
                        loader_func = None
                        if file_ext_unstructured == ".pdf": loader_func = DataLoader.load_pdf
                        elif file_ext_unstructured in [".txt", ".md", ".log", ".py", ".js", ".html", ".css", ".json"]: loader_func = DataLoader.load_text
                        elif file_ext_unstructured == ".docx": loader_func = DataLoader.load_docx
                        else: raise ValueError(f"Unsupported ext '{file_ext_unstructured}'")
                        loaded_docs_list = loader_func(file_path_for_load) if loader_func else []
                        if not loaded_docs_list: raise ValueError("DataLoader empty.")
                        doc_content_unstructured = "\n---\n".join([doc.page_content for doc in loaded_docs_list if hasattr(doc, 'page_content') and doc.page_content])
                        if not doc_content_unstructured: raise ValueError("No text extracted.")
                        doc_type_unstructured = file_ext_unstructured.strip('.')
                    except Exception as load_err: logger.error(f"Load fail {identifier} ({file_path_for_load}): {load_err}", exc_info=True); summary_result = {"identifier": identifier, "error": f"Load fail: {load_err}"}; data_type = "error"
                else: summary_result = {"identifier": identifier or "unknown_unstruc", "error": "Missing content source."}; data_type = "error"

                if data_type != "error" and doc_content_unstructured:
                    try:
                        summary_result = self._generate_llm_summary_from_content(content=doc_content_unstructured, identifier=identifier or "unknown_unstruc_content", doc_type=doc_type_unstructured)
                        if "error" in summary_result and summary_result.get("metadata", {}).get("llm_status") == "failed": data_type = "error"
                    except Exception as llm_err: logger.error(f"LLM unstruc summary fail {identifier}: {llm_err}", exc_info=True); summary_result = {"identifier": identifier or "unknown_unstruc_llm", "error": f"LLM summary fail: {llm_err}"}; data_type = "error"
                elif data_type != "error" and not doc_content_unstructured :
                     summary_result = {"identifier": identifier or "unknown_unstruc_nocontent", "error": "No content after load."}; data_type = "error"

        if identifier and "identifier" not in summary_result: summary_result["identifier"] = identifier
        elif not identifier and "identifier" not in summary_result: summary_result["identifier"] = "unknown_source_final"
        if "source_type" not in summary_result and source_type != "unknown": summary_result["source_type"] = source_type
        if not isinstance(summary_result, dict):
            logger.critical(f"Internal Error: summary_result not dict! Value: {summary_result}");
            summary_result = {"identifier": identifier or "unknown_fmt_crit", "error": "Internal format error.", "source_type": source_type}; data_type = "error"

        path_to_pass = original_file_path if original_file_path else (data_input if isinstance(data_input, str) and source_type == "file" else None)
        
        db_name_out = target_db
        table_name_out = target_table
        if source_type == "database_table":
            db_name_out = target_db or (self.db_utility._engine.url.database if self.db_utility and self.db_utility._engine else "RAG_DB")
            table_name_out = table_name 

        final_output = self._format_output_json(
            summary_data=summary_result, data_type=data_type, source_type=source_type,
            target_db=db_name_out, target_table=table_name_out,
            original_file_path=path_to_pass
        )
        logger.info(f"Summarization complete for: '{final_output.get('metadata', {}).get('identifier', 'unknown_id')}' (Path: {final_output.get('metadata', {}).get('original_file_path', 'N/A')}). ID: {final_output.get('id', 'no_id')}")
        return final_output

    def summarize_all_unstructured_text(self, all_text_content: str, collection_identifier: str) -> Dict[str, Any]:
        if not self.text_gen:
            logger.warning("LLM unavailable. Cannot generate overall unstructured summary.")
            return {
                "identifier": collection_identifier, "document_type": "text_collection_summary",
                "summary": "LLM unavailable for overall summary.", "domain": "N/A", "keywords": [],
                "author": "System", "timestamp": datetime.now(timezone.utc).isoformat(), "headings": [],
                "llm_status": "skipped_no_llm" # Moved metadata one level up for direct access
            }
        logger.info(f"Generating overall LLM summary for collection: {collection_identifier} (total content len: {len(all_text_content)})")
        MAX_CONTENT_LENGTH = 750000
        content_to_send = (all_text_content[:MAX_CONTENT_LENGTH] + "\n[--- CONTENT TRUNCATED ---]") if len(all_text_content) > MAX_CONTENT_LENGTH else all_text_content
        if len(all_text_content) > MAX_CONTENT_LENGTH: logger.warning(f"Overall unstr. content for '{collection_identifier}' truncated.")
        prompt_text = f"The following is a collection of texts from multiple documents. Please provide an overall analysis.\n\n{content_to_send}"
        lc_messages = [ SystemMessage(content=UNSTRUCTURED_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=f"Analyze collection:\n\n{prompt_text}") ]
        response_text = ""
        try:
            response = self.text_gen.invoke(lc_messages); response_text = response.content if hasattr(response, 'content') else str(response)
            json_string = clean_code_snippet(response_text); llm_extracted_data = json.loads(json_string)
            required_keys = ["summary", "domain", "keywords", "author", "timestamp", "headings"]
            for key in required_keys: llm_extracted_data.setdefault(key, "N/A" if key in ["author", "timestamp", "domain", "summary"] else [])
            llm_extracted_data["author"] = "System (Collection Summary)"; llm_extracted_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            result = {
                "identifier": collection_identifier, "document_type": "text_collection_summary",
                **llm_extracted_data, # Spread the LLM extracted data
                "llm_status": "success", "content_length_processed": len(content_to_send)
            }
            # REMOVED "metadata" sub-dictionary to elevate llm_status etc.
            # REMOVED "metadata_id" from here explicitly
            logger.info(f"Generated overall unstructured summary for '{collection_identifier}'.")
            return result
        except Exception as e:
            error_msg = f"LLM overall unstr. summary gen failed for '{collection_identifier}': {e}. Raw: '{response_text}'"
            logger.error(error_msg)
            return { "identifier": collection_identifier, "document_type": "text_collection_summary", "error": error_msg, "llm_status": "failed" }