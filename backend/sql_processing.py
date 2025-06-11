# Backend/sql_processing.py
import logging
import re
from typing import Optional, Dict, Any # Import Dict, Any
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.sql_database.prompt import SQL_PROMPTS

from config import settings

logger = logging.getLogger(__name__)

# --- DB Utilities Initialization and Getters ---
# These establish the connections when the module loads.
# RAGPipeline will get these instances via the getter functions.
db_main: Optional[SQLDatabase] = None
db_uploads: Optional[SQLDatabase] = None
try:
    db_main = SQLDatabase.from_uri(settings.postgres_url, engine_args={"pool_size": 5, "max_overflow": 2})
    logger.info("SQLDatabase connection initialized for main RAG_DB.")
except Exception as e: logger.error(f"Failed to initialize SQLDatabase connection for RAG_DB: {e}", exc_info=True); db_main = None
try:
    db_uploads = SQLDatabase.from_uri(settings.postgres_uploads_url, engine_args={"pool_size": 5, "max_overflow": 2})
    logger.info("SQLDatabase connection initialized for uploads RAG_DB_UPLOADS.")
except Exception as e: logger.error(f"Failed to initialize SQLDatabase connection for RAG_DB_UPLOADS: {e}", exc_info=True); db_uploads = None

def get_rag_db_utility() -> Optional[SQLDatabase]: return db_main
def get_uploads_db_utility() -> Optional[SQLDatabase]: return db_uploads

# --- Helper functions ---
def clean_sql_string(sql: str) -> str:
    """Removes markdown fences and extraneous characters from SQL strings."""
    if not isinstance(sql, str): logger.error(f"Invalid input to clean_sql_string: expected string, got {type(sql)}"); return ""
    logger.debug(f"Attempting to clean SQL (Original): '{sql}'"); cleaned_sql = sql
    # Pattern to capture content within standard markdown code blocks (```sql ... ``` or ``` ... ```)
    pattern = r"```(?:[a-zA-Z0-9]*)?\s*(.*?)\s*```"; match = re.search(pattern, cleaned_sql, re.DOTALL)
    if match: cleaned_sql = match.group(1); logger.debug(f"Cleaned SQL (after regex extraction): '{cleaned_sql}'")
    else:
        # Fallback for cases without standard fences, attempting to remove common prefixes/suffixes
        logger.debug("Regex did not find standard markdown fences."); prefixes_to_remove = ["```sql", "```", "sql"]
        temp_stripped = cleaned_sql.strip(); removed_prefix = False
        # Check prefixes first
        for prefix in prefixes_to_remove:
            if temp_stripped.lower().startswith(prefix):
                 cleaned_sql = temp_stripped[len(prefix):].strip(); logger.debug(f"Cleaned SQL (after removing prefix '{prefix}'): '{cleaned_sql}'"); removed_prefix = True; break
        # If no prefix removed, check suffixes (like trailing ```)
        if not removed_prefix:
            if temp_stripped.endswith("```"): cleaned_sql = temp_stripped[:-3].strip()
            elif temp_stripped.endswith("`"): cleaned_sql = temp_stripped[:-1].strip() # Handle single backtick
            else: cleaned_sql = temp_stripped # No common prefix/suffix found
            logger.debug(f"Cleaned SQL (no prefix detected, basic strip/suffix removal): '{cleaned_sql}'")
        else: # If prefix was removed, apply strip again just in case
             cleaned_sql = cleaned_sql.strip()

    # Final cleanup: remove potential leading/trailing backticks and semicolons
    cleaned_sql = cleaned_sql.strip().strip('`').rstrip(';')
    logger.info(f"Final Cleaned SQL: '{cleaned_sql}'")
    return cleaned_sql

# Removed is_structured_query as it's less critical with metadata-based routing

# --- Function to format table info from metadata ---
# This function is needed again to process the table_metadata passed from rag_pipeline
def format_table_info_from_metadata(metadata: Dict[str, Any]) -> str:
    """Formats schema information for the LLM prompt using the metadata dictionary."""
    table_name = metadata.get("target_table_name", metadata.get("identifier", "UnknownTable"))
    columns = metadata.get("columns", [])

    if not columns:
        # If columns are missing in metadata, try to get basic info from identifier
        # This is a fallback, ideally metadata should always have columns for structured data
        logger.warning(f"Column metadata missing for '{table_name}'. Providing basic info.")
        return f"Table Name: {table_name}\nColumns: No detailed column information available in metadata."

    column_defs = []
    for col in columns:
        col_name = col.get("column", "UnknownColumn")
        col_type = col.get("dtype", "unknown")
        col_desc = col.get("description")
        # Ensure column name is quoted for the schema description sent to LLM
        col_def = f'- "{col_name}" (type: {col_type})'
        if col_desc:
            col_def += f" Description: {col_desc}"
        column_defs.append(col_def)

    table_desc = metadata.get("dataset_description", "")
    if not table_desc and table_name != "UnknownTable":
         table_desc = f"Contains data related to {table_name.replace('_', ' ')}."

    info_str = f'Table Name: "{table_name}"\n' # Quote table name
    if table_desc:
        info_str += f"Table Description: {table_desc}\n"
    info_str += "Columns:\n" + "\n".join(column_defs)

    logger.debug(f"Formatted table info from metadata:\n{info_str}")
    return info_str

# --- Generate SQL Query Function (Restored Signature and Logic) ---
def generate_sql_query(
    user_query: str,
    db_utility: SQLDatabase, # <-- Use passed utility for target DB context
    table_metadata: Dict[str, Any] # <-- Use passed metadata for target table schema
    ) -> str | None:
    """
    Generates the SQL query string using LLM, schema from provided metadata,
    and dialect from the db_utility. Does not execute the query.
    """
    if not db_utility:
        logger.error("Target SQLDatabase utility is not available for getting dialect.")
        return None
    if not table_metadata or not table_metadata.get("columns"):
        logger.error(f"Valid table metadata (with columns) not provided for query: '{user_query[:50]}...'")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model_name,
            temperature=0.0,
        )

        # --- Get dialect from the PASSED db_utility ---
        dialect = db_utility.dialect.lower() if hasattr(db_utility, 'dialect') else 'postgres' # Default if missing
        logger.debug(f"Using dialect: {dialect}")

        # --- Select prompt template (same logic as before) ---
        prompt_template = SQL_PROMPTS.get(dialect, None)
        if prompt_template is None:
             logger.warning(f"{dialect.capitalize()} prompt not found in SQL_PROMPTS. Using a generic fallback prompt.")
             GENERIC_SQL_TEMPLATE = """You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data as per {dialect}.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which table contains which column.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Question: {input}
SQLQuery:""" # <-- IMPORTANT: Use {input} here to match invoke call
             prompt_template = PromptTemplate.from_template(GENERIC_SQL_TEMPLATE)
        # --- End prompt template selection ---

        # --- Format table_info STRING FROM PASSED METADATA ---
        table_info_str = format_table_info_from_metadata(table_metadata)
        # ---

        # Create the chain
        sql_generation_chain = (
            RunnablePassthrough.assign(
                # Pass the necessary inputs dynamically using lambdas
                table_info=lambda _: table_info_str, # Use formatted string from metadata
                input=lambda x: x["input"],         # <<< Use "input" key here
                dialect=lambda _: dialect,
                top_k=lambda _: 10                   # Default top_k value
            )
            | prompt_template
            | llm
            | StrOutputParser()
        )

        logger.info(f"Generating SQL for query: '{user_query[:50]}...' using METADATA schema for table '{table_metadata.get('target_table_name', 'unknown')}'")

        # --- Invoke the chain using the "input" key ---
        raw_sql_output = sql_generation_chain.invoke({"input": user_query})

        logger.info(f"Raw SQL output from LLM: {raw_sql_output}")
        # Extract only the SQL query part if the LLM includes extra text
        # (Refined cleaning might be needed depending on LLM verbosity)
        # A simple approach if LLM follows the format strictly:
        if "SQLQuery:" in raw_sql_output:
             # Extract text after SQLQuery: and potentially before SQLResult:
             sql_part = raw_sql_output.split("SQLQuery:", 1)[-1]
             if "SQLResult:" in sql_part:
                 sql_part = sql_part.split("SQLResult:", 1)[0]
             cleaned_sql = clean_sql_string(sql_part.strip()) # Clean the extracted part
             if cleaned_sql: # Only return if cleaning resulted in non-empty SQL
                  logger.info(f"Extracted SQL from formatted output: {cleaned_sql}")
                  return cleaned_sql
             else:
                  logger.warning("Could not extract valid SQL from formatted LLM output. Falling back to full output.")
                  # Fallback to cleaning the whole raw output if extraction fails
                  return clean_sql_string(raw_sql_output)
        else:
             # If the format isn't strictly followed, just clean the whole output
             logger.warning("LLM output did not contain 'SQLQuery:'. Cleaning raw output.")
             return clean_sql_string(raw_sql_output)

    except Exception as e:
        logger.error(f"Failed to generate SQL query using metadata schema: {e}", exc_info=True)
        return None
# --- End Generate SQL Query Function ---