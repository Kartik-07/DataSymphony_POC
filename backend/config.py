# RAG_Project/MY_RAG/Backend/config.py

import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file before defining settings
load_dotenv()
logger = logging.getLogger(__name__)

# --- Determine Base Directory ---
BASE_DIR = Path(__file__).parent.resolve()

class Settings(BaseSettings):
    # --- Core Settings ---
    # LangSmith
    langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_endpoint: str | None = os.getenv("LANGCHAIN_ENDPOINT")
    langchain_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")

    # LLM/Embedding Providers
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "default_google_key")
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")

    # --- Database ---
    # URL for Vector Store and main RAG DB operations
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:password@localhost:5432/RAG_DB")
    # URL for storing uploaded structured data (e.g., CSV/XLSX tables)
    postgres_uploads_url: str = os.getenv("POSTGRES_UPLOADS_URL", "postgresql+psycopg://postgres:password@localhost:5432/RAG_DB_UPLOADS")
    vector_store_driver: str = os.getenv("VECTOR_STORE_DRIVER", "psycopg2") # Often inferred, but can be set

    # --- Models & Behavior ---
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", 768)) # Defaulting to mpnet dimension
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro") # Main model for generation/SQL
    light_llm_model_name: str = os.getenv("LIGHT_LLM_MODEL_NAME", "gemini-1.5-flash") # LLM for Summarization, Titles, Suggestions
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_collection") # PGVector collection
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    retriever_k: int = int(os.getenv("RETRIEVER_K", 10)) # Initial K for retriever
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", 3)) # Final N after reranking
    use_cohere_rerank: bool = os.getenv("USE_COHERE_RERANK", "True").lower() == "true"

    # --- Data Science Executor --- ### ADDED SECTION ###
    python_executor_url: str = os.getenv("PYTHON_EXECUTOR_URL", "http://localhost:8081/execute")

    # --- Authentication and User History Paths ---
    auth_dir: Path = Path(os.getenv("AUTH_DIR", BASE_DIR / "Authentication"))
    user_history_base_dir: Path = Path(os.getenv("USER_HISTORY_BASE_DIR", BASE_DIR / "Chat_History"))

    # --- NEW: Security Settings (Example for JWT) ---
    secret_key: str = os.getenv("SECRET_KEY", "a_very_secret_key_change_this") # CHANGE THIS IN .env
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # --- NEW: History Management Settings ---
    history_max_turns: int = int(os.getenv("HISTORY_MAX_TURNS", 5)) # Max user+AI turns in active context window
    # Optional: Define a separate model for history summary. If empty, defaults to light_llm_model_name.
    history_summary_llm_model_name: str = os.getenv("LIGHT_LLM_MODEL_NAME", "gemini-1.5-flash")

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra environment variables

# Instantiate settings
settings = Settings()

# --- Environment Variable Setup & Validation ---
if not settings.google_api_key or settings.google_api_key == "default_google_key":
    # Make Google API Key strictly required
    raise ValueError("GOOGLE_API_KEY must be set in the .env file or environment variables.")

if "dummy" in settings.postgres_url or not settings.postgres_url:
     logger.warning("POSTGRES_URL is not properly configured or uses a dummy value.")
     # Decide if this should be fatal
     # raise ValueError("POSTGRES_URL must be configured.")

default_uploads_url = "postgresql+psycopg://postgres:password@localhost:5432/RAG_DB_UPLOADS"
if not settings.postgres_uploads_url or settings.postgres_uploads_url == default_uploads_url:
     logger.warning(f"POSTGRES_UPLOADS_URL not explicitly set or uses the default. Ensure correct configuration if uploading structured data.")

if settings.use_cohere_rerank and not settings.cohere_api_key:
    logger.warning("USE_COHERE_RERANK is True but COHERE_API_KEY is not set. Reranking will be disabled.")
    settings.use_cohere_rerank = False # Disable if key is missing

# --- Check Executor URL --- ### ADDED CHECK ###
if "localhost" in settings.python_executor_url:
    logger.warning(f"PYTHON_EXECUTOR_URL is set to a localhost address ({settings.python_executor_url}). Ensure this is correct for your deployment environment.")

# Set environment variables for LangChain/SDKs that might read them directly
os.environ['GOOGLE_API_KEY'] = settings.google_api_key
if settings.langchain_api_key:
    os.environ['LANGCHAIN_TRACING_V2'] = settings.langchain_tracing_v2
    os.environ['LANGCHAIN_ENDPOINT'] = str(settings.langchain_endpoint)
    os.environ['LANGCHAIN_API_KEY'] = settings.langchain_api_key
if settings.cohere_api_key and settings.use_cohere_rerank:
     os.environ['COHERE_API_KEY'] = settings.cohere_api_key

# Auth and User History Directories
try:
    settings.auth_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Authentication directory checked/created at: {settings.auth_dir}")
except Exception as e:
    logger.error(f"Failed to create auth directory '{settings.auth_dir}': {e}", exc_info=True)

try:
    settings.user_history_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"User chat history base directory checked/created at: {settings.user_history_base_dir}")
except Exception as e:
    logger.error(f"Failed to create user chat history base directory '{settings.user_history_base_dir}': {e}", exc_info=True)

# --- Check SECRET_KEY ---
if settings.secret_key == "a_very_secret_key_change_this":
    logger.warning("SECURITY WARNING: SECRET_KEY is set to the default value. Please change it in your .env file.")

# --- Determine effective summary LLM ---
# Use the specific history summary model if set, otherwise fall back to the light LLM model
effective_summary_model = settings.history_summary_llm_model_name or settings.light_llm_model_name
# Ensure light model is defined if summary model is empty
if not effective_summary_model and settings.light_llm_model_name:
    effective_summary_model = settings.light_llm_model_name
    logger.info("HISTORY_SUMMARY_LLM_MODEL_NAME not set, using LIGHT_LLM_MODEL_NAME for summaries.")
elif not effective_summary_model:
    logger.warning("Neither HISTORY_SUMMARY_LLM_MODEL_NAME nor LIGHT_LLM_MODEL_NAME are set. History summarization may be disabled.")


# Log key settings on startup (Updated Log Format)
logger.info(
    f"Loaded settings: Main LLM={settings.llm_model_name}, "
    f"Light LLM={settings.light_llm_model_name}, "
    f"Summary LLM={effective_summary_model or 'Not Set'}, " # Show effective summary model
    f"Embedding={settings.embedding_model_name}"
)
logger.info(f"Vector Store DB URL (POSTGRES_URL): {settings.postgres_url}")
logger.info(f"Uploads DB URL (POSTGRES_UPLOADS_URL): {settings.postgres_uploads_url}")
logger.info(f"Vector Store Collection: {settings.collection_name}, Reranker Enabled: {settings.use_cohere_rerank}")
logger.info(f"Chat History Directory: {settings.user_history_base_dir.resolve()}") # Log the absolute path
logger.info(f"Python Executor URL: {settings.python_executor_url}")
logger.info(f"History Management: Max Turns={settings.history_max_turns}") # Log history setting