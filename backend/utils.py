# RAG_Project/MY_RAG/Backend/utils.py

# Standard library imports
import logging
import sys
import os # Keep os for basic path operations if needed elsewhere, though not strictly required now
from datetime import datetime # Keep datetime for logging format
from urllib.parse import urlparse

# Third-party imports
import psycopg2
from psycopg2 import pool
# Removed FastAPI imports: UploadFile, HTTPException
# Removed file reading imports: pandas, pdfplumber, docx2txt, openpyxl
# Removed other util imports: uuid, shutil, schedule, time, threading, timedelta, timezone, Path


# --- Logging Setup ---
def setup_logging(level=logging.INFO):
    """Sets up basic logging configuration."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Already configured
        return

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S' # Uses datetime internally
    )
    root_logger.setLevel(level)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    # Optional: File handler
    try:
        # Ensure log file path is correct relative to where the app runs
        # Example: Place it in the base project directory if running from there
        # log_file_path = Path(__file__).resolve().parent.parent / "rag_system.log" # Alternative if Path was kept
        log_file_path = "rag_system.log" # Assume runs from Backend or project root
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Use root logger since file handler failed
        logging.error(f"Failed to configure file logging: {e}", exc_info=True)

    logging.info("Logging configured.")

# Note: Call setup_logging() once during application startup (e.g., in main.py).


# --- Optional: Postgres Pool (only needed for DIRECT SQL outside PGVector/SQLAlchemy) ---
class PostgresConnectionPool:
    """Minimal wrapper around psycopg2 pool for direct connections if needed."""
    _pool = None

    @classmethod
    def initialize(cls, dsn: str, minconn: int = 1, maxconn: int = 5):
        """Initializes the connection pool."""
        if cls._pool is None:
            try:
                cls._pool = pool.SimpleConnectionPool(minconn, maxconn, dsn=dsn)
                logging.info(f"Initialized PostgreSQL connection pool (min={minconn}, max={maxconn}).")
            except Exception as e:
                logging.error(f"Failed to create PostgreSQL connection pool: {e}", exc_info=True)
                cls._pool = None # Ensure it's None on failure
        else:
             logging.warning("PostgreSQL connection pool already initialized.")
        return cls._pool is not None

    @classmethod
    def getconn(cls):
        """Gets a connection from the pool."""
        if cls._pool:
            try:
                return cls._pool.getconn()
            except Exception as e:
                 logging.error(f"Failed to get connection from pool: {e}", exc_info=True)
                 raise ConnectionError("Failed to get connection from pool.") from e
        logging.error("Attempted to get connection from uninitialized pool.")
        raise ConnectionError("Postgres connection pool is not initialized.")

    @classmethod
    def putconn(cls, conn):
        """Returns a connection to the pool."""
        if cls._pool and conn:
             try:
                cls._pool.putconn(conn)
             except Exception as e:
                 logging.error(f"Failed to return connection to pool: {e}", exc_info=True)
                 # Depending on the error, you might want to close the connection instead
                 # conn.close()

    @classmethod
    def closeall(cls):
        """Closes all connections in the pool."""
        if cls._pool:
             cls._pool.closeall()
             logging.info("Closed PostgreSQL connection pool.")
             cls._pool = None

# --- Helper to get DSN for psycopg2 from SQLAlchemy URL ---
def get_psycopg2_dsn(sqlalchemy_url: str) -> str:
    """Converts a SQLAlchemy URL (like postgresql://user:pass@host:port/db) to a psycopg2 DSN string."""
    try:
        parsed = urlparse(sqlalchemy_url)
        # Ensure required components are present
        if not all([
            parsed.scheme and parsed.scheme.startswith('postgres'),
            parsed.username,
            parsed.password,
            parsed.hostname,
            parsed.port,
            parsed.path and parsed.path[1:] # Check path exists and is not just '/'
        ]):
             # Log the problematic URL (mask password)
             masked_url = sqlalchemy_url.replace(parsed.password, "****") if parsed.password else sqlalchemy_url
             logging.error(f"Invalid SQLAlchemy URL format for DSN conversion: {masked_url}")
             raise ValueError("Invalid SQLAlchemy URL format for DSN conversion.")

        dsn = f"dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} host={parsed.hostname} port={parsed.port}"
        return dsn
    except Exception as e:
        # Log the problematic URL (mask password) during exception
        try:
            parsed_for_log = urlparse(sqlalchemy_url)
            masked_url = sqlalchemy_url.replace(parsed_for_log.password, "****") if parsed_for_log.password else sqlalchemy_url
        except:
            masked_url = "URL_Parsing_Failed"
        logging.error(f"Error converting SQLAlchemy URL '{masked_url}' to DSN: {e}", exc_info=True)
        raise ValueError(f"Could not parse SQLAlchemy URL for DSN: {e}") from e

