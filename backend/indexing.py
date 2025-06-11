# RAG_Project/Backend/indexing.py

import logging
import sys
import re
from typing import List, Optional

import psycopg2 # Needed for direct check/setup
from psycopg2 import sql # Import sql module for safe query construction

# Use specific PGVector import if available and preferred
# from langchain_community.vectorstores import PGVector
# Use the newer import path if using more recent langchain versions
from langchain_postgres.vectorstores import PGVector

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever # Import BaseRetriever type

from config import settings
from utils import PostgresConnectionPool, get_psycopg2_dsn # Optional Pool import

logger = logging.getLogger(__name__)

def ensure_pgvector_setup():
    """
    Ensures the vector extension exists in the DB using a direct psycopg2 connection.
    PGVector Langchain integration handles table creation.
    """
    logger.info("Checking for PGVector extension...")
    conn = None # Initialize conn to None
    try:
        # Get a direct connection using psycopg2 DSN format
        dsn = get_psycopg2_dsn(settings.postgres_url)
        conn = psycopg2.connect(dsn) # Direct connection for setup
        conn.autocommit = True # Use autocommit for DDL commands

        with conn.cursor() as cur:
            # Use psycopg2.sql for safe query composition
            query = sql.SQL("CREATE EXTENSION IF NOT EXISTS {};").format(
                sql.Identifier('vector') # Safely quote 'vector' if needed
            )
            cur.execute(query)
            logger.info("PGVector extension check complete (created if not exists).")

    except psycopg2.Error as e:
        logger.error(f"Database error during PGVector setup check: {e}", exc_info=True)
        # Decide if this is fatal - maybe the DB isn't ready yet?
        # sys.exit("Exiting: Failed to ensure PGVector extension.") # Uncomment to make it fatal
        raise ConnectionError(f"Failed to connect/setup PGVector: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during PGVector setup check: {e}", exc_info=True)
        raise
    finally:
        if conn is not None:
             conn.close() # Close direct connection


class Indexer:
    """Handles embedding generation and PGVector interactions."""

    def __init__(self):
        """Initializes the embedding model and the PGVector client."""
        # --- Initialize Embedding Model ---
        try:
            # Consider adding device selection logic (e.g., based on CUDA availability)
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu' # Defaulting to CPU for broader compatibility
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True} # Normalize for cosine similarity
            )
            logger.info(f"Initialized HuggingFaceEmbeddings with model: {settings.embedding_model_name} on device: {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{settings.embedding_model_name}': {e}", exc_info=True)
            raise ValueError(f"Could not initialize embedding model: {e}") from e

        # --- Initialize PGVector Client ---
        try:
             # PGVector uses the connection string from settings.postgres_url
             # It automatically handles table creation based on collection_name if it doesn't exist.
             # The embedding dimension is inferred from the embeddings object.
             self.vector_store = PGVector(
                 connection=settings.postgres_url, # Use the SQLAlchemy-compatible URL
                 embeddings=self.embeddings,
                 collection_name=settings.collection_name,
                 use_jsonb=True # Store metadata in JSONB for better querying/filtering
             )
             logger.info(f"PGVector client initialized. Collection: {settings.collection_name}")
             # Optional: Add a check or method to explicitly create/verify the table schema if needed

        except ImportError as e:
             logger.error(f"Failed to import PGVector dependencies. Install 'langchain-postgres'. Error: {e}")
             raise
        except Exception as e:
             logger.error(f"Failed to initialize PGVector: {e}", exc_info=True)
             # Log connection string details carefully, obfuscating credentials
             safe_conn_string = re.sub(r':\/\/[^:]+:[^@]+@', r'://<user>:<password>@', settings.postgres_url)
             logger.error(f"Check POSTGRES_URL format and DB accessibility. Attempting connection to: {safe_conn_string}")
             raise ValueError(f"Could not initialize PGVector store: {e}") from e

    def index_documents(self, docs: List[Document], ids: Optional[List[str]] = None):
        """
        Embeds and stores documents in the PGVector collection.

        Args:
            docs: A list of LangChain Document objects to index.
            ids: Optional list of unique IDs for each document. If provided,
                 existing documents with the same IDs will be updated.
                 Must be the same length as docs.
        """
        if not docs:
            logger.warning("No documents provided for indexing.")
            return

        try:
            # PGVector's add_documents handles embedding and upsert logic if IDs are provided
            added_ids = self.vector_store.add_documents(docs, ids=ids)
            # Log count and maybe first few IDs for confirmation
            log_ids = added_ids[:5] if added_ids else []
            logger.info(f"Successfully added/updated {len(docs)} documents to PGVector collection '{settings.collection_name}'. IDs (sample): {log_ids}...")
        except Exception as e:
            logger.error(f"Failed to index documents into PGVector: {e}", exc_info=True)
            # Consider more specific error handling based on DB exceptions if needed
            raise # Re-raise the exception to be handled by the caller

    def delete_documents(self, ids: List[str]) -> bool:
        """
        Deletes documents from the vector store by their IDs.

        Args:
            ids: A list of document IDs to delete.

        Returns:
            True if deletion was attempted (regardless of whether IDs existed),
            False if no IDs were provided. Raises exception on DB error.
        """
        if not ids:
            logger.warning("No IDs provided for deletion.")
            return False
        try:
            # The delete method might vary slightly depending on the PGVector version/wrapper
            self.vector_store.delete(ids=ids)
            logger.info(f"Attempted deletion of {len(ids)} documents with IDs (sample: {ids[:5]}...) from collection '{settings.collection_name}'.")
            return True
        except NotImplementedError:
             logger.error("The current PGVector implementation does not support deletion by ID.")
             raise # Or handle differently depending on requirements
        except Exception as e:
             logger.error(f"Failed to delete documents with IDs {ids[:5]}...: {e}", exc_info=True)
             raise # Re-raise the exception

    def get_retriever(self, search_type: str = "similarity", k: Optional[int] = None) -> BaseRetriever:
        """
        Gets a retriever instance from the PGVector store.

        Args:
            search_type: Type of search ('similarity', 'similarity_score_threshold', 'mmr').
            k: The number of documents to retrieve. Defaults to settings.retriever_k.

        Returns:
            A LangChain BaseRetriever instance.
        """
        effective_k = k if k is not None else settings.retriever_k
        search_kwargs = {'k': effective_k}
        # Add threshold if using that search type
        # if search_type == "similarity_score_threshold":
        #     search_kwargs['score_threshold'] = 0.7 # Example threshold

        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            logger.info(f"PGVector retriever created with search_type={search_type}, k={effective_k}")
            return retriever
        except Exception as e:
            logger.error(f"Failed to create PGVector retriever: {e}", exc_info=True)
            raise ValueError(f"Could not create PGVector retriever: {e}") from e