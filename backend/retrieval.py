import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank

# Adjusted import path if Indexer is now in indexing.py
from indexing import Indexer, ensure_pgvector_setup
from config import settings


logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Handles document retrieval, optionally with re-ranking."""

    def __init__(self, indexer: Indexer):
        # Ensure DB setup has potentially run (e.g., during app startup)
        # ensure_pgvector_setup() # Or rely on startup call in api.py
        self.base_retriever: BaseRetriever = indexer.get_retriever(k=settings.retriever_k)
        self.final_retriever: BaseRetriever = self._setup_retriever()

    def _setup_retriever(self) -> BaseRetriever:
        """Sets up the final retriever, potentially adding re-ranking."""
        # --- This logic remains the same, just ensure the base_retriever is from PGVector ---
        if settings.use_cohere_rerank:
            # ... (keep existing CohereRerank setup logic) ...
            # Ensure Cohere key check happens here or in config.py
            if not settings.cohere_api_key:
                logger.warning("Cohere API key missing. Disabling reranker.")
                return self.base_retriever

            try:
                compressor = CohereRerank(
                    # Removed explicit cohere_api_key as it should be in env now
                    model="rerank-english-v3.0",
                    top_n=settings.reranker_top_n
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.base_retriever
                )
                logger.info(f"Cohere Reranker initialized. Will return top {settings.reranker_top_n} documents.")
                return compression_retriever
            except Exception as e:
                logger.error(f"Failed to initialize Cohere Reranker: {e}. Falling back to base retriever.", exc_info=True)
                return self.base_retriever
        else:
            logger.info("Cohere Reranker is disabled. Using base PGVector retriever.")
            return self.base_retriever

    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieves relevant documents for a given query."""
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        try:
            results = self.final_retriever.invoke(query) # Use invoke for LCEL compatibility
            logger.info(f"Retrieved {len(results)} documents after processing/re-ranking.")
            # Ensure metadata needed for citation is present
            for doc in results:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = 'Unknown Source' # Add default if missing
            return results
        except Exception as e:
            logger.error(f"Error during document retrieval for query '{query[:50]}...': {e}", exc_info=True)
            return [] # Return empty list on failure