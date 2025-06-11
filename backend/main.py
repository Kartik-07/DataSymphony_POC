import uvicorn
import logging
from utils import setup_logging # Import the setup function

if __name__ == "__main__":
    # Setup logging before starting the server
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RAG API server...")

    # Run the FastAPI server using Uvicorn
    uvicorn.run(
        "api:app",          # Path to the FastAPI app instance (filename:variable)
        host="0.0.0.0",     # Listen on all available network interfaces
        port=8000,          # Standard port for development
        reload=True,        # Enable auto-reload for development (remove in production)
        log_level="info"    # Uvicorn's log level
    )
    logger.info("RAG API server stopped.")