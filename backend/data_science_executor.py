# RAG_Project/MY_RAG/Backend/data_science_executor.py

import os
import logging
import requests
from typing import Dict, Any, Optional, Tuple

# Attempt to import settings from config.py (adjust path if necessary)
# This assumes you have a config.py file in the same directory or accessible via Python path
# where PYTHON_EXECUTOR_URL is defined.
try:
    # If data_science_executor.py is in the same directory as config.py
    from .config import settings
    EXECUTOR_URL = settings.python_executor_url
except (ImportError, AttributeError, NameError): # Added NameError just in case settings exists but not the attr
    # Fallback to environment variable if config.py or setting is not found
    EXECUTOR_URL = os.getenv("PYTHON_EXECUTOR_URL", "http://localhost:8081/execute") # Default for local dev

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_TIMEOUT_SECONDS = 70 # Slightly longer than the executor's internal timeout (60s)
HEADERS = {"Content-Type": "application/json"}

class DataScienceExecutorError(Exception):
    """Custom exception for errors related to the DataScienceExecutor."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

class DataScienceExecutor:
    """
    Client to interact with a remote Python code execution service.

    Handles sending code, managing requests, and processing responses
    including standard output, standard error, and plot artifacts.
    """

    def __init__(self, executor_url: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        """
        Initializes the DataScienceExecutor client.

        Args:
            executor_url (Optional[str]): The URL of the Python executor service.
                                           Defaults to the value from config/environment.
            timeout (int): Default timeout in seconds for requests to the executor.
        """
        self.executor_url = executor_url or EXECUTOR_URL
        self.timeout = timeout
        self._session = requests.Session() # Use a session for potential connection reuse
        self._session.headers.update(HEADERS)

        if not self.executor_url:
            logger.error("Python executor service URL is not configured. Set PYTHON_EXECUTOR_URL environment variable or configure in settings.")
            # Consider how your main app handles config errors. Raising might be appropriate.
            # For now, we just log and proceed, but execute_analysis will fail.
            # raise ValueError("Executor URL is not configured.") # Option to raise immediately

        logger.info(f"DataScienceExecutor initialized for URL: {self.executor_url}")

    def _parse_executor_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parses the JSON response from the executor service, handling potential errors."""
        try:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            json_response = response.json()

            # Validate expected keys are present, provide defaults if missing
            parsed = {
                "stdout": json_response.get("stdout", ""),
                "stderr": json_response.get("stderr", ""),
                "plot_png_base64": json_response.get("plot_png_base64", None),
                "execution_successful": json_response.get("execution_successful", False),
                "error": None # No client-side error if parsing succeeds
            }
            # If execution failed server-side but returned 200, log stderr
            if not parsed["execution_successful"] and parsed["stderr"]:
                 logger.warning(f"Code execution reported failure by executor. Stderr: {parsed['stderr'][:500]}...") # Log snippet
            elif not parsed["execution_successful"]:
                 logger.warning("Code execution reported failure by executor, but stderr was empty.")

            return parsed

        except requests.exceptions.HTTPError as http_err:
            # Handle specific HTTP errors (e.g., 408 Timeout from executor, 403 Forbidden)
            status_code = http_err.response.status_code
            error_detail = f"Executor returned HTTP {status_code}."
            details_dict = {"raw_response": http_err.response.text[:1000]} # Include raw response snippet
            try:
                # Try to parse error details if executor provided JSON
                error_json = http_err.response.json()
                if isinstance(error_json, dict) and 'detail' in error_json:
                    error_detail += f" Detail: {error_json['detail']}"
                    # If detail is a dict itself, merge it
                    if isinstance(error_json['detail'], dict):
                         details_dict.update(error_json['detail'])
                    else:
                         details_dict["server_detail"] = error_json['detail']
            except ValueError: # JSONDecodeError is a subclass
                logger.warning(f"Could not parse JSON error response from executor for status {status_code}.")

            logger.error(f"HTTP error calling executor: {http_err}")
            # Raise custom error to be handled by the calling code (e.g., rag_pipeline)
            raise DataScienceExecutorError(error_detail, status_code=status_code, details=details_dict) from http_err

        except ValueError as json_err: # Includes JSONDecodeError
            logger.error(f"Failed to decode JSON response from executor: {json_err}. Response text: {response.text[:500]}...")
            raise DataScienceExecutorError(
                "Invalid JSON response received from executor service.",
                details={"raw_response": response.text[:1000]}
            ) from json_err

    def execute_analysis(self, code_to_execute: str) -> Dict[str, Any]:
        """
        Sends Python code to the executor service and returns the result.

        Args:
            code_to_execute (str): The string containing Python code to execute.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'stdout': Standard output from the execution.
                - 'stderr': Standard error from the execution.
                - 'plot_png_base64': Base64 encoded PNG of the plot, if generated.
                - 'execution_successful': Boolean indicating if the code ran without error (exit code 0).
                - 'error': A string describing client-side errors (e.g., connection), None otherwise.
                           (Note: This 'error' key is mainly for client-side connection/timeout issues.
                            Server-side errors are primarily indicated by 'execution_successful': False
                            and content in 'stderr').

        Raises:
            DataScienceExecutorError: If communication with the executor fails critically (e.g., HTTP errors,
                                      invalid JSON response). Connection/Timeout errors are handled internally
                                      and returned in the dictionary structure.
        """
        if not self.executor_url:
             logger.error("Executor URL not configured.")
             # Return error dict matching successful structure but indicating client error
             return {
                 "stdout": "", "stderr": "Executor URL is not configured.", "plot_png_base64": None,
                 "execution_successful": False, "error": "Configuration Error"
             }

        payload = {"code": code_to_execute}
        logger.info(f"Sending code execution request to {self.executor_url}")
        logger.debug(f"Code snippet (first 200 chars): {code_to_execute[:200]}...")

        try:
            response = self._session.post(
                self.executor_url,
                json=payload,
                timeout=self.timeout
            )
            # Parsing logic now handles HTTP errors by raising DataScienceExecutorError
            return self._parse_executor_response(response)

        except requests.exceptions.Timeout:
            logger.error(f"Request to executor service timed out after {self.timeout} seconds.")
            return {
                "stdout": "", "stderr": f"Error: Request timed out after {self.timeout} seconds.",
                "plot_png_base64": None, "execution_successful": False,
                "error": "Request Timeout"
            }
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Could not connect to executor service at {self.executor_url}: {conn_err}")
            return {
                "stdout": "", "stderr": f"Error: Could not connect to executor at {self.executor_url}.",
                "plot_png_base64": None, "execution_successful": False,
                "error": "Connection Error"
            }
        # Catching the specific custom error raised by _parse_executor_response
        except DataScienceExecutorError as dse_err:
             error_message = str(dse_err)
             stderr_content = f"Error communicating with executor: {error_message}"
             # Append details if available
             if dse_err.details:
                 raw_resp = dse_err.details.get("raw_response", "")
                 if raw_resp: stderr_content += f"\nRaw Response Snippet: {raw_resp}"

             return {
                 "stdout": "", "stderr": stderr_content, "plot_png_base64": None,
                 "execution_successful": False, "error": f"Executor Client Error (HTTP {dse_err.status_code})" if dse_err.status_code else "Executor Client Error"
             }
        except Exception as e:
            # Catch any other unexpected exceptions during the request/parsing phase
            logger.exception(f"An unexpected error occurred while calling the executor service: {e}", exc_info=True)
            return {
                "stdout": "", "stderr": f"An unexpected client-side error occurred: {e}",
                "plot_png_base64": None, "execution_successful": False,
                "error": "Unexpected Client Error"
             }

    def ping(self) -> bool:
        """
        Simple check to see if the executor service is reachable.
        """
        if not self.executor_url: return False
        try:
            base_url = self.executor_url.replace("/execute", "")
            # Send a HEAD request to the base URL as a lightweight check
            response = self._session.head(base_url, timeout=5)
            # Consider any non-5xx status code as reachable
            return response.status_code < 500
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ping to executor service at {self.executor_url} failed: {e}")
            return False

    def close(self):
        """Closes the underlying requests session."""
        if self._session:
            self._session.close()
        logger.info("DataScienceExecutor session closed.")

    def __del__(self):
        # Ensure session is closed when the object is garbage collected
        self.close()

# --- No more test execution block (`if __name__ == '__main__':`) ---