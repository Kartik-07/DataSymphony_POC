# RAG_Project/MY_RAG/Backend/session_handler.py

import logging
import uuid
import shutil
import schedule
import time
import threading
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

# Third-party imports
from fastapi import UploadFile, HTTPException
import pandas as pd
import pdfplumber
import docx2txt
import openpyxl # Needed by pandas for xlsx

# Langchain/LLM imports (handle gracefully if not available)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
except ImportError:
    logging.warning("LangChain components for summarization not found. Summarization will be disabled.")
    ChatGoogleGenerativeAI = None
    HumanMessage = None
    SystemMessage = None
    StrOutputParser = None
    PromptTemplate = None

# Local imports
from config import settings # Assuming settings has history_max_turns etc.
from models import ChatMessage # Assuming ChatMessage is defined in models.py

logger = logging.getLogger(__name__)

# --- Configuration ---
# Determine project root relative to this file (session_handler.py)
# Assumes session_handler.py is in Backend directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf", ".csv", ".xlsx"}
TEMP_FILE_LIFESPAN_HOURS = 24
HISTORY_MAX_TURNS = getattr(settings, 'history_max_turns', 5) # Default to 5 turns
HISTORY_SUMMARIZATION_THRESHOLD = HISTORY_MAX_TURNS * 2 # Summarize when messages exceed this
# Token limit threshold (conceptual - implementation requires tokenizer)
# HISTORY_TOKEN_LIMIT = getattr(settings, 'history_token_limit', 3000)

# Ensure the temp directory exists on module load
try:
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary upload directory ensured at: {TEMP_UPLOAD_DIR}")
except Exception as e:
    logging.error(f"Failed to create temporary upload directory {TEMP_UPLOAD_DIR}: {e}", exc_info=True)

# --- Summarization Prompt ---
# (Define it here or import from prompts.py if moved there)
HISTORY_SUMMARIZATION_PROMPT_TEMPLATE = """
You are an expert conversation summarizer. Given the 'Existing Summary' (if any) and the 'Recent Conversation History', create a concise, updated summary.
The summary should capture the key topics, decisions, and unresolved questions from the *entire* conversation, integrating the recent history with the existing summary smoothly.
Focus on information relevant for maintaining context in future interactions. Keep the summary under 150 words.

Existing Summary:
---
{existing_summary}
---

Recent Conversation History (Oldest to Newest):
---
{conversation_history}
---

Updated Concise Summary (Max 150 words):
"""
HISTORY_SUMMARIZATION_PROMPT = PromptTemplate.from_template(HISTORY_SUMMARIZATION_PROMPT_TEMPLATE) if PromptTemplate else None

# --- Session Handler Class ---
class SessionHandler:
    """
    Manages conversational history, including turn limits and summarization.
    Also includes temporary file handling logic.
    """
    def __init__(self):
        self.summarization_llm = None
        self.summarization_chain = None
        if ChatGoogleGenerativeAI and SystemMessage and HumanMessage and StrOutputParser and HISTORY_SUMMARIZATION_PROMPT:
            try:
                summary_model_name = getattr(settings, 'history_summary_llm_model_name', settings.light_llm_model_name)
                self.summarization_llm = ChatGoogleGenerativeAI(
                    model=summary_model_name,
                    temperature=0.2 # Slightly creative for summarization
                )
                self.summarization_chain = (
                    HISTORY_SUMMARIZATION_PROMPT
                    | self.summarization_llm
                    | StrOutputParser()
                )
                logger.info(f"SessionHandler initialized summarization LLM: {summary_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize summarization LLM: {e}", exc_info=True)
                self.summarization_llm = None
                self.summarization_chain = None
        else:
            logger.warning("Summarization dependencies not met. Summarization disabled.")

    def _summarize_history(self, history_to_summarize: List[ChatMessage], existing_summary: Optional[str]) -> Optional[str]:
        """Uses the LLM to summarize older parts of the conversation."""
        if not self.summarization_chain or not history_to_summarize:
            logger.debug("Summarization skipped (no chain or no history to summarize).")
            return existing_summary # Return existing summary if cannot summarize

        logger.info(f"Attempting to summarize {len(history_to_summarize)} messages...")

        # Format history for the prompt
        formatted_history = []
        for msg in history_to_summarize:
            sender = "User" if msg.sender == "user" else "AI"
            formatted_history.append(f"{sender}: {msg.text}")
        history_str = "\n".join(formatted_history)

        summary_input = {
            "existing_summary": existing_summary or "None",
            "conversation_history": history_str
        }

        try:
            new_summary = self.summarization_chain.invoke(summary_input)
            logger.info(f"Successfully generated new summary (length {len(new_summary)}).")
            # Basic validation/cleanup
            if not isinstance(new_summary, str) or not new_summary.strip():
                 logger.warning("Summarization LLM returned empty or invalid summary.")
                 return existing_summary # Return old summary on failure
            return new_summary.strip()
        except Exception as e:
            logger.error(f"Error during history summarization LLM call: {e}", exc_info=True)
            return existing_summary # Return existing summary on error

    def manage_history(self, current_messages: List[ChatMessage], existing_summary: Optional[str]) -> Tuple[str, Optional[str]]:
        """
        Manages the conversation history based on turn limits.

        1.  Determines if summarization is needed (based on message count).
        2.  If needed, calls the summarization LLM.
        3.  Constructs the history context string for the *next* LLM call (summary + last N turns).
        4.  Returns the context string and the *new* summary (if generated).

        Args:
            current_messages: The full list of ChatMessage objects so far.
            existing_summary: The most recent summary string, if any.

        Returns:
            A tuple containing:
            - history_context_for_llm (str): Formatted string including summary (if any) and last N turns.
            - new_summary (Optional[str]): The newly generated summary string, or None if no new summary was created.
        """
        needs_summarization = len(current_messages) > HISTORY_SUMMARIZATION_THRESHOLD
        new_summary = None

        if needs_summarization:
            # Summarize messages *before* the last N turns
            summarization_point = len(current_messages) - (HISTORY_MAX_TURNS * 2)
            history_to_summarize = current_messages[:summarization_point]
            new_summary = self._summarize_history(history_to_summarize, existing_summary)
            # Use the *new* summary going forward
            current_summary_for_context = new_summary
        else:
            # Use the existing summary if not re-summarizing this turn
            current_summary_for_context = existing_summary

        # Get the last N turns (or fewer if history is short)
        last_n_messages = current_messages[-(HISTORY_MAX_TURNS * 2):]

        # Format the final context string for the LLM
        history_context_parts = []
        if current_summary_for_context:
            history_context_parts.append("Conversation Summary:")
            history_context_parts.append(current_summary_for_context)
            history_context_parts.append("\nRecent History (Oldest to Newest):")
        elif last_n_messages:
             history_context_parts.append("Recent History (Oldest to Newest):")
        else:
            history_context_parts.append("No previous conversation history.")


        if last_n_messages:
            for msg in last_n_messages:
                sender = "User" if msg.sender == "user" else "AI"
                history_context_parts.append(f"{sender}: {msg.text}")

        history_context_for_llm = "\n".join(history_context_parts)

        # Return the context string and the potentially updated summary
        # If summarization happened, new_summary will be the updated summary string.
        # If not, new_summary will be None.
        return history_context_for_llm, new_summary

    # --- Temporary File Handling Functions (Moved from utils.py) ---

    def save_temporary_file(self, file: UploadFile) -> str:
        """
        Saves an uploaded file temporarily with a unique name and returns its unique ID (filename).
        Validates file extension and handles potential saving errors.
        """
        if not file.filename:
            logging.warning("Upload attempt with empty filename.")
            raise HTTPException(status_code=400, detail="Filename cannot be empty.")

        _, extension = os.path.splitext(file.filename)
        if extension.lower() not in ALLOWED_EXTENSIONS:
            logging.warning(f"Upload attempt with disallowed file type: {extension}")
            raise HTTPException(status_code=400, detail=f"File type {extension} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

        safe_original_stem = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in Path(file.filename).stem)
        unique_filename = f"{uuid.uuid4()}_{safe_original_stem}{extension}"
        file_path = TEMP_UPLOAD_DIR / unique_filename

        try:
            logging.info(f"Attempting to save temporary file: {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logging.info(f"Successfully saved temporary file: {unique_filename}")
        except Exception as e:
            logging.error(f"Could not save temporary file {unique_filename}: {e}", exc_info=True)
            if file_path.exists():
                try: file_path.unlink()
                except OSError as unlink_err: logging.error(f"Failed to clean up partial file {file_path}: {unlink_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        finally:
            try: file.file.close()
            except Exception as close_err: logging.warning(f"Error closing uploaded file stream: {close_err}", exc_info=True)

        return unique_filename

    def get_temporary_file_path(self, file_id: str) -> Path | None:
        """
        Gets the full, validated path of a temporary file if it exists and is within the designated temp directory.
        Performs basic security checks against path traversal.
        """
        if not file_id or not isinstance(file_id, str):
            logging.warning(f"Invalid file_id type or empty: {file_id}")
            return None
        if '..' in file_id or '/' in file_id or '\\' in file_id:
            logging.warning(f"Potential path traversal attempt detected in file_id: {file_id}")
            return None
        try:
            file_path = (TEMP_UPLOAD_DIR / file_id).resolve()
            if TEMP_UPLOAD_DIR.resolve() in file_path.parents and file_path.exists() and file_path.is_file():
                 return file_path
            else:
                logging.warning(f"Temporary file path validation failed or file not found for ID: {file_id}. Resolved path: {file_path}")
                return None
        except Exception as e:
            logging.error(f"Error resolving or checking temporary file path for ID {file_id}: {e}", exc_info=True)
            return None

    def read_temporary_file_content(self, file_id: str) -> str:
        """
        Reads the content of a temporary file based on its extension, using appropriate libraries.
        Handles potential errors during file reading and library usage.
        """
        file_path = self.get_temporary_file_path(file_id) # Use self.method
        if not file_path:
            logging.error(f"Attempted to read non-existent or invalid temporary file with ID: {file_id}")
            raise HTTPException(status_code=404, detail=f"Temporary file '{file_id}' not found or invalid.")

        extension = file_path.suffix.lower()
        content = ""
        logging.info(f"Reading temporary file: {file_path} (Extension: {extension})")
        try:
            if extension == ".txt":
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f: content = f.read()
            elif extension == ".csv":
                if pd:
                    try: content = pd.read_csv(file_path).to_string()
                    except Exception as e: logging.error(f"Pandas failed CSV {file_id}: {e}"); content = f"[Error reading CSV: {e}]"
                else: content = "[Reading CSV as text - pandas not installed]\n" + file_path.read_text(encoding="utf-8", errors='ignore')
            elif extension == ".pdf":
                if pdfplumber:
                    try:
                        text_parts = []
                        with pdfplumber.open(file_path) as pdf:
                            if pdf.pages:
                                for i, page in enumerate(pdf.pages): text_parts.append(page.extract_text() or f"[Page {i+1} empty]")
                            else: text_parts.append("[PDF no pages]")
                        content = "\n\n".join(text_parts)
                        if not content.strip() or content == "[PDF no pages]": content = "[No text extracted]"
                    except Exception as e: logging.error(f"pdfplumber failed PDF {file_id}: {e}"); content = f"[Error reading PDF: {e}]"
                else: content = "[No PDF reader]"
            elif extension == ".docx":
                if docx2txt:
                    try: content = docx2txt.process(file_path)
                    except Exception as e: logging.error(f"docx2txt failed DOCX {file_id}: {e}"); content = f"[Error reading DOCX: {e}]"
                else: content = "[No DOCX reader]"
            elif extension == ".xlsx":
                if pd and openpyxl:
                    try:
                        xls = pd.ExcelFile(file_path)
                        if not xls.sheet_names: content = "[XLSX no sheets]"
                        else:
                            sheet_contents = [f"--- Sheet: {name} ---\n{pd.read_excel(xls, sheet_name=name).to_string()}" for name in xls.sheet_names]
                            content = "\n\n".join(sheet_contents)
                    except Exception as e: logging.error(f"Pandas failed XLSX {file_id}: {e}"); content = f"[Error reading XLSX: {e}]"
                else: content = "[No XLSX reader: pandas/openpyxl]"
            else: content = "[Unsupported file type]"
        except Exception as e: logging.error(f"Error processing file {file_id}: {e}"); content = f"[Unexpected error]"
        return content


# --- Cleanup Scheduler for Temporary Files ---
# (Keep this logic as moved from utils.py)
def delete_old_temp_files():
    """Scans the temporary directory and deletes files older than the configured lifespan."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=TEMP_FILE_LIFESPAN_HOURS)
    logging.info(f"Running cleanup for temporary files older than {cutoff} in {TEMP_UPLOAD_DIR}...")
    deleted_count = 0; error_count = 0
    try:
        for item in TEMP_UPLOAD_DIR.iterdir():
            if item.is_file():
                try:
                    mod_time = datetime.fromtimestamp(item.stat().st_mtime, timezone.utc)
                    if mod_time < cutoff:
                        item.unlink()
                        logging.info(f"Deleted old temporary file: {item.name}")
                        deleted_count += 1
                except OSError as e: logging.error(f"Error deleting file {item.name}: {e}"); error_count += 1
                except Exception as e: logging.error(f"Error processing file {item.name}: {e}"); error_count += 1
        logging.info(f"Temp file cleanup finished. Deleted: {deleted_count}, Errors: {error_count}.")
    except Exception as e: logging.error(f"Error during temp file cleanup scan: {e}")

def run_scheduler():
    """Runs the cleanup schedule in a loop."""
    logging.info("Background cleanup scheduler thread started.")
    schedule.every(1).hour.do(delete_old_temp_files)
    try: delete_old_temp_files() # Run once on start
    except Exception as e: logging.error(f"Initial run failed: {e}")
    while True:
        try: schedule.run_pending()
        except Exception as e: logging.error(f"Scheduler run failed: {e}")
        time.sleep(60)

_scheduler_thread = None
def start_cleanup_scheduler():
    """Starts the background cleanup scheduler thread if not already running."""
    global _scheduler_thread
    if _scheduler_thread is None or not _scheduler_thread.is_alive():
        logging.info("Starting background cleanup scheduler...")
        _scheduler_thread = threading.Thread(target=run_scheduler, daemon=True, name="TempFileCleanupScheduler")
        _scheduler_thread.start()
    else: logging.warning("Cleanup scheduler already running.")

# Instantiate if needed globally, or handle instantiation in API startup
# session_handler_instance = SessionHandler()