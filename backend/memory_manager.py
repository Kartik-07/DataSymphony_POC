# RAG_Project/MY_RAG/Backend/memory_manager.py
import os
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

# Import the updated models from models.py
from models import ConversationData, ConversationListItem, ChatMessage

from config import settings # Import settings to get user_history_base_dir

logger = logging.getLogger(__name__)

# --- Constants & Paths ---
USER_HISTORY_BASE_DIR: Path = settings.user_history_base_dir

# --- Helper Functions ---
def _clean_email_for_dirname(email: str) -> str:
    """Creates a safe directory name from an email address."""
    # Similar to filename cleaning, but maybe stricter for directory names
    return re.sub(r'[^\w\-@\.]', '_', email)

def _get_user_history_dir(user_email: str) -> Path:
    """Gets the path to the specific user's chat history directory."""
    dir_name = _clean_email_for_dirname(user_email)
    return USER_HISTORY_BASE_DIR / dir_name

def _get_conversation_filepath(user_history_dir: Path, conversation_id: str) -> Optional[Path]:
     """Finds the file path for a given conversation ID within a user's directory."""
     # Use the simpler pattern assuming ID is the filename stem
     pattern = f"{conversation_id}.json"
     matches = list(user_history_dir.glob(pattern))
     if not matches:
         return None
     if len(matches) > 1:
         # If multiple files match (e.g., due to old naming scheme), prioritize the direct ID match if possible
         direct_match = user_history_dir / f"{conversation_id}.json"
         if direct_match in matches:
             logger.warning(f"Multiple files found for conversation ID {conversation_id} in {user_history_dir}. Using exact match: {direct_match.name}")
             return direct_match
         else:
             logger.warning(f"Multiple files found for conversation ID {conversation_id} in {user_history_dir}. Using first match: {matches[0].name}")
             return matches[0] # Fallback to first match
     return matches[0]

def _create_conversation_filename(conversation_id: str) -> str:
     """Creates a filename for a new conversation."""
     # Simpler: just use the UUID as the filename stem
     return f"{conversation_id}.json"


# --- Core Memory Class ---
class MemoryManager:
    """Manages loading and saving chat conversations for specific users."""

    def get_user_history_path(self, user_email: str) -> Path:
        """Ensures the user's history directory exists and returns its path."""
        user_dir = _get_user_history_dir(user_email)
        try:
            user_dir.mkdir(parents=True, exist_ok=True)
            return user_dir
        except OSError as e:
            logger.error(f"Could not create history directory for user {user_email} at {user_dir}: {e}")
            raise  # Re-raise the exception as this is critical

    def load_conversation(self, user_email: str, conversation_id: str) -> Optional[ConversationData]:
        """Loads a specific conversation for a user, including the summary if present."""
        try:
            user_dir = self.get_user_history_path(user_email) # Ensure dir exists
            file_path = _get_conversation_filepath(user_dir, conversation_id)
            if not file_path or not file_path.is_file():
                logger.warning(f"Conversation file not found for user {user_email}, ID {conversation_id}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate data using the updated ConversationData model
            # Pydantic will handle the optional 'summary' field during validation
            conv_data = ConversationData(**data)

            # Security check: Ensure the loaded ID matches the requested ID
            if conv_data.id != conversation_id:
                 logger.error(f"ID mismatch in file {file_path.name}. Expected {conversation_id}, found {conv_data.id}.")
                 return None

            # --- ADDED LOGGING for summary ---
            if conv_data.summary:
                logger.debug(f"Loaded conversation {conversation_id} with existing summary (length {len(conv_data.summary)}).")
            else:
                logger.debug(f"Loaded conversation {conversation_id} without an existing summary.")
            # --- END ADDED LOGGING ---

            return conv_data
        except FileNotFoundError:
             logger.warning(f"Conversation file not found during load for user {user_email}, ID {conversation_id}")
             return None
        except (json.JSONDecodeError, TypeError, ValueError) as e: # Catches Pydantic validation errors too
            logger.error(f"Error loading/validating conversation {conversation_id} for user {user_email}: {e}")
            # Optionally: Handle corrupted files (move/delete)
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading conversation {conversation_id} for user {user_email}: {e}")
            return None

    def save_conversation(self, user_email: str, data: ConversationData) -> bool:
        """Saves a conversation to the user's history directory, including the summary if present."""
        try:
            user_dir = self.get_user_history_path(user_email) # Ensure dir exists
            # Use a consistent filename based on the ID
            filename = _create_conversation_filename(data.id)
            file_path = user_dir / filename

            # --- ADDED LOGGING for summary ---
            if data.summary:
                logger.debug(f"Saving conversation {data.id} with summary (length {len(data.summary)} chars).")
            else:
                logger.debug(f"Saving conversation {data.id} without summary.")
            # --- END ADDED LOGGING ---

            # Pydantic's model_dump_json handles serialization, including the optional 'summary' field
            json_data = data.model_dump_json(indent=4) # Pydantic v2+

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            logger.info(f"Saved conversation {data.id} for user {user_email} to: {file_path.name}")
            return True
        except IOError as e:
            logger.error(f"Error saving conversation {data.id} for user {user_email}: {e}")
            return False
        except Exception as e:
             logger.error(f"Unexpected error saving conversation {data.id} for user {user_email}: {e}")
             return False


    def list_conversations(self, user_email: str) -> List[ConversationListItem]:
        """Lists all conversations for a given user."""
        conversations = []
        try:
            user_dir = self.get_user_history_path(user_email) # Ensure dir exists
            # Use the simpler filename pattern "*.json" if IDs are filename stems
            for file_path in sorted(user_dir.glob("*.json"), key=os.path.getmtime, reverse=True): # Sort by mod time
                try:
                    # For listing, we only need ID, title, and timestamp.
                    # Reading the whole file just for this is slightly inefficient,
                    # but simpler than partial JSON parsing. Keep it for now.
                    conv_data = self.load_conversation(user_email, file_path.stem) # Use filename stem as ID

                    if conv_data:
                        # Use file modification time as the timestamp for listing (reflects last save)
                        m_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                        conversations.append(ConversationListItem(id=conv_data.id, title=conv_data.title, timestamp=m_time))
                    else:
                         logger.warning(f"Skipping invalid conversation file during list: {file_path.name} for user {user_email}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path.name} during list for user {user_email}: {e}")
                    continue # Skip problematic files

            # Sorting is now handled by glob sorting above, but can double-check or resort if needed.
            # conversations.sort(key=lambda x: x.timestamp, reverse=True)
            return conversations
        except Exception as e:
            logger.error(f"Failed to list conversations for user {user_email}: {e}")
            return [] # Return empty list on error


# Instantiate a single manager instance if desired (or use DI in FastAPI)
memory_manager = MemoryManager()