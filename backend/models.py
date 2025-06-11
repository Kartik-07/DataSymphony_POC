# RAG_Project/MY_RAG/Backend/models.py

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# --- Updated Pydantic Import ---
from pydantic import BaseModel, Field, field_validator, EmailStr

logger = logging.getLogger(__name__)

# --- Shared Models ---

class ChatMessage(BaseModel):
    sender: str = Field(..., pattern=r"^(user|ai)$")
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: Optional[List[Dict[str, Any]]] = None # From RAGResponse
    error: Optional[bool] = None

    # --- Updated Pydantic V2 Validator ---
    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_timezone(cls, v: Any) -> datetime:
        """Ensures the timestamp is timezone-aware (UTC) before validation."""
        if isinstance(v, str):
            try:
                # Handle ISO format with Z for UTC
                dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                # Ensure it's UTC if parsed successfully
                return dt.astimezone(timezone.utc)
            except ValueError:
                try:
                    # Handle ISO format without Z
                    dt = datetime.fromisoformat(v)
                    if dt.tzinfo is None:
                        # logger.warning(f"Timestamp string '{v}' is naive. Assuming local time and converting to UTC.")
                        # If timezone is naive, assume it's local and convert to UTC
                        # This might not be ideal, explicitly requiring TZ info is safer
                        # For simplicity here, we'll assume UTC if naive, but logging it
                        logger.warning(f"Timestamp string '{v}' is naive. Assuming UTC.")
                        return dt.replace(tzinfo=timezone.utc)
                    # If timezone info exists, convert to UTC
                    return dt.astimezone(timezone.utc)
                except ValueError:
                    logger.error(f"Could not parse timestamp string: '{v}'. Using current UTC time.")
                    return datetime.now(timezone.utc) # Fallback
        elif isinstance(v, datetime):
            if v.tzinfo is None:
                 # logger.warning(f"Timestamp datetime object is naive. Assuming UTC.")
                 # Assume UTC if naive
                 return v.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if it's not already
                return v.astimezone(timezone.utc)
        else:
            logger.error(f"Unexpected type for timestamp validation: {type(v)}. Using current UTC time.")
            return datetime.now(timezone.utc) # Fallback


class ConversationData(BaseModel):
    id: str # UUID
    title: str
    created_at: datetime
    messages: List[ChatMessage]
    summary: Optional[str] = None
    # --- ADDED STATE FIELD ---
    last_structured_source_id: Optional[str] = Field(None, description="Identifier of the last structured data source used (table or file).")
    # -------------------------

    @field_validator('created_at', mode='before')
    @classmethod
    def ensure_created_at_timezone(cls, v: Any) -> datetime:
        """Ensures the created_at timestamp is timezone-aware (UTC) before validation."""
        # Reusing the same logic as ChatMessage timestamp validation
        return ChatMessage.ensure_timezone(v)


class ConversationListItem(BaseModel):
    id: str
    title: str
    timestamp: datetime # Usually last updated/modified time

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_list_item_timestamp_timezone(cls, v: Any) -> datetime:
        """Ensures the list item timestamp is timezone-aware (UTC) before validation."""
        # Reusing the same logic as ChatMessage timestamp validation
        return ChatMessage.ensure_timezone(v)