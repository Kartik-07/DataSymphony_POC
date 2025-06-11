# RAG_Project/MY_RAG/Backend/auth_manager.py
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any
from passlib.context import CryptContext # For hashing
from pydantic import BaseModel, EmailStr, Field, validator

from config import settings # Import settings to get auth_dir

logger = logging.getLogger(__name__)

# Setup password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Constants & Paths ---
AUTH_DIR: Path = settings.auth_dir
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# --- Models ---
class UserData(BaseModel):
    email: EmailStr
    hashed_password: str
    # Add other user fields if needed, e.g., name, creation_date

class UserRegistration(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# --- Helper Functions ---
def _clean_email_for_filename(email: str) -> str:
    """Creates a safe filename from an email address."""
    return re.sub(r'[^\w\-@\.]', '_', email)

def _get_user_filepath(email: str) -> Path:
    """Gets the expected path for a user's data file."""
    filename = f"{_clean_email_for_filename(email)}.json"
    return AUTH_DIR / filename

# --- Core Auth Class ---
class AuthManager:
    """Manages user authentication using file-based storage."""

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies a plain password against a stored hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generates a hash for a given password."""
        return pwd_context.hash(password)

    def get_user(self, email: EmailStr) -> Optional[UserData]:
        """Retrieves user data from their file."""
        filepath = _get_user_filepath(email)
        if not filepath.is_file():
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Validate data against Pydantic model
            user = UserData(**data)
            # Ensure the email in the file matches the requested email
            if user.email.lower() != email.lower():
                 logger.error(f"Email mismatch in file {filepath.name} for requested email {email}")
                 return None
            return user
        except (json.JSONDecodeError, TypeError, ValueError) as e: # Catch Pydantic validation errors too
            logger.error(f"Error reading or validating user file {filepath.name}: {e}")
            # Optionally: handle corrupted files (e.g., move/delete)
            return None
        except Exception as e:
             logger.error(f"Unexpected error loading user {email}: {e}")
             return None


    def register_user(self, user_in: UserRegistration) -> Optional[UserData]:
        """Registers a new user, saving their data to a file."""
        email_lower = user_in.email.lower()
        filepath = _get_user_filepath(email_lower)

        if filepath.exists():
            logger.warning(f"Registration attempt failed: User '{email_lower}' already exists.")
            return None # User already exists

        hashed_password = self.get_password_hash(user_in.password)
        user_data = UserData(email=email_lower, hashed_password=hashed_password)

        try:
            AUTH_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(user_data.model_dump(), f, indent=4) # Use model_dump for Pydantic v2+
            logger.info(f"User '{email_lower}' registered successfully.")
            return user_data
        except IOError as e:
            logger.error(f"Failed to write user file for '{email_lower}': {e}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error registering user {email_lower}: {e}")
             return None

    def authenticate_user(self, user_in: UserLogin) -> Optional[UserData]:
        """Authenticates a user based on email and password."""
        user = self.get_user(user_in.email)
        if not user:
            logger.warning(f"Authentication failed: User '{user_in.email}' not found.")
            return None # User not found

        if not self.verify_password(user_in.password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password for user '{user_in.email}'.")
            return None # Invalid password

        logger.info(f"User '{user_in.email}' authenticated successfully.")
        return user

# Instantiate a single manager instance if desired (or use DI in FastAPI)
auth_manager = AuthManager()