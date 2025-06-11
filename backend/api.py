# RAG_Project/MY_RAG/Backend/api.py

import logging
import os
import uuid
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Annotated

from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File,
    BackgroundTasks, Body, Path as FastApiPath, status, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, Field, EmailStr

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import settings
from rag_pipeline import RAGPipeline, RAGResponse # RAGResponse now includes used_structured_source_id
from indexing import Indexer, ensure_pgvector_setup
from summarization import DataSummarizer
from data_processing import DataLoader, TextProcessor
from utils import setup_logging
from models import ChatMessage, ConversationData, ConversationListItem
from auth_manager import auth_manager, UserRegistration, UserLogin, UserData as AuthUserData
from memory_manager import memory_manager, MemoryManager
from session_handler import SessionHandler, start_cleanup_scheduler


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced RAG System API (v5.0 - Strict RAG Mode)",
    description="API for querying a RAG system with user auth, history summarization, state management, SQL routing, summarization/ingestion, temp file uploads, and strict RAG mode.",
    version="5.0.0",
)

origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

indexer_instance: Optional[Indexer] = None
rag_pipeline_instance: Optional[RAGPipeline] = None
summarizer_instance: Optional[DataSummarizer] = None
title_llm_instance: Optional[ChatGoogleGenerativeAI] = None
session_handler_instance: Optional[SessionHandler] = None

@app.on_event("startup")
async def startup_event():
    global indexer_instance, rag_pipeline_instance, summarizer_instance, title_llm_instance, session_handler_instance
    setup_logging()
    logger.info("API Startup: Initializing resources...")
    try:
        logger.info(f"Auth directory: {settings.auth_dir}")
        logger.info(f"User history base directory: {settings.user_history_base_dir}")
        ensure_pgvector_setup()
        logger.info("PGVector setup check complete.")
        indexer_instance = Indexer()
        logger.info("Indexer initialized.")
        summarizer_instance = DataSummarizer()
        logger.info("DataSummarizer initialized.")
        rag_pipeline_instance = RAGPipeline(indexer=indexer_instance)
        logger.info("RAGPipeline initialized.")
        title_llm_instance = ChatGoogleGenerativeAI(model=settings.light_llm_model_name, temperature=0.2)
        logger.info(f"Title LLM initialized ({settings.light_llm_model_name}).")
        session_handler_instance = SessionHandler()
        logger.info("SessionHandler initialized.")
        start_cleanup_scheduler()
        logger.info("API Startup: All components initialized successfully.")
    except Exception as e:
        logger.critical(f"API Startup FAILED: {e}", exc_info=True)
        indexer_instance, rag_pipeline_instance, summarizer_instance, title_llm_instance, session_handler_instance = None, None, None, None, None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API Shutdown: Cleaning up resources...")
    logger.info("API Shutdown: Cleanup complete.")

def get_rag_pipeline() -> RAGPipeline:
    if rag_pipeline_instance is None:
        logger.error("RAG pipeline accessed before initialization.")
        raise HTTPException(status_code=503, detail="RAG service unavailable.")
    return rag_pipeline_instance

def get_indexer() -> Indexer:
    if indexer_instance is None:
        logger.error("Indexer accessed before initialization.")
        raise HTTPException(status_code=503, detail="Indexing service unavailable.")
    return indexer_instance

def get_summarizer() -> DataSummarizer:
    if summarizer_instance is None:
        logger.error("Summarizer accessed before initialization.")
        raise HTTPException(status_code=503, detail="Summarization service unavailable.")
    return summarizer_instance

def get_title_llm() -> ChatGoogleGenerativeAI:
    if title_llm_instance is None:
        logger.error("Title LLM accessed before initialization.")
        raise HTTPException(status_code=503, detail="Title generation service unavailable.")
    return title_llm_instance

def get_session_handler() -> SessionHandler:
    if session_handler_instance is None:
        logger.error("SessionHandler accessed before initialization.")
        raise HTTPException(status_code=503, detail="Session handling service unavailable.")
    return session_handler_instance

def get_memory_manager() -> MemoryManager:
    global memory_manager
    if memory_manager is None:
        logger.error("MemoryManager accessed before initialization.")
        raise HTTPException(status_code=503, detail="Memory management service unavailable.")
    return memory_manager

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> AuthUserData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str | None = payload.get("sub")
        if email is None:
            logger.warning("Token payload missing 'sub' (email)")
            raise credentials_exception
    except JWTError as e:
        logger.warning(f"JWTError decoding token: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error decoding or validating token: {e}", exc_info=True)
        raise credentials_exception
    try:
        user = auth_manager.get_user(email)
    except Exception as getUserError:
        logger.error(f"Error fetching user '{email}' from auth manager: {getUserError}", exc_info=True)
        raise credentials_exception
    if user is None:
        logger.warning(f"User '{email}' from valid token not found in auth storage.")
        raise credentials_exception
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

# --- API Models (QueryRequest MODIFIED) ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query text.")
    conversation_id: Optional[str] = Field(None, description="Optional ID of the conversation for context and history update")
    temp_file_id: Optional[str] = Field(None, description="Optional ID of a temporarily uploaded file to use as context.")
    strict_rag_mode: Optional[bool] = Field(False, description="If true, disables general knowledge fallback and supplementation.") # --- NEW FIELD ---

class IngestResponse(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None
    summary_id: Optional[str] = None

class ConversationSaveRequest(BaseModel):
    conversation_id: Optional[str] = None
    title: Optional[str] = Field(None, min_length=1, max_length=100)
    messages: List[ChatMessage]
    last_structured_source_id: Optional[str] = None

class ConversationSaveResponse(BaseModel):
    conversation_id: str
    title: str
    message: str

class UserResponse(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class TempUploadResponse(BaseModel):
    file_id: str
    filename: str

def clean_filename(name: str) -> str:
    if not name: return "untitled"
    name = re.sub(r'[^\w\-_\. ]', '_', name)
    name = name.strip().replace(' ', '_')
    return name[:100]

def generate_chat_title(first_prompt: str, llm: ChatGoogleGenerativeAI) -> str:
    if not first_prompt: return "New Chat"
    logger.info(f"Generating title (sync) based on: '{first_prompt[:50]}...'")
    try:
        system_prompt = "Generate a concise, descriptive title (3-5 words) for a chat conversation starting with the user prompt below. Focus on the core topic. Examples: 'Market Analysis Q1', 'Debugging Python Script', 'Vacation Planning'. Respond ONLY with the title itself."
        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Prompt:\n\"{first_prompt}\"")
        ]
        if not llm:
            logger.error("Title generation LLM is not available.")
            return "Chat"
        title_chain = llm | StrOutputParser()
        title = title_chain.invoke(messages_for_llm)
        title = title.strip().strip('"\'`').replace('\n', ' ').replace(':', '-').strip()
        title = title[:60]
        if not title: title = "Chat"
        logger.info(f"Generated title (sync): '{title}'")
        return title
    except Exception as e:
        logger.error(f"Error generating title (sync): {e}", exc_info=True)
        return "Chat"

def run_file_ingestion(temp_path: str, filename: str, indexer: Indexer, summarizer: DataSummarizer, user_email: Optional[str] = None):
    log_prefix = f"[Background Ingestion Task{' for user '+user_email if user_email else ''}]"
    logger.info(f"{log_prefix} Starting ingestion for: {filename}")
    summary_id = None
    try:
        if not summarizer or not indexer:
            logger.error(f"{log_prefix} Summarizer or Indexer not available. Aborting ingestion for {filename}.")
            return
        logger.info(f"{log_prefix} Summarizing {filename} using method 'auto'...")
        if not os.path.exists(temp_path):
            logger.error(f"{log_prefix} Temporary file {temp_path} not found. Aborting summarization for {filename}.")
            return
        summary_json = summarizer.summarize(temp_path, file_name_override=filename, summary_method='auto')
        if not summary_json or summary_json.get("metadata", {}).get("data_type") == "error":
            error_msg = summary_json.get("metadata", {}).get("error", "Summarization failed, reason unknown.") if summary_json else "Summarizer returned None or empty response."
            logger.error(f"{log_prefix} Summarization failed for {filename}: {error_msg}")
            return
        summary_id = summary_json.get("id", str(uuid.uuid4()))
        logger.info(f"{log_prefix} Summarization successful for {filename}. Summary ID: {summary_id}.")
        metadata_for_doc = summary_json.get("metadata", {"file_name": filename})
        metadata_for_doc["summary_id"] = summary_id
        metadata_for_doc["original_filename"] = filename
        if user_email:
            metadata_for_doc["uploaded_by"] = user_email
        doc_to_index = Document(
            page_content=summary_json.get("document", ""),
            metadata=metadata_for_doc
        )
        logger.info(f"{log_prefix} Indexing summary document for {filename} with ID {summary_id}...")
        indexer.index_documents([doc_to_index], ids=[summary_id])
        logger.info(f"{log_prefix} Successfully summarized and indexed {filename} (Summary ID: {summary_id}).")
    except Exception as e:
        logger.error(f"{log_prefix} Error during ingestion processing for {filename}: {e}", exc_info=True)
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"{log_prefix} Removed temp file: {temp_path}")
        except OSError as e:
            logger.error(f"{log_prefix} Error removing temp file {temp_path}: {e}")

@app.get("/", tags=["Status"], summary="Get API Status and Configuration")
async def get_status():
    is_ok = all([rag_pipeline_instance, indexer_instance, summarizer_instance, title_llm_instance, session_handler_instance])
    status_message = "RAG API is running." if is_ok else "RAG API Service initialization failed."
    effective_summary_model = settings.history_summary_llm_model_name or settings.light_llm_model_name
    return {
        "status": "ok" if is_ok else "error",
        "message": status_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "rag_pipeline": "initialized" if rag_pipeline_instance else "failed",
            "indexer": "initialized" if indexer_instance else "failed",
            "summarizer": "initialized" if summarizer_instance else "failed",
            "title_llm": "initialized" if title_llm_instance else "failed",
            "session_handler": "initialized" if session_handler_instance else "failed",
        },
        "config": {
            "main_llm_model": settings.llm_model_name,
            "light_llm_model": settings.light_llm_model_name,
            "embedding_model": settings.embedding_model_name,
            "reranker_enabled": settings.use_cohere_rerank,
            "collection_name": settings.collection_name,
            "auth_enabled": True,
            "user_history_enabled": True,
            "temp_file_uploads_enabled": True,
            "history_max_turns": settings.history_max_turns,
            "history_summary_model": effective_summary_model or 'Not Set',
            "state_management_enabled": True,
        }
    }

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Authentication"], summary="Register a New User")
def register_new_user(user_in: UserRegistration = Body(...)):
    logger.info(f"Registration attempt for email: {user_in.email}")
    existing_user = auth_manager.get_user(user_in.email)
    if existing_user:
        logger.warning(f"Registration failed: Email '{user_in.email}' already registered.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    user = auth_manager.register_user(user_in)
    if not user:
        logger.error(f"User registration failed unexpectedly for {user_in.email}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not register user due to server error.",
        )
    return UserResponse(email=user.email)

@app.post("/token", response_model=Token, tags=["Authentication"], summary="Login and Get Access Token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    logger.info(f"Login attempt for username (email): {form_data.username}")
    try:
        user_login_data = UserLogin(email=form_data.username, password=form_data.password)
        user = auth_manager.authenticate_user(user_login_data)
    except ValueError as e:
        logger.warning(f"Login failed: Invalid email format '{form_data.username}'. Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
         logger.error(f"Unexpected error during authentication for '{form_data.username}': {e}", exc_info=True)
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail="Authentication service error",
             headers={"WWW-Authenticate": "Bearer"},
         )
    if not user:
        logger.warning(f"Authentication failed for user '{form_data.username}' (user not found or wrong password).")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    logger.info(f"Token issued for user: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse, tags=["Authentication"], summary="Get Current User Info")
async def read_users_me(
    current_user: Annotated[AuthUserData, Depends(get_current_user)]
):
    logger.debug(f"Returning info for authenticated user: {current_user.email}")
    return UserResponse(email=current_user.email)

@app.get("/conversations", response_model=List[ConversationListItem], tags=["Chat History"], summary="List User's Conversations")
def list_conversations_for_user(
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    memory_manager_dep: Annotated[MemoryManager, Depends(get_memory_manager)]
):
    logger.info(f"Listing conversations for user: {current_user.email}")
    conversations = memory_manager_dep.list_conversations(current_user.email)
    return conversations

@app.get("/conversations/{conversation_id}", response_model=ConversationData, tags=["Chat History"], summary="Get Specific Conversation")
def get_conversation_for_user(
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    memory_manager_dep: Annotated[MemoryManager, Depends(get_memory_manager)],
    conversation_id: str = FastApiPath(..., description="The ID of the conversation to retrieve")
):
    logger.info(f"Getting conversation {conversation_id} for user: {current_user.email}")
    conversation_data = memory_manager_dep.load_conversation(current_user.email, conversation_id)
    if conversation_data is None:
        logger.warning(f"Conversation {conversation_id} not found for user {current_user.email}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conversation_data

@app.post("/conversations", response_model=ConversationSaveResponse, tags=["Chat History"], summary="Save or Create Conversation")
def save_conversation_for_user(
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    title_llm: Annotated[ChatGoogleGenerativeAI, Depends(get_title_llm)],
    memory_manager_dep: Annotated[MemoryManager, Depends(get_memory_manager)],
    request: ConversationSaveRequest = Body(...)
):
    user_email = current_user.email
    logger.info(f"Save conversation request for user: {user_email} (Incoming ID: {request.conversation_id})")
    is_new_chat = not request.conversation_id
    conversation_data: Optional[ConversationData] = None
    response_message: str
    try:
        if is_new_chat:
            logger.info(f"Creating new conversation for user: {user_email}")
            if not request.messages:
                logger.warning(f"Attempt to create empty chat for user: {user_email}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot create an empty chat.")
            new_id = str(uuid.uuid4())
            generated_title = request.title
            if not generated_title:
                first_user_message = next((msg for msg in request.messages if msg.sender == 'user'), None)
                prompt_text = first_user_message.text if first_user_message else ""
                generated_title = generate_chat_title(prompt_text, title_llm)
            created_time = datetime.now(timezone.utc)
            conversation_data = ConversationData(
                id=new_id, title=generated_title, created_at=created_time,
                messages=request.messages,
                summary=None,
                last_structured_source_id=request.last_structured_source_id
            )
            response_message = "New conversation created."
            logger.info(f"New conversation created. ID: {new_id}, Title: '{generated_title}' for user {user_email}")
        else:
            conv_id = request.conversation_id
            logger.info(f"Updating conversation {conv_id} for user: {user_email}")
            existing_data = memory_manager_dep.load_conversation(user_email, conv_id)
            if not existing_data:
                logger.warning(f"Update failed: Conversation ID {conv_id} not found for user {user_email}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Conversation ID {conv_id} not found.")
            final_title = request.title if request.title else existing_data.title
            existing_data.title = final_title
            existing_data.messages = request.messages
            if request.last_structured_source_id is not None:
                 existing_data.last_structured_source_id = request.last_structured_source_id
            conversation_data = existing_data
            response_message = "Conversation updated."
            logger.info(f"Conversation title/messages updated. ID: {conv_id}, Title: '{final_title}' for user {user_email}")
        if not memory_manager_dep.save_conversation(user_email, conversation_data):
            logger.error(f"MemoryManager failed to save conversation {conversation_data.id} for user {user_email}.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save conversation data.")
        return ConversationSaveResponse(
            conversation_id=conversation_data.id, title=conversation_data.title, message=response_message
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error saving conversation (ID: {request.conversation_id}) for user {user_email}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save chat due to a server error.")

@app.post("/upload_temp", response_model=TempUploadResponse, tags=["Temporary Files"], summary="Upload Temporary File for Query Context")
async def upload_temporary_file_endpoint(
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    session_handler: Annotated[SessionHandler, Depends(get_session_handler)],
    file: UploadFile = File(...)
):
    user_email = current_user.email
    original_filename = file.filename if file.filename else "unknown_file"
    logger.info(f"Temporary file upload request from user {user_email} for file: {original_filename}")
    try:
        file_id = session_handler.save_temporary_file(file)
        logger.info(f"Temporary file saved successfully for user {user_email}. File ID: {file_id}, Original Name: {original_filename}")
        return TempUploadResponse(file_id=file_id, filename=original_filename)
    except HTTPException as e:
        logger.warning(f"HTTPException during temporary file upload for user {user_email}: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during temporary file upload for user {user_email}, file {original_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred during file upload.")

# --- Query Endpoint (MODIFIED to accept strict_rag_mode) ---
@app.post("/query", response_model=RAGResponse, tags=["RAG Query"], summary="Query RAG Pipeline with History Summary, State, Optional Temp File, and Strict RAG Mode")
def process_query_for_user(
    request: QueryRequest, # Uses the updated QueryRequest model
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    session_handler: Annotated[SessionHandler, Depends(get_session_handler)],
    memory_manager_dep: Annotated[MemoryManager, Depends(get_memory_manager)],
    title_llm: Annotated[ChatGoogleGenerativeAI, Depends(get_title_llm)]
):
    user_email = current_user.email
    conversation_id = request.conversation_id
    temp_file_id = request.temp_file_id
    query_text = request.query
    strict_rag_mode = request.strict_rag_mode # --- NEW: Get the flag ---

    log_msg = f"Received query from user {user_email}: '{query_text[:100]}...' (Strict RAG: {strict_rag_mode})" # Log new flag
    if conversation_id: log_msg += f" (Conversation ID: {conversation_id})"
    if temp_file_id: log_msg += f" (Temp File ID: {temp_file_id})"
    logger.info(log_msg)

    loaded_conv_data: Optional[ConversationData] = None
    current_messages: List[ChatMessage] = []
    current_summary: Optional[str] = None
    current_last_source_id: Optional[str] = None
    temp_file_content: Optional[str] = None
    final_query_text: str = query_text

    if temp_file_id:
        logger.debug(f"Attempting to read temporary file content for ID: {temp_file_id}")
        try:
            temp_file_content = session_handler.read_temporary_file_content(temp_file_id)
            logger.info(f"Successfully read content from temporary file {temp_file_id} (approx {len(temp_file_content)} chars).")
        except HTTPException as e:
            if e.status_code == status.HTTP_404_NOT_FOUND: logger.warning(f"Temporary file {temp_file_id} not found. Proceeding without file content.")
            else: logger.error(f"HTTPException reading temp file {temp_file_id}: {e.detail}", exc_info=True)
            temp_file_content = None
        except Exception as e:
            logger.error(f"Unexpected error reading temp file {temp_file_id}: {e}", exc_info=True)
            temp_file_content = None
    if temp_file_content:
        final_query_text = (
            f"Based on the following document content:\n"
            f"--- Start Document ---\n{temp_file_content}\n--- End Document ---\n\n"
            f"User Query: {query_text}"
        )
        logger.debug(f"Prepended temporary file content to query. New query length approx: {len(final_query_text)}")

    if conversation_id:
        logger.debug(f"Attempting to load history for conversation {conversation_id} for user {user_email}")
        loaded_conv_data = memory_manager_dep.load_conversation(user_email, conversation_id)
        if loaded_conv_data:
            current_messages = loaded_conv_data.messages
            current_summary = loaded_conv_data.summary
            current_last_source_id = loaded_conv_data.last_structured_source_id
            logger.info(f"Loaded {len(current_messages)} messages, summary (exists: {bool(current_summary)}), last_source_id: '{current_last_source_id}' from conversation {conversation_id}.")
        else:
            logger.warning(f"Conversation {conversation_id} provided but not found for user {user_email}. Treating as new chat.")
            conversation_id = None; loaded_conv_data = None; current_last_source_id = None

    history_context_for_llm, new_summary = session_handler.manage_history(current_messages, current_summary)
    if new_summary is not None:
        logger.info(f"New summary generated/updated for conversation {conversation_id or '(new chat)'}")

    try:
        logger.debug(f"Calling RAG pipeline for user {user_email} with final query text, history context, last source ID '{current_last_source_id}', and strict_rag_mode '{strict_rag_mode}'.")
        # --- MODIFIED: Pass strict_rag_mode to pipeline.query ---
        result: RAGResponse = pipeline.query(
            query_text=final_query_text,
            conversation_history_context=history_context_for_llm,
            last_structured_source_id=current_last_source_id,
            strict_rag_mode=strict_rag_mode # Pass the new flag
        )
        logger.debug(f"RAG pipeline returned response for user {user_email}. Used source ID: {result.used_structured_source_id}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during RAG pipeline query for user {user_email} '{query_text[:100]}...': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during RAG processing.")

    should_save = loaded_conv_data is not None or request.conversation_id is None
    if should_save and result.answer:
        logger.debug(f"Attempting to save interaction/summary/state for conversation {conversation_id or '(new chat)'} for user {user_email}")
        try:
            current_time = datetime.now(timezone.utc)
            user_msg = ChatMessage(sender="user", text=query_text, timestamp=current_time)
            ai_msg = ChatMessage(sender="ai", text=result.answer, sources=getattr(result, 'sources', None), timestamp=current_time)
            data_to_save: ConversationData; saved_conv_id: str
            if loaded_conv_data:
                loaded_conv_data.messages.extend([user_msg, ai_msg])
                if new_summary is not None: loaded_conv_data.summary = new_summary
                if result.used_structured_source_id is not None:
                     loaded_conv_data.last_structured_source_id = result.used_structured_source_id
                     logger.info(f"Updating last_structured_source_id to '{result.used_structured_source_id}' for conversation {loaded_conv_data.id}")
                data_to_save = loaded_conv_data; saved_conv_id = loaded_conv_data.id
            else:
                new_conv_id = str(uuid.uuid4()); saved_conv_id = new_conv_id
                title = generate_chat_title(query_text, title_llm)
                data_to_save = ConversationData(
                    id=new_conv_id, title=title, created_at=current_time,
                    messages=[user_msg, ai_msg], summary=new_summary,
                    last_structured_source_id=result.used_structured_source_id
                )
                logger.info(f"First interaction for new chat. Created Conversation ID: {new_conv_id}, Title: '{title}'. Initial last_structured_source_id: '{result.used_structured_source_id}'")
            if memory_manager_dep.save_conversation(user_email, data_to_save):
                logger.info(f"Successfully saved interaction/summary/state for conversation {saved_conv_id} for user {user_email}")
            else:
                logger.error(f"Failed to save updated history/summary/state for conversation {saved_conv_id} (MemoryManager returned False).")
        except Exception as hist_e:
            logger.error(f"Error saving interaction/summary/state for conversation {conversation_id or '(new chat)'}: {hist_e}", exc_info=True)
    elif should_save and not result.answer:
        logger.warning(f"RAG pipeline did not return an answer for query in conversation {conversation_id}. History/summary/state not updated.")

    logger.info(f"Sending RAG response to user {user_email} for query: '{query_text[:100]}...'")
    return result

@app.post("/ingest_file", response_model=IngestResponse, tags=["Indexing"], status_code=status.HTTP_202_ACCEPTED, summary="Ingest and Index File Permanently")
async def ingest_file_endpoint(
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    indexer: Annotated[Indexer, Depends(get_indexer)],
    summarizer: Annotated[DataSummarizer, Depends(get_summarizer)],
    file: UploadFile = File(...)
):
    user_email = current_user.email
    original_filename = file.filename if file.filename else "unknown_file"
    log_prefix = f"Permanent ingest request for user {user_email}: {original_filename}"
    logger.info(log_prefix)
    temp_path_str: Optional[str] = None
    try:
        upload_dir = Path(settings.user_history_base_dir / "../uploads_ingest_temp")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_original_filename = clean_filename(original_filename)
        unique_suffix = str(uuid.uuid4())[:8]
        temp_filename = f"{unique_suffix}_{safe_original_filename}"
        max_len = 200
        if len(temp_filename) > max_len:
            name, ext = os.path.splitext(temp_filename)
            temp_filename = name[:max_len - len(ext) - 1] + '_' + ext if ext else name[:max_len-1] + '_'
        temp_file_path = upload_dir / temp_filename; temp_path_str = str(temp_file_path)
        try:
            file_content = await file.read()
            with open(temp_file_path, "wb") as f: f.write(file_content)
            logger.info(f"Saved temporary file for permanent ingestion: {temp_path_str}")
        except Exception as save_err:
            logger.error(f"Failed to save temporary file {temp_path_str} for permanent ingestion: {save_err}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file.")
        finally:
             try: await file.close()
             except Exception as close_err: logger.warning(f"Error closing file stream for permanent ingestion: {close_err}", exc_info=True)
        background_tasks.add_task(
            run_file_ingestion, temp_path=temp_path_str, filename=original_filename,
            indexer=indexer, summarizer=summarizer, user_email=user_email
        )
        logger.info(f"Scheduled background task for permanent ingestion: {original_filename} by user {user_email}")
        return IngestResponse(
            status="scheduled", message=f"Permanent ingestion task for '{original_filename}' scheduled successfully.",
            filename=original_filename
        )
    except HTTPException as http_exc:
        if temp_path_str and os.path.exists(temp_path_str):
            try: os.remove(temp_path_str)
            except OSError: logger.error(f"Failed to clean up temp file on HTTPException during permanent ingest schedule: {temp_path_str}")
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to schedule permanent ingestion for {original_filename}: {e}", exc_info=True)
        if temp_path_str and os.path.exists(temp_path_str):
            try: os.remove(temp_path_str)
            except OSError: logger.error(f"Failed to clean up temp file on error during permanent ingest schedule: {temp_path_str}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to schedule file ingestion due to a server error.")