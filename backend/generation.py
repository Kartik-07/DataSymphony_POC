# RAG_Project/MY_RAG/Backend/generation.py

import logging
from typing import List, Dict, Any, Tuple, Optional

# --- Langchain Imports ---
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- Local Imports ---
# Import the necessary prompts and helpers from prompts.py
# UPDATED Imports: Removed format_chat_history, create_rag_prompt_template. Added RAG_PROMPT_WITH_HISTORY.
from prompts import (
    RAG_PROMPT_WITH_HISTORY,
    ANSWER_SUFFICIENCY_CHECK_PROMPT
)
from config import settings

# REMOVED: ChatMessage import is no longer needed here
# from models import ChatMessage


logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates answers based on query and retrieved context using Google Gemini.
    Uses a predefined prompt that includes conversation history context.
    Also uses a light LLM to check if the generated answer sufficiently addresses
    the query based *only* on the provided RAG context.
    """

    def __init__(self):
        """Initializes the LLMs used for generation and sufficiency check."""
        self.llm = None # Main LLM for generation
        self.light_llm = None # Light LLM for sufficiency check

        # Initialize Main LLM (from settings.llm_model_name)
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model_name, # Main model
                temperature=0.1, # Low temperature for factual RAG
                convert_system_message_to_human=True
            )
            logger.info(f"Initialized main Generator LLM: {settings.llm_model_name}")
        except Exception as e:
             logger.error(f"Failed to init main Generator LLM ({settings.llm_model_name}): {e}", exc_info=True)

        # Initialize Light LLM (from settings.light_llm_model_name)
        try:
            # Determine effective summary model (fallback to light LLM if specific history summary model isn't set)
            # This ensures light_llm is initialized even if used for summaries elsewhere
            light_model_name_to_use = settings.light_llm_model_name
            if not light_model_name_to_use:
                # Fallback further if even light_llm isn't set (though config should ideally handle this)
                logger.warning("LIGHT_LLM_MODEL_NAME is not set. Sufficiency check will be disabled.")
            else:
                self.light_llm = ChatGoogleGenerativeAI(
                    model=light_model_name_to_use, # Use the determined light model name
                    temperature=0.0, # Zero temperature for deterministic check
                    convert_system_message_to_human=True
                )
                logger.info(f"Initialized light Sufficiency Check LLM: {light_model_name_to_use}")
        except Exception as e:
            logger.error(f"Failed to init light Sufficiency Check LLM ({settings.light_llm_model_name}): {e}", exc_info=True)
            self.light_llm = None


    @staticmethod
    def _format_docs_with_metadata(docs: List[Document]) -> str:
        """
        Formats documents for the RAG prompt, including source metadata for citation.
        Uses the 'identifier' from metadata as the primary source name.
        (Keep this method exactly as in your fetched code)
        """
        if not docs:
            return "No context documents found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            # Prioritize identifier, then source, then fallback
            source_name = doc.metadata.get('identifier', doc.metadata.get('source', f'Unknown Source {i+1}'))
            # Basic cleaning for display
            source_name = str(source_name).replace('_', ' ').strip()
            metadata_str = f"Source: {source_name}"
            content = doc.page_content
            formatted_docs.append(f"--- Document {i+1} [{metadata_str}] ---\n{content}")

        return "\n\n".join(formatted_docs)

    def _check_sufficiency_with_llm(self, query: str, context: str, answer_from_context: str) -> bool:
        """
        Uses the light LLM to check if the answer is sufficient based on context.
        (Keep this method exactly as in your fetched code)
        """
        if not self.light_llm:
            logger.warning("Light LLM for sufficiency check unavailable. Assuming answer is sufficient.")
            return True # Default to sufficient if checker is down

        if not answer_from_context or answer_from_context.startswith("Sorry, I encountered an error"):
             logger.warning("Initial answer was empty or an error message. Checking sufficiency skipped, marking as insufficient.")
             return False

        try:
            check_input = {
                "context": context,
                "question": query,
                "answer_from_context": answer_from_context
            }
            if not ANSWER_SUFFICIENCY_CHECK_PROMPT:
                 logger.error("Answer sufficiency check prompt not available.")
                 return True # Default to sufficient if prompt missing

            sufficiency_chain = ANSWER_SUFFICIENCY_CHECK_PROMPT | self.light_llm | StrOutputParser()
            decision_str = sufficiency_chain.invoke(check_input).strip().upper()
            logger.info(f"Sufficiency Check LLM Decision: '{decision_str}'")
            # Check for the exact word "SUFFICIENT"
            return decision_str == "SUFFICIENT"
        except Exception as e:
            logger.error(f"Error during LLM sufficiency check: {e}", exc_info=True)
            # Default to sufficient on error to prevent fallback due to checker failure
            return True

    # --- MODIFIED generate_answer method ---
    def generate_answer(self,
                        query: str,
                        retrieved_docs: List[Document],
                        # MODIFIED: Accept the pre-formatted context string
                        conversation_history_context: Optional[str] = None
                       ) -> Tuple[str, bool]:
        """
        Generates an answer using the main LLM based on RAG context and conversation history context string.
        Uses the RAG_PROMPT_WITH_HISTORY template.
        Then uses the light LLM to check if that answer was sufficient given *only* the RAG context.

        Args:
            query: The user's query string.
            retrieved_docs: List of documents retrieved for context.
            conversation_history_context: Optional pre-formatted string including summary
                                          and recent turns from SessionHandler.

        Returns:
            Tuple[str, bool]: The generated answer string and a boolean flag
                              indicating if the answer was deemed sufficient based
                              on the provided context (True=sufficient, False=insufficient).
        """
        initial_answer = ""
        is_sufficient = True # Default to sufficient

        # Check if main LLM is available
        if not self.llm:
             logger.error("Cannot generate answer: Main Generator LLM is not initialized.")
             return "Sorry, the answer generation service is currently unavailable.", False

        # Check if RAG context was provided
        if not retrieved_docs:
            logger.warning("No RAG documents provided to generate_answer.")
            # Still might answer from history context/general knowledge via fallback in pipeline
            return "(No specific documents found to answer this query)", False # Indicate no RAG context used

        # 1. Format RAG context
        try:
             context_str = self._format_docs_with_metadata(retrieved_docs)
        except Exception as fmt_err:
             logger.error(f"Error formatting documents: {fmt_err}", exc_info=True)
             return f"Error preparing context for generation: {fmt_err}", False

        # 2. Prepare Inputs for the LLM using RAG_PROMPT_WITH_HISTORY
        try:
            prompt_template = RAG_PROMPT_WITH_HISTORY # Directly use the imported template
            if not prompt_template:
                 logger.error("RAG_PROMPT_WITH_HISTORY template not available.")
                 return "Error: RAG prompt template unavailable.", False

            logger.debug(f"Using prompt template with input variables: {prompt_template.input_variables}")

            llm_inputs = {
                "context": context_str,
                "question": query,
                # Provide the history context string, or a default if None
                "history_context": conversation_history_context or "No previous conversation context provided.",
            }
            logger.debug("Prepared inputs for generation LLM using history context string.")

        except Exception as prep_err:
            logger.error(f"Error preparing inputs or prompt: {prep_err}", exc_info=True)
            return f"Error setting up prompt inputs: {prep_err}", False

        # 3. Generate Initial Answer using Main LLM
        try:
            generation_chain = prompt_template | self.llm | StrOutputParser()
            log_inputs_trunc = {k: (v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) for k, v in llm_inputs.items()}
            logger.debug(f"Invoking generation chain with inputs: {log_inputs_trunc}") # Log truncated inputs
            initial_answer = generation_chain.invoke(llm_inputs)
            logger.info(f"Generated initial answer for query: '{query[:50]}...'")

            # Handle empty answer from LLM
            if not initial_answer:
                logger.warning("Main LLM returned an empty answer based on the provided context/history.")
                initial_answer = "(The language model did not provide an answer based on the provided documents and conversation context.)"
                is_sufficient = False # Empty answer is insufficient

        except Exception as e:
            logger.error(f"Error during initial answer generation for query '{query[:50]}...': {e}", exc_info=True)
            return f"Sorry, I encountered an error while generating the initial answer: {e}", False

        # 4. Check Sufficiency using Light LLM (based on RAG context only)
        if is_sufficient: # Only check if the answer wasn't already deemed insufficient
            logger.debug("Checking answer sufficiency based on retrieved RAG context...")
            # Note: Pass context_str (formatted RAG docs), not history_context, to the check.
            is_sufficient = self._check_sufficiency_with_llm(query, context_str, initial_answer)

        # 5. Return the initial answer and the final sufficiency flag
        logger.debug(f"Returning generated answer. Sufficient: {is_sufficient}")
        return initial_answer, is_sufficient