# RAG_Project/MY_RAG/Backend/rag_pipeline.py

import logging
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import uuid

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from config import settings
    from retrieval import EnhancedRetriever
    from generation import AnswerGenerator
    from indexing import Indexer
    from sql_processing import (
        generate_sql_query,
        clean_sql_string,
        get_rag_db_utility,
        get_uploads_db_utility,
        SQLDatabase
    )
    from prompts import ( # Prompts now instruct LLM for better formatting
        RAG_PROMPT_WITH_HISTORY,
        FALLBACK_PROMPT, FALLBACK_PROMPT_WITH_HISTORY,
        PYTHON_GENERATION_PROMPT
    )
    from data_science_executor import DataScienceExecutor, DataScienceExecutorError
    from routing_logic import QueryRouter
except ImportError as e:
     logging.exception(f"Error importing local modules in rag_pipeline.py: {e}", exc_info=True)
     raise

logger = logging.getLogger(__name__)

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: Optional[List[str]] = None
    used_structured_source_id: Optional[str] = None
    reasoning_log: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Log of internal reasoning steps.")

    @staticmethod
    def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        sources_data = []
        if not docs: return sources_data
        seen_identifiers = set()
        for doc in docs:
            identifier = doc.metadata.get('identifier', doc.metadata.get('source'))
            if not identifier:
                content_hash = hash(doc.page_content[:50])
                identifier = f"unknown_{content_hash}_{len(seen_identifiers)}"
            if identifier in seen_identifiers: continue
            seen_identifiers.add(identifier)
            source_label = doc.metadata.get('file_name', identifier)
            source_type_flag = doc.metadata.get('type', 'vector') # 'type' in metadata from Document
            source_info = {
                "source": source_label,
                "type": source_type_flag, 
                "content_snippet": doc.page_content[:200] + "..." if source_type_flag != 'plot_png_base64' else "Generated Plot",
                "relevance_score": doc.metadata.get('relevance_score'),
                "identifier": identifier,
                "page": doc.metadata.get('page'),
                "summary_id": doc.metadata.get('summary_id'),
                "uploaded_by": doc.metadata.get('uploaded_by'),
                "data": doc.metadata.get('data') if source_type_flag == 'plot_png_base64' else None
            }
            sources_data.append({k: v for k, v in source_info.items() if v is not None})
        return sources_data

class RAGPipeline:
    def __init__(self, indexer: Indexer):
        self.retriever_wrapper = EnhancedRetriever(indexer)
        self.generator = AnswerGenerator()
        self.rag_db_utility: Optional[SQLDatabase] = get_rag_db_utility()
        self.uploads_db_utility: Optional[SQLDatabase] = get_uploads_db_utility()
        self.light_llm = None
        try:
            if settings.light_llm_model_name:
                 self.light_llm = ChatGoogleGenerativeAI(model=settings.light_llm_model_name, temperature=0.1)
                 logger.info(f"Pipeline Light LLM (Routing/Suggestions) initialized: {settings.light_llm_model_name}")
            else:
                 logger.warning("LIGHT_LLM_MODEL_NAME not set, routing/suggestion features may be limited.")
        except Exception as e:
            logger.error(f"Failed to init Pipeline Light LLM: {e}", exc_info=True)

        self.query_router = QueryRouter(
            light_llm=self.light_llm,
            base_retriever=self.retriever_wrapper.base_retriever,
            rag_db_utility_available=bool(self.rag_db_utility),
            uploads_db_utility_available=bool(self.uploads_db_utility)
        )

        self.ds_executor: Optional[DataScienceExecutor] = None
        try:
            self.ds_executor = DataScienceExecutor()
        except Exception as e:
             logger.error(f"Failed to initialize DataScienceExecutor: {e}", exc_info=True)

        logger.info("RAG Pipeline initialized.")
        if not self.generator.llm: logger.warning("Main Generator LLM unavailable in pipeline.")
        if not self.light_llm: logger.warning("Light LLM unavailable in pipeline (routing/suggestions affected).")
        if not self.query_router: logger.warning("QueryRouter unavailable in pipeline.")
        if not self.ds_executor: logger.warning("DataScience Executor unavailable in pipeline.")
        if not self.rag_db_utility: logger.warning("RAG_DB Utility unavailable in pipeline (SQL route affected).")
        if not self.uploads_db_utility: logger.warning("Uploads DB Utility unavailable in pipeline (SQL route affected).")


    def _generate_suggestions(self, user_query: str, ai_answer: str) -> Optional[List[str]]:
        if not user_query or not ai_answer or not self.light_llm: return None
        logger.info("Generating suggestions (sync)...")
        try:
            suggestion_prompt_template = """
Given the user's query and the AI's answer, suggest 3 concise and relevant follow-up questions or actions.
Phrase them as if the user is asking.
Respond ONLY with a valid JSON list of strings, like ["Suggestion 1", "Suggestion 2", "Suggestion 3"].

User Query: "{user_query}"
AI Answer: "{ai_answer}"

Suggested follow-up questions/actions (JSON list):
            """
            suggestion_prompt = PromptTemplate.from_template(suggestion_prompt_template)
            suggestion_chain = suggestion_prompt | self.light_llm | StrOutputParser()
            raw_suggestions_output = suggestion_chain.invoke({"user_query": user_query, "ai_answer": ai_answer})
            match = re.search(r"\[.*\]", raw_suggestions_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    suggestions = json.loads(json_str)
                    if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions): return suggestions[:3]
                except json.JSONDecodeError: pass
            extracted = re.findall(r'"([^"]+)"', raw_suggestions_output);
            if extracted: return extracted[:3]
            logger.warning(f"Could not extract suggestions reliably from LLM output: {raw_suggestions_output}")
            return None
        except Exception as e:
             logger.error(f"Suggestion generation error: {e}", exc_info=True)
             return None

    def _clean_python_code(self, raw_code: str) -> str:
        logger.debug("Cleaning generated Python code...")
        cleaned = re.sub(r"^\s*```python\s*", "", raw_code, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()
        logger.debug(f"Cleaned code snippet: {cleaned[:200]}...")
        return cleaned

    def query(self,
              query_text: str,
              conversation_history_context: Optional[str] = None,
              last_structured_source_id: Optional[str] = None,
              strict_rag_mode: Optional[bool] = False
             ) -> RAGResponse:
        log_context_snippet = f"(History Context: '{conversation_history_context[:100]}...')" if conversation_history_context else "(No History Context)"
        log_last_source = f"(Last Source Hint: {last_structured_source_id})" if last_structured_source_id else ""
        logger.info(f"Processing query: '{query_text[:50]}...' {log_context_snippet} {log_last_source} (Strict RAG: {strict_rag_mode})")

        reasoning_log_entries: List[Dict[str, Any]] = []

        if not self.generator.llm:
            logger.error("Main LLM unavailable in RAG pipeline.")
            reasoning_log_entries.append({"step_title": "System Error", "details": "Main LLM unavailable."})
            return RAGResponse(answer="LLM unavailable.", sources=[], suggestions=None, used_structured_source_id=None, reasoning_log=reasoning_log_entries)

        needs_direct_fallback = False; needs_supplemental_fallback = False
        decision = 'NONE'; target_metadata_from_router: Optional[Dict[str, Any]] = None
        retrieved_docs = []; high_score_docs = []
        initial_answer = ""; final_answer = ""; sources_for_response = []; suggestions = None
        used_structured_source_id_this_turn: Optional[str] = None

        reasoning_log_entries.append({"step_title": "Query Received", "details": query_text[:250] + ("..." if len(query_text)>250 else "")})
        if conversation_history_context:
            reasoning_log_entries.append({"step_title": "Conversation Context", "details": "History context provided for query understanding."})
        if last_structured_source_id:
            reasoning_log_entries.append({"step_title": "Prior Source Hint", "details": f"Last used structured source: {last_structured_source_id}"})
        reasoning_log_entries.append({"step_title": "Mode", "details": f"Strict RAG Mode: {strict_rag_mode}"})


        try:
            if self.query_router:
                reasoning_log_entries.append({"step_title": "Routing", "details": "Determining query route..."})
                decision, target_metadata_from_router = self.query_router.route_query(
                    query_text,
                    conversation_history_context,
                    last_structured_source_id
                )
                reasoning_log_entries.append({"step_title": "Routing Decision", "details": f"Routed to: {decision}"})
                if target_metadata_from_router and target_metadata_from_router.get('identifier'):
                    reasoning_log_entries.append({
                        "step_title": "Targeted Data Source",
                        "details": {
                            "identifier": target_metadata_from_router.get('identifier'),
                            "name": target_metadata_from_router.get('original_filename', target_metadata_from_router.get('target_table_name', 'N/A')),
                            "source_type_from_meta": target_metadata_from_router.get('source_type'),
                            "target_db_from_meta": target_metadata_from_router.get('target_database'),
                            "target_table_from_meta": target_metadata_from_router.get('target_table_name')
                        }
                    })
            else:
                logger.error("QueryRouter not available. Defaulting to VECTOR_STORE.")
                decision = 'VECTOR_STORE'
                target_metadata_from_router = None
                reasoning_log_entries.append({"step_title": "Routing Error", "details": "QueryRouter unavailable. Defaulting to VECTOR_STORE."})

            if decision == 'NONE':
                needs_direct_fallback = True
                reasoning_log_entries.append({"step_title": "Query Handling", "details": "Route decision is 'NONE'. Proceeding to fallback."})
        except Exception as route_err:
             logger.error(f"Fatal routing error: {route_err}", exc_info=True); needs_direct_fallback = True
             reasoning_log_entries.append({"step_title": "Routing Error", "details": f"Fatal routing error: {str(route_err)}"})

        if not needs_direct_fallback:
            try:
                if decision.startswith('SQL:'):
                    parts = decision.split(':', 2)
                    db_key = parts[1]
                    target_identifier = parts[2]
                    target_db_utility = self.uploads_db_utility if db_key == 'UPLOADS' else self.rag_db_utility
                    reasoning_log_entries.append({"step_title": "SQL Execution", "details": f"Preparing for SQL query against {db_key} DB."})

                    if target_db_utility and target_metadata_from_router and target_metadata_from_router.get('identifier') == target_identifier:
                        db_name = getattr(target_db_utility._engine.url, 'database', 'Unknown DB')
                        table_name = target_metadata_from_router.get("target_table_name", target_identifier)
                        logger.info(f"Executing SQL route for: {db_name}/{table_name} (ID: {target_identifier})")
                        reasoning_log_entries.append({
                            "step_title": "SQL Target",
                            "details": { "database": db_name, "table": table_name, "identifier": target_identifier }
                        })
                        try:
                            raw_sql = generate_sql_query(query_text, target_db_utility, target_metadata_from_router)
                            if not raw_sql: raise ValueError("SQL generation failed.")
                            cleaned_sql = clean_sql_string(raw_sql)
                            if not cleaned_sql: raise ValueError("Generated SQL invalid.")
                            reasoning_log_entries.append({"step_title": "Generated SQL", "details": {"raw": raw_sql, "cleaned": cleaned_sql}})

                            db_execution_result_str = target_db_utility.run(cleaned_sql)
                            if db_execution_result_str.strip():
                                final_answer = f"```\n{db_execution_result_str}\n```"
                            else:
                                final_answer = "The SQL query executed successfully but returned no data."
                            reasoning_log_entries.append({"step_title": "SQL Result", "details": str(db_execution_result_str)[:1000] + ("..." if len(str(db_execution_result_str)) > 1000 else "")})

                            sql_source_info = {
                                "source": f"{target_metadata_from_router.get('original_filename', target_identifier)} (DB Table: {table_name})",
                                "type": "sql_result", 
                                "details": "Executed SQL query",
                                "query": cleaned_sql,
                                "content_snippet": str(db_execution_result_str)[:200]+"...",
                                "identifier": target_identifier
                            }
                            if 'summary_id' in target_metadata_from_router: sql_source_info['summary_id'] = target_metadata_from_router['summary_id']
                            sources_for_response = [sql_source_info]
                            used_structured_source_id_this_turn = target_identifier
                        except Exception as e:
                            logger.error(f"SQL execution failed for table '{table_name}': {e}", exc_info=True)
                            final_answer = f"I encountered an error while trying to query the data from '{target_identifier}'. The technical error was: {str(e)}"; sources_for_response = []
                            reasoning_log_entries.append({"step_title": "SQL Error", "details": f"Failed for table '{table_name}': {str(e)}"})
                    else:
                         logger.error(f"SQL route chosen for '{target_identifier}', but configuration or metadata is missing/mismatched.")
                         final_answer = "I can't access the required database or information for this SQL query due to a configuration issue."; sources_for_response = []
                         reasoning_log_entries.append({"step_title": "SQL Error", "details": f"Configuration error for SQL target '{target_identifier}'."})

                elif decision.startswith('PYTHON_ANALYSIS'):
                    logger.info(f"Routed to Python Analysis. Decision: {decision}")
                    reasoning_log_entries.append({"step_title": "Python Analysis", "details": "Initiating Python code generation and execution."})
                    current_structured_target_id: Optional[str] = None

                    if ':' in decision:
                        try:
                            _, target_identifier_from_decision = decision.split(':', 1)
                            if target_metadata_from_router and target_metadata_from_router.get('identifier') == target_identifier_from_decision:
                                current_structured_target_id = target_identifier_from_decision
                                reasoning_log_entries.append({"step_title": "Python Analysis Target", "details": f"Specific dataset targeted: {current_structured_target_id}"})
                            else:
                                reasoning_log_entries.append({"step_title": "Python Analysis Target", "details": f"Mismatched/Missing metadata for specific target {target_identifier_from_decision}. Proceeding with general Python analysis."})
                                target_metadata_from_router = None 
                        except ValueError:
                             reasoning_log_entries.append({"step_title": "Python Analysis Target", "details": f"Could not parse target ID from '{decision}'. Proceeding with general Python analysis."})
                             target_metadata_from_router = None 
                    else: 
                        reasoning_log_entries.append({"step_title": "Python Analysis Target", "details": "General Python analysis (no specific dataset pre-targeted by router)."})
                        target_metadata_from_router = None 

                    if not self.ds_executor:
                         final_answer = "The Python analysis service is currently unavailable."; sources_for_response = [{"source": "System", "type": "error", "content_snippet": final_answer}]
                         reasoning_log_entries.append({"step_title": "Python Analysis Error", "details": "DataScienceExecutor unavailable."})
                    else:
                        data_info = "No specific data source targeted for this Python execution.";
                        initialization_code = "# No specific data source targeted by the router. df will be an empty DataFrame unless loaded by LLM-generated code."
                        user_code_instructions = f"# Write Python code to answer the query: {query_text}. If data loading is required, you must include the necessary code."
                        
                        if target_metadata_from_router and current_structured_target_id:
                            source_id = current_structured_target_id
                            
                            table_name_py = target_metadata_from_router.get('target_table_name')
                            db_name_meta_py = target_metadata_from_router.get('target_database')

                            # REVISED CONDITION: If target_table_name and target_database are present, it's a DB load.
                            if table_name_py and db_name_meta_py:
                                try:
                                    base_db_url_str_py = settings.postgres_uploads_url if db_name_meta_py == 'RAG_DB_UPLOADS' else settings.postgres_url
                                    parsed_url_py = urlparse(base_db_url_str_py)
                                    
                                    db_user_py = parsed_url_py.username
                                    db_pass_py = parsed_url_py.password
                                    if not db_user_py and '@' in base_db_url_str_py.split('://', 1)[-1]:
                                        creds_part = base_db_url_str_py.split('://', 1)[-1].split('@', 1)[0]
                                        if ':' in creds_part:
                                            db_user_py = creds_part.split(':', 1)[0]
                                            db_pass_py = creds_part.split(':', 1)[1]
                                    db_user_py = db_user_py or 'postgres' 
                                    db_pass_py = db_pass_py or 'password' 

                                    db_host_py = 'db' 
                                    db_port_py = 5432 
                                    db_name_py = parsed_url_py.path.lstrip('/') if parsed_url_py.path else db_name_meta_py
                                    if not db_name_py: raise ValueError(f"Could not determine DB name from URL {base_db_url_str_py}")

                                    correct_db_url_py = f"postgresql+psycopg://{db_user_py}:{db_pass_py}@{db_host_py}:{db_port_py}/{db_name_py}"
                                    
                                    dataset_desc_py = target_metadata_from_router.get('document', target_metadata_from_router.get('dataset_description', 'N/A'))
                                    row_count_py = target_metadata_from_router.get('row_count', 'N/A')
                                    columns_py = target_metadata_from_router.get('columns', [])
                                    
                                    data_info_parts_py = [
                                        f"Target: Table `{table_name_py}` from Database `{db_name_py}`.",
                                        f"Original Source File (if applicable): {target_metadata_from_router.get('original_filename', 'N/A')}",
                                        f"Description: {dataset_desc_py}",
                                        f"Number of Rows: {row_count_py}",
                                        "Columns (Name, Type, Semantic Type, Description):"
                                    ]
                                    for col in columns_py:
                                        col_desc_val = col.get('description', 'N/A')
                                        col_semantic_val = col.get('semantic_type', 'N/A')
                                        data_info_parts_py.append(f"  - \"{col.get('column', 'N/A')}\" (Type: {col.get('dtype', 'N/A')}, Semantic: {col_semantic_val}, Desc: {col_desc_val})")
                                    
                                    schema_details = target_metadata_from_router.get("table_schema_details")
                                    if schema_details:
                                        pks = schema_details.get("primary_keys")
                                        fks = schema_details.get("foreign_keys")
                                        comment = schema_details.get("table_comment")
                                        if pks: data_info_parts_py.append(f"Primary Keys: {', '.join(pks)}")
                                        if fks: data_info_parts_py.append(f"Foreign Keys: {json.dumps(fks, indent=2)}")
                                        if comment: data_info_parts_py.append(f"Table Comment: {comment}")

                                    data_info = "\n".join(data_info_parts_py)
                                    
                                    initialization_code = (
                                        f"import pandas as pd\n"
                                        f"from sqlalchemy import create_engine\n"
                                        f"df = pd.DataFrame() # Initialize df as empty DataFrame FIRST\n"
                                        f"engine = None\n" 
                                        f"try:\n"
                                        f"    db_url = '{correct_db_url_py}'\n"
                                        f"    engine = create_engine(db_url)\n"
                                        f"    table_name_to_query = \"{table_name_py}\"\n" 
                                        f"    query = f'SELECT * FROM {{table_name_to_query}} LIMIT 10000'\n"
                                        f"    df_loaded = pd.read_sql_query(query, engine)\n"
                                        f"    if not df_loaded.empty:\n"
                                        f"        df = df_loaded\n"
                                        f"        print(f'Successfully loaded DataFrame `df` from table \"{table_name_py}\" in DB \"{db_name_py}\". Shape: {{df.shape}}')\n"
                                        f"    elif df_loaded.empty:\n" 
                                        f"        print(f'Successfully queried table \"{table_name_py}\" in DB \"{db_name_py}\", but it returned no data. DataFrame `df` remains empty.')\n"
                                        f"except Exception as e_load:\n"
                                        f"    db_url_for_error = '{correct_db_url_py}'\n" 
                                        f"    table_name_for_error = '{table_name_py}'\n"
                                        f"    print(f'Error during DataFrame initialization from table \"{{table_name_for_error}}\" (DB URL: {{db_url_for_error}}): {{e_load}}')\n"
                                        f"finally:\n"
                                        f"    if engine is not None:\n"
                                        f"        try:\n"
                                        f"            engine.dispose()\n"
                                        f"        except Exception as e_dispose:\n"
                                        f"            print(f'Error disposing database engine: {{e_dispose}}')\n"
                                    )
                                    user_code_instructions = f"# Analyze DataFrame 'df' (loaded from table '{table_name_py}'). The user query is: {query_text}"
                                    reasoning_log_entries.append({"step_title": "Python Data Context (DB Load Prepared)", "details": data_info})
                                except Exception as prep_err_py:
                                    data_info = f"Error preparing DB context for Python execution for {source_id}: {prep_err_py}"
                                    initialization_code = f"# DB context preparation failed for {source_id}. df will be an empty DataFrame."
                                    user_code_instructions = f"# General Python code; specific data loading failed for {source_id}. Query: {query_text}"
                                    reasoning_log_entries.append({"step_title": "Python Data Context Error (DB Prep)", "details": data_info})
                                    current_structured_target_id = None 
                            else: 
                                data_info = f"Target {source_id} is structured but is missing DB/table name information in its metadata. Cannot load into DataFrame automatically."
                                initialization_code = f"# Cannot automatically load {source_id} as DB details are missing. df will be an empty DataFrame."
                                user_code_instructions = f"# General Python code. Query: {query_text}"
                                reasoning_log_entries.append({"step_title": "Python Data Context Error (Missing DB Info)", "details": data_info})
                                current_structured_target_id = None
                        
                        code_gen_llm = self.generator.llm
                        if not code_gen_llm or not PYTHON_GENERATION_PROMPT: raise ValueError("Code gen LLM or Prompt unavailable.")
                        code_gen_chain = PYTHON_GENERATION_PROMPT | code_gen_llm | StrOutputParser()
                        code_gen_input = { "question": query_text, "data_info": data_info, "history_context": conversation_history_context or "No context.", "initialization_code": initialization_code, "user_code_instructions": user_code_instructions }
                        raw_analysis_code = code_gen_chain.invoke(code_gen_input); generated_analysis_code = self._clean_python_code(raw_analysis_code)
                        
                        full_code_to_execute = initialization_code + "\n\n# --- Analysis Code Generated by LLM ---\n" + generated_analysis_code
                        
                        reasoning_log_entries.append({"step_title": "Generated Python Code", "details": {"full_code_preview": full_code_to_execute[:1500]+"...", "analysis_code_preview": generated_analysis_code}})
                        logger.info(f"--- Full Code Sent to Executor (Target: {current_structured_target_id or 'General Analysis'}) ---\n{full_code_to_execute[:1000]}...\n---------------------------------")

                        try:
                            logger.info("Executing combined Python code..."); execution_result = self.ds_executor.execute_analysis(full_code_to_execute)
                            exec_stdout = execution_result.get("stdout", "").strip()
                            exec_stderr = execution_result.get("stderr", "").strip()
                            plot_data_received = execution_result.get('plot_png_base64')
                            exec_success = execution_result.get("execution_successful", False); client_error = execution_result.get("error")
                            
                            python_exec_log_details = {
                                "stdout": exec_stdout, 
                                "stderr": exec_stderr, 
                                "plot_generated": bool(plot_data_received), 
                                "execution_successful_on_executor": exec_success, 
                                "client_communication_error": client_error
                            }
                            reasoning_log_entries.append({"step_title": "Python Execution Result", "details": python_exec_log_details})

                            if client_error: 
                                final_answer = f"I encountered an issue communicating with the Python execution service: {client_error}."
                                if exec_stderr: final_answer += f" Details from executor (if any): {exec_stderr}"
                                sources_for_response = [{"source": "Executor Client", "type": "error", "content_snippet": final_answer}]
                            elif not exec_success: 
                                final_answer = "The Python analysis ran into an error."
                                if exec_stderr: final_answer += f" The error reported was: {exec_stderr}"
                                if exec_stdout and not exec_stderr: 
                                     final_answer += f"\nOutput before error: {exec_stdout}" 
                                elif not exec_stdout and not exec_stderr:
                                     final_answer += " No specific error message was captured, but the execution failed."
                                sources_for_response = [{"source": "Python Execution Error", "type": "error", "content_snippet": exec_stderr[:300]+"..." if exec_stderr else final_answer}]
                            else: 
                                final_answer = exec_stdout if exec_stdout else "The Python analysis completed successfully."
                                if plot_data_received and not exec_stdout:
                                    final_answer = "I've generated a plot based on your request."
                                elif plot_data_received and exec_stdout:
                                     final_answer = exec_stdout + "\n\nI've also generated a plot for this analysis."
                                
                                analysis_source_info = {"source": "Python Analysis Output", "type": "code_execution_result", "content_snippet": exec_stdout[:200]+"..." if exec_stdout else "Analysis performed successfully"}
                                if current_structured_target_id and target_metadata_from_router:
                                     analysis_source_info['source'] = f"{target_metadata_from_router.get('original_filename', current_structured_target_id)} (via Python)"
                                     analysis_source_info['identifier'] = current_structured_target_id
                                     if 'summary_id' in target_metadata_from_router: analysis_source_info['summary_id'] = target_metadata_from_router['summary_id']
                                     used_structured_source_id_this_turn = current_structured_target_id
                                sources_for_response = [analysis_source_info]
                                if plot_data_received:
                                    sources_for_response.append({"source": "Generated Plot", "type": "plot_png_base64", "identifier": f"plot_output_{uuid.uuid4().hex[:8]}", "data": plot_data_received})
                        
                        except DataScienceExecutorError as dse_err: 
                            final_answer = f"A critical error occurred with the Python analysis service: {dse_err}"
                            sources_for_response = [{"source": "Executor Service Client", "type": "error", "content_snippet": str(dse_err)[:300]+"..."}]
                            reasoning_log_entries.append({"step_title": "Python Execution Error (Executor Client)", "details": str(dse_err)})
                        except Exception as exec_e: 
                            final_answer = f"An unexpected error occurred during the Python analysis phase: {exec_e}"
                            sources_for_response = [{"source": "System Error", "type": "error", "content_snippet": final_answer}]
                            reasoning_log_entries.append({"step_title": "Python Execution Error (Unexpected)", "details": str(exec_e)})

                elif decision == 'VECTOR_STORE':
                    logger.info("Routed to Vector Store."); RELEVANCE_THRESHOLD = getattr(settings, 'relevance_threshold', 0.75)
                    reasoning_log_entries.append({"step_title": "Vector Retrieval", "details": "Fetching documents from vector store."})
                    retrieved_docs = self.retriever_wrapper.retrieve_documents(query_text)
                    retrieved_doc_details = [{"source": doc.metadata.get('source', f'doc_{i}'), "score": doc.metadata.get('relevance_score', 'N/A'), "content_preview": doc.page_content[:100]+"..."} for i, doc in enumerate(retrieved_docs[:5])]
                    reasoning_log_entries.append({"step_title": "Initial Retrieval", "details": f"Retrieved {len(retrieved_docs)} documents. Top {len(retrieved_doc_details)} (sample):", "data": retrieved_doc_details})

                    if not retrieved_docs:
                        logger.warning("Vector store retrieval returned 0 documents.")
                        reasoning_log_entries.append({"step_title": "Retrieval Result", "details": "No documents found in vector store."})
                        needs_direct_fallback = True
                    else:
                        high_score_docs = [doc for doc in retrieved_docs if float(doc.metadata.get('relevance_score', 0.0)) >= RELEVANCE_THRESHOLD]
                        if not high_score_docs:
                            logger.warning(f"Vector docs retrieved but none met threshold {RELEVANCE_THRESHOLD}.")
                            reasoning_log_entries.append({"step_title": "Filtering Result", "details": f"No documents met relevance threshold ({RELEVANCE_THRESHOLD})."})
                            needs_direct_fallback = True
                        else:
                             logger.info(f"Retrieved {len(high_score_docs)} relevant documents for RAG.")
                             filtered_doc_details = [{"source": doc.metadata.get('source', f'doc_{i}'), "score": doc.metadata.get('relevance_score', 'N/A'), "content_preview": doc.page_content[:100]+"..."} for i, doc in enumerate(high_score_docs[:settings.reranker_top_n])]
                             reasoning_log_entries.append({"step_title": "Filtered/Re-ranked Documents", "details": f"Found {len(high_score_docs)} documents meeting threshold. Using top {len(filtered_doc_details)} for generation (sample):", "data": filtered_doc_details})
                             initial_answer, is_sufficient = self.generator.generate_answer(query=query_text, retrieved_docs=high_score_docs, conversation_history_context=conversation_history_context)
                             reasoning_log_entries.append({"step_title": "Answer Generation", "details": f"Generated answer from RAG context. Sufficient: {is_sufficient}", "answer_preview": initial_answer[:200]+"..."})
                             sources_for_response = RAGResponse.format_sources(high_score_docs)
                             if not is_sufficient and not strict_rag_mode:
                                 needs_supplemental_fallback = True
                                 reasoning_log_entries.append({"step_title": "Sufficiency Check", "details": "Answer insufficient, strict mode OFF. Will attempt supplemental fallback."})
                             elif not is_sufficient and strict_rag_mode:
                                 final_answer = initial_answer
                                 reasoning_log_entries.append({"step_title": "Sufficiency Check", "details": "Answer insufficient, strict mode ON. No fallback."})
                             else:
                                 final_answer = initial_answer
                                 reasoning_log_entries.append({"step_title": "Sufficiency Check", "details": "Answer sufficient."})
                    used_structured_source_id_this_turn = None
            except Exception as e:
                logger.error(f"Error during '{decision}' execution phase: {e}", exc_info=True)
                needs_direct_fallback = True
                final_answer = f"An internal error occurred while processing your request via the '{decision}' route."; sources_for_response = []
                reasoning_log_entries.append({"step_title": f"Error in {decision} Phase", "details": str(e)})

        if needs_direct_fallback:
            logger.info(f"Executing direct fallback for query '{query_text[:50]}...' (Strict RAG: {strict_rag_mode})")
            reasoning_log_entries.append({"step_title": "Fallback Triggered", "details": "Executing direct fallback."})
            used_structured_source_id_this_turn = None 
            if strict_rag_mode:
                final_answer = "I can only answer based on the provided documents, and they do not contain an answer to your query."
                sources_for_response = [{"source": "System Information", "type": "info", "content_snippet": "Query could not be answered from available documents. General knowledge fallback disabled in strict RAG mode."}]
                reasoning_log_entries.append({"step_title": "Strict RAG Fallback", "details": final_answer})
            elif not self.generator.llm:
                final_answer = "The main answering service is currently unavailable."; sources_for_response = []
                reasoning_log_entries.append({"step_title": "Fallback Error", "details": "LLM unavailable for fallback."})
            else:
                try:
                    history_context_for_fallback = "general knowledge"; fallback_prompt_to_use = FALLBACK_PROMPT; fallback_input = {"question": query_text}
                    if conversation_history_context: fallback_prompt_to_use = FALLBACK_PROMPT_WITH_HISTORY; fallback_input = {"question": query_text, "history_context": conversation_history_context}; history_context_for_fallback = "conversation context and general knowledge"
                    reasoning_log_entries.append({"step_title": "General Knowledge Fallback", "details": f"Attempting to answer using {history_context_for_fallback}."})
                    fallback_chain = fallback_prompt_to_use | self.generator.llm | StrOutputParser(); fallback_answer_content = fallback_chain.invoke(fallback_input)
                    
                    final_answer_prefix = ""
                    if not decision or decision == 'NONE' or (decision == 'VECTOR_STORE' and not high_score_docs) :
                        final_answer_prefix = "I couldn't find specific documents related to your query. "
                    
                    final_answer = f"{final_answer_prefix}Based on general knowledge: {fallback_answer_content}"
                    sources_for_response = [{"source": "LLM Internal Knowledge", "type": "internal_knowledge", "content_snippet": f"Answer generated using {history_context_for_fallback}."}]
                    reasoning_log_entries.append({"step_title": "Fallback Answer", "details": fallback_answer_content[:200]+"..."})
                except Exception as fallback_err:
                    logger.error(f"Error during direct fallback generation: {fallback_err}", exc_info=True); final_answer = "I encountered an error while trying to answer from general knowledge."; sources_for_response = []
                    reasoning_log_entries.append({"step_title": "Fallback Error", "details": str(fallback_err)})

        elif needs_supplemental_fallback: 
            logger.info(f"Executing supplemental fallback for query '{query_text[:50]}...' (Strict RAG: {strict_rag_mode})")
            reasoning_log_entries.append({"step_title": "Fallback Triggered", "details": "Executing supplemental fallback as RAG answer was insufficient and strict mode is OFF."})
            if not self.generator.llm:
                final_answer = (initial_answer or "") + "\n\nThe main answering service is unavailable to supplement this."
                reasoning_log_entries.append({"step_title": "Supplemental Fallback Error", "details": "LLM unavailable."})
            else:
                try:
                    fallback_prompt_to_use = FALLBACK_PROMPT; fallback_input = {"question": query_text} 
                    if conversation_history_context: fallback_prompt_to_use = FALLBACK_PROMPT_WITH_HISTORY; fallback_input = {"question": query_text, "history_context": conversation_history_context}
                    
                    reasoning_log_entries.append({"step_title": "Supplemental Knowledge", "details": "Attempting to supplement RAG answer with general knowledge."})
                    fallback_chain = fallback_prompt_to_use | self.generator.llm | StrOutputParser(); fallback_supplement = fallback_chain.invoke(fallback_input)
                    
                    final_answer = f"{initial_answer}\n\nAdditionally, based on general knowledge:\n{fallback_supplement}"
                    
                    supplemental_source_info = {"source": "LLM Internal Knowledge (Supplement)", "type": "internal_knowledge_supplement", "content_snippet": "RAG answer was supplemented with general knowledge."}
                    if not isinstance(sources_for_response, list): sources_for_response = [] 
                    sources_for_response.append(supplemental_source_info)
                    reasoning_log_entries.append({"step_title": "Supplemental Answer Part", "details": fallback_supplement[:200]+"..."})
                except Exception as fallback_err:
                    logger.error(f"Error during supplemental fallback generation: {fallback_err}", exc_info=True);
                    final_answer = (initial_answer or "") + "\n\nI encountered an error while trying to supplement the answer with general knowledge."
                    reasoning_log_entries.append({"step_title": "Supplemental Fallback Error", "details": str(fallback_err)})

        generate_sugg = bool(final_answer and not final_answer.startswith("Sorry") and not final_answer.startswith("I encountered an error") and not final_answer.startswith("The main answering service"))
        if generate_sugg and self.light_llm:
             try:
                 suggestions = self._generate_suggestions(query_text, final_answer)
                 if suggestions:
                     reasoning_log_entries.append({"step_title": "Follow-up Suggestions", "details": "Generated suggestions.", "data": suggestions})
             except Exception as sugg_err:
                 logger.error(f"Suggestion generation failed: {sugg_err}"); suggestions = None
                 reasoning_log_entries.append({"step_title": "Suggestion Error", "details": str(sugg_err)})
        else: suggestions = None

        if not isinstance(final_answer, str) or not final_answer.strip():
            final_answer = "I could not generate a valid response to your query."
            reasoning_log_entries.append({"step_title": "Response Error", "details": "Final answer was empty or invalid."})
        if not isinstance(sources_for_response, list): sources_for_response = []


        logger.info(f"Query processing complete: '{query_text[:50]}...'. Used structured source: {used_structured_source_id_this_turn}")
        reasoning_log_entries.append({"step_title": "Processing Complete", "details": "Response formulated."})
        return RAGResponse(
            answer=final_answer.strip(),
            sources=sources_for_response,
            suggestions=suggestions,
            used_structured_source_id=used_structured_source_id_this_turn,
            reasoning_log=reasoning_log_entries
        )

    def __del__(self):
        if hasattr(self, 'ds_executor') and self.ds_executor:
            try: self.ds_executor.close()
            except Exception as e: logger.error(f"Error closing DataScienceExecutor session: {e}", exc_info=True)