# RAG_Project/MY_RAG/Backend/routing_logic.py

import logging
import json
import re # Import re for regex
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from config import settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_STORE_DIR = PROJECT_ROOT / "Metadata_Store"

ROUTE_TYPE_PROMPT_TEMPLATE = """
You are an expert query classification assistant. Your sole responsibility is to analyze the user's query, conversation context, and an overview of available data (both structured and unstructured summaries) to determine the most appropriate general route for processing the query. You must be precise and output only the determined route type.

---
### Available Data Overview
*High-level summaries of the types of data available for querying. Pay close attention to keywords and descriptions.*

**Structured Data Summary (List of available datasets and their general content):**
{structured_data_overview}
*(Example Format: "1. MSL Engagement Activity (ID: structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67): Covers MSL calls, attendees, HCP/HCO info. 2. Prescription Sales (ID: structured-Prescription_Transactions_SALES_.xlsx-2297d57a): Includes prescription transactions, product details, pricing, customer info.")*

**Unstructured Data Overall Summary (JSON object with overall_summary, domain_themes, keywords):**
{unstructured_overall_summary}
*(Focus on 'overall_summary' for general topics like "lung cancer research", "clinical trials", and 'keywords' like "EGFR", "nivolumab" to understand the textual content.)*
---

### Current Conversation Context
*Includes conversation history and a summary to provide necessary background.*

{history_context}
---

### User Query
*The user's most recent request.*

"{query_text}"
---

### Instructions & Decision Hierarchy

Your decision-making process must follow these steps:

**1. Analyze Query Intent & Data Needs:**
   - Carefully examine the `User Query`. What is the user trying to achieve?
   - Consider the `Conversation Context`. Is this a follow-up or a new line of inquiry?
   - Refer to the `Available Data Overview`.
     - Does the query relate to topics mentioned in the `Unstructured Data Overall Summary` (e.g., "lung cancer," "biomarkers," "EGFR mutations")?
     - Does the query seem to target specific datasets described in the `Structured Data Summary` (e.g., "MSL engagement," "prescription sales," "HCP interactions," "product pricing")?

**2. Determine the Primary Route Type:**

   - **Choose 'SQL':**
     - **IF** the query explicitly asks for precise data points, filtering, or simple aggregations (e.g., "count of calls," "total sales for ProductA," "list all HCPs in Virginia," "what was the status of CALL_ID BT810248?").
     - **AND** the query clearly implies operating on a **SINGLE known structured dataset** mentioned in the `Structured Data Summary` (e.g., MSL Engagement, Prescription Sales).
     - Keywords: "get," "find," "list," "show me records," "count," "total," "status of."

   - **Choose 'PYTHON_ANALYSIS':**
     - **IF** the query implies a need for:
       - Complex analysis on structured data (e.g., "find correlations between call attendees and product discussed," "what are the trends in WAC_PRICE for ProductB over the past year?").
       - Statistical calculations beyond simple SQL aggregates.
       - Data visualization or plotting using structured data (e.g., "plot face-to-face calls by REGION_NAME," "create a chart of COST_PRICE vs. NET_PRICE for ProductC").
       - Data manipulation or transformation involving one or more structured datasets.
     - Keywords: "analyze," "plot," "compare trends," "correlate," "forecast," "transform data," "calculate statistics on."

   - **Choose 'VECTOR_STORE':**
     - **IF** the query is best answered by retrieving information, definitions, or summaries from textual documents, aligning with the `Unstructured Data Overall Summary`'s `overall_summary`, `domain_themes` ("Healthcare"), or `keywords` (e.g., "lung cancer," "clinical trials," "EGFR," "nivolumab").
     - Example Queries: "Explain the molecular pathogenesis of lung cancer," "What are common patient-centered outcomes in oncology trials?", "Tell me about treatments involving nivolumab."
     - **OR IF** it's a general knowledge question related to the domain (e.g., "What is a biomarker?").

   - **Choose 'NONE':**
     - **IF** the query seems unanswerable with any available data type.
     - **OR IF** it's purely conversational with no clear intent for data retrieval or analysis (e.g., "hello," "thank you").
     - **OR IF** the query is out of scope for data-driven responses.

**3. Negative Constraints:**
    - Do **NOT** choose `SQL` or `PYTHON_ANALYSIS` if the query is clearly a general knowledge question about, for example, "lung cancer research" or "biomarkers" that `VECTOR_STORE` can handle based on the `unstructured_overall_summary`.
    - Do **NOT** choose `VECTOR_STORE` if the query is asking for specific, structured data points like "number of virtual calls last month" or "total cost price for ProductA" which are clearly answerable from the datasets in `Structured Data Summary`.

**4. Output Format:**
   - Respond ONLY with the chosen route type: `SQL`, `PYTHON_ANALYSIS`, `VECTOR_STORE`, or `NONE`.
   - Do not include any other text, explanations, or pleasantries.

---
### Examples (Reflecting Your Data)

* **User Query:** "How many face-to-face calls were made by MSLs in the US-East region?"
    * **Structured Data Overview:** Contains "MSL Engagement Activity" data with 'FACE_TO_FACE_CALLS' and 'REGION_NAME'.
    * **Route Type Decision:** `SQL`

* **User Query:** "Can you plot the trend of WAC_PRICE for ProductA over the last 2 years?"
    * **Structured Data Overview:** Contains "Prescription Sales" data with 'WAC_PRICE', 'PRODUCT_NAME', and date columns like 'MONTH_ENDING_DATE'.
    * **Route Type Decision:** `PYTHON_ANALYSIS`

* **User Query:** "What are the latest findings on EGFR mutations in lung cancer treatment?"
    * **Unstructured Data Overall Summary:** Mentions "lung cancer research," "EGFR," "targeted therapy."
    * **Route Type Decision:** `VECTOR_STORE`

* **User Query:** "Compare the number of MSL calls for ProductA versus ProductB by specialty."
    * **Structured Data Overview:** "MSL Engagement Activity" data with 'PRODUCT_NAME', 'CALL_ID', 'SPECIALTY'.
    * **Route Type Decision:** `PYTHON_ANALYSIS` (due to comparison and grouping, likely beyond simple SQL for direct answer).

* **User Query:** "What is a clinical trial?"
    * **Unstructured Data Overall Summary:** Mentions "clinical trials" and "healthcare" domain.
    * **Route Type Decision:** `VECTOR_STORE`
---

**Route Type Decision:**
"""
ROUTE_TYPE_PROMPT = PromptTemplate(
    template=ROUTE_TYPE_PROMPT_TEMPLATE,
    input_variables=["structured_data_overview", "unstructured_overall_summary", "history_context", "query_text"]
)

DATASET_RECOMMENDATION_PROMPT_TEMPLATE = """
You are a precision-focused data source locator. Your specific task, given that a user's query requires structured data access (SQL or Python-based analysis), is to identify and recommend the single most relevant structured data source from the `Available Structured Data Sources`. You must be exact in your recommendation. Ignore any sources with "data_type": "error".

---
### Available Structured Data Sources
*Detailed information for each available structured dataset. This is a list of JSON objects, where each object represents a dataset and includes an 'id', a 'document' (overall description), and 'metadata' which contains 'identifier' (filename), 'columns' (list of column details: 'column' name, 'semantic_type', 'description', 'dtype').*

{available_structured_sources_details}
---

### Current Conversation Context
*Includes conversation history and a summary to provide background.*

{history_context}
---

### User Query
*The user's most recent request that has been determined to need structured data.*

"{query_text}"
---

### Potentially Relevant Last Used Structured Source
*The 'id' of the structured source used in the previous turn, if any (e.g., "structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67").*

{last_structured_source_id}
---

### Instructions & Decision Hierarchy

Your decision-making process must strictly follow these steps in order:

**1. Prioritize Explicit Follow-up (Strongest Signal):**
   - **IF** `{last_structured_source_id}` is **NOT empty** AND its `id` matches one of the `id` fields in the `Available Structured Data Sources` list,
   - **AND** the `User Query` clearly appears to be a direct continuation, refinement, or related question concerning the data within that specific source (identified by `{last_structured_source_id}`),
   - **THEN** you **MUST** recommend `{last_structured_source_id}`.
   - **Action:** Output the `id` string from `{last_structured_source_id}`.

**2. Identify Direct Match from Query Content & Source Identifiers:**
   - **IF** Step 1 does not apply,
   - **AND** the `User Query` explicitly mentions or strongly implies a specific dataset by its filename (from `metadata.identifier` like "MSL Engagement Activity with HCP and HCO.xlsx" or "Prescription Transactions (SALES).xlsx") or refers to specific column names (from `metadata.columns[N].column`) that uniquely point to one of the `Available Structured Data Sources`,
   - **THEN** recommend that source.
   - **Action:** Output the `id` of the matched data source.

**3. Semantic Match based on Query Intent and Source/Column Details:**
   - **IF** Steps 1 and 2 do not apply,
   - **THEN** carefully analyze the `User Query`'s intent against:
     - The overall `document` description of each source.
     - The `metadata.columns` list for each source, paying close attention to:
       - `column` (column name)
       - `semantic_type` (e.g., "product", "price", "status", "identifier", "timestamp", "customer", "address", "specialty")
       - `description` (of the column)
   - Find the single source whose `document` description AND relevant `columns` (by name, semantic type, or description) best align with answering the query. For example, a query about "cost of ProductA" should match a source with 'PRODUCT_NAME' and a 'COST_PRICE' or similar price-related column. A query about "MSL interactions with oncologists" should match a source with MSL activity and 'SPECIALTY' columns.
   - **Action:** Output the `id` of the best semantic match.

**4. Handle Ambiguity or No Specific Match:**
   - **IF** Steps 1, 2, and 3 do not yield a clear, single best source (e.g., the query is too vague, multiple sources seem equally plausible, or no source seems relevant despite the query needing structured data),
   - **THEN** you **MUST** respond with `NO_SPECIFIC_DATASET`.

**5. Output Format:**
   - Respond ONLY with the exact `id` of the chosen data source (e.g., `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67`) OR the literal string `NO_SPECIFIC_DATASET`.
   - Do not include any other text, explanations, or pleasantries.

---
### Examples (Using Your Metadata Structure)

* **User Query:** "Okay, from that MSL data, how many calls were virtual?"
    * **Last Used Structured Source ID:** `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67`
    * **Available Sources:** Includes the source with ID `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67` which has a 'VIRTUAL_CALLS' column.
    * **Recommended Source ID:** `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67`

* **User Query:** "What is the total WAC_PRICE for 'ProductB' from the prescription sales data?"
    * **Last Used Structured Source ID:** (empty or irrelevant)
    * **Available Sources:** Includes a source with `identifier`: "Prescription Transactions (SALES).xlsx" (ID: `structured-Prescription_Transactions_SALES_.xlsx-2297d57a`) which contains 'WAC_PRICE' and 'PRODUCT_NAME' columns.
    * **Recommended Source ID:** `structured-Prescription_Transactions_SALES_.xlsx-2297d57a`

* **User Query:** "I need to see engagement activities with HCPs specializing in Thoracic Oncology."
    * **Last Used Structured Source ID:** (empty or irrelevant)
    * **Available Sources:** The source `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67` has `document` description about "MSL engagement activities with HCPs" and `columns` including 'SPECIALTY' which might contain "Thoracic Oncology".
    * **Recommended Source ID:** `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67`

* **User Query:** "Show me financial performance."
    * **Last Used Structured Source ID:** (empty or irrelevant)
    * **Available Sources:** Both "MSL Engagement" and "Prescription Sales" might have financial aspects, but the query is too vague.
    * **Recommended Source ID:** `NO_SPECIFIC_DATASET`

* **User Query:** "List customers who had a 'Submitted' call status."
    * **Last Used Structured Source ID:** `structured-Prescription_Transactions_SALES_.xlsx-2297d57a`
    * **Available Sources:** The MSL engagement data (`structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67`) contains 'CALL_STATUS' and 'CUSTOMER_NAME'. The last used source is not relevant.
    * **Recommended Source ID:** `structured-MSL_Engagement_Activity_with_HCP_and_HCO.xlsx-4637ee67`
---

**Recommended Source ID:**
"""
DATASET_RECOMMENDATION_PROMPT = PromptTemplate(
    template=DATASET_RECOMMENDATION_PROMPT_TEMPLATE,
    input_variables=["available_structured_sources_details", "history_context", "query_text", "last_structured_source_id"]
)


class QueryRouter:
    def __init__(self,
                 light_llm: ChatGoogleGenerativeAI,
                 base_retriever: BaseRetriever,
                 rag_db_utility_available: bool,
                 uploads_db_utility_available: bool):
        self.light_llm = light_llm
        self.base_retriever = base_retriever
        self.rag_db_utility_available = rag_db_utility_available
        self.uploads_db_utility_available = uploads_db_utility_available
        self.structured_metadata_list: List[Dict[str, Any]] = []
        self.unstructured_summary_content: Dict[str, Any] = {}
        self._load_metadata_files()
        logger.info("QueryRouter initialized.")
        if not self.light_llm: logger.warning("QueryRouter: Light LLM not available.")
        if not self.structured_metadata_list: logger.warning("QueryRouter: structured_metadata.json not loaded/empty.")
        if not self.unstructured_summary_content.get("overall_summary"): logger.warning("QueryRouter: unstructured_summary.json not loaded/empty or missing summary.")

    def _load_metadata_files(self):
        structured_meta_path = METADATA_STORE_DIR / "structured_metadata.json"
        unstructured_summary_path = METADATA_STORE_DIR / "unstructured_summary.json"
        try:
            if structured_meta_path.exists():
                with open(structured_meta_path, 'r', encoding='utf-8') as f: self.structured_metadata_list = json.load(f)
                logger.info(f"Loaded {len(self.structured_metadata_list)} entries from {structured_meta_path}")
            else: logger.warning(f"{structured_meta_path} not found.")
        except Exception as e: logger.error(f"Error loading {structured_meta_path}: {e}", exc_info=True); self.structured_metadata_list = []
        try:
            if unstructured_summary_path.exists():
                with open(unstructured_summary_path, 'r', encoding='utf-8') as f: self.unstructured_summary_content = json.load(f)
                logger.info(f"Loaded unstructured summary from {unstructured_summary_path}")
            else: logger.warning(f"{unstructured_summary_path} not found.")
        except Exception as e: logger.error(f"Error loading {unstructured_summary_path}: {e}", exc_info=True); self.unstructured_summary_content = {}

    def _prepare_context_for_route_type_prompt(self) -> Tuple[str, str]:
        structured_overview_parts = []
        if self.structured_metadata_list:
            for i, item_dict in enumerate(self.structured_metadata_list[:20]):
                meta_content = item_dict.get("metadata", {})
                identifier = meta_content.get("identifier", f"s_src_{i+1}")
                description = item_dict.get("document", meta_content.get("dataset_description", "No desc."))
                source_type = meta_content.get("source_type", "N/A")
                name = meta_content.get("original_filename", meta_content.get("target_table_name", identifier))
                structured_overview_parts.append(f"- {name} (ID: {identifier}, Type: {meta_content.get('data_type', 'structured')} ({source_type})): {description[:100]}...")
            structured_overview_str = "\n".join(structured_overview_parts) if structured_overview_parts else "No structured datasets overview available."
            if len(self.structured_metadata_list) > 20: structured_overview_str += "\n- ... and more."
        else: structured_overview_str = "No structured metadata loaded."
        unstructured_summary_str = self.unstructured_summary_content.get("overall_summary", "No overall summary for unstructured data.")
        if not unstructured_summary_str.strip() or "No overall summary" in unstructured_summary_str : unstructured_summary_str = "General textual info might be available; no specific overall summary loaded."
        return structured_overview_str, unstructured_summary_str

    def _prepare_context_for_dataset_recommendation_prompt(self) -> str:
        if not self.structured_metadata_list: return "No structured data sources loaded for recommendation."
        formatted_sources = []
        for i, item_dict in enumerate(self.structured_metadata_list):
            metadata = item_dict.get("metadata", {})
            identifier = metadata.get("identifier", f"unknown_s_{i}")
            doc_summary = item_dict.get("document", metadata.get("dataset_description", "No desc."))
            name = metadata.get("original_filename", metadata.get("target_table_name", identifier))
            cols_str = ", ".join([col.get("column", "?") for col in metadata.get("columns", [])[:5]]) + ("..." if len(metadata.get("columns", [])) > 5 else "")
            schema_details_raw = metadata.get("table_schema_details", {})
            schema_parts = []
            if schema_details_raw.get("primary_keys"): schema_parts.append(f"PKs: {','.join(schema_details_raw['primary_keys'])}")
            fks_detail = [f"{','.join(fk.get('constrained_columns',[]))}->{fk.get('referred_table')}" for fk in schema_details_raw.get("foreign_keys", [])]
            if fks_detail: schema_parts.append(f"FKs: {'; '.join(fks_detail)}")
            schema_str = f" Schema: {', '.join(schema_parts)}" if schema_parts else ""
            source_entry = f"ID: {identifier}\nName: {name}\nDesc: {doc_summary[:150]}...\nCols: {cols_str}{schema_str}\nTbl: {metadata.get('target_table_name', 'N/A')}, DB: {metadata.get('target_database', 'N/A')}"
            formatted_sources.append(f"Source {i+1}:\n{source_entry}")
        return "\n---\n".join(formatted_sources) if formatted_sources else "No structured data sources formatted."

    def route_query(self, query_text: str, history_context: Optional[str], last_structured_source_id: Optional[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
        logger.info(f"Initiating two-step routing for query: '{query_text[:50]}...'")
        final_decision_type = 'VECTOR_STORE'
        target_metadata: Optional[Dict[str, Any]] = None

        if not self.light_llm:
            logger.error("Routing LLM unavailable. Falling back to VECTOR_STORE.")
            return 'VECTOR_STORE', None

        # --- Step 1: Determine Route Type ---
        try:
            logger.debug("Routing Step 1: Determining route type...")
            structured_overview, unstructured_summary = self._prepare_context_for_route_type_prompt()
            route_type_input = {
                "structured_data_overview": structured_overview,
                "unstructured_overall_summary": unstructured_summary,
                "history_context": history_context or "No conversation history.",
                "query_text": query_text
            }
            route_type_chain = ROUTE_TYPE_PROMPT | self.light_llm | StrOutputParser()
            raw_route_type_decision = route_type_chain.invoke(route_type_input) # Get the raw string
            logger.info(f"Step 1 LLM Raw Output for Route Type: {raw_route_type_decision}")

            # Enhanced parsing for route type
            determined_route_type = "VECTOR_STORE" # Default
            possible_routes = ['SQL', 'PYTHON_ANALYSIS', 'VECTOR_STORE', 'NONE']
            
            # Try to find the keyword after "ROUTE TYPE:" if present
            match = re.search(r"ROUTE TYPE:\s*(" + "|".join(possible_routes) + ")", raw_route_type_decision.upper())
            if match:
                determined_route_type = match.group(1)
            else:
                # Fallback: Check if any keyword is present as a standalone word, prioritizing more specific ones
                # This simple check might need refinement if LLM output is very noisy.
                # We check if the raw output (stripped and uppercased) *ends with* one of the keywords.
                # Or simply contains it, assuming it's the most dominant part of a short answer.
                cleaned_raw_decision = raw_route_type_decision.strip().upper()
                for route in possible_routes:
                    if route in cleaned_raw_decision: # Check for substring presence
                        # If multiple are present, this might pick the first one in `possible_routes`.
                        # A more sophisticated logic might be needed for ambiguous outputs.
                        # For now, if "SQL" is anywhere, and it's the only one, it's SQL.
                        # If the output is *just* the keyword, this works.
                        # If the output has preamble then the keyword, this should also work.
                        # Let's try a direct match first for cleaner outputs
                        if cleaned_raw_decision == route:
                             determined_route_type = route
                             break
                # If no direct match, try substring logic as a fallback
                if determined_route_type == "VECTOR_STORE": # if still default after direct match attempt
                    for route in possible_routes:
                        if route in cleaned_raw_decision:
                            determined_route_type = route
                            logger.info(f"Route type keyword '{route}' found in noisy LLM output.")
                            break # Take the first one found

            logger.info(f"Step 1 LLM - Parsed Route Type: {determined_route_type}")

        except Exception as e:
            logger.error(f"Error in Step 1 (Route Type Determination): {e}", exc_info=True)
            return 'VECTOR_STORE', None

        # --- Step 2: Recommend Dataset (if SQL or PYTHON_ANALYSIS) ---
        if determined_route_type in ['SQL', 'PYTHON_ANALYSIS']:
            logger.debug(f"Routing Step 2: Recommending dataset for route type '{determined_route_type}'...")
            if not self.structured_metadata_list:
                logger.warning(f"No preloaded structured metadata for dataset recommendation (Route: {determined_route_type}).")
                if determined_route_type == 'PYTHON_ANALYSIS': return 'PYTHON_ANALYSIS', None
                else: return 'VECTOR_STORE', None
            try:
                formatted_structured_sources_details_str = self._prepare_context_for_dataset_recommendation_prompt()
                if "No structured data sources" in formatted_structured_sources_details_str : # Check if formatting yielded nothing useful
                     logger.warning(f"Formatting preloaded structured metadata yielded no usable details for dataset recommendation.")
                     if determined_route_type == 'PYTHON_ANALYSIS': return 'PYTHON_ANALYSIS', None
                     else: return 'VECTOR_STORE', None

                dataset_rec_input = {
                    "available_structured_sources_details": formatted_structured_sources_details_str,
                    "history_context": history_context or "No conversation history.",
                    "query_text": query_text,
                    "last_structured_source_id": last_structured_source_id or "None"
                }
                dataset_rec_chain = DATASET_RECOMMENDATION_PROMPT | self.light_llm | StrOutputParser()
                recommended_dataset_id = dataset_rec_chain.invoke(dataset_rec_input).strip() # LLM should return just the ID or NO_SPECIFIC_DATASET
                logger.info(f"Step 2 LLM - Recommended Dataset ID: '{recommended_dataset_id}'")

                if recommended_dataset_id and recommended_dataset_id != 'NO_SPECIFIC_DATASET':
                    chosen_item_dict = next(
                        (item_dict for item_dict in self.structured_metadata_list 
                         if item_dict.get("metadata", {}).get("identifier", '').lower() == recommended_dataset_id.lower()),
                        None)
                    if chosen_item_dict:
                        target_metadata = chosen_item_dict.get("metadata")
                        if not target_metadata:
                            logger.error(f"Item for ID '{recommended_dataset_id}' found but 'metadata' key missing/empty.")
                            target_metadata = None
                        else:
                            logger.info(f"Target dataset '{target_metadata.get('identifier')}' confirmed from structured_metadata.json.")
                        
                        if target_metadata:
                            if determined_route_type == 'SQL':
                                db_key = 'NONE'
                                target_db_name = target_metadata.get('target_database')
                                if target_db_name == 'RAG_DB_UPLOADS' and self.uploads_db_utility_available: db_key = 'UPLOADS'
                                elif target_db_name == 'RAG_DB' and self.rag_db_utility_available: db_key = 'MAIN'
                                if db_key != 'NONE':
                                    if 'target_table_name' not in target_metadata or not target_metadata['target_table_name']:
                                        target_metadata['target_table_name'] = recommended_dataset_id
                                    final_decision_type = f"SQL:{db_key}:{recommended_dataset_id}"
                                else:
                                    logger.warning(f"SQL route chose '{recommended_dataset_id}', DB utility for '{target_db_name}' unavailable. Falling back.")
                                    final_decision_type = 'VECTOR_STORE'; target_metadata = None
                            elif determined_route_type == 'PYTHON_ANALYSIS':
                                final_decision_type = f"PYTHON_ANALYSIS:{recommended_dataset_id}"
                        else: # target_metadata became None after lookup
                            if determined_route_type == 'PYTHON_ANALYSIS': final_decision_type = 'PYTHON_ANALYSIS';
                            else: final_decision_type = 'VECTOR_STORE';
                            target_metadata = None
                    else:
                        logger.warning(f"LLM recommended ID '{recommended_dataset_id}', not found in structured_metadata.json.")
                        if determined_route_type == 'PYTHON_ANALYSIS': final_decision_type = 'PYTHON_ANALYSIS'
                        else: final_decision_type = 'VECTOR_STORE'
                        target_metadata = None
                else: 
                    logger.info(f"LLM recommended 'NO_SPECIFIC_DATASET' or empty for dataset ID.")
                    if determined_route_type == 'PYTHON_ANALYSIS': final_decision_type = 'PYTHON_ANALYSIS'
                    else:
                        logger.warning(f"SQL route determined, but no specific dataset recommended. Falling back to VECTOR_STORE.")
                        final_decision_type = 'VECTOR_STORE'
                    target_metadata = None
            except Exception as e:
                logger.error(f"Error in Step 2 (Dataset Recommendation): {e}", exc_info=True)
                if determined_route_type == 'PYTHON_ANALYSIS': final_decision_type = 'PYTHON_ANALYSIS'
                else: final_decision_type = 'VECTOR_STORE'
                target_metadata = None
        elif determined_route_type == 'VECTOR_STORE':
            final_decision_type = 'VECTOR_STORE'
        elif determined_route_type == 'NONE':
            final_decision_type = 'NONE'
        else: # Unknown route type from Step 1 after parsing
            logger.warning(f"Unknown parsed route type '{determined_route_type}' from Step 1. Defaulting to VECTOR_STORE.")
            final_decision_type = 'VECTOR_STORE'

        logger.info(f"Final routing decision after two steps: {final_decision_type}")
        if target_metadata: logger.debug(f"Target metadata for final decision: ID='{target_metadata.get('identifier')}'")
        else: logger.debug("No specific target metadata for final decision.")
        return final_decision_type, target_metadata