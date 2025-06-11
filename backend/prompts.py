# RAG_Project/MY_RAG/Backend/prompts.py

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

# Langchain imports for prompt templating
from langchain_core.prompts import PromptTemplate

# Import ChatMessage definition (adjust path if necessary, consider shared models.py)
# Note: ChatMessage is not directly used here anymore, but kept for potential future use
from models import ChatMessage

logger = logging.getLogger(__name__)

# --- Prompts ---

# Basic RAG Prompt - Used when no conversation history/context is needed or provided by SessionHandler
# (Kept for potential use cases, though RAG_PROMPT_WITH_HISTORY might always be used now)
RAG_PROMPT_TEMPLATE_NO_HISTORY = """
CONTEXT:
{context}

QUERY:
{question}

INSTRUCTIONS:
Based *only* on the provided CONTEXT, answer the QUERY.
If the context does not contain the answer, state that the context is insufficient.

Use markdown for formatting if it enhances clarity (e.g., lists, bolding).

ANSWER:
"""
RAG_PROMPT_NO_HISTORY = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE_NO_HISTORY,
    input_variables=["context", "question"]
)


# RAG Prompt - Used when conversation context (summary + recent turns) IS available
# Receives the pre-formatted context string from SessionHandler
RAG_PROMPT_TEMPLATE_WITH_HISTORY = """
Conversation Context (may include summary and recent turns):
{history_context}

Retrieved Context Documents:
{context}

QUERY:
{question}

INSTRUCTIONS:
Use the provided 'Conversation Context' as reference to understand the flow and background of the query.
Answer the user's QUERY using *only* the information provided in the 'Retrieved Context Documents'.
If the 'Retrieved Context Documents' do not contain the answer, state that clearly. Do not use the 'Conversation Context' to answer the query directly, only for understanding background.
Be concise, accurate, and helpful.

Use markdown for formatting if it enhances clarity (e.g., lists, bolding).

ANSWER: """
RAG_PROMPT_WITH_HISTORY = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE_WITH_HISTORY,
    input_variables=["history_context", "context", "question"]
)


# Fallback Prompt - Used when no relevant RAG context is found and no history context is available
FALLBACK_PROMPT_TEMPLATE = """
QUERY:
{question}

INSTRUCTIONS:
Please answer the QUERY based on your general knowledge. If the query is highly specific and likely requires external documents you don't have access to, state that you cannot answer accurately without specific context.

Use markdown for formatting if it enhances clarity (e.g., lists, bolding).

ANSWER:
"""
FALLBACK_PROMPT = PromptTemplate.from_template(FALLBACK_PROMPT_TEMPLATE)


# Fallback Prompt - Used when no relevant RAG context is found BUT history context IS available
# Receives the pre-formatted context string from SessionHandler
FALLBACK_PROMPT_WITH_HISTORY_TEMPLATE = """
Conversation Context (may include summary and recent turns):
{history_context}

QUERY:
{question}

INSTRUCTIONS:
Use the provided 'Conversation Context' as reference to inform your response, maintaining continuity where appropriate. Answer the QUERY based on your general knowledge and the conversation context.
If the user's query is highly specific and appears to require external documents or RAG context (which was not found), clearly state that you cannot answer accurately without access to that specific information, even considering the conversation context.

Use markdown for formatting if it enhances clarity (e.g., lists, bolding).

ANSWER:
"""
FALLBACK_PROMPT_WITH_HISTORY = PromptTemplate.from_template(FALLBACK_PROMPT_WITH_HISTORY_TEMPLATE)


# Prompt for checking if the RAG answer is sufficient (Remains the same)
ANSWER_SUFFICIENCY_CHECK_PROMPT_TEMPLATE = """
Analyze the provided Query, Context, and the Answer generated *strictly* from that Context.
Determine if the Answer fully addresses the Query based *only* on the information present in the Context.

Context:
---
{context}
---

Query:
---
{question}
---

Answer Generated from Context:
---
{answer_from_context}
---

INSTRUCTIONS:
- If the "Answer Generated from Context" *fully* answers the "Query" using *only* information found in the "Context", respond with "SUFFICIENT".
- If the "Answer Generated from Context" correctly states limitations or cannot fully answer the "Query" because the necessary details are missing in the "Context", respond with "INSUFFICIENT".

Respond ONLY with the single word "SUFFICIENT" or "INSUFFICIENT".

Decision:
"""
ANSWER_SUFFICIENCY_CHECK_PROMPT = PromptTemplate.from_template(ANSWER_SUFFICIENCY_CHECK_PROMPT_TEMPLATE)


# --- REMOVED ROUTER_PROMPT_TEMPLATE and ROUTER_PROMPT ---


# --- Python Code Generation Prompt Template ---
PYTHON_GENERATION_PROMPT_TEMPLATE = """
You are an expert Python data scientist. Your task is to write Python code to answer the user's query based on the provided information about a pandas DataFrame (`df`).
The code will be executed in an environment where `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, and `seaborn as sns` are already imported. `matplotlib.use('Agg')` is also preset.

User Query:
---
{question}
---

Available Data Information (Metadata for DataFrame `df` loaded from the target source):
---
{data_info}
---
(This section contains metadata about the DataFrame 'df' that has been loaded for you, including description, row count, and column details like Name, Type, and Description.)

Conversation Context (may include summary and recent turns):
---
{history_context}
---
(This section provides context from the ongoing conversation, including a potential summary and the most recent exchanges.)

Instructions for Code Generation:

**1. Understand the Goal:**
   - Write Python code using the pre-loaded DataFrame `df` to directly address the User Query.
   - Use 'Conversation Context' to understand intent, but base code primarily on 'User Query' and 'Available Data Information'.
   - Focus *only* on answering the query based on the provided data.

**2. Available Libraries:**
   - You have `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, and `seaborn as sns`.
   - Use `seaborn` for statistical plots or when aesthetically pleasing defaults are desired (e.g., `sns.histplot()`, `sns.barplot()`, `sns.lineplot()`, `sns.scatterplot()`, `sns.heatmap()`). It often produces better-looking charts with less code.
   - Use `matplotlib.pyplot` for basic plots or fine-grained customization if Seaborn doesn't cover the need directly.
   - `matplotlib.use('Agg')` is already set for non-interactive backend.

**3. Data Handling (CRITICAL):**
   - Refer *strictly* to 'Available Data Information' for DataFrame `df` structure.
   - Use **EXACT** column names from metadata (e.g., `df['Weekly_Sales']`).
   - Pay attention to column 'Type' in metadata.
   - **Date Parsing:** If converting date columns, use `pd.to_datetime(df['Date_Column'], errors='coerce')`. If a specific format is known from metadata samples, use it (e.g., `format='%Y-%m-%d'`). Assign the result back: `df['Date_Column'] = pd.to_datetime(...)`.

**4. Plotting Aesthetics (IMPORTANT for Business Users):**
   - **Aim for a modern, clean, and professional look suitable for business presentations.**
   - **Utilize Seaborn's styling capabilities.** For example, set a theme at the beginning of your plotting code: `sns.set_theme(style='whitegrid', palette='viridis')` or other appropriate styles like 'darkgrid', 'white', 'ticks', and relevant color palettes (e.g., 'muted', 'colorblind').
   - Ensure charts have **clear and descriptive titles, axis labels (for both x and y axes), and legends** where appropriate. Use legible font sizes.
   - **Choose color palettes that are clear and professional.** Consider colorblind-friendly palettes if possible.
   - **Avoid clutter;** ensure the chart clearly communicates the key insights. For instance, if plotting time series data, ensure dates on the x-axis are formatted legibly (e.g., rotate labels if they overlap).
   - Make sure bar charts have appropriate bar widths and spacing.
   - For scatter plots, consider appropriate marker sizes and transparency if there are many points.

**5. Code Output:**
   - **Print Results:** Use `print()` for final answers, calculations, summaries.
   - **Table Formatting:** For DataFrame/list results, consider `tabulate` if complex: `print(tabulate(my_df, headers='keys', tablefmt='psql'))`. Simple `print(my_df)` is also fine.
   - **Plotting (Saving the Figure):**
      - If using Seaborn, create the plot (e.g., `ax = sns.lineplot(...)`). Seaborn plots on the current Matplotlib axes.
      - Add titles and labels using `plt.title()`, `plt.xlabel()`, `plt.ylabel()` or `ax.set_title()`, `ax.set_xlabel()`, etc.
      - Use `plt.legend()` if multiple series are plotted.
      - **Use `plt.tight_layout()`** to prevent labels from overlapping and ensure all elements fit well.
      - **Save the plot EXACTLY as `plot.png`**: `plt.savefig('plot.png', dpi=150)`. Using a dpi like 150 can improve clarity. This is crucial.
      - **Do NOT use `plt.show()`**.
      - After saving, clear the figure for the next potential plot: `plt.clf()` and `plt.close()`.

**6. Constraints & Safety:**
   - Write only the Python code needed. No surrounding text/explanations.
   - Do NOT import `os`, `subprocess`, `sys`, `matplotlib`, `seaborn`, `pandas`, `numpy` (they are pre-imported).
   - Do NOT read/write files except `plot.png`.
   - Stick to `df` operations and allowed libraries.

Write only the Python code required to answer the query.

Python Code:
```python
# Your Python code starts here
{initialization_code} # Code to initialize 'df' if data was loaded

# --- Debug: Print DataFrame Info (if df exists) ---
if 'df' in locals() and isinstance(df, pd.DataFrame):
    print("--- DataFrame Info (from LLM generated code) ---")
    print(f"Columns: {{df.columns.tolist()}}")
    if not df.empty:
        try:
            print(f"Data Types:\\n{{df.dtypes.to_string()}}")
        except Exception as e:
            print(f"Error printing df info: {{e}}")
    else:
        print("DataFrame is empty.")
    print("-------------------------------------------")
else:
    print("--- No DataFrame 'df' loaded or available for analysis ---")
# --- End Debug ---

# --- User Query Analysis Code ---
# Based on the query: {question}
# And data_info: {data_info}
# Apply appropriate Seaborn styling for a professional look, e.g., sns.set_theme(style='whitegrid')
{user_code_instructions} 

# REMINDER TO LLM: If a plot was generated above, ensure the sequence:
# plt.savefig('plot.png', dpi=150, bbox_inches='tight')
# plt.clf()
# plt.close('all')
# was explicitly called.
"""
PYTHON_GENERATION_PROMPT = PromptTemplate(
template=PYTHON_GENERATION_PROMPT_TEMPLATE,
input_variables=["question", "data_info", "history_context", "initialization_code", "user_code_instructions"] # Changed chat_history to history_context
)

#--- Conversational History Summarization Prompt ---
HISTORY_SUMMARIZATION_PROMPT_TEMPLATE = """
You are an expert conversation summarizer. Given the 'Existing Summary' (if any) and the 'Recent Conversation History', create a concise, updated summary.
The summary should capture the key topics, decisions, and unresolved questions from the entire conversation, integrating the recent history with the existing summary smoothly.
Focus on information relevant for maintaining context in future interactions. Keep the summary under 150 words.

Existing Summary:
{existing_summary}
Recent Conversation History (Oldest to Newest):
{conversation_history}
Updated Concise Summary (Max 150 words):
"""
HISTORY_SUMMARIZATION_PROMPT = PromptTemplate.from_template(HISTORY_SUMMARIZATION_PROMPT_TEMPLATE)