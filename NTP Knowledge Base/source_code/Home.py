import math
import os
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# -------------------------------------------------------
# Streamlit page setup
# -------------------------------------------------------
st.set_page_config(page_title="NTP Knowledgebase Chatbot", layout="wide")
st.title("💬 New Type Planning – Knowledgebase Expert Chatbot")

st.markdown(
    """
This chatbot behaves like a **New Type Planning (NTP) project expert**.

It answers **only** based on the curated BBAC NTP knowledgebase (lessons learned),
not from the open internet.

Try questions like:

- "What lessons do we have for conveyor projects in BIW pilot plant?"
- "Typical risks in B-Phase for sorter projects?"
- "What should I watch out for for new type planning of carline V214?"
"""
)

# Optional: if you keep bootstrap.py in the same folder as before
try:
    import bootstrap  # noqa: F401
except Exception:
    pass

# -------------------------------------------------------
# Azure OpenAI client
# -------------------------------------------------------
def get_azure_client() -> AzureOpenAI | None:
    """
    Initialize Azure OpenAI client from Streamlit secrets.

    Expected secrets:
    [azure_openai]
    endpoint = "https://<your-endpoint>.openai.azure.com/"
    api_key = "<your-api-key>"
    deployment_name = "<your-gpt-4o-deployment-name>"
    """
    try:
        client = AzureOpenAI(
            azure_endpoint=st.secrets["azure_openai"]["endpoint"],
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version="2024-02-01",
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {e}")
        return None


# -------------------------------------------------------
# Load knowledge base (static Excel, no upload)
# -------------------------------------------------------
@st.cache_resource(show_spinner="Loading New Type Planning knowledgebase...")
def load_kb_df() -> pd.DataFrame:
    """
    Load the BBAC NTP knowledgebase from a static Excel file.

    Place `Knowledgebase_BBAC_NTP_07-2025-01-30.xlsx` in the same folder as Home.py.
    Expected sheet: 'Knowledgebase LL BBAC'
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(root_dir, "Knowledgebase_BBAC_NTP_07-2025-01-30.xlsx")

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"Knowledgebase file not found at: {kb_path}\n"
            "Make sure 'Knowledgebase_BBAC_NTP_07-2025-01-30.xlsx' is in the app root."
        )

    df = pd.read_excel(kb_path, sheet_name="Knowledgebase LL BBAC")
    df = df.copy()

    # Normalize string columns and fill NaNs
    if "ID" not in df.columns:
        df["ID"] = df.index.astype(str)
    df["ID"] = df["ID"].astype(str).str.strip()

    for col in df.columns:
        df[col] = df[col].fillna("")

    return df


def build_row_text(row: pd.Series) -> str:
    """
    Build a single text blob per row for simple relevance scoring and LLM context.
    Adjust labels/columns if your Excel changes.
    """
    parts = []

    def add(label: str, value: Any):
        val = str(value).strip()
        if val:
            parts.append(f"{label}: {val}")

    add("ID", row.get("ID", ""))
    add("Shop", row.get("Shop", ""))
    add("Project Phase identified", row.get("Project Phase identified", ""))
    add("Relevant Project Phase", row.get("Relevant Project Phase", ""))
    add("Subject", row.get("Subject", ""))
    add("Subject Category", row.get("Subject Category", ""))
    add("Issue Description", row.get("Issue Description", ""))
    add("Cause of issues", row.get("Cause of issues", ""))
    add("Lessons Learned", row.get("Lessons Learned", ""))
    add("Benefits of using the lessons learned", row.get("Benefits of using the lessons learned", ""))
    add("Benefits Category", row.get("Benefits Category", ""))
    add("Responsible for optimisation", row.get("Responsible for optimisation", ""))
    add("Carline/Engine/Battery", row.get("Carline/Engine/Battery", ""))

    return "\n".join(parts)


@st.cache_resource(show_spinner="Indexing knowledgebase...")
def build_kb_index() -> Dict[str, Any]:
    """
    Simple in-memory index:

    - df: original dataframe
    - texts: concatenated text per row (for scoring + context)

    No embeddings here to keep infra simple. You can swap this later
    for an embedding-based index if needed.
    """
    df = load_kb_df()
    texts = [build_row_text(df.iloc[i]) for i in range(len(df))]
    return {"df": df, "texts": texts}


# -------------------------------------------------------
# Simple keyword-based retrieval (no embeddings)
# -------------------------------------------------------
def score_text(query: str, text: str) -> float:
    """
    Very simple relevance scoring:
    - tokenize query
    - score = term frequency / sqrt(text length)
    """
    if not query or not text:
        return 0.0

    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not tokens:
        return 0.0

    text_l = text.lower()
    tf = 0
    for t in tokens:
        tf += text_l.count(t)

    return tf / math.sqrt(len(text_l) + 1.0)


def search_kb(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Rank all lessons by simple keyword score and return top_k.
    Returns list of dicts: {score, row, text}.
    """
    kb = build_kb_index()
    df = kb["df"]
    texts = kb["texts"]

    scored: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        s = score_text(query, text)
        if s > 0:
            scored.append({"score": s, "row": df.iloc[idx], "text": text})

    # If no matches (or all 0), just give a few random-ish rows
    if not scored:
        for idx in range(min(top_k, len(df))):
            scored.append({"score": 0.0, "row": df.iloc[idx], "text": texts[idx]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_context_for_llm(results: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved rows into a context string for the LLM.
    """
    blocks = []
    for r in results:
        row = r["row"]
        blocks.append(
            "-----\n"
            f"Lesson ID: {row.get('ID', '')}\n"
            f"Shop: {row.get('Shop', '')}\n"
            f"Project Phase identified: {row.get('Project Phase identified', '')}\n"
            f"Relevant Project Phase: {row.get('Relevant Project Phase', '')}\n"
            f"Subject: {row.get('Subject', '')}\n"
            f"Subject Category: {row.get('Subject Category', '')}\n"
            f"Issue Description: {row.get('Issue Description', '')}\n"
            f"Cause of issues: {row.get('Cause of issues', '')}\n"
            f"Lessons Learned: {row.get('Lessons Learned', '')}\n"
            f"Benefits: {row.get('Benefits of using the lessons learned', '')}\n"
            f"Benefits Category: {row.get('Benefits Category', '')}\n"
            f"Responsible for optimisation: {row.get('Responsible for optimisation', '')}\n"
            f"Carline/Engine/Battery: {row.get('Carline/Engine/Battery', '')}\n"
        )
    return "\n".join(blocks)


# -------------------------------------------------------
# LLM call – NTP project expert persona
# -------------------------------------------------------
def call_ntp_expert(question: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    1) Retrieve relevant lessons from the knowledgebase
    2) Call Azure OpenAI (GPT-4o, via your deployment)
    3) Return answer + retrieved lessons (for transparency)
    """
    client = get_azure_client()
    if not client:
        return {"answer": "Error: Azure OpenAI client not available.", "retrieved": []}

    retrieved = search_kb(question, top_k=8)
    context = format_context_for_llm(retrieved)

    system_prompt = """
You are a New Type Planning (NTP) project expert at BBAC.

You ONLY use the information from the NTP knowledgebase context provided.
If the context does not contain enough information to answer confidently,
explicitly say that the knowledgebase does not fully cover the topic and
suggest what the planner should check (e.g. carline, phase, shop, stakeholder).

Guidelines:
- Talk like a planning engineer, not like a generic consultant.
- Highlight typical risks, do's & don'ts, and concrete actions.
- Refer to Lesson IDs explicitly when relevant
  (e.g. “According to Lesson ID 12 in BIW pilot plant...”).
- Structure answers with bullet points or short sections.
- Do NOT invent facts beyond the given context.
"""

    # Use short history to keep some conversational continuity
    history_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history[-6:]
    ]

    user_block = f"""
User question:
{question}

Context from BBAC NTP knowledgebase:
{context}

Answer as the BBAC NTP project expert.
"""

    response = client.chat.completions.create(
        model=st.secrets["azure_openai"]["deployment_name"],
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_block},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "retrieved": retrieved}


# -------------------------------------------------------
# UI – single-page chat interface
# -------------------------------------------------------

# Load KB once, mainly to fail fast if file missing
try:
    kb_df = load_kb_df()
    st.caption(f"Knowledgebase loaded: {len(kb_df)} lessons.")
except Exception as e:
    st.error(f"Failed to load NTP knowledgebase: {e}")
    st.stop()

st.divider()

# Chat history in session state
if "ntp_chat_history" not in st.session_state:
    st.session_state["ntp_chat_history"] = []

# Render chat history
for msg in st.session_state["ntp_chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask the NTP expert something…")

if user_input:
    # Add user message
    st.session_state["ntp_chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking like a New Type Planning expert..."):
            result = call_ntp_expert(
                question=user_input,
                chat_history=st.session_state["ntp_chat_history"],
            )
            answer = result["answer"]
            retrieved = result["retrieved"]

            st.markdown(answer)

            # Optional: show which lessons were used
            with st.expander("Show lessons used for this answer"):
                if not retrieved:
                    st.write("No lessons found/used.")
                else:
                    dbg_rows = []
                    for r in retrieved:
                        row = r["row"]
                        dbg_rows.append(
                            {
                                "Score": round(r["score"], 3),
                                "ID": row.get("ID", ""),
                                "Shop": row.get("Shop", ""),
                                "Phase": row.get("Project Phase identified", ""),
                                "Subject": row.get("Subject", ""),
                                "Subject Category": row.get("Subject Category", ""),
                                "Carline": row.get("Carline/Engine/Battery", ""),
                            }
                        )
                    st.dataframe(pd.DataFrame(dbg_rows), use_container_width=True)

    # Add assistant message to history
    st.session_state["ntp_chat_history"].append({"role": "assistant", "content": answer})
