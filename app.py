"""
RAG Chatbot — Streamlit UI backed by ChromaDB + Claude.

Run:
    streamlit run app.py

Make sure to:
1. Copy .env.example to .env and set ANTHROPIC_API_KEY
2. Load documents via load_documents.py or load_confluence.py
"""

import os
from pathlib import Path

import anthropic
import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "knowledge_base"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 5
MAX_HISTORY_TURNS = 6  # last N user+assistant pairs sent to Claude

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.
Use the context below to answer the user's question accurately and concisely.
If the answer is not found in the context, say so clearly — do not make up information."""


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=DefaultEmbeddingFunction())


def retrieve_context(question: str, collection) -> tuple[str, list[str]]:
    results = collection.query(query_texts=[question], n_results=TOP_K, include=["documents", "metadatas"])
    docs = results["documents"][0]
    sources = list({m["source"] for m in results["metadatas"][0]})
    context = "\n\n---\n\n".join(docs)
    return context, sources


def build_messages(context: str, history: list[dict], question: str) -> list[dict]:
    messages = []
    for turn in history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append(turn)
    user_content = f"Context from documents:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})
    return messages


def ask_claude(messages: list[dict]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Knowledge Base Chatbot", page_icon="📚", layout="centered")
st.title("📚 Knowledge Base Chatbot")
st.caption("Ask questions about your documents. Powered by Claude + ChromaDB.")

collection = get_chroma_collection()

if collection.count() == 0:
    st.warning(
        "No documents loaded yet. "
        "Run `python load_documents.py` or `python load_confluence.py` to ingest content."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if collection.count() == 0:
        st.error("Please load documents first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)

    context, sources = retrieve_context(prompt, collection)
    messages_for_claude = build_messages(context, st.session_state.messages, prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_claude(messages_for_claude)
        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
