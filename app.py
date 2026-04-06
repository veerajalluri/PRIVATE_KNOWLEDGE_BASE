"""
RAG Chatbot — Streamlit UI backed by ChromaDB + configurable LLM.

Supported LLM providers (select in sidebar):
  - Claude        (Anthropic)   — requires ANTHROPIC_API_KEY
  - OpenAI                      — requires OPENAI_API_KEY
  - Gemini        (Vertex AI)   — requires GOOGLE_API_KEY
  - Ollama        (local)       — requires Ollama running at OLLAMA_BASE_URL

Run:
    streamlit run app.py
"""

import os
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "knowledge_base"
TOP_K = 5
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.
Use the context below to answer the user's question accurately and concisely.
If the answer is not found in the context, say so clearly — do not make up information."""

# ── LLM defaults ──────────────────────────────────────────────────────────────

LLM_DEFAULTS = {
    "Claude (Anthropic)": {
        "models": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        "env_key": "ANTHROPIC_API_KEY",
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "env_key": "OPENAI_API_KEY",
    },
    "Gemini (Vertex AI)": {
        "models": ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
        "env_key": "GOOGLE_API_KEY",
    },
    "Ollama (local)": {
        "models": ["llama3", "mistral", "gemma3", "phi4", "qwen2.5"],
        "env_key": None,
    },
}


# ── LLM callers ───────────────────────────────────────────────────────────────

def ask_claude(messages: list[dict], model: str) -> str:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set in .env"
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def ask_openai(messages: list[dict], model: str) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set in .env"
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )
    return response.choices[0].message.content


def ask_vertex(messages: list[dict], model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "Error: google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform"

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set in .env"

    try:
        client = genai.Client(api_key=api_key)

        # Build conversation history
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            history=history,
        )
        response = chat.send_message(messages[-1]["content"])
        return response.text
    except Exception as e:
        return f"Error: {e}"


def ask_ollama(messages: list[dict], model: str) -> str:
    import requests
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": False,
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Ollama at {base_url}. Is Ollama running?"
    except Exception as e:
        return f"Error: {e}"


def ask_llm(messages: list[dict], provider: str, model: str) -> str:
    if provider == "Claude (Anthropic)":
        return ask_claude(messages, model)
    elif provider == "OpenAI":
        return ask_openai(messages, model)
    elif provider == "Gemini (Vertex AI)":
        return ask_vertex(messages, model)
    elif provider == "Ollama (local)":
        return ask_ollama(messages, model)
    return "Error: Unknown provider"


# ── ChromaDB ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )


def retrieve_context(question: str, collection) -> tuple[str, list[str]]:
    results = collection.query(query_texts=[question], n_results=TOP_K, include=["documents", "metadatas"])
    docs = results["documents"][0]
    sources = list({m["source"] for m in results["metadatas"][0]})
    context = "\n\n---\n\n".join(docs)
    return context, sources


def build_messages(context: str, history: list[dict], question: str) -> list[dict]:
    messages = list(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": f"Context from documents:\n{context}\n\nQuestion: {question}"})
    return messages


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Knowledge Base Chatbot", page_icon="📚", layout="centered")
st.title("📚 Knowledge Base Chatbot")
st.caption("Ask questions about your documents.")

# Sidebar — LLM config
with st.sidebar:
    st.header("LLM Settings")
    provider = st.selectbox("Provider", list(LLM_DEFAULTS.keys()))
    model = st.selectbox("Model", LLM_DEFAULTS[provider]["models"])

    env_key = LLM_DEFAULTS[provider]["env_key"]
    if env_key and not os.getenv(env_key):
        st.warning(f"{env_key} not set in .env")

    if provider == "Ollama (local)":
        ollama_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        os.environ["OLLAMA_BASE_URL"] = ollama_url

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

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
    messages_for_llm = build_messages(context, st.session_state.messages, prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({provider} / {model})..."):
            answer = ask_llm(messages_for_llm, provider, model)
        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
