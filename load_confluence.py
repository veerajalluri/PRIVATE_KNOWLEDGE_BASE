"""
Fetch pages from Confluence via REST API and load them into ChromaDB.

Usage:
    python load_confluence.py

Configure credentials in .env:
    CONFLUENCE_URL=https://yourcompany.atlassian.net
    CONFLUENCE_EMAIL=you@example.com
    CONFLUENCE_API_TOKEN=your_api_token
    CONFLUENCE_SPACE_KEYS=ENG,DOCS        # comma-separated; omit to fetch all spaces
"""

import os
import uuid
from pathlib import Path

import chromadb
import requests
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PAGE_LIMIT = 50  # pages per API request (max 50)


# ── Confluence API helpers ─────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    email = os.getenv("CONFLUENCE_EMAIL")
    token = os.getenv("CONFLUENCE_API_TOKEN")
    if not email or not token:
        raise ValueError("CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN must be set in .env")
    session = requests.Session()
    session.auth = (email, token)
    session.headers.update({"Accept": "application/json"})
    return session


def _base_url() -> str:
    url = os.getenv("CONFLUENCE_URL", "").rstrip("/")
    if not url:
        raise ValueError("CONFLUENCE_URL must be set in .env")
    return url


def get_space_keys(session: requests.Session) -> list[str]:
    """Return space keys from env, or fetch all spaces from Confluence."""
    env_keys = os.getenv("CONFLUENCE_SPACE_KEYS", "")
    if env_keys:
        return [k.strip() for k in env_keys.split(",") if k.strip()]

    print("No CONFLUENCE_SPACE_KEYS set — fetching all spaces...")
    spaces, start = [], 0
    while True:
        resp = session.get(
            f"{_base_url()}/wiki/rest/api/space",
            params={"limit": 50, "start": start, "type": "global"},
        )
        resp.raise_for_status()
        data = resp.json()
        spaces.extend(r["key"] for r in data["results"])
        if data["results"] and not data.get("_links", {}).get("next"):
            break
        start += 50
    return spaces


def fetch_pages(session: requests.Session, space_key: str) -> list[dict]:
    """Fetch all pages in a space with their body content."""
    pages, start = [], 0
    print(f"  Fetching pages from space: {space_key}")
    while True:
        resp = session.get(
            f"{_base_url()}/wiki/rest/api/content",
            params={
                "spaceKey": space_key,
                "type": "page",
                "status": "current",
                "expand": "body.storage,version",
                "limit": PAGE_LIMIT,
                "start": start,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        pages.extend(results)
        if len(results) < PAGE_LIMIT:
            break
        start += PAGE_LIMIT
    return pages


# ── Text processing ────────────────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def load_confluence() -> None:
    session = _get_session()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )

    # Track already-loaded page IDs to avoid duplicates
    existing = collection.get(include=["metadatas"])
    loaded_ids = {m["confluence_id"] for m in existing["metadatas"] if "confluence_id" in m}

    space_keys = get_space_keys(session)
    print(f"Spaces to sync: {space_keys}\n")

    total_chunks = 0
    for space_key in space_keys:
        pages = fetch_pages(session, space_key)
        print(f"  Found {len(pages)} pages in {space_key}")

        for page in pages:
            page_id = page["id"]
            title = page["title"]
            version = page["version"]["number"]
            source = f"confluence:{space_key}:{page_id}"

            if page_id in loaded_ids:
                print(f"    Skipping (already loaded): {title}")
                continue

            html = page.get("body", {}).get("storage", {}).get("value", "")
            text = html_to_text(html)
            if not text.strip():
                print(f"    Skipping (empty): {title}")
                continue

            chunks = chunk_text(text)
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [
                {
                    "source": title,
                    "confluence_id": page_id,
                    "space_key": space_key,
                    "chunk_index": i,
                    "version": version,
                    "url": f"{_base_url()}/wiki/spaces/{space_key}/pages/{page_id}",
                }
                for i in range(len(chunks))
            ]

            collection.add(ids=ids, documents=chunks, metadatas=metadatas)
            print(f"    Loaded {len(chunks)} chunks from: {title}")
            total_chunks += len(chunks)

    print(f"\nDone. Total chunks in collection: {collection.count()} (+{total_chunks} new)")


if __name__ == "__main__":
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    load_confluence()
