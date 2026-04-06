"""
Load Word (.docx) and PDF documents from data/documents/ into ChromaDB vector store.

Usage:
    python load_documents.py

Place your .docx or .pdf files in data/documents/ before running.
"""

import uuid
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from docx import Document
from pypdf import PdfReader

DOCS_DIR = Path("data/documents")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_docx(path: Path) -> str:
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR) -> None:
    docx_files = list(docs_dir.glob("*.docx"))
    pdf_files = list(docs_dir.glob("*.pdf"))
    all_files = docx_files + pdf_files
    if not all_files:
        print(f"No .docx or .pdf files found in {docs_dir}. Add documents and re-run.")
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )

    existing = collection.get(include=["metadatas"])
    loaded_sources = {m["source"] for m in existing["metadatas"]} if existing["metadatas"] else set()

    total_chunks = 0
    for doc_path in all_files:
        filename = doc_path.name
        if filename in loaded_sources:
            print(f"  Skipping (already loaded): {filename}")
            continue

        print(f"  Processing: {filename}")
        if doc_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(doc_path)
        else:
            text = extract_text_from_docx(doc_path)
        if not text.strip():
            print(f"    Warning: {filename} appears to be empty, skipping.")
            continue

        chunks = chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        print(f"    Loaded {len(chunks)} chunks from {filename}")
        total_chunks += len(chunks)

    print(f"\nDone. Total chunks in collection: {collection.count()}")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    load_documents()
