# Conversational BI Implementation Plan

## Current State

| Component | What exists |
|-----------|-------------|
| Base app  | PrivateGPT/Zylon 0.6.2 — RAG over documents |
| JSON support | `JSONReader` registered but creates **one blob per file** (bad for BI records) |
| LLM backend | Ollama with Mistral (already in settings-local.yaml) |
| Vector store | Qdrant embedded (local, no extra service) |
| Docker | Two separate containers: ollama-cpu + private-gpt-ollama + traefik proxy |
| System prompt | Generic assistant — not BI-aware |
| RAG retrieval | `similarity_top_k: 2` — too low for multi-record BI queries |

## Gaps to Fix

1. **JSON ingestion is wrong for BI data** — current `JSONReader` dumps entire file as one text blob. BI data is an array of records; each record should be its own document so the retriever can fetch relevant rows independently.
2. **No BI-tuned system prompt** — LLM needs to know it is answering analytical questions about business data.
3. **Docker is over-engineered** — Traefik proxy + separate services unnecessary for local BI. We need a lean 2-container setup.
4. **RAG top_k too low** — aggregation queries (e.g. "what is total revenue?") need more chunks retrieved.
5. **No startup data loading** — JSON files in a folder need to be ingested on container start.

---

## What We Will Build

### 1. `private_gpt/components/ingest/bi_json_reader.py` (NEW)

A custom `BaseReader` that:
- Reads a JSON file that is either an **array of objects** or a single object
- Produces **one `Document` per JSON record** (each row = one doc)
- Flattens nested objects into `"key: value\nnested.key: value"` plain text so the LLM can read it naturally
- Stores `record_index` and `source_file` as metadata for traceability

This replaces `llama_index.core.readers.json.JSONReader` for `.json` files.

### 2. `private_gpt/components/ingest/ingest_helper.py` (MODIFY — 1 line)

Swap the `.json` reader entry from `JSONReader` → `BIJsonReader`.

### 3. `settings-bi.yaml` (NEW)

Layered on top of `settings.yaml`. Configures:

```yaml
server:
  env_name: bi-local

llm:
  mode: ollama
  model: mistral          # q4-quantised, runs on CPU ~4 GB RAM
  max_new_tokens: 1024    # BI answers can be long
  context_window: 8192    # Mistral supports 8k

embedding:
  mode: ollama
  ingest_mode: simple     # simple is fine; JSON records are small

vectorstore:
  database: qdrant        # embedded, no extra service

rag:
  similarity_top_k: 10    # retrieve more records for aggregation queries

ui:
  default_mode: "RAG"
  default_query_system_prompt: |   # BI-aware prompt
    You are a business intelligence analyst. The context contains structured
    business records. Answer the user's question precisely using the data
    provided. For numeric questions compute totals/averages/counts from the
    records. Always cite which records support your answer.
    If the answer cannot be derived from the provided context, say so clearly.

ollama:
  llm_model: mistral
  embedding_model: nomic-embed-text
  api_base: http://ollama:11434        # docker service name
  autopull_models: true
```

### 4. `docker-compose.bi.yaml` (NEW)

A lean, self-contained compose file with **just 2 services**:

```
┌────────────────────┐        ┌─────────────────────────────┐
│  ollama            │◄──────│  conv-bi (private-gpt)      │
│  ollama/ollama:    │ :11434 │  zylonai/private-gpt:ollama │
│  latest (CPU)      │        │  PGPT_PROFILES=bi-local     │
│  volume: models/   │        │  volume: local_data/        │
│                    │        │  volume: data/uploads/      │
└────────────────────┘        └─────────────────────────────┘
```

- No Traefik proxy (direct connection)
- No GPU service (CPU-only, zero cloud cost)
- `data/uploads/` volume — drop JSON files here to ingest
- Qdrant runs **inside** the conv-bi container (embedded, no 3rd container)
- `depends_on` with health-check so conv-bi waits for Ollama

### 5. `scripts/ingest_bi_json.py` (NEW)

A script that:
1. Scans a folder (default: `local_data/uploads/`) for `.json` files
2. POSTs each file to `/v1/ingest/file`
3. Prints progress and any errors
4. Can be run ad-hoc: `python scripts/ingest_bi_json.py path/to/data/`

Also added as an `entrypoint` step in `docker-compose.bi.yaml`:
```bash
python scripts/ingest_bi_json.py && python -m private_gpt
```

### 6. `settings-local.yaml` (MODIFY — minor)

Add the BI system prompt and raise `similarity_top_k` so it matches `settings-bi.yaml` behaviour when running locally without Docker.

---

## File Change Summary

| File | Action | Why |
|------|---------|-----|
| `private_gpt/components/ingest/bi_json_reader.py` | **Create** | Per-record JSON documents |
| `private_gpt/components/ingest/ingest_helper.py` | **Modify** (1 line) | Use BIJsonReader for .json |
| `settings-bi.yaml` | **Create** | BI-tuned config layer |
| `docker-compose.bi.yaml` | **Create** | Lean 2-service setup |
| `scripts/ingest_bi_json.py` | **Create** | Data loading helper |
| `settings-local.yaml` | **Modify** (add prompt + top_k) | Local dev parity |

---

## Cost Analysis

| Resource | Cost |
|----------|------|
| Ollama (Mistral 7B Q4) | $0 — runs on CPU, ~4 GB RAM |
| nomic-embed-text | $0 — runs inside Ollama |
| Qdrant embedded | $0 — file-based, no separate container |
| Compute (local Mac/Linux) | $0 |
| Cloud API calls | $0 |

Total: **$0/month** for unlimited queries against your JSON data.

---

## Example Usage After Implementation

```bash
# 1. Drop your BI data
cp sales_q4_2025.json data/uploads/

# 2. Start the stack
docker compose -f docker-compose.bi.yaml up

# 3. Open browser → http://localhost:8001
# Ask: "What was total revenue in Q4?"
# Ask: "Which product had the highest margin?"
# Ask: "Show me top 5 customers by order value"
```
