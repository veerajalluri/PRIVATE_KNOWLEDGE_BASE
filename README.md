# conv-bi — Conversational BI on Zylon/PrivateGPT

> A cost-free, fully local conversational Business Intelligence layer built on top of PrivateGPT/Zylon.
> Drop JSON exports from your data warehouse, ask questions in plain English, get answers grounded in your actual numbers.

---

## Architecture

The pipeline has two phases:

```
── once (or on data refresh) ──────────────────────────────────────────
HeaderResults.json  ─┐
                     ├─ prepare_bi_data.py ──▶ local_data/bi.db
LinesResults.json   ─┘   (read_json_auto)       orders table  165K rows
                                               order_lines table  512K rows

── every query ────────────────────────────────────────────────────────
User question
    │
    ▼
BISqlService.query()
    ├─ 1. Build NSQL prompt  (question + full schema DDL)
    ├─ 2. duckdb-nsql via Ollama  →  SQL
    ├─ 3. Strip markdown fences, extract first statement
    ├─ 4. Execute against bi.db
    │       ✓  format as markdown table  →  return to user
    │       ✗  append error to prompt, retry once (self-correct)
    └─ 5. After 2 failed attempts  →  return error + last SQL
```

No vector embeddings, no Qdrant, no RAG retrieval. The LLM writes SQL; DuckDB runs it against the full raw dataset.

---

## Problems encountered & how they were solved

### Problem 1 — RAG hallucination on aggregation queries

**What happened:** The original approach embedded pre-aggregated JSON summary documents into a
Qdrant vector store and used RAG (`top_k=10`) to answer questions. For aggregation queries
("total revenue", "best margin channel"), the LLM saw only 10 retrieved chunks and fabricated
totals that bore no relation to the actual data.

**Solution:** Replaced the entire RAG path with **text-to-SQL**. `BISqlService`
([`private_gpt/components/sql/bi_sql_service.py`](private_gpt/components/sql/bi_sql_service.py))
translates questions to DuckDB SQL using the `duckdb-nsql` model (fine-tuned for SQL generation),
executes the SQL against the real data, and returns exact results. Aggregations are computed by the
database engine, not estimated by the LLM.

---

### Problem 2 — 678K raw records, days to embed on CPU

**What happened:** The two input files (`HeaderResults.json` 138 MB / 165K orders,
`LinesResults.json` 324 MB / 512K line items) would take hours or days to embed on CPU, and the
resulting vector store would be ~10 GB.

**Solution:** No embedding at all. [`scripts/prepare_bi_data.py`](scripts/prepare_bi_data.py)
uses DuckDB's native `read_json_auto()` to load both files directly into a local `bi.db` file:

```
HeaderResults.json  →  table: orders       (all 165K rows)
LinesResults.json   →  table: order_lines  (all 512K rows)
```

DuckDB streams the JSON files without loading them fully into Python memory. The whole load
completes in seconds. At query time, DuckDB executes SQL against the full dataset in milliseconds —
no sampling, no pre-aggregation, no information loss.

---

### Problem 3 — LLM wraps SQL in markdown fences

**What happened:** The `duckdb-nsql` model frequently returns SQL wrapped in ` ```sql ``` ` code
fences. Passing that raw output to DuckDB raises a parse error.

**Solution:** `_extract_sql()` in `bi_sql_service.py` strips code fences before execution,
handling both ` ```sql ``` ` and ` ``` ``` ` variants.

---

### Problem 4 — No recovery when generated SQL fails

**What happened:** If the model produced invalid SQL (wrong column name, bad syntax), the user
received a dead-end error with no way to recover.

**Solution:** `BISqlService.query()` retries once on failure. The DuckDB error message is appended
to the prompt so the model can self-correct:

```
-- The previous SQL attempt failed with: <error>
-- Corrected SQL:
```

---

### Problem 5 — Docker stack was over-engineered for local BI

**What happened:** The upstream `docker-compose.yaml` runs three services: Traefik proxy, Ollama,
and PrivateGPT. Traefik adds complexity and memory overhead unnecessary for a single-host
local deployment.

**Solution:** [`docker-compose.bi.yaml`](docker-compose.bi.yaml) is a lean 2-service file
(Ollama + conv-bi app) with a two-stage entrypoint:

```
prepare_bi_data.py  →  python -m private_gpt
```

---

## Quick start

### Docker (recommended)

```bash
# 1. Copy your JSON exports
cp HeaderResults.json LinesResults.json local_data/uploads/

# 2. Start the stack — loads data into DuckDB, then serves the UI
#    (downloads duckdb-nsql 7B on first run, ~4 GB)
make docker-up

# 3. Open http://localhost:8001, switch to RAG mode, and ask questions
```

### Local (no Docker)

Requires [Ollama](https://ollama.ai) installed and running.

```bash
ollama pull duckdb-nsql && ollama pull nomic-embed-text

cp HeaderResults.json LinesResults.json local_data/uploads/

make bi    # load raw JSON into local_data/bi.db
make run   # start the app at http://localhost:8001
```

On subsequent data refreshes, re-run `make bi && make run`.

### Example questions

- *"What is the total net sales across all channels?"*
- *"Which sales channel has the highest gross margin percentage?"*
- *"Show me the top 5 products by revenue."*
- *"How did monthly revenue trend over time?"*
- *"What is the average order value for the retail channel?"*
- *"Which customer placed the highest value order?"*

---

## Makefile reference

```
make bi            Load raw JSON into local_data/bi.db (full data pipeline)
make prepare       Load raw JSON in local_data/uploads/ → local_data/bi.db
make run           Start the app (local profile)
make dev           Start with auto-reload for development

make docker-build  Rebuild the conv-bi Docker image
make docker-up     Start the full stack (Ollama + conv-bi)
make docker-down   Stop the stack
make docker-logs   Tail container logs

make test          Run the test suite
make format        Format code with black and ruff
make wipe          Clear the vector store and docstore
make list          Print all available targets
```

---

## Tests

```
make test
```

48 tests, no external services required (all DuckDB in-memory or tmp files, LLM mocked).

### `tests/components/sql/test_bi_sql_service.py` — 23 tests

| Class | Coverage |
|-------|----------|
| `TestExtractSql` | Plain SQL, ` ```sql ``` ` fences, ` ``` ``` ` fences, semicolon truncation, whitespace stripping |
| `TestFormatResult` | Markdown table output, empty result, raises on bad SQL, aggregation correctness |
| `TestBISqlServiceQuery` | Successful query, fenced SQL from LLM, retry on SQL error, error appended to retry prompt, exhausted retries, LLM exception, empty question |

### `tests/scripts/test_prepare_bi_data.py` — 25 tests

| Class | Coverage |
|-------|----------|
| `TestPrepareBiData` | Creates `orders` table, correct columns, creates `order_lines` when present, skips gracefully on missing lines file, exits with code 1 on missing header, creates parent directories, `CREATE OR REPLACE` does not double-count on re-run |
| `TestAggregationsOnOrders` | Total net sales, order count, net sales by channel, order count by channel, distinct channels, average order value, max/min, date-range filter (Jan/Feb), top channel, gross margin by channel, customer lookup |
| `TestAggregationsOnOrderLines` | Total net sales, top SKU, net sales by SKU, distinct SKU count, gross margin per SKU, JOIN to orders by `OriginalReference` |

> **Note:** `CreatedDate` is inferred as `DATE` (not `VARCHAR`) by `read_json_auto()`.
> Date filters must use range comparisons (`>= '2024-01-01'`), not `LIKE '2024-01%'`.

---

## Cost

| Component | Cost |
|-----------|------|
| Ollama — duckdb-nsql 7B (SQL generation) | $0 — CPU inference |
| DuckDB embedded (query engine) | $0 — in-process, file-based |
| Cloud API calls | $0 — none |

---

## Key files

| File | Purpose |
|------|---------|
| [`scripts/prepare_bi_data.py`](scripts/prepare_bi_data.py) | Loads raw JSON into DuckDB (`orders`, `order_lines`) |
| [`private_gpt/components/sql/bi_sql_service.py`](private_gpt/components/sql/bi_sql_service.py) | Text-to-SQL: question → SQL → DuckDB result, with retry |
| [`private_gpt/server/chat/chat_service.py`](private_gpt/server/chat/chat_service.py) | Routes RAG-mode requests to `BISqlService` |
| [`private_gpt/components/ingest/bi_json_reader.py`](private_gpt/components/ingest/bi_json_reader.py) | LlamaIndex reader for `.json` files (non-BI ingest path) |
| [`settings-bi.yaml`](settings-bi.yaml) | BI config: duckdb-nsql model, 5 min timeout, deterministic temp |
| [`settings-local.yaml`](settings-local.yaml) | Local dev config (same as bi, localhost Ollama URL) |
| [`docker-compose.bi.yaml`](docker-compose.bi.yaml) | Lean 2-service Docker stack (Ollama + app) |
| [`Makefile`](Makefile) | Developer workflow targets |
| [`tests/components/sql/test_bi_sql_service.py`](tests/components/sql/test_bi_sql_service.py) | Unit tests for SQL extraction, formatting, and query retry |
| [`tests/scripts/test_prepare_bi_data.py`](tests/scripts/test_prepare_bi_data.py) | Integration tests for data loading and aggregation correctness |

---

# PrivateGPT (upstream)

<a href="https://trendshift.io/repositories/2601" target="_blank"><img src="https://trendshift.io/api/badge/repositories/2601" alt="imartinez%2FprivateGPT | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Tests](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml/badge.svg)](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml?query=branch%3Amain)
[![Website](https://img.shields.io/website?up_message=check%20it&down_message=down&url=https%3A%2F%2Fdocs.privategpt.dev%2F&label=Documentation)](https://docs.privategpt.dev/)
[![Discord](https://img.shields.io/discord/1164200432894234644?logo=discord&label=PrivateGPT)](https://discord.gg/bK6mRVpErU)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZylonPrivateGPT)](https://twitter.com/ZylonPrivateGPT)

![Gradio UI](/fern/docs/assets/ui.png?raw=true)

PrivateGPT is a production-ready AI project that allows you to ask questions about your documents using the power
of Large Language Models (LLMs), even in scenarios without an Internet connection. 100% private, no data leaves your
execution environment at any point.

>[!TIP]
> If you are looking for an **enterprise-ready, fully private AI workspace**
> check out [Zylon's website](https://zylon.ai)  or [request a demo](https://cal.com/zylon/demo?source=pgpt-readme).
> Crafted by the team behind PrivateGPT, Zylon is a best-in-class AI collaborative
> workspace that can be easily deployed on-premise (data center, bare metal...) or in your private cloud (AWS, GCP, Azure...).

The project provides an API offering all the primitives required to build private, context-aware AI applications.
It follows and extends the [OpenAI API standard](https://openai.com/blog/openai-api),
and supports both normal and streaming responses.

The API is divided into two logical blocks:

**High-level API**, which abstracts all the complexity of a RAG (Retrieval Augmented Generation)
pipeline implementation:
- Ingestion of documents: internally managing document parsing,
splitting, metadata extraction, embedding generation and storage.
- Chat & Completions using context from ingested documents:
abstracting the retrieval of context, the prompt engineering and the response generation.

**Low-level API**, which allows advanced users to implement their own complex pipelines:
- Embeddings generation: based on a piece of text.
- Contextual chunks retrieval: given a query, returns the most relevant chunks of text from the ingested documents.

In addition to this, a working [Gradio UI](https://www.gradio.app/)
client is provided to test the API, together with a set of useful tools such as bulk model
download script, ingestion script, documents folder watch, etc.

## 🎞️ Overview
>[!WARNING]
>  This README is not updated as frequently as the [documentation](https://docs.privategpt.dev/).
>  Please check it out for the latest updates!

### Motivation behind PrivateGPT
Generative AI is a game changer for our society, but adoption in companies of all sizes and data-sensitive
domains like healthcare or legal is limited by a clear concern: **privacy**.
Not being able to ensure that your data is fully under your control when using third-party AI tools
is a risk those industries cannot take.

### Primordial version
The first version of PrivateGPT was launched in May 2023 as a novel approach to address the privacy
concerns by using LLMs in a complete offline way.

That version, which rapidly became a go-to project for privacy-sensitive setups and served as the seed
for thousands of local-focused generative AI projects, was the foundation of what PrivateGPT is becoming nowadays;
thus a simpler and more educational implementation to understand the basic concepts required
to build a fully local -and therefore, private- chatGPT-like tool.

If you want to keep experimenting with it, we have saved it in the
[primordial branch](https://github.com/zylon-ai/private-gpt/tree/primordial) of the project.

> It is strongly recommended to do a clean clone and install of this new version of
PrivateGPT if you come from the previous, primordial version.

### Present and Future of PrivateGPT
PrivateGPT is now evolving towards becoming a gateway to generative AI models and primitives, including
completions, document ingestion, RAG pipelines and other low-level building blocks.
We want to make it easier for any developer to build AI applications and experiences, as well as provide
a suitable extensive architecture for the community to keep contributing.

Stay tuned to our [releases](https://github.com/zylon-ai/private-gpt/releases) to check out all the new features and changes included.

## 📄 Documentation
Full documentation on installation, dependencies, configuration, running the server, deployment options,
ingesting local documents, API details and UI features can be found here: https://docs.privategpt.dev/

## 🧩 Architecture
Conceptually, PrivateGPT is an API that wraps a RAG pipeline and exposes its
primitives.
* The API is built using [FastAPI](https://fastapi.tiangolo.com/) and follows
  [OpenAI's API scheme](https://platform.openai.com/docs/api-reference).
* The RAG pipeline is based on [LlamaIndex](https://www.llamaindex.ai/).

The design of PrivateGPT allows to easily extend and adapt both the API and the
RAG implementation. Some key architectural decisions are:
* Dependency Injection, decoupling the different components and layers.
* Usage of LlamaIndex abstractions such as `LLM`, `BaseEmbedding` or `VectorStore`,
  making it immediate to change the actual implementations of those abstractions.
* Simplicity, adding as few layers and new abstractions as possible.
* Ready to use, providing a full implementation of the API and RAG
  pipeline.

Main building blocks:
* APIs are defined in `private_gpt:server:<api>`. Each package contains an
  `<api>_router.py` (FastAPI layer) and an `<api>_service.py` (the
  service implementation). Each *Service* uses LlamaIndex base abstractions instead
  of specific implementations,
  decoupling the actual implementation from its usage.
* Components are placed in
  `private_gpt:components:<component>`. Each *Component* is in charge of providing
  actual implementations to the base abstractions used in the Services - for example
  `LLMComponent` is in charge of providing an actual implementation of an `LLM`
  (for example `LlamaCPP` or `OpenAI`).

## 💡 Contributing
Contributions are welcomed! To ensure code quality we have enabled several format and
typing checks, just run `make check` before committing to make sure your code is ok.
Remember to test your code! You'll find a tests folder with helpers, and you can run
tests using `make test` command.

Don't know what to contribute? Here is the public 
[Project Board](https://github.com/users/imartinez/projects/3) with several ideas. 

Head over to Discord 
#contributors channel and ask for write permissions on that GitHub project.

## 💬 Community
Join the conversation around PrivateGPT on our:
- [Twitter (aka X)](https://twitter.com/PrivateGPT_AI)
- [Discord](https://discord.gg/bK6mRVpErU)

## 📖 Citation
If you use PrivateGPT in a paper, check out the [Citation file](CITATION.cff) for the correct citation.  
You can also use the "Cite this repository" button in this repo to get the citation in different formats.

Here are a couple of examples:

#### BibTeX
```bibtex
@software{Zylon_PrivateGPT_2023,
author = {Zylon by PrivateGPT},
license = {Apache-2.0},
month = may,
title = {{PrivateGPT}},
url = {https://github.com/zylon-ai/private-gpt},
year = {2023}
}
```

#### APA
```
Zylon by PrivateGPT (2023). PrivateGPT [Computer software]. https://github.com/zylon-ai/private-gpt
```

## 🤗 Partners & Supporters
PrivateGPT is actively supported by the teams behind:
* [Qdrant](https://qdrant.tech/), providing the default vector database
* [Fern](https://buildwithfern.com/), providing Documentation and SDKs
* [LlamaIndex](https://www.llamaindex.ai/), providing the base RAG framework and abstractions

This project has been strongly influenced and supported by other amazing projects like 
[LangChain](https://github.com/hwchase17/langchain),
[GPT4All](https://github.com/nomic-ai/gpt4all),
[LlamaCpp](https://github.com/ggerganov/llama.cpp),
[Chroma](https://www.trychroma.com/)
and [SentenceTransformers](https://www.sbert.net/).


