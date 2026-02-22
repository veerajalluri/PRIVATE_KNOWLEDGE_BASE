# conv-bi — Conversational BI on Zylon/PrivateGPT

> A cost-free, fully local conversational Business Intelligence layer built on top of PrivateGPT/Zylon.
> Drop JSON exports from your data warehouse, ask questions in plain English, get answers grounded in your actual numbers.

---

## Problems encountered & how they were solved

### Problem 1 — JSON ingestion produced one blob per file

**What happened:** Zylon's built-in `JSONReader` (from LlamaIndex) reads a JSON file and dumps the entire
contents as a single text document. For a BI export that is an array of records, this means all 165,919
order rows land in one document — too large to embed, too coarse to retrieve usefully.

**Solution:** Replaced `JSONReader` with a custom `BIJsonReader`
([`private_gpt/components/ingest/bi_json_reader.py`](private_gpt/components/ingest/bi_json_reader.py))
that produces **one Document per JSON record**, with nested objects flattened to `key: value` text.
The swap is a single-line change in
[`private_gpt/components/ingest/ingest_helper.py`](private_gpt/components/ingest/ingest_helper.py).

---

### Problem 2 — 678K raw records cannot be embedded on CPU

**What happened:** The two input files (`HeaderResults.json` 138 MB / 165K orders,
`LinesResults.json` 324 MB / 512K line items) produce ~678K documents if embedded record-by-record.
On a CPU-only Ollama setup this would take hours or days, and the resulting vector store would be
~10 GB. More fundamentally, RAG with `top_k=10` over 678K records sees 0.001% of the data —
so any aggregation query ("total revenue", "best margin channel") returns a fabricated answer
based on 10 arbitrary rows.

**Solution:** A pre-aggregation step
([`scripts/prepare_bi_data.py`](scripts/prepare_bi_data.py)) runs **before** ingestion and
uses only the Python standard library to reduce 678K records to **389 summary documents** across 8 files:

| Aggregated file | Records | Answers |
|----------------|---------|---------|
| `header_overall_summary.json` | 1 | Total revenue, orders, date range |
| `header_by_channel.json` | 11 | Revenue / margin by sales channel |
| `header_by_month.json` | 13 | Month-over-month trends |
| `header_top_customers.json` | 100 | Top customers by net sales |
| `header_by_discount.json` | 50 | Discount code usage |
| `lines_overall_summary.json` | 1 | Total cost breakdown |
| `lines_by_sku.json` | 200 | Top 200 products by revenue & margin |
| `lines_by_month.json` | 13 | Monthly line-item trends |

The script runs in ~6 seconds. The 389 summary records embed in minutes on CPU, not days.
`BIJsonReader` also guards against accidental direct ingestion of raw files with a 5,000-record
warning and hard truncation.

---

### Problem 3 — RAG top_k too low for multi-record BI queries

**What happened:** The default `similarity_top_k: 2` means the LLM only sees 2 retrieved
chunks per query. For BI questions that need to reason across multiple channels, months or
products this is insufficient.

**Solution:** `settings-bi.yaml` sets `similarity_top_k: 10` so the LLM sees up to 10
summary records per query. With only 389 total documents, 10 of those cover a meaningful
slice of the data for any well-formed BI question.

---

### Problem 4 — Docker stack was over-engineered for local BI

**What happened:** The upstream `docker-compose.yaml` runs three services for the Ollama mode:
a Traefik reverse proxy, an Ollama container, and the PrivateGPT container. Traefik adds
complexity and memory overhead that is unnecessary for a single-host local deployment.

**Solution:** [`docker-compose.bi.yaml`](docker-compose.bi.yaml) is a lean 2-service file
(Ollama + conv-bi app) with no proxy layer and a three-stage entrypoint:

```
prepare_bi_data.py  →  ingest_bi_json.py  →  python -m private_gpt
```

Qdrant runs embedded inside the app container (no third container needed).

---

## Quick start

### Docker (recommended)

```bash
# 1. Copy your JSON exports
cp HeaderResults.json LinesResults.json data/uploads/

# 2. Start the stack — aggregates data, embeds summaries, then serves the UI
#    (downloads Mistral 7B + nomic-embed-text on first run, ~4 GB)
make docker-up

# 3. Open http://localhost:8001 and ask questions
```

### Local (no Docker)

Requires [Ollama](https://ollama.ai) installed and running.

```bash
ollama pull mistral && ollama pull nomic-embed-text

cp HeaderResults.json LinesResults.json data/uploads/

make bi    # aggregate raw JSON → embed summaries into Qdrant
make run   # start the app at http://localhost:8001
```

On subsequent data refreshes, just re-run `make bi && make run`.

### Example questions

- *"What is the total net sales across all channels?"*
- *"Which sales channel has the highest gross margin percentage?"*
- *"Show me the top 5 products by revenue."*
- *"How did monthly revenue trend month over month?"*
- *"Which discount codes drive the most orders?"*

## Makefile reference

```
make bi            Aggregate raw JSON + embed summaries (full data pipeline)
make prepare       Aggregate raw JSON in data/uploads/ → data/aggregated/
make ingest        Embed aggregated summaries into Qdrant
make run           Start the app (local profile)
make dev           Start with auto-reload for development

make docker-build  Rebuild docker image
make docker-up     Start the full Docker stack (Ollama + conv-bi)
make docker-down   Stop the stack
make docker-logs   Tail container logs

make test          Run the test suite
make format        Format code with black and ruff
make wipe          Clear the vector store and docstore
make list          Print all available targets
```

## Cost

| Component | Cost |
|-----------|------|
| Ollama — Mistral 7B Q4 (LLM) | $0 — CPU inference |
| Ollama — nomic-embed-text (embeddings) | $0 — CPU inference |
| Qdrant (vector store) | $0 — embedded, file-based |
| Cloud API calls | $0 — none |

## Files added / changed

| File | Purpose |
|------|---------|
| [`private_gpt/components/ingest/bi_json_reader.py`](private_gpt/components/ingest/bi_json_reader.py) | One Document per JSON record |
| [`private_gpt/components/ingest/ingest_helper.py`](private_gpt/components/ingest/ingest_helper.py) | Routes `.json` to BIJsonReader |
| [`scripts/prepare_bi_data.py`](scripts/prepare_bi_data.py) | Aggregates raw JSON into summary docs |
| [`scripts/ingest_bi_json.py`](scripts/ingest_bi_json.py) | Ingests aggregated summaries into Qdrant |
| [`settings-bi.yaml`](settings-bi.yaml) | BI-tuned config (top_k, system prompt, Mistral) |
| [`settings-local.yaml`](settings-local.yaml) | Local dev config mirroring settings-bi.yaml |
| [`docker-compose.bi.yaml`](docker-compose.bi.yaml) | Lean 2-service Docker stack |
| [`Makefile`](Makefile) | Developer workflow targets |

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


<!-- to run  -->

docker compose -f docker-compose.bi.yaml up

Then open http://localhost:8001 and ask: "What is the total net sales by channel?" or "Which product has the highest margin?"


scripts/prepare_bi_data.py — ran in 6 seconds, produced 8 files / 389 records:

Aggregated file	Records	Answers questions like
header_overall_summary.json	1	Total revenue, total orders, date range
header_by_channel.json	11	Which channel has best margin?
header_by_month.json	13	Month-over-month trend?
header_top_customers.json	100	Top customers by sales?
header_by_discount.json	50	Which discount codes are used most?
lines_overall_summary.json	1	Total cost breakdown
lines_by_sku.json	200	Top products by revenue/margin?
lines_by_month.json	13	Monthly line-item trends
