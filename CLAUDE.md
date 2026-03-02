# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RECAP Bankruptcy Case Intelligence — an open-source RAG tool for querying bankruptcy case documents from Free Law Project's RECAP Archive and CourtListener API. Users ask natural language questions about federal bankruptcy cases and get AI-generated answers with ECF citations.

## Running the App

```bash
streamlit run app.py
```

Requires a `.env` file with at minimum `COURTLISTENER_API_TOKEN` and an LLM key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). Run `config.validate_config()` to check required keys.

## Architecture

RAG pipeline: **Fetch → Classify → Index → Query**

- **courtlistener.py** — API client for CourtListener REST API (v3/v4). Dataclasses: `RecapDocument`, `DocketEntry`, `BankruptcyCase`. Handles pagination, rate limiting (5000 req/hr), and optional PACER purchases. `get_purchasable_for_entries()` looks up recap document IDs for entries (queries API when no local records exist). PACER docket date params use `de_date_start`/`de_date_end` (not `date_start`/`date_end`).
- **classifier.py** — Regex-based document type classification into `DocType` enum (11 types: Motion, Objection, Order, Plan, Claim, etc.). Pattern matching is ordered most-specific-first.
- **indexer.py** — Chunks documents (512-token target, 50-token overlap), embeds via FLP ModernBERT (default, free, local) or OpenAI, stores in ChromaDB. `CaseIndex` class manages the vector store lifecycle. Two content pools: document chunks (`source: "document"`) and docket entry descriptions (`source: "docket_entry"`). Methods: `query()`, `query_descriptions()`, `query_documents()`, `index_single_document()` (incremental). FLP model uses `search_document:` / `search_query:` prefixes.
- **query.py** — Question intent classification (`classify_question()` → `QuestionIntent`) and smart retrieval routing. Structured listings for date/type queries; two-stage retrieval (descriptions → document chunks) for keyword and analytical queries. Calls LLM (Anthropic Claude or OpenAI GPT), returns answer with source tracking. When PACER creds are configured, requests structured JSON output from LLM with `suggested_purchases` for description-only sources. `_parse_llm_response()` handles JSON extraction with fallback (pure JSON, fenced, prose+JSON, plain text). `query_case()` accepts optional `progress` callback for UI status updates.
- **config.py** — All configuration via environment variables with `python-dotenv`. Provider selection for embeddings and LLM.
- **app.py** — Streamlit UI with session state for case data, index, and chat history. Case caching to `data/cases/`. Layout: compact case header with collapsible "Case Details & Controls" expander, chat area below. Doc type filter and case navigation in sidebar. Purchase suggestion UI with per-document checkboxes. `_escape_dollars()` prevents Streamlit LaTeX rendering of `$` signs. `st.status` shows multi-step RAG pipeline progress.
- **prompts/system_prompt.txt** — LLM system prompt enforcing citation of ECF numbers, bankruptcy terminology, and grounded-only answers.

## Key Patterns

- Domain models are Python `dataclass`es (not Pydantic)
- FLP embedding model is lazily loaded as a module-level global (`_flp_model`)
- ChromaDB is file-based (stored in `data/chroma/`, gitignored)
- Chunk metadata includes `docket_entry_id`, `ecf_number`, `description`, `doc_type`, `date_filed`, `chunk_index`, `source`
- All answers must cite ECF docket entry numbers per the system prompt
- Embedding batching for both ChromaDB inserts and API calls
- Streamlit `$` signs must be escaped (`\$`) to prevent LaTeX rendering — use `_escape_dollars()` for any user-facing text that may contain dollar amounts
- `st.chat_input` must be in the main body (not inside `st.tabs` or containers) to pin to the bottom of the page
- PACER docket purchase date params are `de_date_start`/`de_date_end` (not `date_start`/`date_end`)

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `flp` | `flp` (free, local) or `openai` |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Any supported model ID |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |
| `RETRIEVAL_TOP_K` | `12` | Number of chunks retrieved per query |

## Dependencies

Python 3.10+. Key packages: `streamlit`, `chromadb`, `sentence-transformers`, `torch`, `anthropic`, `openai`, `requests`, `python-dotenv`. Install with `pip install -r requirements.txt`.
