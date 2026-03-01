# RECAP Bankruptcy Case Intelligence

**An open-source AI-powered tool for querying bankruptcy case documents using Free Law Project's RECAP Archive.**

Ask natural language questions about any federal bankruptcy case — "When was the disclosure statement filed?", "Summarize all objections to the plan", "Make me a table of fee applications with amounts requested" — powered by documents freely available through the RECAP Archive and CourtListener API.

## Why This Exists

Bankruptcy cases generate massive dockets. A large Chapter 11 can have thousands of entries, and practitioners, creditors, journalists, and researchers routinely waste hours navigating them. Existing tools either:

- **Provide access but not intelligence** (PACER, CourtListener, PacerPro) — you can find and download documents, but you still have to read everything yourself
- **Provide AI but not for per-case docket Q&A** (Harvey, Octus/Reorg, CoCounsel) — they focus on legal research, deal terms, or document drafting, not interrogating a specific case's filings
- **Are extremely expensive** (Bloomberg Law, Octus) — enterprise pricing for hedge funds and BigLaw

This tool fills the gap: **free, open-source, per-case AI document intelligence** built on top of the public RECAP Archive.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│         (Streamlit app — browser-based)                  │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Case Loader  │  │ Case Dashboard│  │  Q&A Chat      │  │
│  │             │  │              │  │               │  │
│  │ Enter case  │  │ Coverage map │  │ Ask questions │  │
│  │ number/URL  │  │ Doc types    │  │ Get cited     │  │
│  │             │  │ Filing stats │  │ answers       │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
└─────────┼────────────────┼──────────────────┼──────────┘
          │                │                  │
          ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Core Pipeline                          │
│                                                          │
│  1. FETCH         2. CLASSIFY       3. INDEX             │
│  CourtListener ──► Regex-based  ──► Chunk documents      │
│  API: docket,     document type    Embed with FLP's      │
│  entries, docs    classification   ModernBERT model      │
│                                    Store in ChromaDB     │
│                                                          │
│  4. QUERY                                                │
│  User question ──► Retrieve relevant chunks ──► LLM      │
│  (optional type    from vector DB              generates  │
│   filter)                                      answer     │
│                                                w/ ECF     │
│                                                citations  │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                  External Services                       │
│                                                          │
│  CourtListener API ─── Free, requires API token          │
│  (dockets, entries,    https://www.courtlistener.com/    │
│   documents, text)                                       │
│                                                          │
│  FLP ModernBERT ────── Free, runs locally                │
│  Embedding Model       Free-Law-Project/                 │
│                        modernbert-embed-base_finetune_512│
│                                                          │
│  LLM (BYOK) ────────── User provides own API key         │
│  OpenAI / Anthropic    For Q&A generation only           │
│                                                          │
│  PACER (optional) ──── User's own credentials            │
│                        For purchasing missing docs        │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### Case Loading & Dashboard
- Enter a CourtListener docket URL, or court + docket number
- Pull full docket metadata and all available document text via API
- Display coverage stats: "342 of 1,208 docket entries have documents in RECAP"
- Categorize filings by type (motions, objections, orders, plans, claims, etc.)

### Document Classification
- Regex-based classifier for standard bankruptcy filing types
- Categories: Motions, Objections, Orders, Plans/Disclosure Statements, Proofs of Claim, Fee Applications, Schedules/Statements, Monthly Operating Reports, Notices, Other
- Enables filtered queries ("show me only objections to the plan")

### Natural Language Q&A (RAG)
- Ask questions across all indexed documents in the case
- Answers cite specific ECF docket entry numbers
- Supports structured output requests ("make me a table of...")
- Flags when document coverage is incomplete for a given question

### RECAP Ecosystem Integration
- Prompts users to set up @recap.email for automatic document contribution
- Supports purchasing docket sheets and individual documents via CourtListener's PACER API
- Every document acquired through the tool enriches the public RECAP Archive

## Architecture Details

### CourtListener API Integration (`courtlistener.py`)

The CourtListener REST API (v3/v4) provides everything we need:

```
GET /api/rest/v3/dockets/{id}/                    → Case metadata
GET /api/rest/v3/docket-entries/?docket={id}      → All docket entries (paginated)
GET /api/rest/v3/recap-documents/?docket_entry__docket={id}  → Documents with text
POST /api/rest/v4/recap/                          → Purchase from PACER (async)
```

**Key fields on recap-documents:**
- `plain_text` — extracted text content (OCR'd if necessary)
- `description` — docket entry description (used for classification)
- `date_created` — filing date
- `pacer_doc_id` — PACER document identifier
- `is_available` — whether the PDF is in RECAP

**Rate limits:** 5,000 queries/hour for authenticated users. Use field selection to minimize payload size:
```
?fields=id,plain_text,description,date_created,pacer_doc_id,is_available
```

**Pagination:** Results come in pages of 20. The API returns a `next` URL for pagination.

### Document Classification (`classifier.py`)

Bankruptcy docket entries follow predictable naming patterns. A regex-based classifier covers ~80% of filings:

```python
PATTERNS = {
    "motion": r"(?i)(motion|emergency motion|omnibus motion|application.*to)\b",
    "objection": r"(?i)(objection|limited objection|response.*objection)\b",
    "order": r"(?i)(order granting|order denying|order approving|order authorizing)\b",
    "plan": r"(?i)(plan of reorganization|amended plan|disclosure statement|plan supplement)\b",
    "claim": r"(?i)(proof of claim|amended.*claim|claim no\.)\b",
    "fee_application": r"(?i)(application.*compensation|fee application|interim fee)\b",
    "schedule": r"(?i)(schedule[s]?|statement of financial affairs|monthly operating report)\b",
    "notice": r"(?i)(notice of|certificate of service|certificate of no objection)\b",
}
```

Ambiguous entries fall through to "other." Future enhancement: LLM-based classification for edge cases.

### Indexing Pipeline (`indexer.py`)

**Chunking strategy:**
1. Short documents (< 512 tokens): Keep as single chunk, preserving full context
2. Medium documents (512-2048 tokens): Split into ~512-token chunks with 50-token overlap
3. Long documents (> 2048 tokens): Split into ~512-token chunks with 50-token overlap, with section-aware splitting where possible

**Each chunk stores metadata:**
```python
{
    "docket_entry_id": 47,
    "ecf_number": "ECF No. 47",
    "description": "Motion to Approve Disclosure Statement",
    "doc_type": "motion",
    "date_filed": "2024-03-15",
    "filed_by": "Debtor",  # if extractable from entry
    "chunk_index": 0,
    "total_chunks": 3,
}
```

**Embedding model (configurable):**
- **Default: FLP ModernBERT** (`Free-Law-Project/modernbert-embed-base_finetune_512`)
  - Free, runs locally, legal-domain-adapted
  - ~400MB model download, runs on CPU in seconds per chunk
  - Note: uses `search_document:` and `search_query:` prefixes
- **Optional: OpenAI** (`text-embedding-3-small`)
  - API-based, ~$0.12 per large case
  - Set `EMBEDDING_PROVIDER=openai` in config

**Vector store:** ChromaDB (local, file-based, no server required)

### RAG Query Pipeline (`query.py`)

```
User question
    │
    ├──► (Optional) Filter by doc_type metadata
    │
    ├──► Embed question using same model
    │
    ├──► Retrieve top-k chunks from ChromaDB (k=10-15)
    │
    ├──► Assemble context with chunk metadata
    │
    ├──► Send to LLM with system prompt + context + question
    │
    └──► Return answer with ECF citations
```

**LLM options (BYOK):**
- Anthropic Claude (Haiku for cost efficiency, Sonnet for quality)
- OpenAI GPT-4o / GPT-4o-mini

### System Prompt (`prompts/system_prompt.txt`)

The system prompt is critical for answer quality. Key instructions:
- Answer based ONLY on indexed documents from this case
- ALWAYS cite docket entry numbers (e.g., "ECF No. 47")
- Distinguish between what documents state vs. inferences
- When asked for tables/structured output, format as markdown tables
- Flag incomplete coverage: "Note: I only have access to X of Y docket entries"
- Use correct bankruptcy terminology
- When uncertain, say so rather than hallucinate

## Setup & Installation

### Prerequisites
- Python 3.10+
- A CourtListener API token (free: https://www.courtlistener.com/sign-in/)
- An LLM API key (Anthropic or OpenAI)

### Install
```bash
git clone https://github.com/[you]/recap-bankruptcy-ai.git
cd recap-bankruptcy-ai
pip install -r requirements.txt
```

### Configure
```bash
cp .env.example .env
# Edit .env with your API keys:
#   COURTLISTENER_API_TOKEN=your_token_here
#   ANTHROPIC_API_KEY=your_key_here    # or OPENAI_API_KEY
#   EMBEDDING_PROVIDER=flp             # or "openai"
#   LLM_PROVIDER=anthropic             # or "openai"
#   LLM_MODEL=claude-haiku-4-5-20251001  # or claude-sonnet-4-5-20250929, gpt-4o, etc.
```

### Run
```bash
streamlit run app.py
```

## Usage

1. **Load a case:** Enter a CourtListener docket URL or case identifier
2. **Review coverage:** See which documents are available and which are missing
3. **Index:** Click "Index Case" to embed all available documents (takes 1-5 min for a large case)
4. **Ask questions:**
   - "When was the disclosure statement filed?"
   - "What are the main arguments in the objections to the plan?"
   - "Make me a table of all fee applications with applicant, amount requested, and date"
   - "Summarize the debtor's first day motions"
   - "Which creditors filed proofs of claim over $1 million?"

## Expanding Coverage (Free)

The RECAP Archive is crowdsourced. You can help populate it:

### For attorneys appearing in the case:
Set up @recap.email as your "first look" notification email in PACER/ECF. Every filing you receive will be automatically contributed to RECAP — free for you, free for everyone.

1. Log into your CourtListener account
2. Find your unique @recap.email address in settings
3. Add it as a secondary notification email in your PACER ECF account
4. That's it — documents flow in automatically

### For anyone with PACER credentials:
Use this tool's built-in PACER purchase feature to buy specific documents or the full docket sheet. Documents purchased through CourtListener's API are automatically added to RECAP.

## Cost Estimates

| Component | Cost |
|-----------|------|
| Indexing (FLP model, default) | $0.00 |
| Indexing (OpenAI embeddings, large case) | ~$0.12 |
| Q&A queries (Claude Haiku) | ~$0.003/question |
| Q&A queries (Claude Sonnet) | ~$0.02/question |
| Q&A queries (GPT-4o) | ~$0.01/question |
| PACER docket purchase (optional) | $0.10/page, $3.00 cap |

A power user could index a large case and ask 50 questions for well under $1.

## Project Structure

```
recap-bankruptcy-ai/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── config.py              # Configuration & API key management
├── courtlistener.py       # CourtListener API client
├── classifier.py          # Bankruptcy document type classification
├── indexer.py             # Chunking, embedding, vector store
├── query.py               # RAG retrieval + LLM Q&A
├── app.py                 # Streamlit UI
├── prompts/
│   └── system_prompt.txt  # LLM system prompt for Q&A
└── data/                  # Local ChromaDB storage (gitignored)
```

## Contributing

This is an open-source proof of concept. Contributions welcome:

- **Better classification:** LLM-based document type classification for ambiguous entries
- **Claims register parsing:** Structured extraction from proofs of claim
- **Timeline generation:** Automatic case timeline from key docket events
- **Multi-case support:** Compare filings across related cases
- **Export:** Generate case summary reports as PDF/DOCX

## Acknowledgments

- **[Free Law Project](https://free.law/)** — for RECAP, CourtListener, the API, the embedding model, and the mission
- **[LegalQuants](https://legalquants.com/)** — community of lawyers and builders in legal AI
- Built with data from the RECAP Archive. If this tool is useful to you, please [donate to Free Law Project](https://free.law/donate/) or [become a member](https://www.courtlistener.com/sign-in/).

## License

MIT License — see LICENSE file.

## Disclaimer

This tool is an experimental proof of concept for educational and research purposes. It is NOT a legal research tool and does NOT provide legal advice. All outputs should be independently verified by a qualified professional. See the full disclaimer for details.
