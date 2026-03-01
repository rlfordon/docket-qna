# Roadmap

## Near-Term

### PACER Document Purchase Flow
When the system can't fully answer a question due to missing documents, offer to purchase the specific document from PACER, index it, and re-answer.

**Flow:** Answer with what's available → identify the gap → find the ECF number in `case.entries` → show estimated cost → user confirms → purchase via CourtListener API → index incrementally → re-run the question.

**Design decisions:**
- User provides their own PACER login (`PACER_USERNAME`/`PACER_PASSWORD` in `.env`)
- Most PACER documents are capped at $3 per document
- Always show estimated cost and require confirmation before purchasing
- After purchase + indexing, automatically re-run the question

**Requires:**
- Incremental indexing — `add_document()` method on `CaseIndex` that appends to the existing ChromaDB collection without rebuilding the whole index
- Gap detection — parse the LLM's answer or use structured output to identify what document types are missing
- Purchase confirmation UI in Streamlit

### Question Classifier Improvements
The regex-based classifier handles 84 test cases but has known limitations with novel phrasings. Options being evaluated:
- LLM-based classifier using Haiku (~$0.0003/call, ~800ms latency) — tested at 72/81 accuracy without tuning, could improve with few-shot examples
- Hybrid approach — regex as primary, LLM fallback for low-confidence cases
- Expanding the regex test suite with real user questions (tracked in `tests/questions_from_users.md`)

## Medium-Term

### Claims Register Parsing
Structured extraction from proofs of claim — pull creditor names, amounts, claim types, and status into a queryable format. Would enable questions like "Which creditors filed claims over $1 million?" with precise answers.

### Timeline Generation
Automatic case timeline from key docket events. Identify milestones (petition date, first day hearings, bar dates, plan filing, confirmation) and present as a visual timeline in the dashboard.

### Export
Generate case summary reports as PDF/DOCX — case overview, key filings, timeline, and Q&A session history.

## Longer-Term

### Multi-Case Support
Compare filings across related cases (e.g., jointly administered cases, or similar Chapter 11s). Cross-case queries like "How does this plan compare to [other case]?"

### Collaborative Features
Share case indexes and Q&A sessions with colleagues. Annotate docket entries. Track case developments over time with alerts for new filings.
