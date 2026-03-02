# Roadmap

## Near-Term

### ~~PACER Document Purchase Flow~~ ✓ Done
Implemented: LLM identifies description-only sources via structured JSON output, suggests specific documents to purchase. UI shows checkboxes for selective purchase, then fetches via CourtListener API, indexes incrementally, and re-answers.

### UX Improvements

Findings from UX research (March 2026). Prioritized by impact.

**Quick wins:**
- [x] ~~Split dashboard and chat~~ — Compact case header with collapsible "Case Details & Controls" expander; chat immediately below. Tried tabs but case overview was too hidden.
- [x] Replace single spinner with `st.status` for multi-step RAG pipeline (classify → search descriptions → retrieve chunks → generate answer)
- [x] Move doc type filter to sidebar + "Load Different Case" button to sidebar for case navigation
- [ ] Show top 2-3 source cards inline below the answer; only use expander for full list when many sources

**Medium effort:**
- [ ] Stream LLM responses with `st.write_stream()` instead of spinner → full dump
- [ ] Make ECF numbers in answers clickable (link to CourtListener document page)
- [ ] Add per-answer coverage indicator ("Answer based on 8 sources — 5 full text, 3 description only")
- [ ] Use `@st.cache_resource` for embedding model and ChromaDB client

**Fixes:**
- [ ] Work around `st.expander` disappearing inside `st.chat_message` on new submissions (known Streamlit bug) — show sources inline for latest message
- [ ] Reduce excessive `st.rerun()` calls that cause flickering

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
