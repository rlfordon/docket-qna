# RAG Index Improvements

Three improvements to retrieval quality based on review of the current pipeline against industry patterns for legal RAG systems.

---

## 1. Hybrid Search (BM25 + Vector) — High Priority

**Problem:** Pure vector (cosine similarity) search can miss or rank poorly on exact terms that are critical in legal documents — party names, statute numbers ("11 U.S.C. § 363"), dollar amounts, case numbers, defined terms.

**Solution:** Add a parallel BM25 keyword index using `rank_bm25` and merge results with vector scores using Reciprocal Rank Fusion (RRF).

### Implementation Plan

**Files to change:** `indexer.py`, `requirements.txt`

1. Add `rank-bm25>=0.2.2` to `requirements.txt`

2. Create a `BM25Index` class in `indexer.py`:
   - Stores tokenized documents alongside their IDs and metadata
   - Simple whitespace + punctuation tokenizer (`re.findall(r"[a-zA-Z0-9]+", text.lower())`)
   - `add(ids, documents, metadatas)` — tokenizes and stores docs, rebuilds BM25Okapi model
   - `query(question, top_k, where)` — scores docs via BM25, applies ChromaDB-style `where` filter, returns top-k results
   - Need a `_matches_filter(metadata, where)` helper that handles `$and`, `$or`, `$in`, `$eq` operators to reuse the same filter logic as ChromaDB

3. Add `_bm25_index: BM25Index` to `CaseIndex.__init__`

4. In `CaseIndex.index_case()`, after the ChromaDB add loop, call:
   ```python
   self._bm25_index = BM25Index()
   self._bm25_index.add(all_ids, all_chunks, all_metadatas)
   ```

5. In `CaseIndex.index_single_document()`, also call `self._bm25_index.add(...)` with the new chunks

6. Update `CaseIndex.query()` to do hybrid retrieval:
   ```python
   # Vector results (existing)
   vector_results = collection.query(...)

   # BM25 results
   bm25_results = self._bm25_index.query(question, top_k=top_k, where=where_filter)

   # Merge via Reciprocal Rank Fusion
   chunks = _merge_rrf(vector_results_as_dicts, bm25_results, top_k=top_k)
   ```

7. Implement `_merge_rrf(vector_results, bm25_results, top_k, k=60)`:
   - Score each result by `1/(k + rank)` for each list it appears in
   - Sum scores across both lists
   - Sort by combined score, return top_k
   - Use chunk ID as the merge key

### Caveats
- BM25 index is in-memory — needs to be rebuilt when `CaseIndex` is instantiated for an existing case. Options:
  - Rebuild from ChromaDB on first query (lazy): `collection.get()` returns all docs, feed to BM25Index
  - Pickle to disk alongside ChromaDB (adds complexity)
  - Lazy rebuild is simpler and fine for typical case sizes (<10k chunks)
- ChromaDB doesn't natively support BM25, so this is a two-index approach

---

## 2. Document-Level Summaries (Third Content Pool) — Medium Priority

**Problem:** Individual 512-token chunks lack the big picture. Broad questions like "what is this motion about?" or "summarize the plan" get fragmented answers because no single chunk captures the full document scope.

**Solution:** Add a third content pool (`source: "document_summary"`) with extractive summaries for each indexed document.

### Implementation Plan

**Files to change:** `indexer.py`, `query.py`

1. Add `_build_document_summary(text, description, max_words=300)` function in `indexer.py`:
   - Takes first ~150 words + last ~150 words of the document
   - Prepends the document description as a title line
   - This is extractive (no LLM call), so it's fast and free
   - Example output: `"Motion to Approve Sale of Assets\n\n[first 150 words]...\n\n...[last 150 words]"`
   - Skip if document is shorter than ~200 words (the chunks already cover it)

2. In `CaseIndex.index_case()`, after the document chunk loop and before the docket entry loop, add a summary loop:
   ```python
   for entry in case.entries:
       for doc in entry.documents:
           if not doc.plain_text or len(doc.plain_text.split()) < 200:
               continue
           summary = _build_document_summary(doc.plain_text, doc.description or entry.description)
           chunk_id = f"d{case.docket_id}_doc{doc.id}_summary"
           metadata = {
               ...same fields as document chunks...,
               "source": "document_summary",
           }
           all_chunks.append(summary)
           all_metadatas.append(metadata)
           all_ids.append(chunk_id)
   ```

3. In `CaseIndex.index_single_document()`, also create and index a summary chunk

4. Add `CaseIndex.query_summaries()` method — same as `query_descriptions()` but with `source_filter="document_summary"`

5. In `query.py` `query_case()`, for `analytical` and `type_analysis` intents, also query summaries:
   ```python
   summary_hits = index.query_summaries(question, top_k=5, doc_type_filter=effective_type_filter)
   ```
   Include summary hits in the combined results, deduplicating by entry ID against doc chunks

6. Update `format_context()` in `query.py` to label summary sources:
   ```python
   if source == "document_summary":
       desc_tag = " (DOCUMENT SUMMARY)"
   ```

### Caveats
- Extractive summaries (first + last paragraphs) work well for legal docs where the intro states the relief sought and the conclusion summarizes
- This adds ~1 chunk per document, so index size grows modestly
- For a future iteration, could use LLM-generated summaries, but that adds cost and latency at index time

---

## 3. Increase Chunk Overlap — DONE (2026-05-23)

Default `CHUNK_OVERLAP` bumped from 50 → 100 tokens in `config.py`; docs updated. Existing indexed cases still use 50-token overlap until re-indexed — consider surfacing a re-index hint in the UI/logs after config changes.

---

## Priority Order

1. **Hybrid search** (2-4 hrs) — biggest retrieval quality improvement
2. **Document summaries** (1-2 hrs) — helps broad/analytical questions

---

## FOLIO concept tagging

- [x] Spec: docs/superpowers/specs/2026-05-21-folio-concept-tagging-design.md
- [x] Plan: docs/superpowers/plans/2026-05-21-folio-concept-tagging.md
- [x] Implemented (this plan)

### Follow-ups after first real-case eval
- Calibrate FOLIO_MIN_SIMILARITY against tag quality on 1–2 real cases.
- Calibrate FOLIO_RERANK_ALPHA after qualitative review of re-ranking impact.
- Consider pulling doc-artifact subtree if retrieval improvement is real.
- Add OPINION DocType to classifier.py to better separate orders/opinions/pleadings.
- UI multi-select for concept filtering (deferred).
