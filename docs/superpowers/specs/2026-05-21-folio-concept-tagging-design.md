# FOLIO Concept Tagging — Design Spec

**Date:** 2026-05-21
**Status:** Draft for review
**Scope:** Spike — add FOLIO bankruptcy-practice concept tags to indexed chunks, use them to improve retrieval ranking and LLM grounding.

---

## 1. Motivation

The current pipeline classifies each docket entry into one of 12 broad `DocType` buckets (`MOTION`, `ORDER`, `OBJECTION`, etc.). That works for UI filtering but carries no substantive-topic signal — two `MOTION` chunks may be about wildly different topics (employee wages vs. lifting the automatic stay), and retrieval has no way to tell them apart beyond raw vector similarity.

FOLIO (Federated Open Legal Information Ontology) ships an 18,000-concept legal taxonomy with strong bankruptcy coverage — substantive concepts like *Automatic Stay Practice*, *Adequate Protection Practice*, *Bankruptcy Claims Practice*, *Fee Application Process*. Tagging chunks with these concepts gives us a second signal layer orthogonal to `DocType`: *what kind of filing* vs. *what it is about*.

**Goals:**
1. Improve retrieval — surface chunks whose concept tags overlap with the user's query, even when raw vector similarity ranks them lower.
2. Improve answer grounding — let the LLM see concept labels alongside chunk text so it can use precise legal terminology.

**Non-goals (explicit):**
- Replacing [classifier.py](../../../classifier.py). The 12-bucket regex classifier remains the UI filter; FOLIO concepts are additive.
- UI filter chip for concepts (deferred — evaluate first).
- Doc-artifact concepts (separate FOLIO subtree, follow-up).
- An `OPINION` `DocType` to distinguish orders/opinions/pleadings (tracked separately).

## 2. Architecture

```
┌──────────────────────┐
│ scripts/             │   one-time / refresh:
│  fetch_folio.py      │   traverse Bankruptcy Practice subtree
└──────────┬───────────┘   via FOLIO public API
           ▼
┌──────────────────────┐
│ data/folio/          │   cached catalog:
│  concepts.json       │   {iri, short_name, label, alt_labels,
│  concepts.npy        │    definition, embed_text, parent_iri,
└──────────┬───────────┘    children_iris, depth}[]
           ▼                + FLP embeddings, one row per concept
┌──────────────────────┐
│ folio_tags.py        │   load_catalog() → (concepts, embeddings)
│                      │   tag_text(text) → [short_name]
│                      │   tag_embedding(vec) → [short_name]
│                      │   format_for_llm(concepts_str) → "[Concepts: ...]"
└──────┬─────────┬─────┘
       │         │
       ▼         ▼
   indexer.py   query.py
   (tag chunks  (tag query +
   during       re-rank +
   index_case)  inline in prompt)
```

**Key decisions:**

- **Catalog is fetched, not bundled.** `scripts/fetch_folio.py` traverses descendants of the Bankruptcy Practice IRI (`R7LI3BONqNkXbKHKa0t3jyI`) via the public API and writes `concepts.json` + `concepts.npy`. Re-run to refresh.
- **No new dependencies.** Uses the existing FLP embedding model, `requests`, and `numpy`. Does not pull in `folio-python`.
- **Tags stored as a pipe-delimited string** in chunk metadata (`concepts: "automatic_stay|adequate_protection"`). ChromaDB metadata is scalar-only; pipe delimiter avoids label-comma collisions. Re-ranking and LLM inlining happen in Python after retrieval, so we don't need ChromaDB to filter on concept membership.
- **Short-names, not raw IRIs, in metadata** for readability. Mapping back to IRIs lives in `concepts.json`.

## 3. Catalog Fetch

**`scripts/fetch_folio.py`** builds the local catalog.

**Traversal:**
1. Start at the Bankruptcy Practice root IRI (`R7LI3BONqNkXbKHKa0t3jyI`).
2. Fetch the root node via `/{iri}`.
3. For each node, list its direct children via `/search/query?parent_iri=<iri>` (paginated; script handles pagination).
4. Recurse on children. Cap depth at 6 to avoid runaway traversal.
5. Deduplicate by IRI (multiple inheritance is possible).

**Per-concept capture:**
| Field | Source | Use |
|---|---|---|
| `iri` | API | Traceability, mapping back |
| `short_name` | derived: `slugify(label).replace("-practice", "")` | Chunk metadata, LLM inlining |
| `label` | API | Display |
| `alt_labels` | API | Synonym matching |
| `definition` | API | Embedding context |
| `embed_text` | derived: `label + ". " + " ".join(alt_labels) + ". " + definition` | What we embed |
| `parent_iri`, `children_iris` | API | Future use (relations) |
| `depth` | derived | Future use |

Multilingual labels, `seeAlso`, `relatedConcept`, and the `/connections` endpoint are skipped — empty or low-value for bankruptcy concepts.

**Embedding step:** the same script loads the FLP model and calls `embed_texts([c["embed_text"] for c in concepts])`. Output: `data/folio/concepts.npy` — a 2-D float array, row order matches `concepts.json`.

**Re-running:** overwrites the cache. No merge logic — the canonical source is FOLIO.

## 4. Chunk Tagging During Indexing

**Where:** [indexer.py:233](../../../indexer.py:233) — after `embed_texts(all_chunks)`, before the ChromaDB batch insert.

**Algorithm:**
```python
chunk_embeddings = np.array(embeddings)              # (N_chunks, 768)
concepts, concept_embeddings = folio_tags.load_catalog()  # list[Concept], (N_concepts, 768)

similarities = chunk_embeddings @ concept_embeddings.T    # (N_chunks, N_concepts)

for i, meta in enumerate(all_metadatas):
    top = np.argsort(-similarities[i])[:FOLIO_TOP_N_CONCEPTS]
    keep = [j for j in top if similarities[i, j] >= FOLIO_MIN_SIMILARITY]
    meta["concepts"] = "|".join(concepts[j].short_name for j in keep)
    meta["concepts_score"] = round(float(similarities[i, keep[0]]), 3) if keep else 0.0
```

Indexer bypasses the convenience `tag_embedding()` API and computes the similarity matrix in one numpy operation — much faster than per-chunk calls when batching thousands of chunks. `tag_embedding()` and `tag_text()` exist for single-shot use (e.g., tagging a query).

**Encoding:**
- `concepts` (string, pipe-delimited short names). Empty match → `""` (not absent — schema stable).
- `concepts_score` (float). Top match's similarity. Telemetry only; not used in retrieval logic.

**Both content pools tagged:** the document-chunk loop ([indexer.py:181](../../../indexer.py:181)) and the docket-entry-description loop ([indexer.py:201](../../../indexer.py:201)) get the same treatment. Docket-entry descriptions are short — expect 1–2 tags each; document chunks may hit the full top-5.

**Incremental path:** `index_single_document()` ([indexer.py:249](../../../indexer.py:249)) gets the same tagging. Concept embeddings are loaded once at module level via `lru_cache`; per-chunk cost is one matmul against a ~100-row matrix.

**Existing indexed cases:** the `concepts` field is absent on chunks indexed before this change. Query code defaults to `""` when missing — no migration needed. Users who want tags on an existing case re-run the index.

## 5. Query-Side Tagging, Re-Ranking, LLM Inlining

Symmetric to indexing.

**Step 1 — tag the query** (in [query.py](../../../query.py) before retrieval):
```python
query_tags = folio_tags.tag_text(question)   # e.g. ["automatic_stay", "adequate_protection"]
```
Empty result → re-ranking is a no-op; behavior identical to today.

**Step 2 — re-rank retrieved chunks.** Fetch `2 * top_k` from ChromaDB so the re-ranker has room to promote concept-relevant chunks that ranked just outside top-K.

```python
def rerank(chunks, query_tags, alpha=FOLIO_RERANK_ALPHA, k=top_k):
    if not query_tags:
        return chunks[:k]
    qset = set(query_tags)
    for c in chunks:
        c_tags = set(c["metadata"].get("concepts", "").split("|")) - {""}
        overlap = len(qset & c_tags) / len(qset)
        vector_score = 1 - c["distance"]
        c["combined"] = (1 - alpha) * vector_score + alpha * overlap
    return sorted(chunks, key=lambda c: -c["combined"])[:k]
```

`ALPHA = 0.25` default — soft signal, not hard override.

**Step 3 — inline concepts in LLM context.** Each retrieved chunk's header gets a `[Concepts: ...]` suffix when tags are present:
```
[ECF No. 42, Motion, 2024-03-15] [Concepts: automatic_stay, adequate_protection]
<chunk text>
```
Empty tags → no bracket; format is identical to today.

**System prompt addition** ([prompts/system_prompt.txt](../../../prompts/system_prompt.txt)):
> Some chunks include a `[Concepts: ...]` annotation. These are standardized FOLIO legal-concept labels identifying the substantive bankruptcy topics present in the chunk. You may use these labels to inform your reasoning and choose precise terminology in your answer. ECF numbers remain the authoritative citation source.

## 6. Configuration

New entries in [config.py](../../../config.py):
```python
FOLIO_ENABLED = os.getenv("FOLIO_ENABLED", "true").lower() == "true"
FOLIO_CATALOG_DIR = DATA_DIR / "folio"
FOLIO_TOP_N_CONCEPTS = int(os.getenv("FOLIO_TOP_N_CONCEPTS", "5"))
FOLIO_MIN_SIMILARITY = float(os.getenv("FOLIO_MIN_SIMILARITY", "0.40"))
FOLIO_RERANK_ALPHA = float(os.getenv("FOLIO_RERANK_ALPHA", "0.25"))
FOLIO_BANKRUPTCY_ROOT_IRI = "R7LI3BONqNkXbKHKa0t3jyI"
```

`FOLIO_ENABLED=false` disables tagging at index time and re-ranking at query time. Useful for A/B comparison and for users who haven't run the fetch script.

## 7. Error Handling

Tagging is best-effort. No new exceptions raised into [indexer.py](../../../indexer.py) or [query.py](../../../query.py).

| Condition | Behavior |
|---|---|
| `concepts.json` / `concepts.npy` missing | `load_catalog()` logs warning, returns empty catalog. Tagging is a no-op. App functions. |
| Catalog schema mismatch (e.g., old format) | Caught on load, logged, no-op. |
| Fetch script failure (API down, network) | Retry with exponential backoff, fail loudly. Acceptable — offline tooling. |
| Empty chunk text | `tag_text()` returns `[]`. |

## 8. Testing

**`tests/test_folio_tags.py`** — unit tests, fixture catalog (3 hand-crafted concepts + embeddings):
- `tag_text()` returns expected concepts above threshold.
- `tag_text()` drops below-threshold matches.
- `tag_text()` on empty/whitespace input returns `[]`.
- Encoding round-trip: tag → join → split → set preserves membership.

**`tests/test_indexer_folio.py`** — integration with fixture catalog:
- Index a synthetic case; verify chunks have `concepts` metadata.
- `FOLIO_ENABLED=false` produces chunks with `concepts: ""`.

**`tests/test_query_folio.py`**:
- Re-rank with non-empty query tags promotes overlapping chunks.
- Re-rank with empty query tags is identity.
- Over-fetch then truncate to `top_k` after re-rank.

**No live network in tests.** `scripts/fetch_folio.py` is exercised manually, not in CI.

## 9. Rollout

1. Land `folio_tags.py` + `scripts/fetch_folio.py` + tests. Default `FOLIO_ENABLED=true`.
2. README addition: "Run `python scripts/fetch_folio.py` once to enable FOLIO concept tagging."
3. Existing indexed cases keep working untouched; users get tags on next re-index.
4. Update [todo.md](../../../todo.md) to mark FOLIO integration as in-progress; note follow-ups (evaluation on real cases, doc-artifact subtree).

## 10. Out of Scope

- UI multi-select for concepts.
- Doc-artifact concepts (separate field, follow-up).
- Auto-refresh of catalog at app start.
- Per-concept telemetry dashboards.
- `OPINION` `DocType` upgrade to [classifier.py](../../../classifier.py) (separate change).

## 11. Open Questions

- **Threshold calibration.** `FOLIO_MIN_SIMILARITY = 0.40` is a guess. Plan to inspect tag quality on 1–2 real cases after first index and adjust if too noisy / too sparse.
- **`ALPHA` calibration.** `0.25` is a guess. May tune after qualitative review of re-ranking impact.
- **Catalog size.** Bankruptcy Practice subtree has 27 direct children at depth 1, with deeper descendants below. Final count is empirical (run the fetch script to see). If catalog >300 concepts, may revisit depth cap or matching threshold.
