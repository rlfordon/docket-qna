# FOLIO Concept Tagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FOLIO bankruptcy-practice concept tags to indexed chunks via embedding similarity, and use those tags to re-rank retrieval results and inline concept labels into the LLM context.

**Architecture:** New `folio_tags.py` module owns concept fetching, embedding, and matching. A `scripts/fetch_folio.py` script builds a local catalog (one-time / refresh). [indexer.py](../../../indexer.py) tags chunks during `index_case()` / `index_single_document()` by reusing the chunk embeddings it already computes. [query.py](../../../query.py) tags the user's query, fetches `2*top_k` chunks from ChromaDB, re-ranks by weighted vector + concept overlap, and inlines concept labels into chunk headers sent to the LLM. `FOLIO_ENABLED=false` is a kill switch that makes the whole layer a no-op.

**Tech Stack:** Python 3.10+, numpy, requests, sentence-transformers (FLP ModernBERT — already loaded), ChromaDB (already used), pytest, `unittest.mock` (existing convention for mocking chromadb in unit tests).

**Spec reference:** [docs/superpowers/specs/2026-05-21-folio-concept-tagging-design.md](../specs/2026-05-21-folio-concept-tagging-design.md)

**Spec correction applied here:** The spec referenced `prompts/system_prompt.txt` but the actual file is `system_prompt.txt` at the repo root. This plan uses the correct path.

---

## File Structure

**Created:**
- `folio_tags.py` — module with `Concept`, `slugify()`, `load_catalog()`, `tag_embedding()`, `tag_text()`, `format_for_llm()`
- `scripts/fetch_folio.py` — catalog fetcher (manual invocation)
- `scripts/__init__.py` — package marker (empty)
- `tests/test_folio_tags.py` — unit tests for `folio_tags`
- `tests/test_indexer_folio.py` — integration tests for indexer tagging
- `tests/test_query_folio.py` — integration tests for query tagging/rerank
- `tests/fixtures/folio/concepts.json` — tiny hand-crafted catalog (3 concepts)
- `tests/fixtures/folio/concepts.npy` — matching 3×4 hand-crafted embeddings

**Modified:**
- [config.py](../../../config.py) — add FOLIO_* config vars
- [indexer.py](../../../indexer.py) — tag chunks in `index_case()` and `index_single_document()`
- [query.py](../../../query.py) — tag query, re-rank, inline concepts in chunk headers
- [system_prompt.txt](../../../system_prompt.txt) — add `[Concepts: ...]` annotation explanation
- [README.md](../../../README.md) — add fetch-script setup note
- [todo.md](../../../todo.md) — track FOLIO integration status

---

## Task 1: Add FOLIO config

**Files:**
- Modify: [config.py](../../../config.py)

- [ ] **Step 1: Inspect current config.py to find the right insertion point**

Run: `grep -n "DATA_DIR\|CHROMA_DIR" config.py`

Note where `DATA_DIR` is defined — FOLIO_CATALOG_DIR depends on it.

- [ ] **Step 2: Add FOLIO config constants**

In `config.py`, after the existing data-directory definitions, add:

```python
# FOLIO concept tagging
FOLIO_ENABLED = os.getenv("FOLIO_ENABLED", "true").lower() == "true"
FOLIO_CATALOG_DIR = DATA_DIR / "folio"
FOLIO_TOP_N_CONCEPTS = int(os.getenv("FOLIO_TOP_N_CONCEPTS", "5"))
FOLIO_MIN_SIMILARITY = float(os.getenv("FOLIO_MIN_SIMILARITY", "0.40"))
FOLIO_RERANK_ALPHA = float(os.getenv("FOLIO_RERANK_ALPHA", "0.25"))
FOLIO_BANKRUPTCY_ROOT_IRI = "R7LI3BONqNkXbKHKa0t3jyI"
```

Confirm `os` and `DATA_DIR` are already in scope at the insertion point. If `os` is not imported, add `import os` at top.

- [ ] **Step 3: Verify config loads**

Run: `python -c "import config; print(config.FOLIO_ENABLED, config.FOLIO_TOP_N_CONCEPTS, config.FOLIO_CATALOG_DIR)"`

Expected: `True 5 <path-to-data>/folio`

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "Add FOLIO config constants"
```

---

## Task 2: Create `folio_tags.py` scaffold with `Concept` and `slugify`

**Files:**
- Create: `folio_tags.py`
- Test: `tests/test_folio_tags.py`

- [ ] **Step 1: Write the failing test for `slugify`**

Create `tests/test_folio_tags.py`:

```python
"""Tests for folio_tags module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from folio_tags import slugify


def test_slugify_strips_practice_suffix():
    assert slugify("Relief from Stay Practice") == "relief_from_stay"


def test_slugify_handles_punctuation():
    assert slugify("Fee & Employment Practice") == "fee_employment"


def test_slugify_lowercases_and_underscores():
    assert slugify("Adequate Protection") == "adequate_protection"


def test_slugify_collapses_whitespace():
    assert slugify("Chapter 11   Bankruptcy  Plan") == "chapter_11_bankruptcy_plan"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'folio_tags'`

- [ ] **Step 3: Create the module with `Concept` dataclass and `slugify`**

Create `folio_tags.py`:

```python
"""FOLIO concept tagging — embedding-similarity matching of chunks to legal concepts.

The catalog (concepts + embeddings) is built by scripts/fetch_folio.py and
cached under config.FOLIO_CATALOG_DIR. This module reads the cache and
exposes match helpers used by indexer.py and query.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Concept:
    iri: str
    short_name: str
    label: str
    alt_labels: list[str] = field(default_factory=list)
    definition: str = ""
    embed_text: str = ""
    parent_iri: str = ""
    children_iris: list[str] = field(default_factory=list)
    depth: int = 0


_SUFFIX_RE = re.compile(r"_practice$")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")


def slugify(label: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace to underscores,
    and remove a trailing '_practice' suffix.

    Used to derive a stable short_name from a FOLIO label.
    """
    lowered = label.lower().strip()
    underscored = _NONWORD_RE.sub("_", lowered).strip("_")
    return _SUFFIX_RE.sub("", underscored)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add folio_tags.py tests/test_folio_tags.py
git commit -m "Add folio_tags module scaffold with Concept and slugify"
```

---

## Task 3: Build test fixture catalog and implement `load_catalog()`

**Files:**
- Create: `tests/fixtures/folio/concepts.json`
- Create: `tests/fixtures/folio/concepts.npy`
- Create: `tests/conftest.py`
- Modify: `folio_tags.py`
- Modify: `tests/test_folio_tags.py`

The fixture has three concepts and three unit-vector embeddings in 4-D, designed so cosine similarity is exactly predictable:
- `automatic_stay` → `[1, 0, 0, 0]`
- `adequate_protection` → `[0, 1, 0, 0]`
- `proof_of_claim` → `[0, 0, 1, 0]`

- [ ] **Step 1: Create the fixture JSON**

Create `tests/fixtures/folio/concepts.json`:

```json
[
  {
    "iri": "TEST_AS",
    "short_name": "automatic_stay",
    "label": "Automatic Stay",
    "alt_labels": ["Stay", "Bankruptcy Stay"],
    "definition": "Injunction halting collection.",
    "embed_text": "Automatic Stay. Stay. Bankruptcy Stay. Injunction halting collection.",
    "parent_iri": "",
    "children_iris": [],
    "depth": 1
  },
  {
    "iri": "TEST_AP",
    "short_name": "adequate_protection",
    "label": "Adequate Protection",
    "alt_labels": [],
    "definition": "Safeguards for secured creditors.",
    "embed_text": "Adequate Protection. Safeguards for secured creditors.",
    "parent_iri": "",
    "children_iris": [],
    "depth": 1
  },
  {
    "iri": "TEST_PC",
    "short_name": "proof_of_claim",
    "label": "Proof of Claim",
    "alt_labels": [],
    "definition": "Creditor's claim document.",
    "embed_text": "Proof of Claim. Creditor's claim document.",
    "parent_iri": "",
    "children_iris": [],
    "depth": 1
  }
]
```

- [ ] **Step 2: Create the fixture embeddings**

Create a one-time helper script `tests/fixtures/folio/_build_npy.py` (kept as a dev artifact, not run in CI):

```python
"""One-time fixture builder. Run manually if the npy needs to be regenerated.

Usage: python tests/fixtures/folio/_build_npy.py
"""
import numpy as np
from pathlib import Path

vectors = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],   # automatic_stay
        [0.0, 1.0, 0.0, 0.0],   # adequate_protection
        [0.0, 0.0, 1.0, 0.0],   # proof_of_claim
    ],
    dtype=np.float32,
)
out = Path(__file__).parent / "concepts.npy"
np.save(out, vectors)
print(f"Wrote {out} shape={vectors.shape}")
```

Run it once: `python tests/fixtures/folio/_build_npy.py`
Expected: `Wrote .../concepts.npy shape=(3, 4)`

- [ ] **Step 3: Add a pytest fixture for the catalog path**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def folio_catalog_dir():
    """Path to the test FOLIO catalog fixture."""
    return Path(__file__).parent / "fixtures" / "folio"
```

- [ ] **Step 4: Write failing tests for `load_catalog`**

Append to `tests/test_folio_tags.py`:

```python
import numpy as np
from folio_tags import load_catalog


def test_load_catalog_returns_concepts_and_embeddings(folio_catalog_dir):
    concepts, embeddings = load_catalog(folio_catalog_dir)
    assert len(concepts) == 3
    assert concepts[0].short_name == "automatic_stay"
    assert concepts[1].short_name == "adequate_protection"
    assert concepts[2].short_name == "proof_of_claim"
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 4)


def test_load_catalog_missing_files_returns_empty(tmp_path):
    concepts, embeddings = load_catalog(tmp_path)
    assert concepts == []
    assert embeddings.shape == (0,) or embeddings.size == 0


def test_load_catalog_row_order_matches_concepts(folio_catalog_dir):
    concepts, embeddings = load_catalog(folio_catalog_dir)
    # Row 0 should be the automatic_stay vector [1, 0, 0, 0]
    assert embeddings[0, 0] == 1.0
    assert embeddings[1, 1] == 1.0
    assert embeddings[2, 2] == 1.0
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 3 new failures with `ImportError: cannot import name 'load_catalog'`

- [ ] **Step 6: Implement `load_catalog()`**

Append to `folio_tags.py`:

```python
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_catalog(catalog_dir: Path) -> tuple[list[Concept], np.ndarray]:
    """Load the FOLIO concept catalog from disk.

    Reads concepts.json and concepts.npy from catalog_dir. Returns
    (concepts, embeddings). Returns ([], np.empty((0, 0))) if either
    file is missing — tagging then becomes a no-op.

    Row order in embeddings matches concept order in concepts.json.
    """
    catalog_dir = Path(catalog_dir)
    json_path = catalog_dir / "concepts.json"
    npy_path = catalog_dir / "concepts.npy"

    if not json_path.exists() or not npy_path.exists():
        logger.warning(
            f"FOLIO catalog not found at {catalog_dir}. "
            f"Tagging will be a no-op. Run scripts/fetch_folio.py to build it."
        )
        return [], np.empty((0, 0), dtype=np.float32)

    with open(json_path) as f:
        raw = json.load(f)

    concepts = [Concept(**entry) for entry in raw]
    embeddings = np.load(npy_path).astype(np.float32)

    if embeddings.shape[0] != len(concepts):
        logger.error(
            f"FOLIO catalog mismatch: {len(concepts)} concepts vs "
            f"{embeddings.shape[0]} embedding rows. Returning empty catalog."
        )
        return [], np.empty((0, 0), dtype=np.float32)

    return concepts, embeddings


@lru_cache(maxsize=1)
def _cached_catalog(catalog_dir_str: str) -> tuple[list[Concept], np.ndarray]:
    """Internal cache so indexer/query don't re-read on every call."""
    return load_catalog(Path(catalog_dir_str))


def get_catalog(catalog_dir: Path | None = None) -> tuple[list[Concept], np.ndarray]:
    """Module-level cached catalog accessor.

    Use this from indexer.py and query.py; tests should call load_catalog()
    directly with a fixture path.
    """
    import config
    path = catalog_dir if catalog_dir is not None else config.FOLIO_CATALOG_DIR
    return _cached_catalog(str(path))
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 7 passed (4 slugify + 3 load_catalog).

- [ ] **Step 8: Commit**

```bash
git add folio_tags.py tests/test_folio_tags.py tests/conftest.py tests/fixtures/folio/
git commit -m "Implement FOLIO catalog loading + test fixture"
```

---

## Task 4: Implement `tag_embedding()`

**Files:**
- Modify: `folio_tags.py`
- Modify: `tests/test_folio_tags.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_folio_tags.py`:

```python
from folio_tags import tag_embedding


def test_tag_embedding_returns_top_match(folio_catalog_dir):
    concepts, embeddings = load_catalog(folio_catalog_dir)
    # Query vector identical to automatic_stay row → top match
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tags = tag_embedding(vec, concepts, embeddings, top_n=1, min_sim=0.5)
    assert tags == ["automatic_stay"]


def test_tag_embedding_respects_top_n(folio_catalog_dir):
    concepts, embeddings = load_catalog(folio_catalog_dir)
    # Equal-weight blend across all three: similarity ~0.577 each
    vec = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    vec /= np.linalg.norm(vec)
    tags = tag_embedding(vec, concepts, embeddings, top_n=2, min_sim=0.4)
    assert len(tags) == 2
    assert set(tags) <= {"automatic_stay", "adequate_protection", "proof_of_claim"}


def test_tag_embedding_drops_below_threshold(folio_catalog_dir):
    concepts, embeddings = load_catalog(folio_catalog_dir)
    # Orthogonal vector → similarity 0 to all
    vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    tags = tag_embedding(vec, concepts, embeddings, top_n=5, min_sim=0.1)
    assert tags == []


def test_tag_embedding_empty_catalog_returns_empty():
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tags = tag_embedding(vec, [], np.empty((0, 0), dtype=np.float32), top_n=5, min_sim=0.0)
    assert tags == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 4 new failures with `ImportError: cannot import name 'tag_embedding'`

- [ ] **Step 3: Implement `tag_embedding()`**

Append to `folio_tags.py`:

```python
def tag_embedding(
    vec: np.ndarray,
    concepts: list[Concept],
    embeddings: np.ndarray,
    top_n: int,
    min_sim: float,
) -> list[str]:
    """Return short_names of the top-N concepts with cosine similarity >= min_sim.

    Assumes `vec` and rows of `embeddings` are L2-normalized (FLP and our
    fixture vectors satisfy this). If they aren't, this still produces a
    sensible ranking, just not strictly cosine.
    """
    if not concepts or embeddings.size == 0:
        return []

    sims = embeddings @ vec  # (N_concepts,)
    order = np.argsort(-sims)[:top_n]
    return [concepts[int(j)].short_name for j in order if sims[int(j)] >= min_sim]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 11 passed (7 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add folio_tags.py tests/test_folio_tags.py
git commit -m "Implement tag_embedding with top-N + threshold"
```

---

## Task 5: Implement `tag_text()`

**Files:**
- Modify: `folio_tags.py`
- Modify: `tests/test_folio_tags.py`

- [ ] **Step 1: Write failing test (with FLP model mocked)**

Append to `tests/test_folio_tags.py`:

```python
from unittest.mock import patch
from folio_tags import tag_text


def test_tag_text_embeds_then_tags(folio_catalog_dir):
    # Mock the FLP embedder to return the automatic_stay vector
    fake_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("folio_tags._embed_query", return_value=fake_vec):
        tags = tag_text(
            "did the debtor lift the stay?",
            catalog_dir=folio_catalog_dir,
            top_n=1,
            min_sim=0.5,
        )

    assert tags == ["automatic_stay"]


def test_tag_text_empty_string_returns_empty(folio_catalog_dir):
    tags = tag_text("", catalog_dir=folio_catalog_dir, top_n=5, min_sim=0.0)
    assert tags == []


def test_tag_text_whitespace_returns_empty(folio_catalog_dir):
    tags = tag_text("   \n\t  ", catalog_dir=folio_catalog_dir, top_n=5, min_sim=0.0)
    assert tags == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 3 new failures with `ImportError: cannot import name 'tag_text'`

- [ ] **Step 3: Implement `tag_text()` (and the `_embed_query` indirection)**

Append to `folio_tags.py`:

```python
def _embed_query(text: str) -> np.ndarray:
    """Embed a single text via the same model the indexer uses.

    Returns a (1, D) array. Separated as its own function so tests can
    patch it without loading the real FLP model.
    """
    import indexer
    vec = indexer.embed_texts([text], is_query=True)
    return np.array(vec, dtype=np.float32)


def tag_text(
    text: str,
    catalog_dir: Path | None = None,
    top_n: int | None = None,
    min_sim: float | None = None,
) -> list[str]:
    """Tag a free-text string with FOLIO concept short_names.

    Convenience wrapper that embeds the text and calls tag_embedding.
    Empty or whitespace-only input returns []. If the catalog is missing
    or FOLIO_ENABLED is false, returns [].
    """
    import config
    if not config.FOLIO_ENABLED or not text or not text.strip():
        return []

    concepts, embeddings = get_catalog(catalog_dir)
    if not concepts:
        return []

    if top_n is None:
        top_n = config.FOLIO_TOP_N_CONCEPTS
    if min_sim is None:
        min_sim = config.FOLIO_MIN_SIMILARITY

    vec = _embed_query(text)[0]
    return tag_embedding(vec, concepts, embeddings, top_n=top_n, min_sim=min_sim)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add folio_tags.py tests/test_folio_tags.py
git commit -m "Implement tag_text wrapper"
```

---

## Task 6: Implement `format_for_llm()`

**Files:**
- Modify: `folio_tags.py`
- Modify: `tests/test_folio_tags.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_folio_tags.py`:

```python
from folio_tags import format_for_llm


def test_format_for_llm_renders_concepts():
    assert format_for_llm("automatic_stay|adequate_protection") == \
        "[Concepts: automatic_stay, adequate_protection]"


def test_format_for_llm_empty_returns_empty_string():
    assert format_for_llm("") == ""


def test_format_for_llm_single_concept():
    assert format_for_llm("proof_of_claim") == "[Concepts: proof_of_claim]"


def test_format_for_llm_strips_empty_segments():
    # Edge case: a stray trailing pipe shouldn't produce an empty label
    assert format_for_llm("automatic_stay|") == "[Concepts: automatic_stay]"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 4 new failures with `ImportError: cannot import name 'format_for_llm'`

- [ ] **Step 3: Implement `format_for_llm()`**

Append to `folio_tags.py`:

```python
def format_for_llm(concepts_str: str) -> str:
    """Render a pipe-delimited concepts string for inclusion in LLM context.

    "" → ""
    "automatic_stay" → "[Concepts: automatic_stay]"
    "automatic_stay|adequate_protection" → "[Concepts: automatic_stay, adequate_protection]"
    """
    if not concepts_str:
        return ""
    parts = [p for p in concepts_str.split("|") if p]
    if not parts:
        return ""
    return f"[Concepts: {', '.join(parts)}]"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_folio_tags.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add folio_tags.py tests/test_folio_tags.py
git commit -m "Implement format_for_llm helper"
```

---

## Task 7: Create `scripts/fetch_folio.py`

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/fetch_folio.py`

This task is exercised manually (per spec section 8 — no live network in CI).

- [ ] **Step 1: Create package marker**

Create empty `scripts/__init__.py`.

- [ ] **Step 2: Write the fetch script**

Create `scripts/fetch_folio.py`:

```python
"""Fetch the FOLIO Bankruptcy Practice subtree and build a local catalog.

Run manually:  python scripts/fetch_folio.py

Output:
  data/folio/concepts.json   list of Concept records
  data/folio/concepts.npy    matching FLP embeddings (N, 768)

Re-running overwrites both files.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root so we can import config and indexer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import requests

import config
import indexer  # for embed_texts
from folio_tags import Concept, slugify

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "https://folio.openlegalstandard.org"
MAX_DEPTH = 6
PAGE_SIZE = 100
RETRY_DELAYS = [1, 3, 9]  # seconds


def _api_get(path: str, params: dict | None = None) -> dict | list:
    """GET with simple retry/backoff."""
    url = f"{API_BASE}{path}"
    last_err: Exception | None = None
    for delay in [0] + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            logger.warning(f"GET {url} failed ({e}); retrying in {delay}s")
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")


def _fetch_node(iri: str) -> dict:
    """Fetch a single concept node by IRI."""
    return _api_get(f"/{iri}")  # type: ignore[return-value]


def _fetch_children(parent_iri: str) -> list[dict]:
    """Fetch direct children of a concept (all pages)."""
    out: list[dict] = []
    offset = 0
    while True:
        page = _api_get(
            "/search/query",
            params={"parent_iri": parent_iri, "limit": PAGE_SIZE, "offset": offset},
        )
        # The API may return either a bare list or a dict with results — handle both
        if isinstance(page, dict):
            results = page.get("results") or page.get("items") or []
        else:
            results = page
        if not results:
            break
        out.extend(results)
        if len(results) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out


def _normalize(node: dict, depth: int) -> Concept:
    """Convert an API node payload into a Concept dataclass."""
    label = node.get("label") or node.get("prefLabel") or ""
    return Concept(
        iri=node["iri"] if "iri" in node else node.get("id", ""),
        short_name=slugify(label),
        label=label,
        alt_labels=list(node.get("altLabels") or node.get("alt_labels") or []),
        definition=node.get("definition") or "",
        embed_text="",  # filled below
        parent_iri=node.get("parent_iri") or "",
        children_iris=list(node.get("children_iris") or []),
        depth=depth,
    )


def _build_embed_text(c: Concept) -> str:
    parts = [c.label]
    parts.extend(c.alt_labels)
    if c.definition:
        parts.append(c.definition)
    return ". ".join(p for p in parts if p)


def traverse(root_iri: str) -> list[Concept]:
    """BFS from root_iri down to MAX_DEPTH, deduplicating by IRI."""
    seen: dict[str, Concept] = {}
    queue: list[tuple[str, int]] = [(root_iri, 0)]

    while queue:
        iri, depth = queue.pop(0)
        if iri in seen or depth > MAX_DEPTH:
            continue
        try:
            node = _fetch_node(iri)
        except Exception as e:
            logger.warning(f"Skipping {iri}: {e}")
            continue
        c = _normalize(node, depth)
        c.embed_text = _build_embed_text(c)
        seen[iri] = c
        logger.info(f"[d={depth}] {c.label} ({c.short_name})")

        if depth < MAX_DEPTH:
            children = _fetch_children(iri)
            for child in children:
                child_iri = child.get("iri") or child.get("id")
                if child_iri and child_iri not in seen:
                    queue.append((child_iri, depth + 1))

    return list(seen.values())


def main() -> int:
    out_dir = config.FOLIO_CATALOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Traversing FOLIO from root {config.FOLIO_BANKRUPTCY_ROOT_IRI}")
    concepts = traverse(config.FOLIO_BANKRUPTCY_ROOT_IRI)
    logger.info(f"Collected {len(concepts)} concepts")

    if not concepts:
        logger.error("No concepts collected — aborting.")
        return 1

    logger.info("Embedding concept texts via FLP model...")
    vectors = indexer.embed_texts([c.embed_text for c in concepts], is_query=False)
    arr = np.array(vectors, dtype=np.float32)

    json_path = out_dir / "concepts.json"
    npy_path = out_dir / "concepts.npy"

    with open(json_path, "w") as f:
        json.dump([c.__dict__ for c in concepts], f, indent=2)
    np.save(npy_path, arr)

    logger.info(f"Wrote {json_path} ({len(concepts)} concepts)")
    logger.info(f"Wrote {npy_path} shape={arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke test the script**

Run: `python scripts/fetch_folio.py`

Expected (manual check):
- Logs show concepts being collected.
- `data/folio/concepts.json` is written with > 10 concepts (Bankruptcy Practice has ≥27 direct children per the spec).
- `data/folio/concepts.npy` has shape `(N, 768)`.

If the API endpoint shape differs from what `_normalize` / `_fetch_children` expect, adjust per actual response. Log a one-line example of the raw response when first encountering an unexpected payload so a re-run can fix it.

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/fetch_folio.py
git commit -m "Add scripts/fetch_folio.py to build FOLIO catalog"
```

---

## Task 8: Integrate tagging into `indexer.index_case()`

**Files:**
- Modify: [indexer.py](../../../indexer.py)
- Create: `tests/test_indexer_folio.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_indexer_folio.py`:

```python
"""Integration tests for FOLIO tagging in indexer.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_case():
    from courtlistener import BankruptcyCase, DocketEntry, RecapDocument
    return BankruptcyCase(
        docket_id=999,
        case_name="In re Fixture, Inc.",
        docket_number="9:99-bk-99999",
        court="S.D.N.Y.",
        date_filed="2024-06-01",
        date_terminated=None,
        chapter="11",
        trustee=None,
        assigned_to="Judge Test",
        entries=[
            DocketEntry(
                id=1,
                entry_number=1,
                description="Motion to lift the automatic stay",
                date_filed="2024-06-05",
                documents=[
                    RecapDocument(
                        id=1001,
                        ecf_number="1",
                        description="Motion for relief from automatic stay",
                        date_filed="2024-06-05",
                        plain_text=(
                            "The debtor seeks to lift the automatic stay so that "
                            "the creditor may foreclose on the collateral. The motion "
                            "argues that adequate protection has not been provided."
                        ),
                    )
                ],
            )
        ],
    )


@pytest.fixture
def patched_chroma():
    """Patch chromadb so we don't need a real persistent client."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_collection
    with patch("chromadb.PersistentClient", return_value=mock_client):
        yield mock_collection


def test_index_case_tags_chunks_when_enabled(patched_chroma, folio_catalog_dir, monkeypatch):
    import config
    monkeypatch.setattr(config, "FOLIO_ENABLED", True)
    monkeypatch.setattr(config, "FOLIO_CATALOG_DIR", folio_catalog_dir)
    monkeypatch.setattr(config, "FOLIO_TOP_N_CONCEPTS", 2)
    monkeypatch.setattr(config, "FOLIO_MIN_SIMILARITY", 0.4)

    # The FLP model isn't loaded in unit tests; stub embed_texts to return
    # vectors whose first 4 dims match the automatic_stay concept fixture.
    def fake_embed(texts, is_query=False):
        # Pad a [1,0,0,0]-leaning vector out to 4 dims (matches fixture width)
        return [[1.0, 0.1, 0.0, 0.0] for _ in texts]

    import indexer
    monkeypatch.setattr(indexer, "embed_texts", fake_embed)

    from indexer import CaseIndex
    case = _make_case()
    idx = CaseIndex(case.docket_id)
    idx.index_case(case)

    # Inspect the metadatas passed to collection.add
    add_calls = patched_chroma.add.call_args_list
    assert add_calls, "Expected collection.add to be called"
    metadatas = add_calls[0].kwargs["metadatas"]
    assert all("concepts" in m for m in metadatas)
    # Every chunk should be tagged with automatic_stay given our fake embedding
    assert any("automatic_stay" in m["concepts"] for m in metadatas)


def test_index_case_no_concepts_when_disabled(patched_chroma, folio_catalog_dir, monkeypatch):
    import config
    monkeypatch.setattr(config, "FOLIO_ENABLED", False)
    monkeypatch.setattr(config, "FOLIO_CATALOG_DIR", folio_catalog_dir)

    import indexer
    monkeypatch.setattr(indexer, "embed_texts", lambda texts, is_query=False: [[0.0, 0.0, 0.0, 1.0] for _ in texts])

    from indexer import CaseIndex
    case = _make_case()
    idx = CaseIndex(case.docket_id)
    idx.index_case(case)

    metadatas = patched_chroma.add.call_args_list[0].kwargs["metadatas"]
    assert all(m.get("concepts", "") == "" for m in metadatas)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_indexer_folio.py -v`
Expected: 2 failures — `concepts` key missing from metadata.

- [ ] **Step 3: Add tagging to `index_case()`**

In [indexer.py:233](../../../indexer.py:233) (after `embeddings = embed_texts(all_chunks, is_query=False)`), add concept tagging. Insert this block:

```python
        # FOLIO concept tagging — adds `concepts` (pipe-delimited short names)
        # and `concepts_score` (top match similarity) to each chunk's metadata.
        # No-op when FOLIO_ENABLED=false or catalog is missing.
        _attach_folio_tags(all_metadatas, embeddings)
```

Then add a module-level helper in `indexer.py` (near the top, after imports):

```python
def _attach_folio_tags(metadatas: list[dict], embeddings: list[list[float]]) -> None:
    """Tag each chunk's metadata with FOLIO concepts based on its embedding.

    Mutates `metadatas` in place. Writes the `concepts` (pipe-delimited
    short names) and `concepts_score` (float, top similarity) fields. If
    FOLIO is disabled or the catalog is missing, writes empty defaults so
    downstream code can treat the field as always-present.
    """
    import numpy as np

    import folio_tags

    if not config.FOLIO_ENABLED:
        for m in metadatas:
            m.setdefault("concepts", "")
            m.setdefault("concepts_score", 0.0)
        return

    concepts, concept_embs = folio_tags.get_catalog()
    if not concepts:
        for m in metadatas:
            m.setdefault("concepts", "")
            m.setdefault("concepts_score", 0.0)
        return

    chunk_arr = np.array(embeddings, dtype=np.float32)
    sims = chunk_arr @ concept_embs.T  # (N_chunks, N_concepts)

    for i, m in enumerate(metadatas):
        order = np.argsort(-sims[i])[: config.FOLIO_TOP_N_CONCEPTS]
        keep = [int(j) for j in order if sims[i, int(j)] >= config.FOLIO_MIN_SIMILARITY]
        m["concepts"] = "|".join(concepts[j].short_name for j in keep)
        m["concepts_score"] = (
            round(float(sims[i, keep[0]]), 3) if keep else 0.0
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_indexer_folio.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add indexer.py tests/test_indexer_folio.py
git commit -m "Tag chunks with FOLIO concepts during index_case"
```

---

## Task 9: Apply tagging in `index_single_document()`

**Files:**
- Modify: [indexer.py](../../../indexer.py)
- Modify: `tests/test_indexer_folio.py`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_indexer_folio.py`:

```python
def test_index_single_document_tags_chunks(patched_chroma, folio_catalog_dir, monkeypatch):
    import config
    monkeypatch.setattr(config, "FOLIO_ENABLED", True)
    monkeypatch.setattr(config, "FOLIO_CATALOG_DIR", folio_catalog_dir)
    monkeypatch.setattr(config, "FOLIO_TOP_N_CONCEPTS", 2)
    monkeypatch.setattr(config, "FOLIO_MIN_SIMILARITY", 0.4)

    import indexer
    monkeypatch.setattr(
        indexer, "embed_texts",
        lambda texts, is_query=False: [[0.0, 1.0, 0.0, 0.0] for _ in texts],
    )

    # Reset collection.add call history
    patched_chroma.add.reset_mock()

    # get_collection should also return our patched collection
    from indexer import CaseIndex
    case = _make_case()
    idx = CaseIndex(case.docket_id)
    idx.client.get_collection = MagicMock(return_value=patched_chroma)

    entry = case.entries[0]
    doc = entry.documents[0]
    idx.index_single_document(case, entry, doc)

    metadatas = patched_chroma.add.call_args.kwargs["metadatas"]
    assert all("concepts" in m for m in metadatas)
    # The [0,1,0,0] embedding matches adequate_protection in the fixture
    assert any("adequate_protection" in m["concepts"] for m in metadatas)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_indexer_folio.py::test_index_single_document_tags_chunks -v`
Expected: FAIL — concepts not in metadata.

- [ ] **Step 3: Add the same tagging call to `index_single_document()`**

In [indexer.py:307](../../../indexer.py:307) (after `embeddings = embed_texts(all_chunks, is_query=False)` in `index_single_document`), add:

```python
        _attach_folio_tags(all_metadatas, embeddings)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_indexer_folio.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add indexer.py tests/test_indexer_folio.py
git commit -m "Tag chunks in index_single_document incremental path"
```

---

## Task 10: Query-side tagging and re-ranking in `query.py`

**Files:**
- Modify: [query.py](../../../query.py)
- Create: `tests/test_query_folio.py`

- [ ] **Step 1: Locate the retrieval call sites in `query_case()`**

The relevant lines in [query.py](../../../query.py):
- Line 480 (`classify_question`) — right before this is where we'll tag the query.
- Line 525 (`index.query_descriptions(...)`) — Stage 1 retrieval.
- Line 545 and 558 (`index.query_documents(...)`) — Stage 2 retrieval and fallback.
- Line 580 (`all_chunks = doc_chunks + unique_desc_hits`) — re-rank point for the combined list.

- [ ] **Step 2: Write failing tests for the re-rank helper**

Create `tests/test_query_folio.py`:

```python
"""Tests for FOLIO query-side tagging and re-ranking in query.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock chromadb before importing query
sys.modules.setdefault("chromadb", MagicMock())

from query import rerank_by_concepts


def test_rerank_promotes_concept_overlap():
    query_tags = ["automatic_stay"]
    chunks = [
        # Higher vector similarity but no concept overlap
        {"text": "a", "metadata": {"concepts": "proof_of_claim"}, "distance": 0.10},
        # Lower vector similarity but matching concept
        {"text": "b", "metadata": {"concepts": "automatic_stay"}, "distance": 0.30},
    ]
    out = rerank_by_concepts(chunks, query_tags, alpha=0.5, k=2)
    assert out[0]["text"] == "b"


def test_rerank_noop_when_no_query_tags():
    chunks = [
        {"text": "a", "metadata": {"concepts": "proof_of_claim"}, "distance": 0.10},
        {"text": "b", "metadata": {"concepts": "automatic_stay"}, "distance": 0.30},
    ]
    out = rerank_by_concepts(chunks, [], alpha=0.5, k=2)
    # Same order — no re-ranking applied
    assert [c["text"] for c in out] == ["a", "b"]


def test_rerank_truncates_to_k():
    chunks = [
        {"text": f"c{i}", "metadata": {"concepts": "automatic_stay"}, "distance": 0.1 * i}
        for i in range(10)
    ]
    out = rerank_by_concepts(chunks, ["automatic_stay"], alpha=0.5, k=3)
    assert len(out) == 3


def test_rerank_handles_missing_concepts_field():
    chunks = [
        {"text": "a", "metadata": {}, "distance": 0.10},
        {"text": "b", "metadata": {"concepts": "automatic_stay"}, "distance": 0.20},
    ]
    out = rerank_by_concepts(chunks, ["automatic_stay"], alpha=0.5, k=2)
    assert out[0]["text"] == "b"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_query_folio.py -v`
Expected: 4 failures — `ImportError: cannot import name 'rerank_by_concepts'`.

- [ ] **Step 4: Implement `rerank_by_concepts` in query.py**

Add to [query.py](../../../query.py) (top-level, near other helpers):

```python
def rerank_by_concepts(
    chunks: list[dict],
    query_tags: list[str],
    alpha: float | None = None,
    k: int | None = None,
) -> list[dict]:
    """Re-rank ChromaDB results by combining vector similarity with FOLIO
    concept overlap.

    Each chunk's `combined` score is:
        (1 - alpha) * (1 - distance) + alpha * (|query_tags ∩ chunk_tags| / |query_tags|)

    Returns the top-k chunks by combined score. If query_tags is empty,
    returns chunks unchanged (truncated to k).
    """
    import config

    if alpha is None:
        alpha = config.FOLIO_RERANK_ALPHA
    if k is None:
        k = config.RETRIEVAL_TOP_K

    if not query_tags:
        return chunks[:k]

    qset = set(query_tags)
    for c in chunks:
        c_tags = set((c.get("metadata") or {}).get("concepts", "").split("|")) - {""}
        overlap = len(qset & c_tags) / len(qset) if qset else 0.0
        vector_score = 1.0 - float(c.get("distance", 0.0))
        c["combined"] = (1.0 - alpha) * vector_score + alpha * overlap

    return sorted(chunks, key=lambda c: -c["combined"])[:k]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_query_folio.py -v`
Expected: 4 passed.

- [ ] **Step 6: Wire `rerank_by_concepts` into `query_case()`**

In [query.py](../../../query.py), make three surgical edits:

**(a) Add `import folio_tags`** alongside existing imports near the top of the file.

**(b) Tag the query.** After the `_progress("Classifying question...")` line and before `classify_question(...)` (around line 480), add:

```python
    query_tags = folio_tags.tag_text(question) if config.FOLIO_ENABLED else []
    if query_tags:
        logger.info(f"Query tagged with FOLIO concepts: {query_tags}")
```

**(c) Over-fetch in Stage 2 and rerank the combined chunk list.** Locate the existing `top_k=top_k` argument to `index.query_documents(...)` at line 547. When we have query tags, fetch twice as many so the re-ranker has room. Change line 547 from `top_k=top_k,` to:

```python
                top_k=top_k * 2 if query_tags else top_k,
```

And the fallback call at line 560 from `top_k=remaining,` to:

```python
                top_k=remaining * 2 if query_tags else remaining,
```

**(d) Re-rank `all_chunks` after it's assembled at line 580.** Immediately after the line `all_chunks = doc_chunks + unique_desc_hits`, add:

```python
        all_chunks = rerank_by_concepts(all_chunks, query_tags, k=top_k)
```

Also re-rank in the `descriptions_only` branch — change the existing `all_chunks = desc_hits` (line 533) to:

```python
            all_chunks = rerank_by_concepts(desc_hits, query_tags, k=top_k)
```

`rerank_by_concepts` is a no-op when `query_tags` is empty (returns `chunks[:k]`), so the structured-listing branch above line 488 is unaffected.

- [ ] **Step 7: Run all query tests**

Run: `python -m pytest tests/test_query_folio.py -v`
Expected: 4 passed (rerank unit tests). End-to-end wiring is verified by manual smoke test in Task 12 Step 5 and by `format_chunk_for_llm` tests in Task 11.

- [ ] **Step 8: Commit**

```bash
git add query.py tests/test_query_folio.py
git commit -m "Tag query and re-rank retrieval by FOLIO concept overlap"
```

---

## Task 11: Inline concepts in LLM chunk context

**Files:**
- Modify: [query.py](../../../query.py) — extract per-chunk formatting from `format_context()` at lines 431-442 into a new `format_chunk_for_llm()` and inject the `[Concepts: ...]` annotation.
- Modify: `tests/test_query_folio.py`

- [ ] **Step 1: Write failing tests for the chunk-header formatter**

Append to `tests/test_query_folio.py`:

```python
def test_chunk_header_includes_concepts_when_present():
    from query import format_chunk_for_llm
    chunk = {
        "text": "the debtor moves for relief from the stay",
        "metadata": {
            "ecf_number": "ECF No. 42",
            "doc_type": "motion",
            "date_filed": "2024-03-15",
            "description": "Motion for relief",
            "chunk_index": 0,
            "total_chunks": 2,
            "source": "document",
            "concepts": "automatic_stay|adequate_protection",
        },
    }
    out = format_chunk_for_llm(chunk, source_index=1)
    assert "Source 1" in out
    assert "ECF No. 42" in out
    assert "[Concepts: automatic_stay, adequate_protection]" in out
    assert "the debtor moves for relief from the stay" in out


def test_chunk_header_omits_concepts_when_empty():
    from query import format_chunk_for_llm
    chunk = {
        "text": "some chunk",
        "metadata": {
            "ecf_number": "ECF No. 7",
            "doc_type": "order",
            "date_filed": "2024-04-01",
            "description": "Order",
            "chunk_index": 0,
            "total_chunks": 1,
            "source": "document",
            "concepts": "",
        },
    }
    out = format_chunk_for_llm(chunk, source_index=1)
    assert "ECF No. 7" in out
    assert "[Concepts:" not in out


def test_chunk_header_preserves_description_only_tag():
    from query import format_chunk_for_llm
    chunk = {
        "text": "description text",
        "metadata": {
            "ecf_number": "ECF No. 3",
            "doc_type": "notice",
            "date_filed": "2024-05-01",
            "description": "Notice",
            "chunk_index": 0,
            "total_chunks": 1,
            "source": "docket_entry",
            "concepts": "automatic_stay",
        },
    }
    out = format_chunk_for_llm(chunk, source_index=2)
    assert "DESCRIPTION ONLY" in out
    assert "[Concepts: automatic_stay]" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_query_folio.py -v`
Expected: 3 failures with `ImportError: cannot import name 'format_chunk_for_llm'`.

- [ ] **Step 3: Extract `format_chunk_for_llm()` and call it from `format_context()`**

In [query.py](../../../query.py), add a new top-level function just above `format_context()` (around line 415):

```python
def format_chunk_for_llm(chunk: dict, source_index: int) -> str:
    """Render one retrieved chunk as a labeled block for the LLM context.

    Preserves the existing header format and appends `[Concepts: ...]` when
    the chunk has FOLIO concept tags. Empty tags produce no annotation.
    """
    import folio_tags
    meta = chunk["metadata"]
    ecf = meta.get("ecf_number", "Unknown")
    desc = meta.get("description", "")[:200]
    date = meta.get("date_filed", "Unknown date")
    doc_type = meta.get("doc_type", "other")
    chunk_info = f"(chunk {meta.get('chunk_index', 0) + 1}/{meta.get('total_chunks', 1)})"

    source = meta.get("source", "document")
    desc_tag = " (DESCRIPTION ONLY — no document text available)" if source == "docket_entry" else ""

    header = f"[Source {source_index}: {ecf} | {doc_type} | {date} | {desc}{desc_tag} {chunk_info}]"

    concepts_annot = folio_tags.format_for_llm(meta.get("concepts", ""))
    if concepts_annot:
        header = f"{header} {concepts_annot}"

    return f"{header}\n{chunk['text']}"
```

Then replace the existing loop body in `format_context()` (lines 431-444) with:

```python
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(format_chunk_for_llm(chunk, source_index=i))
        meta = chunk["metadata"]
        seen_entries.add(meta.get("entry_number", 0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_query_folio.py -v`
Expected: 7 passed (4 rerank tests + 3 formatter tests).

- [ ] **Step 5: Commit**

```bash
git add query.py tests/test_query_folio.py
git commit -m "Inline FOLIO concept labels into LLM chunk context"
```

---

## Task 12: Update system prompt, README, and todo

**Files:**
- Modify: [system_prompt.txt](../../../system_prompt.txt)
- Modify: [README.md](../../../README.md)
- Modify: [todo.md](../../../todo.md)

- [ ] **Step 1: Add the `[Concepts: ...]` explanation to the system prompt**

Append to `system_prompt.txt`:

```
Some retrieved chunks include a `[Concepts: ...]` annotation in their header line. These are standardized FOLIO legal-concept labels identifying the substantive bankruptcy topics present in the chunk. You may use these labels to inform your reasoning and choose precise terminology. ECF numbers remain the authoritative citation source — always cite ECF numbers when referencing a filing.
```

- [ ] **Step 2: Add setup instructions to README.md**

Find the existing setup section in `README.md` and add a subsection:

```markdown
### FOLIO concept tagging (optional)

The app uses the [FOLIO](https://openlegalstandard.org/) legal ontology to tag indexed chunks with substantive bankruptcy concepts (Automatic Stay, Adequate Protection, etc.). Tags improve retrieval and let the LLM use precise terminology.

To enable, build the local catalog once:

    python scripts/fetch_folio.py

This writes `data/folio/concepts.json` and `data/folio/concepts.npy`. Re-run to refresh.

If the catalog is missing, tagging is automatically a no-op and the app still works. To disable explicitly, set `FOLIO_ENABLED=false` in your `.env`.
```

- [ ] **Step 3: Update todo.md**

In `todo.md`, mark FOLIO integration in-progress (or done after this plan lands) and add follow-up notes:

```markdown
## FOLIO concept tagging

- [x] Spec: docs/superpowers/specs/2026-05-21-folio-concept-tagging-design.md
- [x] Plan: docs/superpowers/plans/2026-05-21-folio-concept-tagging.md
- [ ] Implemented (this plan)

### Follow-ups after first real-case eval
- Calibrate FOLIO_MIN_SIMILARITY against tag quality on 1–2 real cases.
- Calibrate FOLIO_RERANK_ALPHA after qualitative review of re-ranking impact.
- Consider pulling doc-artifact subtree if retrieval improvement is real.
- Add OPINION DocType to classifier.py to better separate orders/opinions/pleadings.
- UI multi-select for concept filtering (deferred).
```

- [ ] **Step 4: Smoke-verify the prompt file**

Run: `grep "Concepts" system_prompt.txt`
Expected: the new line is present.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all pre-existing tests pass + 18 (folio_tags) + 3 (indexer) + 7 (query) FOLIO tests pass.

- [ ] **Step 6: Commit**

```bash
git add system_prompt.txt README.md todo.md
git commit -m "Document FOLIO setup, system prompt note, todo follow-ups"
```

---

## Done When

- All 12 tasks completed and committed.
- `python -m pytest tests/ -v` passes.
- `python scripts/fetch_folio.py` produces a non-empty catalog (manual smoke test).
- Indexing a real case populates `concepts` metadata on chunks.
- Asking a question whose topic matches a concept (e.g., "what motions related to the automatic stay are in the docket?") demonstrably retrieves stay-tagged chunks and the LLM answer references stay terminology.
