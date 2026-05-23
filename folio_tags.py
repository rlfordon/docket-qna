"""FOLIO concept tagging — embedding-similarity matching of chunks to legal concepts.

The catalog (concepts + embeddings) is built by scripts/fetch_folio.py and
cached under config.FOLIO_CATALOG_DIR. This module reads the cache and
exposes match helpers used by indexer.py and query.py.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a copy of arr with each row L2-normalized to unit length.

    Rows with zero magnitude are left as zero (and contribute zero
    similarity to anything), avoiding a divide-by-zero.
    """
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def tag_embedding(
    vec: np.ndarray,
    concepts: list[Concept],
    embeddings: np.ndarray,
    top_n: int,
    min_sim: float,
) -> list[str]:
    """Return short_names of the top-N concepts with cosine similarity >= min_sim.

    Vectors are L2-normalized inside this function so the threshold and
    ranking are true cosine, regardless of the caller's magnitudes.
    (FLP's `model.encode` returns un-normalized vectors by default.)
    """
    if not concepts or embeddings.size == 0:
        return []

    # Guard against embedding-dim mismatch (e.g. catalog built with one
    # embedding provider, queries embedded with another). Log and no-op
    # rather than crashing the caller.
    if vec.shape[-1] != embeddings.shape[1]:
        logger.error(
            f"FOLIO embedding-dim mismatch: query vec is {vec.shape[-1]}-d "
            f"but catalog is {embeddings.shape[1]}-d. Rebuild catalog with "
            f"scripts/fetch_folio.py to match your current EMBEDDING_PROVIDER."
        )
        return []

    # Normalize so similarity is cosine, not raw dot product
    n_vec = vec / (np.linalg.norm(vec) or 1.0)
    n_embs = _l2_normalize_rows(embeddings)
    sims = n_embs @ n_vec  # (N_concepts,)
    order = np.argsort(-sims)[:top_n]
    return [concepts[int(j)].short_name for j in order if sims[int(j)] >= min_sim]


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
