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
