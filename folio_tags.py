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
