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
import folio_tags
import indexer  # for embed_texts
from folio_tags import Concept, slugify

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "https://folio.openlegalstandard.org"
MAX_DEPTH = 6
RETRY_DELAYS = [1, 3, 9]  # seconds


def _normalize_iri(iri: str) -> str:
    """Return iri as a full URL. The root IRI in config is a slug; the
    API returns full URLs in parent_class_of / sub_class_of lists."""
    if iri.startswith("http"):
        return iri
    return f"{API_BASE}/{iri}"


def _api_get(url: str) -> dict:
    """GET a full URL with simple retry/backoff."""
    last_err: Exception | None = None
    for delay in [0] + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            logger.warning(f"GET {url} failed ({e}); retrying in {delay}s")
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")


def _fetch_node(iri: str) -> dict:
    """Fetch a single concept node by (possibly bare) IRI."""
    return _api_get(_normalize_iri(iri))


def _normalize(node: dict, depth: int) -> Concept:
    """Convert an API node payload into a Concept dataclass.

    The FOLIO API uses these field names:
      iri                 — full URL identifier
      label               — display name
      preferred_label     — sometimes set (often null)
      alternative_labels  — synonyms
      definition          — text
      parent_class_of     — list of full-URL IRIs (this node's children)
      sub_class_of        — list of full-URL IRIs (this node's parents)
    """
    label = node.get("label") or node.get("preferred_label") or ""
    parents = list(node.get("sub_class_of") or [])
    children = list(node.get("parent_class_of") or [])
    return Concept(
        iri=node.get("iri") or "",
        short_name=slugify(label),
        label=label,
        alt_labels=list(node.get("alternative_labels") or []),
        definition=node.get("definition") or "",
        embed_text="",  # filled below
        parent_iri=parents[0] if parents else "",
        children_iris=children,
        depth=depth,
    )


def _build_embed_text(c: Concept) -> str:
    parts = [c.label]
    parts.extend(c.alt_labels)
    if c.definition:
        parts.append(c.definition)
    return ". ".join(p for p in parts if p)


def traverse(root_iri: str) -> list[Concept]:
    """BFS from root_iri down to MAX_DEPTH, deduplicating by IRI.

    Children come inline in each node's `parent_class_of` field, so no
    separate "list children" API call is needed.
    """
    seen: dict[str, Concept] = {}
    queue: list[tuple[str, int]] = [(_normalize_iri(root_iri), 0)]

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
            for child_iri in c.children_iris:
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

    # Invalidate the in-memory cache so any long-running process that
    # imports folio_tags picks up the fresh catalog on the next call.
    folio_tags.clear_cache()
    logger.info("Cleared in-memory FOLIO catalog cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
