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
