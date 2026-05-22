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
