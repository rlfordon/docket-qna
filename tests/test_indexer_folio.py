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
                        docket_entry_id=1,
                        ecf_number="1",
                        description="Motion for relief from automatic stay",
                        date_filed="2024-06-05",
                        plain_text=(
                            "The debtor seeks to lift the automatic stay so that "
                            "the creditor may foreclose on the collateral. The motion "
                            "argues that adequate protection has not been provided."
                        ),
                        is_available=True,
                        pacer_doc_id="pdoc_1001",
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
