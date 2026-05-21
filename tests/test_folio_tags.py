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


from unittest.mock import patch
from folio_tags import tag_text


def test_tag_text_embeds_then_tags(folio_catalog_dir, monkeypatch):
    import config
    monkeypatch.setattr(config, "FOLIO_CATALOG_DIR", folio_catalog_dir)
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
