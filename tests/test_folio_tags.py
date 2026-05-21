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
