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
