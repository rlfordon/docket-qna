"""Shared pytest fixtures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def folio_catalog_dir():
    """Path to the test FOLIO catalog fixture."""
    return Path(__file__).parent / "fixtures" / "folio"
