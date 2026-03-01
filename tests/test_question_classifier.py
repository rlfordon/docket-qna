"""Tests for question intent classification in query.py.

Tests classify_question() routing against a comprehensive set of
realistic bankruptcy case questions.

Run:  python -m pytest tests/test_question_classifier.py -v
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock chromadb before importing query (which imports indexer -> chromadb)
sys.modules.setdefault("chromadb", MagicMock())

from courtlistener import BankruptcyCase, DocketEntry
from query import classify_question


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_case():
    """A minimal BankruptcyCase with entries spanning several months."""
    return BankruptcyCase(
        docket_id=1,
        case_name="In re Test Debtor, Inc.",
        docket_number="1:24-bk-12345",
        court="S.D.N.Y.",
        date_filed="2024-06-01",
        date_terminated=None,
        chapter="11",
        trustee=None,
        assigned_to="Judge Smith",
        entries=[
            DocketEntry(id=1, entry_number=1, description="Voluntary Petition", date_filed="2024-06-01"),
            DocketEntry(id=2, entry_number=2, description="Motion for DIP Financing", date_filed="2024-06-05"),
            DocketEntry(id=3, entry_number=3, description="Order Granting DIP Financing", date_filed="2024-06-10"),
            DocketEntry(id=4, entry_number=10, description="Notice of Hearing", date_filed="2024-11-15"),
            DocketEntry(id=5, entry_number=20, description="Motion to Extend Exclusivity", date_filed="2024-12-01"),
            DocketEntry(id=6, entry_number=25, description="Order Approving Sale", date_filed="2024-12-10"),
            DocketEntry(id=7, entry_number=30, description="Objection to Plan", date_filed="2024-12-15"),
            DocketEntry(id=8, entry_number=35, description="Monthly Operating Report", date_filed="2024-12-20"),
        ],
        total_entry_count=8,
        available_doc_count=8,
    )


# ---------------------------------------------------------------------------
# Test cases: (question, expected_category)
# ---------------------------------------------------------------------------

# --- Doc-type listings ---
TYPE_LISTING_QUESTIONS = [
    ("What motions have been filed?", "type_listing"),
    ("List all motions in this case", "type_listing"),
    ("How many motions have been filed?", "type_listing"),
    ("Show me all objections", "type_listing"),
    ("What orders have been entered?", "type_listing"),
    ("How many claims have been filed?", "type_listing"),
    ("List all notices filed in this case", "type_listing"),
    ("What reports have been filed?", "type_listing"),
    ("Show me every stipulation", "type_listing"),
    ("What declarations have been filed?", "type_listing"),
    ("How many fee applications have been submitted?", "type_listing"),
]

# --- Date-range listings ---
DOCKET_LISTING_QUESTIONS = [
    ("List all filings from the past two weeks", "docket_listing"),
    ("What was filed last month?", "docket_listing"),
    ("Show me docket entries from the past 30 days", "docket_listing"),
    ("What filings were entered last week?", "docket_listing"),
    ("What happened last week?", "docket_listing"),
]

# --- Doc-type + date-range listings ---
TYPE_DATE_LISTING_QUESTIONS = [
    ("What orders were entered last month?", "type_date_listing"),
    ("What motions were filed in the past two weeks?", "type_date_listing"),
    ("How many claims were filed last month?", "type_date_listing"),
]

# --- Keyword listings (no doc-type match) ---
KEYWORD_LISTING_QUESTIONS = [
    ("What hearings have occurred in the case?", "keyword_listing"),
    ("List all hearings", "keyword_listing"),
    ("What hearings have been scheduled?", "keyword_listing"),
    ("How many hearings have been held?", "keyword_listing"),
    ("Show me all deadlines in this case", "keyword_listing"),
    ("What sales have occurred?", "keyword_listing"),
    ("List all adversary proceedings", "keyword_listing"),
    ("List all employee-related filings", "keyword_listing"),
    ("Show me all asset sales", "keyword_listing"),
    ("What are all the parties in this case?", "keyword_listing"),
    ("Any new filings?", "keyword_listing"),
]

# --- Type analysis (doc-type + analytical intent) ---
TYPE_ANALYSIS_QUESTIONS = [
    ("What do the objections to the plan argue?", "type_analysis"),
    ("Summarize the key motions in this case", "type_analysis"),
    ("What does the plan of reorganization propose?", "type_analysis"),
    ("Explain the claims objection arguments", "type_analysis"),
    ("What are the key orders in this case?", "type_analysis"),
    ("Summarize the monthly operating reports", "type_analysis"),
    ("Describe the fee applications filed so far", "type_analysis"),
    ("What arguments do the objections raise?", "type_analysis"),
    ("What were the arguments in the objection to the DIP motion?", "type_analysis"),
    ("What relief does the motion to dismiss seek?", "type_analysis"),
    ("Compare the original plan to the amended plan", "type_analysis"),
    ("What did the trustee report?", "type_analysis"),
]

# --- Pure analytical (no doc-type, no listing) ---
ANALYTICAL_QUESTIONS = [
    ("What are the key terms of the DIP facility?", "analytical"),
    ("Who are the largest creditors?", "analytical"),
    ("What caused the debtor to file for bankruptcy?", "analytical"),
    ("Is there a stalking horse bidder?", "analytical"),
    ("What is the proposed timeline for emergence?", "analytical"),
    ("Who is the debtor in possession?", "analytical"),
    ("What are the critical vendor payments?", "analytical"),
    ("Has the exclusivity period been extended?", "analytical"),
    ("What is the status of the automatic stay?", "analytical"),
    ("Are there any preference actions?", "analytical"),
    ("Who is the creditors committee?", "analytical"),
    ("What executory contracts have been assumed or rejected?", "analytical"),
    ("How much cash does the debtor have on hand?", "analytical"),
    ("What is the total amount of secured debt?", "analytical"),
    ("Is there a section 363 sale process?", "analytical"),
    ("What happened at the 341 meeting of creditors?", "analytical"),
    ("Give me a summary of the case", "analytical"),
    ("What is going on in this case?", "analytical"),
    ("Tell me about the DIP financing", "analytical"),
    ("Has the court approved the DIP facility?", "analytical"),
    ("What are the milestones in the restructuring?", "analytical"),
    ("Why was the case converted?", "analytical"),
    ("Walk me through the timeline of this case", "analytical"),
    ("What is the status of confirmation?", "analytical"),
    ("What are the unsecured creditors owed?", "analytical"),
    ("How much did the professionals request in fees?", "analytical"),
]

# --- Edge cases: doc-type keyword appears but question is really analytical ---
# These are the tricky ones where a doc-type word like "claim", "motion", "plan"
# appears in the question but the user isn't asking about that document type.
EDGE_CASES_SHOULD_BE_ANALYTICAL = [
    ("What is the administrative claims bar date?", "analytical", "claim keyword in compound term"),
    ("What are the debtor's first day motions about?", "analytical", "motion keyword but meta-question"),
    ("Who filed the most motions?", "analytical", "motion keyword but who-question"),
    ("When was the plan confirmed?", "analytical", "plan keyword but when-question"),
    ("Has a disclosure statement been approved?", "analytical", "plan keyword but yes/no question"),
    ("Were any claims disallowed?", "analytical", "claim keyword but yes/no question"),
    ("What is the claims bar date?", "analytical", "claim keyword in compound term"),
    ("Did anyone object to the sale?", "analytical", "objection keyword but yes/no question"),
    ("When is the next hearing?", "analytical", "hearing keyword but when-question"),
    ("Summarize all filings", "analytical", "listing + analytical, analytical should win"),
]

# --- Edge cases: keyword listing that currently misses ---
EDGE_CASES_SHOULD_BE_KEYWORD_LISTING = [
    ("What settlements have been reached?", "keyword_listing", "implicit listing via 'have been reached'"),
    ("What professionals have been retained?", "keyword_listing", "implicit listing via 'have been retained'"),
    ("What liens have been asserted?", "keyword_listing", "implicit listing via 'have been asserted'"),
]


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

ALL_BASIC_QUESTIONS = (
    TYPE_LISTING_QUESTIONS
    + DOCKET_LISTING_QUESTIONS
    + TYPE_DATE_LISTING_QUESTIONS
    + KEYWORD_LISTING_QUESTIONS
    + TYPE_ANALYSIS_QUESTIONS
    + ANALYTICAL_QUESTIONS
)


@pytest.mark.parametrize("question,expected", ALL_BASIC_QUESTIONS)
def test_basic_classification(question, expected, sample_case):
    intent = classify_question(question, sample_case)
    assert intent.category == expected, (
        f"'{question}' classified as '{intent.category}', expected '{expected}' "
        f"(doc_type={intent.doc_type}, keywords={intent.keywords})"
    )


@pytest.mark.parametrize("question,expected,note", EDGE_CASES_SHOULD_BE_ANALYTICAL)
def test_edge_cases_analytical(question, expected, note, sample_case):
    """Doc-type keywords that appear incidentally — should still route to analytical."""
    intent = classify_question(question, sample_case)
    assert intent.category == expected, (
        f"'{question}' classified as '{intent.category}', expected '{expected}' "
        f"({note}; doc_type={intent.doc_type})"
    )


@pytest.mark.parametrize("question,expected,note", EDGE_CASES_SHOULD_BE_KEYWORD_LISTING)
def test_edge_cases_keyword_listing(question, expected, note, sample_case):
    """Implicit listing questions that should route to keyword_listing."""
    intent = classify_question(question, sample_case)
    assert intent.category == expected, (
        f"'{question}' classified as '{intent.category}', expected '{expected}' "
        f"({note}; keywords={intent.keywords})"
    )


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def test_type_listing_has_doc_type(sample_case):
    """type_listing results should always have a doc_type set."""
    for question, _ in TYPE_LISTING_QUESTIONS:
        intent = classify_question(question, sample_case)
        if intent.category == "type_listing":
            assert intent.doc_type is not None, f"'{question}' missing doc_type"


def test_docket_listing_has_date_range(sample_case):
    """docket_listing results should always have a date_range set."""
    for question, _ in DOCKET_LISTING_QUESTIONS:
        intent = classify_question(question, sample_case)
        if intent.category == "docket_listing":
            assert intent.date_range is not None, f"'{question}' missing date_range"


def test_keyword_listing_has_keywords(sample_case):
    """keyword_listing results should always have non-empty keywords."""
    for question, _ in KEYWORD_LISTING_QUESTIONS:
        intent = classify_question(question, sample_case)
        if intent.category == "keyword_listing":
            assert intent.keywords, f"'{question}' has empty keywords"
