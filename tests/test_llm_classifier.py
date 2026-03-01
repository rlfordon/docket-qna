"""Compare LLM classifier vs regex classifier on the full question suite.

This is NOT a unit test — it makes real API calls to Anthropic (Haiku).
Run manually:  python tests/test_llm_classifier.py

Cost: ~84 calls x ~400 tokens ≈ 33K tokens ≈ $0.03 with Haiku.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path & mock chromadb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("chromadb", MagicMock())

from courtlistener import BankruptcyCase, DocketEntry
from query import classify_question, classify_question_llm

# Same question suite as test_question_classifier.py
from test_question_classifier import (
    TYPE_LISTING_QUESTIONS,
    DOCKET_LISTING_QUESTIONS,
    TYPE_DATE_LISTING_QUESTIONS,
    KEYWORD_LISTING_QUESTIONS,
    TYPE_ANALYSIS_QUESTIONS,
    ANALYTICAL_QUESTIONS,
    EDGE_CASES_SHOULD_BE_ANALYTICAL,
    EDGE_CASES_SHOULD_BE_KEYWORD_LISTING,
)


def make_case():
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


def main():
    case = make_case()

    # Build full question list with expected categories
    all_questions = []
    for q, expected in (
        TYPE_LISTING_QUESTIONS
        + DOCKET_LISTING_QUESTIONS
        + TYPE_DATE_LISTING_QUESTIONS
        + KEYWORD_LISTING_QUESTIONS
        + TYPE_ANALYSIS_QUESTIONS
        + ANALYTICAL_QUESTIONS
    ):
        all_questions.append((q, expected, ""))

    for q, expected, note in EDGE_CASES_SHOULD_BE_ANALYTICAL:
        all_questions.append((q, expected, note))

    for q, expected, note in EDGE_CASES_SHOULD_BE_KEYWORD_LISTING:
        all_questions.append((q, expected, note))

    print(f"Running {len(all_questions)} questions through both classifiers...\n")
    print(f"{'#':<4} {'REGEX':<7} {'LLM':<7} {'EXPECTED':<20} {'REGEX GOT':<20} {'LLM GOT':<20} QUESTION")
    print("=" * 130)

    regex_pass = 0
    regex_fail = 0
    llm_pass = 0
    llm_fail = 0
    llm_errors = []
    total_time = 0

    for i, (q, expected, note) in enumerate(all_questions, 1):
        # Regex classifier
        regex_intent = classify_question(q, case)
        regex_ok = regex_intent.category == expected

        # LLM classifier
        t0 = time.time()
        llm_intent = classify_question_llm(q)
        elapsed = time.time() - t0
        total_time += elapsed
        llm_ok = llm_intent.category == expected

        if regex_ok:
            regex_pass += 1
        else:
            regex_fail += 1

        if llm_ok:
            llm_pass += 1
        else:
            llm_fail += 1
            llm_errors.append((q, expected, llm_intent, note))

        r_status = "PASS" if regex_ok else "FAIL"
        l_status = "PASS" if llm_ok else "FAIL"
        print(f"{i:<4} {r_status:<7} {l_status:<7} {expected:<20} {regex_intent.category:<20} {llm_intent.category:<20} {q}")

    print(f"\n{'=' * 130}")
    print(f"REGEX:  {regex_pass}/{len(all_questions)} passed")
    print(f"LLM:    {llm_pass}/{len(all_questions)} passed")
    print(f"LLM total time: {total_time:.1f}s  ({total_time/len(all_questions)*1000:.0f}ms avg per question)")

    if llm_errors:
        print(f"\nLLM FAILURES ({len(llm_errors)}):")
        for q, expected, intent, note in llm_errors:
            print(f"  Q: {q}")
            print(f"     expected={expected}  got={intent.category}  doc_type={intent.doc_type}  kw={intent.keywords}")
            if note:
                print(f"     note: {note}")


if __name__ == "__main__":
    main()
