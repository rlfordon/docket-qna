"""Bankruptcy document type classification.

Uses regex patterns on docket entry descriptions to categorize filings.
Covers ~80% of standard bankruptcy filing types. Ambiguous entries
fall through to "other".
"""

import re
from enum import Enum


class DocType(str, Enum):
    MOTION = "motion"
    OBJECTION = "objection"
    ORDER = "order"
    PLAN = "plan"
    CLAIM = "claim"
    FEE_APPLICATION = "fee_application"
    SCHEDULE = "schedule"
    REPORT = "report"
    NOTICE = "notice"
    STIPULATION = "stipulation"
    DECLARATION = "declaration"
    OTHER = "other"

    @property
    def label(self) -> str:
        """Human-readable label for display."""
        labels = {
            "motion": "Motion",
            "objection": "Objection / Response",
            "order": "Order",
            "plan": "Plan / Disclosure Statement",
            "claim": "Proof of Claim",
            "fee_application": "Fee Application",
            "schedule": "Schedules & Statements",
            "report": "Report (MOR, Status, etc.)",
            "notice": "Notice / Certificate",
            "stipulation": "Stipulation / Agreement",
            "declaration": "Declaration / Affidavit",
            "other": "Other",
        }
        return labels.get(self.value, self.value.title())


# Patterns are checked in order — first match wins.
# More specific patterns should come before general ones.
CLASSIFICATION_PATTERNS: list[tuple[DocType, re.Pattern]] = [
    # Plans and disclosure statements (check before motions since
    # "motion to approve disclosure statement" should be a motion)
    (
        DocType.PLAN,
        re.compile(
            r"(?i)\b(plan of reorganization|plan of liquidation|"
            r"amended plan|modified plan|"
            r"disclosure statement|plan supplement|"
            r"chapter 11 plan|joint plan|"
            r"plan and disclosure statement)\b"
        ),
    ),
    # Fee applications (check before general motions)
    (
        DocType.FEE_APPLICATION,
        re.compile(
            r"(?i)\b(application.*(?:compensation|fees|reimbursement)|"
            r"fee application|interim fee|final fee|"
            r"application.*(?:allowance|payment).*(?:fees|compensation)|"
            r"(?:first|second|third|fourth|fifth|final|interim).*"
            r"application.*(?:compensation|fees))\b"
        ),
    ),
    # Proofs of claim
    (
        DocType.CLAIM,
        re.compile(
            r"(?i)\b(proof of claim|amended.*proof.*claim|"
            r"claim no\.|claim #|"
            r"notice of (?:filed|transferred) claim|"
            r"objection to claim)\b"
        ),
    ),
    # Objections and responses (check before motions)
    (
        DocType.OBJECTION,
        re.compile(
            r"(?i)\b(objection(?:s)?(?:\s+to)?|limited objection|"
            r"response.*(?:to|in opposition)|"
            r"opposition to|reply (?:to|in support)|"
            r"sur-?reply|joinder.*(?:objection|opposition))\b"
        ),
    ),
    # Orders
    (
        DocType.ORDER,
        re.compile(
            r"(?i)\b(order (?:granting|denying|approving|authorizing|"
            r"confirming|converting|dismissing|sustaining|overruling|"
            r"extending|modifying|establishing|setting|directing|"
            r"lifting|vacating|reopening)|"
            r"amended order|final order|interim order|"
            r"scheduling order|bar date order|"
            r"(?:findings of fact|conclusions of law).*order)\b"
        ),
    ),
    # Stipulations and agreements
    (
        DocType.STIPULATION,
        re.compile(
            r"(?i)\b(stipulat(?:ion|ed)|(?:settlement|restructuring|"
            r"plan support|lock-?up) agreement|"
            r"agreed order|consent order)\b"
        ),
    ),
    # Declarations and affidavits
    (
        DocType.DECLARATION,
        re.compile(
            r"(?i)\b(declaration|affidavit|verification|"
            r"certification|sworn statement)\b"
        ),
    ),
    # Motions (broad — catches many filing types)
    (
        DocType.MOTION,
        re.compile(
            r"(?i)\b(motion|emergency motion|omnibus motion|"
            r"application (?:to|for)|"
            r"(?:ex parte|urgent) (?:motion|application)|"
            r"motion (?:to|for)|"
            r"memorandum (?:of law |)in support)\b"
        ),
    ),
    # Schedules, statements, and lists
    (
        DocType.SCHEDULE,
        re.compile(
            r"(?i)\b(schedule[s]?\b|statement of financial affairs|"
            r"(?:master|creditor|equity) (?:mailing )?list|"
            r"matrix|means test|"
            r"summary of (?:schedules|assets))\b"
        ),
    ),
    # Reports (Monthly Operating Reports, status reports, etc.)
    (
        DocType.REPORT,
        re.compile(
            r"(?i)\b(monthly operating report|(?:MOR\b)|"
            r"(?:status|progress|quarterly|final) report|"
            r"report of (?:no distribution|sale)|"
            r"(?:trustee|examiner|fee)(?:'s|s)? report|"
            r"post-?confirmation report)\b"
        ),
    ),
    # Notices and certificates (broadest — catch-all for procedural filings)
    (
        DocType.NOTICE,
        re.compile(
            r"(?i)\b(notice of|certificate of (?:service|no objection|counsel)|"
            r"agenda|hearing notice|"
            r"ballot|(?:proof|certificate) of (?:service|mailing|publication))\b"
        ),
    ),
]


def classify_document(description: str) -> DocType:
    """Classify a docket entry by its description.

    Args:
        description: The docket entry description text.

    Returns:
        The best-matching DocType, or DocType.OTHER if no pattern matches.
    """
    if not description:
        return DocType.OTHER

    for doc_type, pattern in CLASSIFICATION_PATTERNS:
        if pattern.search(description):
            return doc_type

    return DocType.OTHER


def classify_entries(entries: list[dict]) -> dict[str, list[dict]]:
    """Classify a list of docket entries and group by type.

    Args:
        entries: List of docket entry dicts with "description" key.

    Returns:
        Dict mapping DocType values to lists of entries.
    """
    grouped: dict[str, list[dict]] = {dt.value: [] for dt in DocType}

    for entry in entries:
        doc_type = classify_document(entry.get("description", ""))
        entry["doc_type"] = doc_type.value
        grouped[doc_type.value].append(entry)

    return grouped


def get_type_summary(entries: list[dict]) -> dict[str, int]:
    """Get a count of each document type in a list of entries.

    Returns:
        Dict mapping DocType labels to counts, sorted by count descending.
    """
    counts: dict[str, int] = {}
    for entry in entries:
        doc_type = classify_document(entry.get("description", ""))
        label = doc_type.label
        counts[label] = counts.get(label, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
