"""CourtListener API client for fetching bankruptcy case data from RECAP Archive.

API docs: https://www.courtlistener.com/help/api/rest/
PACER APIs: https://www.courtlistener.com/help/api/rest/v3/pacer/
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


@dataclass
class RecapDocument:
    """A document from the RECAP Archive."""

    id: int
    docket_entry_id: int
    description: str
    date_filed: Optional[str]
    ecf_number: Optional[str]
    plain_text: str
    is_available: bool
    pacer_doc_id: Optional[str]
    page_count: Optional[int] = None
    filepath_local: Optional[str] = None


@dataclass
class DocketEntry:
    """A single entry on a bankruptcy docket."""

    id: int
    entry_number: Optional[int]
    description: str
    date_filed: Optional[str]
    documents: list[RecapDocument] = field(default_factory=list)


@dataclass
class BankruptcyCase:
    """Represents a bankruptcy case with its docket entries and documents."""

    docket_id: int
    case_name: str
    docket_number: str
    court: str
    date_filed: Optional[str]
    date_terminated: Optional[str]
    chapter: Optional[str]
    trustee: Optional[str]
    assigned_to: Optional[str]
    entries: list[DocketEntry] = field(default_factory=list)
    total_entry_count: int = 0
    available_doc_count: int = 0

    @property
    def coverage_pct(self) -> float:
        if self.total_entry_count == 0:
            return 0.0
        return (self.available_doc_count / self.total_entry_count) * 100


class CourtListenerClient:
    """Client for interacting with the CourtListener REST API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.CL_HEADERS)
        self._request_count = 0
        self._window_start = time.time()

    def _throttle(self):
        """Simple rate limiting: pause if approaching limit."""
        self._request_count += 1
        elapsed = time.time() - self._window_start
        if elapsed > 3600:
            self._request_count = 1
            self._window_start = time.time()
        elif self._request_count >= config.CL_RATE_LIMIT - 100:
            sleep_time = 3600 - elapsed + 1
            logger.warning(f"Approaching rate limit, sleeping {sleep_time:.0f}s")
            time.sleep(sleep_time)
            self._request_count = 0
            self._window_start = time.time()

    def _get(self, url: str, params: dict = None) -> dict:
        """Make a GET request with throttling and error handling."""
        self._throttle()
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _get_paginated(self, url: str, params: dict = None) -> list[dict]:
        """Fetch all pages of a paginated API response."""
        results = []
        params = params or {}
        while url:
            data = self._get(url, params)
            results.extend(data.get("results", []))
            url = data.get("next")
            params = {}  # next URL includes params already
            logger.info(f"  Fetched {len(results)} results so far...")
        return results

    # --- Docket Methods ---

    def get_docket(self, docket_id: int) -> dict:
        """Fetch docket metadata by CourtListener docket ID."""
        url = f"{config.CL_BASE_URL}/dockets/{docket_id}/"
        return self._get(url)

    def search_docket(self, court: str, docket_number: str) -> Optional[dict]:
        """Search for a docket by court and docket number.

        Args:
            court: Court identifier (e.g., "njb" for NJ Bankruptcy)
            docket_number: Case number (e.g., "23-14853")
        """
        url = f"{config.CL_BASE_URL}/dockets/"
        params = {
            "court__id": court,
            "docket_number": docket_number,
        }
        data = self._get(url, params)
        results = data.get("results", [])
        return results[0] if results else None

    def get_docket_entries(self, docket_id: int) -> list[dict]:
        """Fetch all docket entries for a case.

        Uses field selection to minimize payload size.
        """
        url = f"{config.CL_BASE_URL}/docket-entries/"
        params = {
            "docket": docket_id,
            "order_by": "entry_number",
        }
        return self._get_paginated(url, params)

    def get_recap_documents(self, docket_id: int, with_text: bool = True) -> list[dict]:
        """Fetch all RECAP documents for a case, optionally including full text.

        Args:
            docket_id: CourtListener docket ID
            with_text: If True, include plain_text field (larger payloads)
        """
        url = f"{config.CL_BASE_URL}/recap-documents/"
        params = {
            "docket_entry__docket": docket_id,
            "is_available": True,
            "order_by": "docket_entry__entry_number",
        }
        if not with_text:
            params["fields"] = (
                "id,docket_entry,description,date_created,"
                "pacer_doc_id,is_available,page_count,filepath_local"
            )
        return self._get_paginated(url, params)

    # --- Case Assembly ---

    def load_case(self, docket_id: int, fetch_text: bool = True) -> BankruptcyCase:
        """Load a complete bankruptcy case with entries and available documents.

        This is the main entry point for loading a case. It:
        1. Fetches docket metadata
        2. Fetches all docket entries
        3. Fetches all available RECAP documents (with text)
        4. Assembles into a BankruptcyCase object

        Args:
            docket_id: CourtListener docket ID
            fetch_text: Whether to fetch full document text (set False for preview)
        """
        logger.info(f"Loading case {docket_id}...")

        # 1. Docket metadata
        logger.info("  Fetching docket metadata...")
        docket = self.get_docket(docket_id)

        # Extract bankruptcy-specific info if available
        bk_info = docket.get("bankruptcy_information") or {}

        case = BankruptcyCase(
            docket_id=docket_id,
            case_name=docket.get("case_name", "Unknown"),
            docket_number=docket.get("docket_number", "Unknown"),
            court=docket.get("court_id", "Unknown"),
            date_filed=docket.get("date_filed"),
            date_terminated=docket.get("date_terminated"),
            chapter=bk_info.get("chapter"),
            trustee=bk_info.get("trustee_str"),
            assigned_to=docket.get("assigned_to_str"),
        )

        # 2. Docket entries
        logger.info("  Fetching docket entries...")
        raw_entries = self.get_docket_entries(docket_id)
        case.total_entry_count = len(raw_entries)
        logger.info(f"  Found {case.total_entry_count} docket entries")

        # Build entry lookup
        entry_map = {}
        for raw in raw_entries:
            entry = DocketEntry(
                id=raw["id"],
                entry_number=raw.get("entry_number"),
                description=raw.get("description", ""),
                date_filed=raw.get("date_filed"),
            )
            case.entries.append(entry)
            entry_map[entry.id] = entry

        # 3. Available documents
        logger.info("  Fetching available RECAP documents...")
        raw_docs = self.get_recap_documents(docket_id, with_text=fetch_text)
        logger.info(f"  Found {len(raw_docs)} available documents")

        for raw_doc in raw_docs:
            # Link document to its docket entry
            de_url = raw_doc.get("docket_entry", "")
            # Extract docket_entry ID from URL or nested object
            de_id = None
            if isinstance(de_url, str) and "/" in de_url:
                try:
                    de_id = int(de_url.rstrip("/").split("/")[-1])
                except (ValueError, IndexError):
                    pass
            elif isinstance(de_url, int):
                de_id = de_url

            doc = RecapDocument(
                id=raw_doc["id"],
                docket_entry_id=de_id or 0,
                description=raw_doc.get("description", ""),
                date_filed=raw_doc.get("date_created", ""),
                ecf_number=None,  # Will derive from entry_number
                plain_text=raw_doc.get("plain_text", "") or "",
                is_available=raw_doc.get("is_available", False),
                pacer_doc_id=raw_doc.get("pacer_doc_id"),
                page_count=raw_doc.get("page_count"),
            )

            # Attach to entry and set ECF number
            if de_id and de_id in entry_map:
                entry = entry_map[de_id]
                entry.documents.append(doc)
                if entry.entry_number:
                    doc.ecf_number = f"ECF No. {entry.entry_number}"
                # Use entry description if doc description is empty
                if not doc.description and entry.description:
                    doc.description = entry.description
                # Use entry date if doc date is missing
                if not doc.date_filed and entry.date_filed:
                    doc.date_filed = entry.date_filed

        case.available_doc_count = len(raw_docs)
        logger.info(
            f"  Case loaded: {case.case_name} | "
            f"{case.available_doc_count}/{case.total_entry_count} docs available "
            f"({case.coverage_pct:.0f}%)"
        )
        return case

    # --- PACER Purchase (optional) ---

    def purchase_docket(self, docket_id: int) -> dict:
        """Request purchase of a full docket sheet from PACER via CourtListener.

        Requires PACER credentials. The purchase is async — returns a request ID
        that can be polled for completion.
        """
        url = f"{config.CL_V4_BASE_URL}/recap/"
        payload = {
            "docket": docket_id,
            "request_type": 1,  # Docket
            "pacer_username": config.PACER_USERNAME,
            "pacer_password": config.PACER_PASSWORD,
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def purchase_document(self, recap_document_id: int) -> dict:
        """Request purchase of a specific document from PACER via CourtListener."""
        url = f"{config.CL_V4_BASE_URL}/recap/"
        payload = {
            "recap_document": recap_document_id,
            "request_type": 2,  # PDF
            "pacer_username": config.PACER_USERNAME,
            "pacer_password": config.PACER_PASSWORD,
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# --- URL Parsing Helpers ---

def parse_courtlistener_url(url: str) -> Optional[int]:
    """Extract docket ID from a CourtListener URL.

    Handles URLs like:
      https://www.courtlistener.com/docket/67531068/ftx-trading-ltd/
      https://www.courtlistener.com/docket/67531068/
    """
    import re

    match = re.search(r"courtlistener\.com/docket/(\d+)", url)
    if match:
        return int(match.group(1))
    return None
