"""CourtListener API client for fetching bankruptcy case data from RECAP Archive.

API docs: https://www.courtlistener.com/help/api/rest/
PACER APIs: https://www.courtlistener.com/help/api/rest/v3/pacer/
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Callable

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
    last_updated: Optional[str] = None

    @property
    def coverage_pct(self) -> float:
        if self.total_entry_count == 0:
            return 0.0
        return (self.available_doc_count / self.total_entry_count) * 100


DESCRIPTION_QUALITY_THRESHOLD = 50  # chars — descriptions shorter are "stubs"


def get_poor_description_date_range(case: BankruptcyCase) -> tuple[Optional[str], Optional[str]]:
    """Get the date range spanning entries with stub or empty descriptions.

    Returns:
        (date_start, date_end) as ISO date strings, or (None, None) if no
        poor-quality entries have dates (falls back to full docket purchase).
    """
    dates = []
    for entry in case.entries:
        desc = (entry.description or "").strip()
        if not desc or len(desc) < DESCRIPTION_QUALITY_THRESHOLD:
            if entry.date_filed:
                dates.append(entry.date_filed)
    if not dates:
        return None, None
    return min(dates), max(dates)


def description_quality_stats(case: BankruptcyCase) -> dict:
    """Analyze the quality of docket entry descriptions.

    Returns:
        Dict with keys: total, detailed, stub, empty, detailed_pct
    """
    total = len(case.entries)
    empty = 0
    stub = 0
    detailed = 0
    for entry in case.entries:
        desc = (entry.description or "").strip()
        if not desc:
            empty += 1
        elif len(desc) < DESCRIPTION_QUALITY_THRESHOLD:
            stub += 1
        else:
            detailed += 1
    return {
        "total": total,
        "detailed": detailed,
        "stub": stub,
        "empty": empty,
        "detailed_pct": (detailed / total * 100) if total > 0 else 0.0,
    }


def get_purchasable_for_entries(
    case: BankruptcyCase,
    entry_numbers: list[int],
    client: "CourtListenerClient" = None,
) -> list[dict]:
    """Find purchasable documents for the given ECF entry numbers.

    For entries that have no local RecapDocument records (the typical case
    for description-only entries), queries the CourtListener API to find
    document records that can be purchased.

    Args:
        case: The loaded BankruptcyCase.
        entry_numbers: List of ECF entry numbers to check.
        client: CourtListenerClient for API lookups. Required when entries
            lack local document records.

    Returns:
        List of dicts with keys: entry_number, description, date_filed,
        recap_document_id, purchasable.
    """
    entry_by_number = {
        e.entry_number: e for e in case.entries if e.entry_number is not None
    }

    results = []
    for num in entry_numbers:
        entry = entry_by_number.get(num)
        if not entry:
            continue

        info = {
            "entry_number": num,
            "description": entry.description or "",
            "date_filed": entry.date_filed or "",
            "recap_document_id": None,
            "purchasable": False,
        }

        if entry.documents:
            # Prefer a doc that is_available but has no text (purchasable gap)
            target = None
            for doc in entry.documents:
                if doc.is_available and not doc.plain_text:
                    target = doc
                    break
            if target is None:
                target = entry.documents[0]

            info["recap_document_id"] = target.id
            info["purchasable"] = True
        elif client:
            # No local documents — query the API for recap document records
            # on this docket entry (including unavailable ones)
            try:
                url = f"{config.CL_BASE_URL}/recap-documents/"
                params = {
                    "docket_entry__docket": case.docket_id,
                    "docket_entry": entry.id,
                }
                data = client._get(url, params)
                api_docs = data.get("results", [])
                if api_docs:
                    # Use the first document record found
                    info["recap_document_id"] = api_docs[0]["id"]
                    info["purchasable"] = True
                    logger.info(
                        f"Found recap document {api_docs[0]['id']} "
                        f"for entry {num} via API"
                    )
                else:
                    logger.warning(
                        f"No recap document records found for ECF No. {num} "
                        f"(docket_entry={entry.id})"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to look up recap documents for ECF No. {num}: {e}"
                )

        results.append(info)

    return results


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

    def _get(self, url: str, params: dict = None, retries: int = 3) -> dict:
        """Make a GET request with throttling, retry on 5xx, and error handling."""
        for attempt in range(retries):
            self._throttle()
            resp = self.session.get(url, params=params)
            if resp.status_code >= 500 and attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(f"  Server error {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
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
        if isinstance(bk_info, str):
            bk_info = {}

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

        # Build entry lookups (by ID and by entry_number)
        entry_map = {}
        entry_by_number = {}
        for raw in raw_entries:
            entry = DocketEntry(
                id=raw["id"],
                entry_number=raw.get("entry_number"),
                description=raw.get("description", ""),
                date_filed=raw.get("date_filed"),
            )
            case.entries.append(entry)
            entry_map[entry.id] = entry
            if entry.entry_number is not None:
                entry_by_number[entry.entry_number] = entry

        # 3. Available documents
        logger.info("  Fetching available RECAP documents...")
        raw_docs = self.get_recap_documents(docket_id, with_text=fetch_text)
        logger.info(f"  Found {len(raw_docs)} available documents")

        for raw_doc in raw_docs:
            # Link document to its docket entry
            de_url = raw_doc.get("docket_entry", "")
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

            # Attach to entry — try by ID first, then by document_number
            entry = None
            if de_id and de_id in entry_map:
                entry = entry_map[de_id]
            else:
                doc_num = raw_doc.get("document_number")
                if doc_num is not None:
                    try:
                        entry = entry_by_number.get(int(doc_num))
                    except (ValueError, TypeError):
                        pass

            if entry:
                entry.documents.append(doc)
                doc.docket_entry_id = entry.id
                if entry.entry_number:
                    doc.ecf_number = f"ECF No. {entry.entry_number}"
                if not doc.description and entry.description:
                    doc.description = entry.description
                if not doc.date_filed and entry.date_filed:
                    doc.date_filed = entry.date_filed

        case.available_doc_count = len(raw_docs)
        case.last_updated = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"  Case loaded: {case.case_name} | "
            f"{case.available_doc_count}/{case.total_entry_count} docs available "
            f"({case.coverage_pct:.0f}%)"
        )
        return case

    # --- PACER Purchase (optional) ---

    def purchase_docket(
        self,
        case: BankruptcyCase,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> dict:
        """Request purchase of a docket sheet from PACER via CourtListener.

        Requires PACER credentials. The purchase is async — returns a request ID
        that can be polled for completion.

        Args:
            case: The BankruptcyCase to purchase the docket for.
            date_start: Optional start date (inclusive) to scope the purchase.
            date_end: Optional end date (inclusive) to scope the purchase.
        """
        url = f"{config.CL_V4_BASE_URL}/recap-fetch/"
        payload = {
            "request_type": 1,  # Docket
            "docket_number": case.docket_number,
            "court": case.court,
            "pacer_username": config.PACER_USERNAME,
            "pacer_password": config.PACER_PASSWORD,
        }
        if date_start:
            payload["de_date_start"] = date_start
        if date_end:
            payload["de_date_end"] = date_end
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def purchase_document(self, recap_document_id: int) -> dict:
        """Request purchase of a specific document from PACER via CourtListener."""
        url = f"{config.CL_V4_BASE_URL}/recap-fetch/"
        payload = {
            "recap_document": recap_document_id,
            "request_type": 2,  # PDF
            "pacer_username": config.PACER_USERNAME,
            "pacer_password": config.PACER_PASSWORD,
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def poll_purchase_status(
        self,
        request_id: int,
        poll_interval: int = None,
        timeout: int = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Poll a recap-fetch request until completion or timeout.

        Args:
            request_id: The ID from a purchase_docket/purchase_document response.
            poll_interval: Seconds between polls (default from config).
            timeout: Max seconds to wait (default from config).
            progress_callback: Optional callback(status: int, elapsed: float).

        Returns:
            The final response dict on success.

        Raises:
            TimeoutError: If polling exceeds timeout.
            RuntimeError: If the request fails (status 3/6/7).
        """
        poll_interval = poll_interval or config.PACER_POLL_INTERVAL
        timeout = timeout or config.PACER_POLL_TIMEOUT
        url = f"{config.CL_V4_BASE_URL}/recap-fetch/{request_id}/"
        start = time.time()

        # Status codes from CourtListener ProcessingQueue model
        terminal_failures = {3, 6, 7}
        status_labels = {
            1: "Enqueued",
            2: "Complete",
            3: "Failed",
            4: "In Progress",
            5: "Queued for Retry",
            6: "Invalid Content",
            7: "Needs Info",
        }

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"PACER purchase polling timed out after {timeout}s"
                )

            data = self._get(url)
            status = data.get("status")

            if progress_callback:
                progress_callback(status, elapsed)

            if status == 2:
                logger.info("PACER purchase completed successfully")
                return data

            if status in terminal_failures:
                label = status_labels.get(status, f"Unknown status {status}")
                msg = data.get("message", "")
                raise RuntimeError(
                    f"PACER purchase failed: {label}. {msg}".strip()
                )

            logger.info(
                f"  Purchase status: {status_labels.get(status, status)} "
                f"({elapsed:.0f}s elapsed)"
            )
            time.sleep(poll_interval)

    def refresh_docket_entries(self, case: BankruptcyCase) -> int:
        """Re-fetch docket entries from API and merge with existing case data.

        Preserves existing document lists on entries. Only upgrades descriptions
        when the new description is longer than the existing one.

        Args:
            case: The BankruptcyCase to refresh.

        Returns:
            Number of entries whose descriptions were updated.
        """
        logger.info(f"Refreshing docket entries for case {case.docket_id}...")
        raw_entries = self.get_docket_entries(case.docket_id)
        logger.info(f"  Fetched {len(raw_entries)} entries from API")

        # Build lookup of existing entries by ID
        existing_map = {entry.id: entry for entry in case.entries}

        updated_count = 0
        new_entries = []

        for raw in raw_entries:
            entry_id = raw["id"]
            new_desc = raw.get("description", "")

            if entry_id in existing_map:
                existing = existing_map[entry_id]
                # Only upgrade description if new one is longer
                if len(new_desc) > len(existing.description or ""):
                    existing.description = new_desc
                    updated_count += 1
                # Update other fields that may have changed
                if raw.get("date_filed"):
                    existing.date_filed = raw["date_filed"]
                if raw.get("entry_number") is not None:
                    existing.entry_number = raw["entry_number"]
            else:
                # New entry not in our cache
                new_entry = DocketEntry(
                    id=entry_id,
                    entry_number=raw.get("entry_number"),
                    description=new_desc,
                    date_filed=raw.get("date_filed"),
                )
                new_entries.append(new_entry)
                updated_count += 1

        if new_entries:
            case.entries.extend(new_entries)
            logger.info(f"  Added {len(new_entries)} new entries")

        case.total_entry_count = len(case.entries)
        case.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(f"  Updated {updated_count} entry descriptions")
        return updated_count


# --- Case Caching ---

def save_case(case: BankruptcyCase) -> None:
    """Save a BankruptcyCase to disk as JSON for quick reload."""
    path = config.CASES_DIR / f"{case.docket_id}.json"
    path.write_text(json.dumps(asdict(case), default=str), encoding="utf-8")
    logger.info(f"Cached case {case.docket_id} to {path}")


def load_cached_case(docket_id: int) -> Optional[BankruptcyCase]:
    """Load a BankruptcyCase from disk cache, or return None if not cached."""
    path = config.CASES_DIR / f"{docket_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        for e in data.pop("entries", []):
            docs = [RecapDocument(**d) for d in e.pop("documents", [])]
            entries.append(DocketEntry(**e, documents=docs))
        return BankruptcyCase(**data, entries=entries)
    except Exception:
        logger.exception(f"Failed to load cached case {docket_id}")
        return None


def list_cached_cases() -> list[dict]:
    """Return summary info (docket_id, case_name, docket_number) for all cached cases."""
    cases = []
    for path in sorted(config.CASES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cases.append({
                "docket_id": data["docket_id"],
                "case_name": data.get("case_name", "Unknown"),
                "docket_number": data.get("docket_number", ""),
                "court": data.get("court", ""),
            })
        except Exception:
            continue
    return cases


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
