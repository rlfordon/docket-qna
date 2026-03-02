"""Bankruptcy Docket Q&A — Streamlit UI.

A browser-based interface for loading, indexing, and querying
bankruptcy case documents from the RECAP Archive.
"""

import logging
import re

import streamlit as st

import config
from courtlistener import (
    CourtListenerClient, parse_courtlistener_url, BankruptcyCase,
    save_case, load_cached_case, list_cached_cases, description_quality_stats,
    get_poor_description_date_range, get_purchasable_for_entries,
)
from classifier import classify_document, DocType, get_type_summary
from indexer import CaseIndex
from query import query_case

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _escape_dollars(text: str) -> str:
    """Escape bare $ signs so Streamlit doesn't render them as LaTeX."""
    return text.replace("$", "\\$")


def _linkify_ecf_numbers(text: str, docket_id: int, case_name: str) -> str:
    """Replace 'ECF No. 47' citations with clickable CourtListener links."""
    slug = re.sub(r"[^a-z0-9]+", "-", case_name.lower()).strip("-")
    return re.sub(
        r"ECF No\.\s*(\d+)",
        lambda m: (
            f"[ECF No. {m.group(1)}]"
            f"(https://www.courtlistener.com/docket/{docket_id}/{slug}/"
            f"?entry_gte={m.group(1)}&entry_lte={m.group(1)}"
            f"#entry-{m.group(1)})"
        ),
        text,
    )


def _format_date(iso_string: str) -> str:
    """Format an ISO 8601 date string as 'Dec 15, 2025'."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%b %d, %Y").replace(" 0", " ")
    except (ValueError, TypeError):
        return iso_string

# --- Page Config ---
st.set_page_config(
    page_title="Bankruptcy Docket Q&A",
    page_icon="⚖️",
    layout="wide",
)

# --- Session State ---
if "case" not in st.session_state:
    st.session_state.case = None
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pacer_date_range" not in st.session_state:
    st.session_state.pacer_date_range = (None, None)
if "pending_purchases" not in st.session_state:
    st.session_state.pending_purchases = None
if "purchase_in_progress" not in st.session_state:
    st.session_state.purchase_in_progress = False


# --- PACER Confirmation Dialog ---
@st.dialog("Update Docket from PACER")
def _pacer_confirm_dialog(case: BankruptcyCase):
    """Modal dialog for confirming a PACER docket purchase."""
    stats = description_quality_stats(case)
    improvable = stats["stub"] + stats["empty"]
    date_start, date_end = st.session_state.pacer_date_range

    if date_start and date_end:
        st.markdown(
            f"Purchase will be scoped to entries filed "
            f"**{date_start}** to **{date_end}** "
            f"({improvable} entries with poor descriptions)."
        )
    else:
        st.markdown(
            f"Date range could not be determined — "
            f"purchasing the **full docket** ({improvable} entries "
            f"with poor descriptions)."
        )

    st.warning(
        "**PACER costs:** Docket sheets are billed at ~\\$0.10/page. "
        "Large cases may cost \\$5–\\$20+. Charges are non-refundable."
    )

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Confirm Purchase", type="primary", use_container_width=True):
            _execute_pacer_update(case)
            st.rerun()
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pacer_date_range = (None, None)
            st.rerun()


def main():
    st.title("⚖️ Bankruptcy Docket Q&A")
    st.caption(
        "Ask natural language questions about any federal bankruptcy case — "
        "powered by the RECAP Archive and CourtListener API."
    )

    if config.DEMO_MODE:
        st.info(
            "**Live Demo** — Explore bankruptcy court filings with AI.  \n"
            "**1.** Pick a case from the sidebar  \n"
            "**2.** Ask a question in plain English  \n"
            "**3.** Get an answer grounded in the docket with ECF citations"
        )
        with st.expander("💡 Example questions to try"):
            st.markdown(
                "- What motions have been filed in this case?\n"
                "- Who are the largest creditors?\n"
                "- What is the timeline of key events?\n"
                "- Has a plan of reorganization been proposed?\n"
                "- What objections have been raised?"
            )

    case: BankruptcyCase | None = st.session_state.case
    index: CaseIndex | None = st.session_state.index
    is_indexed = index.exists() if index else False

    # --- Sidebar ---
    with st.sidebar:
        # Validate config
        errors = config.validate_config()
        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        if case is not None:
            _render_sidebar_case_info(case, index, is_indexed)
        else:
            _render_sidebar_no_case()

    # --- Main Area ---
    if case is None:
        _render_case_loader()
    else:
        _render_chat(case, index, is_indexed)


def _render_sidebar_no_case():
    """Sidebar content when no case is loaded — recent cases list."""
    st.header("Recent Cases")
    cached = list_cached_cases()
    if cached:
        for c in cached:
            label = f"{c['case_name'][:40]}"
            if st.button(label, key=f"recent_{c['docket_id']}", use_container_width=True):
                _load_from_cache(c["docket_id"])
    else:
        st.caption("No cached cases yet.")

    # --- Footer ---
    st.caption(
        "Powered by [CourtListener](https://www.courtlistener.com/) · "
        "[About](https://github.com/rlfordon/docket-qna)"
    )


def _render_sidebar_case_info(case: BankruptcyCase, index: CaseIndex, is_indexed: bool):
    """Sidebar content when a case is loaded — all case info and controls."""
    slug = re.sub(r"[^a-z0-9]+", "-", case.case_name.lower()).strip("-")

    # --- Case Identity ---
    st.header(case.case_name)
    st.caption(
        f"{case.court.upper()} · Ch. {case.chapter or 'N/A'} · "
        f"Filed {case.date_filed or 'Unknown'}  \n"
        f"[View on CourtListener →](https://www.courtlistener.com/docket/{case.docket_id}/{slug}/)"
    )

    # --- Coverage Stats ---
    stats = description_quality_stats(case)
    doc_entry_dicts = [
        {"description": e.description} for e in case.entries
        if any(d.plain_text for d in e.documents)
    ]
    type_summary = get_type_summary(doc_entry_dicts)

    # Docket coverage popover
    docket_col, doc_col = st.columns(2)
    with docket_col:
        with st.popover(f"Docket {stats['detailed_pct']:.0f}%", use_container_width=True):
            st.caption(f"{stats['detailed']} / {stats['total']} entries")
            st.markdown(f"- Detailed: {stats['detailed']}")
            st.markdown(f"- Stub/Minimal: {stats['stub']}")
            st.markdown(f"- Empty: {stats['empty']}")
    with doc_col:
        with st.popover(f"Docs {case.coverage_pct:.0f}%", use_container_width=True):
            st.caption(f"{case.available_doc_count} / {case.total_entry_count} entries")
            for doc_type_label, count in type_summary.items():
                st.markdown(f"- {doc_type_label}: {count}")

    # Last updated + PACER
    updated_parts = []
    if case.last_updated:
        updated_parts.append(f"Updated {_format_date(case.last_updated)}")
    if config.has_pacer_credentials() and (stats["stub"] + stats["empty"]) > 0:
        updated_parts.append(f"{stats['stub'] + stats['empty']} improvable")
    if updated_parts:
        st.caption(" · ".join(updated_parts))

    if not config.DEMO_MODE:
        if st.button("Refresh from CourtListener", key="sidebar_refresh", use_container_width=True):
            with st.spinner("Refreshing docket entries..."):
                try:
                    client = CourtListenerClient()
                    updated_count = client.refresh_docket_entries(case)
                    save_case(case)
                    if index and index.exists():
                        index.reindex_descriptions(case)
                    st.success(f"Refreshed {updated_count} entry descriptions.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
                    logger.exception("Refresh failed")

        if config.has_pacer_credentials() and (stats["stub"] + stats["empty"]) > 0:
            if st.button("Update from PACER", key="sidebar_pacer", type="secondary", use_container_width=True):
                st.session_state.pacer_date_range = get_poor_description_date_range(case)
                _pacer_confirm_dialog(case)

        # --- Index Controls (only show button when not yet indexed) ---
        if not is_indexed:
            st.divider()
            if st.button("Index Case", type="primary", key="sidebar_index", use_container_width=True):
                with st.spinner("Indexing documents... (this may take a few minutes)"):
                    try:
                        chunk_count = index.index_case(case)
                        st.success(f"Indexed {chunk_count} chunks from {case.available_doc_count} documents")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
                        logger.exception("Indexing failed")

    # --- Query Settings (only when indexed) ---
    if is_indexed:
        st.divider()
        st.caption("QUERY SETTINGS")
        filter_options = ["All document types"] + [dt.label for dt in DocType if dt != DocType.OTHER]
        selected_filter = st.selectbox(
            "Limit answers to a specific filing type:",
            filter_options,
        )
        doc_type_filter = None
        if selected_filter != "All document types":
            for dt in DocType:
                if dt.label == selected_filter:
                    doc_type_filter = dt.value
                    break
        st.session_state._doc_type_filter = doc_type_filter

        st.session_state._descriptions_only = st.toggle(
            "Descriptions only",
            help="Faster, shallower answers using only docket entry descriptions — skips full document text.",
        )

    st.divider()
    with st.popover("Switch Case", use_container_width=True):
        if not config.DEMO_MODE:
            st.markdown("**Load new case:**")
            switch_input = st.text_input(
                "CourtListener URL or docket ID:",
                key="switch_case_input",
                placeholder="https://www.courtlistener.com/docket/...",
            )
            if st.button("Load", key="switch_load_btn", use_container_width=True):
                if switch_input:
                    _load_case_from_input(switch_input)

        cached = list_cached_cases()
        if cached:
            st.markdown("**Recent cases:**" if not config.DEMO_MODE else "**Available cases:**")
            for c in cached:
                if c["docket_id"] == case.docket_id:
                    continue  # Skip the current case
                label = f"{c['case_name'][:40]}"
                if st.button(label, key=f"switch_{c['docket_id']}", use_container_width=True):
                    _load_from_cache(c["docket_id"])

        if config.DEMO_MODE:
            st.info(
                "The full version lets you load any federal bankruptcy case "
                "by URL or docket number."
            )

    # --- RECAP Promo ---
    st.caption(
        "📬 **Expand coverage for free** — set up "
        "[@recap.email](https://www.courtlistener.com/help/recap/email/) as your "
        "ECF notification email."
    )

    # --- Footer ---
    st.caption(
        "Powered by [CourtListener](https://www.courtlistener.com/) · "
        "[About](https://github.com/rlfordon/docket-qna)"
    )


def _load_from_cache(docket_id: int):
    """Load a cached case into session state and rerun."""
    case = load_cached_case(docket_id)
    if case:
        st.session_state.case = case
        st.session_state.index = CaseIndex(docket_id)
        st.session_state.messages = []
        st.rerun()


def _load_case_from_input(case_input: str):
    """Parse input (URL or ID), load case, store in session state."""
    with st.spinner("Loading case from CourtListener..."):
        try:
            docket_id = None
            if case_input.startswith("http"):
                docket_id = parse_courtlistener_url(case_input)
            elif case_input.isdigit():
                docket_id = int(case_input)

            if docket_id is None:
                st.error(
                    "Could not parse input. Please enter a CourtListener URL "
                    "or numeric docket ID."
                )
                return

            # Try cache first
            case = load_cached_case(docket_id)
            if case:
                st.toast("Loaded from cache!")
            else:
                client = CourtListenerClient()
                case = client.load_case(docket_id, fetch_text=True)
                save_case(case)

            st.session_state.case = case
            st.session_state.index = CaseIndex(docket_id)
            st.session_state.messages = []
            st.rerun()

        except Exception as e:
            st.error(f"Error loading case: {e}")
            logger.exception("Failed to load case")


def _execute_pacer_update(case: BankruptcyCase):
    """Purchase updated docket from PACER, refresh entries, and re-index descriptions."""
    client = CourtListenerClient()
    index: CaseIndex = st.session_state.index
    date_start, date_end = st.session_state.pacer_date_range

    try:
        # 1. Submit purchase
        with st.spinner("Submitting docket purchase to PACER..."):
            result = client.purchase_docket(case, date_start=date_start, date_end=date_end)
            request_id = result.get("id")
            if not request_id:
                st.error("No request ID returned from PACER purchase.")
                return

        # 2. Poll for completion
        status_placeholder = st.empty()
        with st.spinner("Waiting for PACER to process the request..."):
            def _progress(status, elapsed):
                labels = {1: "Enqueued", 2: "Complete", 3: "Failed", 4: "In Progress", 5: "Queued for Retry", 6: "Invalid Content", 7: "Needs Info"}
                label = labels.get(status, f"Status {status}")
                status_placeholder.caption(f"PACER status: {label} ({elapsed:.0f}s)")

            client.poll_purchase_status(request_id, progress_callback=_progress)
        status_placeholder.empty()

        # 3. Refresh docket entries
        with st.spinner("Refreshing docket entries from CourtListener..."):
            updated_count = client.refresh_docket_entries(case)

        # 4. Save updated case
        save_case(case)

        # 5. Re-index descriptions if case is indexed
        reindexed = 0
        if index and index.exists():
            with st.spinner("Re-indexing descriptions..."):
                reindexed = index.reindex_descriptions(case)

        st.success(
            f"PACER update complete! Updated {updated_count} entry descriptions."
            + (f" Re-indexed {reindexed} description chunks." if reindexed else "")
        )

    except TimeoutError:
        st.error("PACER purchase timed out. The request may still complete — try refreshing later.")
    except RuntimeError as e:
        st.error(f"PACER purchase failed: {e}")
    except Exception as e:
        st.error(f"Error during PACER update: {e}")
        logger.exception("PACER update failed")


def _execute_doc_purchases():
    """Purchase suggested documents, index them, and re-answer the question."""
    case: BankruptcyCase = st.session_state.case
    index: CaseIndex = st.session_state.index
    pending = st.session_state.pending_purchases
    if not pending:
        return

    question = pending["question"]
    suggestions = pending["suggestions"]
    doc_type_filter = pending.get("doc_type_filter")

    client = CourtListenerClient()

    ecf_numbers = [s["ecf_number"] for s in suggestions]
    purchasable = get_purchasable_for_entries(case, ecf_numbers, client=client)
    purchased_count = 0

    for info in purchasable:
        if not info["purchasable"] or not info["recap_document_id"]:
            st.warning(f"ECF No. {info['entry_number']}: No purchasable document found")
            continue

        try:
            # Purchase
            with st.spinner(f"Purchasing ECF No. {info['entry_number']}..."):
                result = client.purchase_document(info["recap_document_id"])
                request_id = result.get("id")
                if not request_id:
                    st.warning(f"ECF No. {info['entry_number']}: No request ID returned")
                    continue

            # Poll
            with st.spinner(f"Waiting for ECF No. {info['entry_number']}..."):
                client.poll_purchase_status(request_id)

            # Re-fetch the document to get the text
            with st.spinner(f"Fetching document text for ECF No. {info['entry_number']}..."):
                doc_url = f"{config.CL_BASE_URL}/recap-documents/{info['recap_document_id']}/"
                doc_data = client._get(doc_url)
                plain_text = doc_data.get("plain_text", "") or ""

            if not plain_text.strip():
                st.warning(f"ECF No. {info['entry_number']}: Document purchased but no text extracted")
                continue

            # Update the case data
            entry_by_number = {
                e.entry_number: e for e in case.entries if e.entry_number is not None
            }
            entry = entry_by_number.get(info["entry_number"])
            if entry:
                # Find the matching doc and update its text
                for doc in entry.documents:
                    if doc.id == info["recap_document_id"]:
                        doc.plain_text = plain_text
                        break
                else:
                    # Doc not attached yet — shouldn't happen, but handle it
                    from courtlistener import RecapDocument
                    new_doc = RecapDocument(
                        id=info["recap_document_id"],
                        docket_entry_id=entry.id,
                        description=entry.description,
                        date_filed=entry.date_filed,
                        ecf_number=f"ECF No. {entry.entry_number}",
                        plain_text=plain_text,
                        is_available=True,
                        pacer_doc_id=None,
                    )
                    entry.documents.append(new_doc)
                    doc = new_doc

                # Index the document
                with st.spinner(f"Indexing ECF No. {info['entry_number']}..."):
                    chunks_added = index.index_single_document(case, entry, doc)
                    st.caption(f"ECF No. {info['entry_number']}: indexed {chunks_added} chunks")

                purchased_count += 1

        except TimeoutError:
            st.warning(f"ECF No. {info['entry_number']}: Purchase timed out. It may still complete — try again later.")
        except RuntimeError as e:
            err = str(e)
            if "Unable to download PDF" in err:
                st.warning(
                    f"ECF No. {info['entry_number']}: This document couldn't be fetched "
                    f"via the API. It may require direct download from PACER "
                    f"([ecf.nysb.uscourts.gov](https://ecf.nysb.uscourts.gov))."
                )
            else:
                st.warning(f"ECF No. {info['entry_number']}: Purchase failed — {err}")
            logger.warning(f"Purchase failed for ECF No. {info['entry_number']}: {e}")
        except Exception as e:
            st.warning(f"ECF No. {info['entry_number']}: Unexpected error — {e}")
            logger.exception(f"Failed to purchase ECF No. {info['entry_number']}")

    if purchased_count > 0:
        # Save updated case
        save_case(case)
        case.available_doc_count += purchased_count

        # Re-answer the question
        st.info(f"Purchased {purchased_count} document(s). Re-answering your question...")
        with st.spinner("Generating improved answer..."):
            result = query_case(
                question=question,
                case=case,
                index=index,
                doc_type_filter=doc_type_filter,
            )

        st.markdown(_linkify_ecf_numbers(_escape_dollars(result["answer"]), case.docket_id, case.case_name))
        if result["sources"]:
            with st.expander("Sources"):
                for src in result["sources"]:
                    st.caption(
                        f"**{src['ecf_number']}** — "
                        f"{src['description'][:100]} "
                        f"({src['date_filed']})"
                    )

        # Save improved answer to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
    else:
        st.warning("No documents could be purchased. The answer remains unchanged.")

    # Clear pending state
    st.session_state.pending_purchases = None
    st.session_state.purchase_in_progress = False


def _render_case_loader():
    """Render the case loading interface in the main area."""
    if config.DEMO_MODE:
        st.header("Choose a Case")
    else:
        st.header("Load a Case")

        col1, col2 = st.columns([3, 1])
        with col1:
            case_input = st.text_input(
                "Enter a CourtListener docket URL or docket ID:",
                placeholder="https://www.courtlistener.com/docket/67531068/ftx-trading-ltd/",
            )
        with col2:
            load_btn = st.button("Load Case", type="primary", use_container_width=True)

        if load_btn and case_input:
            _load_case_from_input(case_input)

    # Show recent/demo cases in main area too when no case is loaded
    cached = list_cached_cases()
    if cached:
        if not config.DEMO_MODE:
            st.divider()
        st.subheader("Recent Cases" if not config.DEMO_MODE else "Available Cases")
        for c in cached:
            label = f"{c['case_name'][:40]}"
            if st.button(label, key=f"main_recent_{c['docket_id']}", use_container_width=True):
                _load_from_cache(c["docket_id"])


def _render_chat(case: BankruptcyCase, index: CaseIndex, is_indexed: bool):
    """Render the chat-only main area when a case is loaded."""
    if not is_indexed:
        st.info(
            "This case hasn't been indexed yet. Use the **Index Case** button "
            "in the sidebar to enable questions."
        )
        return

    # Chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = _escape_dollars(msg["content"])
            if msg["role"] == "assistant":
                content = _linkify_ecf_numbers(content, case.docket_id, case.case_name)
            st.markdown(content)
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(
                            f"**{src['ecf_number']}** — {src['description'][:100]} "
                            f"({src['date_filed']})"
                        )

    # Clear chat button
    if st.session_state.messages:
        clear_col1, clear_col2 = st.columns([5, 1])
        with clear_col2:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.messages = []
                st.session_state.pending_purchases = None
                st.rerun()

    # Handle pending purchase action
    if st.session_state.purchase_in_progress:
        with st.chat_message("assistant"):
            _execute_doc_purchases()

    # Show pending purchase suggestions (if any, from last answer)
    if st.session_state.pending_purchases and not st.session_state.purchase_in_progress:
        if config.DEMO_MODE:
            pending = st.session_state.pending_purchases
            suggestions = pending["suggestions"]
            st.markdown("**Documents that could improve this answer:**")
            for s in suggestions:
                st.markdown(f"- **ECF No. {s['ecf_number']}** — {s['reason']}")
            st.info(
                "In the full version, these documents can be purchased directly "
                "from PACER and indexed to improve answers."
            )
            if st.button("Dismiss", key="demo_dismiss_purchases"):
                st.session_state.pending_purchases = None
                st.rerun()
        else:
            pending = st.session_state.pending_purchases
            suggestions = pending["suggestions"]

            # Build lookup: entry_number → (page_count, is_transcript)
            entry_info = {}
            for e in case.entries:
                if e.entry_number is not None:
                    page_count = None
                    for d in e.documents:
                        if d.page_count:
                            page_count = d.page_count
                            break
                    is_transcript = bool(
                        e.description and re.search(r"\btranscript\b", e.description, re.IGNORECASE)
                    )
                    entry_info[e.entry_number] = (page_count, is_transcript)

            st.markdown("**Documents that could improve this answer:**")
            has_transcript = False
            selected = []
            for i, s in enumerate(suggestions):
                ecf = s["ecf_number"]
                info = entry_info.get(ecf, (None, False))
                page_count, is_transcript = info
                if is_transcript:
                    has_transcript = True

                # Build label with cost estimate when page count is available
                label = f"ECF No. {ecf} — {s['reason']}"
                if page_count:
                    if is_transcript:
                        cost = page_count * 0.10
                        label += f" · ~{page_count} pp, est. \\${cost:.2f} ⚠️ transcript"
                    else:
                        cost = min(page_count * 0.10, 3.00)
                        label += f" · ~{page_count} pp, est. \\${cost:.2f}"

                checked = st.checkbox(
                    label,
                    value=True,
                    key=f"purchase_cb_{ecf}_{i}",
                )
                if checked:
                    selected.append(s)

            if has_transcript:
                st.warning(
                    "⚠️ **Transcripts are NOT subject to the \\$3 cap** and are billed "
                    "at \\$0.10/page. A 200-page transcript costs \\$20."
                )
            st.caption(
                "Most PACER documents are capped at \\$3 each (\\$0.10/page). "
                "Transcripts are exempt from this cap."
            )

            col_buy, col_skip = st.columns(2)
            with col_buy:
                if st.button(
                    f"Purchase ({len(selected)}) & Re-answer",
                    type="primary",
                    use_container_width=True,
                    disabled=len(selected) == 0,
                ):
                    st.session_state.pending_purchases["suggestions"] = selected
                    st.session_state.purchase_in_progress = True
                    st.rerun()
            with col_skip:
                if st.button("Dismiss", use_container_width=True):
                    st.session_state.pending_purchases = None
                    st.rerun()

    doc_type_filter = getattr(st.session_state, "_doc_type_filter", None)
    descriptions_only = getattr(st.session_state, "_descriptions_only", False)

    # Chat input — in main body so it pins to bottom
    if question := st.chat_input("Ask about this case..."):
        # Clear any pending purchase state from previous question
        st.session_state.pending_purchases = None
        st.session_state.purchase_in_progress = False

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.status("Researching...", expanded=True) as status:
                try:
                    result = query_case(
                        question=question,
                        case=case,
                        index=index,
                        doc_type_filter=doc_type_filter,
                        descriptions_only=descriptions_only,
                        progress=lambda msg: st.write(msg),
                    )
                    status.update(label="Answer ready", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Error", state="error", expanded=False)
                    st.error(f"Error generating answer: {e}")
                    logger.exception("Query failed")
                    result = None

            if result:
                st.markdown(_linkify_ecf_numbers(_escape_dollars(result["answer"]), case.docket_id, case.case_name))

                if result["sources"]:
                    with st.expander("Sources"):
                        for src in result["sources"]:
                            st.caption(
                                f"**{src['ecf_number']}** — "
                                f"{src['description'][:100]} "
                                f"({src['date_filed']})"
                            )

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )

                # Show purchase suggestions if available
                purchases = result.get("suggested_purchases", [])
                if purchases and (config.has_pacer_credentials() or config.DEMO_MODE):
                    st.session_state.pending_purchases = {
                        "question": question,
                        "suggestions": purchases,
                        "doc_type_filter": doc_type_filter,
                    }
                    st.rerun()


if __name__ == "__main__":
    main()
