"""RECAP Bankruptcy Case Intelligence — Streamlit UI.

A browser-based interface for loading, indexing, and querying
bankruptcy case documents from the RECAP Archive.
"""

import logging

import streamlit as st

import config
from courtlistener import (
    CourtListenerClient, parse_courtlistener_url, BankruptcyCase,
    save_case, load_cached_case, list_cached_cases, description_quality_stats,
    get_poor_description_date_range,
)
from classifier import classify_document, DocType, get_type_summary
from indexer import CaseIndex
from query import query_case

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="RECAP Bankruptcy Case Intelligence",
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
if "show_pacer_confirm" not in st.session_state:
    st.session_state.show_pacer_confirm = False
if "pacer_date_range" not in st.session_state:
    st.session_state.pacer_date_range = (None, None)


def main():
    st.title("⚖️ RECAP Bankruptcy Case Intelligence")
    st.caption(
        "Ask natural language questions about any federal bankruptcy case — "
        "powered by the RECAP Archive and CourtListener API."
    )

    # --- Sidebar: Configuration ---
    with st.sidebar:
        st.header("Configuration")

        # Validate config
        errors = config.validate_config()
        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        st.success("✓ API keys configured")
        st.caption(f"Embedding: {config.EMBEDDING_PROVIDER.upper()}")
        st.caption(f"LLM: {config.LLM_MODEL}")

        st.divider()

        # --- Recent Cases ---
        st.header("Recent Cases")
        cached = list_cached_cases()
        if cached:
            for c in cached:
                label = f"{c['case_name'][:40]}"
                if st.button(label, key=f"recent_{c['docket_id']}", use_container_width=True):
                    _load_from_cache(c["docket_id"])
        else:
            st.caption("No cached cases yet.")

        st.divider()

        # --- RECAP Info ---
        st.header("📬 Expand Coverage for Free")
        st.markdown(
            "Set up **@recap.email** as your ECF notification email "
            "to automatically contribute filings to the RECAP Archive.\n\n"
            "[Learn more →](https://free.law/recap/)"
        )

    # --- Main Area: Case Loading ---
    if st.session_state.case is None:
        _render_case_loader()
    else:
        _render_case_dashboard()


def _load_from_cache(docket_id: int):
    """Load a cached case into session state and rerun."""
    case = load_cached_case(docket_id)
    if case:
        st.session_state.case = case
        st.session_state.index = CaseIndex(docket_id)
        st.session_state.messages = []
        st.rerun()


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
        st.rerun()

    except TimeoutError:
        st.error("PACER purchase timed out. The request may still complete — try refreshing later.")
    except RuntimeError as e:
        st.error(f"PACER purchase failed: {e}")
    except Exception as e:
        st.error(f"Error during PACER update: {e}")
        logger.exception("PACER update failed")


def _render_case_loader():
    """Render the case loading interface."""
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
        with st.spinner("Loading case from CourtListener..."):
            try:
                # Parse input
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


def _render_case_dashboard():
    """Render the case dashboard with info, indexing, and Q&A."""
    case: BankruptcyCase = st.session_state.case
    index: CaseIndex = st.session_state.index

    # --- Case Header ---
    st.header(case.case_name)
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", case.case_name.lower()).strip("-")
    st.markdown(f"[View on CourtListener →](https://www.courtlistener.com/docket/{case.docket_id}/{slug}/)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Docket Number", case.docket_number)
    col2.metric("Court", case.court.upper())
    col3.metric("Chapter", case.chapter or "N/A")
    col4.metric("Filed", case.date_filed or "Unknown")

    # --- Coverage Stats ---
    st.subheader("Document Coverage")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Docket Entries", case.total_entry_count)
    col2.metric("Documents in RECAP", case.available_doc_count)
    col3.metric("Coverage", f"{case.coverage_pct:.0f}%")

    # Document type breakdown
    entry_dicts = [
        {"description": e.description} for e in case.entries
    ]
    type_summary = get_type_summary(entry_dicts)

    with st.expander("Filing Types Breakdown"):
        for doc_type_label, count in type_summary.items():
            st.write(f"**{doc_type_label}:** {count}")

    # --- Description Quality Stats ---
    st.subheader("Description Quality")
    stats = description_quality_stats(case)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Detailed Descriptions", stats["detailed"])
    col2.metric("Stub/Minimal", stats["stub"])
    col3.metric("Empty", stats["empty"])
    col4.metric("Quality", f"{stats['detailed_pct']:.0f}%")
    if case.last_updated:
        st.caption(f"Last updated: {case.last_updated}")

    # --- PACER Docket Update ---
    if config.has_pacer_credentials():
        improvable = stats["stub"] + stats["empty"]
        if improvable > 0:
            st.info(
                f"{improvable} entries have minimal or empty descriptions. "
                "Updating the docket from PACER can improve search quality."
            )
        if st.session_state.show_pacer_confirm:
            date_start, date_end = st.session_state.pacer_date_range
            if date_start and date_end:
                scope_msg = (
                    f"Purchase will be scoped to entries filed "
                    f"**{date_start}** to **{date_end}** "
                    f"({improvable} entries with poor descriptions)."
                )
            else:
                scope_msg = (
                    f"Date range could not be determined — "
                    f"purchasing the **full docket** ({improvable} entries "
                    f"with poor descriptions)."
                )
            st.warning(
                f"{scope_msg}\n\n"
                "**PACER costs:** Docket sheets are billed at ~$0.10/page. "
                "Large cases may cost $5-$20+. Charges are non-refundable."
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Confirm Purchase", type="primary", use_container_width=True):
                    st.session_state.show_pacer_confirm = False
                    _execute_pacer_update(case)
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_pacer_confirm = False
                    st.session_state.pacer_date_range = (None, None)
                    st.rerun()
        else:
            if st.button("Update Docket from PACER", use_container_width=True):
                st.session_state.pacer_date_range = get_poor_description_date_range(case)
                st.session_state.show_pacer_confirm = True
                st.rerun()

    # --- Indexing ---
    st.subheader("Index & Search")

    is_indexed = index.exists()
    if is_indexed:
        st.success("✓ Case is indexed and ready for questions")
    else:
        st.info("Case has not been indexed yet. Click below to index available documents.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button(
            "Re-index" if is_indexed else "Index Case",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Indexing documents... (this may take a few minutes)"):
                try:
                    chunk_count = index.index_case(case)
                    st.success(f"Indexed {chunk_count} chunks from {case.available_doc_count} documents")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.exception("Indexing failed")
    with col2:
        if st.button("Load Different Case", use_container_width=True):
            st.session_state.case = None
            st.session_state.index = None
            st.session_state.messages = []
            st.rerun()
    with col3:
        if st.button("Refresh from API", use_container_width=True):
            with st.spinner("Re-fetching case from CourtListener..."):
                try:
                    client = CourtListenerClient()
                    refreshed = client.load_case(case.docket_id, fetch_text=True)
                    save_case(refreshed)
                    st.session_state.case = refreshed
                    st.session_state.index = CaseIndex(case.docket_id)
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Error refreshing case: {e}")
                    logger.exception("Failed to refresh case")

    # --- Q&A Chat ---
    if is_indexed:
        st.divider()
        st.subheader("Ask Questions")

        # Optional doc type filter
        filter_options = ["All document types"] + [dt.label for dt in DocType if dt != DocType.OTHER]
        selected_filter = st.selectbox("Filter by document type (optional):", filter_options)

        doc_type_filter = None
        if selected_filter != "All document types":
            # Reverse lookup from label to value
            for dt in DocType:
                if dt.label == selected_filter:
                    doc_type_filter = dt.value
                    break

        # Chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📄 Sources"):
                        for src in msg["sources"]:
                            st.caption(
                                f"**{src['ecf_number']}** — {src['description'][:100]} "
                                f"({src['date_filed']})"
                            )

        # Chat input
        if question := st.chat_input("Ask about this case..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        result = query_case(
                            question=question,
                            case=case,
                            index=index,
                            doc_type_filter=doc_type_filter,
                        )
                        st.markdown(result["answer"])

                        if result["sources"]:
                            with st.expander("📄 Sources"):
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
                                "sources": result["sources"],
                            }
                        )

                    except Exception as e:
                        error_msg = f"Error generating answer: {e}"
                        st.error(error_msg)
                        logger.exception("Query failed")


if __name__ == "__main__":
    main()
