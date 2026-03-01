"""RECAP Bankruptcy Case Intelligence — Streamlit UI.

A browser-based interface for loading, indexing, and querying
bankruptcy case documents from the RECAP Archive.
"""

import logging

import streamlit as st

import config
from courtlistener import CourtListenerClient, parse_courtlistener_url, BankruptcyCase
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

                client = CourtListenerClient()
                case = client.load_case(docket_id, fetch_text=True)
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
