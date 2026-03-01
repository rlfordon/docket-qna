"""RAG query pipeline: retrieval + LLM generation for bankruptcy case Q&A.

Supports Anthropic (Claude) and OpenAI as LLM providers.
"""

import logging
from pathlib import Path
from typing import Optional

import config
from indexer import CaseIndex
from courtlistener import BankruptcyCase

logger = logging.getLogger(__name__)


def load_system_prompt(case: BankruptcyCase) -> str:
    """Load and populate the system prompt with case-specific context.

    Args:
        case: The loaded BankruptcyCase for context.

    Returns:
        Formatted system prompt string.
    """
    prompt_path = config.SYSTEM_PROMPT_PATH
    if prompt_path.exists():
        template = prompt_path.read_text()
    else:
        template = DEFAULT_SYSTEM_PROMPT

    return template.format(
        case_name=case.case_name,
        docket_number=case.docket_number,
        court=case.court,
        chapter=case.chapter or "Unknown",
        date_filed=case.date_filed or "Unknown",
        assigned_to=case.assigned_to or "Unknown",
        total_entries=case.total_entry_count,
        available_docs=case.available_doc_count,
        coverage_pct=f"{case.coverage_pct:.0f}",
    )


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into context for the LLM.

    Args:
        chunks: List of chunk dicts from CaseIndex.query()

    Returns:
        Formatted context string with chunk metadata.
    """
    if not chunks:
        return "No relevant documents found in the index."

    context_parts = []
    seen_entries = set()

    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        ecf = meta.get("ecf_number", "Unknown")
        desc = meta.get("description", "")[:200]
        date = meta.get("date_filed", "Unknown date")
        doc_type = meta.get("doc_type", "other")
        chunk_info = f"(chunk {meta.get('chunk_index', 0)+1}/{meta.get('total_chunks', 1)})"

        header = f"[Source {i}: {ecf} | {doc_type} | {date} | {desc} {chunk_info}]"
        context_parts.append(f"{header}\n{chunk['text']}")

        seen_entries.add(meta.get("entry_number", 0))

    return "\n\n---\n\n".join(context_parts)


def query_case(
    question: str,
    case: BankruptcyCase,
    index: CaseIndex,
    doc_type_filter: Optional[str] = None,
    top_k: int = config.RETRIEVAL_TOP_K,
) -> dict:
    """Run a full RAG query against a bankruptcy case.

    Args:
        question: The user's natural language question.
        case: The loaded BankruptcyCase.
        index: The CaseIndex for this case.
        doc_type_filter: Optional doc type to filter retrieval.
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with keys: answer, sources, chunks_used
    """
    # 1. Retrieve relevant chunks
    chunks = index.query(
        question=question,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
    )

    if not chunks:
        return {
            "answer": (
                "I couldn't find relevant documents in the index for this question. "
                f"This case currently has {case.available_doc_count} of "
                f"{case.total_entry_count} documents indexed. "
                "You may need to purchase additional documents from PACER."
            ),
            "sources": [],
            "chunks_used": 0,
        }

    # 2. Build context and prompt
    system_prompt = load_system_prompt(case)
    context = format_context(chunks)

    user_message = (
        f"Based on the following documents from this bankruptcy case, "
        f"please answer this question:\n\n"
        f"**Question:** {question}\n\n"
        f"**Retrieved Documents:**\n\n{context}"
    )

    # 3. Call LLM
    answer = _call_llm(system_prompt, user_message)

    # 4. Extract source references
    sources = []
    seen = set()
    for chunk in chunks:
        ecf = chunk["metadata"].get("ecf_number", "")
        if ecf and ecf not in seen:
            seen.add(ecf)
            sources.append(
                {
                    "ecf_number": ecf,
                    "description": chunk["metadata"].get("description", ""),
                    "date_filed": chunk["metadata"].get("date_filed", ""),
                    "doc_type": chunk["metadata"].get("doc_type", ""),
                }
            )

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Call the configured LLM provider.

    Args:
        system_prompt: System-level instructions.
        user_message: The user's message with context.

    Returns:
        The LLM's response text.
    """
    if config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, user_message)
    else:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """Call Anthropic's API."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def _call_openai(system_prompt: str, user_message: str) -> str:
    """Call OpenAI's API."""
    import openai

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


# --- Default System Prompt ---

DEFAULT_SYSTEM_PROMPT = """You are a bankruptcy case research assistant analyzing documents from a specific federal bankruptcy case. You help practitioners, creditors, and researchers understand what has been filed in this case.

## Case Information
- **Case Name:** {case_name}
- **Docket Number:** {docket_number}
- **Court:** {court}
- **Chapter:** {chapter}
- **Date Filed:** {date_filed}
- **Assigned To:** {assigned_to}
- **Document Coverage:** {available_docs} of {total_entries} docket entries have documents indexed ({coverage_pct}%)

## Instructions

1. **Answer based ONLY on the retrieved documents provided.** Do not use outside knowledge about this case or bankruptcy law generally, except to provide context for terminology.

2. **ALWAYS cite specific docket entry numbers** when referencing information. Use the format "ECF No. X" or "(ECF No. X)" for citations. If information comes from multiple documents, cite all relevant ones.

3. **Distinguish between what documents explicitly state vs. your inferences.** Use phrases like "According to ECF No. 47..." for direct information and "Based on the available filings, it appears that..." for inferences.

4. **When asked for structured output** (tables, lists, timelines), format using markdown. For tables, include relevant columns like ECF number, date, filing party, and a summary.

5. **Flag incomplete coverage.** If the question likely requires documents you don't have access to, say so. For example: "Note: I have access to {available_docs} of {total_entries} docket entries. The answer may be incomplete if relevant filings have not been indexed."

6. **Use correct bankruptcy terminology.** Refer to the debtor, creditors, the estate, the trustee, DIP (debtor-in-possession), etc. appropriately.

7. **When you cannot answer a question from the available documents, say so clearly.** Suggest what type of filing might contain the answer (e.g., "This information would likely be in the Disclosure Statement or Monthly Operating Reports").

8. **Be concise but thorough.** Provide the key information without unnecessary elaboration, but don't omit important details or caveats.
"""
