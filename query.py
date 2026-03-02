"""RAG query pipeline: retrieval + LLM generation for bankruptcy case Q&A.

Supports Anthropic (Claude) and OpenAI as LLM providers.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import config
from classifier import DocType, classify_document
from indexer import CaseIndex
from courtlistener import BankruptcyCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question intent classification
# ---------------------------------------------------------------------------

@dataclass
class QuestionIntent:
    """Classified intent of a user question."""
    category: str          # docket_listing, type_listing, type_date_listing,
                           # keyword_listing, type_analysis, analytical
    doc_type: Optional[str] = None        # DocType value if detected
    date_range: Optional[tuple[str, str]] = None  # (start, end) ISO dates
    keywords: Optional[list[str]] = None  # Search terms for keyword-based listing


# Map question keywords to DocType values.  Order doesn't matter here since
# we just scan for the first match.
_QUESTION_DOCTYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (DocType.MOTION.value, re.compile(r"(?i)\bmotion(?:s)?\b")),
    (DocType.OBJECTION.value, re.compile(r"(?i)\b(?:objection|opposition|response)(?:s)?\b")),
    (DocType.ORDER.value, re.compile(r"(?i)\border(?:s)?\b")),
    (DocType.PLAN.value, re.compile(r"(?i)\b(?:plan|disclosure statement)(?:s)?\b")),
    (DocType.CLAIM.value, re.compile(r"(?i)\b(?:claim|proof of claim)(?:s)?\b")),
    (DocType.FEE_APPLICATION.value, re.compile(r"(?i)\bfee application(?:s)?\b")),
    (DocType.SCHEDULE.value, re.compile(r"(?i)\b(?:schedule|statement of financial affairs)(?:s)?\b")),
    (DocType.REPORT.value, re.compile(r"(?i)\b(?:report|MOR|monthly operating report)(?:s)?\b")),
    (DocType.NOTICE.value, re.compile(r"(?i)\bnotice(?:s)?\b")),
    (DocType.STIPULATION.value, re.compile(r"(?i)\bstipulation(?:s)?\b")),
    (DocType.DECLARATION.value, re.compile(r"(?i)\b(?:declaration|affidavit)(?:s)?\b")),
]

_LISTING_PATTERN = re.compile(
    r"(?i)(?:"
    r"\b(?:list|show|all|every|how many|docket entries|filings?)\b"
    r"|what\b.*\b(?:were|have been|has been)\s+(?:filed|entered|submitted|docketed)"
    r"|what\b.*\bhave\s+(?:occurred|happened|taken place"
    r"|been\s+(?:filed|held|scheduled|set|reached|retained|asserted|made|issued|granted|denied|paid|raised|identified))"
    r")"
)

_ANALYTICAL_PATTERN = re.compile(
    r"(?i)\b(?:what (?:does|did|do|is|are|were)|explain|summarize|summary|"
    r"terms of|argue|arguments?|analysis|describe|tell me about|why)\b"
)

# Words to strip when extracting search keywords from the question
_STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "for", "to", "and", "or", "is", "are",
    "was", "were", "been", "be", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "that", "this", "these", "those", "it", "its",
    "all", "any", "every", "each", "some", "no", "not",
    "list", "show", "me", "tell", "give", "find", "get",
    "filed", "occurred", "happened", "taken", "place", "there", "been",
    "case", "docket", "about", "with", "from", "by", "at", "up",
    "many", "much",
}


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful search keywords from a question.

    Strips stopwords and question scaffolding to find the core subject terms.
    """
    words = re.findall(r"[a-zA-Z]+", question.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return keywords


# ---------------------------------------------------------------------------
# LLM-based question classifier (alternative to regex)
# ---------------------------------------------------------------------------

_CLASSIFIER_PROMPT = """\
You classify questions about a federal bankruptcy case into a retrieval category.

## Categories

1. **docket_listing** — The user wants a list of docket entries filtered by date range.
   Examples: "What was filed last month?", "List all filings from the past two weeks"

2. **type_listing** — The user wants a list of filings of a specific document type.
   Examples: "What motions have been filed?", "How many claims have been filed?"

3. **type_date_listing** — Combination: list of a specific doc type within a date range.
   Examples: "What orders were entered last month?"

4. **keyword_listing** — The user wants a list of entries matching a concept that isn't a \
document type (hearings, sales, deadlines, professionals, settlements, etc.).
   Examples: "What hearings have occurred?", "List all asset sales", "What settlements have been reached?"

5. **type_analysis** — The user wants substantive analysis of a specific document type. \
They want to understand the *content* of those documents, not just list them.
   Examples: "What do the objections to the plan argue?", "Summarize the monthly operating reports"

6. **analytical** — General analytical question. No specific document type filter needed. \
Includes factual questions, yes/no questions, and questions where a doc-type word appears \
incidentally (e.g. "claims bar date", "first day motions", "When was the plan confirmed?").
   Examples: "What are the key terms of the DIP facility?", "Who are the largest creditors?", \
"What is the claims bar date?", "Has the court approved the DIP facility?"

## Document types
motion, objection, order, plan, claim, fee_application, schedule, report, notice, stipulation, declaration

## Instructions
Respond with ONLY a JSON object (no markdown fences). Fields:
- "category": one of the six categories above
- "doc_type": the document type if relevant (from the list above), or null
- "keywords": list of search keywords for keyword_listing, or null
- "date_range_detected": true if the question references a relative time period ("last month", "past two weeks"), false otherwise\
"""


def classify_question_llm(question: str) -> QuestionIntent:
    """Classify a user question using an LLM call (Haiku).

    Lightweight alternative to the regex-based classifier. Uses a small,
    fast model to parse intent, doc_type, and keywords from the question.
    Falls back to 'analytical' if the LLM response can't be parsed.
    """
    import json

    try:
        response_text = _call_llm(_CLASSIFIER_PROMPT, question)

        # Strip markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]

        result = json.loads(text)

        category = result.get("category", "analytical")
        valid_categories = {
            "docket_listing", "type_listing", "type_date_listing",
            "keyword_listing", "type_analysis", "analytical",
        }
        if category not in valid_categories:
            category = "analytical"

        doc_type = result.get("doc_type")
        keywords = result.get("keywords")

        return QuestionIntent(
            category=category,
            doc_type=doc_type,
            keywords=keywords if keywords else None,
        )

    except Exception as e:
        logger.warning(f"LLM classifier failed, defaulting to analytical: {e}")
        return QuestionIntent(category="analytical")


# ---------------------------------------------------------------------------
# Regex-based question classifier
# ---------------------------------------------------------------------------

def _detect_date_range(question: str, case: BankruptcyCase) -> Optional[tuple[str, str]]:
    """Detect if a question asks about a date range and return (start, end) ISO dates."""
    q = question.lower()

    # Find the latest filing date in the case to anchor relative dates
    all_dates = [e.date_filed for e in case.entries if e.date_filed]
    if not all_dates:
        return None
    latest = max(all_dates)
    anchor = datetime.strptime(latest[:10], "%Y-%m-%d")

    # "past/last N days/weeks/months"
    m = re.search(r"(?:past|last)\s+(\d+)?\s*(day|week|month|two week)", q)
    if m:
        num = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2)
        if "week" in unit:
            if "two" in unit:
                delta = timedelta(weeks=2)
            else:
                delta = timedelta(weeks=num)
        elif unit == "day":
            delta = timedelta(days=num)
        elif unit == "month":
            delta = timedelta(days=num * 30)
        else:
            return None
        start = (anchor - delta).strftime("%Y-%m-%d")
        end = latest[:10]
        return (start, end)

    return None


# Compound terms where a doc-type keyword is incidental, not referring to
# the document type.  Checked before doc-type detection — if any match,
# the corresponding doc-type keyword is suppressed.
_DOCTYPE_EXCLUDE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "claims bar date", "claims process", "claims agent" — not about claim docs
    (DocType.CLAIM.value, re.compile(r"(?i)\bclaims?\s+(?:bar\s+date|process|agent|trading|objection)")),
    # "first day motions" — a bankruptcy concept, not asking to list/analyze motions
    (DocType.MOTION.value, re.compile(r"(?i)\bfirst\s+day\s+motions?\b")),
    # "in order to/for" — not about court orders
    (DocType.ORDER.value, re.compile(r"(?i)\bin\s+order\s+(?:to|for)\b")),
]

# Questions starting with who/when/where or auxiliary verbs (has/have/is/are/
# was/were/did/does/do) are factual questions — even if they mention a doc-type
# keyword, the user is asking for a fact, not requesting analysis of those docs.
# E.g. "When was the plan confirmed?" → analytical, not type_analysis.
_FACTUAL_QUESTION_PATTERN = re.compile(
    r"(?i)^\s*(?:who|when|where|has|have|had|is|are|was|were|did|does|do)\s"
)


def _detect_doc_type(question: str) -> Optional[str]:
    """Detect if a question references a specific document type.

    Returns None if the doc-type keyword appears only as part of a compound
    term (e.g. 'claims bar date') where it doesn't refer to the document type.
    """
    for doc_type_val, pattern in _QUESTION_DOCTYPE_PATTERNS:
        if pattern.search(question):
            # Check exclusions for this doc type
            excluded = False
            for excl_type, excl_pattern in _DOCTYPE_EXCLUDE_PATTERNS:
                if excl_type == doc_type_val and excl_pattern.search(question):
                    excluded = True
                    break
            if not excluded:
                return doc_type_val
    return None


def classify_question(question: str, case: BankruptcyCase) -> QuestionIntent:
    """Classify a user question into an intent category for retrieval routing.

    Returns a QuestionIntent with the category and any detected filters.
    """
    date_range = _detect_date_range(question, case)
    doc_type = _detect_doc_type(question)
    is_listing = bool(_LISTING_PATTERN.search(question))
    is_analytical = bool(_ANALYTICAL_PATTERN.search(question))
    is_factual = bool(_FACTUAL_QUESTION_PATTERN.search(question))

    # Factual questions (who/when/has/is/did ...) that mention a doc-type keyword
    # are asking for a fact, not requesting analysis of those documents.
    # E.g. "When was the plan confirmed?" or "Were any claims disallowed?"
    # Suppress the doc_type so these route to analytical (broad semantic search).
    if doc_type and is_factual and not is_listing:
        doc_type = None

    # Determine category from the combination of signals
    if date_range and doc_type and is_listing:
        category = "type_date_listing"
    elif date_range and is_listing:
        category = "docket_listing"
    elif date_range and doc_type:
        # Date + type but no explicit listing words — still a structured lookup
        category = "type_date_listing"
    elif date_range:
        category = "docket_listing"
    elif doc_type and is_listing and not is_analytical:
        category = "type_listing"
    elif doc_type and not is_listing and not is_analytical:
        # Ambiguous — e.g. "What about the motions?" — default to type_analysis
        category = "type_analysis"
    elif doc_type and is_analytical:
        category = "type_analysis"
    elif is_listing and not doc_type:
        # Listing intent without a doc_type — keyword search across all entries.
        # e.g. "What hearings have occurred?" or "List all sales"
        # But if a strong analytical verb is present (summarize, explain, etc.),
        # the user wants analysis, not a raw listing.
        if is_analytical and re.search(r"(?i)\b(?:summarize|explain|describe|analyze|compare|tell me about)\b", question):
            category = "analytical"
        else:
            category = "keyword_listing"
            keywords = _extract_keywords(question)
            return QuestionIntent(category=category, keywords=keywords)
    else:
        category = "analytical"

    return QuestionIntent(category=category, doc_type=doc_type, date_range=date_range)


# ---------------------------------------------------------------------------
# Structured listing builders
# ---------------------------------------------------------------------------

def _build_structured_listing(case: BankruptcyCase, intent: QuestionIntent) -> str:
    """Build a formatted listing of docket entries matching the intent filters."""
    entries = []
    for e in case.entries:
        if not e.description:
            continue

        # Date filter
        if intent.date_range:
            if not e.date_filed:
                continue
            d = e.date_filed[:10]
            start, end = intent.date_range
            if not (start <= d <= end):
                continue

        # Doc-type filter
        if intent.doc_type:
            entry_type = classify_document(e.description).value
            if entry_type != intent.doc_type:
                continue

        # Keyword filter — entry description must contain at least one keyword
        if intent.keywords:
            desc_lower = e.description.lower()
            if not any(kw in desc_lower for kw in intent.keywords):
                continue

        entries.append(e)

    entries.sort(key=lambda e: (e.date_filed or "", e.entry_number or 0))

    if not entries:
        parts = []
        if intent.doc_type:
            label = DocType(intent.doc_type).label
            parts.append(f"of type '{label}'")
        if intent.date_range:
            parts.append(f"between {intent.date_range[0]} and {intent.date_range[1]}")
        if intent.keywords:
            parts.append(f"matching '{' '.join(intent.keywords)}'")
        qualifier = " ".join(parts) if parts else "matching the criteria"
        return f"No docket entries found {qualifier}."

    # Build header
    header_parts = []
    if intent.doc_type:
        label = DocType(intent.doc_type).label
        header_parts.append(f"Type: {label}")
    if intent.date_range:
        header_parts.append(f"Date range: {intent.date_range[0]} to {intent.date_range[1]}")
    if intent.keywords:
        header_parts.append(f"Keywords: {', '.join(intent.keywords)}")
    header_info = " | ".join(header_parts) if header_parts else "All entries"

    lines = [f"Docket entries ({header_info}) — {len(entries)} entries:\n"]
    for e in entries:
        ecf = f"ECF No. {e.entry_number}" if e.entry_number else "Unnumbered"
        date = e.date_filed[:10] if e.date_filed else "No date"
        lines.append(f"- **{ecf}** ({date}): {e.description}")

    return "\n".join(lines)


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

    # Include purchase suggestion instructions only when PACER creds are configured
    purchase_block = _PURCHASE_SUGGESTION_INSTRUCTIONS if config.has_pacer_credentials() else ""

    format_kwargs = dict(
        case_name=case.case_name,
        docket_number=case.docket_number,
        court=case.court,
        chapter=case.chapter or "Unknown",
        date_filed=case.date_filed or "Unknown",
        assigned_to=case.assigned_to or "Unknown",
        total_entries=case.total_entry_count,
        available_docs=case.available_doc_count,
        coverage_pct=f"{case.coverage_pct:.0f}",
        purchase_suggestion_block=purchase_block,
    )

    try:
        return template.format(**format_kwargs)
    except KeyError:
        # External prompt file may not have {purchase_suggestion_block} placeholder —
        # append purchase instructions directly if PACER creds are configured
        result = template.format(**{k: v for k, v in format_kwargs.items() if k != "purchase_suggestion_block"})
        if purchase_block:
            result += "\n" + purchase_block
        return result


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

        source = meta.get("source", "document")
        desc_tag = " (DESCRIPTION ONLY — no document text available)" if source == "docket_entry" else ""
        header = f"[Source {i}: {ecf} | {doc_type} | {date} | {desc}{desc_tag} {chunk_info}]"
        context_parts.append(f"{header}\n{chunk['text']}")

        seen_entries.add(meta.get("entry_number", 0))

    return "\n\n---\n\n".join(context_parts)


def query_case(
    question: str,
    case: BankruptcyCase,
    index: CaseIndex,
    doc_type_filter: Optional[str] = None,
    descriptions_only: bool = False,
    top_k: int = config.RETRIEVAL_TOP_K,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run a full RAG query against a bankruptcy case.

    Routes to the appropriate retrieval strategy based on question intent:
    - Structured listing for date-range and doc-type listing questions
    - Filtered semantic search for type-specific analytical questions
    - Pure semantic search for general analytical questions

    Args:
        question: The user's natural language question.
        case: The loaded BankruptcyCase.
        index: The CaseIndex for this case.
        doc_type_filter: Optional doc type to filter retrieval (overrides classifier).
        descriptions_only: If True, skip document chunk retrieval and use only
            docket entry descriptions (faster, shallower results).
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with keys: answer, sources, chunks_used
    """
    _progress = progress or (lambda msg: None)

    _progress("Classifying question...")
    intent = classify_question(question, case)
    logger.info(f"Question classified as '{intent.category}' (doc_type={intent.doc_type}, date_range={intent.date_range})")

    # --- Structured listing: answer from case.entries directly ---
    # Date/type listings use direct filtering — fast and deterministic.
    # keyword_listing uses two-stage semantic search instead (below) because
    # literal keyword matching is too brittle for concepts like "hearings"
    # that may appear as "Status Conference", "Oral Argument", etc.
    if intent.category in ("docket_listing", "type_listing", "type_date_listing"):
        _progress("Building structured listing...")
        listing = _build_structured_listing(case, intent)

        _progress("Generating answer...")
        system_prompt = load_system_prompt(case)
        user_message = (
            f"Based on the following docket entry listing from this bankruptcy case, "
            f"please answer this question:\n\n"
            f"**Question:** {question}\n\n"
            f"**Docket Entries:**\n\n{listing}"
        )

        answer = _call_llm(system_prompt, user_message)
        return {
            "answer": answer,
            "sources": [],
            "chunks_used": 0,
            "suggested_purchases": [],
        }

    # --- Two-stage retrieval ---
    # Used for keyword_listing, type_analysis, and analytical questions.
    # Stage 1: Search docket descriptions (covers full docket) to find
    # relevant entries, then Stage 2: pull document chunks from those entries
    # for deep content.  This gives us breadth (all entries) + depth (doc text).

    effective_type_filter = doc_type_filter or (
        intent.doc_type if intent.category == "type_analysis" else None
    )

    # For keyword_listing, search descriptions with a larger top_k to cast
    # a wide net across the full docket for the concept being asked about.
    desc_top_k = 40 if intent.category == "keyword_listing" else 20

    # Stage 1: find relevant entries via their descriptions
    _progress("Searching docket descriptions...")
    desc_hits = index.query_descriptions(
        question=question,
        top_k=desc_top_k,
        doc_type_filter=effective_type_filter,
    )

    if descriptions_only:
        # Skip document chunk retrieval — use description hits directly
        all_chunks = desc_hits
    else:
        # Collect entry IDs from the description hits for stage 2
        hit_entry_ids = list({
            h["metadata"]["docket_entry_id"]
            for h in desc_hits
            if h["metadata"].get("docket_entry_id")
        })

        # Stage 2: search document chunks, scoped to the entries we found
        _progress("Retrieving document passages...")
        if hit_entry_ids:
            doc_chunks = index.query_documents(
                question=question,
                top_k=top_k,
                doc_type_filter=effective_type_filter,
                entry_ids=hit_entry_ids,
            )
        else:
            doc_chunks = []

        # If stage 2 came up short, also do an unscoped document search to
        # catch relevant content from entries whose descriptions didn't match
        if len(doc_chunks) < top_k:
            remaining = top_k - len(doc_chunks)
            fallback_chunks = index.query_documents(
                question=question,
                top_k=remaining,
                doc_type_filter=effective_type_filter,
            )
            # Deduplicate by chunk text
            seen_texts = {c["text"] for c in doc_chunks}
            for fc in fallback_chunks:
                if fc["text"] not in seen_texts:
                    doc_chunks.append(fc)
                    seen_texts.add(fc["text"])

        # Combine: description hits (for breadth) + document chunks (for depth)
        # Deduplicate and prioritize document chunks over descriptions for the
        # same entry, since doc chunks have the actual content.
        doc_entry_ids = {c["metadata"].get("docket_entry_id") for c in doc_chunks}
        unique_desc_hits = [
            d for d in desc_hits
            if d["metadata"].get("docket_entry_id") not in doc_entry_ids
        ]

        # Put doc chunks first (richer content), then unique description hits
        all_chunks = doc_chunks + unique_desc_hits

    if not all_chunks:
        return {
            "answer": (
                "I couldn't find relevant documents in the index for this question. "
                f"This case currently has {case.available_doc_count} of "
                f"{case.total_entry_count} documents indexed. "
                "You may need to purchase additional documents from PACER."
            ),
            "sources": [],
            "chunks_used": 0,
            "suggested_purchases": [],
        }

    # Build context and prompt
    system_prompt = load_system_prompt(case)
    context = format_context(all_chunks)

    user_message = (
        f"Based on the following documents from this bankruptcy case, "
        f"please answer this question:\n\n"
        f"**Question:** {question}\n\n"
        f"**Retrieved Documents:**\n\n{context}"
    )

    _progress("Generating answer...")
    raw_response = _call_llm(system_prompt, user_message)

    # Parse structured JSON output (answer + purchase suggestions)
    if config.has_pacer_credentials():
        answer, suggested_purchases = _parse_llm_response(raw_response)
    else:
        answer = raw_response
        suggested_purchases = []

    # Extract source references
    sources = []
    seen = set()
    for chunk in all_chunks:
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
        "chunks_used": len(all_chunks),
        "suggested_purchases": suggested_purchases,
    }


def _parse_llm_response(response_text: str) -> tuple[str, list[dict]]:
    """Parse an LLM response that may contain structured JSON output.

    Handles three cases:
    1. Entire response is JSON — ideal
    2. JSON wrapped in markdown fences
    3. Prose followed by JSON (LLM didn't follow instructions perfectly)

    Returns:
        (answer_text, suggested_purchases) — falls back to
        (full response, []) if no valid JSON found.
    """
    text = response_text.strip()

    # Try parsing candidates in priority order
    candidates = [text]

    # Strip markdown fences if present
    if "```" in text:
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

    # Extract JSON object starting at {"answer" (LLM wrote prose then JSON)
    answer_match = re.search(r'\{"answer"', text)
    if answer_match:
        candidates.insert(0, text[answer_match.start():].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = parsed["answer"]
                purchases = parsed.get("suggested_purchases", [])
                valid = []
                for p in purchases:
                    if isinstance(p, dict) and "ecf_number" in p:
                        valid.append({
                            "ecf_number": p["ecf_number"],
                            "reason": p.get("reason", ""),
                        })
                return answer, valid
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    return response_text, []


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

9. **You have access to docket entry descriptions** in addition to full document text. Docket entry descriptions are short summaries from the court's docket sheet (e.g., "Motion to Extend Deadline", "Order Granting Relief"). Use these to answer questions about what was filed, case timelines, and docket activity — even if the full document text is not available.

{purchase_suggestion_block}"""


_PURCHASE_SUGGESTION_INSTRUCTIONS = """
10. **CRITICAL — Response format.** Your ENTIRE response must be a single JSON object. Do NOT write any text before or after the JSON. Some sources above are marked "(DESCRIPTION ONLY — no document text available)". Your response format:

{{"answer": "Your full answer here, with all ECF citations, caveats, and formatting. Use markdown within this string.", "suggested_purchases": [{{"ecf_number": 47, "reason": "One-sentence explanation of why this document would improve the answer"}}]}}

Rules:
- Your ENTIRE response must be this JSON object — nothing else
- Put your complete answer (with all citations, caveats, markdown formatting) inside the "answer" field
- suggested_purchases: only include ECF numbers marked "DESCRIPTION ONLY" above whose full text would materially improve the answer
- If the answer is well-supported, use an empty list: "suggested_purchases": []
- Keep reasons concise (one sentence)
"""
