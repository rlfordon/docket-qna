"""Configuration management for RECAP Bankruptcy Case Intelligence."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# --- API Keys ---
COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PACER_USERNAME = os.getenv("PACER_USERNAME", "")
PACER_PASSWORD = os.getenv("PACER_PASSWORD", "")

# --- Provider Settings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "flp")  # "flp" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

# --- CourtListener API ---
CL_BASE_URL = "https://www.courtlistener.com/api/rest/v3"
CL_V4_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CL_HEADERS = {
    "Authorization": f"Token {COURTLISTENER_API_TOKEN}",
    "Content-Type": "application/json",
}
CL_PAGE_SIZE = 20  # Default page size for API responses
CL_RATE_LIMIT = 5000  # Requests per hour for authenticated users

# --- Embedding Model ---
FLP_MODEL_NAME = "Free-Law-Project/modernbert-embed-base_finetune_512"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- RAG Settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "12"))

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CASES_DIR = DATA_DIR / "cases"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
CASES_DIR.mkdir(exist_ok=True)


def validate_config():
    """Check that required configuration is present."""
    errors = []

    if not COURTLISTENER_API_TOKEN or COURTLISTENER_API_TOKEN == "your_token_here":
        errors.append(
            "COURTLISTENER_API_TOKEN is not set. "
            "Get a free token at https://www.courtlistener.com/sign-in/"
        )

    if LLM_PROVIDER == "anthropic" and (
        not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_key_here"
    ):
        errors.append("ANTHROPIC_API_KEY is not set (required when LLM_PROVIDER=anthropic)")

    if LLM_PROVIDER == "openai" and (
        not OPENAI_API_KEY or OPENAI_API_KEY == "your_key_here"
    ):
        errors.append("OPENAI_API_KEY is not set (required when LLM_PROVIDER=openai)")

    if EMBEDDING_PROVIDER == "openai" and (
        not OPENAI_API_KEY or OPENAI_API_KEY == "your_key_here"
    ):
        errors.append("OPENAI_API_KEY is not set (required when EMBEDDING_PROVIDER=openai)")

    return errors
