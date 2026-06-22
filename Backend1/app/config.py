# config.py — central settings for local and deployed environments

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/new_vector_db")
RAG_DATA_PATH = os.getenv("RAG_DATA_PATH", "./data/rag_data")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
PORT = int(os.getenv("PORT", "8000"))
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "llama-3.1-8b-instant")

for _path in (VECTOR_DB_PATH, RAG_DATA_PATH):
    try:
        os.makedirs(_path, exist_ok=True)
    except OSError:
        pass


def require_groq_api_key() -> str:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to Backend1/.env or your deployment environment."
        )
    return GROQ_API_KEY
