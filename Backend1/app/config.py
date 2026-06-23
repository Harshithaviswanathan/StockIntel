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
# Set ENABLE_RAG=true locally for full vector search. Render free tier: keep false.
ENABLE_RAG = os.getenv("ENABLE_RAG", "false").lower() in ("true", "1", "yes")

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
    if not GROQ_API_KEY.startswith("gsk_"):
        raise ValueError(
            "GROQ_API_KEY looks invalid (must start with gsk_). Update it in Render → Environment."
        )
    return GROQ_API_KEY


def check_groq_api_key() -> dict:
    """Validate Groq key presence and acceptability (lightweight API ping)."""
    if not GROQ_API_KEY:
        return {
            "configured": False,
            "valid": False,
            "message": "GROQ_API_KEY is not set on the server",
        }
    if not GROQ_API_KEY.startswith("gsk_"):
        return {
            "configured": True,
            "valid": False,
            "message": "GROQ_API_KEY format invalid (must start with gsk_)",
        }

    try:
        import requests

        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=5,
        )
        if response.status_code == 200:
            return {"configured": True, "valid": True, "message": "ok"}
        if response.status_code == 401:
            return {
                "configured": True,
                "valid": False,
                "message": "Groq rejected the API key — create a new key at console.groq.com and update Render",
            }
        return {
            "configured": True,
            "valid": False,
            "message": f"Groq API check failed with status {response.status_code}",
        }
    except Exception as exc:
        return {"configured": True, "valid": False, "message": str(exc)}
