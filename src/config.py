"""Centralized configuration loaded from environment variables or Streamlit secrets."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Try to import streamlit for secrets support (only available when running in Streamlit)
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False
    st = None


def _get_secret(key: str, default: str = "") -> str:
    """Get secret from Streamlit secrets or environment variables."""
    if _STREAMLIT_AVAILABLE:
        try:
            # Try Streamlit secrets first
            if hasattr(st, 'secrets') and key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    # Fall back to environment variable
    return os.getenv(key, default)

# Project root: src/config.py -> src -> project root
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Database
_db_path = Path(os.getenv("DB_PATH", "data/learning_path.db"))
DB_PATH: Path = _db_path if _db_path.is_absolute() else BASE_DIR / _db_path

# ChromaDB
_chroma_path = Path(os.getenv("CHROMA_PATH", "data/chroma"))
CHROMA_PATH: Path = _chroma_path if _chroma_path.is_absolute() else BASE_DIR / _chroma_path
CHROMA_COLLECTION_NAME: str = "course_embeddings"

# Sample data directory
SAMPLE_DATA_DIR: Path = BASE_DIR / "excel_data"

# LLM
LLM_PROVIDER: str = _get_secret("LLM_PROVIDER", "openai")
LLM_MODEL: str = _get_secret("LLM_MODEL", "gpt-4o")

# Embeddings
EMBEDDING_PROVIDER: str = _get_secret("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL: str = "text-embedding-3-small"
OPENAI_API_KEY: str = _get_secret("OPENAI_API_KEY", "")
