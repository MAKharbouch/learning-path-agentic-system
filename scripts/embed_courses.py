"""Embed course catalog into ChromaDB for semantic search.

Usage:
    python scripts/embed_courses.py

Loads all courses with summaries from SQLite, combines course_name and
summary_text into a document, and upserts into ChromaDB. The collection's
OpenAIEmbeddingFunction handles embedding generation. Uses upsert() for
idempotent re-runs -- no --force flag needed.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path (same pattern as extract_skills.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.schema import init_db
from vectorstore.embed_courses import embed_all_courses

import _log; _log.setup()
logger = logging.getLogger(__name__)


def main():
    """Initialize database schema and run batch course embedding."""
    argparse.ArgumentParser(
        description="Embed course catalog into ChromaDB for semantic search."
    ).parse_args()

    logger.info("Starting course embedding...")
    init_db()

    result = embed_all_courses()

    logger.info(
        "Embedding complete: %d courses embedded",
        result["embedded"],
    )


if __name__ == "__main__":
    main()
