"""Unified storage initialization script.

Creates both SQLite database (tables + indexes) and ChromaDB collection
in a single command. Idempotent -- safe to run multiple times.

Usage:
    python scripts/init_db.py
"""

from db.schema import init_db


def main() -> None:
    """Initialize SQLite storage.

    ChromaDB collection is created by the embeddings script
    (scripts/embed_courses.py) with the correct OpenAI embedding function.
    """
    init_db()
    print("SQLite database initialized")


if __name__ == "__main__":
    main()
