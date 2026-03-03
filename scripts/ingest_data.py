"""Ingest Excel data files into the SQLite database.

Usage:
    python scripts/ingest_data.py

Loads Course_Master_List.xlsx, User_Master_List.xlsx, and Completion_Data.xlsx
from excel_data/ into the courses, users, and course_completions tables.
Ensures schema exists before ingesting. Idempotent -- safe to run multiple times.
"""

import logging
import sys
from pathlib import Path

# Add project root to path (same pattern as init_db.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.schema import init_db
from db.ingest import ingest_all

import _log; _log.setup()
logger = logging.getLogger(__name__)


def main():
    """Initialize database schema and ingest all Excel data."""
    logger.info("Initializing database schema...")
    init_db()

    logger.info("Starting data ingestion...")
    counts = ingest_all()

    logger.info(
        "Ingestion complete: %d courses, %d users, %d completions",
        counts["courses"],
        counts["users"],
        counts["completions"],
    )


if __name__ == "__main__":
    main()
