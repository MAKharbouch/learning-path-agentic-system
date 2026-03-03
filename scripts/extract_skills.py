"""Extract skills from course summaries using LLM structured output.

Usage:
    python scripts/extract_skills.py
    python scripts/extract_skills.py --force

Calls the LLM for each course that has a summary_text, extracts
skills via Pydantic-validated structured output, and writes them to
course_skills. Skips courses already processed unless --force is given.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path (same pattern as ingest_data.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.schema import init_db
from llm.extract_skills import process_all_courses

import _log; _log.setup()
logger = logging.getLogger(__name__)


def main():
    """Initialize database schema and run batch skill extraction."""
    parser = argparse.ArgumentParser(
        description="Extract skills from course summaries using LLM."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract skills for all courses, even those already processed",
    )
    args = parser.parse_args()

    logger.info("Starting skill extraction...")
    init_db()

    result = process_all_courses(force=args.force)

    logger.info(
        "Skill extraction complete: %d courses processed, %d skipped, %d skills extracted",
        result["processed"],
        result["skipped"],
        result["total_skills"],
    )

    if result["skipped"] > 0:
        logger.warning(
            "%d courses skipped due to LLM refusal or error",
            result["skipped"],
        )


if __name__ == "__main__":
    main()
