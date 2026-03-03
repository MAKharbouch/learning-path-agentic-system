"""Extract prerequisites from course summaries using LLM structured output.

Usage:
    python scripts/extract_prerequisites.py
    python scripts/extract_prerequisites.py --force

Calls the LLM (OpenAI) for each course that has a summary_text, extracts
prerequisites via Pydantic-validated structured output, and writes them to
course_prerequisites. Skips courses already processed unless --force is given.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path (same pattern as extract_skills.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.schema import init_db
from llm.extract_prerequisites import process_all_prerequisites

import _log; _log.setup()
logger = logging.getLogger(__name__)


def main():
    """Initialize database schema and run batch prerequisite extraction."""
    parser = argparse.ArgumentParser(
        description="Extract prerequisites from course summaries using LLM."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract prerequisites for all courses, even those already processed",
    )
    args = parser.parse_args()

    logger.info("Starting prerequisite extraction...")
    init_db()

    result = process_all_prerequisites(force=args.force)

    logger.info(
        "Prerequisite extraction complete: %d courses processed, %d skipped, %d prerequisites extracted",
        result["processed"],
        result["skipped"],
        result["total_prerequisites"],
    )

    if result["skipped"] > 0:
        logger.warning(
            "%d courses skipped due to LLM refusal or error",
            result["skipped"],
        )


if __name__ == "__main__":
    main()
