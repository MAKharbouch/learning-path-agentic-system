"""Load prerequisite data for candidate courses from SQLite.

Provides context for the LLM plan generator to respect prerequisite
ordering when sequencing the learning plan.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prerequisites_for_candidates(
    course_ids: list[int],
    db_path: Path | None = None,
) -> dict[int, list[dict]]:
    """Load prerequisites from course_prerequisites for given course IDs.

    Args:
        course_ids: List of candidate course IDs to load prereqs for.
        db_path: Path to SQLite database. If None, uses config default.

    Returns:
        Dict mapping course_id to list of prerequisite dicts, each with
        keys: prereq_name, relevance_strength, reason_short.
        Courses with no prerequisites will not appear in the dict.
    """
    from db.connection import get_connection

    if not course_ids:
        return {}

    conn = get_connection(db_path)
    try:
        placeholders = ", ".join("?" for _ in course_ids)
        query = f"""
            SELECT course_id, prereq_name, relevance_strength, reason_short
            FROM course_prerequisites
            WHERE course_id IN ({placeholders})
            ORDER BY course_id, relevance_strength DESC
        """
        rows = conn.execute(query, course_ids).fetchall()

        prereqs: dict[int, list[dict]] = {}
        for row in rows:
            cid = row["course_id"]
            if cid not in prereqs:
                prereqs[cid] = []
            prereqs[cid].append({
                "prereq_name": row["prereq_name"],
                "relevance_strength": row["relevance_strength"],
                "reason_short": row["reason_short"],
            })

        logger.info(
            "Loaded prerequisites for %d of %d candidates",
            len(prereqs),
            len(course_ids),
        )

        return prereqs
    finally:
        conn.close()
