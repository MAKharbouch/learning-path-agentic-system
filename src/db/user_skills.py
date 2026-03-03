"""User skill inference from course completion history.

Infers skills for a user by joining course_completions with course_skills.
Pure SQL query -- no LLM calls, no new tables, always-current results.
"""

import logging

logger = logging.getLogger(__name__)


def get_user_skills(portal_id, db_path=None):
    """Infer skills for a user from their completed courses.

    Joins course_completions with course_skills for completed courses,
    grouping by skill_name and taking the highest skill level when a
    skill appears across multiple completed courses.

    Args:
        portal_id: The user's portal ID.
        db_path: Optional database path. Uses config default if None.

    Returns:
        List of UserSkillInferred instances with highest skill level per skill.
    """
    from db.connection import get_connection
    from models.schemas import UserSkillInferred

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                cs.skill_name,
                CASE MAX(CASE cs.skill_level
                    WHEN 'advanced' THEN 3
                    WHEN 'intermediate' THEN 2
                    WHEN 'beginner' THEN 1
                    ELSE 0
                END)
                    WHEN 3 THEN 'advanced'
                    WHEN 2 THEN 'intermediate'
                    WHEN 1 THEN 'beginner'
                    ELSE 'unknown'
                END AS skill_level,
                MAX(cs.skill_confidence) AS skill_confidence,
                cs.topic_name
            FROM course_completions cc
            JOIN course_skills cs ON cc.course_id = cs.course_id
            WHERE cc.portal_id = ?
              AND cc.completion_status = 'Completed'
            GROUP BY cs.skill_name
            ORDER BY cs.skill_name
            """,
            (portal_id,),
        ).fetchall()

        logger.info(
            "Inferred %d skills for portal_id %s", len(rows), portal_id
        )

        return [
            UserSkillInferred(
                skill_name=row["skill_name"],
                skill_level=row["skill_level"],
                skill_confidence=row["skill_confidence"],
                topic_name=row["topic_name"],
            )
            for row in rows
        ]
    finally:
        conn.close()
