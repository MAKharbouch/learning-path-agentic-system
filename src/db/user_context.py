"""User context loading for the online pipeline.

Loads user profile, completed courses, and inferred skills into a single
UserContext model.
"""

import logging

logger = logging.getLogger(__name__)


def get_user_context(portal_id, db_path=None):
    """Load complete user context for learning path generation.

    Combines user profile data, completed courses, and inferred skills
    into a single UserContext model for the online pipeline.

    Args:
        portal_id: The user's portal ID.
        db_path: Optional database path. Uses config default if None.

    Returns:
        UserContext instance with profile, completions, and skills.

    Raises:
        ValueError: If the portal_id does not exist in the users table.
    """
    from db.connection import get_connection
    from models.schemas import UserContext

    conn = get_connection(db_path)
    try:
        # Load user profile
        row = conn.execute(
            "SELECT * FROM users WHERE portal_id = ?",
            (portal_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"User {portal_id} not found")

        # Load completed courses
        completions = conn.execute(
            """
            SELECT cc.course_id, c.course_name
            FROM course_completions cc
            JOIN courses c USING (course_id)
            WHERE cc.portal_id = ?
              AND cc.completion_status = 'Completed'
            """,
            (portal_id,),
        ).fetchall()

        completed_course_ids = [r["course_id"] for r in completions]
        completed_course_names = [r["course_name"] for r in completions]
    finally:
        conn.close()

    # Load inferred skills (manages its own connection)
    from db.user_skills import get_user_skills

    inferred_skills = get_user_skills(portal_id, db_path)

    logger.info(
        "Loaded context for portal_id %s: %d completions, %d skills",
        portal_id,
        len(completed_course_ids),
        len(inferred_skills),
    )

    return UserContext(
        portal_id=row["portal_id"],
        grade=row["grade"],
        employee_type=row["employee_type"],
        training_goal=row["training_goal"],
        country=row["country"],
        completed_course_ids=completed_course_ids,
        completed_course_names=completed_course_names,
        inferred_skills=inferred_skills,
    )
