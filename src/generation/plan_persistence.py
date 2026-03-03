"""Persist a generated learning plan to SQLite.

Writes to learning_path_runs (one row per plan) and
learning_path_courses (one row per course in the plan).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_plan_to_db(
    path_run_id: str,
    portal_id: int,
    training_goal_hours: float | None,
    plan_response,
    db_path: Path | None = None,
) -> None:
    """Persist a LearningPlanResponse to the database.

    Creates one learning_path_runs row and one learning_path_courses row
    per PlannedCourse. Status is set to 'draft' (Phase 8 validates and
    may update to 'validated' or 'failed').

    Args:
        path_run_id: UUID4 string for this plan run.
        portal_id: The user's portal ID.
        training_goal_hours: User's training hour budget (may be None).
        plan_response: A LearningPlanResponse from the LLM.
        db_path: Path to SQLite database. If None, uses config default.
    """
    from db.connection import get_connection

    conn = get_connection(db_path)
    try:
        # Insert the plan run header
        conn.execute(
            """
            INSERT INTO learning_path_runs
                (path_run_id, portal_id, training_goal_hours, status)
            VALUES (?, ?, ?, 'draft')
            """,
            (path_run_id, portal_id, training_goal_hours),
        )

        # Insert each planned course
        for course in plan_response.courses:
            conn.execute(
                """
                INSERT INTO learning_path_courses
                    (path_run_id, course_id, course_name, targeted_skills,
                     targeted_topic, targeted_level, phase, course_order,
                     estimated_duration_hours, reasoning_summary, reasoning_object)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path_run_id,
                    course.course_id,
                    course.course_name,
                    ", ".join(course.targeted_skills),
                    course.targeted_topic,
                    course.targeted_level,
                    course.phase,
                    course.course_order,
                    course.estimated_duration_hours,
                    course.reasoning_summary,
                    json.dumps(course.model_dump()),
                ),
            )

        conn.commit()

        logger.info(
            "Saved plan %s with %d courses for portal_id %s",
            path_run_id,
            len(plan_response.courses),
            portal_id,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_plan_status(path_run_id: str, status: str, db_path: Path | None = None) -> None:
    """Update the status of a learning_path_runs record.

    Args:
        path_run_id: The plan run UUID to update.
        status: New status ('validated' or 'failed').
        db_path: Path to SQLite database. If None, uses config default.
    """
    from db.connection import get_connection

    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE learning_path_runs SET status = ? WHERE path_run_id = ?",
            (status, path_run_id),
        )
        conn.commit()
        logger.info("Updated plan %s status to '%s'", path_run_id, status)
    finally:
        conn.close()
