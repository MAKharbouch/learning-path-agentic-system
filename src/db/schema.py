"""SQLite schema definition and initialization.

Contains all 7 CREATE TABLE statements and 7 CREATE INDEX statements
for the learning path database. Schema initialization is idempotent.
"""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT NOT NULL,
    summary_text TEXT,
    catalog_version TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    processed_by_llm INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    portal_id INTEGER PRIMARY KEY,
    grade INTEGER,
    employee_type TEXT,
    training_goal REAL,
    country TEXT,
    emp_practise TEXT,
    manager_portalid INTEGER,
    hireddate TEXT,
    learning_path_created INTEGER NOT NULL DEFAULT 0,
    path_updated_at TEXT
);

CREATE TABLE IF NOT EXISTS course_completions (
    portal_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    enrolment_date TEXT,
    completion_status TEXT,
    completed_date TEXT,
    PRIMARY KEY (portal_id, course_id, enrolment_date),
    FOREIGN KEY (portal_id) REFERENCES users(portal_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE IF NOT EXISTS course_skills (
    course_id INTEGER NOT NULL,
    skill_name TEXT NOT NULL,
    skill_level TEXT,
    skill_confidence REAL,
    topic_name TEXT,
    topic_weight REAL,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE IF NOT EXISTS course_prerequisites (
    course_id INTEGER NOT NULL,
    prereq_name TEXT NOT NULL,
    relevance_strength REAL,
    reason_short TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE IF NOT EXISTS learning_path_runs (
    path_run_id TEXT PRIMARY KEY,
    portal_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    training_goal_hours REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    FOREIGN KEY (portal_id) REFERENCES users(portal_id)
);

CREATE TABLE IF NOT EXISTS learning_path_courses (
    path_run_id TEXT NOT NULL,
    course_id INTEGER NOT NULL,
    course_name TEXT,
    targeted_skills TEXT,
    targeted_topic TEXT,
    targeted_level TEXT,
    phase TEXT,
    course_order INTEGER,
    estimated_duration_hours REAL,
    reasoning_summary TEXT,
    reasoning_object TEXT,
    PRIMARY KEY (path_run_id, course_id),
    FOREIGN KEY (path_run_id) REFERENCES learning_path_runs(path_run_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_completions_portal ON course_completions(portal_id);
CREATE INDEX IF NOT EXISTS idx_completions_course ON course_completions(course_id);
CREATE INDEX IF NOT EXISTS idx_skills_course ON course_skills(course_id);
CREATE INDEX IF NOT EXISTS idx_skills_name ON course_skills(skill_name);
CREATE INDEX IF NOT EXISTS idx_prereqs_course ON course_prerequisites(course_id);
CREATE INDEX IF NOT EXISTS idx_path_runs_portal ON learning_path_runs(portal_id);
CREATE INDEX IF NOT EXISTS idx_path_courses_run ON learning_path_courses(path_run_id);
"""


def _add_prereqs_extracted_column(conn) -> None:
    """Add prereqs_extracted column to courses table if it doesn't exist.

    This migration supports the prerequisite extraction tracking feature.
    Uses try/except for idempotency -- safe to run multiple times.
    """
    try:
        conn.execute(
            "ALTER TABLE courses ADD COLUMN prereqs_extracted "
            "INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            pass  # Column already exists, migration already applied
        else:
            raise


def _add_prereq_confidence_column(conn) -> None:
    """Add prereq_confidence column to course_prerequisites table if it doesn't exist.

    Brings prerequisite extractions to parity with skill extractions, which
    already store skill_confidence. Uses try/except for idempotency.
    """
    try:
        conn.execute(
            "ALTER TABLE course_prerequisites ADD COLUMN prereq_confidence REAL"
        )
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            pass  # Column already exists
        else:
            raise


def init_db(db_path: Path | None = None) -> None:
    """Create all tables and indexes if they don't exist. Idempotent.

    Args:
        db_path: Path to the SQLite database file. If None, uses config.DB_PATH.
    """
    from db.connection import get_connection

    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        _add_prereqs_extracted_column(conn)
        _add_prereq_confidence_column(conn)
        db_location = db_path or "default"
        print(f"Database initialized at {db_location}")
    finally:
        conn.close()


def get_table_names(db_path: Path | None = None) -> list[str]:
    """Return a sorted list of all table names in the database.

    Args:
        db_path: Path to the SQLite database file. If None, uses config.DB_PATH.

    Returns:
        Sorted list of table names.
    """
    from db.connection import get_connection

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()
