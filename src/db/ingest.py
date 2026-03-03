"""Data ingestion pipeline: Excel files to SQLite.

Provides functions to load courses, users, and completions from Excel
files into the corresponding SQLite tables with data cleaning,
deduplication, and idempotent upserts (INSERT OR REPLACE).
"""

import datetime
import logging
from pathlib import Path

import pandas as pd

from db.connection import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column rename mappings: Excel header -> SQLite column
# ---------------------------------------------------------------------------

_COURSE_COLUMNS = {
    "Course ID": "course_id",
    "Course Full Name": "course_name",
    "summary": "summary_text",
}

_USER_COLUMNS = {
    "Portal ID": "portal_id",
    "Training Goal": "training_goal",
}

_COMPLETION_COLUMNS = {
    "Course ID": "course_id",
    "Portal ID": "portal_id",
    "Enrolment Date": "enrolment_date",
    "Completion Status": "completion_status",
    "Completed Date": "completed_date",
}

# Schema column lists (for selecting only columns that go into the DB)
_USER_SCHEMA_COLS = [
    "portal_id",
    "grade",
    "employee_type",
    "training_goal",
    "country",
    "emp_practise",
    "manager_portalid",
    "hireddate",
]

_COMPLETION_SCHEMA_COLS = [
    "portal_id",
    "course_id",
    "enrolment_date",
    "completion_status",
    "completed_date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_python(val):
    """Convert a single value to a native Python type safe for sqlite3.

    Handles: pandas NaN/NaT -> None, numpy int/float -> native int/float,
    pandas Timestamp -> ISO string.
    """
    if pd.isna(val):
        return None

    # numpy integer types (e.g. numpy.int64)
    import numpy as np

    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()

    return val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_courses(db_path: Path | None = None, data_dir: Path | None = None) -> int:
    """Load courses from Course_Master_List.xlsx into the courses table.

    Uses an upsert that preserves processed_by_llm and prereqs_extracted flags
    so that re-running ingest does not trigger re-extraction of already-processed
    courses.

    catalog_version is read from the Excel file if a 'Catalog Version' column
    exists; otherwise defaults to today's ISO date as the batch version.

    Args:
        db_path: Path to SQLite DB. If None, uses config.DB_PATH via get_connection.
        data_dir: Directory containing Excel files. If None, uses config.SAMPLE_DATA_DIR.

    Returns:
        Number of rows upserted.
    """
    if data_dir is None:
        from config import SAMPLE_DATA_DIR

        data_dir = SAMPLE_DATA_DIR

    df = pd.read_excel(data_dir / "Course_Master_List.xlsx")
    df = df.rename(columns=_COURSE_COLUMNS)

    # Clean whitespace on course names
    df["course_name"] = df["course_name"].str.strip()

    # Resolve catalog_version: use Excel column if present, else today's date
    version_col = next(
        (c for c in df.columns if c.lower() in ("catalog version", "catalog_version", "version")),
        None,
    )
    if version_col:
        df["catalog_version"] = df[version_col].astype(str)
    else:
        df["catalog_version"] = datetime.date.today().isoformat()

    # Select only schema columns
    df = df[["course_id", "course_name", "summary_text", "catalog_version"]]

    # Convert to native Python types
    records = [
        tuple(_to_python(val) for val in row) for row in df.itertuples(index=False)
    ]

    conn = get_connection(db_path)
    try:
        # ON CONFLICT preserves processing flags (processed_by_llm, prereqs_extracted,
        # processed_at) so re-running ingest never resets extraction state.
        conn.executemany(
            "INSERT INTO courses (course_id, course_name, summary_text, catalog_version) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(course_id) DO UPDATE SET "
            "    course_name = excluded.course_name, "
            "    summary_text = excluded.summary_text, "
            "    catalog_version = excluded.catalog_version",
            records,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("Loaded %d courses", len(records))
    return len(records)


def load_users(db_path: Path | None = None, data_dir: Path | None = None) -> int:
    """Load users from User_Master_List.xlsx into the users table.

    Deduplicates on portal_id (keeps last occurrence). Drops extra columns
    not in the schema. Converts NaT dates to None and float manager_portalid
    to int.

    Args:
        db_path: Path to SQLite DB. If None, uses config.DB_PATH via get_connection.
        data_dir: Directory containing Excel files. If None, uses config.SAMPLE_DATA_DIR.

    Returns:
        Number of rows inserted (after deduplication).
    """
    if data_dir is None:
        from config import SAMPLE_DATA_DIR

        data_dir = SAMPLE_DATA_DIR

    df = pd.read_excel(data_dir / "User_Master_List.xlsx")
    df = df.rename(columns=_USER_COLUMNS)

    # Deduplicate on portal_id (keep last)
    before = len(df)
    df = df.drop_duplicates(subset=["portal_id"], keep="last")
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d duplicate user rows", dropped)

    # Select only schema columns
    df = df[_USER_SCHEMA_COLS]

    # Convert to native Python types
    records = [
        tuple(_to_python(val) for val in row) for row in df.itertuples(index=False)
    ]

    conn = get_connection(db_path)
    try:
        # ON CONFLICT preserves learning_path_created and path_updated_at so that
        # re-running ingest does not erase existing user plan state.
        conn.executemany(
            "INSERT INTO users "
            "(portal_id, grade, employee_type, training_goal, country, "
            "emp_practise, manager_portalid, hireddate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(portal_id) DO UPDATE SET "
            "    grade = excluded.grade, "
            "    employee_type = excluded.employee_type, "
            "    training_goal = excluded.training_goal, "
            "    country = excluded.country, "
            "    emp_practise = excluded.emp_practise, "
            "    manager_portalid = excluded.manager_portalid, "
            "    hireddate = excluded.hireddate",
            records,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("Loaded %d users (dropped %d duplicates)", len(records), dropped)
    return len(records)


def load_completions(
    db_path: Path | None = None, data_dir: Path | None = None
) -> int:
    """Load completions from Completion_Data.xlsx into course_completions table.

    Deduplicates on composite PK (portal_id, course_id, enrolment_date),
    preferring rows with a non-null completion_status and the latest
    completed_date.

    Args:
        db_path: Path to SQLite DB. If None, uses config.DB_PATH via get_connection.
        data_dir: Directory containing Excel files. If None, uses config.SAMPLE_DATA_DIR.

    Returns:
        Number of rows inserted (after deduplication).
    """
    if data_dir is None:
        from config import SAMPLE_DATA_DIR

        data_dir = SAMPLE_DATA_DIR

    df = pd.read_excel(data_dir / "Completion_Data.xlsx")
    df = df.rename(columns=_COMPLETION_COLUMNS)

    # Deduplicate composite PK: prefer completed status, latest date
    df = df.sort_values(
        ["completion_status", "completed_date"],
        ascending=[True, False],
        na_position="last",
    )
    before = len(df)
    df = df.drop_duplicates(
        subset=["portal_id", "course_id", "enrolment_date"], keep="first"
    )
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d duplicate completion rows", dropped)

    # Select only schema columns
    df = df[_COMPLETION_SCHEMA_COLS]

    # Convert to native Python types
    records = [
        tuple(_to_python(val) for val in row) for row in df.itertuples(index=False)
    ]

    conn = get_connection(db_path)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO course_completions "
            "(portal_id, course_id, enrolment_date, completion_status, completed_date) "
            "VALUES (?, ?, ?, ?, ?)",
            records,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("Loaded %d completions (dropped %d duplicates)", len(records), dropped)
    return len(records)


def ingest_all(
    db_path: Path | None = None, data_dir: Path | None = None
) -> dict[str, int]:
    """Load all Excel data into SQLite in FK-safe order.

    Loads courses and users first (no FK dependencies), then completions
    (references both courses and users via foreign keys).

    Args:
        db_path: Path to SQLite DB. If None, uses config.DB_PATH via get_connection.
        data_dir: Directory containing Excel files. If None, uses config.SAMPLE_DATA_DIR.

    Returns:
        Dict with row counts: {"courses": N, "users": N, "completions": N}.
    """
    courses = load_courses(db_path, data_dir)
    users = load_users(db_path, data_dir)
    completions = load_completions(db_path, data_dir)

    result = {"courses": courses, "users": users, "completions": completions}
    logger.info(
        "Ingestion complete: %d courses, %d users, %d completions",
        courses,
        users,
        completions,
    )
    return result
