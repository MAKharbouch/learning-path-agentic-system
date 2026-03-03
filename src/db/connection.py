"""Database connection factory for SQLite.

Returns configured connections with WAL mode, foreign keys enabled,
and sqlite3.Row factory for dict-like access.
"""

import sqlite3
from pathlib import Path


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a configured SQLite connection.

    Args:
        db_path: Path to the SQLite database file. If None, uses config.DB_PATH.

    Returns:
        A sqlite3.Connection with WAL mode, foreign keys ON, and Row factory.
    """
    if db_path is None:
        # Import at function level to avoid circular imports
        from config import DB_PATH

        db_path = DB_PATH

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
