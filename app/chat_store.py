"""
app/chat_store.py — Cloud-native Postgres chat history store.

Replaces the original SQLite implementation.  Uses the same public API so
all callers (api/routes.py, api/server.py) require no changes.

Schema (Postgres):
    messages(id BIGSERIAL, thread_id TEXT, role TEXT, content TEXT, timestamp TIMESTAMPTZ)
"""

import os
from datetime import datetime, timezone
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

# ---------------------------------------------------------------------------
# Connection string — comes from the environment (set in .env via load_dotenv)
# ---------------------------------------------------------------------------
_DB_URI: str | None = None


def _get_uri() -> str:
    global _DB_URI
    if _DB_URI is None:
        uri = os.environ.get("SUPABASE_DB_URI")
        if not uri:
            raise EnvironmentError(
                "'SUPABASE_DB_URI' is not set. "
                "Add it to your .env file (e.g. postgresql://user:pass@host:5432/dbname)."
            )
        _DB_URI = uri
    return _DB_URI


@contextmanager
def _get_conn():
    """Yield a Postgres connection with dict-style rows, auto-closed on exit."""
    conn = psycopg.connect(_get_uri(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create the messages table and index if they don't already exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id        BIGSERIAL    PRIMARY KEY,
                thread_id TEXT         NOT NULL,
                role      TEXT         NOT NULL CHECK(role IN ('user', 'assistant')),
                content   TEXT         NOT NULL,
                timestamp TIMESTAMPTZ  NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_thread
            ON chat_messages(thread_id, id)
        """)


def save_message(thread_id: str, role: str, content: str) -> None:
    """Append a single message to the history."""
    ts = datetime.now(timezone.utc)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content, timestamp) "
            "VALUES (%s, %s, %s, %s)",
            (thread_id, role, content, ts),
        )


def get_history(thread_id: str) -> list[dict]:
    """Return all messages for a thread, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp::text FROM chat_messages "
            "WHERE thread_id = %s ORDER BY id",
            (thread_id,),
        ).fetchall()
    return list(rows)


def list_threads() -> list[dict]:
    """Return a summary of all threads (thread_id, message_count, last_active)."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                thread_id,
                COUNT(*)            AS message_count,
                MAX(timestamp)::text AS last_active
            FROM chat_messages
            GROUP BY thread_id
            ORDER BY MAX(timestamp) DESC
        """).fetchall()
    return list(rows)
