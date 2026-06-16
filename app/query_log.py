"""Fire-and-forget user query logging for the v2.3 adaptive data loop.

This module is called from FastAPI BackgroundTasks so the INSERT runs AFTER
the ChatResponse is built and returned — it adds 0ms to user-perceived
latency (Phase 17 CONTEXT D-01).

Failure posture (D-04): swallow ALL exceptions and emit a logger.warning
so a DB failure can never surface to the user or leave an unawaited-task
warning. Fail-open, fail-quiet.
"""

from __future__ import annotations

import logging

from .db import get_conn

logger = logging.getLogger(__name__)


def log_user_query(
    *,
    message: str,
    requested_primary_types: list[str],
    num_stops: int | None,
    rag_label: str,
    session_id: str | None = None,
) -> None:
    """Insert one row into user_query_log.

    Called via background_tasks.add_task() — must swallow all exceptions
    so a logging failure never propagates to the /chat response path or
    leaves an unawaited-exception warning (Phase 17 CONTEXT D-04).

    Parameterised INSERT uses %s placeholders so the raw user message is
    never interpolated into the SQL string (T-17-04 SQL-injection guard).
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_query_log
                    (message, requested_primary_types, num_stops, rag_label, session_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [message, requested_primary_types, num_stops, rag_label, session_id],
            )
            conn.commit()
    except Exception:
        logger.warning("log_user_query failed; skipping", exc_info=True)
