"""Shared sandbox-write guard for the v2.3 adaptive data loop.

``assert_sandbox_write_target()`` is the ONE DRY definition used by both
``scripts/seed_demand_log.py`` (Plan 18-01) and ``scripts/coverage_agent.py``
(Plan 18-03).  Import it directly — no lazy fallback, no re-definition.

Design rationale (REVIEW ROUND-2 H3 refinement + MEDIUM-2):
  The pass condition is the live ``SELECT current_database()`` result on the
  ACTUAL write connection, NOT equality to the env-var SANDBOX_DATABASE_URL's
  parsed dbname.  A mis-set env var that points at prod can therefore never
  whitelist a prod write.
"""

from __future__ import annotations

from contextlib import nullcontext

from app.db import get_conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_SANDBOX_NAME = "city_concierge_sandbox"
KNOWN_PROD_NAMES = frozenset({"city_concierge"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_sandbox_write_target(conn=None) -> None:
    """Raise unless the active write connection targets the sandbox DB.

    Runs ``SELECT current_database()`` against the write connection and
    requires the result to be the known sandbox name
    ``city_concierge_sandbox`` OR a name that CONTAINS the substring
    ``sandbox`` AND is NOT a known-prod name.

    Args:
        conn: An already-open connection to run the check against.  When
            ``None``, a connection is obtained from the shared pool via
            ``get_conn()`` (the same pool target that writes use).

    Raises:
        RuntimeError: When the live ``current_database()`` value is not an
            accepted sandbox name, with the offending dbname in the message.
    """
    ctx = nullcontext(conn) if conn is not None else get_conn()
    with ctx as write_conn, write_conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        row = cur.fetchone()
    live_dbname: str = row[0] if row else ""
    require_sandbox(live_dbname)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def require_sandbox(live_dbname: str) -> None:
    """Raise RuntimeError unless *live_dbname* is an accepted sandbox name.

    Pass condition (in priority order):
    1. ``live_dbname == KNOWN_SANDBOX_NAME`` (canonical sandbox)
    2. ``"sandbox" in live_dbname`` AND ``live_dbname not in KNOWN_PROD_NAMES``
       (configurable-name support for e.g. "my_sandbox_db")

    The decision is SOLELY based on *live_dbname* — the caller must supply
    the value returned by ``SELECT current_database()`` on the real write
    connection.  No env-var equality check is performed here.
    """
    if live_dbname == KNOWN_SANDBOX_NAME:
        return
    if "sandbox" in live_dbname and live_dbname not in KNOWN_PROD_NAMES:
        return
    raise RuntimeError(
        f"assert_sandbox_write_target: refusing to write — live database is "
        f"{live_dbname!r}, which is not an accepted sandbox name "
        f"(expected {KNOWN_SANDBOX_NAME!r} or a name containing 'sandbox'). "
        f"Set DATABASE_URL / SANDBOX_DATABASE_URL to the sandbox DB and retry."
    )
