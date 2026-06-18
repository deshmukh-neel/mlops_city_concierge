"""Deterministic demand-log seed helper for sandbox testing.

Usage (command-line):
    DATABASE_URL=$SANDBOX_DATABASE_URL python scripts/seed_demand_log.py

This script:
1. Calls ``assert_sandbox_write_target()`` BEFORE any INSERT so seed rows
   can never land in a non-sandbox DB (REVIEW HIGH-3 + ROUND-2 MEDIUM-2).
2. Inserts a small deterministic set of catalog-valid demand rows into
   ``user_query_log`` so the demand miner has real data to mine.
3. Includes the (``Outer Sunset``, ``vietnamese``) overlap row so a
   functional test downstream can confirm the gap path fires (D-01, D-04).
"""

from __future__ import annotations

import sys
from contextlib import nullcontext

from app.db import get_conn
from scripts.ingest_places_sf import CUISINES, NEIGHBORHOODS
from scripts.sandbox_guard import assert_sandbox_write_target

# ---------------------------------------------------------------------------
# Fixture rows — every message contains a NEIGHBORHOODS member, every
# requested_primary_types entry maps to a CUISINES member via
# <type>.replace(" Restaurant", "").lower().
# ---------------------------------------------------------------------------

_SEED_ROWS: list[dict] = [
    {
        "message": "Vietnamese restaurants in Outer Sunset",
        "requested_primary_types": ["Vietnamese Restaurant"],
        "num_stops": 3,
        "rag_label": "itinerary",
        "session_id": "seed-001",
    },
    {
        "message": "Ethiopian food near Mission District",
        "requested_primary_types": ["Ethiopian Restaurant"],
        "num_stops": 2,
        "rag_label": "itinerary",
        "session_id": "seed-002",
    },
    {
        "message": "Nepalese restaurants in Inner Sunset",
        "requested_primary_types": ["Nepalese Restaurant"],
        "num_stops": 2,
        "rag_label": "itinerary",
        "session_id": "seed-003",
    },
    {
        "message": "Korean food in Outer Richmond",
        "requested_primary_types": ["Korean Restaurant"],
        "num_stops": 3,
        "rag_label": "itinerary",
        "session_id": "seed-004",
    },
    {
        "message": "Vietnamese dinner in Outer Sunset",
        "requested_primary_types": ["Vietnamese Restaurant"],
        "num_stops": 4,
        "rag_label": "itinerary",
        "session_id": "seed-005",
    },
]

# Verify fixture stays inside the catalog at import time (fail-fast).
for _row in _SEED_ROWS:
    for _pt in _row["requested_primary_types"]:
        _key = _pt.replace(" Restaurant", "").replace(" restaurant", "").lower()
        assert _key in CUISINES, f"seed row type {_pt!r} → {_key!r} not in CUISINES"
    assert any(n in _row["message"] for n in NEIGHBORHOODS), (
        f"seed row message {_row['message']!r} contains no NEIGHBORHOODS member"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def seed_demand_rows() -> list[dict]:
    """Return the deterministic catalog-valid demand fixture rows.

    Returns a new copy of the fixture list on each call so callers can
    mutate it without affecting the module-level constant.
    """
    return list(_SEED_ROWS)


def insert_demand_rows(rows: list[dict], conn=None) -> int:
    """Insert *rows* into ``user_query_log`` in the sandbox DB.

    Calls ``assert_sandbox_write_target()`` BEFORE any INSERT.  If the
    guard raises the function propagates the exception and performs zero
    inserts (REVIEW HIGH-3).

    Args:
        rows: List of dicts with keys matching ``user_query_log`` columns.
        conn: Optional already-open psycopg2 connection.  When ``None``,
            a connection is obtained from the shared pool via ``get_conn()``.

    Returns:
        The number of rows inserted.
    """
    # HIGH-3: guard fires first — no inserts if this raises.
    assert_sandbox_write_target(conn=conn)

    ctx = nullcontext(conn) if conn is not None else get_conn()
    with ctx as write_conn, write_conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO user_query_log
                    (message, requested_primary_types, num_stops, rag_label, session_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    row["message"],
                    row["requested_primary_types"],
                    row["num_stops"],
                    row["rag_label"],
                    row["session_id"],
                ],
            )
        write_conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Seed the sandbox ``user_query_log`` from the command line.

    Set ``DATABASE_URL=$SANDBOX_DATABASE_URL`` before running so the
    shared ``get_conn()`` pool targets the sandbox.  The write guard
    enforces this regardless.
    """
    rows = seed_demand_rows()
    count = insert_demand_rows(rows)
    print(f"Inserted {count} demand rows into user_query_log.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
