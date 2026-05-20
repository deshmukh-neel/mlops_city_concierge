"""Closure-aware itinerary swap node.

Sits between `retime` and END in the agent graph. Detects committed stops
that will be closed at their planned arrival time, deterministically swaps
in walking-distance alternatives of the same category where possible,
escalates to a single user question per turn when not, and remembers every
closure event so refinement turns never re-suggest the same closed place.

See docs/superpowers/specs/2026-05-19-closure-aware-itinerary-swap-design.md
for the architecture rationale and contract details.
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor

from app.agent.state import Stop
from app.db import get_conn

logger = logging.getLogger(__name__)


def _execute_closure_query(
    place_ids: list[str],
    arrivals: list[Any],
) -> dict[str, bool]:
    """One SQL round-trip via `place_is_open`. Returns {place_id: is_open}.

    Mirrors `temporal_coherence` at app/agent/critique/checks.py:69-79; that
    pattern unnests the two arrays in lockstep so an N-stop itinerary is one
    round-trip, not N. Stops missing from the result default to open (no row =
    not in places_raw, which the critique pipeline scores separately).
    """
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT pr.place_id,
                   place_is_open(pr.regular_opening_hours, t.arrival) AS is_open
              FROM unnest(%s::text[], %s::timestamptz[]) AS t(place_id, arrival)
              JOIN places_raw pr ON pr.place_id = t.place_id
            """,
            [place_ids, arrivals],
        )
        return {row["place_id"]: bool(row["is_open"]) for row in cur.fetchall()}


def _per_stop_closure_status(stops: list[Stop]) -> list[bool]:
    """Return [is_closed_at_arrival] per stop, in the same order as `stops`.

    True means "we know this stop is closed at its planned arrival time."
    Stops without an arrival_time, stops missing from places_raw, and full
    DB failures all return False (fail-open — matches checks.py:200-205
    precedent so a DB blip doesn't block the /chat response).
    """
    if not stops:
        return []
    checkable = [(i, s) for i, s in enumerate(stops) if s.arrival_time is not None]
    if not checkable:
        return [False] * len(stops)
    place_ids = [s.place_id for _, s in checkable]
    arrivals = [s.arrival_time for _, s in checkable]
    try:
        is_open_by_id = _execute_closure_query(place_ids, arrivals)
    except Exception as e:  # noqa: BLE001
        logger.warning("closure_swap.db_error: %s", e)
        return [False] * len(stops)
    out = [False] * len(stops)
    for i, stop in checkable:
        # No row -> default to open (matches temporal_coherence semantics).
        is_open = is_open_by_id.get(stop.place_id, True)
        out[i] = not is_open
    return out


__all__ = [
    "_execute_closure_query",
    "_per_stop_closure_status",
]
