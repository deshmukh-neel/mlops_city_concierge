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
from pydantic import BaseModel

from app.agent.planning import haversine_m
from app.agent.state import ClosureContext, Stop
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


class CandidateMatch(BaseModel):
    """Internal record returned by candidate-search helpers."""

    stop: Stop
    distance_m: float
    family_match_score: float
    route_impact_score: float
    total_score: float


# Per-leg walking budget (meters) used as the cutoff for the silent-swap
# path. ~500m at 80 m/min ≈ a 6-minute walk — close enough that swapping
# doesn't change the user's experience materially. Anything beyond this
# escalates to a user question.
_WALKING_DISTANCE_BUDGET_M: int = 500

# Citywide radius used by the fallback search. SF fits comfortably inside
# 30 km from any anchor in the city; `nearby()` requires an explicit
# `radius_m: int` so we pass a large constant rather than introducing a
# separate citywide function.
_CITYWIDE_RADIUS_M: int = 30_000


def _resolve_insert_position(
    closure: ClosureContext,
    stops: list[Stop],
) -> int:
    """Where in `stops` should we insert the proposed alternative?

    Priority rules (matches the ClosureContext docstring in state.py):
      1) insert_after_place_id present in stops -> that index + 1
      2) else insert_before_place_id present in stops -> that index
      3) else stop_index_hint, clamped to [0, len(stops)]
    """
    by_id = {s.place_id: i for i, s in enumerate(stops)}
    if closure.insert_after_place_id and closure.insert_after_place_id in by_id:
        return by_id[closure.insert_after_place_id] + 1
    if closure.insert_before_place_id and closure.insert_before_place_id in by_id:
        return by_id[closure.insert_before_place_id]
    return max(0, min(closure.stop_index_hint, len(stops)))


def _score_candidate(
    candidate: Stop,
    closed_stop: Stop,
    prev_stop: Stop | None,
    next_stop: Stop | None,
    *,
    family_match: bool,
) -> float:
    """Combined score: higher is better.

    Two components, summed equally weighted:
      - family_match_score: 1.0 if the candidate is in the same family as
        the closed stop, else 0.0
      - route_impact_score: 1 - (haversine prev->candidate + candidate->next)
        / 2000, clamped to [0, 1]. Inside the walking radius the family
        bonus dominates any plausible route delta.
    `closed_stop` is currently informational (the prev/next pair carries the
    geometry); it's threaded through so future scoring tweaks (e.g.
    rating/category similarity) have access to it without a signature change.
    """
    _ = closed_stop  # reserved for future heuristics (rating/category)
    fam = 1.0 if family_match else 0.0
    total_dist_m = 0.0
    if (
        prev_stop is not None
        and prev_stop.latitude is not None
        and prev_stop.longitude is not None
        and candidate.latitude is not None
        and candidate.longitude is not None
    ):
        total_dist_m += haversine_m(
            (prev_stop.latitude, prev_stop.longitude),
            (candidate.latitude, candidate.longitude),
        )
    if (
        next_stop is not None
        and next_stop.latitude is not None
        and next_stop.longitude is not None
        and candidate.latitude is not None
        and candidate.longitude is not None
    ):
        total_dist_m += haversine_m(
            (candidate.latitude, candidate.longitude),
            (next_stop.latitude, next_stop.longitude),
        )
    # Linear penalty: 0m -> 1.0, 1000m -> 0.5, 2000m+ -> 0
    route = max(0.0, 1.0 - total_dist_m / 2000.0)
    return fam + route


__all__ = [
    "CandidateMatch",
    "_execute_closure_query",
    "_per_stop_closure_status",
    "_resolve_insert_position",
    "_score_candidate",
]
