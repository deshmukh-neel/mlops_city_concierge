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
from app.agent.state import ClosureContext, ItineraryState, Stop
from app.db import get_conn
from app.tools.filters import SearchFilters, family_of
from app.tools.retrieval import PlaceHit
from app.tools.retrieval import nearby as _nearby_search  # aliased for test patching

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


_VALID_FAMILIES = frozenset({"dessert", "bar", "restaurant", "cafe"})


def _resolve_anchor(state: ItineraryState, closed_stop: Stop) -> str | None:
    """Pick a stable anchor place_id to search around when looking for an
    alternative to the closed stop.

    Prefer the previous stop; if closed_stop is at index 0, fall back to the
    next stop; if neither exists, fall back to the closed stop itself (Google
    returns same-coords neighbors). Returns None only when state.stops is
    empty (defensive — caller guards against this).
    """
    try:
        idx = state.stops.index(closed_stop)
    except ValueError:
        return closed_stop.place_id or None
    if idx > 0:
        return state.stops[idx - 1].place_id
    if idx + 1 < len(state.stops):
        return state.stops[idx + 1].place_id
    return closed_stop.place_id or None


def _candidates_to_matches(
    candidates: list[PlaceHit],
    closed_stop: Stop,
    state: ItineraryState,
) -> list[CandidateMatch]:
    """Score each candidate and sort descending. Family match is computed
    against the closed stop's primary_type. Each candidate becomes a Stop
    inheriting the closed stop's arrival_time + planned_duration_min so the
    chain math stays consistent post-swap.
    """
    closed_family = family_of(closed_stop.primary_type)
    try:
        idx = state.stops.index(closed_stop)
    except ValueError:
        idx = len(state.stops)
    prev_stop = state.stops[idx - 1] if idx > 0 else None
    next_stop = state.stops[idx + 1] if idx + 1 < len(state.stops) else None

    matches: list[CandidateMatch] = []
    for c in candidates:
        candidate_stop = Stop(
            place_id=c.place_id,
            name=c.name,
            address=c.formatted_address,
            rating=c.rating,
            primary_type=c.primary_type,
            latitude=c.latitude,
            longitude=c.longitude,
            arrival_time=closed_stop.arrival_time,
            planned_duration_min=closed_stop.planned_duration_min,
            rationale=f"Walking-distance alternative for {closed_stop.name}",
            source=c.source,
        )
        candidate_family = family_of(c.primary_type)
        family_match = candidate_family is not None and candidate_family == closed_family
        score = _score_candidate(
            candidate_stop, closed_stop, prev_stop, next_stop, family_match=family_match
        )
        matches.append(
            CandidateMatch(
                stop=candidate_stop,
                distance_m=c.dist_m if c.dist_m is not None else 0.0,
                family_match_score=1.0 if family_match else 0.0,
                route_impact_score=score - (1.0 if family_match else 0.0),
                total_score=score,
            )
        )
    matches.sort(key=lambda m: m.total_score, reverse=True)
    return matches


def _excluded_place_ids_from_state(
    state: ItineraryState,
    extra: list[str] | None = None,
) -> list[str]:
    """All place_ids the swap node must not re-propose: current stops + every
    closure_context entry's source place_id + extras.

    Every outcome contributes per spec — once recorded, never re-suggested.
    Sorted for deterministic SQL params (helps test assertions and pg log
    correlation).
    """
    excluded = {s.place_id for s in state.stops}
    excluded.update(entry.place_id for entry in state.closure_context)
    if extra:
        excluded.update(extra)
    return sorted(excluded)


def _try_walking_distance_swap(
    state: ItineraryState,
    closure: ClosureContext,
    *,
    anchor_place_id: str,
) -> CandidateMatch | None:
    """Search within `_WALKING_DISTANCE_BUDGET_M` for an alternative of the
    same family that's open at the closed stop's attempted_arrival.

    Returns the highest-scoring match or None if no candidates fit. DB errors
    return None and log a warning (fail-open: a DB blip won't block the
    response, it just causes escalation to the user-question path).
    """
    closed_stop = next((s for s in state.stops if s.place_id == closure.place_id), None)
    if closed_stop is None:
        return None
    if closure.family not in _VALID_FAMILIES:
        # Without a resolved family we can't do a category-matched search;
        # caller escalates to the user-question path.
        return None
    excluded = _excluded_place_ids_from_state(state)
    filters = SearchFilters(
        primary_type_family=closure.family,  # type: ignore[arg-type]
        excluded_place_ids=excluded,
        open_at=closure.attempted_arrival,
    )
    try:
        candidates = _nearby_search(
            place_id=anchor_place_id,
            radius_m=_WALKING_DISTANCE_BUDGET_M,
            filters=filters,
            k=8,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("closure_swap.db_error during walking-distance search: %s", e)
        return None
    if not candidates:
        return None
    matches = _candidates_to_matches(candidates, closed_stop, state)
    if not matches:
        return None
    return matches[0]


def _try_any_distance_search(
    state: ItineraryState,
    closure: ClosureContext,
    *,
    anchor_place_id: str,
) -> CandidateMatch | None:
    """Citywide fallback — used only to populate the user-facing question's
    proposed_alternative when the walking-distance pass failed.

    Uses `_CITYWIDE_RADIUS_M` (30 km, covers all of SF). Same family +
    exclusion rules as walking-distance.
    """
    closed_stop = next((s for s in state.stops if s.place_id == closure.place_id), None)
    if closed_stop is None:
        return None
    if closure.family not in _VALID_FAMILIES:
        return None
    excluded = _excluded_place_ids_from_state(state)
    filters = SearchFilters(
        primary_type_family=closure.family,  # type: ignore[arg-type]
        excluded_place_ids=excluded,
        open_at=closure.attempted_arrival,
    )
    try:
        candidates = _nearby_search(
            place_id=anchor_place_id,
            radius_m=_CITYWIDE_RADIUS_M,
            filters=filters,
            k=5,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("closure_swap.db_error during any-distance search: %s", e)
        return None
    if not candidates:
        return None
    matches = _candidates_to_matches(candidates, closed_stop, state)
    if not matches:
        return None
    return matches[0]


__all__ = [
    "CandidateMatch",
    "_execute_closure_query",
    "_per_stop_closure_status",
    "_resolve_anchor",
    "_resolve_insert_position",
    "_score_candidate",
    "_try_any_distance_search",
    "_try_walking_distance_swap",
]
