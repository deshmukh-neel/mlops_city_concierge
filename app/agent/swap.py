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
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from app.agent.commit import enrich_stops_with_booking
from app.agent.planning import chain_arrival_times, haversine_m
from app.agent.revision import summarize_stops
from app.agent.state import (
    MAX_CLOSURE_CONTEXT_ENTRIES,
    ClosureContext,
    ItineraryState,
    Stop,
)
from app.db import get_conn
from app.tools.directions import route_legs
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


async def _bounded_retime_after_swap(stops: list[Stop]) -> list[Stop]:
    """One extra `route_legs` call after a swap -> re-chain arrival_times.

    Strictly bounded: no recursion, no loop. Called at most once per swap
    node invocation. Falls back to the input stops on any failure (mirrors
    `retime()` in graph.py:319-323).
    """
    coords = [
        (s.latitude, s.longitude)
        for s in stops
        if s.latitude is not None and s.longitude is not None
    ]
    if len(coords) < 2 or len(coords) != len(stops):
        return stops
    try:
        result = await route_legs(coords, mode="walk")
        leg_min = [leg.duration_s / 60 for leg in result.legs]
        retimed = chain_arrival_times(stops, leg_min)
    except Exception as e:  # noqa: BLE001
        logger.warning("closure_swap.retime_failure: %s", e)
        return stops
    return retimed


def _apply_swap(
    state: ItineraryState,
    stop_index: int,
    replacement: Stop,
    leg_durations_min: list[float],
) -> list[Stop]:
    """Replace stops[stop_index] with `replacement` and re-chain arrivals.

    Returns the new stops list. Caller is responsible for substituting it
    into state. Booking enrichment is re-run on the new list so the swapped
    stop's card fields (booking URL, refreshed address) stay accurate.
    """
    new_stops = list(state.stops)
    new_stops[stop_index] = replacement
    if leg_durations_min and new_stops and new_stops[0].arrival_time is not None:
        new_stops = chain_arrival_times(new_stops, leg_durations_min)
    enrich_stops_with_booking(new_stops, state)
    return new_stops


def _promote_pending(
    closure_context: list[ClosureContext],
) -> list[ClosureContext]:
    """If there is no pending entry, promote the first queued one (if any).

    Returns a new list. Caller substitutes it into state.closure_context.
    No-op when a pending entry is already present (v1 surfaces one at a
    time, in arrival order).
    """
    if any(c.outcome == "pending_user_decision" for c in closure_context):
        return list(closure_context)
    promoted = list(closure_context)
    for i, c in enumerate(promoted):
        if c.outcome == "queued_user_decision":
            promoted[i] = c.model_copy(update={"outcome": "pending_user_decision"})
            break
    return promoted


def _miles_from_meters(m: float) -> int:
    """Round to nearest mile for user-facing text. 1609m -> 1mi, 4800m -> 3mi."""
    return round(m / 1609.34)


def _formulate_closure_question(pending: ClosureContext) -> str:
    """User-facing question text for a pending closure decision.

    Two shapes:
      - With proposed_alternative: 'The closest open <family> is <name>,
        about <N> mi (drive/transit). Want it, or pick something else?'
      - Without: 'I couldn't find an open <family> alternative for <name>.
        Want me to skip that stop, or pick a different category?'
    """
    if pending.proposed_alternative is not None:
        distance = pending.proposed_distance_m or 0.0
        miles = _miles_from_meters(distance)
        mode = "drive" if distance > 1500 else "walk/transit"
        return (
            f"{pending.place_name} is closed at the planned arrival time. "
            f"The closest open {pending.family} place is "
            f"{pending.proposed_alternative.name}, about {miles} mi ({mode}). "
            f"Want me to add it, or pick something else?"
        )
    return (
        f"{pending.place_name} is closed at the planned arrival time and I "
        f"couldn't find an open {pending.family} alternative. Want me to "
        f"skip that stop, or pick a different category?"
    )


def _inject_closure_exclusions(
    tool_name: str,
    args: dict[str, Any],
    closure_context: list[ClosureContext],
) -> dict[str, Any]:
    """Merge closure_context place_ids into a tool call's exclusion argument.

    Server-side belt-and-suspenders enforcement so the prompt guidance is an
    optimization, not the only line of defense. Routes by tool name because
    the exclusion argument lives in different places per tool:

      - semantic_search / nearby -> args["filters"]["excluded_place_ids"]
      - kg_traverse              -> args["excluded_place_ids"]  (top-level)
      - anything else            -> no-op

    Returns a NEW args dict with `filters` as a plain dict — NOT a Pydantic
    `SearchFilters` instance. The result must be `json.dumps`-safe because
    the caller stores it (or a copy) inside `AIMessage.tool_calls`, which
    langchain serializes on every subsequent OpenAI API call. A Pydantic
    instance there causes `TypeError: Object of type SearchFilters is not
    JSON serializable` on the next plan() step.

    Every closure_context outcome contributes — auto_swapped through
    pending_user_decision all exclude the source closed place_id. The
    proposed_alternative.place_id is NOT in this set unless that place was
    itself later recorded as a closure.
    """
    if not closure_context:
        return dict(args)
    excluded = {c.place_id for c in closure_context}
    new_args = dict(args)
    if tool_name in ("semantic_search", "nearby"):
        existing_filters = new_args.get("filters")
        if existing_filters is None:
            llm_excluded: set[str] = set()
            base: dict[str, Any] = {}
        elif isinstance(existing_filters, SearchFilters):
            base = existing_filters.model_dump(exclude_none=True)
            llm_excluded = set(base.get("excluded_place_ids") or [])
        else:
            # LangChain delivers `filters` as a plain dict in tool_call args.
            # Validate it through SearchFilters once (defensive — rejects
            # unknown fields) then re-emit as a dict so the result stays
            # JSON-serializable.
            llm = SearchFilters.model_validate(existing_filters)
            base = llm.model_dump(exclude_none=True)
            llm_excluded = set(base.get("excluded_place_ids") or [])
        base["excluded_place_ids"] = sorted(llm_excluded | excluded)
        new_args["filters"] = base
        return new_args
    if tool_name == "kg_traverse":
        llm_excluded = set(new_args.get("excluded_place_ids") or [])
        new_args["excluded_place_ids"] = sorted(llm_excluded | excluded)
        return new_args
    return new_args


def _cap_closure_context(entries: list[ClosureContext]) -> list[ClosureContext]:
    """Append-and-drop-oldest to `MAX_CLOSURE_CONTEXT_ENTRIES`."""
    if len(entries) <= MAX_CLOSURE_CONTEXT_ENTRIES:
        return entries
    dropped = len(entries) - MAX_CLOSURE_CONTEXT_ENTRIES
    logger.warning("closure_context.cap_exceeded: dropped %d oldest entries", dropped)
    return entries[dropped:]


def _resolve_family_for_stop(stop: Stop) -> str:
    """family from primary_type. Returns "" when nothing resolves so the
    caller can still record the closure (without searching for a swap)."""
    fam = family_of(stop.primary_type) if stop.primary_type else None
    return fam or ""


def _build_closure_context_entry(
    stops: list[Stop],
    closed_index: int,
    proposed: CandidateMatch | None,
    outcome: str,
) -> ClosureContext:
    """Build a ClosureContext entry for a closed stop at `closed_index`,
    with stable anchors derived from neighboring stops."""
    closed = stops[closed_index]
    insert_after = stops[closed_index - 1].place_id if closed_index > 0 else None
    insert_before = stops[closed_index + 1].place_id if closed_index + 1 < len(stops) else None
    return ClosureContext(
        place_id=closed.place_id,
        place_name=closed.name,
        family=_resolve_family_for_stop(closed),
        attempted_arrival=closed.arrival_time
        or datetime.fromtimestamp(0, tz=ZoneInfo("America/Los_Angeles")),
        outcome=outcome,  # type: ignore[arg-type]
        insert_after_place_id=insert_after,
        insert_before_place_id=insert_before,
        stop_index_hint=closed_index,
        proposed_alternative=proposed.stop if proposed else None,
        proposed_distance_m=proposed.distance_m if proposed else None,
    )


async def swap_closed_stops(state: ItineraryState) -> dict[str, Any]:
    """LangGraph node — closure-aware swap pass.

    1. Per-stop closure check on real arrival times.
    2. For each closed stop, try a walking-distance swap of the same family.
    3. Auto-swaps batched; one bounded retime + re-check covers all of them.
    4. Any remaining closures: first becomes pending_user_decision, the rest
       queued_user_decision. Citywide fallback search populates the pending
       entry's proposed_alternative when possible.
    5. Final reply = the question text if anything is pending, else the
       regenerated summary.

    Returns the LangGraph update dict (subset of ItineraryState fields).
    No-op (empty update) when no closures are detected.
    """
    if not state.stops:
        return {}

    closed = _per_stop_closure_status(state.stops)
    if not any(closed):
        return {}

    # Phase 1: try a walking-distance swap for each closed stop.
    working_stops = list(state.stops)
    auto_swapped_entries: list[tuple[int, CandidateMatch]] = []
    pending_indices: list[int] = []

    for idx, is_closed in enumerate(closed):
        if not is_closed:
            continue
        closed_stop = working_stops[idx]
        family = _resolve_family_for_stop(closed_stop)
        if not family:
            pending_indices.append(idx)
            continue
        anchor = _resolve_anchor(state, closed_stop)
        if anchor is None:
            pending_indices.append(idx)
            continue
        probe_ctx = ClosureContext(
            place_id=closed_stop.place_id,
            place_name=closed_stop.name,
            family=family,
            attempted_arrival=closed_stop.arrival_time or datetime.now(),
            outcome="pending_user_decision",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=idx,
        )
        match = _try_walking_distance_swap(state, probe_ctx, anchor_place_id=anchor)
        if match is None:
            pending_indices.append(idx)
            continue
        auto_swapped_entries.append((idx, match))

    # Apply auto-swaps in one pass.
    new_closure_entries: list[ClosureContext] = []
    if auto_swapped_entries:
        for idx, match in auto_swapped_entries:
            working_stops[idx] = match.stop
            new_closure_entries.append(
                _build_closure_context_entry(
                    state.stops, idx, proposed=match, outcome="auto_swapped"
                )
            )
        # Phase 2: one bounded retime + re-check.
        retimed = await _bounded_retime_after_swap(working_stops)
        # Re-check on the retimed plan to catch a swap that's open at the
        # OLD projected arrival but not the NEW one after re-routing.
        re_closed = _per_stop_closure_status(retimed)
        # Pull DB enrichment in once on the retimed set so cards stay fresh.
        enrich_stops_with_booking(retimed, state)
        working_stops = retimed
        for idx, is_closed in enumerate(re_closed):
            if is_closed and idx not in pending_indices:
                pending_indices.append(idx)

    # Phase 3: escalate unresolved closures.
    pending_indices.sort()
    for n, idx in enumerate(pending_indices):
        closed_stop = working_stops[idx]
        family = _resolve_family_for_stop(closed_stop)
        outcome = "pending_user_decision" if n == 0 else "queued_user_decision"

        proposal: CandidateMatch | None = None
        if family:
            anchor = _resolve_anchor(state, closed_stop)
            if anchor:
                probe_ctx = ClosureContext(
                    place_id=closed_stop.place_id,
                    place_name=closed_stop.name,
                    family=family,
                    attempted_arrival=closed_stop.arrival_time or datetime.now(),
                    outcome="pending_user_decision",
                    insert_after_place_id=None,
                    insert_before_place_id=None,
                    stop_index_hint=idx,
                )
                proposal = _try_any_distance_search(state, probe_ctx, anchor_place_id=anchor)
        new_closure_entries.append(
            _build_closure_context_entry(state.stops, idx, proposed=proposal, outcome=outcome)
        )

    # Drop the closed (unswapped) stops from working_stops so the summary
    # doesn't show a place we're asking about.
    pending_set = set(pending_indices)
    final_stops = [s for i, s in enumerate(working_stops) if i not in pending_set]

    merged_context = _cap_closure_context([*state.closure_context, *new_closure_entries])

    pending_entry = next(
        (c for c in new_closure_entries if c.outcome == "pending_user_decision"),
        None,
    )
    if pending_entry is not None:
        final_reply = _formulate_closure_question(pending_entry)
    else:
        probe_state = state.model_copy(update={"stops": final_stops})
        final_reply = summarize_stops(probe_state)

    return {
        "stops": final_stops,
        "closure_context": merged_context,
        "final_reply": final_reply,
    }


__all__ = [
    "CandidateMatch",
    "_apply_swap",
    "_bounded_retime_after_swap",
    "_build_closure_context_entry",
    "_execute_closure_query",
    "_formulate_closure_question",
    "_inject_closure_exclusions",
    "_per_stop_closure_status",
    "_promote_pending",
    "_resolve_anchor",
    "_resolve_family_for_stop",
    "_resolve_insert_position",
    "_score_candidate",
    "_try_any_distance_search",
    "_try_walking_distance_swap",
    "swap_closed_stops",
]
