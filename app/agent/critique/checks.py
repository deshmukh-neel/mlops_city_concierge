"""Deterministic itinerary checks. Pure functions of state plus DB lookups.

Canonical home for these checks. W6's eval pipeline imports from here so
request-time critique and offline eval share one implementation.

Each check returns a 0.0-1.0 score; `itinerary_violations(state)` returns the
list of check names that fell below their threshold.
"""

from __future__ import annotations

import logging

from psycopg2.extras import RealDictCursor

from app.agent.planning import haversine_m
from app.agent.state import ItineraryState
from app.db import get_conn

_log = logging.getLogger(__name__)

CRITIQUE_THRESHOLDS: dict[str, float] = {
    "constraints_satisfied": 0.8,
    "geographic_coherence": 1.0,
    "stop_count_satisfied": 1.0,
    "temporal_coherence": 1.0,
    "walking_budget_respected": 1.0,
    "no_hallucinated_place_ids": 1.0,
}


def no_hallucinated_place_ids(state: ItineraryState) -> float:
    """1.0 iff every committed place_id resolves in places_raw. Zero tolerance."""
    if not state.stops:
        return 1.0
    pids = [s.place_id for s in state.stops]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT place_id FROM places_raw WHERE place_id = ANY(%s)",
            [pids],
        )
        found = {row[0] for row in cur.fetchall()}
    return 1.0 if all(p in found for p in pids) else 0.0


def stop_count_satisfied(state: ItineraryState) -> float:
    """1.0 iff an explicit requested stop count matches committed stops."""
    requested = state.constraints.num_stops
    if requested is None:
        return 1.0
    return 1.0 if len(state.stops) == requested else 0.0


def temporal_coherence(state: ItineraryState) -> float:
    """1.0 iff every stop is open at its planned arrival_time per place_is_open.

    Stops without an arrival_time are skipped (we can't check what we don't
    know). Stops without hours data are treated as open (matches the SQL
    helper's empty-hours behavior — the agent's filter would not have picked
    them on `must_be_open`).

    Coalesces all stops into one parametrized query via `unnest` so a 5-stop
    itinerary is one round-trip, not five."""
    checkable = [s for s in state.stops if s.arrival_time is not None]
    if not checkable:
        return 1.0
    pids = [s.place_id for s in checkable]
    arrivals = [s.arrival_time for s in checkable]
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT pr.place_id,
                   place_is_open(pr.regular_opening_hours, t.arrival) AS is_open
              FROM unnest(%s::text[], %s::timestamptz[]) AS t(place_id, arrival)
              JOIN places_raw pr ON pr.place_id = t.place_id
            """,
            [pids, arrivals],
        )
        results = {row["place_id"]: bool(row["is_open"]) for row in cur.fetchall()}
    # Stops missing from the result (no row) treat as open per docstring.
    open_count = sum(1 for s in checkable if results.get(s.place_id, True))
    return open_count / len(checkable)


def geographic_coherence(state: ItineraryState) -> float:
    """1.0 iff every consecutive pair fits within a per-leg walking budget.

    Per-leg budget = walking_budget_m / max(num_stops - 1, 1). Pairs missing
    coordinates are skipped — we report on what we can measure."""
    stops = state.stops
    if len(stops) < 2:
        return 1.0
    measurable_legs: list[float] = []
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if a.latitude is None or a.longitude is None or b.latitude is None or b.longitude is None:
            continue
        measurable_legs.append(haversine_m((a.latitude, a.longitude), (b.latitude, b.longitude)))
    if not measurable_legs:
        return 1.0
    per_leg_budget = state.constraints.walking_budget_m / max(len(stops) - 1, 1)
    fit = sum(1 for d in measurable_legs if d <= per_leg_budget)
    return fit / len(measurable_legs)


def walking_budget_respected(state: ItineraryState) -> float:
    """1.0 iff total haversine across the chain ≤ walking_budget_m."""
    stops = state.stops
    if len(stops) < 2:
        return 1.0
    total = 0.0
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if a.latitude is None or a.longitude is None or b.latitude is None or b.longitude is None:
            continue
        total += haversine_m((a.latitude, a.longitude), (b.latitude, b.longitude))
    return 1.0 if total <= state.constraints.walking_budget_m else 0.0


def constraints_satisfied(state: ItineraryState) -> float:
    """Fraction of expressed constraints that the produced stops actually
    satisfy. Looked up via places_raw so we use authoritative DB values, not
    what the agent claimed.

    Only constraints the user actually expressed are scored; unset ones are
    not penalized. Returns 1.0 if no constraints were expressed."""
    if not state.stops:
        return 1.0
    c = state.constraints
    expressed: list[str] = []
    if c.price_level_max is not None:
        expressed.append("price_level_max")
    if c.min_rating is not None:
        expressed.append("min_rating")
    if c.min_user_rating_count is not None:
        expressed.append("min_user_rating_count")
    want_neighborhood: str | None = None
    if c.neighborhood:
        expressed.append("neighborhood")
        want_neighborhood = c.neighborhood.lower()
    if not expressed:
        return 1.0

    pids = [s.place_id for s in state.stops]
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT place_id,
                   price_level_rank(price_level) AS price_rank,
                   rating, user_rating_count,
                   neighborhood_of(source_json) AS neighborhood,
                   formatted_address
              FROM places_raw
             WHERE place_id = ANY(%s)
            """,
            [pids],
        )
        rows = {row["place_id"]: row for row in cur.fetchall()}

    satisfied = 0
    total = 0
    for stop in state.stops:
        row = rows.get(stop.place_id)
        if row is None:
            # Hallucinated place_ids are scored separately; here we just skip.
            continue
        for con in expressed:
            total += 1
            if con == "price_level_max":
                pr = row["price_rank"]
                if pr is None or pr <= c.price_level_max:
                    satisfied += 1
            elif con == "min_rating":
                r = row["rating"]
                # Missing rating data passes — we can't fault what we can't measure.
                if r is None or r >= c.min_rating:
                    satisfied += 1
            elif con == "min_user_rating_count":
                cnt = row["user_rating_count"]
                if cnt is None or cnt >= c.min_user_rating_count:
                    satisfied += 1
            elif con == "neighborhood" and want_neighborhood is not None:
                hood = (row["neighborhood"] or "").lower()
                addr = (row["formatted_address"] or "").lower()
                if hood == want_neighborhood or want_neighborhood in addr:
                    satisfied += 1
    if total == 0:
        return 1.0
    return satisfied / total


def itinerary_violations(state: ItineraryState) -> list[str]:
    """Return the list of check names that fell below their threshold.

    Fails open on DB errors: if a check that needs DB access can't reach the
    database, it is skipped rather than treated as a violation. The user
    gets their plan; the missed check shows up in logs."""
    failed: list[str] = []

    def _try(name: str, fn) -> None:
        try:
            score = fn(state)
        except Exception as e:  # noqa: BLE001
            _log.warning("itinerary check %s failed; skipping: %s", name, e)
            return
        if score < CRITIQUE_THRESHOLDS[name]:
            failed.append(name)

    # Order matters: hallucinated_place_ids comes first because every other
    # check assumes the place_ids are real. Stop count comes next because a
    # partially rejected commit is not a complete itinerary even if every
    # committed place is individually valid.
    _try("no_hallucinated_place_ids", no_hallucinated_place_ids)
    _try("stop_count_satisfied", stop_count_satisfied)
    _try("temporal_coherence", temporal_coherence)
    _try("geographic_coherence", geographic_coherence)
    _try("walking_budget_respected", walking_budget_respected)
    _try("constraints_satisfied", constraints_satisfied)
    return failed
