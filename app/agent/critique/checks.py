"""Deterministic itinerary checks. Pure functions of state plus DB lookups.

Canonical home for these checks. W6's eval pipeline imports from here so
request-time critique and offline eval share one implementation.

Each check returns a 0.0-1.0 score; `itinerary_violations(state)` returns the
list of check names that fell below their threshold.
"""

from __future__ import annotations

from psycopg2.extras import RealDictCursor

from app.agent.planning import haversine_m
from app.agent.state import ItineraryState
from app.db import get_conn

CRITIQUE_THRESHOLDS: dict[str, float] = {
    "constraints_satisfied": 0.8,
    "geographic_coherence": 1.0,
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


def temporal_coherence(state: ItineraryState) -> float:
    """1.0 iff every stop is open at its planned arrival_time per place_is_open.

    Stops without an arrival_time are skipped (we can't check what we don't
    know). Stops without hours data are treated as open (matches the SQL
    helper's empty-hours behavior — the agent's filter would not have picked
    them on `must_be_open`)."""
    checkable = [s for s in state.stops if s.arrival_time is not None]
    if not checkable:
        return 1.0
    results: dict[str, bool] = {}
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for stop in checkable:
            cur.execute(
                "SELECT place_is_open(regular_opening_hours, %s::timestamptz) AS is_open "
                "FROM places_raw WHERE place_id = %s",
                [stop.arrival_time, stop.place_id],
            )
            row = cur.fetchone()
            results[stop.place_id] = bool(row["is_open"]) if row else True
    open_count = sum(1 for s in checkable if results.get(s.place_id, True))
    return open_count / len(checkable)


def geographic_coherence(state: ItineraryState) -> float:
    """1.0 iff every consecutive pair fits within a per-leg walking budget.

    Per-leg budget = walking_budget_m / max(num_stops - 1, 1). Pairs missing
    coordinates are skipped — we report on what we can measure."""
    stops = state.stops
    if len(stops) < 2:
        return 1.0
    legs = [(stops[i], stops[i + 1]) for i in range(len(stops) - 1)]
    measurable = [
        (a, b)
        for a, b in legs
        if a.latitude is not None
        and a.longitude is not None
        and b.latitude is not None
        and b.longitude is not None
    ]
    if not measurable:
        return 1.0
    per_leg_budget = state.constraints.walking_budget_m / max(len(stops) - 1, 1)
    fit = sum(
        1
        for a, b in measurable
        if haversine_m((a.latitude, a.longitude), (b.latitude, b.longitude)) <= per_leg_budget
    )
    return fit / len(measurable)


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
    if c.neighborhood:
        expressed.append("neighborhood")
    if not expressed:
        return 1.0

    pids = [s.place_id for s in state.stops]
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT place_id,
                   price_level_rank(price_level) AS price_rank,
                   rating, user_rating_count, neighborhood, formatted_address
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
                if r is not None and r >= c.min_rating:
                    satisfied += 1
            elif con == "min_user_rating_count":
                cnt = row["user_rating_count"]
                if cnt is not None and cnt >= c.min_user_rating_count:
                    satisfied += 1
            elif con == "neighborhood":
                want = c.neighborhood.lower()
                hood = (row["neighborhood"] or "").lower()
                addr = (row["formatted_address"] or "").lower()
                if hood == want or want in addr:
                    satisfied += 1
    if total == 0:
        return 1.0
    return satisfied / total


def itinerary_violations(state: ItineraryState) -> list[str]:
    """Return the list of check names that fell below their threshold.

    Order matters: hallucinated_place_ids comes first because every other
    check assumes the place_ids are real."""
    failed: list[str] = []
    if no_hallucinated_place_ids(state) < CRITIQUE_THRESHOLDS["no_hallucinated_place_ids"]:
        failed.append("no_hallucinated_place_ids")
    if temporal_coherence(state) < CRITIQUE_THRESHOLDS["temporal_coherence"]:
        failed.append("temporal_coherence")
    if geographic_coherence(state) < CRITIQUE_THRESHOLDS["geographic_coherence"]:
        failed.append("geographic_coherence")
    if walking_budget_respected(state) < CRITIQUE_THRESHOLDS["walking_budget_respected"]:
        failed.append("walking_budget_respected")
    if constraints_satisfied(state) < CRITIQUE_THRESHOLDS["constraints_satisfied"]:
        failed.append("constraints_satisfied")
    return failed
