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
from app.agent.state import ItineraryState, Stop
from app.db import get_conn
from app.tools.filters import _PRIMARY_TYPE_FAMILIES, family_of

_log = logging.getLogger(__name__)

CRITIQUE_THRESHOLDS: dict[str, float] = {
    "constraints_satisfied": 0.8,
    "geographic_coherence": 1.0,
    "stop_count_satisfied": 1.0,
    "temporal_coherence": 1.0,
    "walking_budget_respected": 1.0,
    "no_hallucinated_place_ids": 1.0,
    "category_compliance": 1.0,
    "category_compliance_strict": 1.0,
    "rationale_stop_alignment": 1.0,
    "refinement_minimal_edit": 1.0,
}


def _build_family_keywords() -> dict[str, frozenset[str]]:
    """Derive per-family keyword sets from filters._PRIMARY_TYPE_FAMILIES.

    Combines `types` (snake_case array column values) and `primary_types`
    (Title Case scalar values), splits multi-word entries on underscores and
    whitespace, lowercases, and dedupes. Generic stop-words like 'restaurant',
    'bar', 'cafe' are kept — they are still informative signal that the
    rationale describes the right family of place.
    """
    keywords: dict[str, frozenset[str]] = {}
    for family, columns in _PRIMARY_TYPE_FAMILIES.items():
        words: set[str] = set()
        for value in (*columns["types"], *columns["primary_types"]):
            for token in value.replace("_", " ").lower().split():
                if token:
                    words.add(token)
        keywords[family] = frozenset(words)
    return keywords


_FAMILY_KEYWORDS: dict[str, frozenset[str]] = _build_family_keywords()

_STRICT_TYPE_KEYWORDS: dict[str, frozenset[str]] = {
    "omakase": frozenset({"Sushi Restaurant", "Japanese Restaurant", "Fine Dining Restaurant"}),
    "sushi": frozenset({"Sushi Restaurant", "Japanese Restaurant"}),
    "ramen": frozenset({"Ramen Restaurant", "Japanese Restaurant"}),
    "tacos": frozenset({"Mexican Restaurant", "Restaurant"}),
    "cocktails": frozenset({"Cocktail Bar", "Bar"}),
    "dessert": frozenset({"Dessert Shop", "Bakery", "Ice Cream Shop"}),
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


def category_compliance(state: ItineraryState) -> float | None:
    """Per-slot category match between requested slots and committed stops (EVAL-01).

    D-11-03 / WR-12 abstain contract: returns None when there are zero committed
    stops. Decisiveness failure is already the hard-gated `committed_itinerary_rate`
    signal; a zero-stop run carries no category-compliance signal and must not
    score as a perfect 1.0 (which would inflate medians for decisiveness-failing
    providers). None propagates cleanly — aggregate_results already filters
    `if score is not None`. This guard fires BEFORE the D-03 empty-requested guard.

    D-03 abstain contract: returns 1.0 when the user did not name category
    slots (state.constraints.requested_primary_types is empty). The scorer
    only fires when the user gave structured slot info.

    Otherwise: for each index i in range(min(len(requested), len(stops))), a
    "match" requires both family_of(requested[i]) and family_of(stop[i].
    primary_type) to be non-None and equal. primary_type=None on a stop is
    a strict mismatch (we can't measure but we can't pass — mirrors the
    geographic_coherence "score what we can measure" precedent toward the
    strict end).

    Score = matches / max(len(requested), len(stops)). Length mismatches
    dilute the score so an agent can't game it by over- or under-committing
    stops. (Considered alternative: abstain on length mismatch — rejected
    because it would let an agent commit zero stops or extra stops and still
    score 1.0, defeating the scorer's purpose.)

    Pure function of state: no DB access. Cannot be broken by DB outages.
    """
    if not state.stops:
        return None  # WR-12 / D-11-03: abstain — zero-stop runs carry no category signal; excluded from aggregation
    requested = state.constraints.requested_primary_types
    if not requested:
        return 1.0
    overlap = min(len(requested), len(state.stops))
    denom = max(len(requested), len(state.stops))
    matches = 0
    for i in range(overlap):
        want = family_of(requested[i])
        got = family_of(state.stops[i].primary_type)
        if want is not None and got is not None and want == got:
            matches += 1
    return matches / denom


def category_compliance_strict(state: ItineraryState) -> float:
    """Strict per-slot category match between requested keywords and stops.

    Pure function of state: no DB access. Abstains with 1.0 when the user did
    not name category slots (D-03) and fail-opens with 1.0 when no stops were
    committed. Mapped keywords use exact Title Case primary_type membership;
    unmapped keywords fall back to the same family-level comparison used by
    category_compliance so strict scoring is never worse than family scoring
    for requests outside the lookup table.
    """
    requested = state.constraints.requested_primary_types
    if not requested:
        return 1.0
    if not state.stops:
        return 1.0
    overlap = min(len(requested), len(state.stops))
    denom = max(len(requested), len(state.stops))
    matches = 0
    for i in range(overlap):
        stop_primary_type = state.stops[i].primary_type
        expected = _STRICT_TYPE_KEYWORDS.get(requested[i].lower())
        if expected is not None:
            if stop_primary_type in expected:
                matches += 1
            continue
        want = family_of(requested[i])
        got = family_of(stop_primary_type)
        if want is not None and got is not None and want == got:
            matches += 1
    return matches / denom


def is_rationale_aligned(stop: Stop) -> bool:
    """Per-stop rationale-alignment rule. Public helper (plan 04-05).

    Both `rationale_stop_alignment` (scorer) and `_first_misaligned_stop_index`
    (revision dispatcher in `app/agent/revision.py`) call this so the per-stop
    rule is single-sourced — no duplicated interpretation of "aligned" across
    the scorer and the dispatcher (DRY).

    Returns True iff the stop's rationale contains either (a) the stop's name
    (case-insensitive substring) or (b) at least one keyword from the family
    derived from `family_of(stop.primary_type)` via `_FAMILY_KEYWORDS`. Returns
    False for empty / None rationale, for None primary_type with no name match,
    or when neither path fires.
    """
    rationale_lower = stop.rationale.lower() if stop.rationale else ""
    if not rationale_lower:
        return False
    if stop.name and stop.name.lower() in rationale_lower:
        return True
    family = family_of(stop.primary_type)
    if family is None:
        return False
    keywords = _FAMILY_KEYWORDS.get(family, frozenset())
    return any(kw in rationale_lower for kw in keywords)


def rationale_stop_alignment(state: ItineraryState) -> float:
    """Per-stop rationale-to-stop alignment (EVAL-02).

    For each stop, a "match" is the boolean returned by `is_rationale_aligned`:
    either (a) the stop's name appears in its rationale (case-insensitive
    substring) or (b) at least one keyword from the stop's family appears in
    the rationale (also case-insensitive substring).

    Score = matches / len(stops). Empty stops returns 1.0 (fail-open).

    Catches two failure modes:
    - Refinement-turn rationale drift (a rewritten rationale that talks about
      a different stop or no specific place).
    - Closure-swap placeholder bleed: app/agent/swap.py:238 sets a stub
      rationale "Walking-distance alternative for {closed_stop.name}". That
      string names the CLOSED stop, not the swap candidate, and contains no
      family keyword — so this scorer returns 0.0 for any stop that still
      carries the placeholder when committed.

    Pure function of state: no DB access. Cannot be broken by DB outages.

    Per-stop logic lives in `is_rationale_aligned` so the revision dispatcher
    can identify the offending stop using the SAME rule (plan 04-05 ADVISORY 3).
    """
    if not state.stops:
        return 1.0
    matches = sum(1 for stop in state.stops if is_rationale_aligned(stop))
    return matches / len(state.stops)


def refinement_minimal_edit(state: ItineraryState) -> float:
    """Refinement minimal-edit scorer (REF-01 merge gate; PROMPT-03 extended).

    Computes the fraction of PRIOR non-target stops that survive byte-equal
    (same `place_id` in the same slot) in the current `state.stops`. Strict
    1.0 == every prior non-target stop preserved exactly; anything else
    indicates the refinement turn over-edited beyond the requested slot.

    Contract sources:
    - D-06-08 (06-CONTEXT.md): scorer math + abstain semantics.
    - D-06-09 (06-CONTEXT.md): strict 1.0 merge gate (REF-01 is binary).
    - 06-REVIEWS.md § HIGH-2: denominator iterates PRIOR non-target slots,
      NOT current non-target slots. A dropped prior non-target slot must
      fail the scorer (< 1.0), not be silently excluded from the denom.
    - 06-REVIEWS.md § Pass 2 N-2: a NEW explicit
      `state.scratch['refinement_context']: bool` flag (populated by 06-06
      for every refinement scenario regardless of turn-0 commit outcome)
      lets the scorer distinguish "non-refinement / ad-hoc invocation →
      abstain 1.0" from "refinement scenario where turn 0 produced no
      prior stops → fail-loud 0.0". Closes the silent-pass path where
      turn 0 fails to commit and the merge gate misses the failure.
    - 06-REVIEWS.md § Pass 2 N-3: the abstain/fail branching is rewritten
      with explicit five-branch precedence so each branch is independently
      testable (a regression in any one branch produces a precise CI name).

    Phase 7 / D-07-05 extension (PROMPT-03):
        Branch 5 additionally enforces same-`primary_type` on the TARGET
        (replacement) slot. The behavioral rule prompt rule 10 used to
        prescribe ("SAME `primary_type` / Google Place category as the
        original") now lives here where the binary 1.0 merge gate
        (`CRITIQUE_THRESHOLDS["refinement_minimal_edit"] = 1.0`) can
        enforce it deterministically. The change is IN-PLACE in Branch 5
        only — Branches 1-4 are byte-identical to Phase 6 (abstain /
        fail-loud-empty / fail-loud-malformed / lone-stop semantics are
        category-blind). No new scorer, no new threshold key, no new
        dispatcher entry in `itinerary_violations`.

        Scratch contract extension (plan 07-02 / D-07-06):
            `prior_committed_stops` entries now carry an optional
            `primary_type` field per entry alongside `slot` and
            `place_id`: `{slot: int, place_id: str, primary_type: str | None}`.
            The eval runner reads `Stop.primary_type` from the turn-0
            committed itinerary; the model-facing JSON block (`io.py`)
            is NOT extended (HIGH-4 prompt-injection mitigation preserved
            byte-identically).

        D-07-07 four-cell `primary_type` matrix in Branch 5
        (evaluated AFTER computing the byte-equality fraction):

            | prior primary_type | current primary_type | Branch 5 returns       |
            |--------------------|----------------------|------------------------|
            | None / missing     | (any)                | byte_fraction unchanged|
            |                    |                      | (D-07-07 abstain —     |
            |                    |                      |  migration path for    |
            |                    |                      |  legacy 06-06 scratch  |
            |                    |                      |  payloads)             |
            | present            | None                 | 0.0 (fail-loud — the   |
            |                    |                      |  commit dropped a real |
            |                    |                      |  field; defect)        |
            | present            | present, different   | 0.0 (category mismatch |
            |                    |                      |  zeros the byte-       |
            |                    |                      |  fraction; binary gate)|
            | present            | present, equal       | byte_fraction unchanged|

        Comparison is EXACT STRING EQUALITY (D-07-05): no `family_of()`
        lookup, no case folding. The prompt rule being moved was "SAME
        `primary_type` / Google Place category", not "same family".

    Five-branch precedence (mutually exclusive, evaluated in order):
        Branch 1 (abstain): `refinement_context` absent or False → 1.0.
            Covers (a) first-turn / non-refinement scratch and (b) the
            ad-hoc invocation by `itinerary_violations` in the revision
            loop where no refinement context is present. Mirrors
            `category_compliance`'s fail-open shape. Category check
            does NOT fire on Branch 1.
        Branch 2 (fail-loud): `refinement_context == True` AND
            (`prior_committed_stops` is None / missing / empty list OR
            `refinement_target_slot` is missing) → 0.0. Turn 0 was supposed
            to commit but didn't; the merge gate must surface the failure.
            Category check does NOT fire on Branch 2.
        Branch 3 (fail-loud): `refinement_context == True` AND prior is
            non-empty but every entry is malformed (missing `slot` or
            `place_id`, or `place_id` is empty/non-string) such that
            `prior_by_slot` collapses to empty → 0.0. Eval-runner contract
            violation; surface it. Category check does NOT fire on
            Branch 3.
        Branch 4 (legitimate zero-denom): `refinement_context == True`
            AND `prior_by_slot` non-empty AND every surviving entry has
            `slot == target_slot` (single-stop-target case where the lone
            prior stop IS the one being changed) → 1.0. Nothing to preserve.
            Category check does NOT fire on Branch 4 (per Phase 7
            PATTERNS.md "Preserve abstain semantics on Branch 4" — the
            lone-stop case is a degenerate refinement shape and the byte-
            equality denominator is already zero; treating it as a
            category abstain keeps the scorer's no-data semantics
            consistent).
        Branch 5 (normal): all four prior branches inapplicable → compute
            `byte_fraction = matches / len(prior_non_target_slots)` then
            apply the D-07-07 four-cell `primary_type` matrix above.

    Scratch keys read (1-indexed slots, matching the YAML convention from
    plan 06-04 and the `is_refinement_request` return convention from
    plan 06-02):
        - `prior_committed_stops`: `list[dict]` with
          `{slot: int, place_id: str, primary_type: str | None}` per entry
          (the `primary_type` key is Phase-7 / D-07-06 additive; legacy
          06-06 payloads without it trigger the D-07-07 abstain branch).
          Populated by the eval runner (plan 06-06 / 07-02) between turns;
          NOT populated on the `/chat` production path (production never
          runs this scorer mid-flight — it's an offline-eval scorer).
        - `refinement_target_slot`: `int` (1-indexed). The slot the user
          asked to change. Excluded from the byte-equality denominator and
          used as the lookup index for the D-07-07 category sub-check.
        - `refinement_context`: `bool` (per N-2). True iff the eval
          runner identified this turn as a refinement scenario. Used to
          disambiguate Branches 1 vs 2.

    Current-state field read (Phase 7 / D-07-06):
        - `state.stops[target_slot - 1].primary_type`: populated on commit
          by `commit_itinerary` via places_raw lookups (CAT-01..CAT-04;
          plan 04). Missing → D-07-07 fail-loud (0.0) — that path is a
          real defect, not a migration case.

    HIGH-2 regression guards (covered in
    `tests/unit/test_critique_checks.py::TestRefinementMinimalEdit`):
        - prior 3 stops, target_slot=2, current dropped slot 3 entirely
          → 0.5 (NOT 1.0).
        - prior 3 stops, target_slot=2, current inserted a NEW slot 4
          alongside preserved slots 1+3 → 1.0 (insertions are neutral).

    Pure function of state: no DB access. The `_try(...)` fail-open in
    `itinerary_violations` therefore never trips on a DB error here.
    """
    # Branch 1: abstain when not in refinement context.
    # D-10-04 invariant: score_checks is only called on completed (status="ok")
    # runs. Exceptions in the eval runner are converted to ERROR records
    # before reaching this function, so this abstain fires only for genuinely
    # non-refinement completed runs — never from exception-corrupted partial state.
    refinement_context = bool(state.scratch.get("refinement_context", False))
    if not refinement_context:
        return 1.0

    prior = state.scratch.get("prior_committed_stops")
    target_slot = state.scratch.get("refinement_target_slot")

    # Branch 2: refinement context but prior data missing → fail-loud.
    if target_slot is None or prior is None:
        return 0.0
    if isinstance(prior, list) and len(prior) == 0:
        return 0.0

    # Build prior_by_slot defensively — skip malformed entries.
    # Phase 7 / D-07-06 / D-07-07: build a parallel
    # `prior_primary_type_by_slot` keyed on the same well-formed slots.
    # Implementation choice (CONTEXT.md "Claude's Discretion item 3"):
    # parallel dict rather than upgrading `prior_by_slot` to carry the
    # full entry — preserves the existing place_id-keyed dict shape so
    # the Branch 5 byte-equality logic is byte-identical to Phase 6,
    # which keeps the diff localized and makes the new lookup self-
    # documenting. Legacy entries without a `primary_type` key map to
    # None (D-07-07 abstain marker).
    prior_by_slot: dict[int, str] = {}
    prior_primary_type_by_slot: dict[int, str | None] = {}
    for entry in prior:
        if not isinstance(entry, dict):
            continue
        slot = entry.get("slot")
        place_id = entry.get("place_id")
        if not isinstance(slot, int):
            continue
        if not isinstance(place_id, str) or not place_id:
            continue
        prior_by_slot[slot] = place_id
        # Tolerate missing key OR explicit None (legacy 06-06 payload).
        # Tolerate non-string values defensively — coerce to None so the
        # D-07-07 abstain branch fires rather than an EQ comparison
        # crashing on, e.g., an int sneaking through the scratch contract.
        pt = entry.get("primary_type")
        prior_primary_type_by_slot[slot] = pt if isinstance(pt, str) else None

    # Branch 3: every prior entry was malformed → fail-loud.
    if not prior_by_slot:
        return 0.0

    prior_non_target_slots = [s for s in prior_by_slot if s != target_slot]

    # Branch 4: every surviving prior entry IS the target slot (lone-stop case).
    # Per Phase 7 PATTERNS.md, the D-07-05 category check does NOT fire
    # here — see Branch 4 doc above.
    if not prior_non_target_slots:
        return 1.0

    # Branch 5: normal path — byte-equality fraction first, then D-07-07
    # category sub-check.
    current_by_slot: dict[int, str] = {i + 1: s.place_id for i, s in enumerate(state.stops)}
    matches = sum(
        1 for slot in prior_non_target_slots if current_by_slot.get(slot) == prior_by_slot[slot]
    )
    byte_fraction = matches / len(prior_non_target_slots)

    # Phase 7 / D-07-05 + D-07-07: target-slot primary_type sub-check.
    prior_target_pt = prior_primary_type_by_slot.get(target_slot)
    current_target_idx = target_slot - 1  # 1-indexed slot → 0-indexed list
    # Defensive bounds-check: refinement turn that LOST the target slot
    # entirely already produces a byte-fraction reflecting the loss; the
    # category sub-check abstains in that case (no current target to
    # compare against). This mirrors Branch 4's "no-data → abstain" shape.
    if current_target_idx < 0 or current_target_idx >= len(state.stops):
        return byte_fraction
    current_target_pt = state.stops[current_target_idx].primary_type

    # D-07-07 four-cell matrix (order matters — prior-None abstain takes
    # precedence over current-None fail-loud so legacy scratch payloads
    # never get penalized for a current-side defect they cannot observe).
    if prior_target_pt is None:
        # Abstain — legacy scratch payload OR commit-time data not yet
        # populated. Return byte-fraction unchanged.
        return byte_fraction
    if current_target_pt is None:
        # Fail-loud — prior carried a real value but commit dropped it.
        return 0.0
    if prior_target_pt != current_target_pt:
        # Category mismatch zeros the byte-fraction (binary merge-gate
        # semantic; D-07-05 forbids fractional penalties).
        return 0.0
    # Both present and equal → category check passes, return byte_fraction.
    return byte_fraction


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
        if score is None:
            # Scorer abstained (e.g. category_compliance on zero-stop runs).
            # Abstain = no signal = no violation. Do not append to failed.
            return
        if score < CRITIQUE_THRESHOLDS[name]:
            failed.append(name)

    # Order matters: hallucinated_place_ids comes first because every other
    # check assumes the place_ids are real. Stop count comes next because a
    # partially rejected commit is not a complete itinerary even if every
    # committed place is individually valid.
    _try("no_hallucinated_place_ids", no_hallucinated_place_ids)
    _try("stop_count_satisfied", stop_count_satisfied)
    _try("category_compliance", category_compliance)
    _try("category_compliance_strict", category_compliance_strict)
    _try("temporal_coherence", temporal_coherence)
    _try("geographic_coherence", geographic_coherence)
    _try("walking_budget_respected", walking_budget_respected)
    _try("constraints_satisfied", constraints_satisfied)
    _try("rationale_stop_alignment", rationale_stop_alignment)
    # refinement_minimal_edit is grouped adjacent to rationale_stop_alignment
    # because both are refinement-related. Per its Branch 1 abstain, this
    # call returns 1.0 every time `state.scratch['refinement_context']` is
    # absent (the ad-hoc revision-loop invocation case), so it never produces
    # a spurious violation in the standard /chat critique path.
    _try("refinement_minimal_edit", refinement_minimal_edit)
    return failed
