# PlaceCard address / rating / price_level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry `address`, `rating`, and `price_level` from the already-fetched `PlaceDetails` onto `Stop` so `PlaceCard`s the frontend renders are no longer silently null.

**Architecture:** Extend `Stop` with three optional fields plus a `price_level_to_rank` helper (string enum → int 0..4, mirroring SQL `price_level_rank()`). Populate the fields inside the existing batched `get_details_many` loop in `enrich_stops_with_booking` — zero new DB calls — and *before* the `when is None` early-continue so card data lands even without a booking time. `state_to_cards` passes the fields through.

**Tech Stack:** Python 3.10, Pydantic v2, pytest (`asyncio_mode=auto`), `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-05-16-placecard-address-rating-price-design.md`

---

### Task 1: `price_level_to_rank` helper + `Stop` fields

**Files:**
- Modify: `app/agent/state.py` (add helper after `default_duration_for`, add 3 fields to `Stop` at lines 113-124)
- Test: `tests/unit/test_state.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_state.py`:

```python
"""Unit tests for app.agent.state helpers."""

from __future__ import annotations

import pytest

from app.agent.state import Stop, price_level_to_rank


@pytest.mark.parametrize(
    ("enum", "expected"),
    [
        ("PRICE_LEVEL_FREE", 0),
        ("PRICE_LEVEL_INEXPENSIVE", 1),
        ("PRICE_LEVEL_MODERATE", 2),
        ("PRICE_LEVEL_EXPENSIVE", 3),
        ("PRICE_LEVEL_VERY_EXPENSIVE", 4),
        ("PRICE_LEVEL_UNSPECIFIED", None),
        ("garbage", None),
        ("", None),
        (None, None),
    ],
)
def test_price_level_to_rank(enum: str | None, expected: int | None) -> None:
    assert price_level_to_rank(enum) == expected


def test_stop_defaults_new_fields_to_none() -> None:
    s = Stop(place_id="p1", name="A", rationale="r", source="google_places")
    assert s.address is None
    assert s.rating is None
    assert s.price_level is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'price_level_to_rank'`

- [ ] **Step 3: Write minimal implementation**

In `app/agent/state.py`, add this helper immediately after `default_duration_for` (currently ends line 110):

```python
_PRICE_LEVEL_RANK: dict[str, int] = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def price_level_to_rank(value: str | None) -> int | None:
    """Map Google's price_level enum string to the 0..4 rank PlaceCard expects.

    Mirrors the SQL `price_level_rank()` function (alembic c428add573d7) so the
    card's integer tier matches the rank used in SearchFilters comparisons.
    Unknown / unspecified / None all collapse to None.
    """
    if value is None:
        return None
    return _PRICE_LEVEL_RANK.get(value)
```

Then extend the `Stop` model (currently lines 113-124) — add three fields after `name`:

```python
class Stop(BaseModel):
    place_id: str
    name: str
    address: str | None = None
    rating: float | None = None
    price_level: int | None = None  # 0..4, mapped via price_level_to_rank
    arrival_time: datetime | None = None
    planned_duration_min: int = DEFAULT_STOP_DURATION_MIN_FALLBACK
    rationale: str
    source: str  # 'google_places' | 'editorial'
    latitude: float | None = None
    longitude: float | None = None
    primary_type: str | None = None
    booking_url: str | None = None
    booking_provider: BookingProvider | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_state.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add app/agent/state.py tests/unit/test_state.py
git commit -m "feat(agent): add address/rating/price_level to Stop + rank helper"
```

---

### Task 2: Populate the fields in `enrich_stops_with_booking`

**Files:**
- Modify: `app/agent/graph.py` (the `for stop in stops:` loop in `enrich_stops_with_booking`, currently lines 129-147)
- Test: `tests/unit/test_agent_graph.py` (add tests near the existing enrichment tests)

The critical correctness point: today the loop does `when = ...; if when is None: continue` (line 130-137) *before* resolving `details`. The new fields must be stamped from `details` **before** that `continue`, so a stop with no booking time still gets address/rating/price. That means we resolve `details` and stamp the three fields at the top of the loop body, then keep the existing `when is None` / `details is None` / booking logic.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_agent_graph.py`. First, extend the `_details_for` helper usage — add a new fixture-style helper near `_details_for` (line 28):

```python
def _rich_details_for(place_id: str) -> PlaceDetails:
    """PlaceDetails carrying address/rating/price for card-field tests."""
    return PlaceDetails(
        place_id=place_id,
        name=f"Place {place_id}",
        source="google_places",
        similarity=0.0,
        formatted_address=f"{place_id} Main St, San Francisco",
        rating=4.4,
        price_level="PRICE_LEVEL_MODERATE",
    )
```

Then add these tests (place them after `test_enrich_stops_with_booking_mutates_in_place`, ~line 507):

```python
def test_enrich_populates_card_fields_from_details() -> None:
    """address/rating/price_level flow from PlaceDetails onto Stop."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id, provider="tock", booking_url="https://x"
        )

    with (
        patch(
            "app.agent.graph.get_details_many",
            return_value={"p1": _rich_details_for("p1")},
        ),
        patch("app.agent.graph.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert stops[0].address == "p1 Main St, San Francisco"
    assert stops[0].rating == 4.4
    assert stops[0].price_level == 2


def test_enrich_card_fields_set_even_when_no_booking_time() -> None:
    """Regression guard: the no-time path skips booking but must NOT skip
    address/rating/price. These do not depend on `when`."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))  # when=None

    with (
        patch(
            "app.agent.graph.get_details_many",
            return_value={"p1": _rich_details_for("p1")},
        ),
        patch("app.agent.graph.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)

    mock_build.assert_not_called()  # no booking without a time
    assert stops[0].booking_url is None
    # ...but card fields still landed:
    assert stops[0].address == "p1 Main St, San Francisco"
    assert stops[0].rating == 4.4
    assert stops[0].price_level == 2


def test_enrich_card_fields_none_when_details_missing() -> None:
    """place_id missing from DB at enrichment time → fields stay None
    (same degradation as booking links)."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    with patch("app.agent.graph.get_details_many", return_value={}):
        enrich_stops_with_booking(stops, state)

    assert stops[0].address is None
    assert stops[0].rating is None
    assert stops[0].price_level is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_agent_graph.py -k "card_fields" -v`
Expected: FAIL — `assert None == 'p1 Main St, San Francisco'` (fields not yet populated)

- [ ] **Step 3: Write minimal implementation**

In `app/agent/graph.py`, replace the loop body in `enrich_stops_with_booking`. Current code (lines 129-147):

```python
    for stop in stops:
        when = stop.arrival_time or state.constraints.when
        if when is None:
            # No time → no booking link. Falling back to datetime.now() would
            # embed wall-clock time in the URL, breaking re-commit idempotency
            # (same inputs, different URL each call) and meaning nothing to
            # the user. The card ships without a booking link; downstream can
            # re-enrich once the user supplies a time.
            continue
        details = details_by_id.get(stop.place_id)
        if details is None:
            # place_id grounded in scratch but missing from DB at enrichment
            # time — race condition on the deletion side, or a stale id. Same
            # recoverable case as the old ValueError("unknown place_id"): skip.
            logger.warning("booking enrichment skipped: place_id=%s not in DB", stop.place_id)
            continue
        proposal = propose_booking_from_details(details, when, party_size)
        stop.booking_url = proposal.booking_url
        stop.booking_provider = proposal.provider
```

Replace with (resolve `details` first, stamp card fields before any `continue`):

```python
    for stop in stops:
        details = details_by_id.get(stop.place_id)
        if details is None:
            # place_id grounded in scratch but missing from DB at enrichment
            # time — race condition on the deletion side, or a stale id. Same
            # recoverable case as the old ValueError("unknown place_id"): skip
            # both card-field and booking enrichment for this stop.
            logger.warning("enrichment skipped: place_id=%s not in DB", stop.place_id)
            continue

        # Card fields do NOT depend on a booking time — stamp them before the
        # `when is None` skip so a timeless stop still renders a full card.
        stop.address = details.formatted_address
        stop.rating = details.rating
        stop.price_level = price_level_to_rank(details.price_level)

        when = stop.arrival_time or state.constraints.when
        if when is None:
            # No time → no booking link. Falling back to datetime.now() would
            # embed wall-clock time in the URL, breaking re-commit idempotency
            # (same inputs, different URL each call) and meaning nothing to
            # the user. The card ships without a booking link; downstream can
            # re-enrich once the user supplies a time.
            continue
        proposal = propose_booking_from_details(details, when, party_size)
        stop.booking_url = proposal.booking_url
        stop.booking_provider = proposal.provider
```

Add the import — find the existing import of state symbols in `app/agent/graph.py` and add `price_level_to_rank` to it. Verify with:

```bash
grep -n "from app.agent.state import" app/agent/graph.py
```

Add `price_level_to_rank` to that import list (alphabetically/where it fits the existing style).

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_agent_graph.py -v`
Expected: PASS — all existing enrichment tests AND the 3 new `card_fields` tests pass. (The reworded log string `"enrichment skipped: place_id=%s not in DB"` — confirm no existing test asserts the old `"booking enrichment skipped"` text; if one does, update that assertion.)

- [ ] **Step 5: Commit**

```bash
git add app/agent/graph.py tests/unit/test_agent_graph.py
git commit -m "feat(agent): populate Stop card fields in enrich_stops_with_booking"
```

---

### Task 3: Pass fields through `state_to_cards`

**Files:**
- Modify: `app/agent/io.py:38-51` (`state_to_cards`)
- Test: `tests/unit/test_io.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_io.py`:

```python
"""Unit tests for app.agent.io projection."""

from __future__ import annotations

from app.agent.io import state_to_cards
from app.agent.state import ItineraryState, Stop


def test_state_to_cards_carries_address_rating_price() -> None:
    state = ItineraryState(
        stops=[
            Stop(
                place_id="p1",
                name="Tartine",
                address="600 Guerrero St, San Francisco",
                rating=4.5,
                price_level=2,
                rationale="great pastries",
                source="google_places",
            )
        ]
    )

    cards = state_to_cards(state)

    assert len(cards) == 1
    assert cards[0]["address"] == "600 Guerrero St, San Francisco"
    assert cards[0]["rating"] == 4.5
    assert cards[0]["price_level"] == 2


def test_state_to_cards_defaults_missing_fields_to_none() -> None:
    state = ItineraryState(
        stops=[Stop(place_id="p1", name="X", rationale="r", source="google_places")]
    )

    cards = state_to_cards(state)

    assert cards[0]["address"] is None
    assert cards[0]["rating"] is None
    assert cards[0]["price_level"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_io.py -v`
Expected: FAIL — `assert None == '600 Guerrero St, San Francisco'` (fields not passed through)

- [ ] **Step 3: Write minimal implementation**

In `app/agent/io.py`, update the `PlaceCard(...)` construction in `state_to_cards` (lines 41-49) to pass the three fields:

```python
def state_to_cards(state: ItineraryState) -> list[dict[str, Any]]:
    """Project committed stops into the PlaceCard shape the frontend renders."""
    return [
        PlaceCard(
            place_id=s.place_id,
            name=s.name,
            address=s.address,
            rating=s.rating,
            price_level=s.price_level,
            primary_type=s.primary_type,
            arrival_time=s.arrival_time,
            rationale=s.rationale,
            booking_url=s.booking_url,
            booking_provider=s.booking_provider,
        ).model_dump(mode="json")
        for s in state.stops
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_io.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/agent/io.py tests/unit/test_io.py
git commit -m "feat(agent): pass address/rating/price_level through state_to_cards"
```

---

### Task 4: Full-suite regression + typecheck

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `poetry run pytest tests/unit/ -q`
Expected: PASS — no regressions. Pay attention to `test_agent_graph.py`, `test_booking.py` (they exercise `enrich_stops_with_booking`).

- [ ] **Step 2: Typecheck**

Run: `poetry run mypy app/`
Expected: no new errors. `price_level_to_rank` returns `int | None`; `PlaceCard.price_level` is `int | None` — types align.

- [ ] **Step 3: Lint**

Run: `poetry run ruff check app/ tests/unit/test_state.py tests/unit/test_io.py`
Expected: clean. (Pre-commit also runs ruff on commit; this is a pre-flight.)

- [ ] **Step 4: Commit (only if Step 1-3 surfaced fixable nits)**

If nothing changed, skip. Otherwise:

```bash
git add -A
git commit -m "chore: lint/type fixups for PlaceCard field passthrough"
```

---

### Task 5: Update implementation-plan tracking + FUTURE_WATCH

Per `CLAUDE.md`: this fix resolves a FUTURE_WATCH item (not a workstream merge, so README.md status table is untouched — there's no W-row for it). Mark the FUTURE_WATCH item resolved instead.

**Files:**
- Modify: `implementation_plan/james/FUTURE_WATCH.md` (the "`PlaceCard` is missing address / rating / price_level" section, lines 46-68)

- [ ] **Step 1: Mark the item resolved**

In `implementation_plan/james/FUTURE_WATCH.md`, replace the `**Trigger:**` paragraph of that section (lines 57-65) with a resolution note:

```markdown
**Resolved (2026-05-16):** Fixed via Approach 1 on branch
`fix/placecard-address-rating-price`. `Stop` now carries
`address`/`rating`/`price_level`; populated in `enrich_stops_with_booking`
from the already-fetched `PlaceDetails` (zero extra DB calls). `price_level`
is mapped enum→int 0..4 via `price_level_to_rank` (mirrors SQL
`price_level_rank()`). Card fields are stamped before the `when is None`
booking skip so timeless stops still render full cards.
```

- [ ] **Step 2: Commit**

```bash
git add implementation_plan/james/FUTURE_WATCH.md
git commit -m "docs(future-watch): mark PlaceCard card-fields item resolved"
```

---

## Notes for the executor

- Use `poetry run` for pytest/mypy/ruff (project is poetry editable-installed; never add `sys.path` hacks).
- Do NOT run `ruff` manually before committing as a *gate* — the pre-commit hook owns that. The ruff step in Task 4 is a read-only pre-flight; if the hook reformats on commit, re-stage and amend.
- The user merges PRs themselves — stop after CI is green; do not `gh pr merge`.
- Commits are small and single-line per project convention.
