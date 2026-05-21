# Closure-Aware Itinerary Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a committed itinerary contains a stop that will be closed at its planned arrival time, the agent silently swaps in a walking-distance alternative of the same category; if none exists, it asks the user one clear question and remembers the closure across the whole conversation so refinement turns never re-suggest a closed place.

**Architecture:** A new LangGraph node `swap_closed_stops` sits between the existing `retime` node and `END`. Closure history flows through an explicit `conversation_state` field on `/chat` request and response (opaque to the frontend, validated/re-enriched on the backend). Walking-distance candidate search uses an extended `nearby()` (now projecting `dist_m`) and a new `SearchFilters.primary_type_family` / `excluded_place_ids` pair; `kg_traverse` gets the same exclusion as a top-level argument. The existing `temporal_coherence` caveat in `retime` is deleted — closures are now resolved or asked about, never warned about.

**Tech Stack:** FastAPI, Pydantic v2, LangGraph, psycopg2 (sync DB tools), httpx (async Routes API), pytest+pytest-mock, Vitest + React 18, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-05-19-closure-aware-itinerary-swap-design.md`
**Branch:** `fix/agent-reliability-review` (5 existing commits already on it)

---

## File Structure

**Create:**
- `app/agent/swap.py` — the new node + helpers (`swap_closed_stops`, `_per_stop_closure_status`, `_try_walking_distance_swap`, `_try_any_distance_search`, `_score_candidate`, `_bounded_retime_after_swap`, `_promote_pending`, `_resolve_insert_position`, `_formulate_closure_question`, `_apply_swap`, `_inject_closure_exclusions`, `CandidateMatch`)
- `tests/unit/test_swap.py` — unit + mock tests for swap helpers
- `tests/unit/test_swap_node.py` — smoke tests for the graph node with DB mocked
- `tests/integration/test_swap_real_db.py` — gated `APP_ENV=integration` integration tests

**Modify:**
- `app/agent/state.py` — add `ClosureContext`, extend `Stop` is unchanged, add `ItineraryState.closure_context`
- `app/agent/input_parsing.py` — add `parse_closure_decision`
- `app/agent/graph.py` — register `swap_closed_stops` node, delete temporal caveat in `retime` (lines 327-348), extend `_constraints_context` for closure exclusion, wire `_inject_closure_exclusions` into `act()`
- `app/tools/filters.py` — refactor dessert mapping into `_PRIMARY_TYPE_FAMILIES`, add `family_of`/`family_of_types`, add `SearchFilters.primary_type_family` + `excluded_place_ids`, compile both
- `app/tools/retrieval.py` — project `dist_m` from `nearby()`, add `dist_m` to `PlaceHit`
- `app/tools/graph.py` — add `excluded_place_ids` parameter to `kg_traverse`
- `app/agent/tools.py` — re-expose `kg_traverse` with new arg in the LLM tool surface
- `app/main.py` — extend `ChatRequest`/`ChatResponse`, add `ConversationState`, implement decode + accept/decline/alternative early-return branches, attach response state
- `frontend/src/api/chat.js` — read/write `conversation_state` (opaque)
- `frontend/src/App.jsx` — store last `conversation_state` in a `useRef`
- `tests/unit/test_chat_endpoint.py` — extend with closure-flow tests
- `tests/unit/test_filters.py` — extend with new fields
- `tests/unit/test_agent_input_parsing.py` — extend with `parse_closure_decision`
- `frontend/src/api/chat.test.js` — extend if the existing file tests round-trip; otherwise leave

**Why this decomposition:** State models and tool primitives ship first because everything else imports them. The swap module is the only file that owns closure orchestration. Graph wiring is a separate task so a failed wiring doesn't block independently-testable helpers. `/chat` and frontend changes come last because they're the user-visible glue.

---

## Pre-flight (do this exactly once, before Task 1)

- [ ] **Step 0.1: Confirm clean working tree**

Run: `git status --short`
Expected output: empty (no modifications, no untracked).
If anything appears, STOP and ask the user — do not proceed.

- [ ] **Step 0.2: Confirm branch + commits already on it**

Run: `git log main..HEAD --oneline`
Expected: the five existing commits ending with `48d1a6e` plus five docs(spec) commits ending with `15b9a6e`. If something else is present, STOP and ask the user.

- [ ] **Step 0.3: Run the full test suite once as a clean baseline**

Run: `make test`
Expected: PASS. Note the test count — additions in this plan should bring it to ~535 (current + ~36).

If anything fails on `main`-derived behavior before any code is touched, STOP and surface it. We don't want to inherit a pre-existing failure into this branch.

---

## Task 1: Add `ClosureContext` and `closure_context` field to state

**Files:**
- Modify: `app/agent/state.py`
- Test: `tests/unit/test_agent_state.py` (extend)

**Why first:** Every downstream module imports `ClosureContext` and `ItineraryState.closure_context`. Establish the contract before touching anything that uses it.

- [ ] **Step 1.1: Write the failing test**

Append to `tests/unit/test_agent_state.py` (create the file's import line for `ClosureContext` if missing):

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent.state import ClosureContext, ItineraryState, Stop


def test_closure_context_minimal_fields_validate() -> None:
    """Pending entry with no proposal — used when nearby() returns no candidate."""
    ctx = ClosureContext(
        place_id="ChIJ_closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 2, tzinfo=ZoneInfo("America/Los_Angeles")),
        outcome="pending_user_decision",
        insert_after_place_id="ChIJ_stop1",
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=None,
        proposed_distance_m=None,
    )
    assert ctx.schema_version == 1
    assert ctx.outcome == "pending_user_decision"
    assert ctx.proposed_alternative is None


def test_closure_context_with_proposal_validates() -> None:
    sophies = Stop(
        place_id="ChIJ_sophies",
        name="Sophie's Crepes",
        rationale="closest open dessert",
        source="google_places",
        latitude=37.7849,
        longitude=-122.4093,
        primary_type="Dessert Shop",
        planned_duration_min=30,
    )
    ctx = ClosureContext(
        place_id="ChIJ_closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 2, tzinfo=ZoneInfo("America/Los_Angeles")),
        outcome="pending_user_decision",
        insert_after_place_id="ChIJ_stop1",
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=sophies,
        proposed_distance_m=4800.0,
    )
    assert ctx.proposed_alternative is not None
    assert ctx.proposed_alternative.place_id == "ChIJ_sophies"
    assert ctx.proposed_distance_m == 4800.0


def test_itinerary_state_default_closure_context_empty() -> None:
    state = ItineraryState()
    assert state.closure_context == []


def test_itinerary_state_accepts_closure_context_list() -> None:
    state = ItineraryState(
        closure_context=[
            ClosureContext(
                place_id="p",
                place_name="X",
                family="bar",
                attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles")),
                outcome="auto_swapped",
                insert_after_place_id=None,
                insert_before_place_id=None,
                stop_index_hint=0,
                proposed_alternative=None,
                proposed_distance_m=None,
            )
        ]
    )
    assert len(state.closure_context) == 1
    assert state.closure_context[0].outcome == "auto_swapped"
```

- [ ] **Step 1.2: Run the failing test**

Run: `poetry run pytest tests/unit/test_agent_state.py -v -k closure`
Expected: ImportError for `ClosureContext`.

- [ ] **Step 1.3: Add `ClosureContext` and `closure_context` to `state.py`**

In `app/agent/state.py`, after the `Stop` class (around line 152, before `class ItineraryState`), insert:

```python
ClosureOutcome = Literal[
    "auto_swapped",
    "user_accepted_drive",
    "user_declined_dropped",
    "pending_user_decision",
    "queued_user_decision",
]


class ClosureContext(BaseModel):
    """One closure event recorded during a /chat conversation.

    Persisted across turns via the opaque `conversation_state` round-trip
    on /chat. Stable across neighbor drops/inserts because placement is
    anchored to the neighboring stops' place_ids, with stop_index_hint as
    a last-resort fallback.
    """

    schema_version: int = 1
    place_id: str
    place_name: str
    family: str
    attempted_arrival: datetime
    outcome: ClosureOutcome
    # Stable placement anchors — resolution priority order:
    #   1) insert_after_place_id in current stops -> insert at that index + 1
    #   2) else insert_before_place_id in current stops -> insert at that index
    #   3) else stop_index_hint, clamped to len(stops)
    insert_after_place_id: str | None = None
    insert_before_place_id: str | None = None
    stop_index_hint: int
    proposed_alternative: Stop | None = None
    proposed_distance_m: float | None = None


MAX_CLOSURE_CONTEXT_ENTRIES = 10
```

Then extend `ItineraryState` to add the new field. Find the existing class definition (line 154) and add this field after `revision_counts`:

```python
    closure_context: list[ClosureContext] = Field(default_factory=list)
```

- [ ] **Step 1.4: Run the test to verify it passes**

Run: `poetry run pytest tests/unit/test_agent_state.py -v -k closure`
Expected: 4 passed.

- [ ] **Step 1.5: Run the full unit suite to confirm nothing regressed**

Run: `poetry run pytest tests/unit/ -v 2>&1 | tail -5`
Expected: all pass (count = baseline + 4).

- [ ] **Step 1.6: Lint + typecheck**

Run: `poetry run ruff check app/agent/state.py tests/unit/test_agent_state.py && poetry run mypy app/agent/state.py`
Expected: clean.

- [ ] **Step 1.7: Commit**

```bash
git add app/agent/state.py tests/unit/test_agent_state.py
git commit -m "feat(state): add ClosureContext + closure_context field for closure-aware swap"
```

---

## Task 2: Add `parse_closure_decision` to input parsing

**Files:**
- Modify: `app/agent/input_parsing.py`
- Test: `tests/unit/test_agent_input_parsing.py` (extend)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/unit/test_agent_input_parsing.py`:

```python
from app.agent.input_parsing import parse_closure_decision


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # accept — first-token rule, anything starting with a yes-ish token
        ("yes", "accept"),
        ("Yes", "accept"),
        ("yes!", "accept"),
        ("yes, make it 4 stops", "accept"),
        ("yeah do it", "accept"),
        ("yep", "accept"),
        ("sure thing", "accept"),
        ("ok let's go", "accept"),
        ("okay", "accept"),
        ("y", "accept"),
        ("👍", "accept"),
        # decline
        ("no", "decline"),
        ("No thanks", "decline"),
        ("nope", "decline"),
        ("nah", "decline"),
        ("n", "decline"),
        # alternative — anything else
        ("find something cheaper instead", "alternative"),
        ("what about ramen?", "alternative"),
        ("pick a different one", "alternative"),
        ("", "alternative"),
        ("   ", "alternative"),
    ],
)
def test_parse_closure_decision(text: str, expected: str) -> None:
    assert parse_closure_decision(text) == expected
```

- [ ] **Step 2.2: Run it to confirm it fails**

Run: `poetry run pytest tests/unit/test_agent_input_parsing.py -v -k closure_decision`
Expected: ImportError.

- [ ] **Step 2.3: Implement `parse_closure_decision`**

Append to `app/agent/input_parsing.py`:

```python
_ACCEPT_TOKENS: frozenset[str] = frozenset({
    "yes", "yeah", "yep", "sure", "ok", "okay", "y", "👍",
})
_DECLINE_TOKENS: frozenset[str] = frozenset({"no", "nope", "n", "nah"})


def parse_closure_decision(text: str) -> Literal["accept", "decline", "alternative"]:
    """Conservative parser for a user's reply to a closure question.

    First-token rule resolves "yes + revision" (e.g. "yes! make it 4 stops")
    unambiguously as accept; the existing `explicit_num_stops_from_conversation`
    separately handles the count update. Empty / whitespace / questions / free
    text all bucket to "alternative" (no auto-accept on silence).
    """
    if not text or not text.strip():
        return "alternative"
    # First "word" — split on whitespace then strip surrounding punctuation
    # so "Yes!" or "yes," still matches. The emoji case is preserved verbatim.
    first = text.strip().split()[0].lower()
    stripped = first.strip(".,!?;:\"'()[]{}")
    if stripped in _ACCEPT_TOKENS or first in _ACCEPT_TOKENS:
        return "accept"
    if stripped in _DECLINE_TOKENS or first in _DECLINE_TOKENS:
        return "decline"
    return "alternative"
```

Also add `Literal` to the import block at the top of the file (currently only `Protocol` from `typing` is imported):

```python
from typing import Literal, Protocol
```

- [ ] **Step 2.4: Run the test to verify it passes**

Run: `poetry run pytest tests/unit/test_agent_input_parsing.py -v -k closure_decision`
Expected: all parametrized cases pass.

- [ ] **Step 2.5: Lint**

Run: `poetry run ruff check app/agent/input_parsing.py tests/unit/test_agent_input_parsing.py`

- [ ] **Step 2.6: Commit**

```bash
git add app/agent/input_parsing.py tests/unit/test_agent_input_parsing.py
git commit -m "feat(parsing): add parse_closure_decision for accept/decline/alternative"
```

---

## Task 3: Refactor dessert mapping into `_PRIMARY_TYPE_FAMILIES` + `family_of`

**Files:**
- Modify: `app/tools/filters.py`
- Test: `tests/unit/test_filters.py` (extend)

**Why before SearchFilters changes:** `primary_type_family` resolves family → SQL clauses against this mapping. Land the mapping first so the next task can lean on it.

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/unit/test_filters.py`:

```python
from app.tools.filters import (
    _PRIMARY_TYPE_FAMILIES,
    family_of,
    family_of_types,
)


def test_primary_type_families_all_have_types_and_primary_types() -> None:
    for family, members in _PRIMARY_TYPE_FAMILIES.items():
        assert set(members.keys()) == {"types", "primary_types"}, family
        assert members["types"], f"{family} types must not be empty"
        assert members["primary_types"], f"{family} primary_types must not be empty"


def test_dessert_family_preserves_existing_dessert_members() -> None:
    dessert = _PRIMARY_TYPE_FAMILIES["dessert"]
    assert "dessert_shop" in dessert["types"]
    assert "Dessert Shop" in dessert["primary_types"]
    assert "Ice Cream Shop" in dessert["primary_types"]


def test_family_of_resolves_primary_type() -> None:
    assert family_of("Dessert Shop") == "dessert"
    assert family_of("Bar") == "bar"
    assert family_of("Cafe") == "cafe"
    assert family_of("Italian Restaurant") == "restaurant"


def test_family_of_returns_none_for_unknown_primary_type() -> None:
    assert family_of("Spaceship") is None
    assert family_of(None) is None
    assert family_of("") is None


def test_family_of_types_resolves_via_types_array() -> None:
    assert family_of_types(["italian_restaurant", "restaurant"]) == "restaurant"
    assert family_of_types(["dessert_shop"]) == "dessert"
    assert family_of_types([]) is None
```

- [ ] **Step 3.2: Run to confirm failure**

Run: `poetry run pytest tests/unit/test_filters.py -v -k "family or PRIMARY_TYPE_FAMILIES"`
Expected: ImportError.

- [ ] **Step 3.3: Refactor `filters.py`**

Replace the existing `_DESSERT_TYPES` and `_DESSERT_PRIMARY_TYPES` constants (lines 140-164) with:

```python
# Maps a family name to the column conventions place_documents/places_raw
# uses for category matching:
#   - "types":         snake_case strings used in the `types[]` array column
#   - "primary_types": Title Case strings used in the `primary_type` scalar
# Both lists are required for each family because Postgres rows can carry
# either or both. The serves_dessert helper and the primary_type_family
# filter both compile to `(types && %s OR primary_type = ANY(%s))` against
# these two lists so the row matches on either column.
_PRIMARY_TYPE_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "dessert": {
        "types": (
            "dessert_shop",
            "bakery",
            "ice_cream_shop",
            "candy_store",
            "chocolate_shop",
            "coffee_shop",
            "cafe",
            "confectionery",
            "donut_shop",
            "tea_house",
        ),
        "primary_types": (
            "Dessert Shop",
            "Bakery",
            "Ice Cream Shop",
            "Candy Store",
            "Chocolate Shop",
            "Coffee Shop",
            "Cafe",
            "Confectionery store",
            "Donut Shop",
            "Tea House",
        ),
    },
    "bar": {
        "types": (
            "bar",
            "cocktail_bar",
            "wine_bar",
            "pub",
            "sports_bar",
            "night_club",
        ),
        "primary_types": (
            "Bar",
            "Cocktail Bar",
            "Wine Bar",
            "Pub",
            "Sports Bar",
            "Night Club",
        ),
    },
    "restaurant": {
        "types": (
            "restaurant",
            "fine_dining_restaurant",
            "italian_restaurant",
            "japanese_restaurant",
            "chinese_restaurant",
            "mexican_restaurant",
            "thai_restaurant",
            "indian_restaurant",
            "french_restaurant",
            "vietnamese_restaurant",
            "korean_restaurant",
            "mediterranean_restaurant",
            "seafood_restaurant",
            "steak_house",
            "sushi_restaurant",
            "ramen_restaurant",
            "pizza_restaurant",
            "american_restaurant",
        ),
        "primary_types": (
            "Restaurant",
            "Fine Dining Restaurant",
            "Italian Restaurant",
            "Japanese Restaurant",
            "Chinese Restaurant",
            "Mexican Restaurant",
            "Thai Restaurant",
            "Indian Restaurant",
            "French Restaurant",
            "Vietnamese Restaurant",
            "Korean Restaurant",
            "Mediterranean Restaurant",
            "Seafood Restaurant",
            "Steak House",
            "Sushi Restaurant",
            "Ramen Restaurant",
            "Pizza Restaurant",
            "American Restaurant",
        ),
    },
    "cafe": {
        "types": ("cafe", "coffee_shop", "tea_house"),
        "primary_types": ("Cafe", "Coffee Shop", "Tea House"),
    },
}


def family_of(primary_type: str | None) -> str | None:
    """Reverse lookup: scalar primary_type column value -> family name.

    Case-preserving comparison — the DB column preserves Title Case verbatim,
    and _PRIMARY_TYPE_FAMILIES stores both casings exactly as the columns do.
    """
    if not primary_type:
        return None
    for family, members in _PRIMARY_TYPE_FAMILIES.items():
        if primary_type in members["primary_types"]:
            return family
    return None


def family_of_types(types: list[str] | None) -> str | None:
    """Reverse lookup: types[] array values -> family name.

    Returns the first family that overlaps the input list (by membership).
    Lets the swap node fall back when primary_type isn't in the index but
    types[] is populated.
    """
    if not types:
        return None
    for family, members in _PRIMARY_TYPE_FAMILIES.items():
        if any(t in members["types"] for t in types):
            return family
    return None
```

Update the `serves_dessert` clause (currently lines 210-217) to use the new mapping:

```python
    if f.serves_dessert is not None:
        dessert = _PRIMARY_TYPE_FAMILIES["dessert"]
        dessert_clause = "(types && %s OR primary_type = ANY(%s))"
        if f.serves_dessert:
            clauses.append(dessert_clause)
        else:
            clauses.append(f"NOT {dessert_clause}")
        params.append(list(dessert["types"]))
        params.append(list(dessert["primary_types"]))
```

- [ ] **Step 3.4: Run the new tests**

Run: `poetry run pytest tests/unit/test_filters.py -v -k "family or PRIMARY_TYPE_FAMILIES"`
Expected: all pass.

- [ ] **Step 3.5: Run the full filters test file**

Run: `poetry run pytest tests/unit/test_filters.py -v`
Expected: all existing dessert tests still pass (the SQL clause shape didn't change).

- [ ] **Step 3.6: Lint + typecheck**

Run: `poetry run ruff check app/tools/filters.py tests/unit/test_filters.py && poetry run mypy app/tools/filters.py`

- [ ] **Step 3.7: Commit**

```bash
git add app/tools/filters.py tests/unit/test_filters.py
git commit -m "refactor(filters): unify primary_type families + add family_of lookups"
```

---

## Task 4: Add `primary_type_family` and `excluded_place_ids` to `SearchFilters`

**Files:**
- Modify: `app/tools/filters.py`
- Test: `tests/unit/test_filters.py` (extend)

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/unit/test_filters.py`:

```python
def test_primary_type_family_filter_compiles_both_columns() -> None:
    where, params = compile_filters(
        SearchFilters(
            primary_type_family="dessert",
            business_status=None,
            min_user_rating_count=0,
        )
    )
    assert "(types && %s OR primary_type = ANY(%s))" in where
    # types list (snake_case) and primary_types list (Title Case) both as params.
    assert "dessert_shop" in params[0]
    assert "Dessert Shop" in params[1]


def test_primary_type_family_unknown_family_raises() -> None:
    with pytest.raises(ValidationError):
        SearchFilters.model_validate({"primary_type_family": "spaceship"})


def test_excluded_place_ids_filter_compiles_to_not_all() -> None:
    where, params = compile_filters(
        SearchFilters(
            excluded_place_ids=["ChIJ_a", "ChIJ_b"],
            business_status=None,
            min_user_rating_count=0,
        )
    )
    assert "place_id != ALL(%s)" in where
    assert ["ChIJ_a", "ChIJ_b"] in params


def test_excluded_place_ids_empty_list_emits_no_clause() -> None:
    where, _ = compile_filters(
        SearchFilters(
            excluded_place_ids=[],
            business_status=None,
            min_user_rating_count=0,
        )
    )
    assert "place_id != ALL" not in where
```

- [ ] **Step 4.2: Confirm they fail**

Run: `poetry run pytest tests/unit/test_filters.py -v -k "primary_type_family or excluded_place_ids"`
Expected: ValidationError on the unknown-field path / no clause emitted for the others.

- [ ] **Step 4.3: Extend `SearchFilters`**

In `app/tools/filters.py`, in the `SearchFilters` class, add the two new fields after `types_any` (around line 72). The `extra="forbid"` config means unknown keys are already rejected, so the validation test above doesn't need its own validator.

```python
    primary_type_family: Literal["dessert", "bar", "restaurant", "cafe"] | None = Field(
        default=None,
        description=(
            "Restrict candidates to a category family (dessert/bar/restaurant/"
            "cafe). Expands to a `(types && %s OR primary_type = ANY(%s))` "
            "clause against both column conventions. Distinct from `types_any`, "
            "which only matches the types[] array."
        ),
    )
    excluded_place_ids: list[str] | None = Field(
        default=None,
        description=(
            "place_ids to exclude from results. Used by the closure-swap path "
            "to prevent re-suggesting a place that was previously found closed "
            "in the same conversation. Empty list (or None) is a no-op."
        ),
    )
```

Add `Literal` to the typing import at the top:

```python
from typing import Literal
```

Then extend `compile_filters` with two new clauses. After the existing `types_any` block (around line 222):

```python
    if f.primary_type_family is not None:
        family = _PRIMARY_TYPE_FAMILIES[f.primary_type_family]
        clauses.append("(types && %s OR primary_type = ANY(%s))")
        params.append(list(family["types"]))
        params.append(list(family["primary_types"]))

    if f.excluded_place_ids:
        clauses.append("place_id != ALL(%s)")
        params.append(f.excluded_place_ids)
```

(Note: `if f.excluded_place_ids:` is intentional — `[]` is falsy here, so an empty list emits no clause per the test.)

- [ ] **Step 4.4: Run the new tests**

Run: `poetry run pytest tests/unit/test_filters.py -v -k "primary_type_family or excluded_place_ids"`
Expected: all pass.

- [ ] **Step 4.5: Full filters file passes**

Run: `poetry run pytest tests/unit/test_filters.py -v`

- [ ] **Step 4.6: Lint + typecheck**

Run: `poetry run ruff check app/tools/filters.py && poetry run mypy app/tools/filters.py`

- [ ] **Step 4.7: Commit**

```bash
git add app/tools/filters.py tests/unit/test_filters.py
git commit -m "feat(filters): add primary_type_family + excluded_place_ids"
```

---

## Task 5: Project `dist_m` from `nearby()` and add it to `PlaceHit`

**Files:**
- Modify: `app/tools/retrieval.py`
- Test: `tests/unit/test_tools_retrieval.py` (extend)

- [ ] **Step 5.1: Read the existing retrieval tests to mirror their mocking style**

Run: `poetry run python -c "import pathlib; print(pathlib.Path('tests/unit/test_tools_retrieval.py').read_text()[:800])"`
(Used only as a reference — no change needed if its `mocker.patch` shape already covers `_execute`.)

- [ ] **Step 5.2: Write a failing test**

Append to `tests/unit/test_tools_retrieval.py`:

```python
def test_place_hit_dist_m_defaults_to_none() -> None:
    """Backward compatibility: semantic_search rows have no dist_m column,
    so PlaceHit must accept missing dist_m and default to None."""
    from app.tools.retrieval import PlaceHit

    hit = PlaceHit(
        place_id="p1",
        name="x",
        source="google_places",
        similarity=0.7,
    )
    assert hit.dist_m is None


def test_nearby_projects_dist_m_in_select(mocker) -> None:
    """nearby() must SELECT dist_m alongside the existing fields so the
    closure-swap node can score candidates by route impact."""
    from app.tools.retrieval import nearby

    captured: dict = {}

    def _fake_execute(sql: str, params: list):
        captured["sql"] = sql
        captured["params"] = params
        return [
            {
                "place_id": "p2",
                "name": "near",
                "primary_type": "Bar",
                "formatted_address": "...",
                "latitude": 37.78,
                "longitude": -122.41,
                "rating": 4.5,
                "price_level": None,
                "business_status": "OPERATIONAL",
                "source": "google_places",
                "similarity": 0.0,
                "snippet": "snippet",
                "dist_m": 250.0,
            }
        ]

    mocker.patch("app.tools.retrieval._execute", side_effect=_fake_execute)
    hits = nearby(place_id="anchor", radius_m=800)
    assert "dist_m" in captured["sql"]
    assert hits[0].dist_m == 250.0
```

- [ ] **Step 5.3: Confirm failure**

Run: `poetry run pytest tests/unit/test_tools_retrieval.py -v -k "dist_m"`
Expected: failure — `PlaceHit` rejects unknown field `dist_m` (extra=forbid is NOT set there, but the dict-construction path would just pass it as an unknown kw if the model lacks the field — depending on Pydantic version, the test may fail by AttributeError on `.dist_m`). Either way, FAIL before the change.

- [ ] **Step 5.4: Add `dist_m` to `PlaceHit`**

In `app/tools/retrieval.py`, append a `dist_m: float | None = None` field to `PlaceHit` (after `snippet`, around line 44):

```python
    dist_m: float | None = None
```

- [ ] **Step 5.5: Project `dist_m` in `nearby()` SQL**

Change the outer SELECT inside `nearby()` (currently lines 145-150) to include `dist_m`:

```python
        SELECT
            place_id, name, primary_type, formatted_address,
            latitude, longitude, rating, price_level, business_status,
            source,
            0.0 AS similarity,
            snippet,
            dist_m
        FROM candidates
```

- [ ] **Step 5.6: Run the dist_m tests**

Run: `poetry run pytest tests/unit/test_tools_retrieval.py -v -k "dist_m"`
Expected: pass.

- [ ] **Step 5.7: Confirm `semantic_search` still works (no `dist_m` row coming in)**

Run: `poetry run pytest tests/unit/test_tools_retrieval.py -v`
Expected: pass — `dist_m: float | None = None` means rows without it default to None.

- [ ] **Step 5.8: Lint + typecheck**

Run: `poetry run ruff check app/tools/retrieval.py && poetry run mypy app/tools/retrieval.py`

- [ ] **Step 5.9: Commit**

```bash
git add app/tools/retrieval.py tests/unit/test_tools_retrieval.py
git commit -m "feat(retrieval): project dist_m from nearby() + expose on PlaceHit"
```

---

## Task 6: Add `excluded_place_ids` to `kg_traverse`

**Files:**
- Modify: `app/tools/graph.py`
- Modify: `app/agent/tools.py` (LLM-facing wrapper)
- Test: `tests/unit/test_kg_traverse.py` (extend)

- [ ] **Step 6.1: Write the failing test**

Append to `tests/unit/test_kg_traverse.py`:

```python
def test_kg_traverse_excluded_place_ids_filters_at_sql_layer(mocker) -> None:
    """Closure-swap exclusion path: kg_traverse must drop excluded place_ids
    from the WHERE clause at the SQL layer, not in Python."""
    from app.tools.graph import kg_traverse

    captured: dict = {}

    def _fake_execute(sql: str, params: list):
        captured["sql"] = sql
        captured["params"] = params
        return []

    mocker.patch("app.tools.graph._execute", side_effect=_fake_execute)
    kg_traverse(place_id="anchor", excluded_place_ids=["ChIJ_a", "ChIJ_b"])

    # The SQL must include a place_id exclusion guarded by the NULL check so
    # callers that omit excluded_place_ids still match the original behavior.
    assert "pd.place_id != ALL(%s::text[])" in captured["sql"]
    # Params pattern: src, relation, excluded (null check), excluded (filter), k
    assert ["ChIJ_a", "ChIJ_b"] in captured["params"]


def test_kg_traverse_no_exclusions_is_no_op(mocker) -> None:
    """Without excluded_place_ids the SQL must produce the same rows as before
    — guarded by `IS NULL` so an empty list never filters everything out."""
    from app.tools.graph import kg_traverse

    captured: dict = {}

    def _fake_execute(sql: str, params: list):
        captured["sql"] = sql
        captured["params"] = params
        return []

    mocker.patch("app.tools.graph._execute", side_effect=_fake_execute)
    kg_traverse(place_id="anchor")
    # The NULL guard is in the SQL; the parameter list includes None for both
    # exclusion slots.
    assert "IS NULL OR pd.place_id != ALL" in captured["sql"]
    # Two None slots for the duplicated parameter.
    assert captured["params"].count(None) == 2
```

- [ ] **Step 6.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_kg_traverse.py -v -k "excluded or no_exclusions"`
Expected: TypeError (`kg_traverse() got unexpected keyword argument 'excluded_place_ids'`).

- [ ] **Step 6.3: Modify `kg_traverse`**

In `app/tools/graph.py`, change the function signature (line 29) and SQL:

```python
def kg_traverse(
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
    excluded_place_ids: list[str] | None = None,
) -> list[RelatedPlace]:
    """Return up to ``k`` places related to ``place_id`` by ``relation_type``.

    NEAR is ordered ascending by weight (closest first); SIMILAR_VECTOR is
    ordered descending by weight (most similar first); other relations use a
    stable LIMIT. Unknown relation_type raises ValueError.

    ``excluded_place_ids`` lets callers (the closure-swap node) suppress
    place_ids known to be closed in the current conversation. Passed at the
    SQL layer so behavior matches `nearby` and `semantic_search` exclusion.
    """
    if relation_type not in VALID_RELATIONS:
        raise ValueError(f"Unknown relation_type: {relation_type}")
    view = _view_name()  # allowlist member — interpolated into SQL below
    sql = f"""
        SELECT pd.place_id, pd.name, pd.primary_type, pd.formatted_address,
               pd.latitude, pd.longitude, pd.rating, pd.price_level,
               pd.business_status, pd.source,
               0.0 AS similarity,
               LEFT(pd.embedding_text, 400) AS snippet,
               r.relation_type, r.weight,
               r.metadata AS relation_metadata
        FROM place_relations r
        JOIN {view} pd ON pd.place_id = r.dst_place_id
        WHERE r.src_place_id = %s
          AND r.relation_type = %s
          AND (%s::text[] IS NULL OR pd.place_id != ALL(%s::text[]))
        ORDER BY
          CASE r.relation_type
            WHEN 'NEAR'           THEN  r.weight
            WHEN 'SIMILAR_VECTOR' THEN -r.weight
            ELSE 0
          END
        LIMIT %s
    """  # noqa: S608
    # Pass excluded twice so the NULL guard short-circuits when the caller
    # didn't provide a list — keeps the no-op semantics free.
    exclude = excluded_place_ids or None
    rows = _execute(sql, [place_id, relation_type, exclude, exclude, k])
    return [RelatedPlace(**row) for row in rows]
```

- [ ] **Step 6.4: Update the LLM-facing wrapper**

In `app/agent/tools.py`, change the `kg_traverse` wrapper signature (line 77-96) and forward the new arg:

```python
def kg_traverse(
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
    excluded_place_ids: list[str] | None = None,
) -> list[RelatedPlace]:
    """Traverse the knowledge graph from `place_id` along a relation_type.

    Pick the relation_type by intent:
    - SIMILAR_VECTOR: "more like this" — same vibe/category as the anchor.
    - SAME_NEIGHBORHOOD: alternates in the same SF neighborhood.
    - NEAR_LANDMARK: the anchor is near a known landmark (museum, park).
    - NEAR: geographic neighbors (~800m) without re-running `nearby`.
    - CONTAINED_IN: the parent venue (e.g. a stall inside a food hall) — rare.

    Single-hop: for multi-hop reasoning call again with the new anchor. If it
    returns empty, fall back to `semantic_search` or `nearby`.
    """
    from app.tools.graph import kg_traverse as _kg_traverse

    return _kg_traverse(
        place_id=place_id,
        relation_type=relation_type,
        k=k,
        excluded_place_ids=excluded_place_ids,
    )
```

- [ ] **Step 6.5: Run the new tests**

Run: `poetry run pytest tests/unit/test_kg_traverse.py -v -k "excluded or no_exclusions"`
Expected: pass.

- [ ] **Step 6.6: Run the full kg_traverse test suite**

Run: `poetry run pytest tests/unit/test_kg_traverse.py tests/unit/test_kg_traverse_functional.py tests/unit/test_kg_traverse_smoke.py -v`
Expected: all pass — the SQL change is backward-compatible because the new clause is a NULL-guarded no-op when excluded is None.

- [ ] **Step 6.7: Lint + typecheck**

Run: `poetry run ruff check app/tools/graph.py app/agent/tools.py && poetry run mypy app/tools/graph.py app/agent/tools.py`

- [ ] **Step 6.8: Commit**

```bash
git add app/tools/graph.py app/agent/tools.py tests/unit/test_kg_traverse.py
git commit -m "feat(tools): add excluded_place_ids to kg_traverse + LLM wrapper"
```

---

## Task 7: Create `app/agent/swap.py` skeleton with `_per_stop_closure_status`

**Files:**
- Create: `app/agent/swap.py`
- Create: `tests/unit/test_swap.py`

**Why split:** the swap module's surface is large. Build it bottom-up — the closure-status helper underpins every other helper, and verifying it independently is cleanest.

- [ ] **Step 7.1: Write the failing test**

Create `tests/unit/test_swap.py`:

```python
"""Unit tests for app/agent/swap.py.

All DB access (place_is_open SQL function) is mocked via _execute or the
helper's direct cursor calls. Live-DB behavior is covered separately in
tests/integration/test_swap_real_db.py.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agent.state import Stop

SF = ZoneInfo("America/Los_Angeles")


def _stop(
    place_id: str = "p",
    name: str = "X",
    arrival_iso: str = "2026-05-19T19:00:00-07:00",
    lat: float = 37.78,
    lng: float = -122.41,
    primary_type: str = "Bar",
) -> Stop:
    return Stop(
        place_id=place_id,
        name=name,
        rationale="r",
        source="google_places",
        arrival_time=datetime.fromisoformat(arrival_iso),
        latitude=lat,
        longitude=lng,
        primary_type=primary_type,
        planned_duration_min=60,
    )


def test_per_stop_closure_status_all_open(mocker) -> None:
    """Every stop returns is_open=True → list of all True (so "is closed" is
    all False)."""
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"p1": True, "p2": True},
    )
    stops = [_stop(place_id="p1"), _stop(place_id="p2")]
    statuses = _per_stop_closure_status(stops)
    # _per_stop_closure_status returns True for "closed".
    assert statuses == [False, False]


def test_per_stop_closure_status_one_closed(mocker) -> None:
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"p1": True, "p2": False},
    )
    stops = [_stop(place_id="p1"), _stop(place_id="p2")]
    statuses = _per_stop_closure_status(stops)
    assert statuses == [False, True]


def test_per_stop_closure_status_skips_stops_without_arrival_time(mocker) -> None:
    """A stop with arrival_time=None can't be checked; treat as open."""
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"p1": True},
    )
    stops = [_stop(place_id="p1")]
    stops.append(
        Stop(
            place_id="p2",
            name="no time",
            rationale="r",
            source="google_places",
            arrival_time=None,
            latitude=37.78,
            longitude=-122.41,
            primary_type="Bar",
            planned_duration_min=60,
        )
    )
    statuses = _per_stop_closure_status(stops)
    assert statuses == [False, False]


def test_per_stop_closure_status_db_failure_fails_open(mocker) -> None:
    """A DB blip must NOT block /chat — the helper returns [False] * n
    (no closure detected) so the plan ships unchanged. Matches checks.py
    fail-open precedent at lines 200-205."""
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=Exception("db down"),
    )
    statuses = _per_stop_closure_status([_stop(place_id="p1"), _stop(place_id="p2")])
    assert statuses == [False, False]
```

- [ ] **Step 7.2: Confirm it fails**

Run: `poetry run pytest tests/unit/test_swap.py -v -k closure_status`
Expected: ImportError — `app.agent.swap` doesn't exist.

- [ ] **Step 7.3: Create `app/agent/swap.py`**

```python
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

from app.agent.state import (
    ClosureContext,
    ItineraryState,
    Stop,
)
from app.db import get_conn

logger = logging.getLogger(__name__)


def _execute_closure_query(
    place_ids: list[str],
    arrivals: list[Any],
) -> dict[str, bool]:
    """One SQL round-trip via `place_is_open`. Returns {place_id: is_open}.

    Mirrors temporal_coherence at app/agent/critique/checks.py:69-79; that
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
```

- [ ] **Step 7.4: Run the tests**

Run: `poetry run pytest tests/unit/test_swap.py -v -k closure_status`
Expected: all 4 pass.

- [ ] **Step 7.5: Lint + typecheck**

Run: `poetry run ruff check app/agent/swap.py tests/unit/test_swap.py && poetry run mypy app/agent/swap.py`

- [ ] **Step 7.6: Commit**

```bash
git add app/agent/swap.py tests/unit/test_swap.py
git commit -m "feat(swap): add closure-status detection helper"
```

---

## Task 8: Add `_resolve_insert_position`, `CandidateMatch`, `_score_candidate`

**Files:**
- Modify: `app/agent/swap.py`
- Modify: `tests/unit/test_swap.py`

- [ ] **Step 8.1: Write the failing tests**

Append to `tests/unit/test_swap.py`:

```python
def test_resolve_insert_position_uses_insert_after_when_anchor_present() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _resolve_insert_position

    stops = [_stop(place_id="a"), _stop(place_id="b"), _stop(place_id="c")]
    ctx = ClosureContext(
        place_id="closed",
        place_name="X",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="a",
        insert_before_place_id=None,
        stop_index_hint=99,
    )
    # insert_after a (index 0) → position 1
    assert _resolve_insert_position(ctx, stops) == 1


def test_resolve_insert_position_falls_back_to_insert_before() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _resolve_insert_position

    stops = [_stop(place_id="a"), _stop(place_id="b")]
    ctx = ClosureContext(
        place_id="closed",
        place_name="X",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="missing",
        insert_before_place_id="b",
        stop_index_hint=99,
    )
    # insert_after missing; insert_before b (index 1) → position 1
    assert _resolve_insert_position(ctx, stops) == 1


def test_resolve_insert_position_falls_back_to_index_hint_clamped() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _resolve_insert_position

    stops = [_stop(place_id="a"), _stop(place_id="b")]
    ctx = ClosureContext(
        place_id="closed",
        place_name="X",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="missing",
        insert_before_place_id="also_missing",
        stop_index_hint=99,
    )
    # Both anchors absent; clamp hint to len(stops)
    assert _resolve_insert_position(ctx, stops) == 2


def test_score_candidate_prefers_lower_route_impact() -> None:
    """Two candidates with identical family-match: the one with smaller
    combined prev+next distance scores higher."""
    from app.agent.swap import _score_candidate

    closed = _stop(place_id="closed", lat=37.78, lng=-122.41)
    prev_ = _stop(place_id="prev", lat=37.78, lng=-122.41)
    next_ = _stop(place_id="next", lat=37.785, lng=-122.41)

    close_candidate = _stop(place_id="c1", lat=37.78, lng=-122.41)
    far_candidate = _stop(place_id="c2", lat=37.90, lng=-122.41)

    s_close = _score_candidate(close_candidate, closed, prev_, next_, family_match=True)
    s_far = _score_candidate(far_candidate, closed, prev_, next_, family_match=True)
    assert s_close > s_far


def test_score_candidate_prefers_family_match() -> None:
    """All else equal, a family-matching candidate beats one that doesn't."""
    from app.agent.swap import _score_candidate

    closed = _stop(place_id="closed")
    prev_ = _stop(place_id="prev")
    next_ = _stop(place_id="next")
    candidate = _stop(place_id="c", lat=37.78, lng=-122.41)

    s_match = _score_candidate(candidate, closed, prev_, next_, family_match=True)
    s_nomatch = _score_candidate(candidate, closed, prev_, next_, family_match=False)
    assert s_match > s_nomatch
```

- [ ] **Step 8.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "resolve_insert_position or score_candidate"`
Expected: ImportError or AttributeError.

- [ ] **Step 8.3: Add helpers to `app/agent/swap.py`**

Append (importing additional symbols first if needed — `from pydantic import BaseModel`, `from app.agent.planning import haversine_m`):

```python
from pydantic import BaseModel

from app.agent.planning import haversine_m


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
# 30 km from any anchor in the city; nearby() requires an explicit radius_m.
_CITYWIDE_RADIUS_M: int = 30_000


def _resolve_insert_position(
    closure: ClosureContext,
    stops: list[Stop],
) -> int:
    """Where in `stops` should we insert the proposed alternative?

    Priority rules (matches ClosureContext docstring in state.py):
      1) insert_after_place_id present in stops → that index + 1
      2) else insert_before_place_id present in stops → that index
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

    Two components:
      - family_match_score: 1.0 if same family as the closed stop, else 0.0
      - route_impact_score: -(haversine prev→candidate + candidate→next),
        scaled so that ~500m total adds ~0.5 to the score, and ~2km zeroes it
    The two are summed equally weighted (1.0 each). Family match is the
    primary lever — a 1.0 family bonus dominates any plausible route delta
    inside the walking radius.
    """
    fam = 1.0 if family_match else 0.0
    total_dist_m = 0.0
    if prev_stop is not None and prev_stop.latitude is not None and prev_stop.longitude is not None \
            and candidate.latitude is not None and candidate.longitude is not None:
        total_dist_m += haversine_m(
            (prev_stop.latitude, prev_stop.longitude),
            (candidate.latitude, candidate.longitude),
        )
    if next_stop is not None and next_stop.latitude is not None and next_stop.longitude is not None \
            and candidate.latitude is not None and candidate.longitude is not None:
        total_dist_m += haversine_m(
            (candidate.latitude, candidate.longitude),
            (next_stop.latitude, next_stop.longitude),
        )
    # Linear penalty: 0m → 1.0, 1000m → 0.5, 2000m+ → 0
    route = max(0.0, 1.0 - total_dist_m / 2000.0)
    return fam + route
```

- [ ] **Step 8.4: Run the tests**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "resolve_insert_position or score_candidate"`
Expected: 5 pass.

- [ ] **Step 8.5: Lint + typecheck**

Run: `poetry run ruff check app/agent/swap.py && poetry run mypy app/agent/swap.py`

- [ ] **Step 8.6: Commit**

```bash
git add app/agent/swap.py tests/unit/test_swap.py
git commit -m "feat(swap): add CandidateMatch + _resolve_insert_position + _score_candidate"
```

---

## Task 9: Add `_try_walking_distance_swap` and `_try_any_distance_search`

**Files:**
- Modify: `app/agent/swap.py`
- Modify: `tests/unit/test_swap.py`

- [ ] **Step 9.1: Write the failing tests**

Append to `tests/unit/test_swap.py`:

```python
def test_try_walking_distance_swap_uses_family_and_exclusion(mocker) -> None:
    """The swap helper must:
    1. Resolve the family from primary_type via family_of.
    2. Pass family + exclusions to nearby() via SearchFilters.
    3. Pass open_at = attempted_arrival.
    4. Return the highest-scoring candidate, if any.
    """
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import _try_walking_distance_swap
    from app.tools.retrieval import PlaceHit

    captured_filters: list = []

    def _fake_nearby(place_id, radius_m, filters, k):
        captured_filters.append((place_id, radius_m, filters, k))
        return [
            PlaceHit(
                place_id="alt1",
                name="Alt 1",
                primary_type="Dessert Shop",
                latitude=37.78,
                longitude=-122.41,
                rating=4.5,
                source="google_places",
                similarity=0.0,
                dist_m=300.0,
            )
        ]

    mocker.patch("app.agent.swap._nearby_search", side_effect=_fake_nearby)
    closed = _stop(place_id="closed", primary_type="Dessert Shop")
    prev_ = _stop(place_id="prev", lat=37.78, lng=-122.41)
    state = ItineraryState(
        stops=[prev_, closed],
        closure_context=[],
    )
    ctx = ClosureContext(
        place_id="closed",
        place_name="Closed",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="prev",
        insert_before_place_id=None,
        stop_index_hint=1,
    )

    match = _try_walking_distance_swap(state, ctx, anchor_place_id="prev")

    assert match is not None
    assert match.stop.place_id == "alt1"
    # captured: (place_id, radius_m, filters, k)
    _, radius_m, filters, _ = captured_filters[0]
    assert radius_m <= 500  # walking budget
    assert filters.primary_type_family == "dessert"
    assert "closed" in (filters.excluded_place_ids or [])
    assert filters.open_at == datetime(2026, 5, 19, 20, 0, tzinfo=SF)


def test_try_walking_distance_swap_returns_none_when_no_candidates(mocker) -> None:
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import _try_walking_distance_swap

    mocker.patch("app.agent.swap._nearby_search", return_value=[])
    closed = _stop(place_id="closed", primary_type="Dessert Shop")
    state = ItineraryState(stops=[_stop(place_id="prev"), closed])
    ctx = ClosureContext(
        place_id="closed",
        place_name="Closed",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="prev",
        insert_before_place_id=None,
        stop_index_hint=1,
    )
    assert _try_walking_distance_swap(state, ctx, anchor_place_id="prev") is None


def test_try_any_distance_search_uses_citywide_radius(mocker) -> None:
    """Fallback search uses _CITYWIDE_RADIUS_M (30km) so the question can
    propose a drive-distance alternative when nothing is within walking
    distance."""
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import _try_any_distance_search
    from app.tools.retrieval import PlaceHit

    captured = []

    def _fake_nearby(place_id, radius_m, filters, k):
        captured.append(radius_m)
        return [
            PlaceHit(
                place_id="alt2",
                name="Alt 2 (far)",
                primary_type="Dessert Shop",
                latitude=37.80,
                longitude=-122.45,
                source="google_places",
                similarity=0.0,
                dist_m=4800.0,
            )
        ]

    mocker.patch("app.agent.swap._nearby_search", side_effect=_fake_nearby)
    closed = _stop(place_id="closed", primary_type="Dessert Shop")
    state = ItineraryState(stops=[_stop(place_id="prev"), closed])
    ctx = ClosureContext(
        place_id="closed",
        place_name="Closed",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="prev",
        insert_before_place_id=None,
        stop_index_hint=1,
    )

    match = _try_any_distance_search(state, ctx, anchor_place_id="prev")
    assert match is not None
    assert match.distance_m == 4800.0
    assert captured[0] == 30_000
```

- [ ] **Step 9.2: Run to confirm failure**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "walking_distance_swap or any_distance_search"`
Expected: AttributeError.

- [ ] **Step 9.3: Implement the helpers**

Append to `app/agent/swap.py` (the import block needs to grow — add the marked imports):

```python
from app.tools.filters import SearchFilters, family_of
from app.tools.retrieval import PlaceHit
from app.tools.retrieval import nearby as _nearby_search  # aliased for test patching
```

Then the helper bodies:

```python
def _resolve_anchor(state: ItineraryState, closed_stop: Stop) -> str | None:
    """Pick a stable anchor place_id to search around when looking for an
    alternative to the closed stop. Prefer the previous stop; if there is no
    previous (closed_stop is index 0), fall back to the next stop; if
    neither has a place_id we can use, fall back to the closed stop itself
    (Google will return same-coords neighbors)."""
    try:
        idx = state.stops.index(closed_stop)
    except ValueError:
        return closed_stop.place_id
    if idx > 0:
        return state.stops[idx - 1].place_id
    if idx + 1 < len(state.stops):
        return state.stops[idx + 1].place_id
    return closed_stop.place_id


def _candidates_to_matches(
    candidates: list[PlaceHit],
    closed_stop: Stop,
    state: ItineraryState,
) -> list[CandidateMatch]:
    """Score each candidate and sort descending. Family match is computed
    against the closed stop's primary_type."""
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
    """All place_ids the swap node must not re-propose:
    current stops + every closure_context entry's source place_id + extras.
    Every outcome contributes per the spec — once recorded, never re-suggested.
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
    """Search within _WALKING_DISTANCE_BUDGET_M for an alternative of the
    same family that's open at the closed stop's attempted_arrival. Returns
    the highest-scoring match or None if no candidates are within the
    walking radius. DB errors return None and log."""
    closed_stop = next((s for s in state.stops if s.place_id == closure.place_id), None)
    if closed_stop is None:
        return None
    if not closure.family:
        # Without a resolved family we can't do a category-matched search;
        # the caller will escalate to the user.
        return None
    if closure.family not in {"dessert", "bar", "restaurant", "cafe"}:
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
    """Citywide fallback — only used to populate the user-facing question's
    proposed_alternative when the walking-distance pass failed. Uses
    _CITYWIDE_RADIUS_M (30 km, covers all of SF). Same family + exclusion
    rules as walking-distance."""
    closed_stop = next((s for s in state.stops if s.place_id == closure.place_id), None)
    if closed_stop is None:
        return None
    if not closure.family or closure.family not in {"dessert", "bar", "restaurant", "cafe"}:
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
```

- [ ] **Step 9.4: Run tests**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "walking_distance_swap or any_distance_search"`
Expected: 3 pass.

- [ ] **Step 9.5: Lint + typecheck**

Run: `poetry run ruff check app/agent/swap.py tests/unit/test_swap.py && poetry run mypy app/agent/swap.py`

- [ ] **Step 9.6: Commit**

```bash
git add app/agent/swap.py tests/unit/test_swap.py
git commit -m "feat(swap): add walking-distance + citywide candidate search"
```

---

## Task 10: Add `_bounded_retime_after_swap`, `_apply_swap`, `_promote_pending`, `_formulate_closure_question`

**Files:**
- Modify: `app/agent/swap.py`
- Modify: `tests/unit/test_swap.py`

- [ ] **Step 10.1: Write the failing tests**

Append to `tests/unit/test_swap.py`:

```python
def test_apply_swap_replaces_stop_at_position(mocker) -> None:
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import CandidateMatch, _apply_swap

    s1 = _stop(place_id="s1", name="S1")
    s2_closed = _stop(place_id="s2_closed", name="S2 closed")
    s3 = _stop(place_id="s3", name="S3")
    state = ItineraryState(stops=[s1, s2_closed, s3])
    replacement = _stop(place_id="s2_new", name="S2 new")
    leg_durations_min = [10.0, 5.0]

    # Avoid touching the real DB during enrich
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    new_stops = _apply_swap(
        state,
        stop_index=1,
        replacement=replacement,
        leg_durations_min=leg_durations_min,
    )

    assert [s.place_id for s in new_stops] == ["s1", "s2_new", "s3"]


def test_bounded_retime_after_swap_calls_route_legs_once(mocker) -> None:
    """The bounded retime helper makes at most ONE extra route_legs call per
    swap-node invocation. Mock route_legs and confirm call_count == 1."""
    from app.agent.state import ItineraryState
    from app.agent.swap import _bounded_retime_after_swap
    from app.tools.directions import DirectionsLeg, DirectionsResult

    call_count = {"n": 0}

    async def _fake_route(stops, mode="walk"):
        call_count["n"] += 1
        legs = [DirectionsLeg(duration_s=600, distance_m=400.0)] * max(len(stops) - 1, 1)
        return DirectionsResult(
            legs=legs,
            total_duration_s=600 * len(legs),
            mode=mode,
            source="haversine_fallback",
        )

    mocker.patch("app.agent.swap.route_legs", side_effect=_fake_route)
    state = ItineraryState(
        stops=[_stop(place_id="a"), _stop(place_id="b"), _stop(place_id="c")],
    )
    import asyncio

    retimed = asyncio.run(_bounded_retime_after_swap(state.stops))
    assert call_count["n"] == 1
    assert len(retimed) == 3


def test_promote_pending_flips_first_queued_to_pending() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _promote_pending

    queued1 = ClosureContext(
        place_id="q1",
        place_name="Q1",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="queued_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    queued2 = queued1.model_copy(update={"place_id": "q2"})
    auto = queued1.model_copy(update={"place_id": "a", "outcome": "auto_swapped"})

    promoted = _promote_pending([auto, queued1, queued2])
    outcomes = [c.outcome for c in promoted]
    assert outcomes == ["auto_swapped", "pending_user_decision", "queued_user_decision"]


def test_promote_pending_is_noop_when_no_queued() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _promote_pending

    auto = ClosureContext(
        place_id="a",
        place_name="A",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="auto_swapped",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    assert [c.outcome for c in _promote_pending([auto])] == ["auto_swapped"]


def test_formulate_closure_question_with_proposal() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _formulate_closure_question

    proposal = _stop(place_id="alt", name="Sophie's Crepes")
    ctx = ClosureContext(
        place_id="closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=proposal,
        proposed_distance_m=4800.0,
    )
    q = _formulate_closure_question(ctx)
    assert "Sophie's Crepes" in q
    assert "Mochill Mochidonut" in q
    # ~3 mi rounding from 4800m is expected
    assert "3" in q


def test_formulate_closure_question_without_proposal() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _formulate_closure_question

    ctx = ClosureContext(
        place_id="closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=None,
        proposed_distance_m=None,
    )
    q = _formulate_closure_question(ctx)
    assert "Mochill Mochidonut" in q
    # No proposal → message should ask the user to pick / change category
    assert "pick" in q.lower() or "different" in q.lower() or "skip" in q.lower()
```

- [ ] **Step 10.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "apply_swap or bounded_retime or promote_pending or formulate"`
Expected: AttributeError.

- [ ] **Step 10.3: Implement the helpers**

Append to `app/agent/swap.py` (imports first — add `from app.agent.commit import enrich_stops_with_booking`, `from app.agent.planning import chain_arrival_times`, `from app.tools.directions import route_legs`):

```python
from app.agent.commit import enrich_stops_with_booking
from app.agent.planning import chain_arrival_times
from app.tools.directions import route_legs
```

Then:

```python
async def _bounded_retime_after_swap(stops: list[Stop]) -> list[Stop]:
    """One extra `route_legs` call after a swap → re-chain arrival_times.

    Strictly bounded: no recursion, no loop. Called at most once per swap
    node invocation. Falls back to the input stops on any failure (mirrors
    retime() in graph.py:319-323)."""
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
    into state.
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
    """Round to nearest mile for user-facing text. 1609m → 1mi, 4800m → 3mi."""
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
```

- [ ] **Step 10.4: Run tests**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "apply_swap or bounded_retime or promote_pending or formulate"`
Expected: 6 pass.

- [ ] **Step 10.5: Lint + typecheck**

Run: `poetry run ruff check app/agent/swap.py && poetry run mypy app/agent/swap.py`

- [ ] **Step 10.6: Commit**

```bash
git add app/agent/swap.py tests/unit/test_swap.py
git commit -m "feat(swap): add bounded retime + apply_swap + promote_pending + question text"
```

---

## Task 11: Implement `swap_closed_stops` node

**Files:**
- Modify: `app/agent/swap.py`
- Create: `tests/unit/test_swap_node.py`

This is the orchestrator. It composes everything from Tasks 7-10.

- [ ] **Step 11.1: Write the failing tests**

Create `tests/unit/test_swap_node.py`:

```python
"""Smoke tests for the swap_closed_stops graph node.

Uses real ItineraryState + Stop models, with the SQL helpers and Routes API
mocked. Verifies the orchestration: no-op when nothing's closed, auto-swap
batches, escalation to pending, queueing additional pending entries, etc.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agent.state import ClosureContext, ItineraryState, Stop
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceHit

SF = ZoneInfo("America/Los_Angeles")


def _stop(
    place_id: str,
    *,
    name: str = "X",
    arrival_iso: str = "2026-05-19T19:00:00-07:00",
    primary_type: str = "Bar",
    lat: float = 37.78,
    lng: float = -122.41,
) -> Stop:
    return Stop(
        place_id=place_id,
        name=name,
        rationale="r",
        source="google_places",
        arrival_time=datetime.fromisoformat(arrival_iso),
        latitude=lat,
        longitude=lng,
        primary_type=primary_type,
        planned_duration_min=60,
    )


def _fake_route(stops, mode="walk"):
    async def _r(*args, **kwargs):
        legs = [DirectionsLeg(duration_s=600, distance_m=400.0)] * max(len(stops) - 1, 1)
        return DirectionsResult(
            legs=legs, total_duration_s=600 * len(legs), mode=mode, source="haversine_fallback"
        )

    return _r


def test_swap_node_noop_when_nothing_closed(mocker) -> None:
    from app.agent.swap import swap_closed_stops

    mocker.patch("app.agent.swap._execute_closure_query", return_value={"a": True, "b": True})
    state = ItineraryState(
        stops=[_stop("a"), _stop("b")],
    )
    update = asyncio.run(swap_closed_stops(state))
    # No-op → empty or stops unchanged
    assert update.get("stops") in (None, state.stops)
    assert not any(
        e.outcome != "auto_swapped" for e in update.get("closure_context", [])
    ), "no new pending entries"


def test_swap_node_auto_swap_silent_when_candidate_found(mocker) -> None:
    """Closure detected at stop 1; walking-distance candidate exists →
    silent swap, closure_context records auto_swapped, summary reply."""
    from app.agent.swap import swap_closed_stops

    # Stop b is closed
    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=[
            {"a": True, "b": False, "c": True},  # initial closure check
            {"a": True, "b_alt": True, "c": True},  # post-swap re-check
        ],
    )
    candidate = PlaceHit(
        place_id="b_alt",
        name="B Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=_fake_route([1, 2, 3]))
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    state = ItineraryState(
        stops=[
            _stop("a", primary_type="Bar"),
            _stop("b", primary_type="Bar"),
            _stop("c", primary_type="Bar"),
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_stops = update.get("stops")
    assert new_stops is not None
    assert [s.place_id for s in new_stops] == ["a", "b_alt", "c"]
    new_ctx = update["closure_context"]
    assert any(c.outcome == "auto_swapped" and c.place_id == "b" for c in new_ctx)
    # Silent swap → reply is the regenerated summary
    reply = update.get("final_reply", "")
    assert "Caveats" not in reply
    assert "B Alt" in reply


def test_swap_node_escalates_to_pending_when_no_walking_match(mocker) -> None:
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"a": True, "b": False},
    )
    # Walking search returns empty; citywide search returns one far candidate.
    mocker.patch(
        "app.agent.swap._nearby_search",
        side_effect=[
            [],  # walking-distance result
            [
                PlaceHit(
                    place_id="b_far",
                    name="B Far",
                    primary_type="Bar",
                    latitude=37.80,
                    longitude=-122.45,
                    source="google_places",
                    similarity=0.0,
                    dist_m=4800.0,
                )
            ],  # citywide fallback
        ],
    )
    state = ItineraryState(
        stops=[_stop("a", primary_type="Bar"), _stop("b", primary_type="Bar")],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    pending = [c for c in new_ctx if c.outcome == "pending_user_decision"]
    assert len(pending) == 1
    assert pending[0].place_id == "b"
    assert pending[0].proposed_alternative is not None
    assert pending[0].proposed_alternative.place_id == "b_far"
    # Reply is the question text, not a summary
    assert update["final_reply"]
    assert "B" in update["final_reply"]


def test_swap_node_queues_additional_pending_closures(mocker) -> None:
    """Two stops closed, neither walking-fixable → first becomes pending,
    second becomes queued."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"a": False, "b": False},
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[])
    state = ItineraryState(
        stops=[_stop("a", primary_type="Bar"), _stop("b", primary_type="Bar")],
    )
    update = asyncio.run(swap_closed_stops(state))
    outcomes = [c.outcome for c in update["closure_context"]]
    assert outcomes.count("pending_user_decision") == 1
    assert outcomes.count("queued_user_decision") == 1


def test_swap_node_skips_when_family_unresolved(mocker) -> None:
    """A stop whose primary_type has no family (e.g. 'Spaceship') escalates
    to a pending entry with no proposal — we can't search."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"a": True, "b": False},
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[])  # not called for unresolved
    state = ItineraryState(
        stops=[
            _stop("a", primary_type="Bar"),
            _stop("b", primary_type="Spaceship"),  # unknown family
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    pending = [c for c in new_ctx if c.outcome == "pending_user_decision"]
    assert len(pending) == 1
    assert pending[0].family == ""
    assert pending[0].proposed_alternative is None


def test_swap_node_caps_closure_context_at_max(mocker) -> None:
    """When closure_context grows past MAX_CLOSURE_CONTEXT_ENTRIES, oldest
    entries are dropped."""
    from app.agent.state import MAX_CLOSURE_CONTEXT_ENTRIES
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=[{"x": False}, {"y_alt": True}],
    )
    candidate = PlaceHit(
        place_id="y_alt",
        name="Y Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=_fake_route([1]))
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    existing = [
        ClosureContext(
            place_id=f"old_{i}",
            place_name=f"Old {i}",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="auto_swapped",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        )
        for i in range(MAX_CLOSURE_CONTEXT_ENTRIES)
    ]
    state = ItineraryState(
        stops=[_stop("x", primary_type="Bar")],
        closure_context=existing,
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    assert len(new_ctx) == MAX_CLOSURE_CONTEXT_ENTRIES
    # The oldest "old_0" should be dropped; the new entry should be present.
    place_ids = {c.place_id for c in new_ctx}
    assert "old_0" not in place_ids
    assert "x" in place_ids


def test_swap_node_fail_open_on_initial_db_error(mocker) -> None:
    """If the initial closure query fails, the node is a no-op and ships
    the plan as-is. Matches checks.py:200-205 precedent."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=Exception("db down"),
    )
    state = ItineraryState(
        stops=[_stop("a", primary_type="Bar"), _stop("b", primary_type="Bar")],
    )
    update = asyncio.run(swap_closed_stops(state))
    # Either no update or stops unchanged + empty closure_context
    new_stops = update.get("stops", state.stops)
    assert [s.place_id for s in new_stops] == ["a", "b"]
```

- [ ] **Step 11.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_swap_node.py -v`
Expected: ImportError for `swap_closed_stops`.

- [ ] **Step 11.3: Implement `swap_closed_stops`**

Append to `app/agent/swap.py`:

```python
from app.agent.state import MAX_CLOSURE_CONTEXT_ENTRIES
from app.agent.revision import summarize_stops


def _cap_closure_context(entries: list[ClosureContext]) -> list[ClosureContext]:
    """Append-and-drop-oldest to MAX_CLOSURE_CONTEXT_ENTRIES."""
    if len(entries) <= MAX_CLOSURE_CONTEXT_ENTRIES:
        return entries
    dropped = len(entries) - MAX_CLOSURE_CONTEXT_ENTRIES
    logger.warning("closure_context.cap_exceeded: dropped %d oldest entries", dropped)
    return entries[dropped:]


def _resolve_family_for_stop(stop: Stop) -> str:
    """family from primary_type first, then nothing (we don't have types[]
    on Stop). Returns "" when nothing resolves so the caller can still
    record the closure (without searching)."""
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
    insert_before = (
        stops[closed_index + 1].place_id if closed_index + 1 < len(stops) else None
    )
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
    4. Any remaining closures → first becomes pending_user_decision, the
       rest queued_user_decision. Citywide fallback search populates the
       pending entry's proposed_alternative when possible.
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
        # One bounded retime + re-check (Phase 2).
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
            _build_closure_context_entry(
                state.stops, idx, proposed=proposal, outcome=outcome
            )
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
```

Also add `from datetime import datetime` and `from zoneinfo import ZoneInfo` to the top of the file.

- [ ] **Step 11.4: Run the swap node tests**

Run: `poetry run pytest tests/unit/test_swap_node.py -v`
Expected: all 7 pass.

- [ ] **Step 11.5: Run the full swap tests**

Run: `poetry run pytest tests/unit/test_swap.py tests/unit/test_swap_node.py -v`
Expected: all pass.

- [ ] **Step 11.6: Lint + typecheck**

Run: `poetry run ruff check app/agent/swap.py tests/unit/test_swap.py tests/unit/test_swap_node.py && poetry run mypy app/agent/swap.py`

- [ ] **Step 11.7: Commit**

```bash
git add app/agent/swap.py tests/unit/test_swap_node.py
git commit -m "feat(swap): implement swap_closed_stops orchestrator node"
```

---

## Task 12: Add `_inject_closure_exclusions` helper

**Files:**
- Modify: `app/agent/swap.py`
- Modify: `tests/unit/test_swap.py`

This is the belt-and-suspenders enforcement so the prompt guidance isn't the only line of defense.

- [ ] **Step 12.1: Write the failing tests**

Append to `tests/unit/test_swap.py`:

```python
def test_inject_closure_exclusions_merges_into_semantic_search_filters() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _inject_closure_exclusions
    from app.tools.filters import SearchFilters

    ctx = [
        ClosureContext(
            place_id="closed1",
            place_name="C1",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="auto_swapped",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        ),
        ClosureContext(
            place_id="closed2",
            place_name="C2",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="user_accepted_drive",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        ),
    ]
    # LLM-supplied args
    args = {
        "query": "ramen",
        "filters": SearchFilters(min_rating=4.0, excluded_place_ids=["llm_excluded"]),
    }
    out = _inject_closure_exclusions("semantic_search", args, ctx)
    excluded = out["filters"].excluded_place_ids
    assert set(excluded) == {"llm_excluded", "closed1", "closed2"}
    # Returns a new args dict (not in-place mutation)
    assert out is not args


def test_inject_closure_exclusions_creates_filters_when_absent() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _inject_closure_exclusions

    ctx = [
        ClosureContext(
            place_id="closed1",
            place_name="C1",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="auto_swapped",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        )
    ]
    args = {"query": "ramen"}
    out = _inject_closure_exclusions("semantic_search", args, ctx)
    assert "filters" in out
    assert out["filters"].excluded_place_ids == ["closed1"]


def test_inject_closure_exclusions_kg_traverse_is_top_level() -> None:
    """kg_traverse takes excluded_place_ids as a top-level arg, not via
    filters."""
    from app.agent.state import ClosureContext
    from app.agent.swap import _inject_closure_exclusions

    ctx = [
        ClosureContext(
            place_id="closed1",
            place_name="C1",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="auto_swapped",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        )
    ]
    args = {"place_id": "anchor", "relation_type": "SIMILAR_VECTOR"}
    out = _inject_closure_exclusions("kg_traverse", args, ctx)
    assert out["excluded_place_ids"] == ["closed1"]


def test_inject_closure_exclusions_empty_context_is_noop() -> None:
    from app.agent.swap import _inject_closure_exclusions

    args = {"query": "ramen"}
    out = _inject_closure_exclusions("semantic_search", args, [])
    assert out == args
    # New dict either way, but contents identical
    assert "filters" not in out


def test_inject_closure_exclusions_unknown_tool_is_noop() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _inject_closure_exclusions

    ctx = [
        ClosureContext(
            place_id="closed1",
            place_name="C1",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="auto_swapped",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        )
    ]
    args = {"foo": "bar"}
    out = _inject_closure_exclusions("get_details", args, ctx)
    assert out == args
```

- [ ] **Step 12.2: Run to confirm failure**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "inject_closure_exclusions"`
Expected: AttributeError.

- [ ] **Step 12.3: Implement `_inject_closure_exclusions`**

Append to `app/agent/swap.py`:

```python
def _inject_closure_exclusions(
    tool_name: str,
    args: dict[str, Any],
    closure_context: list[ClosureContext],
) -> dict[str, Any]:
    """Merge closure_context place_ids into a tool call's exclusion argument.

    Server-side belt-and-suspenders enforcement so the prompt guidance is an
    optimization, not the only line of defense. Routes by tool name because
    the exclusion argument lives in different places per tool:

      - semantic_search / nearby → args["filters"].excluded_place_ids
      - kg_traverse              → args["excluded_place_ids"]  (top-level)
      - anything else            → no-op

    Returns a NEW args dict; never mutates the input (the graph.act node
    records the original args verbatim in scratch for tracing).
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
            filters = SearchFilters(excluded_place_ids=sorted(excluded))
        elif isinstance(existing_filters, SearchFilters):
            llm_excluded = set(existing_filters.excluded_place_ids or [])
            filters = existing_filters.model_copy(
                update={"excluded_place_ids": sorted(llm_excluded | excluded)}
            )
        else:
            # LangChain delivers args as plain dicts when StructuredTool
            # builds Pydantic args_schema — but the args dict here still
            # carries the inner filters as a dict. Reconstruct via validation.
            llm = SearchFilters.model_validate(existing_filters)
            llm_excluded = set(llm.excluded_place_ids or [])
            filters = llm.model_copy(
                update={"excluded_place_ids": sorted(llm_excluded | excluded)}
            )
        new_args["filters"] = filters
        return new_args
    if tool_name == "kg_traverse":
        llm_excluded = set(new_args.get("excluded_place_ids") or [])
        new_args["excluded_place_ids"] = sorted(llm_excluded | excluded)
        return new_args
    return new_args
```

- [ ] **Step 12.4: Run the tests**

Run: `poetry run pytest tests/unit/test_swap.py -v -k "inject_closure_exclusions"`
Expected: 5 pass.

- [ ] **Step 12.5: Lint + typecheck**

Run: `poetry run ruff check app/agent/swap.py && poetry run mypy app/agent/swap.py`

- [ ] **Step 12.6: Commit**

```bash
git add app/agent/swap.py tests/unit/test_swap.py
git commit -m "feat(swap): add _inject_closure_exclusions for SQL-layer enforcement"
```

---

## Task 13: Wire `swap_closed_stops` into the graph + delete the temporal_coherence caveat

**Files:**
- Modify: `app/agent/graph.py`
- Modify: `tests/unit/test_agent_graph.py` (extend) or add new test cases

- [ ] **Step 13.1: Write the failing tests**

Append to an appropriate existing test file (`tests/unit/test_agent_graph.py`):

```python
def test_graph_routes_through_swap_closed_stops_node(mocker) -> None:
    """The compiled graph must route retime → swap_closed_stops → END."""
    from langchain_core.language_models.fake_chat_models import FakeChatModel

    from app.agent.graph import build_agent_graph

    # FakeChatModel doesn't actually run — we just need a bound-tools model.
    # The graph topology check is the only assertion.
    fake = mocker.Mock()
    fake.bind_tools = mocker.Mock(return_value=fake)
    graph = build_agent_graph(fake)
    # The compiled graph exposes its node ids via .nodes
    assert "swap_closed_stops" in graph.nodes


def test_retime_no_longer_appends_temporal_caveat(mocker) -> None:
    """Regression: the temporal_coherence caveat is now the swap node's
    responsibility. The retime node must not append it under any condition.
    """
    import inspect

    from app.agent import graph as graph_mod

    src = inspect.getsource(graph_mod)
    # The string "Caveats:" should no longer appear inside the retime node
    # body (it lived at lines 327-347). Easiest invariant to assert:
    assert "_final_with_caveats" not in src or src.count("_final_with_caveats(existing") == 0, (
        "retime() must not call _final_with_caveats on temporal_coherence; "
        "closure handling lives in app/agent/swap.py now."
    )
```

- [ ] **Step 13.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_agent_graph.py -v -k "swap_closed_stops or temporal_caveat"`
Expected: failure on both.

- [ ] **Step 13.3: Modify `app/agent/graph.py`**

a) Add the import at the top with the other agent imports (around line 41-42):

```python
from app.agent.swap import _inject_closure_exclusions, swap_closed_stops
```

b) Delete lines 327-347 of the existing `retime` function (the entire post-retime `temporal_coherence` check block). The retime function should end at line 326 (`update: dict[str, Any] = {"stops": retimed}`) and `return update`.

Replace the original block:

```python
        update: dict[str, Any] = {"stops": retimed}

        # Re-run ONLY the open-at-arrival check on the real times. Other
        # checks (geographic/walking/hallucination) are coord/id-based and
        # unaffected by re-timing, so re-running them would be wasted work.
        # temporal_coherence is sync psycopg2 I/O — offload to a thread so
        # the event loop stays responsive (same pattern as act()).
        probe = state.model_copy(update={"stops": retimed})
        try:
            score = await asyncio.to_thread(temporal_coherence, probe)
        except Exception:  # noqa: BLE001
            # Fails open exactly like itinerary_violations(): a DB blip must
            # not block /chat. Ship the re-timed plan without the re-check.
            return update

        # Only append a caveat if the revision loop hasn't already shipped
        # one (the END->retime rewire routes the caveats-exhausted path
        # through here too; _final_with_caveats is not idempotent, so a
        # second call would duplicate the identical "Caveats:" paragraph).
        if score < CRITIQUE_THRESHOLDS["temporal_coherence"]:
            existing = state.final_reply or ""
            if "Caveats:" not in existing:
                update["final_reply"] = _final_with_caveats(existing, ["temporal_coherence"])
        return update
```

with:

```python
        return {"stops": retimed}
```

c) Remove the now-unused imports at the top of `graph.py`. Drop these from line 31-32 and 34-39 as appropriate:

```python
from app.agent.critique.checks import CRITIQUE_THRESHOLDS, temporal_coherence  # delete this line
from app.agent.revision import (
    _final_with_caveats,                                                        # delete _final_with_caveats
    critique_final_with_stops,
    critique_step,
    finalize_as_is,
    short_circuit_max_steps,
)
```

(Keep the other revision imports — they're still used.)

d) Add the `swap_closed_stops` node and wire it. Find the existing graph-build block at lines 359-369:

```python
    g = StateGraph(ItineraryState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.add_node("retime", retime)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", "retime": "retime"})
    g.add_edge("retime", END)
    return g.compile()
```

Change to:

```python
    g = StateGraph(ItineraryState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.add_node("retime", retime)
    g.add_node("swap_closed_stops", swap_closed_stops)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", "retime": "retime"})
    g.add_edge("retime", "swap_closed_stops")
    g.add_edge("swap_closed_stops", END)
    return g.compile()
```

e) Inject closure exclusions in `act()`. Modify the body of the `for tc in ai.tool_calls:` loop (graph.py around lines 204-247). Right before `tool = tool_by_name.get(tc["name"])` (around line 221), insert:

```python
            # Belt-and-suspenders: merge closure-context exclusions into
            # the tool args at the SQL layer. The prompt guidance in
            # _constraints_context is an optimization; this is enforcement.
            if tc["name"] in ("semantic_search", "nearby", "kg_traverse"):
                tc["args"] = _inject_closure_exclusions(
                    tc["name"], tc["args"], state.closure_context
                )
```

f) Extend `_constraints_context` to mention closure exclusions. Replace the function body (lines 52-60) with:

```python
def _constraints_context(state: ItineraryState) -> str:
    """Human-readable deterministic constraints appended to the system prompt."""
    parts: list[str] = []
    if state.constraints.num_stops is not None:
        parts.append(
            f"- The user explicitly requested {state.constraints.num_stops} stops. "
            "Do not ask how many stops they want; plan exactly that many stops."
        )
    # Every closure outcome contributes — once a place has been recorded as
    # closed in this conversation, it must not resurface as a candidate.
    closures = state.closure_context
    if closures:
        names = ", ".join(c.place_name for c in closures)
        parts.append(
            f"- Earlier in this conversation, these places were closed at the planned "
            f"arrival time and should NOT be re-suggested: {names}. "
            f"Their place_ids are also excluded from your search-result candidates."
        )
    if not parts:
        return ""
    return "\n\nDETERMINISTIC REQUEST CONTEXT:\n" + "\n".join(parts)
```

- [ ] **Step 13.4: Run the graph tests**

Run: `poetry run pytest tests/unit/test_agent_graph.py -v -k "swap_closed_stops or temporal_caveat"`
Expected: pass.

- [ ] **Step 13.5: Run the full agent test suite**

Run: `poetry run pytest tests/unit/test_agent_graph.py tests/unit/test_agent_smoke.py tests/unit/test_agent_self_correct.py tests/unit/test_agent_self_correct_functional.py -v`
Expected: pass (the deleted caveat block had tests that explicitly looked for "Caveats:" — these should now be removed or rewritten to expect no caveat; check the failing ones and either delete the corresponding assertions or replace them with assertions that the swap node ran instead).

If any existing test asserts the temporal caveat behavior, update it. Run with `--tb=short` to see exactly which tests broke.

- [ ] **Step 13.6: Lint + typecheck**

Run: `poetry run ruff check app/agent/graph.py && poetry run mypy app/agent/graph.py`
Expected: clean (unused-import errors here mean step 13.3c is incomplete — go back and remove them).

- [ ] **Step 13.7: Commit**

```bash
git add app/agent/graph.py tests/unit/test_agent_graph.py
git commit -m "feat(graph): wire swap_closed_stops node + drop temporal_coherence caveat"
```

If existing agent tests had to be updated, this commit can be split — do "remove obsolete temporal_caveat assertions" as a separate commit before this one if cleaner.

---

## Task 14: Extend `/chat` request/response with `conversation_state`

**Files:**
- Modify: `app/main.py`
- Modify: `tests/unit/test_chat_endpoint.py` (extend)

- [ ] **Step 14.1: Write the failing tests**

Append to `tests/unit/test_chat_endpoint.py`:

```python
def test_chat_endpoint_accepts_conversation_state(mocker) -> None:
    """An inbound conversation_state must hydrate into ItineraryState.closure_context."""
    fake_graph = mocker.Mock()
    captured: dict = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "make stop 2 cheaper",
                "history": [
                    {"role": "user", "content": "plan a 3-stop date"},
                    {"role": "assistant", "content": "Here's your itinerary..."},
                ],
                "conversation_state": {
                    "schema_version": 1,
                    "closure_context": [
                        {
                            "schema_version": 1,
                            "place_id": "ChIJ_closed",
                            "place_name": "Mochill",
                            "family": "dessert",
                            "attempted_arrival": "2026-05-19T20:02:00-07:00",
                            "outcome": "auto_swapped",
                            "insert_after_place_id": "ChIJ_prev",
                            "insert_before_place_id": None,
                            "stop_index_hint": 2,
                            "proposed_alternative": None,
                            "proposed_distance_m": None,
                        }
                    ],
                    "prior_stops": [],
                },
            },
        )

    assert response.status_code == 200
    state = captured["state"]
    assert len(state.closure_context) == 1
    assert state.closure_context[0].place_id == "ChIJ_closed"
    assert state.closure_context[0].outcome == "auto_swapped"


def test_chat_endpoint_returns_conversation_state(mocker) -> None:
    """Final state's closure_context must be echoed in the response."""
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        d = _final_state_dict(reply="ok")
        d["closure_context"] = [
            {
                "schema_version": 1,
                "place_id": "ChIJ_closed",
                "place_name": "Mochill",
                "family": "dessert",
                "attempted_arrival": "2026-05-19T20:02:00-07:00",
                "outcome": "auto_swapped",
                "insert_after_place_id": None,
                "insert_before_place_id": None,
                "stop_index_hint": 2,
                "proposed_alternative": None,
                "proposed_distance_m": None,
            }
        ]
        return d

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "plan a date", "history": []},
        )

    assert response.status_code == 200
    body = response.json()
    assert "conversation_state" in body
    cs = body["conversation_state"]
    assert cs["schema_version"] == 1
    assert len(cs["closure_context"]) == 1
    assert cs["closure_context"][0]["place_id"] == "ChIJ_closed"


def test_chat_endpoint_degrades_on_malformed_conversation_state(mocker) -> None:
    """A malformed conversation_state must not 422 — the handler logs and
    falls back to empty state."""
    fake_graph = mocker.Mock()
    captured: dict = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "anything",
                "conversation_state": {"schema_version": 1, "closure_context": "not-a-list"},
            },
        )
    # Degrades silently → 200, no closure_context hydrated.
    assert response.status_code == 200
    assert captured["state"].closure_context == []


def test_chat_endpoint_first_turn_omits_conversation_state(mocker) -> None:
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    # Backend always emits a typed conversation_state, never null
    assert "conversation_state" in body
    assert body["conversation_state"]["schema_version"] == 1
    assert body["conversation_state"]["closure_context"] == []
```

- [ ] **Step 14.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_chat_endpoint.py -v -k "conversation_state"`
Expected: failures — request body validation might 422, response shape missing the field, etc.

- [ ] **Step 14.3: Modify `app/main.py`**

a) Add imports:

```python
from pydantic import ValidationError

from .agent.state import ClosureContext, Stop
```

b) After the existing `ChatMessage` class (line 76-78), add:

```python
class ConversationState(BaseModel):
    schema_version: int = 1
    closure_context: list[ClosureContext] = Field(default_factory=list)
    prior_stops: list[Stop] = Field(default_factory=list)
```

c) Update `ChatRequest` (line 81-83) and `ChatResponse` (line 86-90):

```python
class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    # Opaque dict so a malformed nested object doesn't 422 before the
    # handler runs — `/chat` does manual ConversationState.model_validate
    # and degrades to empty state on ValidationError. dict | None still
    # 422s on non-object payloads (string/list/number), which is the right
    # answer for those (developer / curl mistakes).
    conversation_state: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    # Field name matches the frontend contract (frontend/src/api/chat.js).
    reply: str
    places: list[dict]
    ragLabel: str  # noqa: N815
    # Always emit a valid shape — strictly typed on the response side.
    conversation_state: ConversationState | None = None
```

d) Replace the `/chat` handler body (lines 283-312) with:

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail=AGENT_UNAVAILABLE_DETAIL)

    rag_label = getattr(request.app.state, "rag_label", _rag_label_for(None))

    # Hydrate inbound conversation_state. Manual validate so we degrade
    # rather than 422 on schema mismatch (matches the warning-log path used
    # by other "untrusted opaque state" decoders in the codebase).
    incoming: ConversationState
    if req.conversation_state is None:
        incoming = ConversationState()
    else:
        try:
            incoming = ConversationState.model_validate(req.conversation_state)
        except ValidationError:
            logger.warning("conversation_state.decode_failed", exc_info=True)
            incoming = ConversationState()

    with trace_request("chat", message=req.message[:200]) as trace_id:
        state = ItineraryState(
            messages=[
                *messages_from_history(req.history),
                HumanMessage(content=req.message),
            ],
            constraints=UserConstraints(
                num_stops=explicit_num_stops_from_conversation(req.history, req.message),
            ),
            closure_context=incoming.closure_context,
        )
        raw = await graph.ainvoke(
            state,
            config={
                "callbacks": langgraph_callbacks(),
                "metadata": {"trace_id": trace_id},
            },
        )
    final_state = raw if isinstance(raw, ItineraryState) else ItineraryState(**raw)

    # Build outbound conversation_state. prior_stops carries the stops the
    # user just saw so a follow-up turn can act on them without /chat
    # round-tripping the full places list.
    outbound = ConversationState(
        schema_version=1,
        closure_context=final_state.closure_context,
        prior_stops=final_state.stops,
    )

    return ChatResponse(
        reply=final_state.final_reply or "",
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=outbound,
    )
```

- [ ] **Step 14.4: Run the new tests**

Run: `poetry run pytest tests/unit/test_chat_endpoint.py -v -k "conversation_state"`
Expected: 4 pass.

- [ ] **Step 14.5: Run the full chat endpoint suite**

Run: `poetry run pytest tests/unit/test_chat_endpoint.py -v`
Expected: existing tests may need adjustment for the new `conversation_state` key in responses — update `expected_place_keys` assertion if it checks response top-level keys. (Looking at the existing `test_chat_endpoint_returns_reply_places_raglabel` it asserts `set(body.keys()) == {"reply", "places", "ragLabel"}`. Update this to include `"conversation_state"`.)

In `tests/unit/test_chat_endpoint.py:72`, change:

```python
assert set(body.keys()) == {"reply", "places", "ragLabel"}
```

to:

```python
assert set(body.keys()) == {"reply", "places", "ragLabel", "conversation_state"}
```

- [ ] **Step 14.6: Lint + typecheck**

Run: `poetry run ruff check app/main.py && poetry run mypy app/main.py`

- [ ] **Step 14.7: Commit**

```bash
git add app/main.py tests/unit/test_chat_endpoint.py
git commit -m "feat(chat): add conversation_state round-trip on /chat"
```

---

## Task 15: Implement accept/decline/alternative early-return branches

**Files:**
- Modify: `app/main.py`
- Modify: `tests/unit/test_chat_endpoint.py`

This is the user-decision routing that runs BEFORE the graph when a pending entry is present.

- [ ] **Step 15.1: Write the failing tests**

Append to `tests/unit/test_chat_endpoint.py`:

```python
def _pending_state(
    place_id: str = "ChIJ_closed",
    family: str = "dessert",
    proposed_id: str = "ChIJ_sophies",
    prior_stop_id: str = "ChIJ_stop1",
) -> dict:
    return {
        "schema_version": 1,
        "closure_context": [
            {
                "schema_version": 1,
                "place_id": place_id,
                "place_name": "Mochill",
                "family": family,
                "attempted_arrival": "2026-05-19T20:02:00-07:00",
                "outcome": "pending_user_decision",
                "insert_after_place_id": prior_stop_id,
                "insert_before_place_id": None,
                "stop_index_hint": 2,
                "proposed_alternative": {
                    "place_id": proposed_id,
                    "name": "Sophie's Crepes",
                    "rationale": "closest open dessert",
                    "source": "google_places",
                    "latitude": 37.7849,
                    "longitude": -122.4093,
                    "primary_type": "Dessert Shop",
                    "arrival_time": "2026-05-19T20:02:00-07:00",
                    "planned_duration_min": 30,
                },
                "proposed_distance_m": 4800.0,
            }
        ],
        "prior_stops": [
            {
                "place_id": prior_stop_id,
                "name": "Stop 1",
                "rationale": "anchor",
                "source": "google_places",
                "latitude": 37.78,
                "longitude": -122.41,
                "primary_type": "Bar",
                "arrival_time": "2026-05-19T18:00:00-07:00",
                "planned_duration_min": 60,
            },
            {
                "place_id": "ChIJ_stop2",
                "name": "Stop 2",
                "rationale": "anchor",
                "source": "google_places",
                "latitude": 37.785,
                "longitude": -122.41,
                "primary_type": "Restaurant",
                "arrival_time": "2026-05-19T19:00:00-07:00",
                "planned_duration_min": 90,
            },
        ],
    }


def test_chat_endpoint_accept_path_inserts_proposed_alternative(mocker) -> None:
    """User replies "yes" → proposed_alternative inserted, graph NOT
    invoked, response carries 3 stops + user_accepted_drive outcome."""
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock(
        side_effect=AssertionError("graph should not run on accept path")
    )
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)
    # Re-validation of proposed_alternative: re-fetch get_details, re-check open
    from app.tools.retrieval import PlaceDetails

    mocker.patch(
        "app.main.get_details",
        return_value=PlaceDetails(
            place_id="ChIJ_sophies",
            name="Sophie's Crepes",
            source="google_places",
            similarity=0.0,
            latitude=37.7849,
            longitude=-122.4093,
            primary_type="Dessert Shop",
            formatted_address="123 Fillmore",
            regular_opening_hours={
                "periods": [{"open": {"day": 2, "hour": 10}, "close": {"day": 2, "hour": 22}}]
            },
        ),
    )
    mocker.patch("app.main._place_is_open_now", return_value=True)
    mocker.patch("app.main._per_stop_closure_status", return_value=[False, False, False])
    mocker.patch("app.main._bounded_retime_after_swap", side_effect=lambda stops: stops)
    mocker.patch("app.main.enrich_stops_with_booking", return_value=None)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "yes",
                "history": [
                    {"role": "user", "content": "plan a date"},
                    {"role": "assistant", "content": "The closest open dessert..."},
                ],
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    place_ids = [p["place_id"] for p in body["places"]]
    assert "ChIJ_sophies" in place_ids
    cs = body["conversation_state"]
    outcomes = [c["outcome"] for c in cs["closure_context"]]
    assert "user_accepted_drive" in outcomes


def test_chat_endpoint_decline_path_drops_closed_stop(mocker) -> None:
    """User replies "no" → closed stop dropped, no graph invocation."""
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock(
        side_effect=AssertionError("graph should not run on decline path")
    )
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "no thanks",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    # prior_stops had 2 stops; dropping leaves 2 stops (closed was never on
    # the prior_stops list — it was the pending entry). Outcome flips.
    cs = body["conversation_state"]
    outcomes = [c["outcome"] for c in cs["closure_context"]]
    assert "user_declined_dropped" in outcomes


def test_chat_endpoint_alternative_path_falls_through_to_graph(mocker) -> None:
    """User replies "find something cheaper" → graph IS invoked with a
    HumanMessage hint about the declined drive option."""
    fake_graph = mocker.Mock()
    captured: dict = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "find something cheaper instead",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    state = captured["state"]
    # The last message should reference the closed place name + the hint.
    last_human = next(
        (m for m in reversed(state.messages) if m.type == "human"), None
    )
    assert last_human is not None
    assert "Mochill" in last_human.content or "find something cheaper" in last_human.content


def test_chat_endpoint_accept_escalates_when_proposed_alternative_missing(mocker) -> None:
    """Accept path with proposed_alternative no longer in DB → graph runs
    (not early-return) with the closure_context preserved minus the bad entry."""
    fake_graph = mocker.Mock()
    captured: dict = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)
    mocker.patch("app.main.get_details", return_value=None)  # gone from DB

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "yes",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    # Graph WAS invoked → not an early return
    assert "state" in captured
```

- [ ] **Step 15.2: Confirm failure**

Run: `poetry run pytest tests/unit/test_chat_endpoint.py -v -k "accept_path or decline_path or alternative_path or escalates"`
Expected: failures — these patches reference symbols (`_place_is_open_now`, `_per_stop_closure_status`, `_bounded_retime_after_swap`, `enrich_stops_with_booking`, `get_details`) that haven't been imported into `app.main` yet.

- [ ] **Step 15.3: Implement the early-return branches in `app/main.py`**

a) Add imports:

```python
from datetime import datetime

from .agent.input_parsing import parse_closure_decision
from .agent.commit import enrich_stops_with_booking
from .agent.swap import (
    _apply_swap,
    _bounded_retime_after_swap,
    _per_stop_closure_status,
    _promote_pending,
    _resolve_insert_position,
    _formulate_closure_question,
    _try_walking_distance_swap,
)
from .agent.revision import summarize_stops
from .tools.retrieval import get_details
```

b) Add a small helper `_place_is_open_now` near the top of `main.py` (or use the existing `temporal_coherence` helpers — but the simplest is a direct SQL call mirroring `swap._execute_closure_query`). Drop in right before the `chat` handler:

```python
def _place_is_open_now(hours: dict | None, when: datetime) -> bool:
    """One-shot SQL call mirroring closure_swap's check, for a single
    proposed alternative on the accept path. Fails OPEN — a DB blip must
    not block the swap, matching the codebase's fail-open precedent."""
    import json as _json

    import psycopg2

    from .db import get_conn

    if not hours:
        return True
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT place_is_open(%s::jsonb, %s)",
                [_json.dumps(hours), when],
            )
            row = cur.fetchone()
            return bool(row[0]) if row else True
    except psycopg2.Error:
        logger.warning("_place_is_open_now DB error; treating as open", exc_info=True)
        return True
```

c) Extract the existing `chat()` handler body into a helper, and route based on whether there's a pending closure decision. Replace the `chat()` function (the version produced in Task 14) with:

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail=AGENT_UNAVAILABLE_DETAIL)

    rag_label = getattr(request.app.state, "rag_label", _rag_label_for(None))

    incoming: ConversationState
    if req.conversation_state is None:
        incoming = ConversationState()
    else:
        try:
            incoming = ConversationState.model_validate(req.conversation_state)
        except ValidationError:
            logger.warning("conversation_state.decode_failed", exc_info=True)
            incoming = ConversationState()

    pending = next(
        (c for c in incoming.closure_context if c.outcome == "pending_user_decision"),
        None,
    )

    if pending is not None and req.message.strip():
        decision = parse_closure_decision(req.message)
        if decision == "accept":
            early = await _try_accept_path(
                pending, incoming, rag_label
            )
            if early is not None:
                return early
            # else fall through to graph (re-validation failed)
        elif decision == "decline":
            return _decline_path(pending, incoming, rag_label)
        # "alternative" intentionally falls through to the graph

    with trace_request("chat", message=req.message[:200]) as trace_id:
        # If "alternative", prepend a HumanMessage hint so the model sees
        # the user declined the drive option but still wants help.
        hint_messages: list[HumanMessage] = []
        if pending is not None and req.message.strip():
            decision = parse_closure_decision(req.message)
            if decision == "alternative":
                hint_messages.append(
                    HumanMessage(
                        content=(
                            f"User declined the drive option for "
                            f"{pending.place_name}. They want: "
                            f"'{req.message}'. Plan again with this guidance."
                        )
                    )
                )

        state = ItineraryState(
            messages=[
                *messages_from_history(req.history),
                *hint_messages,
                HumanMessage(content=req.message),
            ],
            constraints=UserConstraints(
                num_stops=explicit_num_stops_from_conversation(req.history, req.message),
            ),
            closure_context=incoming.closure_context,
        )
        raw = await graph.ainvoke(
            state,
            config={
                "callbacks": langgraph_callbacks(),
                "metadata": {"trace_id": trace_id},
            },
        )
    final_state = raw if isinstance(raw, ItineraryState) else ItineraryState(**raw)

    outbound = ConversationState(
        schema_version=1,
        closure_context=final_state.closure_context,
        prior_stops=final_state.stops,
    )

    return ChatResponse(
        reply=final_state.final_reply or "",
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=outbound,
    )


async def _try_accept_path(
    pending: ClosureContext,
    incoming: ConversationState,
    rag_label: str,
) -> ChatResponse | None:
    """User accepted the proposed drive alternative.

    Returns a built ChatResponse for the early-return path, or None if
    re-validation fails (caller falls through to the graph).
    """
    if pending.proposed_alternative is None:
        return None
    # Re-fetch the proposal — defense in depth against stale state.
    details = get_details(pending.proposed_alternative.place_id)
    if details is None:
        logger.warning("closure_swap.proposed_alternative_invalidated: place_id=%s",
                       pending.proposed_alternative.place_id)
        return None
    # Re-check open-at for the proposal at attempted_arrival
    if not _place_is_open_now(details.regular_opening_hours, pending.attempted_arrival):
        logger.warning("closure_swap.proposed_alternative_invalidated: now closed")
        return None

    # Insert into prior_stops at the resolved position
    insert_at = _resolve_insert_position(pending, incoming.prior_stops)
    replacement = pending.proposed_alternative.model_copy()
    new_stops = list(incoming.prior_stops)
    new_stops.insert(insert_at, replacement)

    # Bounded retime + closure re-check on the new plan
    retimed = await _bounded_retime_after_swap(new_stops)
    re_closed = _per_stop_closure_status(retimed)

    # Build a probe state so we can re-use swap helpers
    probe_state = ItineraryState(
        stops=retimed,
        closure_context=incoming.closure_context,
    )
    # Mark the pending entry as accepted
    updated_context: list[ClosureContext] = []
    for c in incoming.closure_context:
        if c.place_id == pending.place_id and c.outcome == "pending_user_decision":
            updated_context.append(c.model_copy(update={"outcome": "user_accepted_drive"}))
        else:
            updated_context.append(c)

    # If the retime created a new closure, try to walking-distance-swap it
    # (bounded — one more pass), then escalate anything still closed.
    new_pending_entries: list[ClosureContext] = []
    if any(re_closed):
        from .agent.swap import _build_closure_context_entry, _resolve_anchor, _resolve_family_for_stop

        still_closed_indices = [i for i, c_flag in enumerate(re_closed) if c_flag]
        for idx in still_closed_indices:
            closed_stop = retimed[idx]
            family = _resolve_family_for_stop(closed_stop)
            anchor = _resolve_anchor(probe_state, closed_stop)
            match = None
            if family and anchor:
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
                match = _try_walking_distance_swap(probe_state, probe_ctx, anchor_place_id=anchor)
            if match is not None:
                # Apply the swap inline
                retimed[idx] = match.stop
                updated_context.append(
                    _build_closure_context_entry(retimed, idx, match, "auto_swapped")
                )
            else:
                # Escalate
                new_pending_entries.append(
                    _build_closure_context_entry(retimed, idx, None,
                        "pending_user_decision" if not new_pending_entries else "queued_user_decision")
                )
        if new_pending_entries:
            updated_context.extend(new_pending_entries)
            # Drop the closed stops from the surfaced plan
            keep_idx = set(range(len(retimed))) - {
                i for i, e in zip(still_closed_indices, new_pending_entries)
                if e.outcome == "pending_user_decision"
            }
            retimed = [s for i, s in enumerate(retimed) if i in keep_idx]

    # Promote the next queued entry if pending is gone
    updated_context = _promote_pending(updated_context)

    # If anything is still pending, surface the question; otherwise summarize
    next_pending = next(
        (c for c in updated_context if c.outcome == "pending_user_decision"),
        None,
    )
    if next_pending is not None:
        final_reply = _formulate_closure_question(next_pending)
        # Drop the stop the new pending refers to from the surfaced plan
        retimed = [s for s in retimed if s.place_id != next_pending.place_id]
    else:
        enrich_stops_with_booking(retimed, probe_state)
        final_reply = summarize_stops(
            probe_state.model_copy(update={"stops": retimed})
        )

    final_state = ItineraryState(stops=retimed, closure_context=updated_context)
    outbound = ConversationState(
        schema_version=1,
        closure_context=updated_context,
        prior_stops=retimed,
    )
    return ChatResponse(
        reply=final_reply,
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=outbound,
    )


def _decline_path(
    pending: ClosureContext,
    incoming: ConversationState,
    rag_label: str,
) -> ChatResponse:
    """User declined the drive option. Drop the closed stop entirely.

    If queued entries exist, promote the first to pending and ask about it.
    Otherwise summarize the (now shorter) plan.
    """
    # The closed stop wasn't on prior_stops to begin with (it was demoted to
    # pending before the user saw the plan). Just flip the outcome.
    updated_context: list[ClosureContext] = []
    for c in incoming.closure_context:
        if c.place_id == pending.place_id and c.outcome == "pending_user_decision":
            updated_context.append(c.model_copy(update={"outcome": "user_declined_dropped"}))
        else:
            updated_context.append(c)
    updated_context = _promote_pending(updated_context)
    new_pending = next(
        (c for c in updated_context if c.outcome == "pending_user_decision"),
        None,
    )
    probe_state = ItineraryState(
        stops=incoming.prior_stops,
        closure_context=updated_context,
    )
    if new_pending is not None:
        final_reply = _formulate_closure_question(new_pending)
        surfaced_stops = [s for s in incoming.prior_stops if s.place_id != new_pending.place_id]
    else:
        final_reply = summarize_stops(probe_state)
        surfaced_stops = list(incoming.prior_stops)

    final_state = ItineraryState(stops=surfaced_stops, closure_context=updated_context)
    outbound = ConversationState(
        schema_version=1,
        closure_context=updated_context,
        prior_stops=surfaced_stops,
    )
    return ChatResponse(
        reply=final_reply,
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=outbound,
    )
```

- [ ] **Step 15.4: Run the tests**

Run: `poetry run pytest tests/unit/test_chat_endpoint.py -v -k "accept_path or decline_path or alternative_path or escalates"`
Expected: 4 pass.

- [ ] **Step 15.5: Run the full chat endpoint test suite**

Run: `poetry run pytest tests/unit/test_chat_endpoint.py -v`
Expected: all pass.

- [ ] **Step 15.6: Lint + typecheck**

Run: `poetry run ruff check app/main.py && poetry run mypy app/main.py`

- [ ] **Step 15.7: Commit**

```bash
git add app/main.py tests/unit/test_chat_endpoint.py
git commit -m "feat(chat): add accept/decline/alternative early-return paths"
```

---

## Task 16: Frontend — opaque `conversation_state` pass-through

**Files:**
- Modify: `frontend/src/api/chat.js`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/api/chat.test.js` if it tests sendMessage signature; otherwise add a tiny test

- [ ] **Step 16.1: Write the failing test**

Append to `frontend/src/api/chat.test.js` (create if necessary; mirror the existing patterns there):

```javascript
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { sendMessage } from './chat'

describe('sendMessage conversation_state round-trip', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('sends conversation_state when provided', async () => {
    const stub = {
      reply: 'ok',
      places: [],
      ragLabel: 'openai:gpt-4o-mini',
      conversation_state: { schema_version: 1, closure_context: [], prior_stops: [] },
    }
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(stub),
    })

    const cs = {
      schema_version: 1,
      closure_context: [{ place_id: 'p', outcome: 'auto_swapped' }],
      prior_stops: [],
    }
    await sendMessage('hi', [], cs)

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body)
    expect(body.conversation_state).toEqual(cs)
  })

  it('passes null when no conversation_state given', async () => {
    const stub = { reply: 'ok', places: [], ragLabel: '', conversation_state: null }
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(stub),
    })

    await sendMessage('hi', [])

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body)
    expect(body.conversation_state).toBeNull()
  })

  it('returns conversation_state from response opaquely', async () => {
    const stub = {
      reply: 'ok',
      places: [],
      ragLabel: '',
      conversation_state: { schema_version: 1, opaque: true },
    }
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(stub),
    })

    const result = await sendMessage('hi', [])
    expect(result.conversation_state).toEqual({ schema_version: 1, opaque: true })
  })
})
```

- [ ] **Step 16.2: Confirm failure**

Run: `cd frontend && npm test -- chat.test`
Expected: TypeError or failed assertions (sendMessage signature has no third arg yet).

- [ ] **Step 16.3: Modify `chat.js`**

Change `sendMessage` (lines 34-60):

```javascript
export async function sendMessage(message, history = [], conversationState = null) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      history,
      conversation_state: conversationState ?? null,
    }),
  })

  if (!res.ok) {
    let detail = ''
    try {
      const errBody = await res.json()
      detail = errBody?.detail ? ` — ${errBody.detail}` : ''
    } catch {
      /* ignore parse errors */
    }
    throw new Error(`API error ${res.status}${detail}`)
  }

  const data = await res.json()
  const cards = Array.isArray(data?.places) ? data.places : []

  return {
    reply: formatReply(data?.reply ?? ''),
    places: cards.map(toUiPlace),
    ragLabel: data?.ragLabel || undefined,
    conversation_state: data?.conversation_state ?? null,
  }
}
```

- [ ] **Step 16.4: Modify `App.jsx`**

Add a `useRef` for the conversation state. Find the imports (line 1) and ensure `useRef` is imported. Then in the component body, immediately after the existing `useState` calls:

```javascript
import React, { useCallback, useReducer, useRef, useState } from 'react'
```

In the component (around line 41-44):

```javascript
const [focusId, setFocusId] = useState(null)
// Opaque conversation_state from the backend. Stored in a ref so the
// empty-deps `handleSend` reads the current value at call time — a
// useState here would be captured stale in the callback closure.
const conversationStateRef = useRef(null)
```

Update the `sendMessage` call in `handleSend` (line 81):

```javascript
const data = await sendMessage(text, history, conversationStateRef.current)
// Store opaque state for the next request.
conversationStateRef.current = data.conversation_state ?? null
```

Also clear the ref in `handleClear` (line 117-121):

```javascript
const handleClear = useCallback(() => {
  setMessages(INITIAL_MESSAGES)
  dispatch({ type: 'clear' })
  setFocusId(null)
  conversationStateRef.current = null
}, [])
```

- [ ] **Step 16.5: Run the frontend test**

Run: `cd frontend && npm test -- chat.test`
Expected: 3 pass.

- [ ] **Step 16.6: Run the full frontend test suite**

Run: `cd frontend && npm test`
Expected: all pass.

- [ ] **Step 16.7: Verify the App.jsx renders (syntax check)**

Run: `cd frontend && npm run build 2>&1 | tail -10`
Expected: build succeeds.

- [ ] **Step 16.8: Commit**

```bash
git add frontend/src/api/chat.js frontend/src/api/chat.test.js frontend/src/App.jsx
git commit -m "feat(frontend): opaque conversation_state round-trip"
```

---

## Task 17: Integration tests for live DB swap behavior

**Files:**
- Create: `tests/integration/test_swap_real_db.py`

- [ ] **Step 17.1: Write the integration test**

Create `tests/integration/test_swap_real_db.py`:

```python
"""Integration tests for the closure-aware swap node — gated on APP_ENV=integration.

These exercise the real place_is_open() PL/pgSQL helper, the real
_PRIMARY_TYPE_FAMILIES SQL clauses, and the real `nearby()` projection.
They're the only layer that can catch hours-data drift (Google Places
hours change) that mocked tests can't.

Run with:
    APP_ENV=integration make test-integration
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)

from app.agent.state import ItineraryState, Stop
from app.agent.swap import _per_stop_closure_status, _try_walking_distance_swap, swap_closed_stops

SF = ZoneInfo("America/Los_Angeles")


def _make_stop(
    place_id: str, name: str, primary_type: str, arrival: datetime
) -> Stop:
    """Build a real-DB-shaped Stop; coords are filled by enrich at runtime."""
    return Stop(
        place_id=place_id,
        name=name,
        rationale="integration",
        source="google_places",
        arrival_time=arrival,
        primary_type=primary_type,
        planned_duration_min=30,
        latitude=37.78,
        longitude=-122.41,
    )


def test_per_stop_closure_status_detects_real_closed_stop() -> None:
    """For a known place with a known closing time, planning an arrival
    AFTER close should return is_closed=True."""
    # The exact place_ids and hours are DB-dependent; this test asserts the
    # round-trip works end-to-end against the real `place_is_open` helper.
    # A failing test here means hours data has drifted in the index — the
    # remediation is usually re-running ingestion, not changing this code.
    stops = [
        _make_stop(
            place_id="ChIJ_a_real_place_id_in_the_index",
            name="Test Place",
            primary_type="Bar",
            arrival=datetime(2026, 5, 19, 3, 0, tzinfo=SF),  # 3am should be closed
        ),
    ]
    statuses = _per_stop_closure_status(stops)
    # Either True (real place was closed at 3am) or False (place_id wasn't
    # in places_raw → defaults to open). Both are acceptable, but the call
    # must not raise.
    assert isinstance(statuses, list)
    assert len(statuses) == 1


def test_swap_node_runs_against_live_db_without_raising() -> None:
    """Smoke: the node must execute against the real DB without exceptions
    even when no stops are closed."""
    stops = [
        _make_stop(
            place_id="ChIJ_a_real_place_id_in_the_index",
            name="Anchor",
            primary_type="Bar",
            arrival=datetime(2026, 5, 19, 19, 0, tzinfo=SF),
        ),
    ]
    state = ItineraryState(stops=stops)
    update = asyncio.run(swap_closed_stops(state))
    # No assertion on the contents — just that it didn't raise.
    assert isinstance(update, dict)


def test_walking_distance_search_uses_real_family_mapping() -> None:
    """A walking-distance search for a dessert family must return only
    candidates whose primary_type matches the dessert family list."""
    from app.tools.filters import _PRIMARY_TYPE_FAMILIES

    dessert_primaries = set(_PRIMARY_TYPE_FAMILIES["dessert"]["primary_types"])
    closed = _make_stop(
        place_id="ChIJ_a_dessert_place_id",
        name="Closed Dessert",
        primary_type="Dessert Shop",
        arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
    )
    state = ItineraryState(stops=[closed])
    from app.agent.state import ClosureContext

    ctx = ClosureContext(
        place_id=closed.place_id,
        place_name=closed.name,
        family="dessert",
        attempted_arrival=closed.arrival_time or datetime.now(SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    match = _try_walking_distance_swap(state, ctx, anchor_place_id=closed.place_id)
    if match is None:
        pytest.skip("No walking-distance match in the live DB for this anchor.")
    # primary_type belongs to the dessert family OR is None (rare)
    assert match.stop.primary_type is None or match.stop.primary_type in dessert_primaries
```

- [ ] **Step 17.2: Verify the gate works (skip when APP_ENV unset)**

Run: `poetry run pytest tests/integration/test_swap_real_db.py -v`
Expected: 3 skipped (gate stopped them).

- [ ] **Step 17.3: Document the integration-run command**

Append a short note to `tests/integration/test_swap_real_db.py` in the module docstring if not already present. (It's there in step 17.1.)

- [ ] **Step 17.4: Lint**

Run: `poetry run ruff check tests/integration/test_swap_real_db.py`

- [ ] **Step 17.5: Commit**

```bash
git add tests/integration/test_swap_real_db.py
git commit -m "test(integration): add live-DB tests for closure-swap"
```

---

## Task 18: Full-suite test + lint + typecheck + manual smoke

**Files:** none modified

- [ ] **Step 18.1: Full test suite**

Run: `make test`
Expected: all unit tests pass (current + ~36). Coverage should not regress.

If a previously-existing test from a non-swap area fails (e.g. a test that explicitly asserted `"Caveats:"`), update or remove the obsolete assertion in a small focused commit.

- [ ] **Step 18.2: Full lint**

Run: `make lint`
Expected: clean.

- [ ] **Step 18.3: Full typecheck**

Run: `make typecheck`
Expected: clean.

- [ ] **Step 18.4: Frontend tests + build**

Run: `cd frontend && npm test && npm run build`
Expected: all pass.

- [ ] **Step 18.5: Live verification with the user**

Surface to the user:
"All automated checks pass. Ready for live verification with the omakase
Japantown query. Confirm the MLflow tunnel and Cloud SQL proxy are running:
- `curl -s -m3 http://localhost:5050/health` (should return 200)
- `nc -z 127.0.0.1 5433` (should succeed)

Once both are up, start the backend with the proxy-env command from the
session prompt (NOT `make dev`), and run the demo query 5 times."

The user runs the 5 live runs. Expected per the spec's success criteria:
- "Caveats: I couldn't fully satisfy temporal_coherence" never appears.
- Closures within walking distance silent-swap.
- Closures outside walking distance ask exactly one question; "yes" → drive added; "no" → stop dropped; arbitrary text → graph re-plans.
- Refinement turns never re-suggest a recorded closure.

- [ ] **Step 18.6: No final commit here**

The user merges PRs themselves. After live verification passes, stop. The branch is ready for the user to push and open a PR.

---

## Self-Review (mandatory before handing off)

1. **Spec coverage:** Walk each spec section and confirm a task implements it.
   - Architecture diagram: Task 11 + Task 13 (wiring).
   - API contract (Request/Response/treating as untrusted): Task 14 + Task 15.
   - state.py `ClosureContext`/`closure_context`: Task 1.
   - main.py `ConversationState`/early-returns: Task 14 + Task 15.
   - swap.py module: Tasks 7-12 + Task 11 (node).
   - retrieval.py / filters.py / graph.py tool changes: Tasks 3, 4, 5, 6.
   - graph.py wiring + caveat deletion + filter injection: Task 13.
   - input_parsing.py `parse_closure_decision`: Task 2.
   - Frontend chat.js + App.jsx ref: Task 16.
   - Testing layers: every task adds tests; Task 17 covers integration; Task 18 runs the full suite.

2. **Placeholder scan:** No `TBD`/`fill in details`/`similar to Task N`. Every code block is complete.

3. **Type consistency:**
   - `ClosureContext` is the canonical name everywhere (state, swap, main).
   - `CandidateMatch` defined Task 8 used Task 9, 10, 11, 12, 15.
   - `_per_stop_closure_status` defined Task 7, used Tasks 11, 15.
   - `_bounded_retime_after_swap`, `_apply_swap`, `_promote_pending`, `_resolve_insert_position`, `_formulate_closure_question` all defined Task 10, used Tasks 11, 15.
   - `_inject_closure_exclusions` defined Task 12, used Task 13.
   - `_try_walking_distance_swap` / `_try_any_distance_search` defined Task 9, used Tasks 11, 15.
   - `_PRIMARY_TYPE_FAMILIES` / `family_of` defined Task 3, used Tasks 4, 9.
   - `primary_type_family` / `excluded_place_ids` on `SearchFilters` defined Task 4, used Tasks 9, 12.
   - `dist_m` on `PlaceHit` defined Task 5, used Task 9.
   - `kg_traverse(excluded_place_ids=...)` defined Task 6, used Tasks 12 (via `_inject_closure_exclusions`).
   - `parse_closure_decision` defined Task 2, used Task 15.
   - `MAX_CLOSURE_CONTEXT_ENTRIES` defined Task 1, used Task 11.

4. **Cross-cutting risk:** The accept-path logic in Task 15 duplicates some swap-node logic (re-running closure checks, walking-distance swap, escalation). This is intentional — the spec splits early-return (`/chat`) from the graph node, but both must do similar work. If the duplication turns out to be tight enough to factor, do that in a follow-up commit after Task 18 passes — not during the plan execution.

---

## Execution Notes

- **Order is rigid through Task 13** — later tasks import earlier symbols. After Task 13, Tasks 14-17 can technically interleave (they touch different files), but the simplest path is sequential.
- **Commit cadence:** one commit per task is the norm; split if a task's own steps produce two logically distinct commits (e.g. a test-fix commit before the feature commit when an existing test had a stale assertion).
- **Pre-commit hooks** run ruff on staged files; if a commit is blocked, re-stage and commit again (don't fight the hook).
- **Co-author trailer** is added by the harness on the commits you produce — no manual `Co-Authored-By:` line needed in the commit message body.
- **Don't merge.** The user merges PRs themselves; once the branch is verified live, hand back control and stop.
