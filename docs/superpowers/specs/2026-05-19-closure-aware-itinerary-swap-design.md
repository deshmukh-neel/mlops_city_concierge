# Closure-Aware Itinerary Swap — Design

**Status:** Draft, awaiting user approval
**Date:** 2026-05-19
**Branch:** `fix/agent-reliability-review` (will spawn a sub-PR or new branch on implementation)

## Problem

The retime node currently appends `"Caveats: I couldn't fully satisfy
temporal_coherence after revisions."` to the final reply whenever real
Google Directions travel times push a stop's arrival past that stop's
closing hours. On the verified demo query this fires ~50% of runs, and the
caveat is real (the agent genuinely scheduled e.g. Mochill Mochidonut at
20:02 when it closes at 19:00). The current handling — ship a broken plan
with a jargon-laden warning — is the worst of both worlds: users see ugly
text *and* would walk into a locked shop. We need a flow that resolves the
closure where possible and asks the user when not.

## Goal

When a committed itinerary contains a stop that will be closed at the
agent's planned arrival time (re-checked against real travel times):

1. Try to deterministically swap in a walking-distance alternative of the
   same category open at the planned arrival. If a candidate exists, swap
   silently — the user sees only the corrected plan.
2. If no walking-distance match exists, ask the user *"the closest open
   option is X (~Nmi / drive or transit). Want it, or pick something else?"*
3. Remember every closure event for the whole conversation so refinement
   turns ("swap stop 2 for something cheaper") don't re-suggest places we
   already learned are closed.

The temporal_coherence caveat is removed. Closure handling is the new
responsibility of a dedicated node.

## Non-goals

- Generic preference memory ("I don't like ramen") — closure_context is
  closure-specific.
- Multi-conversation persistence — context is scoped to one `/chat`
  conversation thread (`req.history`).
- Real server-side session state — out of scope; this design uses
  history-encoded markers (Option 2 from brainstorming; Option 3 was
  punted as a multi-PR architectural shift).
- Frontend changes — the agent asking a question is an existing supported
  state; no new UI work.

## Architecture

Approach B from brainstorming: a new LangGraph node `swap_closed_stops`
sits between `retime` and `END`. `retime` keeps its existing scope (Google
Directions retime + chain_arrival_times) but loses its post-retime
temporal check and caveat-append. All closure-related decisions move to
the new node.

```
plan → act → critique → (retime → swap_closed_stops → END)
                       ↑                              ↓
                       └────── revision loop ─────────┘
```

The swap node:

1. Per-stop `place_is_open(hours, retimed.arrival_time)` check (one SQL
   round-trip via the existing helper).
2. If no closures: no-op return.
3. For each closed stop, try a deterministic walking-distance swap via the
   existing `nearby()` tool with `open_at=retimed_arrival`, primary-type
   family match, and exclusion of place_ids already in `state.stops` plus
   any in `state.closure_context`.
4. Successful swaps are recorded as `outcome="auto_swapped"` and applied
   silently; the reply is regenerated via `summarize_stops(state)` so the
   visible plan reflects the new stops.
5. Closed stops with no walking-distance match: record as
   `outcome="pending_user_decision"`, generate ONE clarifying question
   (batched if multiple), set `state.awaiting_closure_decision`, return.

## Components

### 1. `app/agent/state.py`

New `ClosureContext` Pydantic model and `closure_context` field on
`ItineraryState`:

```python
class ClosureContext(BaseModel):
    schema_version: int = 1
    place_id: str
    place_name: str
    family: str                       # "dessert", "bar", "restaurant", "cafe"
    attempted_arrival: datetime
    outcome: Literal[
        "auto_swapped",
        "user_accepted_drive",
        "user_declined_dropped",
        "pending_user_decision",
    ]
    proposed_alternative: Stop | None
    proposed_distance_m: float | None

class ItineraryState(BaseModel):
    # ... existing fields ...
    closure_context: list[ClosureContext] = Field(default_factory=list)
```

Cap at `MAX_CLOSURE_CONTEXT_ENTRIES = 10`, append-and-drop-oldest.

### 2. `app/tools/filters.py`

- Refactor existing `_DESSERT_TYPES` into a shared
  `_PRIMARY_TYPE_FAMILIES: dict[str, tuple[str, ...]]` with entries for
  `dessert`, `bar`, `restaurant`, `cafe`.
- Add `family_of(primary_type: str) -> str | None`.
- Rewire `serves_dessert` to use `_PRIMARY_TYPE_FAMILIES["dessert"]` so
  there's one source of truth.
- Add `excluded_place_ids: list[str] | None` field to `SearchFilters` with
  SQL clause `place_id != ALL(%s)` when populated.

### 3. `app/agent/graph.py`

- **Delete** lines 327-347 of `retime` (post-retime `temporal_coherence`
  check + `_final_with_caveats` append).
- Register the new node:
  ```python
  g.add_node("swap_closed_stops", swap_closed_stops)
  g.add_edge("retime", "swap_closed_stops")
  g.add_edge("swap_closed_stops", END)
  ```
- Extend `_constraints_context` to include closure exclusion guidance when
  `state.closure_context` is non-empty:
  ```
  - Earlier in this conversation, these places were closed at the planned
    arrival time and should NOT be re-suggested: <comma-separated names>.
    Their place_ids are also excluded from your search-result candidates.
  ```

### 4. `app/agent/swap.py` (new module)

Public node entry point:

```python
async def swap_closed_stops(state: ItineraryState) -> dict[str, Any]:
    """LangGraph node. See design for the full flow."""
```

Internal helpers (underscore-prefixed; white-box tests import them):

- `_per_stop_closure_status(stops: list[Stop]) -> list[bool]` — single SQL
  round-trip via `place_is_open`; fail-open on DB error (matches
  graph.py:336-338 precedent).
- `_try_walking_distance_swap(state, stop_index, per_leg_budget_m) -> Stop | None`
  — calls `nearby()` with `open_at`, family filter, and place_id
  exclusions; returns top-similarity candidate or None.
- `_try_any_distance_search(state, closed_stop, per_leg_budget_m) -> Stop | None`
  — fallback after walking-distance fails; broader `nearby()` (no budget)
  used to populate `proposed_alternative` for the user question.
- `_formulate_closure_question(closure_entries: list[ClosureContext]) -> str`
  — builds question text; differs when `proposed_alternative is None`
  ("no alternatives at any distance") vs. populated ("Sophie's, 3mi —
  drive or pick different?").
- `_apply_swap(state, stop_index, replacement) -> ItineraryState` —
  replaces stop, re-chains arrival_times from that index onward,
  re-runs `enrich_stops_with_booking` for the new stop, regenerates
  `final_reply` via `summarize_stops`.

### 5. `app/agent/closure_marker.py` (new module)

Marker encoding/decoding for history-resident closure context.

```python
def encode(closure_context: list[ClosureContext]) -> str:
    """Returns '<!-- CC_CLOSURE_CONTEXT:<base64-json> -->'."""

def decode_from_history(history: list[ChatMessage]) -> list[ClosureContext]:
    """Walks history backward; reads the most-recent assistant message that
    has a marker; returns the decoded closure_context, or [] on absence /
    parse failure (warning logged)."""
```

The encoded JSON includes `schema_version`; mismatched versions decode as
empty (graceful degradation).

### 6. `app/agent/input_parsing.py`

Extend with:

```python
def parse_closure_decision(text: str) -> Literal["accept", "decline", "alternative"]:
    """Conservative parser. Empty/whitespace → 'alternative' (no auto-accept)."""
```

Accepted patterns:
- `accept`: "yes", "yeah", "yep", "sure", "ok", "okay", "👍", "y"
- `decline`: "no", "nope", "n", "nah"
- `alternative`: anything else (including empty, including questions, including the user proposing something specific)

### 7. `app/main.py`

Extend `/chat`:

1. Call `closure_marker.decode_from_history(req.history)` →
   `closure_context: list[ClosureContext]`.
2. Find any `pending_user_decision` entry (at most one — see Edge cases).
3. If pending and `req.message` is non-empty: parse decision.
4. **Early-return branches** (before invoking graph):
   - `accept` → build state with `state.stops` + `proposed_alternative`
     inserted at the appropriate index, re-chain arrivals, mark the entry
     `user_accepted_drive`, emit `summarize_stops` + new marker, return.
   - `decline` → build state with `state.stops` minus the closed stop,
     re-chain arrivals, mark the entry `user_declined_dropped`, emit
     summary + new marker, return.
   - `alternative` → fall through to the graph with a HumanMessage hint
     prepended: `"User declined the drive option for <closed_place>.
     They want: '<user message>'. Plan again with this guidance."`
5. Otherwise (no pending): normal flow, build `ItineraryState` with the
   decoded `closure_context` passed in.

The final assistant reply on every itinerary-shipping turn ends with a
fresh marker reflecting the current `closure_context`.

## Data Flow

### Single-turn (closure detected, walking-distance match found)

```
/chat
  ↓
read marker → closure_context=[]
parse num_stops, no pending decision
  ↓
ItineraryState built; graph runs:
  plan → act → critique → retime → swap_closed_stops
  ↓
swap detects 1 closure; _try_walking_distance_swap succeeds
state.stops[K] replaced; arrival_times re-chained
closure_context += ClosureContext(outcome="auto_swapped", ...)
final_reply = summarize_stops(state)
  ↓
response: {reply: "<itinerary>" + marker, places: [3 cards]}
```

### Two-turn (closure detected, no walking-distance match, user accepts drive)

**Turn 1:**
```
swap_closed_stops:
  closure detected
  _try_walking_distance_swap → None
  _try_any_distance_search → Sophie's Crepes (3.0 mi)
  closure_context += ClosureContext(outcome="pending_user_decision", ...)
  state.final_reply = "The closest open dessert place is Sophie's Crepes, about 3 mi (drive/transit). Want me to add it, or pick something else?"
response: reply + marker carrying the pending entry, places: 2 cards
```

`places` on a pending turn contains only the successfully-placed stops
(the closed one is dropped from the card payload). The frontend renders
the question in chat; the existing 2 stops stay on the map. When the user
accepts on turn 2, the response's `places` returns to 3 cards.

**Turn 2 (user replies "yes"):**
```
/chat: read marker → 1 pending entry, parse "yes" → accept
EARLY RETURN: insert Sophie's into stops, re-chain, mark entry "user_accepted_drive"
response: summarize_stops + new marker (1 user_accepted_drive entry), places: 3 cards
```

### Refinement turn ("make stop 2 cheaper")

```
/chat: read marker → closure_context=[ClosureContext(Mochill, "auto_swapped", ...)]
no pending decision; normal flow
  ↓
ItineraryState.closure_context populated
graph runs; _constraints_context tells the model:
  "...these places were closed and should NOT be re-suggested: Mochill Mochidonut..."
agent's semantic_search tool call uses SearchFilters with
  excluded_place_ids=[ChIJYTeHK0SBhYARyEEvKzjrBl8]  (Mochill)
SQL filter enforces exclusion at retrieval time
  ↓
agent commits a different (cheaper) dessert place; swap_closed_stops re-runs;
no new closures detected (it's open); final_reply via summarize_stops
response: reply + marker (same auto_swapped entry preserved), places: 3 cards
```

## Error Handling

Five paths; principle is graceful degradation over correctness theatre.

1. **No `nearby()` candidates at any distance** → `pending_user_decision`
   with `proposed_alternative=None`; question text shifts to "no
   alternatives at all, want to skip / pick different category?"

2. **`nearby()` raises (psycopg2 error)** → caught in swap.py; log
   warning; record `pending_user_decision` with no proposal; append a
   soft note to the reply ("I had trouble checking for alternatives —
   double-check hours before heading out").

3. **Marker decode failure** → `closure_marker.decode_from_history`
   catches `binascii.Error`, `json.JSONDecodeError`,
   `pydantic.ValidationError`; returns `[]` with warning log; conversation
   proceeds without prior closure memory.

4. **`place_is_open` mid-graph DB failure** → `_per_stop_closure_status`
   fail-open returns `[False] * n` (matches the existing fail-open
   precedent at graph.py:336-338); no swap happens, no question, plan
   ships as-is.

5. **Unparseable user reply to a closure question** → falls into
   `alternative` bucket by design; routed back into the graph with a
   HumanMessage hint.

Structured warning logs for each path:
- `closure_swap.db_error`
- `closure_marker.decode_failed`
- `closure_context.cap_exceeded`

## Testing

~34 tests across three layers per the project's test-layering convention.

| File | New? | Tests | Layer |
|---|---|---|---|
| `tests/unit/test_closure_marker.py` | new | 7 | unit |
| `tests/unit/test_agent_input_parsing.py` | extend | +5 | unit |
| `tests/unit/test_swap.py` | new | 8 | unit + mock |
| `tests/unit/test_swap_node.py` | new | 6 | smoke (graph runs, LLM scripted, DB mocked) |
| `tests/unit/test_chat_endpoint.py` | extend | +5 | functional (TestClient, graph mocked) |
| `tests/integration/test_swap_real_db.py` | new | 3 | integration (`APP_ENV=integration` gate) |

Notable coverage:
- Marker round-trip including the 10-entry cap boundary.
- Two-turn accept/decline/alternative paths verified at the /chat layer.
- Refinement turn verified to thread closure_context into the graph's
  SearchFilters at the SQL layer (belt-and-suspenders for the prompt
  guidance).
- Integration tests detect closure data drift (Google Places hours
  changes) that mocked tests can't catch.

## Edge cases (explicitly handled)

- **Multiple closed stops in one plan** → batched: walking-distance
  swaps applied for the ones that can be, single question covers all
  that can't (one entry per pending stop in closure_context, one question
  text covering them all).
- **User accepts but the proposed alternative just closed** (clock
  advanced past *its* closing during the back-and-forth) → re-run
  `place_is_open(proposed_alternative.place_id, now)` in the accept
  early-return; if closed, escalate to a new question rather than
  shipping a now-closed stop.
- **closure_context cap exceeded** → drop oldest, log
  `closure_context.cap_exceeded`. With cap=10 this is a long-conversation
  edge case.
- **User asks "what were the closed places?"** → no special handling;
  the LLM gets closure_context via `_constraints_context` and narrates
  naturally.

## Edge cases (explicitly NOT handled)

- **Marker forgery / replay** — not a security concern in this
  single-tenant app.
- **Marker on user messages** — encoder reads only assistant messages.
- **Stale marker after backend redeploy** — `schema_version` mismatch
  decodes as empty; next assistant turn writes a fresh marker.

## Scope estimate

| | LOC |
|---|---|
| Source | ~290-320 |
| Tests | ~150 |
| **Total** | **~440-470** |

One PR. Branch is `fix/agent-reliability-review`; this design either
extends that branch or spawns a successor branch depending on the
implementation plan.

## Out-of-scope follow-ups noted but deferred

- **Real session state (Option 3)** — proper architecture, multi-PR
  investment. Document as future work.
- **User explicit overrides** ("include Mochill anyway") — out of scope
  for v1; user can restart the conversation.
- **Closure context for stops the user manually clicked through on the
  map** — no such event surface exists today.

---

**Status footer (filled in on merge):**

- [ ] Spec approved by user: _________
- [ ] Implementation plan written: _________
- [ ] PR merged: _________
