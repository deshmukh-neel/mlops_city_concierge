# Closure-Aware Itinerary Swap — Design

**Status:** Draft, awaiting user approval (revised post-review)
**Date:** 2026-05-19
**Branch:** `fix/agent-reliability-review`

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
responsibility of a dedicated node + an explicit conversation-state API
field.

## Non-goals

- Generic preference memory ("I don't like ramen") — closure_context is
  closure-specific.
- Multi-conversation persistence — state is scoped to one /chat
  conversation thread.
- Real server-side session state (Redis / per-user store) — out of scope;
  this design uses a stateless API with an explicit opaque
  `conversation_state` field round-tripped by the frontend.
- More than one unresolved closure decision per turn — v1 resolves one
  user-facing decision at a time even if multiple stops are pending.
  (Auto-swaps still batch.)

## Design summary (what changed in revision)

The original draft hid closure state inside an HTML-comment marker on the
assistant reply and reconstructed it server-side from chat history. Code
review (correctly) flagged four blocking issues:

- The frontend escapes `<`/`>` in replies via `formatReply()` →
  `escapeHtml()`, so the marker would be mangled to `&lt;!-- ... --&gt;`
  before being sent back in history.
- `/chat` doesn't round-trip prior `places`, so an "early-return accept"
  path can't reconstruct the full prior itinerary just from the marker.
- `chain_arrival_times` needs leg durations as input; re-chaining after a
  swap with the *prior* legs gives stale times. A second bounded
  `route_legs` call is needed.
- `nearby()` projects `0.0 AS similarity` and doesn't return `dist_m`,
  and `SearchFilters` has no family/types filter. The spec's "top
  similarity within family" was impossible against today's tooling.

The revised design replaces hidden markers with an explicit
`conversation_state` API field, expands `SearchFilters` and the `nearby`
projection to support real candidate selection, adds a bounded second
retime after swaps, and clarifies one-pending-at-a-time semantics.

## Architecture

A new LangGraph node `swap_closed_stops` sits between `retime` and `END`.
`retime` keeps its existing scope (Google Directions retime +
`chain_arrival_times`) but loses its post-retime temporal check and
caveat-append.

```
plan → act → critique → (retime → swap_closed_stops → END)
                       ↑                              ↓
                       └────── revision loop ─────────┘
```

The swap node:

1. Per-stop `place_is_open(hours, retimed.arrival_time)` check (one SQL
   round-trip via the existing helper).
2. If no closures: no-op return.
3. For each closed stop, run a deterministic walking-distance swap via
   `nearby()` with `open_at=retimed_arrival`, primary-type family match,
   and exclusion of place_ids already in `state.stops` plus any in
   `state.closure_context`. Successful walking-distance swaps are batched
   — all of them applied silently in one pass.
4. After all auto-swaps, run **one** additional `route_legs(...)` call on
   the updated stops, re-chain arrival times, and re-run
   `_per_stop_closure_status` once more. (Bounded: no loops; at most one
   retime + one re-check after the swap pass.) This catches the case
   where a replacement is open at the *old* projected arrival but not at
   the *new* one after re-routing.
5. After the bounded re-check, any remaining closures need a user
   decision. **v1 promotes only the first one** to
   `outcome="pending_user_decision"`. The rest are recorded as
   `outcome="queued_user_decision"` and surface on later turns once the
   current one is resolved.
6. The reply is regenerated via `summarize_stops(state)` for the new
   stops. If a pending decision exists, the reply is the question text
   instead of the summary.

## API contract change

This is the heart of the revision.

### Request

```json
POST /chat
{
  "message": "yes",
  "history": [
    {"role": "user", "content": "plan a 3-stop omakase date night..."},
    {"role": "assistant", "content": "The closest open dessert place is..."}
  ],
  "conversation_state": {
    "schema_version": 1,
    "closure_context": [
      {
        "place_id": "ChIJ...",
        "place_name": "Mochill Mochidonut",
        "family": "dessert",
        "attempted_arrival": "2026-05-19T20:02:33-07:00",
        "outcome": "pending_user_decision",
        "stop_index": 2,
        "proposed_alternative": {
          "place_id": "ChIJ...sophies...",
          "name": "Sophie's Crepes",
          "latitude": 37.7849,
          "longitude": -122.4093,
          "primary_type": "Dessert Shop",
          "rating": 4.5,
          "address": "...",
          "arrival_time": "2026-05-19T20:02:33-07:00",
          "planned_duration_min": 60,
          "rationale": "Walking-distance alternative since Mochill closes at 19:00",
          "source": "google_places"
        },
        "proposed_distance_m": 4800.0
      }
    ],
    "prior_stops": [
      { "place_id": "...", "name": "...", "latitude": ..., "longitude": ..., ... },
      { ... }
    ]
  }
}
```

Both `conversation_state` and its inner fields are **optional**. First-turn
requests omit it entirely.

### Response

```json
{
  "reply": "<plain text — no hidden markers>",
  "places": [ /* cards as today */ ],
  "ragLabel": "openai:gpt-4o-mini",
  "conversation_state": {
    "schema_version": 1,
    "closure_context": [ /* updated list */ ],
    "prior_stops": [ /* the stops just shown, so a follow-up turn can act on them */ ]
  }
}
```

### Treating it as untrusted

The backend must rebuild a usable `ItineraryState.closure_context` from
the incoming `conversation_state`, but it does **not** trust the contents.
Defense-in-depth:

- Decode via Pydantic — schema mismatches degrade to empty context with a
  warning log.
- On an accept path: re-fetch `proposed_alternative` details via
  `get_details(place_id)` to confirm it still exists in the DB. If it
  doesn't, escalate to "this place is no longer in our index — pick
  different?"
- On an accept path: re-run `place_is_open(proposed.hours,
  proposed.attempted_arrival)` before applying the swap. If the proposed
  alternative is *now* closed (e.g. the user took an hour to reply and
  the alt's closing time also passed), escalate rather than accept.
- On any received `prior_stops`: re-enrich via
  `enrich_stops_with_booking` so card field freshness comes from current
  DB state, not whatever was cached client-side.
- Missing / malformed `conversation_state` always falls back to the
  normal-planning path (no early return). Conversation degrades
  gracefully — the agent forgets, but nothing breaks.

### Why opaque to the frontend

The frontend treats `conversation_state` as a black box: stores it
verbatim on each response, sends it verbatim on each request. It never
parses, validates, or renders it. The shape can evolve server-side
without frontend changes (only the schema_version moves). Two consequences:

- The frontend `api/chat.js` adds two lines (read the field, store it;
  send the stored field on the next request). No other frontend code
  touches it.
- The contract is purely server-side; the Pydantic schema is the source
  of truth.

## Components

### 1. `app/agent/state.py`

New `ClosureContext` and `Stop`-snapshot Pydantic models, plus a new field
on `ItineraryState`:

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
        "queued_user_decision",
    ]
    stop_index: int                   # position in the itinerary where the closure occurred
    proposed_alternative: Stop | None
    proposed_distance_m: float | None

class ItineraryState(BaseModel):
    # ... existing fields ...
    closure_context: list[ClosureContext] = Field(default_factory=list)
```

Cap: `MAX_CLOSURE_CONTEXT_ENTRIES = 10`, append-and-drop-oldest.

### 2. `app/main.py`

Extend `ChatRequest` and `ChatResponse`:

```python
class ConversationState(BaseModel):
    schema_version: int = 1
    closure_context: list[ClosureContext] = Field(default_factory=list)
    prior_stops: list[Stop] = Field(default_factory=list)

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    conversation_state: ConversationState | None = None

class ChatResponse(BaseModel):
    reply: str
    places: list[dict]
    ragLabel: str
    conversation_state: ConversationState | None = None
```

Extend `/chat` to:

1. Hydrate state from request:
   ```python
   incoming = req.conversation_state or ConversationState()
   closure_context = _validated_closure_context(incoming.closure_context)
   prior_stops     = _revalidated_prior_stops(incoming.prior_stops)  # re-enrich via DB
   ```
2. Find any `pending_user_decision` entry (exactly one if present per v1
   semantics).
3. If pending and `req.message` is non-empty: parse decision via
   `parse_closure_decision(req.message)`.
4. **Early-return branches** (before invoking graph):
   - `accept` → re-validate the proposed alternative (re-fetch details,
     re-check open-at). If still good, insert into `prior_stops` at
     `stop_index`, mark entry `user_accepted_drive`, run **one bounded
     retime** on the new stops, re-run `summarize_stops`, return.
     If the alternative is no longer valid, escalate to a fresh question.
   - `decline` → drop the closed stop entirely; if queued entries exist,
     promote the first to `pending_user_decision` and ask about it.
     Otherwise re-run `summarize_stops` and return.
   - `alternative` → fall through to the graph with a HumanMessage hint:
     `"User declined the drive option for <closed_place>. They want:
     '<user message>'. Plan again with this guidance."`
5. Otherwise (no pending): build `ItineraryState` with the decoded
   `closure_context` populated and run the graph normally.
6. On response: serialize the updated `closure_context` and the
   currently-shown stops into `ConversationState`, attach to
   `ChatResponse`.

### 3. `app/agent/swap.py` (new module)

Public node entry point:

```python
async def swap_closed_stops(state: ItineraryState) -> dict[str, Any]:
    """LangGraph node. See design Architecture for flow."""
```

Internal helpers (underscore-prefixed; white-box tests import them):

- `_per_stop_closure_status(stops: list[Stop]) -> list[bool]` — one SQL
  round-trip via `place_is_open`; fail-open on DB error (matches
  graph.py:336-338 precedent).
- `_try_walking_distance_swap(state, stop_index, per_leg_budget_m) -> CandidateMatch | None`
  — calls `nearby()` with `open_at`, `primary_types_any` (family
  members), and place_id exclusions; scores returned candidates and
  returns the best, or None.
- `_try_any_distance_search(state, closed_stop) -> CandidateMatch | None`
  — fallback after walking-distance fails; broader `nearby()` (no
  budget) for the pending-question proposal.
- `_score_candidate(candidate, closed_stop, prev_stop, next_stop) -> float`
  — combined score: category match + route impact (distance from prev +
  distance to next, both relative to closed_stop's original position).
  Distance comes from the new projected `dist_m` (see #4).
- `_bounded_retime_after_swap(state) -> ItineraryState` — one extra
  `route_legs` call on the updated stops, re-chains arrival times. NO
  loop, NO recursion — at most one extra call per swap node invocation.
- `_promote_pending(closure_context) -> closure_context` — when the
  current pending entry is resolved, promote the first queued entry to
  pending. Returns the updated list.
- `_formulate_closure_question(pending: ClosureContext) -> str` — builds
  question text; differs when `proposed_alternative is None` vs
  populated.
- `_apply_swap(state, stop_index, replacement, leg_durations) -> ItineraryState`
  — replaces stop, re-chains arrival_times using `leg_durations` from
  the second `route_legs`, re-runs `enrich_stops_with_booking` for the
  new stop, regenerates `final_reply` via `summarize_stops`.

```python
class CandidateMatch(BaseModel):
    stop: Stop
    distance_m: float
    family_match_score: float
    route_impact_score: float
    total_score: float
```

### 4. `app/tools/retrieval.py` and `app/tools/filters.py`

**`retrieval.py` change to `nearby()`:**

- Project `dist_m` into the result so callers can use it. Add a
  `dist_m: float | None = None` field to `PlaceHit` (default None so
  `semantic_search` results — which don't have it — stay backward
  compatible). Update `nearby` SQL to select `dist_m` in the final
  projection alongside the existing fields.

**`filters.py` changes:**

- Refactor existing `_DESSERT_TYPES` into a shared
  `_PRIMARY_TYPE_FAMILIES: dict[str, tuple[str, ...]]` with entries for
  `dessert`, `bar`, `restaurant`, `cafe`. Single source of truth.
- Add `family_of(primary_type: str) -> str | None`.
- Rewire `serves_dessert` to use `_PRIMARY_TYPE_FAMILIES["dessert"]`.
- Add to `SearchFilters`:
  - `primary_types_any: list[str] | None = None` — compiles to
    `primary_type = ANY(%s)` for category constraint. (Distinct from
    existing `types_any` which queries the `types` array column.)
  - `excluded_place_ids: list[str] | None = None` — compiles to
    `place_id != ALL(%s)` for exclusion.

### 5. `app/agent/graph.py`

- **Delete** lines 327-347 of `retime` (post-retime `temporal_coherence`
  check + `_final_with_caveats` append).
- Register the new node:
  ```python
  g.add_node("swap_closed_stops", swap_closed_stops)
  g.add_edge("retime", "swap_closed_stops")
  g.add_edge("swap_closed_stops", END)
  ```
- Extend `_constraints_context` to include closure exclusion guidance
  when `state.closure_context` has any `auto_swapped` /
  `user_declined_dropped` / `queued_user_decision` entries:
  ```
  - Earlier in this conversation, these places were closed at the planned
    arrival time and should NOT be re-suggested: <comma-separated names>.
    Their place_ids are also excluded from your search-result candidates.
  ```

### 6. `app/agent/input_parsing.py`

Extend with:

```python
def parse_closure_decision(text: str) -> Literal["accept", "decline", "alternative"]:
    """Conservative parser. Empty/whitespace → 'alternative' (no auto-accept)."""
```

Accepted patterns:
- `accept`: "yes", "yeah", "yep", "sure", "ok", "okay", "👍", "y"
- `decline`: "no", "nope", "n", "nah"
- `alternative`: anything else (including questions, free text, "yes! 4 stops")

The function returns `accept` even if `req.message` also contains a
num_stops change ("yes! make it 4 stops"); the existing
`explicit_num_stops_from_conversation` handles the count separately.

### 7. Frontend: `frontend/src/api/chat.js` (only frontend change)

Two-line change:

```js
// On the way out — send any state we got back from the last response.
body: JSON.stringify({ message, history, conversation_state: conversationState ?? null }),
// ...
// On the way in — store opaque state for next call.
return {
  reply: formatReply(data?.reply ?? ''),
  places: cards.map(toUiPlace),
  ragLabel: data?.ragLabel || undefined,
  conversation_state: data?.conversation_state ?? null,  // opaque pass-through
}
```

Caller code in `App.jsx` stores the last `conversation_state` in state and
passes it to the next `sendMessage` call. ~3 lines in `App.jsx`. The
frontend never inspects the field's contents.

## Data Flow

### Single-turn (closure detected, walking-distance match found)

```
/chat (no conversation_state)
  ↓
ItineraryState built; graph runs:
  plan → act → critique → retime → swap_closed_stops
    detects 1 closure
    _try_walking_distance_swap succeeds → CandidateMatch
    _apply_swap replaces state.stops[K]
    _bounded_retime_after_swap → one extra route_legs call, re-chain
    _per_stop_closure_status re-run → no closures
    closure_context += ClosureContext(outcome="auto_swapped", stop_index=K, ...)
    final_reply = summarize_stops(state)
  ↓
response: {
  reply: <itinerary text>,
  places: [3 cards],
  conversation_state: { closure_context: [auto_swapped × 1], prior_stops: [3] }
}
```

### Two-turn (closure detected, no walking-distance match, user accepts drive)

**Turn 1:**
```
swap_closed_stops:
  closure detected at stop_index=2
  _try_walking_distance_swap → None
  _try_any_distance_search → CandidateMatch(Sophie's Crepes, 4800m)
  closure_context += ClosureContext(outcome="pending_user_decision",
                                    stop_index=2,
                                    proposed_alternative=Sophie's,
                                    proposed_distance_m=4800.0)
  state.final_reply = "The closest open dessert place is Sophie's Crepes,
                      about 3 mi (drive/transit). Want me to add it, or
                      pick something else?"
response: {
  reply: <question>,
  places: [2 cards],   # closed stop dropped
  conversation_state: { closure_context: [pending × 1], prior_stops: [2] }
}
```

**Turn 2 (user replies "yes"):**
```
/chat:
  incoming conversation_state has 1 pending entry
  _revalidated_prior_stops: re-fetch+re-enrich 2 stops → confirmed
  parse_closure_decision("yes") → accept
  re-fetch get_details(proposed_alternative.place_id) → exists ✓
  re-run place_is_open(proposed.hours, proposed.attempted_arrival) → open ✓
  EARLY RETURN:
    insert Sophie's into prior_stops at index=2
    _bounded_retime on the new 3-stop set (one route_legs call)
    closure_context entry → mark "user_accepted_drive"
    final_reply = summarize_stops(state)
response: {
  reply: <itinerary>,
  places: [3 cards],
  conversation_state: { closure_context: [user_accepted_drive × 1], prior_stops: [3] }
}
```

### Two-turn with degraded state (corrupted / stale conversation_state)

**Turn 2 alt path** if `proposed_alternative.place_id` no longer in DB:
```
/chat:
  parse decision → accept
  get_details(proposed.place_id) → None
  ESCALATE: do NOT early-return; build a fresh state with the
  closure_context preserved minus the bad pending entry, prepend a
  HumanMessage: "The previously proposed alternative is no longer
  available. Plan again, excluding [list of closed places]."
  run graph normally
```

### Refinement turn ("make stop 2 cheaper")

```
/chat:
  incoming conversation_state has [auto_swapped(Mochill) × 1] in
  closure_context and prior_stops with 3 entries
  no pending decision
  normal graph flow
  state.closure_context populated from incoming
  _constraints_context tells the LLM:
    "...these places were closed and should NOT be re-suggested: Mochill..."
  agent's semantic_search tool call uses SearchFilters with
    excluded_place_ids=[Mochill's place_id]
  agent commits a different (cheaper) dessert stop
  swap node re-runs; no new closures detected
  final_reply via summarize_stops
response: {
  reply: <itinerary>,
  places: [3 cards],
  conversation_state: { closure_context: [auto_swapped × 1 (preserved)], prior_stops: [3] }
}
```

## Error Handling

Five paths; graceful degradation over correctness theatre.

1. **No `nearby()` candidates at any distance** → `pending_user_decision`
   with `proposed_alternative=None`; question text shifts to "no
   alternatives at all, want to skip / pick different category?"

2. **`nearby()` raises (psycopg2 error)** → caught in swap.py; log
   warning `closure_swap.db_error`; record `pending_user_decision` with
   no proposal; append a soft note ("I had trouble checking for
   alternatives — double-check hours before heading out").

3. **`conversation_state` decode failure** → Pydantic validation rejects
   it; `/chat` falls back to empty state with warning log
   `conversation_state.decode_failed`. The agent "forgets" prior
   closures, but the request succeeds.

4. **`place_is_open` mid-graph DB failure** → `_per_stop_closure_status`
   fail-open returns `[False] * n` (matches graph.py:336-338 precedent);
   no swap happens, no question, plan ships as-is.

5. **Unparseable user reply to a closure question** → falls into
   `alternative` bucket by design; routed back into the graph with a
   HumanMessage hint.

6. **Re-validation failures on accept path** (proposed alternative
   missing from DB, or its hours flipped during the back-and-forth) →
   do NOT silently swap; do NOT silently fail. Escalate to a fresh
   question or a new graph run with the bad place excluded.

Structured warning logs:
- `closure_swap.db_error`
- `conversation_state.decode_failed`
- `closure_context.cap_exceeded`
- `closure_swap.proposed_alternative_invalidated`
- `closure_swap.retime_failure`

## Testing

~36 tests across three layers per the project's test-layering convention.

| File | New? | Tests | Layer |
|---|---|---|---|
| `tests/unit/test_agent_input_parsing.py` | extend | +5 | unit |
| `tests/unit/test_swap.py` | new | 10 | unit + mock |
| `tests/unit/test_swap_node.py` | new | 7 | smoke (graph runs, LLM scripted, DB mocked) |
| `tests/unit/test_chat_endpoint.py` | extend | +8 | functional (TestClient, graph mocked) |
| `tests/unit/test_filters.py` | extend | +3 | unit (primary_types_any + excluded_place_ids + family_of) |
| `tests/integration/test_swap_real_db.py` | new | 3 | integration (`APP_ENV=integration` gate) |

Notable coverage:
- `parse_closure_decision` covering accept / decline / alternative /
  empty.
- Candidate scoring: route impact + family match deterministic.
- Two-turn accept/decline/alternative paths verified at the /chat layer
  with explicit `conversation_state` round-trip.
- Refinement turn verified to thread closure_context into the graph's
  SearchFilters at the SQL layer (the "belt-and-suspenders" enforcement
  for the prompt guidance).
- Re-validation on accept: proposed_alternative missing from DB →
  escalation; proposed_alternative now closed → escalation.
- Bounded retime: ensure `_bounded_retime_after_swap` calls
  `route_legs` at most once per swap node invocation (mock and count).
- Integration tests detect closure data drift (Google Places hours
  changes) that mocked tests can't catch.

Tests for the HTML-comment marker approach are NOT in this design — the
marker is gone.

## Edge cases (explicitly handled)

- **Multiple closed stops in one plan, all walking-distance fixable** →
  all swapped silently in one swap pass; closure_context has multiple
  `auto_swapped` entries. One bounded retime covers all of them.
- **Multiple closed stops, some need user input** → auto-swaps applied
  in batch; remaining unresolved closures: the first becomes
  `pending_user_decision`, the rest `queued_user_decision`. v1 only
  surfaces one question per turn. Once the user resolves the pending
  one (accept/decline early-return path in `/chat`),
  `_promote_pending(closure_context)` runs **inside that same early
  return** — it scans for the first `queued_user_decision`, flips it to
  `pending_user_decision`, and the response's `final_reply` becomes the
  next question. No separate turn / no graph invocation needed to
  promote.
- **User accepts but the proposed alternative just closed** → re-run
  `place_is_open(proposed.hours, proposed.attempted_arrival)` in the
  accept early-return; if closed, escalate.
- **closure_context cap exceeded** → drop oldest, log
  `closure_context.cap_exceeded`. With cap=10 this is a long-conversation
  edge case.
- **conversation_state missing on a turn that should have it** (frontend
  bug, dev console, curl test) → falls back to no-prior-context normal
  flow.

## Edge cases (explicitly NOT handled)

- **conversation_state forgery** — single-tenant app; the worst outcome
  is the agent excludes places it shouldn't or accepts a bogus
  alternative (which would fail re-validation anyway).
- **Concurrent /chat requests for the same conversation** — no shared
  in-memory state; each request is independent.
- **Stale conversation_state after backend redeploy** — schema_version
  mismatch decodes as empty; the next assistant turn writes a fresh
  one.

## Scope estimate

| | LOC |
|---|---|
| Backend source | ~320-360 |
| Frontend changes (chat.js + App.jsx wiring) | ~15 |
| Tests | ~180 |
| **Total** | **~520-560** |

One PR, on `fix/agent-reliability-review`.

## Out-of-scope follow-ups noted but deferred

- **Real session state (server-side store)** — proper architecture,
  multi-PR investment. Documented as future work.
- **Batched multi-pending UI** (one question covers multiple unresolved
  closures with grouped yes/no) — v1 keeps it one-at-a-time.
- **User explicit overrides** ("include Mochill anyway") — out of scope
  for v1; user can restart the conversation.

---

**Status footer (filled in on merge):**

- [ ] Spec approved by user: _________
- [ ] Implementation plan written: _________
- [ ] PR merged: _________
