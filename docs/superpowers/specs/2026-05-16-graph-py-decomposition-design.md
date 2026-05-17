# `app/agent/graph.py` decomposition — design

**Date:** 2026-05-16
**Branch:** `refactor/split-agent-graph` (off `main` *after* PR #84 merged — `main` is at `aaea296`)
**Source:** `implementation_plan/james/FUTURE_WATCH.md` → "`app/agent/` directory size"

## Problem

`app/agent/graph.py` is **604 lines** (post-#84). The FUTURE_WATCH trigger fires
when `graph.py` exceeds ~400 lines OR `app/agent/` exceeds 10 top-level files.
The line trigger is met; the file-count trigger is not (7 top-level files). So
this is a single-file decomposition, **not** a subdir restructure. FUTURE_WATCH
explicitly warns "do NOT pre-split — empty subdirs hurt readability."

## Goal

Split `graph.py` into three flat sibling modules (mirroring the existing
`planning.py` / `io.py` pattern) with **zero behavior change**, verified by the
existing test suite passing with only import lines updated.

## Module breakdown

### `app/agent/commit.py` (~105 lines)

Turn LLM-proposed stops into validated `Stop`s + enrich with booking/card data.

| Symbol | Visibility | Note |
|---|---|---|
| `commit_stops(state, raw_stops) -> tuple[list[Stop], dict]` | **public** | renamed from `_commit_stops` (now cross-module entry point) |
| `enrich_stops_with_booking(stops, state) -> None` | public | already public, unchanged name |
| `_grounded_place_ids(scratch) -> set[str]` | private | module-internal helper of `commit.py` |

Imports: `app.agent.state`, `app.tools.booking`, `app.tools.retrieval`,
stdlib `logging`/`psycopg2`. Owns its own `logger = logging.getLogger(__name__)`.

### `app/agent/revision.py` (~250 lines)

All critique / revision-diagnosis logic.

| Symbol | Visibility |
|---|---|
| `critique_step(state) -> dict` | **public** (renamed from `_critique_step`) |
| `critique_final_with_stops(state, last, judge_llm) -> dict` | **public** (renamed from `_critique_final_with_stops`) |
| `short_circuit_max_steps(state) -> dict` | **public** (renamed from `_short_circuit_max_steps`) |
| `finalize_as_is(state, last) -> dict` | **public** (renamed from `_finalize_as_is`) |
| `_can_retry`, `_bumped_counts`, `_scratch_entries_for_last_round`, `_most_restrictive_filter`, `_diagnose_one`, `_diagnose_last_tool_result`, `_hint_for_violation`, `_final_with_caveats`, `_last_ai_content` | private (module-internal) |
| `LOW_SIMILARITY_THRESHOLD = 0.55`, `MAX_REVISIONS_PER_REASON = 2` | module constants |

Imports: `app.agent.state`, `app.agent.critique` (`CRITIQUE_ITINERARY`,
`CRITIQUE_STEP`, `CRITIQUE_VIBE`, `vibe`), `app.agent.critique.checks`
(`itinerary_violations`).

**Naming rule:** a symbol gets a public (no-underscore) name *only* if it is
called across module boundaries (by `graph.py`). Symbols that are conceptually
internal but reached by white-box tests (`_diagnose_last_tool_result`,
`_final_with_caveats`, `_hint_for_violation`) keep their underscore — tests
import them by their private name from `revision.py`. We do not inflate the
public API to satisfy a test reaching into internals (that white-box pattern
already exists in `test_agent_self_correct.py` and is accepted).

### `app/agent/graph.py` (~250 lines, remains)

Keeps: module docstring, `_prune_for_llm`, `_serialize_tool_result` (LLM-wire
concerns, tightly coupled to the `plan`/`act` node closures), and
`build_agent_graph` with its closures (`plan`, `act`, `critique`,
`route_after_plan`, `route_after_critique`).

New imports:
```python
from app.agent.commit import commit_stops
from app.agent.revision import (
    critique_final_with_stops,
    critique_step,
    finalize_as_is,
    short_circuit_max_steps,
)
```
Call sites inside the closures change `_commit_stops(...)` → `commit_stops(...)`,
`_critique_step(state)` → `critique_step(state)`, etc.

**`enrich_stops_with_booking` is NOT imported by `graph.py`.** Verified: its
only call site is line 91, *inside* `_commit_stops` itself. It moves to
`commit.py` alongside `commit_stops`, which calls it internally. `graph.py`
never references it directly, so importing it there would be a dead import.
`test_agent_graph.py` imports it directly from `commit.py`.

## Blast radius — test files to update (imports only)

Pure code-move ⇒ **only import statements change, no assertion/logic edits.**

1. **`tests/unit/test_agent_graph.py`** (lines 17-22): split the
   `from app.agent.graph import (...)` block —
   - `_commit_stops` → `from app.agent.commit import commit_stops` (and rename
     all ~7 call sites `_commit_stops(` → `commit_stops(` in this file)
   - `enrich_stops_with_booking` → `from app.agent.commit import enrich_stops_with_booking`
   - `_prune_for_llm`, `build_agent_graph` → stay `from app.agent.graph import ...`

2. **`tests/unit/test_booking.py`** (line 8 + ~6 call sites): `from
   app.agent.graph import _commit_stops` → `from app.agent.commit import
   commit_stops`; rename call sites `_commit_stops(` → `commit_stops(` (lines
   ~327, 352, 369, 384). Comment at line 286 mentioning `_commit_stops` updated
   to `commit_stops` for accuracy.

3. **`tests/unit/test_agent_self_correct.py`**:
   - top import block (lines 24-28): `LOW_SIMILARITY_THRESHOLD`,
     `MAX_REVISIONS_PER_REASON` → `from app.agent.revision import ...`;
     `build_agent_graph` stays from `app.agent.graph`
   - inline imports inside test bodies: `from app.agent.graph import
     _diagnose_last_tool_result` (lines 530, 542, 633) →
     `from app.agent.revision import _diagnose_last_tool_result`;
     `_final_with_caveats` (581, 593) → `from app.agent.revision import
     _final_with_caveats`; `_hint_for_violation` (620) →
     `from app.agent.revision import _hint_for_violation`

No other files reference moved symbols (`test_agent_smoke.py` and
`test_chat_functional.py` only import `build_agent_graph`, which stays).

## Test strategy

No new tests — moving code adds no behavior. Verification = behavior-preserving
proof:

- `poetry run pytest tests/unit/test_agent_graph.py tests/unit/test_booking.py
  tests/unit/test_agent_self_correct.py tests/unit/test_agent_smoke.py
  tests/unit/test_chat_functional.py -q` → all green with only import-line edits
- `poetry run mypy app/` → clean (catches any missed reference / circular import)
- Full `poetry run pytest tests/unit/ -q` → same pass count as `main`
  (the pre-existing `test_chat_runs_real_graph_with_tool_call` full-suite
  pollution flake — proven on `main`, unrelated — is the only acceptable
  non-pass, identical to before)

## Circular-import check

`commit.py` and `revision.py` both import only from `app.agent.state`,
`app.agent.critique`, `app.tools.*` — none import each other or `graph.py`.
`graph.py` imports both. Acyclic: `graph → {commit, revision} → {state,
critique, tools}`. `mypy` + import at test collection time will catch any
violation.

## Commit sequencing

Three atomic commits, each independently green (code move + every referencing
import fixed in the *same* commit so the suite never goes red mid-series):

1. `refactor(agent): extract commit.py from graph.py` — create `commit.py`,
   remove the 3 functions from `graph.py`, update `graph.py` call sites +
   imports, update `test_agent_graph.py` + `test_booking.py`.
2. `refactor(agent): extract revision.py from graph.py` — create `revision.py`,
   remove the ~13 symbols + 2 constants from `graph.py`, update `graph.py` +
   `test_agent_self_correct.py`.
3. `docs(future-watch): mark app/agent size item resolved` — FUTURE_WATCH note.

## FUTURE_WATCH resolution note (commit 3)

Replace the **Trigger** paragraph of the "`app/agent/` directory size" section
with a resolution note: `graph.py` split into `graph.py` (~250) / `commit.py`
(~105) / `revision.py` (~250) on `refactor/split-agent-graph`; flat-module
pattern kept (no subdir, per the "do NOT pre-split" guidance); file-count
trigger (10+) still not met so `critique/` stays the only subdir.

## Out of scope

- `app/agent/critique/` is already its own dir — untouched.
- No `booking/` or `kg/` subdir (FUTURE_WATCH lists these as *future*
  considerations gated on the file-count trigger, which is not met).
- No behavior, signature, or logic changes — imports/names/file-location only.
- `_prune_for_llm` / `_serialize_tool_result` stay in `graph.py` (LLM-wire
  concern bound to node closures; moving them adds churn without cohesion gain).
