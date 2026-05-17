# `app/agent/graph.py` Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 604-line `app/agent/graph.py` into three flat sibling modules (`graph.py`, `commit.py`, `revision.py`) with zero behavior change.

**Architecture:** Pure code-move refactor. Two cohesive clusters (stop-commit/booking enrichment; critique/revision-diagnosis) move to new flat modules mirroring the existing `planning.py`/`io.py` pattern. Cross-module entry points get public names (underscore dropped); test-only internals keep their underscore. Behavior preservation is proven by the existing test suite passing with only import-line edits.

**Tech Stack:** Python 3.10, pytest (asyncio_mode=auto), mypy, poetry editable-install.

**Spec:** `docs/superpowers/specs/2026-05-16-graph-py-decomposition-design.md`

**CRITICAL — this is a MOVE, not a rewrite.** When a step says "move function `X`", copy the function body **verbatim** from the source — do not retype, re-paraphrase, reformat, or "improve" it. The only permitted edits are: (a) renaming `_commit_stops`→`commit_stops` and the 4 revision entry points (and their call sites), (b) import statements. Any logic/comment/whitespace change is a plan violation. Verify with `git diff` that moved code is identical modulo the rename.

---

### Task 1: Extract `app/agent/commit.py`

**Files:**
- Create: `app/agent/commit.py`
- Modify: `app/agent/graph.py` (remove `_grounded_place_ids`, `_commit_stops`, `enrich_stops_with_booking`; update imports + the `act` closure call site)
- Modify: `tests/unit/test_agent_graph.py` (import lines 17-22 + `_commit_stops` call-site renames)
- Modify: `tests/unit/test_booking.py` (import line 8 + `_commit_stops` call-site renames + comment line ~286)

- [ ] **Step 1: Create `app/agent/commit.py`**

Create the file with this exact content (the three functions are moved verbatim from `graph.py` lines 45-148; `_commit_stops` is renamed to `commit_stops` and its internal call to `enrich_stops_with_booking` is unchanged):

```python
"""Commit LLM-proposed stops into validated Stops + enrich with booking/card data.

Split out of graph.py (FUTURE_WATCH: app/agent/ directory size). `commit_stops`
is the cross-module entry point called by the graph's `act` node;
`enrich_stops_with_booking` is also public so a future constraint-edit path can
refresh booking URLs in place without re-committing through the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg2

from app.agent.state import ItineraryState, Stop, price_level_to_rank
from app.tools.booking import propose_booking_from_details
from app.tools.retrieval import get_details_many

logger = logging.getLogger(__name__)


def _grounded_place_ids(scratch: dict[str, Any]) -> set[str]:
    """All place_ids the agent has actually seen via prior tool results."""
    grounded: set[str] = set()
    for entries in scratch.values():
        for entry in entries:
            result = entry.get("result")
            if isinstance(result, list):
                for hit in result:
                    pid = getattr(hit, "place_id", None) or (
                        hit.get("place_id") if isinstance(hit, dict) else None
                    )
                    if pid:
                        grounded.add(pid)
            elif result is not None:
                pid = getattr(result, "place_id", None) or (
                    result.get("place_id") if isinstance(result, dict) else None
                )
                if pid:
                    grounded.add(pid)
    return grounded


def commit_stops(state: ItineraryState, raw_stops: Any) -> tuple[list[Stop], dict[str, Any]]:
    """Validate and coerce LLM-supplied stops into Stop models.

    Returns (committed_stops, tool_result_payload). The payload is what the
    LLM sees back as the tool result; rejected place_ids surface there so the
    model can self-correct in W3.
    """
    if not isinstance(raw_stops, list):
        return [], {"error": "stops must be a list"}
    grounded = _grounded_place_ids(state.scratch)
    committed: list[Stop] = []
    rejected: list[dict[str, Any]] = []
    for raw in raw_stops:
        if not isinstance(raw, dict):
            rejected.append({"reason": "stop must be an object", "value": str(raw)})
            continue
        pid = raw.get("place_id")
        if not pid or pid not in grounded:
            rejected.append({"place_id": pid, "reason": "place_id not seen via prior tool result"})
            continue
        try:
            committed.append(Stop(**raw))
        except Exception as e:  # noqa: BLE001
            rejected.append({"place_id": pid, "reason": f"invalid stop: {e}"})
    enrich_stops_with_booking(committed, state)
    return committed, {
        "committed": [s.place_id for s in committed],
        "rejected": rejected,
    }


def enrich_stops_with_booking(stops: list[Stop], state: ItineraryState) -> None:
    """Stamp booking_url + booking_provider on each committed stop in-place.

    Deterministic — URL construction is a pure transform of (PlaceDetails,
    when, party_size), so the LLM is not involved.

    One batched DB read (get_details_many) covers all stops; previously this
    was an N+1 over commit_itinerary. Per-stop URL building is pure and
    cannot raise ValueError/psycopg2.Error, so the error-handling moved to
    the single batched read: a DB blip skips enrichment for this commit
    (cards ship without booking links), bugs propagate.

    Called by commit_stops on initial commit. Public (no leading underscore)
    so a future constraint-edit path — "make it 4 people instead of 2" or
    "shift dinner to 8pm" — can refresh URLs in place without round-tripping
    through the LLM and re-committing the same place_ids.
    """
    party_size = state.constraints.party_size or 2
    place_ids = [stop.place_id for stop in stops]
    try:
        details_by_id = get_details_many(place_ids)
    except psycopg2.Error:
        # Single point of DB failure for the whole enrichment. Skip enrichment
        # for the entire commit; cards still ship without booking links.
        logger.warning(
            "booking enrichment DB read failed for %d stops",
            len(place_ids),
            exc_info=True,
        )
        return

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

- [ ] **Step 2: Remove the three functions from `graph.py`**

In `app/agent/graph.py`, delete `_grounded_place_ids`, `_commit_stops`, and `enrich_stops_with_booking` (the contiguous block, original lines 45-148 — from `def _grounded_place_ids` through the end of `enrich_stops_with_booking`, up to but NOT including `_RECENT_TOOL_EXCHANGES_KEPT = 2`). Leave `_RECENT_TOOL_EXCHANGES_KEPT` and everything after it.

- [ ] **Step 3: Fix `graph.py` imports and the `act` call site**

In `graph.py`, the import block currently has (lines ~13-37):
```python
from app.agent.state import ItineraryState, RevisionHint, Stop, price_level_to_rank
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools
from app.tools.booking import propose_booking_from_details
from app.tools.retrieval import get_details_many
```
`price_level_to_rank`, `propose_booking_from_details`, `get_details_many` are now used ONLY by the moved code. Change these lines to:
```python
from app.agent.commit import commit_stops
from app.agent.state import ItineraryState, RevisionHint, Stop
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools
```
Delete the now-unused `from app.tools.booking import propose_booking_from_details` and `from app.tools.retrieval import get_details_many` lines, and remove `price_level_to_rank` from the `app.agent.state` import. Also check: `import psycopg2` (line ~20) — `git grep -n "psycopg2" app/agent/graph.py` after the move; if no references remain, delete the `import psycopg2` line. Keep `import asyncio`, `import json`, `import logging` (still used by remaining code — verify each with grep; remove any with zero remaining references).

In the `act` closure, the call site (original line ~515):
```python
                stops, payload = _commit_stops(state, tc["args"].get("stops"))
```
becomes:
```python
                stops, payload = commit_stops(state, tc["args"].get("stops"))
```

- [ ] **Step 4: Update `tests/unit/test_agent_graph.py`**

Current import block (lines 17-22):
```python
from app.agent.graph import (
    _commit_stops,
    _prune_for_llm,
    build_agent_graph,
    enrich_stops_with_booking,
)
```
Replace with:
```python
from app.agent.commit import commit_stops, enrich_stops_with_booking
from app.agent.graph import _prune_for_llm, build_agent_graph
```
Then rename every `_commit_stops(` call to `commit_stops(` in this file (run `grep -n "_commit_stops" tests/unit/test_agent_graph.py` — expect call sites near lines 355, 376, 395, 418, 442-443, plus a docstring/comment mention near 322; rename the comment too for accuracy).

- [ ] **Step 5: Update `tests/unit/test_booking.py`**

Line 8: `from app.agent.graph import _commit_stops` → `from app.agent.commit import commit_stops`.
Rename every `_commit_stops(` → `commit_stops(` (run `grep -n "_commit_stops" tests/unit/test_booking.py` — expect ~327, 352, 369, 384) and update the prose comment at ~line 286 (`# These tests drive _commit_stops end-to-end`) to say `commit_stops`.

- [ ] **Step 6: Run the affected suites + mypy**

Run:
```bash
poetry run pytest tests/unit/test_agent_graph.py tests/unit/test_booking.py -q
poetry run mypy app/
```
Expected: all pass; mypy `Success: no issues found`. If mypy reports an unused-import or undefined-name error in `graph.py`, fix the import per Step 3's grep instruction and re-run.

- [ ] **Step 7: Verify the move was verbatim**

Run:
```bash
git stash --include-untracked --quiet 2>/dev/null; git stash pop --quiet 2>/dev/null
git diff --stat
poetry run pytest tests/unit/ -q 2>&1 | tail -2
```
Expected: full unit suite shows the SAME pass/fail counts as `main` — i.e. all pass EXCEPT the known pre-existing `tests/unit/test_chat_functional.py::test_chat_runs_real_graph_with_tool_call` full-suite pollution flake (proven identical on `main`, unrelated to this change; passes in isolation). If ANY other test fails, the move was not verbatim — `git diff main -- app/agent/` and find the unintended change.

- [ ] **Step 8: Commit**

```bash
git add app/agent/commit.py app/agent/graph.py tests/unit/test_agent_graph.py tests/unit/test_booking.py
git commit -m "refactor(agent): extract commit.py from graph.py

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Extract `app/agent/revision.py`

**Files:**
- Create: `app/agent/revision.py`
- Modify: `app/agent/graph.py` (remove the critique/revision cluster + 2 constants; update imports + the `critique` closure call sites)
- Modify: `tests/unit/test_agent_self_correct.py` (top import block + 3 inline imports)

The cluster to move is `graph.py` original lines 218-465: `_can_retry`, `_bumped_counts`, `_scratch_entries_for_last_round`, `_most_restrictive_filter`, `_diagnose_one`, `_diagnose_last_tool_result`, `_hint_for_violation`, `_final_with_caveats`, `_last_ai_content`, `_short_circuit_max_steps`, `_finalize_as_is`, `_critique_final_with_stops`, `_critique_step`. Plus the 2 constants `LOW_SIMILARITY_THRESHOLD = 0.55` and `MAX_REVISIONS_PER_REASON = 2` (original lines 41-42). Four functions are renamed (underscore dropped): `_short_circuit_max_steps`→`short_circuit_max_steps`, `_critique_final_with_stops`→`critique_final_with_stops`, `_finalize_as_is`→`finalize_as_is`, `_critique_step`→`critique_step`. All other functions keep their underscore names. Internal call sites between moved functions must use the renamed names (e.g. `_critique_step` is only called from `graph.py`'s `critique` closure, but `_critique_final_with_stops` calls `_can_retry`/`_bumped_counts`/`_hint_for_violation`/`_final_with_caveats`/`vibe.*` which all move together and keep their names).

- [ ] **Step 1: Create `app/agent/revision.py`**

Create `app/agent/revision.py`. Header:
```python
"""Critique / revision-diagnosis logic for the agent graph.

Split out of graph.py (FUTURE_WATCH: app/agent/ directory size). Public entry
points (critique_step, critique_final_with_stops, short_circuit_max_steps,
finalize_as_is) are called by the graph's `critique` node. Everything else is
module-internal (underscore-prefixed); white-box tests import those by their
private name, matching the existing test pattern.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.critique import CRITIQUE_ITINERARY, CRITIQUE_STEP, CRITIQUE_VIBE, vibe
from app.agent.critique.checks import itinerary_violations
from app.agent.state import ItineraryState, RevisionHint

LOW_SIMILARITY_THRESHOLD = 0.55
MAX_REVISIONS_PER_REASON = 2
```
Then move the 13-function cluster from `graph.py` (original lines 218-465) **verbatim** below the header, applying ONLY these renames at definition + call sites:
- `def _short_circuit_max_steps` → `def short_circuit_max_steps`
- `def _critique_final_with_stops` → `def critique_final_with_stops`
- `def _finalize_as_is` → `def finalize_as_is`
- `def _critique_step` → `def critique_step`

No other function is renamed. Inside the moved code, the only inter-function calls are: `_critique_final_with_stops` (now `critique_final_with_stops`) calls `itinerary_violations`, `_can_retry`, `_hint_for_violation`, `_bumped_counts`, `_final_with_caveats`, `_last_ai_content`, `vibe.vibe_check`, `vibe.VIBE_THRESHOLD`; `_critique_step` (now `critique_step`) calls `_diagnose_last_tool_result`, `_can_retry`, `_bumped_counts`; `_diagnose_last_tool_result` calls `_scratch_entries_for_last_round` + `_diagnose_one`; `_diagnose_one` calls `_most_restrictive_filter`. None of these callees are renamed, so only the 4 `def` lines change. Confirm no callee rename is needed by `grep -n "_short_circuit_max_steps\|_critique_final_with_stops\|_finalize_as_is\|_critique_step" app/agent/revision.py` after writing — these 4 tokens must appear ONLY at their `def` line within revision.py.

- [ ] **Step 2: Remove the cluster + constants from `graph.py`**

In `graph.py`: delete the 2 constants `LOW_SIMILARITY_THRESHOLD`/`MAX_REVISIONS_PER_REASON` (they now live in `revision.py`), and delete the contiguous function block (original lines 218-465, `def _can_retry` through the end of `def _critique_step`). Everything between `_serialize_tool_result` (stays) and `build_agent_graph` (stays) that is part of the cluster is removed.

- [ ] **Step 3: Fix `graph.py` imports + `critique` call sites**

After Task 1, `graph.py`'s critique-related imports are:
```python
from app.agent.critique import (
    CRITIQUE_ITINERARY,
    CRITIQUE_STEP,
    CRITIQUE_VIBE,
    vibe,
)
from app.agent.critique.checks import itinerary_violations
```
The `critique` closure uses `vibe.is_enabled()` / `vibe.make_judge()` (in `build_agent_graph`) — so `vibe` is still needed. But `CRITIQUE_ITINERARY`, `CRITIQUE_STEP`, `CRITIQUE_VIBE`, `itinerary_violations`, `RevisionHint` are now used ONLY by moved code. Run `grep -n "CRITIQUE_ITINERARY\|CRITIQUE_STEP\|CRITIQUE_VIBE\|itinerary_violations\|RevisionHint\|HumanMessage\|ToolMessage" app/agent/graph.py` and remove every import token with zero remaining references. Expected end state for the critique import:
```python
from app.agent.critique import vibe
```
(Remove `from app.agent.critique.checks import itinerary_violations` entirely; remove `RevisionHint` from the `app.agent.state` import; `HumanMessage`/`ToolMessage` — keep only if still referenced by `_prune_for_llm`/`_serialize_tool_result`/closures per grep.)

Add the revision import:
```python
from app.agent.revision import (
    critique_final_with_stops,
    critique_step,
    finalize_as_is,
    short_circuit_max_steps,
)
```

In the `critique` closure (original lines ~570-587), the 4 call sites:
```python
        if state.step_count >= max_steps:
            return _short_circuit_max_steps(state)
        ...
        if finalizing and state.stops:
            return _critique_final_with_stops(state, last, judge_llm)
        if finalizing:
            return _finalize_as_is(state, last)
        return _critique_step(state)
```
become `short_circuit_max_steps(state)`, `critique_final_with_stops(state, last, judge_llm)`, `finalize_as_is(state, last)`, `critique_step(state)` respectively.

- [ ] **Step 4: Update `tests/unit/test_agent_self_correct.py`**

Top import block (lines 24-28):
```python
from app.agent.graph import (
    LOW_SIMILARITY_THRESHOLD,
    MAX_REVISIONS_PER_REASON,
    build_agent_graph,
)
```
Replace with:
```python
from app.agent.graph import build_agent_graph
from app.agent.revision import LOW_SIMILARITY_THRESHOLD, MAX_REVISIONS_PER_REASON
```
Inline imports inside test bodies — change the module path only (names unchanged, they stay private):
- line ~530: `from app.agent.graph import _diagnose_last_tool_result` → `from app.agent.revision import _diagnose_last_tool_result`
- line ~542: same rename as above
- line ~581: `from app.agent.graph import _final_with_caveats` → `from app.agent.revision import _final_with_caveats`
- line ~593: same `_final_with_caveats` rename
- line ~620: `from app.agent.graph import _hint_for_violation` → `from app.agent.revision import _hint_for_violation`
- line ~633: `from app.agent.graph import _diagnose_last_tool_result` → `from app.agent.revision import _diagnose_last_tool_result`

Run `grep -n "from app.agent.graph import" tests/unit/test_agent_self_correct.py` after — the only remaining one should be `build_agent_graph`.

- [ ] **Step 5: Run affected suites + mypy**

```bash
poetry run pytest tests/unit/test_agent_self_correct.py tests/unit/test_agent_graph.py tests/unit/test_booking.py tests/unit/test_agent_smoke.py -q
poetry run mypy app/
```
Expected: all pass; mypy clean. mypy will catch any circular import (`graph → revision → ...` is acyclic per spec; `revision` imports nothing from `graph`/`commit`).

- [ ] **Step 6: Verify verbatim + full suite parity**

```bash
git diff --stat
poetry run pytest tests/unit/ -q 2>&1 | tail -2
wc -l app/agent/graph.py app/agent/commit.py app/agent/revision.py
```
Expected: full suite same pass/fail as `main` (only the known `test_chat_runs_real_graph_with_tool_call` full-suite flake may fail — proven pre-existing on `main`). Approximate post-split sizes (exact values from `wc -l`, not asserted): `graph.py` ~255, `commit.py` ~140, `revision.py` ~255 — all well under the 400-line trigger. If any unexpected test fails, `git diff main -- app/agent/graph.py` and find the non-verbatim edit.

- [ ] **Step 7: Commit**

```bash
git add app/agent/revision.py app/agent/graph.py tests/unit/test_agent_self_correct.py
git commit -m "refactor(agent): extract revision.py from graph.py

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Mark FUTURE_WATCH item resolved

**Files:**
- Modify: `implementation_plan/james/FUTURE_WATCH.md` (the "`app/agent/` directory size" section, the `**Trigger:**` paragraph at lines 79-84)

- [ ] **Step 1: Replace the Trigger paragraph**

In `implementation_plan/james/FUTURE_WATCH.md`, find the section `### \`app/agent/\` directory size`. Replace its `**Trigger:**` paragraph (the one starting "when `app/agent/graph.py` exceeds ~400 lines OR a flat directory") with:

```markdown
**Resolved (2026-05-16):** `graph.py` (604 lines, post-#84) split on branch
`refactor/split-agent-graph` into `graph.py` (~255: node closures + graph
assembly + `_prune_for_llm`/`_serialize_tool_result`), `app/agent/commit.py`
(~140: `commit_stops` + `enrich_stops_with_booking`), and
`app/agent/revision.py` (~255: critique/revision-diagnosis). Flat-module
pattern kept — no subdir, per the "do NOT pre-split" guidance. The 10+
top-level-file trigger was never met (still 9 after the split), so
`critique/` remains the only subdir; `booking/` and `kg/` subdirs stay
deferred until that count is actually reached.
```

(If the resulting top-level `.py` count differs from 9, state the real number — count with `git ls-tree --name-only HEAD app/agent/ | grep -vE '/$' | grep '\.py$' | wc -l` before writing.)

- [ ] **Step 2: Commit**

```bash
git add implementation_plan/james/FUTURE_WATCH.md
git commit -m "docs(future-watch): mark app/agent size item resolved

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Notes for the executor

- This is a behavior-preserving MOVE. The proof of correctness is: existing tests pass with ONLY import-line + rename edits, and `mypy app/` is clean. If you find yourself editing test assertions or function logic, STOP — you've deviated from the plan.
- Use `poetry run` for pytest/mypy. Never add `sys.path` hacks (app is poetry editable-installed).
- Pre-commit hook owns ruff — if it reformats on commit, re-stage and amend (do not pre-run ruff as a gate).
- The pre-existing `tests/unit/test_chat_functional.py::test_chat_runs_real_graph_with_tool_call` full-suite flake is OUT OF SCOPE and proven identical on `main` — do not try to fix it; it passes in isolation.
- User merges PRs themselves — stop after CI is green; do not `gh pr merge`.
- Single-line commit messages per project convention.
