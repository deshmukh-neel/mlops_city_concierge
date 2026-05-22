---
phase: 03-eval-harness-extension
plan: 03
subsystem: critique-scorers
tags: [eval, critique, scorers, tdd, deterministic]
requires:
  - "app/agent/critique/checks.py:CRITIQUE_THRESHOLDS"
  - "app/agent/critique/checks.py:itinerary_violations"
  - "app/agent/state.py:UserConstraints.requested_primary_types (plan 03-02)"
  - "app/tools/filters.py:family_of"
  - "app/tools/filters.py:_PRIMARY_TYPE_FAMILIES"
  - "scripts/eval_agent.py:DETERMINISTIC_CHECKS"
provides:
  - "app/agent/critique/checks.py:category_compliance (EVAL-01 scorer)"
  - "app/agent/critique/checks.py:rationale_stop_alignment (EVAL-02 scorer)"
  - "app/agent/critique/checks.py:_FAMILY_KEYWORDS (module-level keyword sets)"
  - "scripts/eval_agent.py:DETERMINISTIC_CHECKS (extended with both scorers; emits {name}_mean per scorer)"
affects:
  - "scripts/eval_agent.py:aggregate_results (writes category_compliance_mean + rationale_stop_alignment_mean)"
  - "Phase 04 (Category Compliance Fix): consumes baseline category_compliance_mean for merge gate"
  - "Phase 05 (Rationale Alignment Fix): consumes baseline rationale_stop_alignment_mean for merge gate"
  - "Phase 06 (Closure-aware Swap): rationale_stop_alignment regression-tests the Walking-distance placeholder bleed"
tech-stack:
  added: []
  patterns:
    - "Deterministic (state) -> float scorer contract (mirrors stop_count_satisfied)"
    - "_try wrapper inheritance in itinerary_violations -> fail-open on DB error"
    - "Module-level keyword derivation from filters._PRIMARY_TYPE_FAMILIES (no hard-coded keyword lists)"
    - "Pure-function scorer (no DB access) so DB outages cannot break the check"
    - "TDD RED -> GREEN per task (3 RED commits, 3 GREEN commits)"
key-files:
  created: []
  modified:
    - "app/agent/critique/checks.py"
    - "scripts/eval_agent.py"
    - "tests/unit/test_critique_checks.py"
    - "tests/unit/test_eval_agent.py"
decisions:
  - "category_compliance length-mismatch policy: dilute the score via denom = max(len_requested, len_stops) so the agent can't game it by over- or under-committing stops. Abstain alternative explicitly rejected (would let zero-stop commits score 1.0)."
  - "category_compliance primary_type=None is a strict mismatch, not an abstention. Mirrors geographic_coherence's 'score what we can measure' on the strict end so missing data doesn't silently pass."
  - "rationale_stop_alignment keyword derivation runs at import time (one-shot, frozenset) from filters._PRIMARY_TYPE_FAMILIES — no runtime overhead per call, and renaming a primary_type in the families dict automatically updates the keyword set."
  - "Generic stop-words ('restaurant', 'bar', 'cafe', 'shop', 'house') KEPT in the keyword set per plan guidance. They are still informative — if the rationale doesn't even mention the family-name word, the rationale almost certainly doesn't describe the right kind of place."
  - "rationale_stop_alignment positioned LAST in itinerary_violations because rationale quality is the softest of the new checks — name/family substring matching has more false-positives than the structured checks."
  - "category_compliance positioned BETWEEN stop_count_satisfied and temporal_coherence so the state-only checks (no DB) are grouped together, matching the existing ordering convention."
  - "DETERMINISTIC_CHECKS in eval_agent.py reordered to ALPHABETICAL-BY-KEY (per plan guidance) so the two new entries slot in deterministically rather than appended at the bottom."
  - "query_result test helper expanded to mirror DETERMINISTIC_CHECKS shape; comment names the invariant (any scorer added to DETERMINISTIC_CHECKS must also appear here, otherwise aggregate_results KeyErrors)."
metrics:
  duration_minutes: 20
  tasks_completed: 3
  files_modified: 4
  tests_added: 24
  red_commits: 3
  green_commits: 3
  completed: "2026-05-21"
requirements:
  - EVAL-01
  - EVAL-02
---

# Phase 3 Plan 03: Deterministic Scorers Summary

One-liner: Shipped `category_compliance` (EVAL-01) and `rationale_stop_alignment` (EVAL-02) as pure-function `(state) -> float` scorers, registered in both `CRITIQUE_THRESHOLDS` / `itinerary_violations` (request-time critique path) and `scripts/eval_agent.py:DETERMINISTIC_CHECKS` (eval-report path), with full RED-GREEN TDD coverage including the closure-swap `"Walking-distance alternative for X"` placeholder bleed regression.

## What Shipped

### Task 1 — `category_compliance` scorer (EVAL-01)

Pure-function scorer added to `app/agent/critique/checks.py`. Imports `family_of` from `app.tools.filters`. The behavior contract:

- Empty `requested_primary_types` → **1.0** (D-03 abstain — the scorer only fires when the user named category slots).
- Empty `stops` → 1.0 (fail-open).
- Otherwise: iterate `min(len(requested), len(stops))` pairs, count matches via `family_of(stop.primary_type) == family_of(requested[i])` (both non-None and equal). Score = `matches / max(len(requested), len(stops))` — length mismatches dilute the score so an agent can't game it by committing extra or fewer stops.
- `primary_type=None` → strict mismatch (cannot pass without measurable data).

Registered with threshold 1.0 in `CRITIQUE_THRESHOLDS` and as the third `_try()` call in `itinerary_violations()` (right after `stop_count_satisfied`, before `temporal_coherence` — grouping all state-only, DB-free checks together).

**Tests (11 new):** D-03 abstain, fail-open empty stops, single exact match, multi-slot all match, single-stop family mismatch, partial 2-of-2 match, length mismatch in both directions, `primary_type=None` strict mismatch, threshold registered, and a no-DB-access smoke test (asserts `get_conn` is never called).

**Commits:**
- `3eea6c0` — **RED:** failing tests (collection-time `ImportError`)
- `5c9f201` — **GREEN:** scorer + threshold + itinerary_violations registration

### Task 2 — `rationale_stop_alignment` scorer (EVAL-02)

Pure-function scorer added to `app/agent/critique/checks.py`. Builds a module-level `_FAMILY_KEYWORDS: dict[str, frozenset[str]]` at import time from `filters._PRIMARY_TYPE_FAMILIES` — splits multi-word entries on underscores and whitespace, lowercases, dedupes. Generic family-name words (`restaurant`, `bar`, `cafe`) are kept per plan guidance (still informative signal).

The behavior contract:

- Empty `stops` → 1.0 (fail-open).
- Per stop, a "match" requires either (a) `stop.name.lower() in rationale_lower` OR (b) any family keyword from `_FAMILY_KEYWORDS[family_of(stop.primary_type)]` appears in `rationale_lower`. Score = `matches / len(stops)`.
- `primary_type=None` with no name substring → no match (no derivable family keywords; only name substring can save the stop).

Registered with threshold 1.0 in `CRITIQUE_THRESHOLDS` and as the **last** `_try()` call in `itinerary_violations()` (rationale quality is the softest of the new checks — substring matching is more prone to false positives than structured checks).

**Tests (10 new):** fail-open empty, name substring match, family keyword match (restaurant), **closure-swap placeholder bleed regression**, bar-family keyword, multi-stop fractional, `primary_type=None`+no-name-match, case-insensitive name match, threshold registered, no-DB-access smoke test.

The closure-swap regression test uses the EXACT live placeholder string `"Walking-distance alternative for Kaiseki Yuzu"` against a stop named `"Lazy Bear"` with `primary_type="American Restaurant"`. The placeholder names the closed (origin) stop, contains no restaurant-family keyword, and does not contain the candidate stop's name → scorer returns 0.0. A code comment in the test names `app/agent/swap.py:238` as the owner of the live string so renames are easy to track.

**Commits:**
- `a35e365` — **RED:** failing tests (collection-time `ImportError`)
- `383cc2a` — **GREEN:** scorer + `_FAMILY_KEYWORDS` + threshold + itinerary_violations registration

### Task 3 — `DETERMINISTIC_CHECKS` registration in `scripts/eval_agent.py`

Three coordinated edits across `scripts/eval_agent.py` and `tests/unit/test_eval_agent.py`:

1. **Import block** (`scripts/eval_agent.py:25-33`) — added `category_compliance` and `rationale_stop_alignment` to the existing `from app.agent.critique.checks import (...)` group.
2. **`DETERMINISTIC_CHECKS` dict** (`scripts/eval_agent.py:46-54`) — reordered to alphabetical-by-key and slotted both new scorers into the correct positions. `aggregate_results`'s existing `for name in DETERMINISTIC_CHECKS` loop now emits `category_compliance_mean` and `rationale_stop_alignment_mean` per report.
3. **`query_result` test helper** (`tests/unit/test_eval_agent.py:55-72`) — extended the `checks` dict to mirror `DETERMINISTIC_CHECKS` shape so `aggregate_results` doesn't `KeyError` when iterating over the expanded scorer set. Inline comment names the invariant.

**Tests (3 new):** registration assertion, aggregate-emits-both-mean-keys assertion, and the EVAL-08 / P1 `json.dumps(asdict(QueryEvalResult(...)))` round-trip assertion (regression guard against non-JSON-safe values smuggled into a dataclass field — same shape as the PR #94 `AIMessage.tool_calls` Pydantic-args fix).

**Commits:**
- `290b8e2` — **RED:** 3 failing tests (registration assertion, missing aggregate keys, helper missing entries)
- `de52bb7` — **GREEN:** import + DETERMINISTIC_CHECKS + helper extension

## Verification Runs

```text
poetry run pytest tests/unit/test_critique_checks.py -v
  42 passed in 0.10s    (32 prior + 10 new across category + rationale)

poetry run pytest tests/unit/test_eval_agent.py -v
  27 passed in 0.51s    (24 prior + 3 new)

poetry run make test-unit
  645 passed, 9 warnings in 31.12s    (621 prior baseline -> +24 new tests, zero regressions)

poetry run mypy app/
  Success: no issues found in 34 source files

poetry run ruff check app/agent/critique/checks.py scripts/eval_agent.py \
  tests/unit/test_critique_checks.py tests/unit/test_eval_agent.py
  All checks passed!
```

**Plan-level cross-registry sync verification:**

```python
poetry run python -c "
from scripts.eval_agent import DETERMINISTIC_CHECKS
from app.agent.critique.checks import CRITIQUE_THRESHOLDS
assert set(DETERMINISTIC_CHECKS) - {'stop_count_satisfied'} == set(CRITIQUE_THRESHOLDS) - {'stop_count_satisfied'}
print('Sync OK')
"
  Sync OK
```

The asymmetry between `DETERMINISTIC_CHECKS` (no `stop_count_satisfied`) and `CRITIQUE_THRESHOLDS` (has `stop_count_satisfied`) was preserved exactly — Phase 3 did not change that intentional design.

## Acceptance Criteria Status

All acceptance criteria from the plan were verified passing:

- ✅ Task 1: `grep -n "def category_compliance" app/agent/critique/checks.py` → 1 line; `grep -n '"category_compliance"' app/agent/critique/checks.py` → 2 lines (threshold + `_try()`); state-only import smoke test passes; 11 tests collected and passing under `-k category_compliance`; mypy clean; no `get_conn` in the function body.
- ✅ Task 2: `grep -n "def rationale_stop_alignment" app/agent/critique/checks.py` → 1 line; `grep -n '"rationale_stop_alignment"' app/agent/critique/checks.py` → 2 lines; `grep -n "Walking-distance alternative" tests/unit/test_critique_checks.py` → 5 lines (test fixture + 4 doc/test references); 10 tests collected and passing under `-k rationale_stop_alignment`; mypy clean; no `get_conn`.
- ✅ Task 3: `grep -n "category_compliance" scripts/eval_agent.py` → 2 lines; `grep -n "rationale_stop_alignment" scripts/eval_agent.py` → 2 lines; `DETERMINISTIC_CHECKS` membership smoke test passes; full unit suite passes; mypy clean on the whole `app/` package; cross-registry sync verified.

## Deviations from Plan

**None — plan executed exactly as written.**

The plan called out one design choice in advance that I followed without amplification: `_FAMILY_KEYWORDS` derives from `filters._PRIMARY_TYPE_FAMILIES` at import time (one-shot, frozenset values), keeping generic family-name stop-words as informative signal. This is exactly what the `<action>` block specified.

No Rule 1/2/3 auto-fixes were needed:

- The implementation surfaces map 1:1 to the plan's behavior contracts.
- No pre-existing bugs were uncovered in `family_of()`, `_PRIMARY_TYPE_FAMILIES`, or `UserConstraints.requested_primary_types`.
- The legacy `sys.path.insert(0, str(REPO_ROOT))` block in `scripts/eval_agent.py:21-23` was left untouched per the plan's explicit instruction (predates the Poetry editable install; out of scope for this plan).

## Authentication Gates

None — pure-function scorer additions, no external services touched.

## Known Stubs

None. Both scorers are immediately consumable by Phase 4 (`category_compliance_mean` baseline) and Phase 5 (`rationale_stop_alignment_mean` baseline). The eval-report aggregate emits per-scorer mean metrics on every run.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, schema changes, or trust boundaries introduced. Both scorers are pure functions of in-memory `ItineraryState` — no DB calls, no I/O, no logging beyond the existing `_log` in `itinerary_violations`'s `_try` wrapper (which already fail-opens on `Exception`).

## TDD Gate Compliance

All three tasks followed RED → GREEN. No refactor commits were needed (the additions are small and the GREEN-as-final-shape passed lint + mypy + the full suite on first run).

| Task | RED       | GREEN     | Test count delta |
| ---- | --------- | --------- | ---------------- |
| 1    | `3eea6c0` | `5c9f201` | +11 (and 3 mock updates to violations tests for category) |
| 2    | `a35e365` | `383cc2a` | +10 (and 3 mock updates to violations tests for rationale) |
| 3    | `290b8e2` | `de52bb7` | +3 |

Each RED commit fails at test collection time (`ImportError` for the new symbols), which is the strongest form of RED — the absence of the API itself is the failure, not a behavioral assertion that happens to evaluate false.

## Self-Check: PASSED

- FOUND: `app/agent/critique/checks.py` — modified, contains `def category_compliance` (line 194), `def rationale_stop_alignment` (line 256), `_FAMILY_KEYWORDS` build, both `_try()` registrations in `itinerary_violations`
- FOUND: `scripts/eval_agent.py` — modified, both new scorers in import block and `DETERMINISTIC_CHECKS` dict
- FOUND: `tests/unit/test_critique_checks.py` — modified, 21 new tests (10 category + 10 rationale + 1 threshold-extension; plus mock updates in 3 existing violations tests)
- FOUND: `tests/unit/test_eval_agent.py` — modified, 3 new tests + helper extension
- FOUND: commit `3eea6c0` (Task 1 RED)
- FOUND: commit `5c9f201` (Task 1 GREEN)
- FOUND: commit `a35e365` (Task 2 RED)
- FOUND: commit `383cc2a` (Task 2 GREEN)
- FOUND: commit `290b8e2` (Task 3 RED)
- FOUND: commit `de52bb7` (Task 3 GREEN)
