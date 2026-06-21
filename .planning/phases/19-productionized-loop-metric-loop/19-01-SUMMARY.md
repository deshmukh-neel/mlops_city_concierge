---
phase: 19-productionized-loop-metric-loop
plan: "01"
subsystem: loop/metric
tags: [scoring, recall, floor, gate, pure-function, tdd]
dependency_graph:
  requires: []
  provides: [compute_recall_at_k, RecallAtKResult, FLOOR, decide_loop_exit]
  affects: [app/loop/falsifier_core.py, tests/unit/test_falsifier_core_recall.py]
tech_stack:
  added: []
  patterns: [frozen-dataclass-result, IN-02-assertion, pure-stdlib-core, tdd-red-green]
key_files:
  created:
    - tests/unit/test_falsifier_core_recall.py
  modified:
    - app/loop/falsifier_core.py
decisions:
  - "FLOOR defaults to 0.0 (strict-positive-delta only on first run, D-05); orchestrator overrides via LOOP_HIT_RATE_FLOOR env / --floor CLI"
  - "decide_loop_exit omits the before_rate!=0.0→EXIT_INFRA branch from loop_falsifier.decide_exit because the populated baseline legitimately scores before_hit@k=0 against the v2-diff target set (D-03)"
  - "compute_recall_at_k and compute_hit_rate share the same IN-02 assertion style (noqa S101) for single source of truth on retrieval window"
metrics:
  duration: "166s"
  completed: "2026-06-21T00:13:54Z"
  tasks: 2
  files: 2
requirements: [METRIC-01, METRIC-02, METRIC-03]
---

# Phase 19 Plan 01: Falsifier Core Recall Metrics Summary

**One-liner:** Pure recall@k scoring + floor-aware gate added to falsifier_core with stdlib-only implementation and 13 TDD unit tests at zero API cost.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for compute_recall_at_k + RecallAtKResult | ce3dcfc | tests/unit/test_falsifier_core_recall.py |
| 1+2 (GREEN) | Add compute_recall_at_k, RecallAtKResult, FLOOR, decide_loop_exit | 9422ea0 | app/loop/falsifier_core.py |

## What Was Built

**`app/loop/falsifier_core.py` additions:**

1. **`RecallAtKResult`** — frozen dataclass with `found_count: int`, `total_count: int`, `recall: float`; mirrors `HitRateResult` house style.

2. **`compute_recall_at_k`** — pure function computing recall@K as distinct-union coverage across paraphrases. Key properties:
   - Empty `newly_ingested_ids` → `RecallAtKResult(0, 0, 0.0)` with no ZeroDivisionError
   - IN-02 assertion (same style as `compute_hit_rate`) — any top-k list exceeding K raises AssertionError
   - Distinct counting via set union (`found |= set(topk) & newly_ingested_ids`)
   - Stdlib-only — no mlflow/psycopg2/app.db/semantic_search imports

3. **`FLOOR: float = 0.0`** — module constant in the constants section (L24-41); runtime-tunable via env `LOOP_HIT_RATE_FLOOR` / CLI `--floor`; first run = strict-positive-delta only (D-05).

4. **`decide_loop_exit`** — pure gate function combining:
   - Priority 1: `guard_violation.ok == False` → `EXIT_INFRA`
   - Priority 2: `embed_added_count == 0` → `EXIT_INFRA` (D-02)
   - Priority 3: `is_strictly_positive_delta(before_rate, after_rate) and after_rate >= floor` → `EXIT_PASS`
   - Default: `EXIT_FAIL`
   - Reuses `is_strictly_positive_delta` (no reimplemented delta math)
   - `loop_falsifier.py` `decide_exit` is left byte-for-byte unchanged (D-07)

**`tests/unit/test_falsifier_core_recall.py` — 13 tests across 3 classes:**

- `TestComputeRecallAtK` (5 cases): full coverage → 1.0, partial → 0.5, empty set → 0.0, >K assertion, distinct union dedup
- `TestDecideLoopExit` (6 cases): guard infra, zero-embed infra, pass, below-floor fail, non-positive delta fail, floor==0 strict-delta reduction
- `TestFLOORConstant` (2 cases): default 0.0, float type

## Verification

- `poetry run pytest tests/unit/test_falsifier_core_recall.py -q` — 13 passed
- `poetry run mypy app/loop/falsifier_core.py` — no issues
- `git diff --quiet scripts/loop_falsifier.py` — exit 0 (unchanged)
- Import contract: no new mlflow/psycopg2/app.db/semantic_search imports added

## TDD Gate Compliance

- RED commit: `ce3dcfc` — `test(19-01): add failing tests for compute_recall_at_k, FLOOR, decide_loop_exit`
- GREEN commit: `9422ea0` — `feat(19-01): add compute_recall_at_k, RecallAtKResult, FLOOR, decide_loop_exit`
- REFACTOR: not needed (implementation was clean on first pass)

## Deviations from Plan

None — plan executed exactly as written.

The two tasks shared the test file and all imports at the top of the file; Task 1's RED tests imported `FLOOR` and `decide_loop_exit` (Task 2's symbols), making the test file unusable until both were implemented. This is expected: the plan uses a single test file for both tasks and the RED phase committed the full test file upfront. Both implementations were committed in one GREEN commit since they share the same source file.

## Threat Flags

None — pure function additions with no network, DB, or I/O surface.

## Self-Check: PASSED

- [x] `app/loop/falsifier_core.py` contains `def compute_recall_at_k(` — confirmed
- [x] `app/loop/falsifier_core.py` contains `class RecallAtKResult` — confirmed
- [x] `app/loop/falsifier_core.py` contains `FLOOR: float = 0.0` — confirmed
- [x] `app/loop/falsifier_core.py` contains `def decide_loop_exit(` — confirmed
- [x] `tests/unit/test_falsifier_core_recall.py` created — confirmed
- [x] ce3dcfc exists in git log — confirmed
- [x] 9422ea0 exists in git log — confirmed
- [x] 13/13 tests pass — confirmed
- [x] mypy clean — confirmed
- [x] loop_falsifier.py unchanged — confirmed
