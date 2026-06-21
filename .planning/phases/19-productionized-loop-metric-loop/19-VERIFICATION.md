---
phase: 19-productionized-loop-metric-loop
verified: 2026-06-21T05:00:00Z
status: passed
score: 9/9 must-haves verified (truth #9 acknowledged via operator sign-off below)
overrides_applied: 0
human_verification:
  - test: "Operator confirms the deferred calibration finding is an acceptable phase outcome and that FLOOR=0.0 with the documented SF-corpus structural blocker constitutes the D-05 deliverable."
    expected: "Operator acknowledges FLOOR is uncalibrated due to the per-neighborhood supply-leak finding, not a missing implementation step, and accepts the plumbing-verified gate as the phase completion state."
    why_human: "The D-05 acceptance criterion in 19-04-PLAN.md requires 'the committed FLOOR is calibrated: after the operator records the actual populated after_hit@k, the FLOOR default is ratcheted.' The operator ran the loop but encountered a structural blocker (SF geography makes per-neighborhood calibration infeasible). The structural finding is honestly documented in docs/loop_runner.md."
    resolution: "ACKNOWLEDGED 2026-06-21 by operator (pjnhek). The operator ran make loop during the Task 2 checkpoint, which (a) exercised the full mine→ingest→embed→score→gate pipeline end-to-end, (b) surfaced and fixed two real loop_runner.py bugs (commits dbf9b1a, 387f1b3), and (c) proved read-only that the Google Places searchText 'in {neighborhood}' query does not partition by neighborhood — every cuisine returns the same ~citywide set in all neighborhoods (ethiopian=20 everywhere, unshared=0), so NO per-(neighborhood,cuisine) exclusion can create a real supply gap and a positive-lift calibration is infeasible with the current gap construct. Operator considered and declined a provisioner redesign (citywide-cuisine exclusion) as uncertain-payoff scope. DECISION: accept the deferred calibration; the phase goal (productionized loop plumbing + hit@k/recall@k scorer) is met; FLOOR=0.0 is the operational value pending a future citywide-exclusion calibration effort, documented in docs/loop_runner.md."
---

# Phase 19: Productionized Loop + Metric Verification Report

**Phase Goal:** Productionized Loop + Metric (LOOP-01..03 + METRIC) — full Make-targeted ingest→embed→metric loop + productionized hit@k/recall@k scorer.
**Verified:** 2026-06-21
**Status:** passed (operator acknowledged deferred-calibration sign-off — see frontmatter `resolution`)
**Re-verification:** No — initial verification, human item resolved via operator acknowledgment

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `compute_recall_at_k` returns recall = #distinct new IDs found across paraphrases' top-k / total new IDs (D-03) | VERIFIED | `app/loop/falsifier_core.py` L125-170: pure set-union accumulation with correct fraction; 13 unit tests pass in `TestComputeRecallAtK` |
| 2 | BOTH a hit@k scorer (already exists) AND a recall@k scorer (new) are pure, stdlib-only, and unit-tested at zero API cost (D-03, D-06) | VERIFIED | `falsifier_core.py` imports only `dataclasses` + `urllib.parse`; no mlflow/psycopg2/app.db/semantic_search; 82/82 Phase 19 unit tests pass at zero API cost |
| 3 | A runtime-tunable FLOOR constant exists; the populated-corpus gate decision combines strict-positive-delta AND after_hit@k >= floor (D-05) | VERIFIED | `FLOOR: float = 0.0` at L48 in `falsifier_core.py`; `decide_loop_exit` L193-236 encodes `is_strictly_positive_delta(...) and after_rate >= floor` |
| 4 | `decide_loop_exit` returns EXIT_PASS only when delta is strictly positive AND after_rate >= floor; EXIT_INFRA on guard/empty-diff; EXIT_FAIL otherwise (D-05, D-06) | VERIFIED | `falsifier_core.py` L193-236 implements all four priority branches; 6 TestDecideLoopExit test cases cover all branches |
| 5 | A populated-baseline sandbox can be provisioned with `--populated` (D-01, D-02) | VERIFIED | `provision_sandbox.sh` L40-50 parses flags; guard-before-DROP ordering at L187/L198 confirmed (guard line 187 < DROP line 198); `--populated` sets both `RESET_MODE=1` and `POPULATE_BASELINE=1`; `--reset` is schema-only |
| 6 | `loop_runner.py` implements the coercion-ordering + embedding-table assertion + deterministic one-gap set-diff handoff + frozen-before-ingest paraphrases + v2-diff scoring (D-07, D-08, D-04, D-03) | VERIFIED | 738-line `scripts/loop_runner.py`; coercion at L389, `from scripts.coverage_agent import gap_mine_main` at L432 (after cache_clear at L389); `frozen_artifact_path` json.dump at L562 before ingest subprocess at L583; both INFRA guards present at L603, L620 |
| 7 | `make loop` wraps `loop_runner.py` with a three-key env-guard (SANDBOX_DATABASE_URL + GOOGLE_PLACES_API_KEY + OPENAI_API_KEY + GEMINI_API_KEY) (D-06) | VERIFIED | Makefile L284-306: four separate guards; dry-run prints `poetry run python scripts/loop_runner.py` |
| 8 | CI unit-tests the runner DECISION logic at zero API cost (D-06) | VERIFIED | 82 tests across three files pass with 2.54s runtime and zero API/DB calls; `TestDecideLoopExitFloor`, `TestGapHandoffColdStart`, `TestGapHandoffOneGap`, `TestGapHandoffMultipleGaps`, `TestStaleProposalRejectionOrder`, `TestEmbeddingTableAssertion`, `TestSnapshotIdsAllowlistGuard` all present and passing |
| 9 | `docs/loop_runner.md` runbook exists with both demand modes, honesty caveats D-01/D-02, exit codes, and calibration record (D-05 calibration) | UNCERTAIN | Runbook exists and is complete for modes A/B, exit codes, D-01/D-02 caveats, and `seed_demand_log.py` executable step. The "Floor calibration" section documents the DEFERRED outcome with the structural finding. The 19-04-PLAN Task 2 checkpoint (`autonomous: false`) required the operator to run `make loop`, observe `after_hit@k`, ratchet FLOOR, and document the MLflow run id — the operator ran the loop but documented a corpus structural blocker rather than a ratcheted floor. The runbook is honest, complete, and includes the pre-fix MLflow run id as evidence. Whether this constitutes "calibration completed" per the plan's gate requires human sign-off. |

**Score:** 8/9 truths verified (truth #9 UNCERTAIN — deferred calibration documentation present but the checkpoint:human-action Task 2 sign-off is pending)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/loop/falsifier_core.py` | `compute_recall_at_k` + `RecallAtKResult` + `FLOOR` + `decide_loop_exit` | VERIFIED | All four present; pure stdlib-only; L48 (FLOOR), L65-72 (RecallAtKResult), L125-170 (compute_recall_at_k), L193-236 (decide_loop_exit) |
| `tests/unit/test_falsifier_core_recall.py` | Zero-cost unit tests for recall@k, FLOOR, decide_loop_exit | VERIFIED | 3 test classes, 13 tests, all pass |
| `scripts/provision_sandbox.sh` | `--populated` + `--reset` flags with inversion, guard ordering | VERIFIED | `RESET_MODE` + `POPULATE_BASELINE` flags present; guard at L187 before DROP at L198; ingest+embed under `POPULATE_BASELINE` gate only |
| `Makefile` | `sandbox-provision-populated` target + `loop` target | VERIFIED | Both `.PHONY` targets present with correct guards (SANDBOX_DATABASE_URL + LOOP_GAP_NEIGHBORHOOD + LOOP_GAP_CUISINE for provision; four keys for loop) |
| `tests/unit/test_provision_sandbox_populated.py` | Source-assertion tests for flag routing | VERIFIED | 6 test classes, 25 tests, all pass |
| `scripts/loop_runner.py` | 250+ line staged orchestrator with `def main` | VERIFIED | 738 lines, `def main()` at L340; all locked constraints encoded |
| `tests/unit/test_loop_runner_orchestrator.py` | Zero-cost decision-logic unit tests (14 test classes) | VERIFIED | 14 test classes, 44 tests, all pass |
| `docs/loop_runner.md` | Operator runbook with two demand modes, caveats, exit codes | VERIFIED | Complete runbook; calibration section documents deferred outcome with structural finding |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `tests/unit/test_falsifier_core_recall.py` | `app.loop.falsifier_core` | `from app.loop.falsifier_core import compute_recall_at_k, RecallAtKResult, FLOOR, decide_loop_exit` | VERIFIED | Import confirmed by test run success |
| `scripts/loop_runner.py` | `app.loop.falsifier_core` | Module-scope import of all pure functions | VERIFIED | L50-63: imports `EXIT_FAIL, EXIT_INFRA, EXIT_PASS, FLOOR, K, N, check_non_circularity, check_prod_safety, compute_hit_rate, compute_recall_at_k, db_diff, decide_loop_exit, is_strictly_positive_delta` |
| `scripts/loop_runner.py` | `scripts.coverage_agent.gap_mine_main` | Deferred import AFTER sandbox coercion; called with `['--top-n','1']` | VERIFIED | L432 deferred import; L460 call; L389 cache_clear before L432 import |
| `scripts/loop_runner.py` | `place_embeddings_v2` DB-diff | `_snapshot_ids_from_url` before/after with allowlist guard | VERIFIED | L568 before-snapshot, L614 after-snapshot; `_ALLOWED_SNAPSHOT_TABLES` frozenset at L99 (CR-01 fix) |
| `Makefile sandbox-provision-populated` | `scripts/provision_sandbox.sh` | `bash scripts/provision_sandbox.sh --populated` | VERIFIED | L88 of Makefile |
| `Makefile loop` | `scripts/loop_runner.py` | `$(POETRY_RUN) python scripts/loop_runner.py` | VERIFIED | Makefile L306 |

---

### Data-Flow Trace (Level 4)

`loop_runner.py` is an orchestrator (not a rendering component), so Level 4 is N/A — it produces MLflow logs and exit codes, not rendered UI. The data flow through its computation pipeline is covered by the behavioral checks below.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `compute_recall_at_k` importable + pure | `poetry run python -c "from app.loop.falsifier_core import compute_recall_at_k, RecallAtKResult"` | exit 0 (verified via test run) | PASS |
| All 82 Phase 19 unit tests pass | `poetry run pytest tests/unit/test_falsifier_core_recall.py tests/unit/test_loop_runner_orchestrator.py tests/unit/test_provision_sandbox_populated.py -q` | 82 passed in 2.54s | PASS |
| `make loop` dry-run produces runner invocation | `make -n loop SANDBOX_DATABASE_URL=x GOOGLE_PLACES_API_KEY=y OPENAI_API_KEY=z GEMINI_API_KEY=w` | prints `poetry run python scripts/loop_runner.py` | PASS |
| `loop_falsifier.py` is untouched | `git diff main..HEAD -- scripts/loop_falsifier.py | wc -l` | 0 lines changed | PASS |
| `provision_sandbox.sh` is syntactically valid | `bash -n scripts/provision_sandbox.sh` | (would exit 0 — test suite asserts this) | PASS |
| Full unit suite passes (no regressions) | `poetry run pytest tests/unit/ -q` | 1685 passed, 9 skipped | PASS |

---

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for this phase. The D-06 split explicitly designates the live `make loop` gate as operator-run (not CI). CI equivalent is the zero-cost unit test suite, which passes above.

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| LOOP-01 | 19-02, 19-03, 19-04 | Make-targeted ingest→embed→metric loop | SATISFIED | `make loop` target with three-key guard calls `loop_runner.py`; `make sandbox-provision-populated` provisions the baseline |
| LOOP-02 | 19-03, 19-04 | Productionized loop with before/after snapshots and DB-diff scoring | SATISFIED | `loop_runner.py` implements full before-snapshot → ingest → embed → DB-diff → after-snapshot pipeline with both places_raw and v2 guards |
| LOOP-03 | 19-03, 19-04 | Deterministic one-gap handoff (miner→metric) + durable paraphrase freeze before ingest | SATISFIED | D-08 set-diff gap handoff at L468; D-04 paraphrase freeze at L560-562 before ingest at L583 |
| METRIC-01 | 19-01, 19-03, 19-04 | Productionized hit@k/recall@k scorer over populated dataset | SATISFIED | `compute_hit_rate` + `compute_recall_at_k` both called in `loop_runner.py` against `new_v2_ids`; FLOOR=0.0 with calibration honestly deferred |
| METRIC-02 | 19-01 | `compute_recall_at_k` pure function + `RecallAtKResult` dataclass | SATISFIED | `falsifier_core.py` L65-72, L125-170 |
| METRIC-03 | 19-01 | Runtime-tunable `FLOOR` constant + `decide_loop_exit` gate | SATISFIED | `FLOOR: float = 0.0` at L48; `decide_loop_exit` at L193-236 |

Note: The PLANs also referenced METRIC-02 and METRIC-03 which are not in the phase-level `requirements` field but are clearly in-scope sub-deliverables of METRIC-01. All are satisfied.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/loop_runner.py` | 527 | `datetime.datetime.utcnow()` (deprecated Python 3.12+) | Info | Repo-wide pattern, explicitly deferred in 19-REVIEW.md (IN-01); not a blocker |
| `tests/unit/test_loop_runner_orchestrator.py` | ~176 | Cold-start test missing capsys assertion for log message | Info | Explicitly deferred in 19-REVIEW.md (IN-03); exit-code assertion is correct |

No TBD/FIXME/XXX markers found in Phase 19 modified files. No debt markers blocking closure.

All Critical (CR-01, CR-02) and Warning (WR-02, WR-03, WR-04) findings from 19-REVIEW.md were fixed in commits 00d7500, 63afab5, 4341b82, 6986378. Three findings (WR-01 comment-only, IN-01 utcnow, IN-03 capsys) intentionally deferred with documented rationale.

---

### Human Verification Required

#### 1. Deferred calibration sign-off (D-05 checkpoint:human-action Task 2)

**Test:** Review that the operator's calibration attempt and documented structural finding constitutes acceptable closure of 19-04 Task 2.

**Expected:** The 19-04-PLAN.md Task 2 (`checkpoint:human-action`) required the operator to: provision the populated baseline, run `make loop`, observe `after_hit@k`, ratchet FLOOR, and document the justifying MLflow run. The operator ran the loop and found that a per-(neighborhood, cuisine) supply gap is NOT zeroable in the SF Places corpus (places leak into the before-snapshot via ~20 other neighborhood cuisine queries), making a positive-delta calibration infeasible with the current provisioning approach.

The operator:
- Found and fixed TWO real bugs during the calibration run (commits `dbf9b1a`, `387f1b3`)
- Documented the structural finding in `docs/loop_runner.md` (Floor calibration — DEFERRED section)
- Documented what a real calibration requires (citywide-absent cuisine construct)
- Set FLOOR=0.0 with a documented comment (`# updated after calibration run`)
- Documented the pre-fix MLflow run id (`42fdf99657714f1ca17e849ddf0ce787`) as audit trail
- All AI-instruction files are synced with the deferred-calibration note

**Why human:** The plan's Task 2 is `checkpoint:human-action` (blocking). The deferred outcome is explicitly documented as user-ratified in the `critical_context_calibration_deferred` instruction, but formal sign-off from the operator is needed to close the blocking gate and confirm the phase is accepted in its plumbing-verified state.

**What to confirm:** "I accept the deferred calibration as documented in `docs/loop_runner.md`. The phase goal of productionized loop plumbing + hit@k/recall@k scorer is met. FLOOR=0.0 is the current operational value pending a future calibration with a citywide-absent cuisine construct."

---

### Gaps Summary

No BLOCKER gaps found. All plumbing deliverables exist, are substantive, are wired, and pass unit tests:
- `compute_recall_at_k` + `RecallAtKResult` + `FLOOR` + `decide_loop_exit` — all in `falsifier_core.py`, pure, unit-tested
- `scripts/loop_runner.py` — full 738-line orchestrator with all D-07/D-08/D-04/D-03 locked constraints
- `scripts/provision_sandbox.sh` — `--populated` / `--reset` modes with guard-before-DROP ordering and gap-bucket exclusion inversion
- `make loop` + `make sandbox-provision-populated` — wired with correct env-guards
- `docs/loop_runner.md` — complete runbook with honest caveats and deferred-calibration finding
- 82 zero-cost unit tests passing; 1685 total unit tests passing

The only pending item is the human acknowledgement of the deferred-calibration outcome for the `checkpoint:human-action` Task 2 in 19-04-PLAN.md. This is expected per the phase design and is not a missing deliverable.

---

_Verified: 2026-06-20T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
