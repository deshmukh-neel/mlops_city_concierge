---
phase: 12-decisiveness-instrumentation-comparison-floor
verified: 2026-06-12T00:00:00Z
status: human_needed
score: 7/7 must-haves verified
overrides_applied: 0
gaps: []
human_verification:
  - test: "Run `make eval-matrix` then `make eval-falsifier` (no RUN_DIR override) and confirm the output names the resolved run directory and the gpt-5-mini pooled rate printed matches that run — not a prior refinement-matrix run"
    expected: "The report header prints the resolved run-dir path (e.g. eval_reports/2026-…) and the per-scenario breakdown reflects only the omakase scenario from eval_matrix.yaml; no refinement scenario contaminates the pool"
    why_human: "WR-06 (advisory): _latest_run_dir picks by ISO8601 name with no validation that the summary came from configs/eval_matrix.yaml. If a refinement run was executed more recently, the falsifier silently grades the wrong artifact. Can only be triggered by running the full live matrix; grep cannot detect the runtime ordering."
  - test: "Run `make eval-matrix` then `make eval-falsifier` with a gpt-4o-mini anchor score below its committed baseline (e.g. by temporarily lowering the baseline JSON to 0.6) and verify exit code is 1 and verdict line reads FAIL (anchor regression)"
    expected: "Exit code 1, verdict line 'FAIL (anchor regression)', gpt-4o-mini pooled < baseline printed"
    why_human: "CR-01 (advisory): anchor non-regression in run-dir mode pools the live run across its scenario universe and the baselines summary across all committed baselines — including refinement_cheaper which is absent from a standard eval_matrix.yaml run. With both anchor medians currently at 1.0 the bug is invisible. Verifying the correct intersection behaviour requires a live run with a non-1.0 anchor result."
---

# Phase 12: Decisiveness Instrumentation + Comparison Floor Verification Report

**Phase Goal:** The eval harness emits per-run decisiveness telemetry and the honest comparison floor (all matrix cells except the deferred cells) is complete, so every experiment arm in Phase 13 can be judged objectively against the same falsifier.
**Verified:** 2026-06-12
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Each completed eval run's JSON contains `first_commit_call_step`, `viable_candidates_per_step`, and `rule8_met_per_step` fields readable without post-processing (SC-1, INST-01/02/03) | VERIFIED | `DeterministicEvalResult` in `scripts/eval_agent.py:120-148` carries all 7 new fields; `report_to_dict` at line 1390-1392 is a plain `asdict(report)` with zero filtering; `query_result_from_state` wires all fields at lines 844-850 |
| 2 | Each run JSON records `step_telemetry` with per-step LLM-call and tool-execution wall times (SC-2, INST-04) | VERIFIED | `app/agent/state.py:278-287` defines `step_telemetry: list[dict[str, Any]]`; `graph.py` timing hooks at lines 316-345 (plan) and 357-472 (act); `scripts/eval_agent.py:849` forwards `list(state.step_telemetry)` verbatim via the unfiltered `asdict` write path |
| 3 | `make eval-falsifier` prints per-model numbers and a PASS/FAIL verdict with exit code 0/1/2 (SC-3, INST-05) | VERIFIED | `scripts/eval_falsifier.py` exists; `make -n eval-falsifier` prints the expected command; live run against real baselines exits 1 with per-model numbers printed (gpt-5-mini pooled 0.500 < 0.6, gpt-4o-mini baseline-mode report); exit 2 confirmed on missing run-dir |
| 4 | `gemini/gemini-3.1-pro-preview` baseline is deferred (D-12-09) and both deferred cells remain in `_DEFERRED_BASELINE_CELLS`; every non-deferred matrix cell is honest n=5 (SC-4, ANCH-02/03) | VERIFIED | `tests/unit/test_eval_matrix.py:121-133` retains both gemini + anthropic in `_DEFERRED_BASELINE_CELLS` with D-12-09 comment; parity test `test_baseline_provider_cells_match_matrix_entries` passes (2 selected, 2 passed); docs updated in `docs/eval_gates.md:92`, `docs/baseline_regen.md:249` |
| 5 | `step_telemetry` survives the LangGraph state reducer between plan() and act() without crashing the next plan step | VERIFIED | `step_telemetry` field is a plain `list[dict[str, Any]]` (replace semantics — no Annotated reducer); plan() returns `[*state.step_telemetry, {...}]` (copy-and-append); act() copies and patches via `list(state.step_telemetry)`; 3 unit tests in `test_agent_graph.py:1388+` assert JSON-safety and accumulation across steps; 213 graph tests pass |
| 6 | `LOW_SIMILARITY_THRESHOLD` is imported never hardcoded in eval_agent.py viability logic | VERIFIED | `scripts/eval_agent.py:38` imports constant; no new `0.55` literal in viability logic paths; `viable_candidates_per_step_from_state` takes `viability_threshold` as parameter; `viability_threshold=threshold` recorded in every run |
| 7 | ROADMAP success criterion 4 and REQUIREMENTS ANCH-02/ANCH-03 match D-12-09 (comparison floor = matrix minus anthropic AND gemini) | VERIFIED | ROADMAP.md line 68 states gemini deferred D-12-09; REQUIREMENTS.md lines 25-26 mark ANCH-02/03 complete with D-12-09 wording; traceability table updated; `grep -n "D-12-09"` matches across all 5 bookkeeping surfaces |

**Score:** 7/7 truths verified

### Requirement ID Traceability

All 7 requirement IDs from PLAN frontmatter cross-referenced against REQUIREMENTS.md:

| Requirement | Plan | Description | Status | Evidence |
|-------------|------|-------------|--------|----------|
| INST-04 | 12-01 | Per-turn latency decomposition (LLM call time vs tool-execution time, per plan step) | SATISFIED | `app/agent/state.py:278`, `app/agent/graph.py` timing hooks, 3 unit tests in `test_agent_graph.py` |
| INST-01 | 12-02 | steps-to-first-commit-consideration per run | SATISFIED | `first_commit_call_step_from_state` at `scripts/eval_agent.py:514`; wired in `query_result_from_state` at line 844 |
| INST-02 | 12-02 | per-step viable-candidate counts | SATISFIED | `viable_candidates_per_step_from_state` at `scripts/eval_agent.py:537`; wired at line 846 |
| INST-03 | 12-02 | rule-8 precondition met/not-met per step | SATISFIED | `rule8_met_per_step_from_state` at `scripts/eval_agent.py:602`; wired at line 847 |
| INST-05 | 12-03 | Executable falsifier report | SATISFIED | `scripts/eval_falsifier.py` + `make eval-falsifier` target; 17 tests pass; smoke test against real baselines exits {0,1} |
| ANCH-02 | 12-04 | gemini n=5 baseline (deferred-with-note per D-12-09) | SATISFIED (deferred-with-note) | `_DEFERRED_BASELINE_CELLS` retains gemini entry with D-12-09 comment; both docs updated; REQUIREMENTS marked complete |
| ANCH-03 | 12-04 | Non-deferred matrix cells honest n=5 | SATISFIED (reinterpreted per D-12-09) | Parity test `test_baseline_provider_cells_match_matrix_entries` passes; floor = matrix minus anthropic AND gemini |

**REQUIREMENTS.md checklist discrepancy (documentation gap, NOT a code gap):** INST-01, INST-02, INST-03 remain marked `[ ] Pending` in the REQUIREMENTS.md checklist and traceability table even though Plan 12-02 fully implemented them. Plan 12-04 only updated ANCH-02/03 status. The code is correct and complete; the checklist was not updated by the executor after Plan 12-02. This is a bookkeeping miss, not a missing implementation.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/agent/state.py` | `step_telemetry` field on `ItineraryState` | VERIFIED | Line 278: `step_telemetry: list[dict[str, Any]] = Field(default_factory=list, ...)`; D-12-01 in description; all 4 keys named |
| `app/agent/graph.py` | In-graph timing hooks in `plan()` and `act()` | VERIFIED | 8 `step_telemetry` references; 4 `time.monotonic()` calls; plan() returns `step_telemetry` key at line 345; act() patches and returns at line 472 |
| `tests/unit/test_agent_graph.py` | Unit tests for step_telemetry production | VERIFIED | 3 new tests at lines 1388-1573; cover key set, types, JSON-safety, accumulation, commit path |
| `scripts/eval_agent.py` | Three derived-field helpers + extended `DeterministicEvalResult` | VERIFIED | All 3 helpers exist; `DeterministicEvalResult` has all 7 INST fields; `query_result_from_state` wires them; `make_error_record` has safe defaults; `report_to_dict` is unfiltered `asdict` |
| `tests/unit/test_eval_agent.py` | Unit tests for three helpers | VERIFIED | 12 new tests in 3 test classes; cover per-step semantics, cumulative rule-8, cosine-only fallback, JSON safety |
| `scripts/eval_falsifier.py` | Artifact-reading falsifier with pooled rate + anchor non-regression | VERIFIED | File exists; `_latest_run_dir`, `_pooled_commit_rate`, `main` present; zero live SDK imports; exit 2 on missing run-dir; real verdict on baselines-mode |
| `tests/unit/test_eval_falsifier.py` | Unit + smoke tests for falsifier | VERIFIED | 17 tests: pooling math, bar logic, baselines-mode path, smoke against real `configs/eval_baselines` |
| `Makefile` | `eval-falsifier` target + `RUN_DIR` var | VERIFIED | `RUN_DIR ?=` at line 111; `.PHONY: eval-falsifier` at line 214; `## INST-05:` help comment; `$(if $(RUN_DIR),--run-dir $(RUN_DIR),)` conditional |
| `tests/unit/test_eval_matrix.py` | `_DEFERRED_BASELINE_CELLS` retains gemini + anthropic with D-12-09 rationale | VERIFIED | Lines 121-133: gemini comment cites D-12-09; both cells present |
| `docs/eval_gates.md` | Gemini deferral section (D-12-09) parallel to Anthropic deferral | VERIFIED | `## Gemini deferral (2026-06-11)` section at line 92; cites D-12-09; logged-not-gated status |
| `docs/baseline_regen.md` | Gemini deferral section (D-12-09) | VERIFIED | `### Gemini deferral (D-12-09)` section at line 249; promotion path preserved |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `app/agent/graph.py plan()` | `ItineraryState.step_telemetry` | `return {'step_telemetry': new_telemetry}` | WIRED | Line 345 returns dict with `step_telemetry` key |
| `app/agent/graph.py act()` | `ItineraryState.step_telemetry` | patch entry with `tool_exec_seconds` + `tool_calls_this_step` | WIRED | Lines 457-472 copy, patch, and return updated telemetry list |
| `scripts/eval_agent.py query_result_from_state` | `ItineraryState.scratch['commit_itinerary'] step indices` | `first_commit_call_step_from_state` | WIRED | Lines 514-534 read scratch, return min step |
| `scripts/eval_agent.py viability helper` | `app/agent/revision.LOW_SIMILARITY_THRESHOLD` | import + per-hit similarity >= threshold check | WIRED | Line 38 imports constant; line 591 uses it |
| `ItineraryState.step_telemetry` | `scripts/eval_agent.py DeterministicEvalResult.step_telemetry` | `query_result_from_state` forwards `list(state.step_telemetry)`; `asdict()` serializes verbatim | WIRED | Line 849; `report_to_dict` at 1390-1392 is plain `asdict(report)` |
| `scripts/eval_falsifier.py` | `scripts.check_eval_gates._build_summary_from_baselines` | lazy import inside `main()` | WIRED | Lines 163-165; no reimplementation (D-12-07) |
| `Makefile eval-falsifier` | `scripts/eval_falsifier.py` | `$(POETRY_RUN) python scripts/eval_falsifier.py` | WIRED | Lines 214-219 |
| `tests/unit/test_eval_matrix.py _DEFERRED_BASELINE_CELLS` | `configs/eval_matrix.yaml + eval_matrix_refinement.yaml` | `test_baseline_provider_cells_match_matrix_entries` parity assertion | WIRED | Lines 149-180; parity test passes |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `scripts/eval_agent.py` `query_result_from_state` | `first_commit_call_step` | `state.scratch["commit_itinerary"]` (live graph scratch written by `act()`) | Yes | FLOWING |
| `scripts/eval_agent.py` `query_result_from_state` | `viable_candidates_per_step` | `state.scratch` semantic_search/nearby entries; gates on `LOW_SIMILARITY_THRESHOLD` (imported constant, not hardcoded) | Yes — with known measurement bias (WR-01: nearby hits hardcode similarity=0.0 so they can never be viable) | FLOWING (advisory bias) |
| `scripts/eval_agent.py` `query_result_from_state` | `step_telemetry` | `state.step_telemetry` from in-graph timing hooks | Yes | FLOWING |
| `scripts/eval_falsifier.py` `_pooled_commit_rate` | `committed_itinerary_rate` | `summary.json` scorers block from `eval_reports/` run dir OR `_build_summary_from_baselines` | Yes — from committed baseline JSONs in `configs/eval_baselines/` | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `eval_falsifier.py --baselines-mode` exits 0 or 1 (real verdict, not 2) | `poetry run python scripts/eval_falsifier.py --baselines-mode --baselines-dir configs/eval_baselines; echo "exit=$?"` | exit=1 (FAIL — gpt-5-mini pooled 0.500 < 0.6); per-model lines printed | PASS |
| `eval_falsifier.py --run-dir /nonexistent` exits 2 | `poetry run python scripts/eval_falsifier.py --run-dir /nonexistent/path; echo "exit=$?"` | exit=2; "summary.json not found" to stderr | PASS |
| `DeterministicEvalResult` has all 7 INST fields | `poetry run python -c "..."` (dataclasses.fields assertion) | All 7 fields present, no missing | PASS |
| All Phase 12 unit tests pass | `poetry run pytest tests/unit/test_eval_matrix.py tests/unit/test_agent_graph.py tests/unit/test_eval_agent.py tests/unit/test_eval_falsifier.py -q` | 281 passed, 2 warnings (RuntimeWarning: coroutine never awaited — test infrastructure noise, not a test failure) | PASS |

### Probe Execution

No probe scripts declared or discovered for this phase. Step 7c: SKIPPED (no probe-*.sh files in conventional paths).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INST-01 | 12-02 | steps-to-first-commit per run | SATISFIED (code complete; checklist not updated) | `first_commit_call_step_from_state` + 4 unit tests + wired in `query_result_from_state` |
| INST-02 | 12-02 | per-step viable-candidate counts | SATISFIED (code complete; checklist not updated) | `viable_candidates_per_step_from_state` + tests + wired |
| INST-03 | 12-02 | rule-8 precondition per step | SATISFIED (code complete; checklist not updated) | `rule8_met_per_step_from_state` + tests + wired |
| INST-04 | 12-01 | per-turn latency decomposition | SATISFIED | `step_telemetry` field + graph timing hooks + 3 unit tests |
| INST-05 | 12-03 | Executable falsifier | SATISFIED | `scripts/eval_falsifier.py` + `make eval-falsifier` + 17 tests |
| ANCH-02 | 12-04 | gemini baseline (deferred-with-note) | SATISFIED (deferred-with-note, D-12-09) | `_DEFERRED_BASELINE_CELLS` + docs + ROADMAP + REQUIREMENTS updated |
| ANCH-03 | 12-04 | non-deferred cells honest n=5 | SATISFIED (reinterpreted per D-12-09) | Parity test passes; 2 deferred cells documented |

**Note:** REQUIREMENTS.md traceability table still shows INST-01/02/03 as "Pending" despite code being complete (Plan 12-02 implemented them; Plan 12-04 only updated ANCH-02/03 in the traceability table). This is a bookkeeping miss by the executor — not a code gap. The code, tests, and PLAN all align correctly.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/eval_falsifier.py` | 139-141 | `--gates-config` arg parsed but `args.gates_config` never referenced in `main()` | WARNING (WR-05 from review) | Operator editing `eval_gates.yaml` expecting the falsifier to follow will be silently ignored; Makefile passes `--gates-config configs/eval_gates.yaml` which is accepted and discarded |
| `scripts/eval_agent.py` | 530-534 | `first_commit_call_step_from_state` collects step values without `isinstance(step, int)` guard; sibling helpers have the guard at lines 571-572, 652-653 | WARNING (WR-08 from review) | `min([None])` or mixed int/str raises `TypeError` crashing the query row; docstring promises None on malformed input |
| `scripts/eval_agent.py` | 644 | `rule8_met_per_step_from_state` hardcodes `viability_threshold = LOW_SIMILARITY_THRESHOLD` internally while `viable_candidates_per_step_from_state` takes it as a parameter | WARNING (WR-09 from review) | `viability_threshold` field in run JSON claims to be "the threshold used for viability judgments" but only binds one of the two metrics; pass a different threshold and they silently diverge |
| `scripts/eval_agent.py` | 563-679 | `nearby` hits read from scratch but `app/tools/retrieval.py:152,175,201` hardcodes `0.0 AS similarity` — no nearby hit can ever count as viable | WARNING (WR-01 from review) | `viable_candidates_per_step` systematically undercounts on nearby-driven flows; `rule8_met_per_step` stays False when model legitimately had viable nearby results; decisiveness gap signal biased low |
| `scripts/eval_agent.py` | 632-679 | `rule8_met_per_step_from_state` typed path uses `covered_types >= set(requested_types)` collapsing duplicate types; no-types fallback sums per-step counts without deduplication by `place_id` | WARNING (WR-02 from review) | Rule-8 met flag can be True too early (one restaurant venue covers two restaurant stops); `rule8_met_but_kept_searching_steps` overstates the decisiveness gap Phase 13 diagnoses against |
| `scripts/eval_falsifier.py` | 232-277 | Anchor non-regression check pools live-run and baseline summaries over structurally different scenario universes (run = omakase only; baselines = omakase + refinement_cheaper) | WARNING (CR-01 from review — the review marks this Critical, not a code STUB) | With both medians currently at 1.0 the verdict happens to be correct; the moment either baseline is regenerated below 1.0 the comparison becomes apples-to-oranges and can produce false anchor regressions or false PASSes |

**No TBD/FIXME/XXX debt markers** found in any file modified by this phase.

**Stub classification:** All anti-patterns above are measurement-semantic biases or a missing guard, not placeholder/stub implementations. Data does flow; the above issues affect the *accuracy* of what flows. None prevent Phase 13 from starting, but CR-01 and WR-01/02/08 should be addressed before the Phase 13 gate verdict is accepted as authoritative.

### Human Verification Required

#### 1. Falsifier grades correct matrix run in run-dir mode

**Test:** Execute `make eval-matrix` (standard, not refinement), then immediately `make eval-falsifier` with no RUN_DIR override. Inspect the output for a printed run-dir path and confirm that path corresponds to the just-completed eval_matrix.yaml run (not a prior refinement-matrix run).

**Expected:** The report header or first line names the resolved run directory; the per-scenario breakdown for gpt-5-mini shows only `omakase_mission_open_ended` (no `refinement_cheaper` from a different matrix appearing in the gpt-5-mini pooled calculation).

**Why human:** WR-06: `_latest_run_dir` picks by ISO8601 name with no validation that `summary.json` came from `configs/eval_matrix.yaml`. Only observable at runtime when run ordering can be controlled. Grep cannot detect ordering.

#### 2. Anchor non-regression fails correctly when anchor score drops below baseline

**Test:** Temporarily lower one field in `configs/eval_baselines/openai__gpt-4o-mini__omakase_mission_open_ended.json` (e.g. set `committed_itinerary_rate.median` to `0.9`) then run `make eval-matrix` + `make eval-falsifier`. Verify exit code is 1 and the verdict line names "anchor regression" as the cause. Restore the file.

**Expected:** Exit 1, output includes "FAIL (anchor regression)", gpt-4o-mini pooled rate printed as less than the modified baseline floor.

**Why human:** CR-01: the anchor non-regression comparison pools over different scenario universes (run = omakase only; baselines = omakase + refinement). With both currently at 1.0 the bug is dormant. Verifying the code correctly handles non-1.0 anchor rates requires a live matrix run against a temporarily modified baseline.

### Gaps Summary

No structural gaps block the phase goal. All 7 must-have truths are verified in the codebase. The phase delivers:
- In-graph `step_telemetry` (INST-04): implemented, tested, JSON-safe, prod-safe
- Three harness decisiveness fields (INST-01/02/03): implemented, tested, wired end-to-end via unfiltered `asdict` write path
- Falsifier report (INST-05): artifact-reading, per-model numbers, 0/1/2 exit codes, 17 tests passing, smoke test against real baselines
- Comparison floor bookkeeping (ANCH-02/03): deferred cells documented consistently across 5 files, parity test confirming non-deferred floor

Two human-verification items remain because the falsifier's run-dir selection and anchor comparison have structural runtime-ordering sensitivities that cannot be verified by static analysis. These are advisory until Phase 13 produces its first make eval-matrix run.

**Advisory (from 12-REVIEW.md):** The code review found 1 critical + 8 warning measurement-semantic issues. None are stubs or missing implementations, but CR-01 (mismatched anchor scenario universes) and WR-01/02/08 (nearby never viable, duplicate-type collapse, missing int guard) will distort Phase 13 verdict quality if not addressed before the go/no-go decision is made.

---

_Verified: 2026-06-12_
_Verifier: Claude (gsd-verifier)_
