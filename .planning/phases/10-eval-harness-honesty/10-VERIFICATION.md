---
phase: 10-eval-harness-honesty
verified: 2026-06-11T06:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/6
  gaps_closed:
    - "CR-01 (EVAL-03): check_eval_gates.py now walks summary['scenarios'][*]['providers'][family]; integration test feeds real aggregate_cell_jsons() output and asserts exit 1 on hard-gate violation"
    - "CR-03 (EVAL-02): main() passes eval_queries_config=load_eval_queries(args.eval_queries) (with try/except fallback to None) so baseline_eligible reaches real summary.json"
    - "CR-02 (EVAL-01): _constraints_for_case guards with `case.expected_results is not None` before dereferencing min_stops; all 30 hand_written cases run without crash"
    - "CR-04 (EVAL-05): hardcoded /Users/pnhek path replaced with REPO_ROOT = Path(__file__).resolve().parents[2]"
    - "CR-05 (EVAL-05): response_metadata, usage_metadata, tool_calls all routed through _redact(json.dumps(...)); post-write guard extended via _scan_fixture_for_secrets to check _SECRET_ENV_VARS values"
  gaps_remaining: []
  regressions: []
---

# Phase 10: Eval Harness Honesty Verification Report

**Phase Goal:** The eval harness distinguishes infrastructure failure from model failure, measures only prod-shaped behavior, and documents merge gates that are actually satisfiable — so that Phase 11's baseline regen is trustworthy on the first attempt.
**Verified:** 2026-06-11T06:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (plans 10-07, 10-08, 10-09 executed)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A turn-0/turn-1 exception produces an ERROR-status record excluded from score aggregation, surfaced in summary.json as an error count — never 1.0 or 0.0 (EVAL-01) | VERIFIED | `make_error_record` wired in both threading branches; `aggregate_results` filters `status == "ok"`; `RaisingChatModel` replay tests assert `n_scored==0`; `_constraints_for_case` None-guard (CR-02) ensures all 30 hand_written cases build constraints without AttributeError — empirically confirmed on all 5 clarification cases |
| 2 | `late_night_closure_cascade` is explicitly quarantined from baselines and merge gates, with the decision recorded next to the scenario config (EVAL-02) | VERIFIED | `baseline_eligible: false` in eval_queries.yaml; `EvalQuery.baseline_eligible` field in config.py; `main()` now passes `eval_queries_config=load_eval_queries(args.eval_queries)` at line 808 (CR-03 fixed); `test_main_aggregation_surfaces_baseline_eligible` pins the callsite and asserts `baseline_eligible=False` for late_night in written summary.json |
| 3 | Per-family merge gates are re-derived from honest anchor data and enforced by an executable Makefile target that exits non-zero on regression (EVAL-03) | VERIFIED | `configs/eval_gates.yaml` has 7 entries with honest values (no strict-1.0 gate); `_check_gate` now walks `summary.get("scenarios", {})` (CR-01 fixed); integration test `test_integration_real_aggregate_output_fires_hard_gate` feeds real `aggregate_cell_jsons()` output and asserts `main(...) == 1`; empirically confirmed: running checker against a nested-shape summary with `committed_itinerary_rate median=0.4` exits 1 with "HARD GATE VIOLATION: ['openai/gpt-4o-mini']" |
| 4 | A test asserts baseline JSON provider cells match matrix YAML entries in both directions, modulo documented deferrals (EVAL-04) | VERIFIED | `test_baseline_provider_cells_match_matrix_entries` parametrized over both matrix files; `test_late_night_scenario_is_baseline_ineligible` passes; late_night quarantine is distinct from deferred-baseline status |
| 5 | A per-provider live-probe Make target exists, is documented as mandatory pre-matrix, captured real-wire responses are checked in as fixtures consumed by adapter tests, and redaction is fail-closed (EVAL-05) | VERIFIED | `probe_provider_capture.py` routes all three value-bearing fields through `_redact(json.dumps(...))` (CR-05 fixed); `_scan_fixture_for_secrets` covers both regex and `_SECRET_ENV_VARS` channels; `test_post_write_guard_catches_env_var_sourced_secret` plants a non-regex-shaped secret and asserts fixture deleted + return 2; hardcoded author path replaced with `REPO_ROOT = Path(__file__).resolve().parents[2]` (CR-04 fixed); 11 probe tests pass |
| 6 | The `build_chat_model` gpt-5 dispatch branch has factory-level tests; `ScriptedChatModel` is exercised via `ainvoke`; `vibe_check` sync call is non-blocking or flag-documented (EVAL-06) | VERIFIED | `test_build_chat_model_gpt5_returns_openai_reasoning_chat_model` asserts `use_responses_api=True`; `test_scripted_chat_model_ainvoke_works` proves ainvoke executor fallback; `vibe.py:78-80` has D-10-17 executor comment; behavior unchanged |

**Score:** 6/6 truths VERIFIED

### Empirical Check Results (Re-verification Focus)

| Check | Command | Result | Status |
|-------|---------|--------|--------|
| CR-01: check_eval_gates exits non-zero on real nested summary with below-gate cell | `poetry run python scripts/check_eval_gates.py /tmp/nested_summary.json` (rate=0.4 < 0.8 gate) | "HARD GATE VIOLATION: ['openai/gpt-4o-mini']"; EXIT CODE: 1 | PASS |
| CR-01: flat `providers` key gone from checker | `grep -n 'summary.get("providers"' scripts/check_eval_gates.py` | no matches | PASS |
| CR-03: main() passes eval_queries_config | `grep -n "eval_queries_config=_eval_queries_cfg" scripts/eval_matrix.py` | line 808 | PASS |
| CR-02: _constraints_for_case on all 5 clarification cases | `poetry run python -c "...all 5 cases..."` | ALL CLARIFICATION CASES: OK (num_stops=None, no crash) | PASS |
| CR-04: no hardcoded /Users/pnhek path | `grep -n "/Users/pnhek" tests/unit/test_probe_provider_capture.py` | no matches (exit 1) | PASS |
| CR-05: all three fields routed through _redact | `grep -n "_redact(json.dumps" scripts/probe_provider_capture.py` | lines 226, 230, 235 | PASS |
| CR-05: post-write guard checks _SECRET_ENV_VARS | `grep -n "_SECRET_ENV_VARS" scripts/probe_provider_capture.py` | lines 73, 93, 119 (definition + _redact + _scan_fixture_for_secrets) | PASS |
| Full test suite | `poetry run pytest tests/ -q --ignore=tests/integration` | 1127 passed, 11 skipped | PASS |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/check_eval_gates.py` | gate checker that walks nested scenarios->providers shape | VERIFIED | `_check_gate` iterates `summary.get("scenarios", {}).items()`; skips `baseline_eligible=False` blocks; no top-level `providers` lookup remains |
| `tests/unit/test_check_eval_gates.py` | tests using real nested summary shape + aggregate_cell_jsons integration test | VERIFIED | `_make_summary` emits `{"scenarios": {"refinement_cheaper": {"providers": ...}}}` shape; 16 tests pass; `test_integration_real_aggregate_output_fires_hard_gate` feeds real aggregator output |
| `scripts/eval_matrix.py` | main() wires eval_queries_config into aggregate_cell_jsons | VERIFIED | `load_eval_queries` imported; line 796 loads config; line 808 passes `eval_queries_config=_eval_queries_cfg`; try/except fallback to None on OSError/ValueError |
| `tests/unit/test_eval_matrix.py` | test that main() aggregation emits baseline_eligible | VERIFIED | `test_main_aggregation_surfaces_baseline_eligible` and `test_main_aggregation_survives_missing_eval_queries_file` at lines 1631, 1691 |
| `scripts/eval_agent.py` | None-guarded expected_results fallback in _constraints_for_case | VERIFIED | Line 649: `if num_stops is None and case.expected_results is not None:` |
| `tests/unit/test_eval_agent.py` | regression test over every hand_written case asserting no crash | VERIFIED | `TestConstraintsForCaseClarificationGuard` at line 1975 with `test_no_crash_over_all_hand_written_cases` at 2004 |
| `scripts/probe_provider_capture.py` | fail-closed redaction + env-var-aware post-write guard | VERIFIED | All 3 value-bearing fields through `_redact(json.dumps(...))`; `_scan_fixture_for_secrets` checks regex + env-var channels |
| `tests/unit/test_probe_provider_capture.py` | portable subprocess test + env-var post-write-guard test | VERIFIED | `REPO_ROOT = Path(__file__).resolve().parents[2]` at line 19; `test_post_write_guard_catches_env_var_sourced_secret` at line 172; 11 tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_check_gate` | `summary['scenarios'][*]['providers'][family]` | scenarios loop + baseline_eligible skip | WIRED | Lines 167-173; quarantined blocks skipped; empirically exits 1 on below-gate cell |
| `main()` | `aggregate_cell_jsons eval_queries_config param` | `load_eval_queries(args.eval_queries)` at line 796 | WIRED | try/except fallback to None on failure; `_eval_queries_cfg` passed at line 808 |
| `_constraints_for_case` | `case.expected_results.min_stops` | guarded by `case.expected_results is not None` | WIRED | Line 649; all 5 clarification cases return `UserConstraints(num_stops=None)` |
| `probe_provider_capture.py response_metadata/usage_metadata/tool_calls` | `_redact` | `json.loads(_redact(json.dumps(..., default=str)))` | WIRED | Lines 226, 230, 235 |
| `probe_provider_capture.py post-write guard` | `_SECRET_ENV_VARS` env values | `_scan_fixture_for_secrets` covers both channels | WIRED | Lines 119-122; planted non-regex secret triggers delete + return 2 |
| `Makefile eval-gates-check` | `scripts/check_eval_gates.py` | POETRY_RUN target | WIRED | Target present; exits non-zero when gate fires |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVAL-01 | Plans 10-01, 10-02, 10-08 | ERROR-status records; fail-open scorer paths closed; full suite no crash | SATISFIED | `make_error_record` wired; `aggregate_results` status=="ok" filter; CR-02 None-guard; 5 clarification cases confirmed no crash |
| EVAL-02 | Plans 10-03, 10-08 | `late_night_closure_cascade` quarantined with decision recorded | SATISFIED | `baseline_eligible: false` in eval_queries.yaml; main() passes eval_queries_config (CR-03 fixed); `test_main_aggregation_surfaces_baseline_eligible` pins callsite |
| EVAL-03 | Plans 10-04, 10-07 | Per-family gates satisfiable + executable gate checker exits non-zero | SATISFIED | `eval_gates.yaml` has honest values; `_check_gate` walks nested scenarios shape (CR-01 fixed); integration test proves exit 1 against real `aggregate_cell_jsons` output |
| EVAL-04 | Plan 10-03 | Baseline<->matrix parity test | SATISFIED | `test_baseline_provider_cells_match_matrix_entries` covers both matrix files; `test_late_night_scenario_is_baseline_ineligible` passes |
| EVAL-05 | Plans 10-05, 10-09 | Per-provider live-probe + fixtures + adapter tests + fail-closed redaction | SATISFIED | Probe, target, fixture dir, adapter tests all present and correct; CR-05 fixed (3 fields redacted); CR-04 fixed (REPO_ROOT resolution); 11 probe tests pass |
| EVAL-06 | Plan 10-06 | Factory tests for gpt-5 dispatch + ScriptedChatModel ainvoke + vibe.py doc | SATISFIED | All three goals verified; tests pass; behavior unchanged |

### Anti-Patterns Found

No blockers or warnings in files modified by plans 10-07, 10-08, 10-09. No TBD/FIXME/XXX debt markers. No hardcoded paths. No stub implementations.

### Human Verification Required

None. All must-haves are verifiable programmatically. The empirical checks in this report cover all prescribed re-verification points.

### Gaps Summary

No gaps remain. All five CRs from the initial verification are closed:

- **CR-01 (EVAL-03, BLOCKER):** CLOSED — `_check_gate` walks `summary['scenarios'][*]['providers'][family]`; integration test proves exit 1 against real `aggregate_cell_jsons` output.
- **CR-02 (EVAL-01, BLOCKER):** CLOSED — `if num_stops is None and case.expected_results is not None:` guard at line 649 of `eval_agent.py`; all 30 hand_written cases confirmed crash-free.
- **CR-03 (EVAL-02, BLOCKER):** CLOSED — `main()` at line 808 passes `eval_queries_config=_eval_queries_cfg`; `test_main_aggregation_surfaces_baseline_eligible` pins the callsite.
- **CR-04 (EVAL-05, BLOCKER):** CLOSED — `REPO_ROOT = Path(__file__).resolve().parents[2]` replaces hardcoded author path; test passes from any CWD.
- **CR-05 (EVAL-05, WARNING):** CLOSED — `_scan_fixture_for_secrets` covers regex + env-var channels; all three value-bearing fixture fields route through `_redact(json.dumps(...))`.

---

_Verified: 2026-06-11T06:00:00Z_
_Verifier: Claude (gsd-verifier)_
