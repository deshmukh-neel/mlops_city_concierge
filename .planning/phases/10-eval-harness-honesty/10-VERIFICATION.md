---
phase: 10-eval-harness-honesty
verified: 2026-06-11T04:32:32Z
status: gaps_found
score: 3/6 must-haves verified
overrides_applied: 0
gaps:
  - truth: "make eval-gates-check exits non-zero when a summary's hard-gated cell drops below its gate value (EVAL-03)"
    status: failed
    reason: "check_eval_gates.py reads summary.get('providers', {}) — a top-level providers key that aggregate_cell_jsons never writes. The real summary shape is {'scenarios': {'<id>': {'providers': {...}}}}. Empirically verified: running the checker against a correct nested summary prints NOT-EVALUABLE for every hard-gated family and exits 0. Hard gates can never fire. The unit tests bake in the same wrong flat schema so CI is green while the integration is broken. (CR-01)"
    artifacts:
      - path: "scripts/check_eval_gates.py"
        issue: "Line 155: summary.get('providers', {}) reads a top-level 'providers' key that does not exist in aggregate_cell_jsons output"
      - path: "tests/unit/test_check_eval_gates.py"
        issue: "_make_summary() constructs {'providers': providers} — the flat shape that the real pipeline never produces"
    missing:
      - "Fix _check_gate to walk summary.get('scenarios', {}).items() and look up the family under each scenario_block['providers']"
      - "Rewrite _make_summary in tests to produce the real {'scenarios': {'<id>': {'providers': {...}}}} shape"
      - "Add an integration-level test that feeds an actual aggregate_cell_jsons() output to the checker"

  - truth: "The D-10-09 quarantine flag (baseline_eligible) is honored in real summary.json output (EVAL-02)"
    status: failed
    reason: "aggregate_cell_jsons accepts an optional eval_queries_config parameter that writes baseline_eligible into each scenario block — but main() at line 791 calls aggregate_cell_jsons(output_dir, llm_provider_override=args.llm_provider_override) with no config. Every real summary.json therefore omits baseline_eligible entirely. Phase 11 baseline tooling will see no flag and default scenarios to eligible, negating the quarantine. Unit tests pass because they call aggregate_cell_jsons with the config explicitly. (CR-03)"
    artifacts:
      - path: "scripts/eval_matrix.py"
        issue: "Line 791: main() calls aggregate_cell_jsons without eval_queries_config; baseline_eligible never reaches real summary.json"
    missing:
      - "Pass eval_queries_config=load_eval_queries(args.eval_queries) to the aggregate_cell_jsons call in main()"
      - "Add a test that invokes main() and asserts baseline_eligible appears in the written summary.json"

  - truth: "Running the full default eval suite (all hand_written cases) does not crash (EVAL-01 / harness trustworthiness)"
    status: failed
    reason: "_constraints_for_case at line 650 unconditionally dereferences case.expected_results.min_stops when explicit_num_stops_from_text returns None. expected_results is None by design for expects_clarification_or_relaxation=true cases. Empirically confirmed: 5 of 30 hand_written cases have expected_results=None (impossible_four_am_five_star, impossible_cheap_michelin, impossible_north_beach_sushi_4am, overconstrained_walkable_three_neighborhoods, closed_monday_brunch); calling _constraints_for_case on any of them raises AttributeError. The exception escapes evaluate_case (try/finally only) and aborts the entire run. The matrix runner dodges this only because it passes --scenario-ids + --max-queries 1 scoped to cases with expected_results. (CR-02)"
    artifacts:
      - path: "scripts/eval_agent.py"
        issue: "Lines 649-651: num_stops fallback unconditionally accesses case.expected_results.min_stops without checking for None; 5 clarification cases with expected_results=None crash the runner"
    missing:
      - "Guard the fallback: if num_stops is None and case.expected_results is not None: (before dereferencing min_stops)"
      - "Add a regression test that calls _constraints_for_case over every hand_written case in eval_queries.yaml and asserts no exception"

  - truth: "The post-write secret-scan guard in probe_provider_capture.py is fail-closed (EVAL-05)"
    status: failed
    reason: "Three fixture fields bypass _redact: response_metadata goes through _sanitize_response_metadata (blanks only 2 fixed keys, does not recurse), usage_metadata is written raw, tool_calls are written raw. _redact is only applied to additional_kwargs_values. Additionally, the post-write guard (lines 224-233) scans only the 4 _SECRET_PATTERNS regexes — it does NOT check env-var-sourced secret values (the D-10-13 substitution that _redact applies before writing is absent from the re-read guard). A non-regex-shaped API key appearing in response_metadata or tool_calls would be written to a checked-in fixture and survive the guard. The SUMMARY claims 'fail-closed' but the implementation has three unredacted write paths and an incomplete post-write scan. (CR-05)"
    artifacts:
      - path: "scripts/probe_provider_capture.py"
        issue: "Lines 196-214: response_metadata, usage_metadata, and tool_calls bypass _redact; post-write guard at lines 224-233 omits env-var secret value scan"
    missing:
      - "Route response_metadata, usage_metadata, and tool_calls through _redact before writing to fixture"
      - "Extend the post-write guard to check env-var-sourced secret values (mirror the _redact env-var substitution)"
---

# Phase 10: Eval Harness Honesty Verification Report

**Phase Goal:** The eval harness distinguishes infrastructure failure from model failure, measures only prod-shaped behavior, and documents merge gates that are actually satisfiable — so that Phase 11's baseline regen is trustworthy on the first attempt.
**Verified:** 2026-06-11T04:32:32Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A turn-0/turn-1 exception produces an ERROR-status record excluded from score aggregation and surfaced in summary.json as an error count — never 1.0 or 0.0; the 21-14-30Z conditions yield error records and zero scored cells (EVAL-01) | VERIFIED | `make_error_record` function exists at eval_agent.py:170; `aggregate_results` filters on `status == "ok"` at line 1026; `RaisingChatModel` replay tests at test_eval_agent.py:1316-1380 assert n_scored==0 and refinement_minimal_edit_mean==0.0 for all-error runs; `partial_state` scoring path confirmed removed (only 2 comment references remain) |
| 2 | `late_night_closure_cascade` is explicitly quarantined from baselines and merge gates, with the decision recorded next to the scenario config (EVAL-02) | PARTIAL | `baseline_eligible: false` field exists in EvalQuery (config.py:186) and is set in eval_queries.yaml:428. The flag IS parsed and honored by `aggregate_cell_jsons` when `eval_queries_config` is provided. However, `main()` never passes `eval_queries_config` (line 791), so real summary.json output never carries the flag. The quarantine decision IS recorded in three config locations but the runtime path is broken (CR-03). |
| 3 | Per-family merge gates are re-derived from honest anchor data and enforced by an executable Makefile target that exits non-zero on regression (EVAL-03) | FAILED | `configs/eval_gates.yaml` exists with 7 entries and honest values; `docs/eval_gates.md` documents semantics without duplicating numbers; `scripts/check_eval_gates.py` exists with correct exit-code convention. However, the checker reads `summary.get("providers", {})` at line 155 — a top-level key that `aggregate_cell_jsons` never produces. Empirically verified: checker prints NOT-EVALUABLE for every gated family and exits 0 against a correctly-shaped summary.json. `make eval-gates-check` is structurally a no-op. (CR-01) |
| 4 | A test asserts baseline JSON provider cells match matrix YAML entries in both directions, modulo documented deferrals (EVAL-04) | VERIFIED | `test_baseline_provider_cells_match_matrix_entries` at test_eval_matrix.py:116 is parametrized over `_MATRIX_TO_BASELINES` (both eval_matrix.yaml and eval_matrix_refinement.yaml); test passes. `test_late_night_scenario_is_baseline_ineligible` at line 1601 also passes. `late_night_closure_cascade` is NOT in `_DEFERRED_BASELINE_CELLS` (quarantine distinct from deferral). |
| 5 | A per-provider live-probe Make target exists, is documented as mandatory pre-matrix, and captured real-wire responses are checked in as fixtures consumed by adapter tests (EVAL-05) | PARTIAL | `scripts/probe_provider_capture.py` exists with `--provider` argparse; `make probe-providers` target exists (Makefile:158-159); `tests/fixtures/provider_payloads/.gitkeep` exists; `test_adapter_capture_on_real_wire_fixture` parametrized test exists in test_adapters.py. However, the post-write secret-scan guard is not fail-closed: `response_metadata`, `usage_metadata`, and `tool_calls` bypass `_redact` and the guard does not check env-var-sourced secrets. The EVAL-05 must-have "redaction is fail-closed" is not met. (CR-05) |
| 6 | The `build_chat_model` gpt-5 dispatch branch (`use_responses_api=True`) has factory-level tests; `ScriptedChatModel` is exercised via `ainvoke`; `vibe_check` sync call is non-blocking or flag-documented (EVAL-06) | VERIFIED | `test_build_chat_model_gpt5_returns_openai_reasoning_chat_model` at test_llm_factory.py:730 asserts `use_responses_api=True`; `test_build_chat_model_gpt4o_mini_stays_plain_chat_openai` calls `OpenAIReasoningChatModel.assert_not_called()`; `test_scripted_chat_model_ainvoke_works` at line 769 proves ainvoke executor fallback; `app/agent/critique/vibe.py:78-80` has executor comment referencing D-10-17 with no behavior change. |

**Score:** 2/6 truths fully VERIFIED; 2/6 PARTIAL (EVAL-02 is partial because runtime path is broken by CR-03; EVAL-05 is partial because security claim is not met by CR-05); 2/6 FAILED (EVAL-03 due to CR-01; direct eval run crashes due to CR-02).

Note: CR-02 is a harness trustworthiness issue — while the matrix runner itself is not affected (it passes --scenario-ids scoped to expected_results cases), the default `python scripts/eval_agent.py` run crashes on 5 of 30 cases with AttributeError. This directly contradicts EVAL-01's goal of a trustworthy harness, and was explicitly noted in the code review as a gap this phase failed to address.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/eval_agent.py` | ERROR-status record path in both threading branches | VERIFIED | make_error_record exists; both branches use it; scored_results filter on status=="ok" |
| `tests/unit/test_eval_agent.py` | 21-14-30Z replay tests with RaisingChatModel | VERIFIED | RaisingChatModel at line 694; 3 replay tests at lines 1316-1380 |
| `scripts/eval_matrix.py` | n_scored/n_errored/cell_valid threading through aggregate_cell_jsons | VERIFIED | Fields present in aggregate_cell_jsons output block |
| `tests/unit/test_eval_matrix.py` | quarantine flag + parity test coverage | VERIFIED | test_late_night_scenario_is_baseline_ineligible at 1601; parity test at 116 |
| `app/eval/config.py` | baseline_eligible field on EvalQuery | VERIFIED | Field at line 186 with default True |
| `configs/eval_baselines/late_night_closure_cascade.json` | _observations annotation | VERIFIED | _observations key present at line 2 |
| `configs/eval_gates.yaml` | per-family gates with hard/advisory/status/rationale | VERIFIED | 7 entries; all required fields present; no strict-1.0 gate |
| `scripts/check_eval_gates.py` | summary.json gate-checker with correct exit codes | STUB | File exists and exit codes are correct in tests, but _check_gate reads wrong summary key — checker is a no-op against real output |
| `docs/eval_gates.md` | narrative gate semantics linking to YAML | VERIFIED | Contains aspirational, quarantined-legacy-threading, retired; links to configs/eval_gates.yaml |
| `Makefile` | eval-gates-check + probe-providers targets | VERIFIED | Both .PHONY targets present at lines 158 and 170 |
| `scripts/probe_provider_capture.py` | generalized --provider probe with fail-closed redaction | PARTIAL | Probe exists with --provider; redaction partial; 3 fields bypass _redact; post-write guard incomplete |
| `tests/unit/test_probe_provider_capture.py` | redaction unit tests | VERIFIED | 10 tests covering OpenAI/Anthropic/Google key patterns and post-write guard |
| `tests/fixtures/provider_payloads/.gitkeep` | directory tracked for fixtures | VERIFIED | .gitkeep exists |
| `tests/unit/test_adapters.py` | parametrized fixture-loading tests | VERIFIED | test_adapter_capture_on_real_wire_fixture parametrized over 4 providers; 4 skipped when fixtures absent |
| `tests/unit/test_llm_factory.py` | gpt-5 dispatch + gpt-4o-mini anchor + ScriptedChatModel ainvoke tests | VERIFIED | 3 tests added; use_responses_api assertion present |
| `app/agent/critique/vibe.py` | executor comment recording D-10-17 finding | VERIFIED | Comment at line 78-80; behavior unchanged |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_run_prod_threading` exception | make_error_record(status=error) | except clause | WIRED | Line 892 confirmed |
| `_run_legacy_threading` exception | make_error_record(status=error) | except clause | WIRED | Line 773 confirmed |
| `aggregate_results` | scored_results (status=="ok" filter) | list comprehension | WIRED | Line 1026 confirmed |
| `configs/eval_queries.yaml` late_night | EvalQuery.baseline_eligible=False | YAML parse | WIRED | Confirmed by poetry run python assert |
| `EvalQuery.baseline_eligible` | summary.json scenario block | aggregate_cell_jsons eval_queries_config param | NOT_WIRED | main() at line 791 does not pass eval_queries_config — flag never reaches real summary.json |
| `configs/eval_gates.yaml` | `scripts/check_eval_gates.py` gate lookup | summary.get("providers", {}) | PARTIAL | Gates YAML loads correctly; but checker reads wrong summary key (top-level providers vs nested scenarios.*.providers) |
| `Makefile eval-gates-check` | `scripts/check_eval_gates.py` | POETRY_RUN target | WIRED | Target present and invocable |
| `probe_provider_capture.py` | `tests/fixtures/provider_payloads/{provider}.json` | write on run | WIRED | Path confirmed via grep |
| `tests/fixtures/provider_payloads/{provider}.json` | `tests/unit/test_adapters.py` | parametrized loader | WIRED | _FIXTURE_DIR and skip logic confirmed |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| check_eval_gates exits non-zero when gpt-4o-mini committed_itinerary_rate below 0.8 (real nested summary shape) | `poetry run python scripts/check_eval_gates.py /tmp/real_summary.json` (nested scenarios shape) | NOT-EVALUABLE for all families; exit 0 | FAIL — hard gate never fires on real summary shape |
| check_eval_gates fires on flat-providers shape (the shape tests use) | `poetry run python scripts/check_eval_gates.py /tmp/fake_summary_flat.json` (rate=0.7 below 0.8) | HARD GATE VIOLATION; exit 1 | PASS — but this shape is not what aggregate_cell_jsons produces |
| _constraints_for_case on clarification case (expected_results=None) | `poetry run python3 -c "...from scripts.eval_agent import _constraints_for_case; _constraints_for_case(impossible_four_am_five_star_case)"` | AttributeError: 'NoneType' object has no attribute 'min_stops' | FAIL — 5 of 30 cases crash |
| ScriptedChatModel ainvoke works | `poetry run pytest tests/unit/test_llm_factory.py -q -k ainvoke` | 1 passed | PASS |
| Unit suite | `poetry run pytest tests/unit/test_eval_agent.py test_eval_matrix.py test_check_eval_gates.py test_adapters.py test_probe_provider_capture.py test_llm_factory.py -q` | 252 passed, 4 skipped | PASS — all unit tests pass (tests use wrong schema so CI is green while integration is broken) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/unit/test_probe_provider_capture.py` | 158 | Hardcoded absolute path `/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge` | BLOCKER | Test fails on CI (Ubuntu) and every other machine — guaranteed failure outside author's laptop (CR-04) |
| `scripts/eval_agent.py` | 650 | `case.expected_results.min_stops` without None guard | BLOCKER | 5 of 30 eval cases crash with AttributeError when run without --scenario-ids scoping (CR-02) |
| `scripts/check_eval_gates.py` | 155 | `summary.get("providers", {})` reads top-level key that aggregate_cell_jsons never produces | BLOCKER | eval-gates-check is structurally a no-op against real summary.json output (CR-01) |
| `scripts/eval_matrix.py` | 791 | `aggregate_cell_jsons(output_dir, ...)` called without eval_queries_config | BLOCKER | baseline_eligible never reaches real summary.json; quarantine inert in pipeline (CR-03) |
| `scripts/probe_provider_capture.py` | 196-233 | response_metadata/usage_metadata/tool_calls bypass _redact; post-write guard omits env-var check | WARNING | Redaction claim is not fail-closed: non-regex-shaped secrets in response_metadata or tool_calls would survive (CR-05) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVAL-01 | Plans 10-01, 10-02 | ERROR-status records; fail-open scorer paths closed | SATISFIED | make_error_record wired; aggregate_results filters status=="ok"; replay tests pass. Note: direct eval_agent.py run on all cases crashes (CR-02), which the code review flagged as a related gap not fully closed. |
| EVAL-02 | Plan 10-03 | late_night_closure_cascade quarantined | BLOCKED | Flag parsed and recorded in three places; but main() does not pass eval_queries_config so baseline_eligible is inert in real pipeline (CR-03) |
| EVAL-03 | Plan 10-04 | Per-family gates in machine-readable source, satisfiable, enforced | BLOCKED | YAML exists with honest values; checker script exists with correct exit codes; but checker reads wrong summary shape — no gate can ever fire against real output (CR-01) |
| EVAL-04 | Plan 10-03 | Baseline<->matrix parity test | SATISFIED | test_baseline_provider_cells_match_matrix_entries covers both matrix files; passes; late_night not added to deferred list |
| EVAL-05 | Plan 10-05 | Per-provider live-probe, fixtures, adapter tests | PARTIALLY SATISFIED | Probe, target, fixture dir, adapter tests all present. Redaction security claim not fully met (CR-05); portability broken (CR-04). Core capability (probe script + fixture-backed adapter tests) is functional. |
| EVAL-06 | Plan 10-06 | Factory tests for gpt-5 dispatch + ScriptedChatModel ainvoke + vibe.py doc | SATISFIED | All three goals achieved; tests pass; behavior unchanged |

### Gaps Summary

Four blockers prevent the phase goal from being fully achieved.

**Root cause cluster 1 — Gate checker schema mismatch (CR-01, EVAL-03):** `check_eval_gates.py` reads `summary.get("providers", {})` but `aggregate_cell_jsons` writes `{"scenarios": {"<id>": {"providers": {...}}}}`. The checker has never been able to find any gated family in a real `summary.json`. All hard gates are permanently NOT-EVALUABLE. The unit tests use the wrong flat schema so CI stays green. Phase 11's BASE-03 ("lock per-family merge gates in CI") depends on this working correctly.

**Root cause cluster 2 — Wiring gap in main() (CR-03, EVAL-02):** `aggregate_cell_jsons` was extended with an `eval_queries_config` parameter that writes `baseline_eligible` — but `main()` never passes it. The quarantine decision is real (flag, YAML, annotation, 3-place recording), but the runtime output is wrong. Phase 11 baseline tooling reading summary.json will see no flag.

**Root cause cluster 3 — Crash on clarification cases (CR-02, harness trustworthiness):** `_constraints_for_case` crashes on 5 of 30 eval_queries.yaml cases. The matrix runner is unaffected (scopes to one scenario at a time via --scenario-ids), but direct use of the harness ("run all hand_written cases") is broken. This is a Phase 10 harness-honesty gap.

**Root cause cluster 4 — Portability (CR-04):** Hardcoded absolute path in `test_probe_provider_capture.py` breaks `make test` on any machine other than the author's. This was present in the submitted phase state and affects CI portability.

**CR-05** (probe redaction not fully fail-closed) is a WARNING rather than a hard BLOCKER — the probe functions, redaction covers the most common key shapes, and the post-write guard catches regex-shaped secrets. However the claim of "fail-closed" in the SUMMARY and docs is inaccurate.

---

_Verified: 2026-06-11T04:32:32Z_
_Verifier: Claude (gsd-verifier)_
