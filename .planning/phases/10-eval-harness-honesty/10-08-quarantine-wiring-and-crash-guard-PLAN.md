---
phase: 10-eval-harness-honesty
plan: 08
type: execute
wave: 1
gap_closure: true
closes_crs: [CR-03, CR-02]
depends_on: []
files_modified:
  - scripts/eval_matrix.py
  - tests/unit/test_eval_matrix.py
  - scripts/eval_agent.py
  - tests/unit/test_eval_agent.py
autonomous: true
requirements: [EVAL-02, EVAL-01]
must_haves:
  truths:
    - "main() passes eval_queries_config to aggregate_cell_jsons so baseline_eligible reaches the real summary.json"
    - "A test invokes the main() aggregation path and asserts baseline_eligible appears in the written summary.json scenario blocks"
    - "_constraints_for_case does not crash on hand_written cases whose expected_results is None (the 5 clarification cases)"
    - "A regression test runs _constraints_for_case over every hand_written case in configs/eval_queries.yaml and asserts no exception"
  artifacts:
    - path: "scripts/eval_matrix.py"
      provides: "main() wires eval_queries_config into aggregate_cell_jsons"
      contains: "eval_queries_config"
    - path: "scripts/eval_agent.py"
      provides: "None-guarded expected_results fallback in _constraints_for_case"
      contains: "expected_results is not None"
    - path: "tests/unit/test_eval_matrix.py"
      provides: "test that main()'s aggregation emits baseline_eligible"
      contains: "baseline_eligible"
    - path: "tests/unit/test_eval_agent.py"
      provides: "regression test over every hand_written case asserting no crash"
      contains: "_constraints_for_case"
  key_links:
    - from: "scripts/eval_matrix.py main()"
      to: "aggregate_cell_jsons eval_queries_config param"
      via: "load_eval_queries(args.eval_queries)"
      pattern: "eval_queries_config"
    - from: "scripts/eval_agent.py _constraints_for_case"
      to: "case.expected_results.min_stops"
      via: "guarded by `case.expected_results is not None`"
      pattern: "expected_results is not None"
---

<objective>
Close two BLOCKER gaps that share the eval runner/matrix surface:

CR-03 (EVAL-02): `aggregate_cell_jsons` accepts an `eval_queries_config` parameter that writes
`baseline_eligible` into each scenario block, but `main()` at line 791 calls it WITHOUT that config
(`aggregate_cell_jsons(output_dir, llm_provider_override=args.llm_provider_override)`). Every real
summary.json therefore omits `baseline_eligible` entirely, and Phase 11 baseline tooling will default
the quarantined `late_night_closure_cascade` scenario to eligible — silently negating the D-10-09
quarantine.

CR-02 (EVAL-01 / harness trustworthiness): `_constraints_for_case` at lines 649-653 unconditionally
dereferences `case.expected_results.min_stops` / `.max_stops` when `explicit_num_stops_from_text`
returns None. 5 of 30 hand_written cases (the `expects_clarification_or_relaxation=true` cases:
impossible_four_am_five_star, impossible_cheap_michelin, impossible_north_beach_sushi_4am,
overconstrained_walkable_three_neighborhoods, closed_monday_brunch) have `expected_results=None` by
design and crash the default full-suite run with `AttributeError`, aborting the whole run.

Purpose: Make the quarantine flag real in pipeline output (EVAL-02) and make the default
`python scripts/eval_agent.py` full-suite run survive every checked-in case (EVAL-01 harness honesty).
Output: One-line wiring fix in `main()`; one-line guard in `_constraints_for_case`; two regression tests.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-eval-harness-honesty/10-VERIFICATION.md
@.planning/phases/10-eval-harness-honesty/10-02-SUMMARY.md
@.planning/phases/10-eval-harness-honesty/10-03-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire eval_queries_config into aggregate_cell_jsons in main() (CR-03)</name>
  <files>scripts/eval_matrix.py, tests/unit/test_eval_matrix.py</files>
  <read_first>
    - scripts/eval_matrix.py lines 44-51 (the `from app.eval.config import (...)` block — `load_eval_matrix` is already imported here; `load_eval_queries` is NOT and must be added), lines 209-384 (`aggregate_cell_jsons` signature `(output_dir, llm_provider_override=None, eval_queries_config=None)` and how it writes `scenario_block["baseline_eligible"]` only when `eval_queries_config is not None`), lines 591-595 (the `--eval-queries` argparse arg, default `_DEFAULT_EVAL_QUERIES_REL = "configs/eval_queries.yaml"`), lines 788-808 (the `main()` callsite: `summary = aggregate_cell_jsons(output_dir, llm_provider_override=args.llm_provider_override)`)
    - app/eval/config.py lines 263-270 — `load_eval_queries(path=DEFAULT_EVAL_QUERIES_PATH) -> EvalQueriesConfig` is the loader to call; `args.eval_queries` is a string path.
    - configs/eval_queries.yaml — confirm `late_night_closure_cascade` carries `baseline_eligible: false` (set in 10-03) so the wired path actually surfaces a False flag.
    - tests/unit/test_eval_matrix.py lines 312-440 (`_write_cell` / `_write_cell_with_aggregate` helpers) for how to seed per-cell JSONs; lines 540-590 for how existing tests call `aggregate_cell_jsons` directly.
  </read_first>
  <behavior>
    - After the fix, running the aggregation as `main()` wires it (config loaded from `args.eval_queries`) produces a summary whose scenario blocks each carry a `baseline_eligible` key.
    - For the `late_night_closure_cascade` scenario the surfaced value is `False`; for an ordinary scenario (e.g. `refinement_cheaper`) it is `True`.
    - Existing direct `aggregate_cell_jsons(tmp_path)` tests (no config) are unaffected — they still omit `baseline_eligible` (the param defaults to None).
  </behavior>
  <action>
    In scripts/eval_matrix.py: add `load_eval_queries` to the existing `from app.eval.config import (...)` import block (lines 45-51). Change the `main()` callsite (line 791) from
    `summary = aggregate_cell_jsons(output_dir, llm_provider_override=args.llm_provider_override)`
    to pass `eval_queries_config=load_eval_queries(args.eval_queries)` as a third keyword argument. Wrap the `load_eval_queries(args.eval_queries)` call in a try/except (OSError, ValueError) that, on failure, logs a warning via the module `_log` and falls back to `eval_queries_config=None` — a missing/malformed queries file must NOT block summary writing (the review explicitly notes "Wrap the load in try/except if a missing queries file should not block summary writing"). Do not change `aggregate_cell_jsons` itself, the argparse definition, or `run_matrix`. In tests/unit/test_eval_matrix.py: add `test_main_aggregation_surfaces_baseline_eligible` (or extend an existing main-path test) that seeds per-cell JSONs for `late_night_closure_cascade` and one ordinary scenario, drives the aggregation through the same call main uses — either by invoking `main()` with `--output-dir` pointed at the seeded dir, `--dry-run`-incompatible so instead mock `run_matrix` to return `(0, [])` and let main fall through to aggregation + summary write, OR (simpler and equivalent for this assertion) call `aggregate_cell_jsons(cell_dir, eval_queries_config=load_eval_queries("configs/eval_queries.yaml"))` and assert the same wiring main now uses. Prefer driving `main()` with a mocked `run_matrix` so the test pins the actual production callsite; assert the written summary.json contains `baseline_eligible: false` under the late_night scenario block and `baseline_eligible: true` under the ordinary scenario block.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -q -k "baseline_eligible or main"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "load_eval_queries" scripts/eval_matrix.py` returns >= 2 (import + main callsite).
    - `grep -c "eval_queries_config=load_eval_queries" scripts/eval_matrix.py` returns 1 (the wired call in main).
    - The new/extended test asserts the written summary's late_night scenario block has `baseline_eligible` == False and an ordinary scenario block has `baseline_eligible` == True.
    - `poetry run pytest tests/unit/test_eval_matrix.py -q` exits 0.
    - A missing/malformed eval-queries file does not raise out of main()'s aggregation step (covered by the try/except; assert via a test that points `--eval-queries` at a nonexistent path and still gets a written summary, OR document the fallback path is exercised).
  </acceptance_criteria>
  <done>main() passes `eval_queries_config=load_eval_queries(args.eval_queries)` (with a try/except fallback to None) so baseline_eligible reaches real summary.json; a test pins the production callsite and asserts the late_night flag is False in the output.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: None-guard the expected_results fallback in _constraints_for_case (CR-02)</name>
  <files>scripts/eval_agent.py, tests/unit/test_eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py lines 629-657 — `_constraints_for_case`. The crash is at lines 649-653: after `num_stops = explicit_num_stops_from_text(case.query)`, the `if num_stops is None:` branch reads `case.expected_results.min_stops` and `.max_stops` with no None check. `case.expected_results` is `None` for clarification cases.
    - app/eval/config.py — `EvalQuery` model: confirm `expected_results` is `Optional` (None-able) and that `expects_clarification_or_relaxation` cases set it to None; `min_stops`/`max_stops` live on the `expected_results` object.
    - configs/eval_queries.yaml — the 5 cases with `expected_results` absent / `expects_clarification_or_relaxation: true`: impossible_four_am_five_star, impossible_cheap_michelin, impossible_north_beach_sushi_4am, overconstrained_walkable_three_neighborhoods, closed_monday_brunch.
    - tests/unit/test_eval_agent.py lines 11-21 (imports already include `_constraints_for_case`, `load_eval_queries`, `EvalQuery`), lines 89-176 (existing `_constraints_for_case` tests that load `hand_written` cases from `configs/eval_queries.yaml` — the regression test mirrors the `load_eval_queries("configs/eval_queries.yaml").hand_written` loop already used at lines 167/176).
  </read_first>
  <behavior>
    - `_constraints_for_case` on any of the 5 clarification cases (expected_results=None, no explicit "N stops" in the query) returns a `UserConstraints` with `num_stops=None` — no AttributeError.
    - `_constraints_for_case` on a case with explicit text stops ("3 stops") still extracts num_stops from text (text precedence unchanged).
    - `_constraints_for_case` on a case with expected_results present and min==max still falls back to that count (existing behavior preserved).
  </behavior>
  <action>
    In scripts/eval_agent.py change the fallback guard at line 649 from `if num_stops is None:` to `if num_stops is None and case.expected_results is not None:` so the `case.expected_results.min_stops` / `.max_stops` dereferences at lines 650-651 only run when `expected_results` exists. Leave the inner `if min_s is not None and max_s is not None and min_s == max_s:` logic and the `UserConstraints(...)` return unchanged — when `expected_results` is None, `num_stops` simply stays None and is passed through. In tests/unit/test_eval_agent.py add `test_constraints_for_case_no_crash_over_all_hand_written_cases` that loads `cases = load_eval_queries("configs/eval_queries.yaml").hand_written` and, in a loop over every case, calls `_constraints_for_case(case)` inside the test body (any exception fails the test naturally — optionally assert it returns a `UserConstraints` instance). Add a focused assertion that for at least one known clarification case (e.g. `impossible_four_am_five_star`) the returned `num_stops` is None.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -q -k "constraints_for_case"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "expected_results is not None" scripts/eval_agent.py` returns >= 1 (the guard is present in `_constraints_for_case`).
    - `poetry run python -c "from scripts.eval_agent import _constraints_for_case; from app.eval.config import load_eval_queries; [(_constraints_for_case(c)) for c in load_eval_queries('configs/eval_queries.yaml').hand_written]; print('OK')"` prints `OK` and exits 0 (every hand_written case builds constraints without raising).
    - The regression test iterates over EVERY `hand_written` case and passes; it asserts `num_stops is None` for `impossible_four_am_five_star`.
    - `poetry run pytest tests/unit/test_eval_agent.py -q` exits 0.
  </acceptance_criteria>
  <done>_constraints_for_case guards the expected_results dereference with `case.expected_results is not None`; all 30 hand_written cases build constraints without AttributeError; a regression test loops over every case.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| configs/eval_queries.yaml -> eval runner | Checked-in, trusted config; no untrusted external input |
| summary.json baseline_eligible -> Phase 11 baseline tooling | Internal handoff; the flag's absence (CR-03) is a correctness, not security, risk |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-08-01 | Tampering | baseline_eligible quarantine flag | mitigate | Test pins main()'s aggregation callsite so a future refactor that drops eval_queries_config makes the test go red (the exact CR-03 regression) |
| T-10-08-02 | Denial of Service | full-suite eval run aborts on 1 case | mitigate | None-guard + all-cases regression test ensures one clarification case can no longer crash the entire run (the CR-02 abort) |
| T-10-08-03 | Tampering | malformed eval-queries file | accept | try/except fallback to None means a corrupt config degrades summary (no flag) rather than crashing; low risk, internal tooling |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py tests/unit/test_eval_agent.py -q` exits 0.
- Manual: `grep -n "eval_queries_config=load_eval_queries" scripts/eval_matrix.py` and `grep -n "expected_results is not None" scripts/eval_agent.py` both show the fixes.
- Full suite (per project memory — real-graph tests leak a DB pool unless run together): `make test` exits 0.
</verification>

<success_criteria>
- baseline_eligible reaches real summary.json via main() (CR-03 closed; EVAL-02 PARTIAL -> VERIFIED).
- The default full eval suite no longer crashes on clarification cases (CR-02 closed; EVAL-01 harness-trustworthiness gap closed).
- The verification truths "The D-10-09 quarantine flag (baseline_eligible) is honored in real summary.json output (EVAL-02)" and "Running the full default eval suite (all hand_written cases) does not crash (EVAL-01)" flip to VERIFIED.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-08-SUMMARY.md` when done.
</output>
