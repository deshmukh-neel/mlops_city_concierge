---
phase: 10-eval-harness-honesty
plan: 02
type: execute
wave: 2
depends_on: ["10-01"]
files_modified:
  - scripts/eval_matrix.py
  - tests/unit/test_eval_matrix.py
autonomous: true
requirements: [EVAL-01]
must_haves:
  truths:
    - "summary.json carries per-cell n_scored and n_errored aggregated from each cell JSON's status counts"
    - "summary.json carries a top-level errors array listing each errored cell's stage/type"
    - "A cell with any errored run is flagged cell_valid=false in summary.json"
    - "eval_matrix exits non-zero when any cell errored, with error count distinct from violation count in output"
    - "The structural-check mode validates the new error-schema fields without live calls"
  artifacts:
    - path: "scripts/eval_matrix.py"
      provides: "n_scored/n_errored/errors threading through aggregate_cell_jsons + error-aware exit code + structural-check extension"
      contains: "n_errored"
    - path: "tests/unit/test_eval_matrix.py"
      provides: "aggregation error-threading test + structural-check error-schema test"
      contains: "n_errored"
  key_links:
    - from: "scripts/eval_agent.py per-run JSON (status field)"
      to: "scripts/eval_matrix.py::aggregate_cell_jsons"
      via: "aggregator reads status / n_scored / n_errored from each cell JSON"
      pattern: "n_scored|n_errored"
    - from: "scripts/eval_matrix.py error count"
      to: "process exit code"
      via: "non-zero when n_errored > 0, distinct stderr line"
      pattern: "n_errored"
---

<objective>
Thread the EVAL-01 error-status semantics through the matrix aggregation layer (EVAL-01,
second half). Plan 10-01 made each per-run JSON carry `status`, `n_scored`, `n_errored`, and
`errors`. This plan makes `aggregate_cell_jsons` surface those counts per cell in
`summary.json`, adds a top-level `errors` array and per-cell `cell_valid` flag (D-10-03), makes
`eval_matrix.py` exit non-zero on any errored cell with the error count printed distinctly from
the score-violation count, and extends the no-subprocess `--structural-check` mode to validate
the new error-schema fields (PATTERNS.md: copy the existing five-check structural pattern).

Purpose: a matrix operator (and Phase 11's baseline writer) must see at a glance that a cell is
INVALID_FOR_BASELINE because runs errored — not read a phantom 1.0 median.
Output: summary.json that distinguishes scored from errored runs; an error-aware matrix exit code.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/10-eval-harness-honesty/10-CONTEXT.md
@.planning/phases/10-eval-harness-honesty/10-PATTERNS.md
@.planning/phases/10-eval-harness-honesty/10-01-SUMMARY.md
@scripts/eval_matrix.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Thread n_scored / n_errored / errors through aggregate_cell_jsons + error-aware exit</name>
  <files>scripts/eval_matrix.py</files>
  <read_first>
    - scripts/eval_matrix.py (read _scorer_means_from_cell :158-190; read _stats_for_values :193-205; read aggregate_cell_jsons :208-260 — the per-cell payload walk; read run_matrix :360-441 — the failures list + rc computation; read main :536-684 — summary write at :668-684 and the return rc)
    - .planning/phases/10-eval-harness-honesty/10-01-SUMMARY.md (the exact per-run JSON shape 10-01 wrote: status, aggregate.n_scored, aggregate.n_errored, aggregate.errors)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-03 cell validity; matrix exit code distinguishes error count from violation count)
    - scripts/check_baselines_fresh.py (exit-code convention to imitate: 0 clean / 1 violation / 2 infra)
  </read_first>
  <behavior>
    - aggregate_cell_jsons reads each cell JSON's aggregate.n_scored and aggregate.n_errored and writes them into the per-provider block of summary.json, plus a cell_valid bool (n_errored == 0).
    - The top-level summary dict gains an errors array collecting each errored cell's {cell, stage, type} entries.
    - main() returns non-zero when any cell has n_errored > 0; stderr prints the error count on a line distinct from the failures/violations line.
  </behavior>
  <action>
    Extend `aggregate_cell_jsons` so that for each cell JSON it reads `aggregate.n_scored`, `aggregate.n_errored`, and `aggregate.errors` (written by 10-01) and surfaces them in the per-provider-key output block alongside the existing `scorers` sub-dict: add `n_scored` (int), `n_errored` (int), and `cell_valid` (bool, `n_errored == 0`) per D-10-03. Accumulate a top-level `summary["errors"]` list of `{"cell": <cell_filename>, "stage": <stage>, "type": <type>}` entries drawn from every errored cell (Claude's Discretion: exact error-array shape per CONTEXT). Cells missing the new fields (legacy JSON) default to n_errored=0, n_scored=n (backward-compatible). In `main`, after writing summary.json, compute `total_errored = sum of n_errored across cells` and make the return code non-zero when `total_errored > 0` OR `failures` non-empty; print TWO distinct stderr lines: one for subprocess `failures` (existing) and one new line `eval_matrix: <N> cell(s) had errored runs (INVALID_FOR_BASELINE)`. Keep the existing rc=2 gate paths (APP_ENV, bad --runs) unchanged.
  </action>
  <verify>
    <automated>poetry run python -c "from scripts.eval_matrix import aggregate_cell_jsons; import inspect; assert 'n_errored' in inspect.getsource(aggregate_cell_jsons)"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "n_errored" scripts/eval_matrix.py` shows it read inside `aggregate_cell_jsons` and used in the exit-code computation in `main`.
    - `grep -n "cell_valid" scripts/eval_matrix.py` returns at least one match.
    - The top-level summary dict produced by `aggregate_cell_jsons` over a directory containing one errored cell JSON contains a non-empty `errors` list.
    - `poetry run ruff check scripts/eval_matrix.py` passes.
  </acceptance_criteria>
  <done>summary.json surfaces per-cell n_scored/n_errored/cell_valid and a top-level errors array; the matrix exit code is error-aware with distinct stderr accounting.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Extend --structural-check with error-schema validation + aggregation tests</name>
  <files>scripts/eval_matrix.py, tests/unit/test_eval_matrix.py</files>
  <read_first>
    - scripts/eval_matrix.py (read the `if args.structural_check:` block :547-637 — the five existing checks; add Check 6 in the same style, stderr + return 1 on failure, return 0 with an OK line on success)
    - tests/unit/test_eval_matrix.py (read existing aggregation tests and the structural-check test to mirror style; read _DEFERRED_BASELINE_CELLS at :101-104)
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (structural-check extension pattern — synthetic_error_cell asserting status/error/stage membership)
  </read_first>
  <behavior>
    - --structural-check validates that the error-schema contract is well-formed: a synthetic error cell with {status:"error", error:{stage,type,message}} passes the membership checks (stage in {setup,turn0,turnN}); a malformed one fails with a descriptive stderr line and return 1.
    - A unit test builds a temp dir with one OK cell JSON and one errored cell JSON, runs aggregate_cell_jsons, and asserts the OK cell's scorer means are present, the errored cell has cell_valid=false, and summary["errors"] is non-empty.
    - A unit test runs the matrix in structural-check mode and asserts exit 0 with the new error-schema check present.
  </behavior>
  <action>
    Add "Check 6" inside the existing `if args.structural_check:` block (after Check 5), following the established pattern exactly: construct a synthetic error cell dict `{"status": "error", "error": {"stage": "turn0", "type": "RateLimitError", "message": "quota"}}`, assert `status` and `error` keys present and `error["stage"] in {"setup","turn0","turnN"}`; on any failure print a descriptive stderr line and `return 1`; fold the success into the existing final OK print. In `tests/unit/test_eval_matrix.py`, add (a) `test_aggregate_cell_jsons_threads_error_counts` — write two cell JSONs to a tmp_path (one with aggregate.n_scored=N/n_errored=0, one with n_errored>=1 and an errors list), call `aggregate_cell_jsons`, assert per-cell n_scored/n_errored/cell_valid and top-level `errors` non-empty; (b) `test_structural_check_validates_error_schema` — invoke `scripts.eval_matrix.main(["--matrix-config", <refinement yaml path>, "--structural-check"])` and assert it returns 0. Mirror the no-subprocess, no-live-key constraint (structural-check never calls subprocess.run).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -q -k "error or structural or aggregate"</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/eval_matrix.py` structural-check block contains an assertion that `error["stage"] in {"setup","turn0","turnN"}` (or equivalent membership check) as a 6th check.
    - `tests/unit/test_eval_matrix.py` contains `test_aggregate_cell_jsons_threads_error_counts` asserting `cell_valid` is False for the errored cell and `errors` is non-empty.
    - `make eval-matrix-refinement-structural-check` exits 0 (existing CI hard gate still passes after the Check-6 addition).
    - `poetry run pytest tests/unit/test_eval_matrix.py -q` exits 0.
  </acceptance_criteria>
  <done>The CI structural-check gate now validates the error-schema shape with no live calls, and aggregation error-threading is unit-tested.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| per-cell JSON → aggregator | a cell JSON's status/n_errored fields drive baseline eligibility; a missing/forged field could mask an errored cell |
| matrix process → CI exit code | CI trusts the exit code to block a bad matrix |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-02-01 | Tampering | `aggregate_cell_jsons` cell-validity computation | mitigate | cell_valid derived from n_errored read directly from each cell JSON; structural-check Check 6 validates the schema shape so a malformed error block is caught in CI |
| T-10-02-02 | Repudiation | matrix exit code | mitigate | n_errored>0 forces non-zero exit with a distinct stderr line; an errored matrix cannot exit 0 and be mistaken for clean |
| T-10-02-03 | Denial of service | structural-check synthetic data | accept | Check 6 uses only in-memory synthetic dicts (no subprocess, no live calls); zero external attack surface |
| T-10-02-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py -q` exits 0.
- `make eval-matrix-refinement-structural-check` exits 0.
- `poetry run ruff check scripts/eval_matrix.py tests/unit/test_eval_matrix.py` passes.
</verification>

<success_criteria>
- summary.json distinguishes scored runs from errored runs per cell and at the top level (EVAL-01).
- The matrix exit code is non-zero when any cell errored, with the error count distinct from violations.
- The CI structural-check gate validates the new error-schema fields without live calls.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-02-SUMMARY.md` when done.
</output>
