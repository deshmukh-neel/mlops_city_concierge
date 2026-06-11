---
phase: 10-eval-harness-honesty
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/eval_agent.py
  - app/agent/critique/checks.py
  - tests/unit/test_eval_agent.py
autonomous: true
requirements: [EVAL-01]
must_haves:
  truths:
    - "A turn-0 exception produces a per-run record with status=error, never a scored 1.0"
    - "A turn-N (N>=1) exception produces a per-run record with status=error, stage=turnN"
    - "An errored run's QueryEvalResult is excluded from scorer aggregation (not scored 1.0/0.0)"
    - "Replaying the 2026-06-05T21-14-30Z conditions yields error records and zero scored cells"
    - "The Branch-1 abstain in refinement_minimal_edit still returns 1.0 only for completed non-refinement runs"
  artifacts:
    - path: "scripts/eval_agent.py"
      provides: "ERROR-status record path replacing partial_state scoring in both threading branches"
      contains: 'status'
    - path: "tests/unit/test_eval_agent.py"
      provides: "21-14-30Z replay acceptance test + per-branch error-path unit tests"
      contains: "RaisingChatModel"
  key_links:
    - from: "scripts/eval_agent.py::_run_prod_threading"
      to: "RunErrorRecord (status=error)"
      via: "except Exception clause returns error record, not query_result_from_state"
      pattern: 'status.*=.*error'
    - from: "scripts/eval_agent.py::aggregate_results"
      to: "n_scored / n_errored / errors[]"
      via: "errored results excluded from scorer means; counted separately"
      pattern: 'n_scored|n_errored'
---

<objective>
Close the three fail-open scorer paths in the eval runner (EVAL-01). Today, when a turn
raises (429 quota, temp-400, DB down), both `_run_prod_threading` and `_run_legacy_threading`
catch the exception, build a `partial_state`, and return it through `query_result_from_state`
— which runs the scorers on corrupted partial state. The `refinement_minimal_edit` Branch-1
abstain then returns 1.0 for any non-refinement-context state, so a fully-failed matrix reads
1.0 medians (proven: `eval_reports/2026-06-05T21-14-30Z/` — all cells errored, refinement mean
read 1.0).

This plan removes the partial-state scoring path (D-10-02), makes any turn exception produce an
ERROR-status record (D-10-01), excludes errored runs from scorer aggregation, and surfaces
per-run error counts. Scorers are NEVER reached on exception. The Branch-1 abstain in
`checks.py` is retained verbatim — it is legitimate for completed non-refinement runs; it
becomes unreachable from error paths because exceptions no longer reach the scorer.

Purpose: Phase 11's baseline regen must distinguish infra failure from model failure on the
first attempt. A baseline cut from a quota-429 matrix is worthless.
Output: An eval runner that writes `status: error` records and aggregates only scored runs.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/10-eval-harness-honesty/10-CONTEXT.md
@.planning/phases/10-eval-harness-honesty/10-PATTERNS.md
@scripts/eval_agent.py
@app/agent/critique/checks.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Define RunErrorRecord schema and error-record builder in eval_agent.py</name>
  <files>scripts/eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py (read the dataclass block at :84-157 — CheckResult, QueryEvalResult, EvalRunReport — to mirror style; read query_result_from_state at :514-546; read write_report at :1090-1096; read aggregate_results at :1038-1045)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-01 schema; D-10-02 removal mandate)
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (RunErrorRecord shape; error-record-on-exception section)
  </read_first>
  <behavior>
    - A RunErrorRecord with status="error" and error={stage, type, message} serializes to a dict with exactly keys {status, error}; error has keys {stage, type, message}.
    - A helper `make_error_record(case_id, stage, exc)` returns a RunErrorRecord with stage in {"setup","turn0","turnN"}, type == type(exc).__name__, message == str(exc)[:500].
    - QueryEvalResult gains a `status: str = "ok"` field (default "ok") so existing scored rows are status="ok" with no other change; existing tests that build QueryEvalResult without status still pass via the default.
  </behavior>
  <action>
    Add a `status: str = "ok"` field to the QueryEvalResult dataclass (default keeps all existing scored rows status="ok"; this is the discriminator the aggregator reads). Add a new dataclass `RunErrorRecord` with `status: str = "error"` and `error: dict[str, str]` (keys: stage, type, message). Add a module-level builder `make_error_record(case: EvalQuery, stage: str, exc: BaseException) -> QueryEvalResult` that returns a QueryEvalResult with status="error", a populated `error`-shaped diagnostic, and scored fields set to neutral/empty so serialization never crashes — the canonical schema is D-10-01: status="error", error={"stage": <stage>, "type": type(exc).__name__, "message": str(exc)[:500]}. Stage values are exactly "setup" (graph/LLM construction), "turn0", "turnN" (any turn index >= 1) per D-10-01 / CONTEXT specifics. Decide (Claude's Discretion per D-10-01) whether the error block rides on QueryEvalResult (add an `error: dict[str,str] | None = None` field) or a sibling type; the aggregator in Task 3 must be able to read both `status` and `error` from each result. Do NOT put fenced code or full state-scoring in the error path. The error record carries NO scored checks.
  </action>
  <verify>
    <automated>poetry run python -c "from scripts.eval_agent import QueryEvalResult, make_error_record; import dataclasses; assert 'status' in {f.name for f in dataclasses.fields(QueryEvalResult)}"</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/eval_agent.py` QueryEvalResult dataclass contains a field named `status` with default `"ok"`.
    - `scripts/eval_agent.py` defines `make_error_record` and it sets `status="error"`, `error["type"] == type(exc).__name__`, `error["message"] == str(exc)[:500]`, and `error["stage"]` in `{"setup","turn0","turnN"}`.
    - `poetry run python -c "from scripts.eval_agent import make_error_record"` exits 0.
    - `poetry run pytest tests/unit/test_eval_agent.py -q` exits 0 (existing tests unaffected by the default-valued field).
  </acceptance_criteria>
  <done>QueryEvalResult has a status discriminator; make_error_record builds D-10-01-shaped error rows; existing scored-row construction is unchanged.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Replace partial-state scoring with error records in both threading branches</name>
  <files>scripts/eval_agent.py, app/agent/critique/checks.py</files>
  <read_first>
    - scripts/eval_agent.py (read _run_prod_threading :734-927 including the except block at :839-863 that builds partial_state and returns query_result_from_state; read _run_legacy_threading :652-730 including its except block at :696-726; read evaluate_multi_turn_case :619-649)
    - app/agent/critique/checks.py (read refinement_minimal_edit Branch-1 abstain at :488-491 — RETAINED verbatim; only a clarifying comment may be added)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-02 removal; D-10-04 Branch-1 retained for completed non-refinement runs only)
  </read_first>
  <behavior>
    - When graph.ainvoke raises in _run_prod_threading at turn index 0, the function returns a QueryEvalResult with status="error", error.stage="turn0"; scorers (score_checks) are NOT invoked on the failed run.
    - When graph.ainvoke raises at turn index >= 1, the returned record has status="error", error.stage="turnN".
    - _run_legacy_threading exhibits identical behavior (turn0 / turnN error records, no partial-state scoring).
    - refinement_minimal_edit still returns 1.0 for a COMPLETED state whose scratch lacks refinement_context (Branch-1) — proven by an unchanged existing Branch-1 test.
  </behavior>
  <action>
    In `_run_prod_threading`, DELETE the `except Exception as exc:` block at :839-863 (the partial_state construction, scratch re-stamp, multi_turn_runner append, and `return query_result_from_state(...)`) per D-10-02. Replace it with an except clause that computes `total_latency` and returns `(make_error_record(case, stage, exc), <a non-scored state sentinel>)` — the function returns a tuple `(QueryEvalResult, ItineraryState)`, so return the error-status QueryEvalResult plus the last-known state (or a fresh ItineraryState) WITHOUT calling query_result_from_state on it. Compute `stage = "turn0" if index == 0 else "turnN"`. Apply the identical change in `_run_legacy_threading` (delete the partial_state/multi_turn_runner block at :696-726, return `make_error_record(case, stage, exc)`; this branch returns a bare QueryEvalResult). Wrap the graph construction / pre-loop setup such that a setup-stage exception (e.g. in `_constraints_for_case` or message assembly before the loop) produces stage="setup" — only if such a failure is reachable; otherwise document that setup errors surface at the build_report level (Task added behavior is turn-level). Add ONE clarifying comment above `checks.py:488-491` Branch-1 stating the invariant: score_checks is only reached on completed (status="ok") runs, so Branch-1 abstain fires only for genuinely-non-refinement completed runs (D-10-04). Do NOT change Branch-1/Branch-2 logic.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py tests/unit/test_critique_checks.py -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "partial_state" scripts/eval_agent.py` returns zero matches (the partial-state scoring path is fully removed).
    - `grep -n "make_error_record" scripts/eval_agent.py` shows it called inside the except clauses of both `_run_prod_threading` and `_run_legacy_threading`.
    - `app/agent/critique/checks.py` Branch-1 abstain (`return 1.0` after `if not refinement_context`) is byte-unchanged except for one added comment line referencing D-10-04.
    - `poetry run pytest tests/unit/test_critique_checks.py -q` exits 0 (Branch-1/Branch-2 scorer behavior preserved for completed runs).
  </acceptance_criteria>
  <done>Both threading branches return error records on exception; scorers never run on failed turns; the legitimate Branch-1 abstain is preserved and documented.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Aggregate scored-only + emit error counts; add the 21-14-30Z replay acceptance test</name>
  <files>scripts/eval_agent.py, tests/unit/test_eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py (read aggregate_results :1038-1045 — the scorer-mean loop; read evaluate_cases :930-944; read the aggregate dict assembly around :981-1037; read report_has_errors :1053-1055 and report_has_violations :1058-1060; read main exit-code :1099-1112)
    - tests/unit/test_eval_agent.py (read the eval_case() builder + parametrize patterns at :47-59 and :149-159 to mirror test style)
    - tests/_helpers/scripted_llm.py (the raising-on-exhaustion stub to base a RaisingChatModel on)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-03 cell validity n_scored/n_errored; D-10-04 acceptance = 21-14-30Z replay)
  </read_first>
  <behavior>
    - aggregate_results computes scorer means over ONLY results with status=="ok"; an all-error result list yields scorer means of 0.0 (or omitted), never 1.0, and n_scored==0.
    - The aggregate dict gains n_scored (count status=="ok"), n_errored (count status=="error"), and an errors list of {stage, type, message} per errored result.
    - The acceptance test simulates the three 21-14-30Z paths (turn-0 LLM exception, turn-1 LLM exception, retrieval-only/tool exception) with a RaisingChatModel and asserts: (a) turn-0/turn-1 LLM exceptions produce status="error" records, (b) the report's n_scored==0 for an all-error run, (c) the former fail-open outcomes (Branch-1 abstain 1.0 on partial state, prior-vs-itself 1.0) do NOT appear.
  </behavior>
  <action>
    In `aggregate_results`, filter `results` to scored-only (`[r for r in results if r.status == "ok"]`) before computing every `{name}_mean` so errored runs contribute nothing to scorer means (D-10-03). Add `aggregate["n_scored"]`, `aggregate["n_errored"]`, and `aggregate["errors"]` (list of each errored result's `error` dict). Keep `check_error_count` (individual-check exceptions on COMPLETED runs) distinct from `n_errored` (whole-run failures) — they are different concepts per PATTERNS.md. Add `cell_valid = (n_errored == 0)` semantics to the aggregate (D-10-03: a cell with any errored run is INVALID_FOR_BASELINE). Extend `report_has_errors` (or main's exit logic) so a report with `n_errored > 0` returns a non-zero exit distinct in stderr from a score violation. In `tests/unit/test_eval_agent.py`, add a `RaisingChatModel(BaseChatModel)` test stub (raises a configurable exception type on `_generate`) and three async tests using `evaluate_multi_turn_case` / `evaluate_case` against a graph built on RaisingChatModel: turn-0 raise, turn-1 raise (use a ScriptedChatModel-then-raise sequence), and the all-error report aggregation. Assert status=="error", stage values, n_scored==0, and that `refinement_minimal_edit` mean is NOT 1.0 on the all-error run (the former fail-open). Use `asyncio_mode = "auto"` (already set in pyproject.toml — no asyncio.run boilerplate needed).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -q -k "error or raising or replay or status"</automated>
  </verify>
  <acceptance_criteria>
    - `aggregate_results` filters on `status == "ok"` before computing scorer means (source assertion: `grep -n 'status == "ok"' scripts/eval_agent.py` returns at least one match inside aggregate_results).
    - The aggregate dict produced by an all-error run contains `n_scored == 0`, `n_errored >= 1`, and a non-empty `errors` list.
    - The new acceptance test asserts the all-error report's `refinement_minimal_edit_mean` is not 1.0, and asserts at least one record has `status == "error"` with `error["stage"] in {"turn0","turnN"}`.
    - `poetry run pytest tests/unit/test_eval_agent.py -q` exits 0.
  </acceptance_criteria>
  <done>Aggregation scores only completed runs and reports error counts; the 21-14-30Z fail-open is proven gone by a cheap stub-driven replay test (no live calls).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| provider LLM → eval runner | provider responses / exceptions cross into scoring; a swallowed exception became a false 1.0 (the bug this plan fixes) |
| eval runner → per-run JSON on disk | report content is written to `eval_reports/`; downstream baseline tooling trusts it |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-01-01 | Repudiation | `_run_prod_threading` / `_run_legacy_threading` exception handling | mitigate | Every turn exception writes an auditable status=error record with stage+type+message (D-10-01); no silent swallow |
| T-10-01-02 | Tampering | `aggregate_results` scorer means | mitigate | Errored runs excluded from means (status=="ok" filter); a quota-429 matrix can no longer masquerade as a 1.0 baseline |
| T-10-01-03 | Information disclosure | `error["message"] = str(exc)[:500]` | accept | Exception messages may include partial config but never raw keys (provider SDKs do not echo keys in exception strings); truncated to 500 chars; eval_reports/ is not published. Re-evaluated in 10-04 redaction work for fixtures (different surface) |
| T-10-01-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs in this plan; no install tasks present |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_agent.py tests/unit/test_critique_checks.py -q` exits 0.
- `grep -c "partial_state" scripts/eval_agent.py` returns 0.
- `poetry run ruff check scripts/eval_agent.py tests/unit/test_eval_agent.py` passes.
</verification>

<success_criteria>
- A turn-0 or turn-1 exception yields a status=error record excluded from scoring (EVAL-01).
- The 21-14-30Z replay test proves the three fail-open paths (Branch-1 abstain 1.0, prior-vs-itself 1.0, retrieval-0.0 asymmetry) no longer produce phantom scores.
- The legitimate Branch-1 abstain is retained verbatim for completed non-refinement runs.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-01-SUMMARY.md` when done.
</output>
