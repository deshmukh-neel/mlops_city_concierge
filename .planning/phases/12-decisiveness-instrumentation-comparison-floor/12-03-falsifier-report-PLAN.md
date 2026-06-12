---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/eval_falsifier.py
  - tests/unit/test_eval_falsifier.py
  - Makefile
autonomous: true
requirements: [INST-05]
must_haves:
  truths:
    - "make eval-falsifier prints one line per model with the numbers: did gpt-5-mini hit >=0.6 pooled committed_itinerary_rate, did gpt-4o-mini hold >= its honest baseline"
    - "The falsifier reads a completed eval_reports run dir (latest by default, overridable) and does NOT fan out any live API call"
    - "The gpt-5-mini commit-rate bar is pooled across all scored scenario cells in configs/eval_matrix.yaml, with a per-scenario breakdown also printed"
    - "Exit code is 0 on PASS, 1 on expected falsifier FAIL, 2 on infrastructure failure"
    - "A committed smoke test exercises the script against the real configs/eval_baselines and asserts a real verdict exit code (0 or 1), not just synthetic in-memory summaries"
  artifacts:
    - path: "scripts/eval_falsifier.py"
      provides: "artifact-reading falsifier report with pooled gpt-5-mini rate + anchor non-regression"
      exports: ["main"]
    - path: "tests/unit/test_eval_falsifier.py"
      provides: "unit tests over synthetic summaries (PASS/FAIL/infra) + a smoke test against real configs/eval_baselines via --baselines-mode"
      contains: "_load_script"
    - path: "Makefile"
      provides: "eval-falsifier target + RUN_DIR var"
      contains: "eval-falsifier"
  key_links:
    - from: "scripts/eval_falsifier.py"
      to: "scripts.check_eval_gates baseline machinery"
      via: "import _build_summary_from_baselines, _get_metric_value"
      pattern: "from scripts.check_eval_gates import"
    - from: "Makefile eval-falsifier"
      to: "scripts/eval_falsifier.py"
      via: "poetry run python scripts/eval_falsifier.py"
      pattern: "eval_falsifier.py"
---

<objective>
Make the milestone falsifier (INST-05) executable as a single report. `make
eval-falsifier` reads a completed eval_reports run directory (latest by default,
`RUN_DIR=` to override) and answers two questions with explicit per-model numbers and a
PASS/FAIL verdict:

1. Did `openai/gpt-5-mini` hit `committed_itinerary_rate >= 0.6` POOLED across all scored
   scenario cells in `configs/eval_matrix.yaml` (D-12-08)? A per-scenario breakdown is
   always printed (gpt-5-mini splits sharply by scenario — omakase ~1.0 vs refinement
   0.0 — so an omakase-only bar would be vacuous; pooling keeps it meaningful at ~0.5
   today).
2. Did `openai/gpt-4o-mini` hold `>= its honest baseline` floor (no anchor regression),
   checked per-metric against `configs/eval_baselines/` using the existing gate machinery?

Per D-12-06 the report reads existing artifacts only — it NEVER fans out live API calls
(workflow stays two-step: `make eval-matrix` is the expensive live run, `make
eval-falsifier` is the cheap repeatable report). Per D-12-07 it reuses
`eval_matrix.py`/`check_eval_gates.py` machinery rather than duplicating either. Exit
codes 0/1/2 let Phase 13 consume it mechanically.

Purpose: the single objective falsifier every Phase 13 arm is judged against.
Output: `scripts/eval_falsifier.py`, `make eval-falsifier` target, unit + smoke tests.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create scripts/eval_falsifier.py (artifact-reading, pooled rate, anchor non-regression)</name>
  <files>scripts/eval_falsifier.py</files>
  <read_first>
    - scripts/check_eval_gates.py (the shebang/docstring/import shape at the top; `_parse_args`
      at line 392; `main` at line 441; `_build_summary_from_baselines` at lines 131-204
      [synthesises a summary-shaped dict from configs/eval_baselines/ JSONs];
      `_get_metric_value` at lines 224-... [reads `cell["scorers"][metric]["median"]`, falls
      back to mean]; `_load_baseline_eligibility` at line 99; the 0/1/2 exit-code contract
      in main())
    - scripts/eval_matrix.py (`_DEFAULT_OUTPUT_BASE` at line 62 = REPO_ROOT/"eval_reports";
      `aggregate_cell_jsons` at line 212 and its documented output shape `scenarios.<id>.
      providers.<provider/model>.scorers.<scorer>.{median,n,min,max,stdev}` plus per-scenario
      `baseline_eligible`; the D-11-02 thread at lines 295-305 that puts committed_itinerary_rate
      under scorers; `_parse_cell_filename` at line 142)
    - configs/eval_matrix.yaml (the scored provider cells: openai/gpt-4o-mini,
      deepseek/deepseek-chat, openai/gpt-5-mini, anthropic/claude-sonnet-4-6,
      deepseek/deepseek-reasoner — note gemini is NOT in this file; the falsifier targets
      gpt-5-mini and gpt-4o-mini, both present)
    - configs/eval_gates.yaml (gpt-5-mini family carries the 0.6 committed_itinerary_rate
      value as aspirational; gpt-4o-mini family carries the 0.8 anchor floor — reference
      INST-05/D-12-08 for the 0.6 bar, do not duplicate gate YAML semantics)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (section "scripts/eval_falsifier.py (new)" — the docstring, _latest_run_dir,
      _pooled_commit_rate, argparse, and main() patterns to copy)
  </read_first>
  <action>
    Create `scripts/eval_falsifier.py` modeled on `check_eval_gates.py`. Top: shebang,
    docstring citing INST-05 / D-12-06..08 and the exit-code contract, `from __future__
    import annotations`, stdlib imports (argparse, json, sys, pathlib, typing). NO top-level
    LLM-SDK imports (artifact-reading only, D-12-06). Reuse machinery via
    `from scripts.check_eval_gates import _build_summary_from_baselines,
    _load_baseline_eligibility, _get_metric_value`. Define `REPO_ROOT =
    Path(__file__).resolve().parents[1]` and `_DEFAULT_OUTPUT_BASE = REPO_ROOT /
    "eval_reports"`. Implement `_latest_run_dir(base) -> Path` (most recent subdir by ISO8601
    name sort; raise OSError if none). Implement `_pooled_commit_rate(summary,
    provider_key) -> tuple[float | None, dict[str, float | None]]`: iterate
    `summary["scenarios"]`; skip any scenario whose `baseline_eligible` is explicitly False;
    for the provider cell read `cell["scorers"]["committed_itinerary_rate"]` and use its
    `median` (guard None/bool/non-numeric) weighted by `n` to compute a pooled rate
    `sum(median*n)/sum(n)`; record per-scenario median in the breakdown dict (None when the
    cell or metric is absent); return (pooled_or_None, per_scenario). Define `_FALSIFIER_BAR
    = 0.6` and `_GPT5_KEY = "openai/gpt-5-mini"`, `_ANCHOR_KEY = "openai/gpt-4o-mini"`,
    `_ANCHOR_METRIC = "committed_itinerary_rate"`. Implement `main(argv) -> int`: parse
    `--run-dir` (default latest under _DEFAULT_OUTPUT_BASE), `--baselines-mode` (read
    configs/eval_baselines via `_build_summary_from_baselines` instead of a run dir —
    enables live-key-free CI like `eval-gates-check-baselines`), `--baselines-dir` (default
    configs/eval_baselines), `--eval-queries` (default configs/eval_queries.yaml, used by
    `_load_baseline_eligibility`). Resolve the summary dict (from `run_dir/summary.json` in
    run-dir mode, or from baselines in baselines-mode), routing missing/malformed to a
    `return 2` via the `(OSError, ValueError)` handler. Compute the gpt-5-mini pooled rate
    and compare to `_FALSIFIER_BAR`; compute the anchor floor from
    `_build_summary_from_baselines(Path(args.baselines_dir))` (or read the gate value) and
    compare the run's pooled/per-scenario anchor committed_itinerary_rate against it. Print
    one line per model with the actual numbers (e.g. "gpt-5-mini: pooled
    committed_itinerary_rate = 0.52 < 0.6  FAIL" and the per-scenario breakdown; "gpt-4o-mini:
    pooled = 1.00 >= baseline 0.80  PASS"), then a final verdict line. Set `failed=True`
    on any failed check; `return 1 if failed else 0`. Guard with `if __name__ ==
    "__main__": raise SystemExit(main())`.
  </action>
  <verify>
    <automated>poetry run python scripts/eval_falsifier.py --baselines-mode --baselines-dir configs/eval_baselines; echo "exit=$?"</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/eval_falsifier.py` exists and `poetry run python -c "import importlib.util,sys; spec=importlib.util.spec_from_file_location('ef','scripts/eval_falsifier.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); assert hasattr(m,'main') and hasattr(m,'_pooled_commit_rate') and hasattr(m,'_latest_run_dir')"` succeeds
    - `grep -n "from scripts.check_eval_gates import" scripts/eval_falsifier.py` shows reuse of _build_summary_from_baselines / _get_metric_value (no reimplementation, D-12-07)
    - `grep -c "import openai\|import anthropic\|from langchain\|build_chat_model" scripts/eval_falsifier.py` is 0 (no live-SDK imports, D-12-06)
    - Running with a nonexistent --run-dir returns exit code 2 (infra failure), verified by `poetry run python scripts/eval_falsifier.py --run-dir /nonexistent/path; test $? -eq 2`
    - `grep -n "0.6\|_FALSIFIER_BAR" scripts/eval_falsifier.py` shows the 0.6 pooled bar is present and the per-scenario breakdown is computed
    - Running `--baselines-mode` produces per-model lines and a PASS/FAIL verdict with exit code 0 or 1 (not 2)
  </acceptance_criteria>
  <done>eval_falsifier.py reads artifacts only, pools gpt-5-mini commit rate across scored cells, checks anchor non-regression, prints per-model numbers, and returns 0/1/2.</done>
</task>

<task type="auto">
  <name>Task 2: Unit + smoke tests for the falsifier (synthetic PASS/FAIL/infra + real-baselines smoke + --baselines-mode path)</name>
  <files>tests/unit/test_eval_falsifier.py</files>
  <read_first>
    - tests/unit/test_check_eval_gates.py (lines 1-43: the `_load_script()` /
      `importlib.util.spec_from_file_location` module-loading pattern; lines ~50-68: the
      synthetic summary-dict fixtures built in-test — never touch the filesystem or live
      keys; if there is an existing test calling `main([...])` against real configs/baselines
      live-key-free, mirror that smoke pattern)
    - scripts/eval_falsifier.py (the `_pooled_commit_rate` semantics and `main()` exit-code
      contract from Task 1, including the `--baselines-mode` / `--baselines-dir` flags, to
      write fixtures that produce known pooled rates and to exercise the baselines path)
    - configs/eval_baselines/ (the real committed baseline JSONs the smoke test reads via
      --baselines-mode — confirm the directory exists and is checked in)
    - .planning/STATE.md (project memory `feedback_test_layering`: new modules need
      unit/mock + smoke + functional + integration coverage, not just unit — the smoke test
      below is the committed smoke artifact for this script)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (section "tests/unit/test_eval_falsifier.py (new)" — the module-loading pattern and
      synthetic-summary fixture approach)
  </read_first>
  <action>
    Create `tests/unit/test_eval_falsifier.py` using the `_load_script()` +
    `spec_from_file_location("eval_falsifier", SCRIPT_PATH)` pattern (REPO_ROOT =
    parents[2], SCRIPT_PATH = scripts/eval_falsifier.py) and a `script` fixture returning the
    loaded module. (A) UNIT tests over synthetic summary dicts matching the
    `scenarios.<id>.providers.<provider>.scorers.committed_itinerary_rate.{median,n}` shape:
    `_pooled_commit_rate` two scenarios with gpt-5-mini medians [1.0 (n=5), 0.0 (n=5)] pool
    to 0.5 and the per-scenario breakdown reports both; a scenario with
    `baseline_eligible: False` is excluded from the pool; an absent provider cell yields a
    None per-scenario entry and is excluded from the pooled denominator; a summary with no
    eligible scenarios returns (None, ...). Bar-logic: pooled 0.5 fails the 0.6 bar; pooled
    0.8 passes. Drive `main()` against a tmp_path run dir containing a synthetic
    `summary.json`: exit code 1 when gpt-5-mini < 0.6, 0 when above and anchor holds, 2 when
    summary.json is missing. (B) --baselines-mode UNIT path: build a synthetic baselines
    directory under tmp_path (the minimal JSON shape `_build_summary_from_baselines` expects)
    and call `main(["--baselines-mode", "--baselines-dir", str(tmp_dir)])`, asserting exit
    code is 0 or 1 (a real verdict, not 2) — this exercises the baselines code path the
    run-dir tests skip. (C) SMOKE test (committed test artifact, satisfies
    feedback_test_layering): call
    `main(["--baselines-mode", "--baselines-dir", "configs/eval_baselines"])` against the
    REAL checked-in baselines directory and assert the return code is in `{0, 1}` (NOT 2) —
    i.e. the script produces a real verdict against real artifacts, no infra failure. Mark
    it clearly (name like `test_smoke_runs_against_real_baselines`) and use the repo-root
    relative path so it runs from any cwd. Do NOT call any live API or read real
    eval_reports run dirs (artifact-reading, live-key-free).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_falsifier.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `tests/unit/test_eval_falsifier.py` loads the script via importlib (no sys.path mutation, matches test_check_eval_gates pattern)
    - A test asserts two-scenario medians [1.0, 0.0] at equal n pool to 0.5
    - A test asserts a `baseline_eligible: False` scenario is excluded from the pool
    - A test asserts pooled 0.5 < 0.6 produces a FAIL path and pooled 0.8 produces a PASS path
    - A test asserts `main()` returns 2 when summary.json is missing from the run dir
    - A `--baselines-mode` unit test against a synthetic baselines dir asserts exit code 0 or 1 (not 2), covering the baselines code path
    - A committed SMOKE test calls `main(["--baselines-mode", "--baselines-dir", "configs/eval_baselines"])` against the real baselines and asserts the exit code is in {0, 1} (not 2)
    - `poetry run pytest tests/unit/test_eval_falsifier.py -q` passes
  </acceptance_criteria>
  <done>Unit tests cover pooling math, baseline_eligible exclusion, the 0.6 bar, the 0/1/2 exit codes, and the --baselines-mode path; a committed smoke test exercises the script against real configs/eval_baselines (feedback_test_layering satisfied).</done>
</task>

<task type="auto">
  <name>Task 3: Add the eval-falsifier Makefile target + RUN_DIR var</name>
  <files>Makefile</files>
  <read_first>
    - Makefile (the Variables block at lines 4-6: `POETRY_RUN := poetry run`; the eval
      targets block — `eval-gates-check` ~line 171 with `$(SUMMARY)`, `eval-gates-check-baselines`
      ~line 182, `write-baselines` ~line 191; the `## help` doc-comment convention used by
      the `help` target at lines 8-11)
    - scripts/eval_falsifier.py (its argparse flags from Task 1: --run-dir, --baselines-mode,
      --baselines-dir, --eval-queries — so the target passes the right flags)
  </read_first>
  <action>
    Add a `RUN_DIR ?=` variable in the Variables block (near POETRY_RUN), defaulting empty
    so the script picks the latest run dir. Add a `.PHONY: eval-falsifier` target after
    `write-baselines` with a `## INST-05: ...` help comment. The recipe runs
    `$(POETRY_RUN) python scripts/eval_falsifier.py` passing
    `$(if $(RUN_DIR),--run-dir $(RUN_DIR),)` and `--baselines-dir configs/eval_baselines`.
    Mirror the existing eval targets' formatting (tab-indented recipe, backslash line
    continuations). Do NOT change any existing target.
  </action>
  <verify>
    <automated>make -n eval-falsifier && grep -n "eval-falsifier" Makefile</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "eval-falsifier:" Makefile` matches a `.PHONY`-declared target with a `## INST-05` help comment
    - `grep -n "RUN_DIR" Makefile` shows the variable declared and used via `$(if $(RUN_DIR),--run-dir $(RUN_DIR),)`
    - `make -n eval-falsifier` prints a `poetry run python scripts/eval_falsifier.py ...` command (dry-run, no execution error)
    - `make help` lists `eval-falsifier` with its description (the `## ...` comment is picked up)
    - No existing Makefile target is modified (diff touches only the new var line and the new target block)
  </acceptance_criteria>
  <done>make eval-falsifier (with optional RUN_DIR=) invokes the falsifier report; help lists it.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| falsifier ← eval_reports artifacts | reads JSON written by a prior run; malformed/missing input must fail closed to exit 2, never green |
| Phase 13 ← falsifier exit code | downstream consumes 0/1/2 mechanically; a wrong exit code silently mis-grades an arm |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-07 | Spoofing | live API call masquerading as a report | mitigate | no live-SDK imports (Task 1 acceptance gate: grep count 0); D-12-06 artifact-reading only |
| T-12-08 | Tampering | empty/missing input read as PASS | mitigate | reuse _build_summary_from_baselines fail-closed contract; missing run dir/summary returns exit 2 (Task 1 + Task 2 assert it); smoke test confirms a real verdict against real baselines |
| T-12-09 | Repudiation | vacuous omakase-only bar | mitigate | D-12-08 pooled scope across all scored cells with per-scenario breakdown printed; unit test pins the pooling math |
| T-12-10 | Elevation of privilege | wrong exit code mis-grades arm | mitigate | three-level 0/1/2 contract unit-tested (incl. --baselines-mode path); (OSError, ValueError) always routes to 2 |
| T-12-SC | Tampering | npm/pip/cargo installs | mitigate | no new package installs (stdlib + in-repo imports); no slopcheck needed |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_falsifier.py -q` passes
- `make lint` passes (ruff E,F,I,N,UP,B,SIM per CLAUDE.md) on scripts/eval_falsifier.py and tests/unit/test_eval_falsifier.py
- `make -n eval-falsifier` prints the expected command; `make help` lists the target
- `poetry run python scripts/eval_falsifier.py --baselines-mode --baselines-dir configs/eval_baselines` exits 0 or 1 (a real verdict, not an infra error) and prints per-model numbers
- The committed smoke test (`--baselines-mode` against real configs/eval_baselines) returns an exit code in {0, 1}
- `poetry run python scripts/eval_falsifier.py --run-dir /nonexistent` exits 2
- `grep -c "import openai\|from langchain\|build_chat_model" scripts/eval_falsifier.py` is 0
</verification>

<success_criteria>
- INST-05 satisfied: a single `make eval-falsifier` answers the milestone question with per-model numbers and a PASS/FAIL verdict, exit code 0/1/2
- gpt-5-mini bar is pooled across all scored scenario cells (D-12-08), per-scenario breakdown printed; anchor non-regression checked per-metric against committed baselines
- Report reads artifacts only — zero live API calls — and reuses existing machinery (D-12-06/07)
- Test layering satisfied: unit (synthetic) + --baselines-mode unit path + a committed smoke against real configs/eval_baselines (feedback_test_layering)
</success_criteria>

<output>
Create `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-03-SUMMARY.md` when done
</output>
</content>
