---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: 05
type: execute
wave: 1
depends_on: []
gap_closure: true
files_modified:
  - scripts/eval_falsifier.py
  - tests/unit/test_eval_falsifier.py
autonomous: true
requirements: [INST-05]
must_haves:
  truths:
    - "In run-dir mode, when the resolved summary shares ZERO scenarios with configs/eval_matrix.yaml's declared scenarios (and the matrix config was readable / non-empty), eval_falsifier prints the existing wrong-matrix diagnosis and returns exit 2 — it does NOT print any VERDICT line and does NOT emit a PASS/FAIL"
    - "The exit-2 refusal fires BEFORE the gpt-5-mini and anchor checks run, so a wrong-matrix run can never produce a spurious milestone PASS nor a spurious FAIL for scripted/CI consumers that read only the exit code"
    - "--baselines-mode is unaffected: it still reads scenarios from the committed baselines themselves and produces a real verdict (exit 0/1)"
    - "When the matrix config is unreadable/empty (expected_scenarios is empty), the guard does NOT fire — a best-effort empty expected set must not turn every run into an infra refusal"
  artifacts:
    - path: "scripts/eval_falsifier.py"
      provides: "zero-overlap guard in run-dir mode that escalates the WR-06 diagnosis from warn-and-continue to refuse-with-exit-2"
      contains: "return 2"
    - path: "tests/unit/test_eval_falsifier.py"
      provides: "unit test asserting zero-overlap run-dir summary -> exit 2 with no VERDICT line; plus a regression test that an in-matrix run still grades normally"
      contains: "VERDICT"
  key_links:
    - from: "scripts/eval_falsifier.py zero-overlap guard"
      to: "_expected_matrix_scenarios() / summary['scenarios'] intersection"
      via: "reuse the existing found_scenarios & expected_scenarios computation, escalate to return 2"
      pattern: "found_scenarios & expected_scenarios"
    - from: "tests/unit/test_eval_falsifier.py"
      to: "scripts/eval_falsifier.py main()"
      via: "main(['--run-dir', <refinement-only summary>]) asserts rc == 2 and 'VERDICT' not in out"
      pattern: "assert rc == 2"
---

<objective>
Close the single UAT gap (12-HUMAN-UAT.md, Test 1, severity major): in run-dir
mode the WR-06 wrong-matrix detection currently WARNS and then continues to emit a
PASS/FAIL verdict and exit 0/1. Scripted/CI consumers (Phase 13 arm judging) read
only the exit code, so a wrong-matrix run produces a spurious FAIL today and could
produce a spurious milestone PASS if a refinement run happens to clear the 0.6 bar.

Per the script's own documented 0/1/2 exit-code contract, zero scenario overlap with
configs/eval_matrix.yaml is an infrastructure/usage error, not a falsifier FAIL.
Escalate the existing diagnostic from warn-and-continue to refuse-with-exit-2: print
the same wrong-matrix message (it already names the mismatch and suggests --run-dir),
then `return 2` BEFORE any verdict is computed or printed.

Scope is narrow and surgical:
- Applies to run-dir mode only (auto-resolved latest OR explicit --run-dir).
- --baselines-mode is untouched (it derives scenarios from the baselines themselves).
- The guard only fires when expected_scenarios is non-empty (matrix config readable)
  AND the intersection with the summary's scenarios is empty. An unreadable/empty
  matrix config yields an empty expected set; the guard must NOT fire then (best-effort).
- Keep the existing diagnostic text verbatim (do not reword the message).
- Do NOT remove the downstream "no scenario overlap -> warn and pass" anchor branch
  (lines ~370-374) — it becomes unreachable in the wrong-matrix case but stays as a
  defensive branch for honest partial-overlap runs.

Purpose: make the falsifier fail closed so Phase 13 can consume its exit code mechanically.
Output: a zero-overlap guard in scripts/eval_falsifier.py + unit tests pinning exit 2 / no VERDICT.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-HUMAN-UAT.md
@scripts/eval_falsifier.py
@tests/unit/test_eval_falsifier.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Escalate run-dir zero-overlap from warn-and-continue to refuse-with-exit-2</name>
  <files>scripts/eval_falsifier.py</files>
  <read_first>
    - scripts/eval_falsifier.py — focus on three regions, all already in context:
        • Module docstring (lines 1-23): the 0/1/2 exit-code contract that names
          exit 2 = "infrastructure failure (missing run dir, malformed JSON)". The
          docstring should be extended to also name zero-overlap-with-eval_matrix.yaml
          as an exit-2 case so the contract documents the new refusal.
        • The run-dir-mode WR-06 warning block (lines 264-278): in the `else` branch
          (NOT baselines-mode) it computes
          `expected_scenarios = _expected_matrix_scenarios()`,
          `found_scenarios = set(scenarios_block) if isinstance(...) else set()`,
          then `if expected_scenarios and not (found_scenarios & expected_scenarios):`
          prints the WARNING block. THIS is the exact predicate to reuse. Today the
          block prints and falls through to the gpt-5-mini check at line 281+.
        • The gpt-5-mini check (lines 281-301) and final verdict (lines 387-396) that
          must NOT be reached on a zero-overlap run.
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-HUMAN-UAT.md
      (the Gaps block: truth, reason, artifacts, missing[] — the executable contract
      for this fix).
  </read_first>
  <action>
    In `main()` in scripts/eval_falsifier.py, inside the existing run-dir-mode `else`
    branch (the one that prints `source: run dir {run_dir}` and computes
    `expected_scenarios` / `found_scenarios`), change the zero-overlap predicate from
    "print a WARNING and continue" to "print the SAME diagnosis, then `return 2`".

    Concretely: keep the predicate `if expected_scenarios and not (found_scenarios &
    expected_scenarios):` and keep the existing diagnostic print verbatim (the message
    that names configs/eval_matrix.yaml's expected scenarios, the found scenarios, the
    "may belong to a different matrix (e.g. eval_matrix_refinement.yaml)" hint, and the
    "pass --run-dir explicitly if so" suggestion). Immediately after that print,
    escalate: write a one-line refusal to stderr (e.g. "eval_falsifier: refusing to
    grade — resolved run dir shares zero scenarios with configs/eval_matrix.yaml "
    "(wrong-matrix run); exit 2, no verdict.") and `return 2`. This MUST execute before
    the gpt-5-mini pooled-rate check, so no VERDICT line and no PASS/FAIL is ever
    printed for a wrong-matrix run.

    Guard scoping (do NOT widen): the refusal fires ONLY when `expected_scenarios` is
    truthy (the matrix config was readable and non-empty) AND the intersection is empty.
    When `_expected_matrix_scenarios()` returns an empty set (unreadable/missing config),
    `expected_scenarios` is falsy, the predicate is False, and grading proceeds exactly
    as before — preserve that behavior (the helper is best-effort by design, lines
    69-88). Do NOT touch the --baselines-mode branch (lines 264-265): baselines-mode
    reads scenarios from the baselines and is out of scope.

    Update the module docstring exit-code contract (lines 19-23) so the `2 =` line also
    names this case, e.g. add "or a run-dir summary that shares zero scenarios with
    configs/eval_matrix.yaml (wrong-matrix run)". Do NOT remove or alter the downstream
    anchor-side "no scenario overlap -> warn and pass" branch (lines ~370-374): it is
    now unreachable in the wrong-matrix case but remains a defensive branch for honest
    partial-overlap runs (per UAT constraint). Keep the change additive — do not
    reorder or delete the existing WARNING text; only escalate what happens after it.
  </action>
  <verify>
    <automated>poetry run python -c "import importlib.util; s=importlib.util.spec_from_file_location('ef','scripts/eval_falsifier.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)" && grep -n "return 2" scripts/eval_falsifier.py | grep -v '^[0-9]*:#'</automated>
  </verify>
  <acceptance_criteria>
    - The run-dir zero-overlap predicate (`if expected_scenarios and not (found_scenarios & expected_scenarios):`) now ends in `return 2` after printing the existing diagnostic — verified by `grep -n "found_scenarios & expected_scenarios" scripts/eval_falsifier.py` followed by a `return 2` within that block
    - The escalation runs BEFORE the gpt-5-mini check: the `return 2` is positioned above the `_pooled_commit_rate(summary, _GPT5_KEY)` call for the first check (line ~256/281 region)
    - The existing WARNING diagnostic text is preserved verbatim (the "may belong to a different matrix (e.g. eval_matrix_refinement.yaml)" and "pass --run-dir explicitly" strings still present): `grep -c "eval_matrix_refinement" scripts/eval_falsifier.py` >= 1
    - The guard is scoped to `expected_scenarios` being truthy — an empty expected set does NOT trigger exit 2 (the `if expected_scenarios and ...` short-circuit is intact)
    - The --baselines-mode branch is unchanged: `grep -n "baselines-mode" scripts/eval_falsifier.py` shows the source line still printing committed baselines with no early return
    - The defensive anchor "no scenario overlap" branch is still present: `grep -c "no scenario overlap" scripts/eval_falsifier.py` >= 1
    - The module docstring exit-2 contract now mentions the wrong-matrix/zero-overlap case: `grep -ci "zero scenario\|wrong-matrix\|zero scenarios" scripts/eval_falsifier.py` >= 1
    - `grep -c "import openai\|import anthropic\|from langchain\|build_chat_model" scripts/eval_falsifier.py` is 0 (no live-SDK imports introduced — artifact-reading only, D-12-06)
  </acceptance_criteria>
  <done>Run-dir mode refuses to grade a zero-overlap (wrong-matrix) summary: it prints the existing diagnosis, writes a refusal to stderr, and returns exit 2 before any verdict; baselines-mode and the empty-expected-set fallback are untouched.</done>
</task>

<task type="auto">
  <name>Task 2: Unit test — zero-overlap run-dir summary returns exit 2 with no VERDICT line; in-matrix run still grades</name>
  <files>tests/unit/test_eval_falsifier.py</files>
  <read_first>
    - tests/unit/test_eval_falsifier.py — reuse the existing scaffolding, all in context:
        • `_load_script()` + `script` fixture (lines 31-45) — module loaded via importlib.
        • `_make_summary()` (lines 56-69) and `_cell_with_cir()` (lines 72-82) helpers.
        • `_write_run_summary(tmp_path, summary)` (lines 466-471) — writes summary.json
          to a tmp run dir and returns the path.
        • `class TestResolvedSourceVisibility` (lines 596-659), especially
          `test_warns_when_no_expected_matrix_scenario_present` (line 627): it drives a
          `refinement_cheaper`-only summary through main() and asserts the WARNING text
          prints. NOTE: that test does NOT assert the return code, so escalating to
          exit 2 does not break it — but the NEW tests below must assert rc == 2 and
          that no VERDICT was emitted. Confirm the helper `_run_with_scenario` ignores
          rc before relying on this.
        • `test_no_warning_when_expected_scenario_present` (line 637): an
          `omakase_mission_open_ended` summary must still grade (no wrong-matrix refusal).
        • `script._expected_matrix_scenarios()` (test at line 652) confirms the real
          config contains `omakase_mission_open_ended` — use that scenario id for the
          "in-matrix still grades" regression test and a non-matrix id (e.g.
          `refinement_cheaper` or a clearly-bogus `scenario_not_in_matrix`) for the
          zero-overlap test.
    - scripts/eval_falsifier.py (the Task 1 guard) — to assert the exact contract:
      zero-overlap run-dir summary -> stderr refusal + exit 2 + NO "VERDICT" in stdout.
    - .planning/STATE.md memory `feedback_test_layering`: new behavior needs assertion
      depth, not a single happy-path check — cover exit code AND the no-verdict invariant
      AND the negative case (in-matrix run unaffected) AND the empty-expected-set fallback.
  </read_first>
  <action>
    Add a new test class (e.g. `class TestZeroOverlapRefusesWithExit2`) to
    tests/unit/test_eval_falsifier.py using the existing `script` fixture,
    `_make_summary`, `_cell_with_cir`, and `_write_run_summary` helpers — no new
    loading machinery, no live API, no real eval_reports run dirs.

    Tests to add (each drives `script.main([...])` against a tmp run dir):

    (A) test_zero_overlap_run_dir_returns_exit_2: build a summary whose ONLY scenario
        is NOT in configs/eval_matrix.yaml (e.g. `refinement_cheaper`, or to be robust
        against future matrix edits, assert the chosen id is absent via
        `assert chosen_id not in script._expected_matrix_scenarios()` first). Give it a
        gpt-5-mini cell at 0.8 and an anchor cell at 1.0 so that, absent the guard, it
        would PASS — proving the guard prevents a spurious milestone PASS. Write it via
        `_write_run_summary`, call `main(["--run-dir", str(run_dir), "--baselines-dir",
        str(REPO_ROOT / "configs" / "eval_baselines")])`, and assert `rc == 2`.

    (B) test_zero_overlap_emits_no_verdict_line: same setup as (A); capture stdout with
        `capsys` and assert `"VERDICT" not in captured.out` AND
        `"VERDICT = PASS" not in captured.out` AND `"VERDICT = FAIL" not in captured.out`.
        Also assert the existing diagnosis still prints
        (`"may belong to a different matrix" in captured.out` or the
        `"eval_matrix_refinement"` hint), confirming the message was kept, not removed.

    (C) test_in_matrix_run_still_grades: build a summary whose scenario IS in the matrix
        (`omakase_mission_open_ended`) with gpt-5-mini 0.8 + anchor 1.0; assert
        `main([...])` returns 0 (or at least in {0, 1}, a real verdict — pick 0 since
        0.8 >= 0.6 and anchor holds against committed baselines) and that
        `"VERDICT" in captured.out`. This is the negative control proving the guard is
        scoped to wrong-matrix runs only.

    (D) test_empty_expected_set_does_not_refuse: monkeypatch
        `script._expected_matrix_scenarios` to return `set()` (simulating an
        unreadable/empty matrix config), build a summary with any scenario id and a
        passing gpt-5-mini + anchor, and assert `main([...])` returns a real verdict
        (in {0, 1}, NOT 2) — proving the best-effort empty-set path still grades and
        the guard does not over-fire. Use `monkeypatch.setattr(script,
        "_expected_matrix_scenarios", lambda *a, **k: set())`.

    Keep all tests live-key-free and cwd-independent (use `REPO_ROOT` for the real
    baselines dir, already defined at the top of the file). Do not modify or weaken any
    existing test; only add the new class.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_falsifier.py -q</automated>
  </verify>
  <acceptance_criteria>
    - A test asserts a zero-overlap run-dir summary (scenario absent from `script._expected_matrix_scenarios()`, with otherwise-passing gpt-5-mini + anchor cells) makes `main()` return exit 2
    - A test asserts that on the zero-overlap path, `"VERDICT"` does NOT appear in captured stdout (no PASS/FAIL emitted), while the existing wrong-matrix diagnosis text IS still printed
    - A negative-control test asserts an in-matrix scenario (`omakase_mission_open_ended`) still grades to a real verdict (exit 0) with `"VERDICT"` in stdout
    - A test monkeypatches `_expected_matrix_scenarios` to `set()` and asserts the guard does NOT fire (exit in {0, 1}, not 2)
    - No existing test is modified or weakened; `poetry run pytest tests/unit/test_eval_falsifier.py -q` passes (all prior tests + new class green)
  </acceptance_criteria>
  <done>tests/unit/test_eval_falsifier.py pins the new contract: zero-overlap run-dir -> exit 2 with no VERDICT line, in-matrix run still grades, and an empty expected set does not over-trigger the guard.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Phase 13 arm-judging ← falsifier exit code | downstream consumes 0/1/2 mechanically; a wrong-matrix run that emits 0/1 silently mis-grades an arm (spurious PASS or FAIL) |
| falsifier ← resolved run-dir summary.json | the auto-resolved "latest" run may belong to a different matrix (eval_matrix_refinement writes to the same eval_reports/ base); zero scenario overlap with eval_matrix.yaml is a usage error, not a verdict |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-11 | Elevation of privilege | wrong-matrix run emits a verdict and exit 0/1, mis-grading a Phase 13 arm | mitigate | run-dir zero-overlap escalates to `return 2` BEFORE any verdict (Task 1); unit test pins exit 2 + no VERDICT line (Task 2) |
| T-12-12 | Tampering | a refinement run that clears 0.6 read as a spurious milestone PASS | mitigate | refusal fires before the gpt-5-mini check; test (A) uses an otherwise-passing summary to prove the guard prevents the spurious PASS |
| T-12-13 | Denial of service (over-fire) | an unreadable/empty matrix config (empty expected set) turns every run into exit 2 | mitigate | guard scoped to `expected_scenarios` truthy via short-circuit; Task 2 test (D) monkeypatches empty set and asserts grading still proceeds |
| T-12-SC | Tampering | npm/pip/cargo installs | accept | no new package installs (stdlib + in-repo edits only); no slopcheck needed |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_falsifier.py -q` passes (all prior tests + the new zero-overlap class)
- `make lint` passes (ruff E,F,I,N,UP,B,SIM, line-length 100 per CLAUDE.md) on scripts/eval_falsifier.py and tests/unit/test_eval_falsifier.py
- `make test` passes (full suite, no DB-pool contamination — per project memory run the full suite, not just the changed file)
- Manual contract check (run-dir zero-overlap → exit 2, no verdict): construct a tmp run dir whose summary's only scenario is not in eval_matrix.yaml and confirm `poetry run python scripts/eval_falsifier.py --run-dir <that dir>; echo "exit=$?"` prints the wrong-matrix diagnosis, NO "VERDICT" line, and `exit=2`
- `--baselines-mode` regression: `poetry run python scripts/eval_falsifier.py --baselines-mode --baselines-dir configs/eval_baselines; echo "exit=$?"` still exits 0 or 1 (real verdict, unaffected by the guard)
- `grep -c "import openai\|from langchain\|build_chat_model" scripts/eval_falsifier.py` is 0 (no live-SDK imports introduced)
</verification>

<success_criteria>
- The UAT gap is closed: in run-dir mode a zero-scenario-overlap (wrong-matrix) summary makes eval_falsifier print the existing diagnosis and exit 2 with NO PASS/FAIL verdict line — scripted/CI consumers can no longer read a spurious FAIL or a spurious milestone PASS off a wrong-matrix run
- The escalation is surgical: --baselines-mode is unchanged; an empty/unreadable matrix config (empty expected set) does NOT trigger the refusal; the existing WR-06 diagnostic text is preserved verbatim; the defensive anchor "no scenario overlap → warn and pass" branch is retained
- Behavior is pinned by tests: exit 2 + no VERDICT on zero overlap, an in-matrix run still grades, and the empty-expected-set fallback still grades — all live-key-free, cwd-independent
</success_criteria>

<output>
Create `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-05-SUMMARY.md` when done
</output>
