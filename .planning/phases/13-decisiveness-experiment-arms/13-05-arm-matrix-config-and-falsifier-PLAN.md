---
phase: 13-decisiveness-experiment-arms
plan: 05
type: execute
wave: 2
depends_on: ["13-01"]
files_modified:
  - configs/eval_matrix_arm.yaml
  - scripts/eval_falsifier.py
  - Makefile
  - tests/unit/test_eval_falsifier.py
autonomous: true
requirements: [DEC-04, DEC-05]
must_haves:
  truths:
    - "An arm matrix config runs exactly 3 providers (openai/gpt-4o-mini anchor, openai/gpt-5-mini, deepseek/deepseek-reasoner) over exactly 2 scenarios (omakase_mission_open_ended + refinement_cheaper); anthropic and gemini are absent (deferred); scenario IDs match committed baseline scenario IDs exactly so the falsifier's zero-overlap exit-2 guard is satisfied"
    - "The falsifier accepts a --matrix-config flag so arm run dirs are graded against configs/eval_matrix_arm.yaml's scenario universe (not the omakase-only default), and the zero-overlap exit-2 guard reads that universe"
    - "The falsifier reports the model-initiated vs forced commit split per model by reading individual run JSONs' commit_forced field (D-13-04), printed in the A2 verdict line"
    - "Makefile exposes eval-matrix-arm and eval-falsifier-arm targets for running an arm and grading its run dir"
  artifacts:
    - path: "configs/eval_matrix_arm.yaml"
      provides: "3-provider x 2-scenario arm matrix config (anthropic/gemini deferred)"
      contains: "refinement_cheaper"
    - path: "scripts/eval_falsifier.py"
      provides: "--matrix-config flag + model-initiated/forced split reader"
      contains: "matrix-config"
    - path: "Makefile"
      provides: "eval-matrix-arm + eval-falsifier-arm targets"
      contains: "eval-matrix-arm"
  key_links:
    - from: "scripts/eval_falsifier.py split reader"
      to: "run JSON deterministic.commit_forced"
      via: "per-run JSON read aggregating model-initiated vs forced commits"
      pattern: "commit_forced"
    - from: "scripts/eval_falsifier.py --matrix-config"
      to: "configs/eval_matrix_arm.yaml scenarios"
      via: "zero-overlap exit-2 guard reads the arm scenario universe"
      pattern: "matrix"
---

<objective>
Build the run-and-grade plumbing for the arms: the committed arm matrix config
(3 providers x 2 scenarios, anthropic/gemini deferred), the falsifier extension
that grades an arm run dir against that two-scenario universe AND prints the
D-13-04 model-initiated vs forced commit split, and the Makefile targets to run
an arm and grade it.

Purpose: D-13-02 (two-scenario universe so the falsifier is not vacuous;
omakase-only is vacuous for gpt-5-mini per D-12-08), D-13-04 (forced-commit
honesty split visible in the verdict), and the canonical_refs requirement that
arm run dirs satisfy the falsifier's zero-overlap exit-2 guard (scenario IDs must
match committed baseline IDs exactly). Depends on 13-01 for the commit_forced
field the split reader consumes. Touches only configs/eval_falsifier.py/Makefile —
no overlap with the graph/prompt/revision plans.

Output: configs/eval_matrix_arm.yaml (new), a --matrix-config flag + split reader
in eval_falsifier.py, eval-matrix-arm/eval-falsifier-arm Makefile targets, and
falsifier unit tests. Live runs themselves are plan 13-06.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md
@.planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md
@.planning/phases/13-decisiveness-experiment-arms/13-01-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create the arm matrix config (3 providers x 2 scenarios)</name>
  <files>configs/eval_matrix_arm.yaml</files>
  <read_first>
    - configs/eval_matrix_refinement.yaml (sibling arm-config shape to copy: entries list with provider/model, scenarios list)
    - configs/eval_matrix.yaml (default matrix — note omakase_mission_open_ended is the only default scenario; arm config ADDS refinement_cheaper)
    - configs/eval_baselines/ directory listing (committed baseline scenario file names — arm scenario IDs MUST match these exactly for the falsifier zero-overlap guard: omakase_mission_open_ended, refinement_cheaper)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "configs/eval_matrix_arm.yaml (new)" (exact entries + scenarios)
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-02 (provider set + scenario universe + anthropic/gemini deferred + late_night_closure_cascade quarantined)
  </read_first>
  <action>
    Create `configs/eval_matrix_arm.yaml` with `entries:` listing exactly three providers — `openai/gpt-4o-mini` (anchor; comment: must show no behavioral change with flags off), `openai/gpt-5-mini`, `deepseek/deepseek-reasoner` — and `scenarios:` listing exactly `omakase_mission_open_ended` and `refinement_cheaper`. Do NOT include anthropic or gemini entries (D-12-09 deferred). Do NOT include late_night_closure_cascade (D-10-09 quarantined). Add a header comment explaining: arm runs select arm behavior via env flags passed at run time (VIABILITY_CONTRACT_ENABLED / FORCED_COMMIT_STEP / PARALLEL_TOOL_EXECUTION_ENABLED), NOT via per-cell `env` here — the arm flags are graph-build-time reads (plan 13-04). The two scenario IDs MUST match the committed baseline file stems exactly (omakase_mission_open_ended, refinement_cheaper) so the falsifier's zero-overlap exit-2 guard passes. Do NOT put gate values in this file (gates live in configs/eval_gates.yaml — D-10-05). If MatrixEntry requires `env` as YAML strings, follow the eval_matrix_refinement.yaml string-quoting rule; otherwise omit env entirely.
  </action>
  <verify>
    <automated>poetry run python -c "import yaml; d=yaml.safe_load(open('configs/eval_matrix_arm.yaml')); assert [e['model'] for e in d['entries']]==['gpt-4o-mini','gpt-5-mini','deepseek-reasoner'], d['entries']; assert set(d['scenarios'])=={'omakase_mission_open_ended','refinement_cheaper'}, d['scenarios']; print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - configs/eval_matrix_arm.yaml has exactly 3 entries: gpt-4o-mini, gpt-5-mini, deepseek-reasoner (no anthropic, no gemini).
    - scenarios are exactly omakase_mission_open_ended + refinement_cheaper (no late_night_closure_cascade).
    - The two scenario IDs match committed baseline file stems in configs/eval_baselines/ (verified: `ls configs/eval_baselines/omakase_mission_open_ended.json configs/eval_baselines/refinement_cheaper.json` both exist).
    - `poetry run python scripts/eval_matrix.py --matrix-config configs/eval_matrix_arm.yaml --runs 1 --llm-provider-override scripted` parses the config without a YAML/pydantic error.
  </acceptance_criteria>
  <done>The arm matrix config runs the 3-provider x 2-scenario universe with deferred/quarantined cells excluded and scenario IDs that satisfy the falsifier's overlap guard.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Extend the falsifier with --matrix-config and the forced-commit split reader</name>
  <files>scripts/eval_falsifier.py, tests/unit/test_eval_falsifier.py</files>
  <read_first>
    - scripts/eval_falsifier.py lines 67-90 (_expected_matrix_scenarios + the zero-overlap exit-2 guard at 272-287 — must read the arm scenario universe when --matrix-config is given)
    - scripts/eval_falsifier.py lines 179-216 (_parse_args — add the --matrix-config argument here)
    - scripts/eval_falsifier.py lines 259-310 (gpt-5-mini pooled check + per-scenario print — the split line is added near here for the A2 verdict)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "scripts/eval_falsifier.py — extend for arm run-dir mode" (--matrix-config arg + _commit_split_from_run_dir helper that reads individual run JSONs, filters by scenario from filename, aggregates commit_forced vs first_commit_call_step)
    - tests/unit/test_eval_falsifier.py (existing importlib-based test pattern for the falsifier; fixtures for run dirs / summaries)
  </read_first>
  <behavior>
    - eval_falsifier accepts --matrix-config PATH; the zero-overlap exit-2 guard reads scenarios from that path (default configs/eval_matrix.yaml) so an arm run dir grades against the two-scenario universe.
    - _commit_split_from_run_dir(run_dir, provider_key) returns (model_initiated, forced) counts by reading individual *.json run files (not summary.json): commit_forced True counts as forced; a non-forced run with first_commit_call_step not None counts as model-initiated.
    - The A2 verdict line prints the split, e.g. "gpt-5-mini: ... committed_itinerary_rate = 0.800 (model-initiated 2/5, forced 2/5)".
    - Exit-code contract unchanged: 0 PASS, 1 FAIL, 2 infra (including zero-overlap with the configured matrix's scenarios).
  </behavior>
  <action>
    In `scripts/eval_falsifier.py`: (1) Add a `--matrix-config` argument to `_parse_args` (default None → resolve to `_DEFAULT_MATRIX_CONFIG`). Thread it into `_expected_matrix_scenarios(matrix_path=...)` so the zero-overlap exit-2 guard reads the arm scenario universe when grading an arm run dir. (2) Add `_commit_split_from_run_dir(run_dir, provider_key, scenario_ids=None) -> tuple[int,int]` that globs individual run JSONs (`{provider--model}--*.json`, NOT summary.json), reads each file's `deterministic.commit_forced` and `deterministic.first_commit_call_step`, and aggregates: `commit_forced` True → forced += 1; else `first_commit_call_step is not None` → model_initiated += 1. Optionally filter by scenario parsed from the filename (pattern provider--model--scenario--run-N.json). Guard all reads (OSError/ValueError → skip file). (3) In `main()`, after the gpt-5-mini pooled rate is computed (and for the anchor too), call the split reader and print the split annotation on the verdict line (D-13-04 / D-13-09 format: `(model-initiated {mi}/{total}, forced {fc}/{total})`). Keep the exit-code contract and the median-weighted honesty note unchanged. Add tests to `tests/unit/test_eval_falsifier.py`: a fixture run dir with some commit_forced=True and some model-initiated runs asserts _commit_split_from_run_dir returns the right (mi, forced) tuple, and that --matrix-config changes which scenario universe the overlap guard checks.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_falsifier.py -v -k "split or matrix_config"</automated>
  </verify>
  <acceptance_criteria>
    - `poetry run python scripts/eval_falsifier.py --help` shows `--matrix-config`.
    - A test builds a fixture run dir with 2 forced + 2 model-initiated + 1 never-committed run for a provider and asserts _commit_split_from_run_dir returns (2, 2).
    - A test asserts that passing --matrix-config configs/eval_matrix_arm.yaml makes the overlap guard accept a run dir containing refinement_cheaper (which is not in the default eval_matrix.yaml).
    - Existing test_eval_falsifier.py tests still pass; exit codes 0/1/2 contract unchanged.
  </acceptance_criteria>
  <done>The falsifier grades arm run dirs against the two-scenario universe and prints the model-initiated vs forced split for the A2 verdict.</done>
</task>

<task type="auto">
  <name>Task 3: Add eval-matrix-arm and eval-falsifier-arm Makefile targets</name>
  <files>Makefile</files>
  <read_first>
    - Makefile lines 121-148 (eval-matrix + eval-matrix-refinement targets — the --matrix-config + --runs + LLM_OVERRIDE pattern to mirror; note the flag is --runs not --n)
    - Makefile lines 213-218 (eval-falsifier target — the --run-dir + --baselines-dir pattern; add --matrix-config to the arm variant)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "Makefile — arm eval targets" (eval-matrix-arm + eval-falsifier-arm shapes)
  </read_first>
  <action>
    Append two targets to the eval block in `Makefile`. (1) `eval-matrix-arm` — runs `scripts/eval_matrix.py --matrix-config configs/eval_matrix_arm.yaml --runs $(RUNS) $(if $(LLM_OVERRIDE),--llm-provider-override $(LLM_OVERRIDE),)`; document in the `##` help that arm behavior is selected by exporting the env flags before invocation (e.g. `VIABILITY_CONTRACT_ENABLED=1 make eval-matrix-arm RUNS=5` for A1; `FORCED_COMMIT_STEP=6 make eval-matrix-arm RUNS=5` for A2; `PARALLEL_TOOL_EXECUTION_ENABLED=1 make eval-matrix-arm RUNS=5` for A3) and that APP_ENV=eval + provider API keys are required for live runs. (2) `eval-falsifier-arm` — runs `scripts/eval_falsifier.py $(if $(RUN_DIR),--run-dir $(RUN_DIR),) --matrix-config $(if $(MATRIX_CONFIG),$(MATRIX_CONFIG),configs/eval_matrix_arm.yaml) --baselines-dir configs/eval_baselines`. Add a `MATRIX_CONFIG ?=` parameter variable near the existing RUN_DIR ?= declaration. Use the existing `$(POETRY_RUN)` prefix and `.PHONY` declarations consistent with neighboring targets. Use `--runs` (not `--n`) to match eval_matrix.py's argparse.
  </action>
  <verify>
    <automated>grep -q "eval-matrix-arm" Makefile && grep -q "eval-falsifier-arm" Makefile && grep -q "configs/eval_matrix_arm.yaml" Makefile && poetry run python scripts/eval_matrix.py --matrix-config configs/eval_matrix_arm.yaml --runs 1 --llm-provider-override scripted >/dev/null 2>&1 && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - `make -n eval-matrix-arm` prints a command invoking eval_matrix.py with --matrix-config configs/eval_matrix_arm.yaml and --runs.
    - `make -n eval-falsifier-arm RUN_DIR=eval_reports/x` prints a command invoking eval_falsifier.py with --run-dir and --matrix-config.
    - The `##` help text for eval-matrix-arm documents the per-arm env-flag exports.
    - A scripted dry run (`--llm-provider-override scripted`) of the arm matrix config exits 0.
  </acceptance_criteria>
  <done>Operators can run any arm via env-flag export + make eval-matrix-arm and grade the resulting run dir via make eval-falsifier-arm.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| run JSON files on disk → falsifier split reader | Eval artifacts read from the local filesystem; malformed JSON must not crash the verdict |
| --matrix-config path → scenario universe | Operator-supplied config path drives the overlap guard |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-05-01 | Tampering | malformed run JSON crashing the split reader | mitigate | _commit_split_from_run_dir wraps each file read in try/except (OSError, ValueError) and skips bad files — never raises (preserves the exit-2-only-on-infra contract) |
| T-13-05-02 | Spoofing | arm run dir with mismatched scenario IDs gaming the overlap guard | mitigate | scenario IDs in the arm config must match committed baseline stems exactly; the zero-overlap exit-2 guard refuses to grade a run that shares no scenarios with the configured matrix |
| T-13-05-03 | Repudiation | forced commits silently inflating commit rate | mitigate | the split reader makes model-initiated vs forced explicit in the verdict line (D-13-04) so a forced-inflated rate is never reported as plain commit rate |
| T-13-05-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs (yaml/json are existing deps); slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_falsifier.py -v` passes (including pre-existing tests).
- `poetry run python scripts/eval_matrix.py --matrix-config configs/eval_matrix_arm.yaml --runs 1 --llm-provider-override scripted` exits 0 (scripted, no live keys).
- `make -n eval-matrix-arm` and `make -n eval-falsifier-arm` print well-formed commands.
- `make lint` passes.
</verification>

<success_criteria>
- Arm matrix config runs the 3-provider x 2-scenario universe (anthropic/gemini deferred, late_night quarantined) with overlap-guard-safe scenario IDs.
- Falsifier grades arm run dirs against the two-scenario universe and prints the model-initiated vs forced split (D-13-04).
- Makefile exposes eval-matrix-arm + eval-falsifier-arm.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-05-SUMMARY.md` when done.
</output>
