---
phase: 13-decisiveness-experiment-arms
plan: 10
type: execute
wave: 3
gap_closure: true
depends_on: ["13-09"]
files_modified:
  - .planning/ROADMAP.md
  - docs/decisiveness_arm_verdicts.md
  - app/config.py
  - app/agent/graph.py
  - app/agent/revision.py
  - scripts/eval_agent.py
  - tests/unit/test_config.py
autonomous: true
requirements: [DEC-04]
must_haves:
  truths:
    - "Phase 13 ROADMAP success criterion 3 is respecified to require absolute gpt-4o-mini tool-execution latency recorded in run JSON for future-baseline use (the unmeasurable-reduction constraint annotated) — NOT a reduction-vs-Phase-12 delta, which is structurally impossible because the Phase-12 floor runs predate INST-04 step_telemetry"
    - "The A3 section of docs/decisiveness_arm_verdicts.md echoes the respecification: the discovered constraint (Phase-12 floor has step_telemetry=None) is named and the absolute tool_exec_seconds values are recorded as the future-baseline artifact"
    - "VIABILITY_CONTRACT_ENABLED is read at one place/time consistent with graph-build-time co-tuning of DEC-01 and DEC-03 (WR-02 split-read risk closed) — the prompt addendum and the critique scoping never desync"
    - "A single env_flag helper parses the truthy set ({1,true,yes,on}) and is used at every boolean-flag site that previously inlined the idiom (WR-09 DRY) so the graph and the run-JSON arm_flags self-description provably parse identically"
  artifacts:
    - path: "app/config.py"
      provides: "env_flag(name) truthy-set helper (single source of truth)"
      contains: "def env_flag"
    - path: ".planning/ROADMAP.md"
      provides: "Respecified Phase 13 success criterion 3 with the discovered-constraint annotation"
      contains: "absolute"
    - path: "docs/decisiveness_arm_verdicts.md"
      provides: "A3 respecification echo + future-baseline artifact note"
      contains: "future baseline"
  key_links:
    - from: "app/agent/revision.py low_similarity scoping flag"
      to: "app/agent/graph.py build-time VIABILITY_CONTRACT_ENABLED read"
      via: "single flag read shared by DEC-01 prompt addendum and DEC-03 critique scoping"
      pattern: "VIABILITY_CONTRACT_ENABLED"
    - from: "scripts/eval_agent.py arm_flags self-description"
      to: "app/config.py env_flag"
      via: "shared truthy-set parser so the report matches graph behavior"
      pattern: "env_flag"
---

<objective>
Close the SC-3 latency gap the cheapest honest way, and clear the two low-risk
WARNING findings (WR-02, WR-09) flagged for inclusion only if cheap.

SC-3 decision (locked-decision-consistent): the DEC-04 success criterion's "measurable
latency reduction" is STRUCTURALLY unmeasurable against the specified baseline — the
Phase-12 comparison-floor run dirs predate INST-04 step_telemetry (tool_exec_seconds is
None), so no valid before-point exists. The verification offered two closures: (a) a
same-phase flag-off control run for the gpt-4o-mini anchor at n=5, or (b) respecify the
criterion as "absolute latency recorded for future baseline use." Path (a) requires a
5TH full live matrix run — but D-13-02 hard-caps the phase at <=4 full live matrix runs
total and all 4 are already consumed (A1, A2, A3, and the A4 decision slot), and CONTEXT
forbids billing top-ups. A 5th run would BREAK a locked decision. Therefore path (b) —
zero-spend respecification — is the only closure consistent with the locked decisions,
and it is what 13-VERIFICATION.md prefers as "the cheapest honest closure."

Purpose: make the roadmap criterion honest (it asks for something the specified baseline
cannot provide) and record the absolute tool_exec_seconds as the future-baseline artifact,
so a later phase that regenerates the Phase-12 floor with telemetry CAN compute the delta.

WR-02 and WR-09 are co-tuning-integrity and DRY fixes (per CLAUDE.md DRY is critical) that
remove a flag desync risk and a six-fold duplicated parser. Both are flag-off-byte-identical.

Output: respecified ROADMAP criterion 3 + A3 verdict echo; one env_flag helper used at
every boolean-flag site; VIABILITY_CONTRACT_ENABLED read once consistent with build-time
co-tuning. No flags enabled by default; no baselines written; no live API spend.
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
@.planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md
@.planning/phases/13-decisiveness-experiment-arms/13-VERIFICATION.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Respecify ROADMAP criterion 3 + echo the constraint in the A3 verdict section</name>
  <files>.planning/ROADMAP.md, docs/decisiveness_arm_verdicts.md</files>
  <read_first>
    - .planning/ROADMAP.md lines 82-95 (Phase 13 section; success criterion 3 at line 91: "measurable gpt-4o-mini latency reduction at n=5 recorded in run JSON")
    - docs/decisiveness_arm_verdicts.md lines 204-318 (the A3 section: the latency pass criterion at 214-216, the CRITICAL FINDING at 277-281 that Phase-12 floor runs have step_telemetry=None, the unmeasurable delta table at 283-295, the raw tool_exec_seconds block at 289-295, and the A3 Closing verdict at 304-318)
    - .planning/phases/13-decisiveness-experiment-arms/13-VERIFICATION.md (the SC-3 gap entry: missing[] offers the respecification "absolute latency recorded for future baseline use"; the Gaps Summary paragraph naming the structural unmeasurability)
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-02 (<=4 full live matrix runs hard cap; no billing top-ups) — the reason path (b) not (a)
  </read_first>
  <action>
    In `.planning/ROADMAP.md`, rewrite Phase 13 success criterion 3 (line 91) so it no longer demands a reduction DELTA the specified baseline cannot supply. New criterion text (per DEC-04, D-13-08): the parallel-tool-execution arm runs all tool calls within one act() plan step concurrently with results order-stable, AND the absolute gpt-4o-mini tool-execution latency at n=5 (INST-04 `tool_exec_seconds`, summed per run) is recorded in run JSON for future-baseline use. Append a one-line parenthetical annotation naming the discovered constraint: the reduction-vs-Phase-12-floor delta is structurally unmeasurable because the Phase-12 comparison-floor run dirs predate the INST-04 step_telemetry instrumentation (tool_exec_seconds=None there); a future phase that regenerates the floor with telemetry can compute the delta. Do NOT touch criteria 1, 2, 4, 5. Keep the Phase 13 line marked complete (criterion 3 is now satisfiable as respecified — absolute latency IS recorded).

    In `docs/decisiveness_arm_verdicts.md` A3 section, echo the respecification: add a short note in/under the Latency Analysis (near lines 277-295) that the ROADMAP criterion 3 has been respecified to "absolute latency recorded for future-baseline use", that the unmeasurable delta is a discovered constraint (Phase-12 floor has step_telemetry=None), and that the recorded absolute tool_exec_seconds values (the per-run block already present) ARE the future-baseline artifact a later telemetry-equipped floor regen will diff against. Do NOT change the A3 anchor-regression FAIL finding (refinement_cheaper 0.000 vs 1.000 baseline) or the honest null result.
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && grep -qi "absolute" .planning/ROADMAP.md && grep -qi "future-baseline\|future baseline" .planning/ROADMAP.md && grep -qi "future baseline\|future-baseline" docs/decisiveness_arm_verdicts.md && grep -qi "step_telemetry" docs/decisiveness_arm_verdicts.md && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - ROADMAP Phase 13 criterion 3 no longer contains the unqualified phrase "latency reduction" as a required deliverable; it requires absolute tool_exec_seconds recorded for future-baseline use.
    - ROADMAP criterion 3 carries the discovered-constraint annotation naming the Phase-12 step_telemetry=None gap.
    - The A3 verdict section echoes the respecification and names the absolute tool_exec_seconds as the future-baseline artifact.
    - The A3 anchor-regression FAIL finding and the closing null result are unchanged (`grep -c "anchor regression" docs/decisiveness_arm_verdicts.md` unchanged; `grep -c "No arm cleared" docs/decisiveness_arm_verdicts.md` >= 1).
    - No live run was performed (`git status configs/eval_baselines/` clean; no new eval_reports/ run dir created).
  </acceptance_criteria>
  <done>SC-3 is closed at zero spend: the criterion is respecified to the satisfiable, honest form (absolute latency for future baseline) with the discovered constraint annotated in both the ROADMAP and the A3 verdict; the structural-unmeasurability is named, not hidden.</done>
</task>

<task type="auto">
  <name>Task 2: Add env_flag helper (WR-09 DRY) and use it at every boolean-flag site</name>
  <files>app/config.py, app/agent/graph.py, scripts/eval_agent.py, app/agent/revision.py, tests/unit/test_config.py</files>
  <read_first>
    - app/config.py (full file — module conventions, get_settings/Settings; add the helper here as a module-level function, NOT a Settings field, since flags are read live from os.environ)
    - app/agent/graph.py lines 304-310 (the two inlined truthy-set reads: VIABILITY_CONTRACT_ENABLED and PARALLEL_TOOL_EXECUTION_ENABLED; FORCED_COMMIT_STEP is an int read, NOT a boolean — leave it as int())
    - scripts/eval_agent.py lines 928-940 (arm_flags assembly: the two inlined `... in {"1","true","yes","on"}` boolean reads) and line 1180 (`flag_enabled = flag_raw.strip().lower() in {"1","true","yes","on"}`)
    - app/agent/revision.py lines 35-37 (the inlined VIABILITY_CONTRACT_ENABLED read — Task 3 changes its READ POINT; this task only swaps the parser idiom)
    - app/main.py lines 753-763 (the canonical REFINEMENT_STRUCTURED_PLAN_ENABLED truthy-set precedent the helper must reproduce exactly: `os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}`)
  </read_first>
  <action>
    Add a single helper to `app/config.py`: `def env_flag(name: str) -> bool` that returns `os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}` (import `os` at the top of config.py if not already imported). This is the ONE truthy-set definition. Then replace every inlined boolean truthy-set read with a call to `env_flag(...)` at these sites: app/agent/graph.py (VIABILITY_CONTRACT_ENABLED and PARALLEL_TOOL_EXECUTION_ENABLED reads, lines 305-310 — keep the FORCED_COMMIT_STEP int() read as-is), scripts/eval_agent.py (the two arm_flags boolean reads at 929-937 and the flag_enabled read at ~1180). Do NOT change the int parse for FORCED_COMMIT_STEP. Add app/main.py's REFINEMENT_STRUCTURED_PLAN_ENABLED read to use env_flag too IF it is touched, but it is out of scope here unless trivial — leave it unless the executor confirms it is a pure one-line swap with no test churn. Add a unit test in tests/unit/test_config.py asserting env_flag returns True for each of {"1","true","yes","on"} (and case/whitespace variants like " TRUE ") and False for "", "0", "off", "no", and unset. The acceptance gain (D-13-05 / WR-09): the graph's flag reads and the run-JSON arm_flags self-description now parse through the SAME function — they cannot desync.
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && grep -q "def env_flag" app/config.py && poetry run pytest tests/unit/test_config.py -q && poetry run pytest tests/unit/test_eval_agent.py tests/unit/test_graph_forced_commit.py -q && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - `app/config.py` defines `def env_flag(name: str) -> bool` with the canonical truthy set `{"1", "true", "yes", "on"}`.
    - `grep -rc 'strip().lower() in {"1", "true", "yes", "on"}' app/agent/graph.py scripts/eval_agent.py` is 0 for the boolean sites (replaced by env_flag) — the FORCED_COMMIT_STEP int read remains.
    - graph.py, eval_agent.py, and revision.py import and call `env_flag` (or graph/eval_agent do; revision's call site is finalized in Task 3).
    - A new test in tests/unit/test_config.py pins env_flag truthiness for all accepted tokens and rejects "", "0", "off", "no", unset.
    - `poetry run pytest tests/unit/test_config.py tests/unit/test_eval_agent.py tests/unit/test_graph_forced_commit.py -q` exits 0.
    - Flag-off behavior is byte-identical (the helper reproduces the exact prior truthy set) — existing flag-off tests still pass.
  </acceptance_criteria>
  <done>One env_flag helper is the single source of truth for boolean-flag parsing; every prior inline copy calls it; the graph and the report's arm_flags self-description provably parse identically (WR-09 closed).</done>
</task>

<task type="auto">
  <name>Task 3: Close the VIABILITY_CONTRACT_ENABLED split-read risk (WR-02)</name>
  <files>app/agent/revision.py, app/agent/graph.py, tests/unit/test_agent_revision.py</files>
  <read_first>
    - app/agent/revision.py lines 35-37 (`_VIABILITY_CONTRACT_ENABLED` read at MODULE IMPORT time) and lines 185-212 (`_diagnose_last_tool_result`: the DEC-03 low_similarity suppression gated on `_VIABILITY_CONTRACT_ENABLED` at line 206 — the critique half of the co-tuning)
    - app/agent/graph.py lines 304-312 (the build-time read of VIABILITY_CONTRACT_ENABLED → `_viability_contract_enabled` and the prompt addendum at 312 — the prompt half of the co-tuning, read at graph-build time)
    - .planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md WR-02 (D-13-05 requires DEC-01 prompt addendum and DEC-03 critique scoping to flip TOGETHER; import-time vs build-time reads can split when env changes between import and build; recommended fix: read the flag in one place at one time)
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-05 (sharing one flag MECHANICALLY enforces co-tuning — that invariant must hold by construction, not by env-freeze timing)
    - tests/unit/test_agent_revision.py lines 161-267 (the flag-on tests that importlib.reload(rev) — confirm the chosen fix keeps these green; WR-03 module-leak is a separate finding, do NOT attempt it here unless trivial)
  </read_first>
  <action>
    Eliminate the split between revision.py's import-time `_VIABILITY_CONTRACT_ENABLED` and graph.py's build-time read so DEC-01 (prompt) and DEC-03 (critique scoping) cannot desync. Choose the lowest-risk construction: have `_diagnose_last_tool_result` (and any other revision.py consumer of the flag) read the flag LIVE per call via `env_flag("VIABILITY_CONTRACT_ENABLED")` (from app.config, added in Task 2) instead of reading the module-level `_VIABILITY_CONTRACT_ENABLED` constant — this makes the critique half read at the same effective time as graph.py's build-time read in same-process eval runs, and removes the import-time freeze hazard. Keep the flag-off byte-identity guarantee: the live read stays INSIDE the existing `if hint.reason == "low_similarity" and <flag>:` gate so flag-off runs never call all_slots_viable (T-13-03-02). The cheapest correct change: replace the `_VIABILITY_CONTRACT_ENABLED` reference at line 206 with `env_flag("VIABILITY_CONTRACT_ENABLED")`; either remove the now-unused module-level constant or keep it but stop relying on it for the gate (prefer removing it to avoid a dead/misleading symbol — confirm no other importer references it via grep before removing). Verify the existing test_agent_revision.py flag-on tests still pass (they set the env then reload — a live read honors the env without needing reload, so they remain green; adjust only if a test asserts the module-level constant directly).
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && grep -q 'env_flag("VIABILITY_CONTRACT_ENABLED")' app/agent/revision.py && poetry run pytest tests/unit/test_agent_revision.py tests/unit/test_agent_prompts.py -q && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - The DEC-03 low_similarity suppression gate in `_diagnose_last_tool_result` reads the flag live via `env_flag("VIABILITY_CONTRACT_ENABLED")` (no longer the import-time module constant).
    - If the module-level `_VIABILITY_CONTRACT_ENABLED` constant is removed, `grep -rn "_VIABILITY_CONTRACT_ENABLED" app/ scripts/ tests/` returns no live importer references; if kept, it is no longer the gate's read source.
    - Flag-off byte-identity holds: the live read remains inside the `low_similarity` + flag gate, so flag-off runs never call all_slots_viable.
    - `poetry run pytest tests/unit/test_agent_revision.py tests/unit/test_agent_prompts.py -q` exits 0 (DEC-01 prompt locks and DEC-03 scoping tests both green).
    - The Phase-7 grep gate stays green: `poetry run pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` exits 0.
  </acceptance_criteria>
  <done>The DEC-01 prompt addendum and DEC-03 critique scoping read the same flag at the same effective time; the import-time/build-time split-read hazard (WR-02) is closed without breaking flag-off byte-identity or the prompt locks.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | This plan edits planning/docs files, a config helper, and internal env-flag read points behind off-by-default flags. No new external input, network, or auth surface. The env_flag helper reads `os.environ` (process-trusted) only; no user-controlled data crosses a new boundary. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-10-01 | Repudiation | respecified ROADMAP criterion hides the unmeasurability instead of recording it | mitigate | Criterion 3 explicitly annotates the discovered constraint (Phase-12 step_telemetry=None); A3 verdict echoes it; the absolute tool_exec_seconds is preserved as the future-baseline artifact |
| T-13-10-02 | Tampering | env_flag refactor silently changes the accepted truthy set, desyncing graph behavior from arm_flags | mitigate | tests/unit/test_config.py pins the exact truthy set {1,true,yes,on}; single helper used at all sites so graph and report parse identically |
| T-13-10-03 | Tampering | WR-02 fix breaks flag-off byte-identity (DEC-01/DEC-03 co-tuning fires when flag off) | mitigate | The live flag read stays inside the existing low_similarity+flag gate; flag-off path never calls all_slots_viable; flag-off tests and the Phase-7 grep gate must stay green |
| T-13-10-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A — no new dependencies added |
</threat_model>

<verification>
- ROADMAP criterion 3 respecified with the discovered-constraint annotation; A3 verdict echoes it; null result and anchor-regression finding unchanged.
- `poetry run pytest tests/unit/test_config.py tests/unit/test_agent_revision.py tests/unit/test_agent_prompts.py tests/unit/test_eval_agent.py tests/unit/test_graph_forced_commit.py -q` exits 0.
- The Phase-7 grep gate stays green: `poetry run pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` exits 0.
- Full suite mandatory before merge (DB-pool contamination risk with real-graph tests): `make test` passes.
- No live API spend; no 5th matrix run (D-13-02 four-run cap respected); no baselines written (`git status configs/eval_baselines/` clean); no flags enabled by default.
</verification>

<success_criteria>
- SC-3 closed at zero spend and consistent with the locked <=4-run cap: criterion 3 respecified to absolute-latency-for-future-baseline with the constraint annotated in ROADMAP and the A3 verdict.
- WR-09 closed: one env_flag helper; every boolean-flag site uses it; graph and arm_flags parse identically.
- WR-02 closed: VIABILITY_CONTRACT_ENABLED read consistently for the co-tuned DEC-01/DEC-03 halves; flag-off byte-identity and prompt locks preserved.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-10-SUMMARY.md` when done.
</output>
