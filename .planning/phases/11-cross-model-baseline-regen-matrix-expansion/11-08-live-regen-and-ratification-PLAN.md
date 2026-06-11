---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 08
type: execute
wave: 5
depends_on: ["11-01", "11-02", "11-03", "11-04", "11-05", "11-06", "11-07"]
files_modified:
  - docs/baseline_regen.md
  - configs/eval_baselines/_snapshots/omakase_mission_open_ended.pre-phase11.json
  - configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase11.json
  - configs/eval_baselines/_snapshots/late_night_closure_cascade.pre-phase11.json
  - configs/eval_baselines/omakase_mission_open_ended.json
  - configs/eval_baselines/refinement_cheaper.json
  - configs/eval_gates.yaml
  - docs/eval_gates.md
  - tests/unit/test_eval_matrix.py
autonomous: false
requirements: [BASE-01, BASE-02, BASE-03]
user_setup:
  - service: cloud-sql-proxy
    why: "Wave-2 live regen needs Cloud SQL reachable (or local Postgres) for the semantic_search retrieval path"
    dashboard_config:
      - task: "Start cloud-sql-proxy on port 5433 for instance mlops-491820:us-central1:mlops--city-concierge (or bring up local Docker Postgres)"
        location: "Local terminal — see docs/baseline_regen.md preconditions"
  - service: provider-api-keys
    why: "All four providers run live at n=5"
    env_vars:
      - name: OPENAI_API_KEY
        source: "OpenAI dashboard (also covers shared embeddings quota)"
      - name: DEEPSEEK_API_KEY
        source: "DeepSeek dashboard"
      - name: ANTHROPIC_API_KEY
        source: "Anthropic console"
      - name: GOOGLE_API_KEY
        source: "Google AI Studio (gemini logged-not-gated)"
must_haves:
  truths:
    - "docs/baseline_regen.md documents the ordered regen procedure: preconditions, probe, snapshot, matrix runs at RUNS=5 for both YAML configs, write_baselines, baselines-mode gate check, commit"
    - "pre-phase11 snapshots of all three baseline JSONs exist in _snapshots/ before regen overwrites the canonical files"
    - "configs/eval_baselines/*.json are regenerated at n=5 by scripts/write_baselines.py with the Wave-0 measurement fixes and Phase-9 adapters active; committed_itinerary_rate is present in the regenerated scorers blocks"
    - "anthropic/claude-sonnet-4-6 gate moves provisional-n1 -> active; gpt-5-mini stays aspirational; deepseek/gemini stay logged - each change carries a D-11 rationale line"
    - "make eval-gates-check-baselines passes against the regenerated committed baselines (or surfaces only the known aspirational gpt-5-mini miss)"
  artifacts:
    - path: "docs/baseline_regen.md"
      provides: "BASE-01 regen runbook (D-11-08)"
    - path: "configs/eval_baselines/_snapshots/omakase_mission_open_ended.pre-phase11.json"
      provides: "Pre-regen audit trail of the fail-open-saturated baselines (D-11-09)"
    - path: "configs/eval_gates.yaml"
      provides: "Re-ratified statuses/values with D-11-20 rationale lines + NOTE-block update"
  key_links:
    - from: "make eval-matrix RUNS=5 / make eval-matrix-refinement RUNS=5"
      to: "scripts/write_baselines.py -> configs/eval_baselines/*.json"
      via: "summary.json with committed_itinerary_rate threaded (D-11-02)"
      pattern: "committed_itinerary_rate"
    - from: "configs/eval_gates.yaml anthropic status"
      to: "regenerated refinement_cheaper.json anthropic n=5 cell"
      via: "D-11-20 re-ratification"
      pattern: "active"
---

<objective>
Run the Wave-2 live regen - the LAST wave, after every measurement-semantics fix and tooling change has landed, so baselines are written exactly once (D-11-01). Author the `docs/baseline_regen.md` runbook (D-11-08), snapshot the three pre-phase11 baselines (D-11-09), run both matrices live at n=5 (D-11-10), write the new baselines via the Wave-1 tool (refusing partial/quarantined cells), re-ratify the gates from fresh n=5 data (D-11-20), and remove the now-landed cross-model cells from the parity test's deferred set. Gemini erroring is a documented deferral (D-11-11); a gpt-4o-mini commit_rate below 0.8 is a STOP-and-investigate real regression.

Purpose: BASE-01 replaces the fail-open-saturated v2.0 baselines with honest n=5 measurements under DB-up conditions; BASE-02 produces the four-provider results without errors; BASE-03's gate re-ratification anchors the committed empirical record. This plan requires live infra and human verification, so it is non-autonomous.
Output: Runbook, snapshots, regenerated baselines, re-ratified gates, and an updated parity test.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-05-SUMMARY.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-06-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Author docs/baseline_regen.md runbook</name>
  <files>docs/baseline_regen.md</files>
  <read_first>
    - configs/eval_baselines/_snapshots/README.md - the structural template for the runbook's "what this is / lifecycle" framing and the snapshot naming convention.
    - docs/eval_gates.md - the ordered-step + Makefile-command doc style to imitate (renders YAML, never duplicates numbers).
    - Makefile - read the `probe-providers` (158-163), `eval-matrix` (120-125), `eval-matrix-refinement` (132-137), `snapshot-baselines` and `write-baselines` (added in 11-05), and `eval-gates-check-baselines` (added in 11-06) targets so the runbook cites real target names + params.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md - the `docs/baseline_regen.md (NEW)` section shows the full ordered-step structure and the failure-branch section.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md - D-11-08 ordered steps, D-11-10 regen scope (omakase + refinement; late_night NOT regenerated), D-11-11 gemini failure branch, External Preconditions (DB instance is double-dash mlops--city-concierge; embeddings sanity probe; all 4 keys), and specifics (gpt-4o-mini < 0.8 = STOP).
  </read_first>
  <action>
    Create `docs/baseline_regen.md` (BASE-01 / D-11-08) with: a Preconditions section (OpenAI embeddings sanity probe - a single semantic_search must return results not 429, the exact 21-14-30Z poison condition; Cloud SQL reachable via cloud-sql-proxy port 5433 for instance `mlops-491820:us-central1:mlops--city-concierge` or local Postgres with a `SELECT 1` check; all four keys live: OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY). An ordered Steps section: (1) `make probe-providers` (D-10-14 mandatory pre-matrix), (2) `make snapshot-baselines` (D-11-09), (3) `APP_ENV=eval make eval-matrix RUNS=5`, (4) `APP_ENV=eval make eval-matrix-refinement RUNS=5`, (5) `make write-baselines SUMMARY=eval_reports/{ts}/summary.json RUNS=5` for each matrix summary, (6) `make eval-gates-check-baselines` (same code path CI runs), (7) commit baselines + snapshots together (small focused commit; live-regen results committed separately from code per feedback_small_focused_commits). A Failure branches section: gemini cells erroring -> write_baselines refuses them, record in `_DEFERRED_BASELINE_CELLS`, retry per D-11-11; gated families (gpt-4o-mini, anthropic) erroring BLOCKS - rerun until clean; gpt-4o-mini commit_rate < 0.8 on honest regen -> STOP and investigate (real anchor regression, not noise). State explicitly that late_night_closure_cascade is NOT regenerated (D-10-09/10 standing). Note per-family thinking/temperature policies stay as-shipped in app/llm_factory.py (no tuning, D-11-10).
  </action>
  <verify>
    <automated>test -f docs/baseline_regen.md && grep -q "probe-providers" docs/baseline_regen.md && grep -q "snapshot-baselines" docs/baseline_regen.md && grep -q "RUNS=5" docs/baseline_regen.md && grep -q "eval-gates-check-baselines" docs/baseline_regen.md && grep -q "0.8" docs/baseline_regen.md && grep -q "late_night" docs/baseline_regen.md && echo RUNBOOK_OK</automated>
  </verify>
  <acceptance_criteria>
    - `docs/baseline_regen.md` exists and contains a Preconditions section naming the embeddings sanity probe and all four provider key env vars
    - The Steps section cites `make probe-providers`, `make snapshot-baselines`, `make eval-matrix RUNS=5`, `make eval-matrix-refinement RUNS=5`, `make write-baselines`, and `make eval-gates-check-baselines` in order
    - A Failure branches section documents the gemini deferral (D-11-11) and the gpt-4o-mini < 0.8 STOP condition
    - The doc states late_night is NOT regenerated (D-10-09/10 standing)
  </acceptance_criteria>
  <done>The runbook is a complete, executable ordered procedure citing real Makefile targets and the failure branches.</done>
</task>

<task type="checkpoint:human-action" gate="blocking">
  <name>Task: Verify live-infra preconditions before regen</name>
  <what-built>The runbook, the write_baselines tool (11-05), the baselines-mode gate (11-06), and all Wave-0 measurement fixes are in place. The remaining work requires live infrastructure that Claude cannot provision: Cloud SQL reachability, all four provider API keys, and OpenAI embeddings quota.</what-built>
  <how-to-verify>
    Before approving, the developer must confirm the runbook preconditions (docs/baseline_regen.md):
    1. Start cloud-sql-proxy on port 5433 (instance mlops-491820:us-central1:mlops--city-concierge) OR bring up local Docker Postgres; verify `psql $DATABASE_URL -c "SELECT 1"`.
    2. Export OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY.
    3. Run the embeddings sanity probe (a single semantic_search) and confirm it returns results, NOT a 429 - this is the exact condition that poisoned the 2026-06-05T21-14-30Z run.
  </how-to-verify>
  <resume-signal>Type "preconditions met" once the DB is reachable, all four keys are exported, and the embeddings probe returns results. Type the blocker if any precondition fails.</resume-signal>
</task>

<task type="auto">
  <name>Task 2: Snapshot, run both matrices at n=5, write baselines, re-ratify gates, update parity</name>
  <files>configs/eval_baselines/_snapshots/omakase_mission_open_ended.pre-phase11.json, configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase11.json, configs/eval_baselines/_snapshots/late_night_closure_cascade.pre-phase11.json, configs/eval_baselines/omakase_mission_open_ended.json, configs/eval_baselines/refinement_cheaper.json, configs/eval_gates.yaml, docs/eval_gates.md, tests/unit/test_eval_matrix.py</files>
  <read_first>
    - docs/baseline_regen.md - the runbook authored in Task 1; follow its ordered steps exactly.
    - configs/eval_gates.yaml - read the full file. The `anthropic/claude-sonnet-4-6` entry is `status: provisional-n1` (lines 43-50); the `openai/gpt-5-mini` entry is `status: aspirational` (lines 34-41); deepseek/gemini are `logged`. The NOTE block (lines 17-19) says committed_itinerary_rate is "not yet wired" - that caveat is now obsolete after D-11-02.
    - configs/eval_baselines/refinement_cheaper.json - read the existing anthropic cell `_observations` (the n=1 SHIPPED-WITH-GAP record the n=5 re-measure replaces).
    - tests/unit/test_eval_matrix.py - read `_DEFERRED_BASELINE_CELLS` (set to the three cross-model providers by 11-04); the three landed cells are removed here after regen.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md - D-11-20 (re-ratification rules: anthropic provisional-n1 -> active with value re-derived under the D-10-07 "absorb one stochastic miss at n=5" rule; gpt-5-mini STAYS aspirational; deepseek/gemini stay logged; advisory medians stay advisory; every change gets a D-11 rationale line) and specifics (expected post-regen picture; gpt-4o-mini commit_rate < 0.8 = STOP).
  </read_first>
  <action>
    Execute the runbook against live infra. (1) `make snapshot-baselines` - verify the three `_snapshots/*.pre-phase11.json` files now exist (D-11-09). (2) `APP_ENV=eval make eval-matrix RUNS=5` and `APP_ENV=eval make eval-matrix-refinement RUNS=5`. If any GATED-family cell (gpt-4o-mini, anthropic) errors, rerun until clean per D-11-11; if gemini cells error, leave them deferred. If gpt-4o-mini's committed_itinerary_rate median is below 0.8, STOP, do not write baselines, and report the regression (do not proceed). (3) `make write-baselines SUMMARY=<omakase summary> RUNS=5` and again for the refinement summary - the tool refuses partial/quarantined cells. Verify `committed_itinerary_rate` is present in each regenerated scorers block (D-11-02 proof). Carry the anthropic n=5 re-measure forward replacing the n=1 _observations; record the gemini-first-ever n=5 (or its deferral). (4) Re-ratify `configs/eval_gates.yaml` (D-11-20): change `anthropic/claude-sonnet-4-6` `status: provisional-n1 -> active` with its value re-derived from the n=5 data under the D-10-07 absorb-one-miss rule, and add a `D-11-20` rationale line; keep `openai/gpt-5-mini` aspirational (add a D-11-20 rationale noting it remains the v2.2 target); keep deepseek/gemini logged; remove the now-obsolete "not yet wired" caveat from the NOTE block (D-11-02 landed). Update `docs/eval_gates.md` prose only if it references the obsolete not-evaluable state - never duplicate numeric gate values (they live only in the YAML). (5) `make eval-gates-check-baselines` against the regenerated baselines - confirm it passes (only the known gpt-5-mini aspirational miss may be reported, non-blocking). (6) In `tests/unit/test_eval_matrix.py`, remove from `_DEFERRED_BASELINE_CELLS["eval_matrix.yaml"]` exactly the cross-model providers whose live baseline cells were written (gpt-5-mini, claude-sonnet-4-6, deepseek-reasoner); if gemini or any provider remains deferred, document it with a matching matrix-YAML comment per D-11-11; run the parity test to confirm it passes atomically. Commit baselines + snapshots together, separate from the gate-config / test changes per feedback_small_focused_commits.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -k "baseline_provider_cells_match" -v && make eval-gates-check-baselines && poetry run python -c "import json; d=json.load(open('configs/eval_baselines/omakase_mission_open_ended.json')); assert any('committed_itinerary_rate' in c.get('scorers',{}) for c in d['providers'].values()), 'committed_itinerary_rate missing from regenerated scorers'; print('REGEN_OK')"</automated>
  </verify>
  <acceptance_criteria>
    - All three `configs/eval_baselines/_snapshots/*.pre-phase11.json` files exist on disk
    - `configs/eval_baselines/omakase_mission_open_ended.json` and `refinement_cheaper.json` contain `committed_itinerary_rate` in at least one provider's `scorers` block (D-11-02 threaded through to committed baselines)
    - Each regenerated provider cell carries `generated_by: scripts/write_baselines.py` and a fresh `generated_at`
    - `configs/eval_gates.yaml` `anthropic/claude-sonnet-4-6` `status` is `active` with a `D-11-20` rationale line; `openai/gpt-5-mini` stays `aspirational`
    - `make eval-gates-check-baselines` exits 0 (or reports only the gpt-5-mini aspirational miss, non-blocking)
    - `_DEFERRED_BASELINE_CELLS["eval_matrix.yaml"]` no longer lists any cross-model provider whose baseline cell was written; the parity test exits 0
    - The late_night baseline JSON is unchanged from its snapshot (not regenerated)
  </acceptance_criteria>
  <done>Baselines regenerated honestly at n=5 with committed_itinerary_rate threaded; gates re-ratified with D-11 rationale; parity test atomically consistent; baselines-mode gate passes.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task: Verify regenerated baselines are honest and the anchor holds</name>
  <what-built>The regenerated baselines, re-ratified gates, runbook, and parity-test update.</what-built>
  <how-to-verify>
    1. Inspect `configs/eval_baselines/omakase_mission_open_ended.json` and `refinement_cheaper.json`: confirm honest n=5 numbers (not all-1.0 fail-open), `committed_itinerary_rate` present, fresh `generated_at`/`generated_by`.
    2. Confirm `gpt-4o-mini` committed_itinerary_rate median >= 0.8 (anchor not regressed). If below 0.8, this is a real regression - do NOT ship; reopen investigation.
    3. Confirm the anthropic n=5 cell replaced the n=1 _observations record; review the gemini cell or its documented deferral.
    4. Run `make eval-gates-check-baselines` and `make test` locally; confirm both pass (gpt-5-mini aspirational miss is acceptable, non-blocking).
    5. Confirm the three `_snapshots/*.pre-phase11.json` preserve the old fail-open numbers for audit.
  </how-to-verify>
  <resume-signal>Type "approved" if the regenerated baselines are honest and the anchor holds at >= 0.8, or describe the regression / anomaly to investigate.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| live provider/embeddings APIs -> regenerated baseline JSON | a quota/DB outage during regen can poison every cell (the 21-14-30Z disaster); the embeddings probe + error-status semantics guard this |
| provider API keys (env) -> committed baseline + snapshots | keys must never appear in committed baseline JSONs, snapshots, or the runbook |
| regenerated baseline -> CI gate verdict | the committed n=5 baselines become the empirical record every future PR is gated against |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-20 | Tampering | quota/DB-poisoned regen | mitigate | runbook embeddings sanity probe + DB reachability precondition + EVAL-01 error-status semantics (no fail-open 1.0) prevent baking poisoned cells |
| T-11-21 | Information disclosure | provider keys in committed artifacts | mitigate | baselines carry only numeric scorer stats + provenance stamps; runbook names env-var sources only, never key values; snapshots are copies of numeric baselines |
| T-11-22 | Tampering | silent anchor regression | mitigate | human-verify checkpoint + runbook STOP rule: gpt-4o-mini < 0.8 halts before commit; baselines-mode gate fires on the committed regression |
| T-11-23 | Repudiation | overwriting auditable history | mitigate | D-11-09 pre-phase11 snapshots preserve the fail-open-saturated v2.0 numbers before overwrite |
| T-11-08-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH section Package Legitimacy Audit: none); regen uses the existing pinned toolchain |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py -k "baseline_provider_cells_match" -v` passes (parity atomic).
- `make eval-gates-check-baselines` passes against the regenerated committed baselines (gpt-5-mini aspirational miss acceptable).
- `make test` full suite passes.
- The three `_snapshots/*.pre-phase11.json` files exist and preserve the pre-regen numbers.
- No provider key strings appear in any committed file (`grep -rIl "sk-\|AIzaSy" configs/eval_baselines docs/baseline_regen.md` returns nothing).
</verification>

<success_criteria>
- BASE-01 baselines regenerated honestly at n=5 under DB-up with Wave-0 fixes + Phase-9 adapters; fail-open v2.0 baselines replaced and snapshotted.
- BASE-02 four-provider results with n_errored == 0 on gated/logged cells (gemini deferral allowed).
- BASE-03 gates re-ratified from fresh data with D-11 rationale; baselines-mode gate green.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-08-SUMMARY.md` when done.
</output>
