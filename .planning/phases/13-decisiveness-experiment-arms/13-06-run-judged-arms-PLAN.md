---
phase: 13-decisiveness-experiment-arms
plan: 06
type: execute
wave: 3
depends_on: ["13-03", "13-04", "13-05"]
files_modified:
  - docs/decisiveness_arm_verdicts.md
autonomous: false
requirements: [DEC-05]
must_haves:
  truths:
    - "docs/decisiveness_arm_verdicts.md exists with one section per arm (A1, A2, A3, A4-conditional) carrying the D-13-09 field set: flag config, matrix config + run-dir paths, per-model pooled commit rate with per-scenario breakdown, model-initiated vs forced split (A2), latency decomposition deltas (A3), falsifier verdict line + exit code"
    - "Each judged arm (A1, A2, A3) ran smoke n=1 BEFORE the full n=5 spend, and the smoke run JSON's arm_flags self-description was verified to match the intended arm config (D-13-02, D-13-05)"
    - "Each full arm run is 3 models x 2 scenarios x n=5, temp=1.0, sequential, graded by make eval-falsifier-arm against the Phase-12 comparison floor; A3 is judged on latency reduction + zero scorer regression, NOT commit rate (D-13-01)"
    - "The A2 section reports the model-initiated vs forced split per model explicitly, and flags any gpt-4o-mini anchor behavior change as a red flag (D-13-04)"
    - "If quota/budget dies mid-arm, the partial result is recorded honestly in the verdict doc and the arm stops — no partial-cell baselines are ever written (D-13-02, D-11-14)"
  artifacts:
    - path: "docs/decisiveness_arm_verdicts.md"
      provides: "Per-arm verdict sections with run-dir paths, per-model numbers, splits, and falsifier exit codes for A1/A2/A3"
      min_lines: 60
  key_links:
    - from: "docs/decisiveness_arm_verdicts.md"
      to: "eval_reports/{run-dir}/summary.json + per-run JSONs"
      via: "recorded run-dir paths + falsifier output pasted per arm"
      pattern: "eval_reports"
    - from: "docs/decisiveness_arm_verdicts.md A2 section"
      to: "scripts/eval_falsifier.py _commit_split_from_run_dir output"
      via: "model-initiated vs forced split line pasted from the falsifier print"
      pattern: "model-initiated"
---

<objective>
Execute the three judged experiment arms live (A1 viability contract + critique
recalibration, A2 forced-commit-at-step-6, A3 parallel tool execution) against
the Phase-12 comparison floor, smoke n=1 before every full n=5 spend, and record
each arm's verdict section in `docs/decisiveness_arm_verdicts.md` — the document
Phase 14's conditional entry gate and Phase 15's promotion both read.

Purpose: DEC-05 requires per-arm n=5 commit-rate numbers for gpt-5-mini,
deepseek-reasoner, and the gpt-4o-mini anchor, judged by the INST-05 falsifier.
D-13-01 fixes the arm structure (three judged arms; A3 judged on latency, not
commit rate; hard cap of 4 full live matrix runs total — this plan consumes 3).
D-13-02 fixes the run protocol (2-scenario universe, n=5, temp=1.0, sequential,
smoke-first, no billing top-ups). D-13-09 fixes the verdict doc field set.

This plan is NOT autonomous: live runs cost real API spend and require the
operator's keys + go-ahead; arm ordering and partial-failure handling need
human judgment.

Output: A1/A2/A3 sections of docs/decisiveness_arm_verdicts.md filled with real
numbers and run-dir paths. The A4 conditional decision and the closing verdict
are plan 13-07.
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
@.planning/phases/13-decisiveness-experiment-arms/13-04-SUMMARY.md
@.planning/phases/13-decisiveness-experiment-arms/13-05-SUMMARY.md
@docs/decisiveness_dec03_decision.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Scaffold docs/decisiveness_arm_verdicts.md with the D-13-09 field set</name>
  <files>docs/decisiveness_arm_verdicts.md</files>
  <read_first>
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-09 (the per-arm field list this doc must carry) and D-13-01 (arm definitions A1-A4, A3's latency-not-commit-rate judging, A4 conditionality)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "docs/decisiveness_arm_verdicts.md (new)" (the per-arm section template: flag config, matrix config, run dirs, per-model results table with pooled + per-scenario columns + falsifier verdict, exit code, closing verdict)
    - docs/eval_gates.md and docs/baseline_regen.md (the doc-as-contract style to match)
    - docs/decisiveness_dec03_decision.md (the A1 first-run decision: LOW_SIMILARITY_THRESHOLD_OVERRIDE stays unset on the first A1 run)
  </read_first>
  <action>
    Create `docs/decisiveness_arm_verdicts.md` with: a header explaining the doc's role (DEC-05 record; Phase-14 conditional entry gate input — Phase 14 executes ONLY if no arm clears the INST-05 bar; Phase-15 promotion input), the INST-05 falsifier definition (gpt-5-mini pooled committed_itinerary_rate >= 0.6 at n=5 AND gpt-4o-mini holds >= its honest baseline), and the run-budget contract (hard cap 4 full live matrix runs per D-13-01; smoke n=1 before each; no billing top-ups; partial results recorded honestly, never written as baselines per D-11-14). Then four arm sections following the PATTERNS.md template: A1 (flag config `VIABILITY_CONTRACT_ENABLED=1`, override unset per docs/decisiveness_dec03_decision.md), A2 (flag config `FORCED_COMMIT_STEP=6`; its per-model table includes a model-initiated vs forced split column in the D-13-04 format `commit_rate 0.8 (model-initiated 0.4, forced 0.4)`; note the anchor red-flag rule: gpt-4o-mini commits before step 6 on its own, so ANY anchor behavior change fails A2), A3 (flag config `PARALLEL_TOOL_EXECUTION_ENABLED=1`; judged on measured latency reduction from INST-04 step_telemetry deltas + zero scorer regression — NOT commit rate, per D-13-01), and A4 (marked CONDITIONAL: run only if neither A1 nor A2 clears alone but both show positive signal; decision recorded in plan 13-07). Every section gets placeholders for: matrix config path (configs/eval_matrix_arm.yaml), run-dir paths (smoke + full), per-model pooled commit rate with omakase/refinement_cheaper breakdown, falsifier verdict line + exit code. End with an empty "Closing verdict" section (filled by 13-07).
  </action>
  <verify>
    <automated>test -f docs/decisiveness_arm_verdicts.md && grep -q "VIABILITY_CONTRACT_ENABLED" docs/decisiveness_arm_verdicts.md && grep -q "FORCED_COMMIT_STEP" docs/decisiveness_arm_verdicts.md && grep -q "PARALLEL_TOOL_EXECUTION_ENABLED" docs/decisiveness_arm_verdicts.md && grep -q "model-initiated" docs/decisiveness_arm_verdicts.md && grep -qi "conditional" docs/decisiveness_arm_verdicts.md && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - The doc has four arm sections (A1/A2/A3/A4) plus a Closing verdict section.
    - The A2 section template includes the model-initiated vs forced split format and the anchor red-flag rule.
    - The A3 section states it is judged on latency + zero scorer regression, not commit rate.
    - The A4 section states the D-13-01 conditionality and that the hard cap is 4 full live matrix runs.
    - The header names the INST-05 bar (>= 0.6 gpt-5-mini pooled, anchor non-regression) and the no-partial-baselines rule.
  </acceptance_criteria>
  <done>The verdict document skeleton exists with every D-13-09 field placeholdered, ready to receive real run numbers.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Approve live spend and confirm run environment</name>
  <what-built>Plans 13-01..13-05 shipped all three arms behind default-off env flags, the arm matrix config (3 models x 2 scenarios), and the falsifier arm mode. The verdict doc skeleton from Task 1 is ready. The next task fans out LIVE API runs: 3 arms x (smoke n=1 + full n=5) x 3 models x 2 scenarios — real OpenAI + DeepSeek spend.</what-built>
  <how-to-verify>
    1. Confirm API keys are available in the shell environment: OPENAI_API_KEY and DEEPSEEK_API_KEY set, APP_ENV=eval exported, and the database is reachable (eval retrieval needs the embeddings DB — run `make db-up` if running locally).
    2. Confirm budget: roughly 3 full matrix runs (each 3 models x 2 scenarios x 5 runs = 30 agent episodes) plus 3 smoke runs (6 episodes each). If budget is constrained, say which arms to prioritize (suggested order: A1, A2, A3 per D-13-01).
    3. Confirm the arm ordering or adjust it.
  </how-to-verify>
  <action>Present the spend summary and environment checklist above, then PAUSE and wait for the operator's signal. Do not start any live run before approval.</action>
  <resume-signal>Type "approved" (optionally with an arm order, e.g. "approved, run A2 first") or describe what is missing.</resume-signal>
</task>

<task type="auto">
  <name>Task 3: Run A1, A2, A3 — smoke n=1 first, then full n=5 — and record each verdict section</name>
  <files>docs/decisiveness_arm_verdicts.md</files>
  <read_first>
    - docs/decisiveness_arm_verdicts.md (the skeleton from Task 1 — fill its placeholders, do not restructure)
    - Makefile eval-matrix-arm + eval-falsifier-arm targets (from plan 13-05 — the exact run/grade commands)
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-02 (run protocol: sequential, temp=1.0, smoke-first, stop honestly on budget death) and "Specific Ideas" (paste falsifier per-scenario breakdowns directly; arm run dirs must satisfy the zero-overlap guard)
  </read_first>
  <action>
    For each arm IN THE APPROVED ORDER (default A1 → A2 → A3), execute the D-13-02 protocol. Per arm: (1) SMOKE — export the arm's flag(s) (A1: `VIABILITY_CONTRACT_ENABLED=1`, override unset; A2: `FORCED_COMMIT_STEP=6`; A3: `PARALLEL_TOOL_EXECUTION_ENABLED=1`) and run `make eval-matrix-arm RUNS=1`; open one run JSON from the new run dir and VERIFY `deterministic.arm_flags` self-describes the intended arm config exactly (D-13-05) — if it does not, STOP and report (do not proceed to full spend with a mis-flagged graph). (2) FULL — run `make eval-matrix-arm RUNS=5` with the same exports (temp=1.0 is the harness default — confirm the run metadata records it). (3) GRADE — run `make eval-falsifier-arm RUN_DIR=<full-run-dir>` and capture the full output including exit code (`echo $?`). (4) RECORD — fill the arm's section in docs/decisiveness_arm_verdicts.md: smoke + full run-dir paths, the falsifier's per-scenario breakdown pasted directly (do not recompute), per-model pooled commit rate, exit code, and the verdict line. A2 additionally: paste the model-initiated vs forced split per model, and compare the gpt-4o-mini anchor numbers against its baseline — flag ANY anchor change as a red flag per D-13-04(c). A3 additionally: compute the gpt-4o-mini latency delta from INST-04 step_telemetry (sum of tool_exec_seconds per run, arm vs the Phase-12 comparison-floor run dirs) and record scorer non-regression — A3's verdict is latency + zero scorer regression, NOT commit rate (D-13-01). HONESTY RULES: runs are sequential (D-11-14); if quota/budget dies mid-arm, record the partial result in the arm's section labeled PARTIAL with what completed, and stop — never write baselines from these runs and never re-run cells to cherry-pick. Unset each arm's env flags before the next arm's smoke (one arm's flags must never leak into another's run).
  </action>
  <verify>
    <automated>grep -c "eval_reports/" docs/decisiveness_arm_verdicts.md | awk '{exit ($1>=3)?0:1}' && grep -q "exit code" docs/decisiveness_arm_verdicts.md && echo OK</automated>
    <human-check>Read the three filled arm sections: run-dir paths resolve, the A2 split is present per model, the A3 latency delta is present, and every number traces to a falsifier print or run JSON (no hand-derived figures).</human-check>
  </verify>
  <acceptance_criteria>
    - Each of A1/A2/A3 has BOTH a smoke run-dir and a full run-dir recorded, with the smoke arm_flags verification noted.
    - Each arm section carries the falsifier per-scenario breakdown, pooled per-model commit rates (gpt-5-mini, deepseek-reasoner, gpt-4o-mini), and the falsifier exit code (0/1/2).
    - The A2 section reports the model-initiated vs forced split per model and an explicit anchor red-flag assessment.
    - The A3 section reports the gpt-4o-mini latency delta vs the comparison floor and scorer non-regression — its verdict line does not use commit rate.
    - No baselines were written (`git status configs/eval_baselines/` is clean).
    - Any incomplete arm is labeled PARTIAL with an honest account, not omitted.
  </acceptance_criteria>
  <done>All three judged arms have honest, falsifier-graded n=5 verdict sections (or honestly-labeled partials), consuming exactly 3 of the 4 capped live matrix runs.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| operator env (API keys, arm flags) → live eval runs | Real spend + arm selection controlled by exported env vars |
| run artifacts → verdict document | Recorded numbers must trace to falsifier output, not manual derivation |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-06-01 | Spoofing | mis-flagged run graded as the wrong arm | mitigate | smoke n=1 arm_flags self-description check is a hard gate before every full spend (D-13-05); flags unset between arms |
| T-13-06-02 | Repudiation | verdict numbers not traceable to artifacts | mitigate | run-dir paths recorded per arm; falsifier per-scenario breakdowns pasted verbatim; exit codes captured |
| T-13-06-03 | Tampering | partial/cherry-picked results inflating an arm | mitigate | sequential runs, PARTIAL labeling rule, no re-runs to cherry-pick, no baselines written (D-11-14 guard: write_baselines.py refuses partial cells anyway) |
| T-13-06-04 | Information Disclosure | API keys leaking into the verdict doc or run JSONs | mitigate | record only run-dir paths and flag names in the doc; keys live in the operator's shell env, never in committed files |
| T-13-06-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A |
</threat_model>

<verification>
- docs/decisiveness_arm_verdicts.md has filled A1/A2/A3 sections with run-dir paths, per-model numbers, splits (A2), latency deltas (A3), and falsifier exit codes.
- `git status configs/eval_baselines/` is clean (no baselines written from arm runs).
- Exactly 3 full live matrix runs consumed (A4 budget preserved for plan 13-07).
</verification>

<success_criteria>
- Three judged arms ran smoke-first at n=5 temp=1.0 against the Phase-12 comparison floor and are honestly recorded per the D-13-09 field set (roadmap criterion 5 inputs complete for A1/A2/A3).
- A2's forced-commit honesty split is explicit (roadmap criterion 2 + D-13-04).
- A3's measurable gpt-4o-mini latency reduction at n=5 is recorded from run JSON telemetry (roadmap criterion 3).
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-06-SUMMARY.md` when done.
</output>
