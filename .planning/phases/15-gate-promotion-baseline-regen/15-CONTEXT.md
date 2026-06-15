# Phase 15: Gate Promotion + Baseline Regen - Context

**Gathered:** 2026-06-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 15 **closes the v2.2 Reasoning-Model Decisiveness milestone**. Its scope was
already ratified by the user at the D-14-08 checkpoint (recorded in
`docs/replay_arm_verdicts.md`, "USER CHECKPOINT RESOLVED 2026-06-12"):

> Phase 15 = **A2 retest on the fixed synthesizer** + **refinement_cheaper root-cause
> analysis** + **gate promotion / baseline regen**. gpt-4o-mini anchor **ratified**.
> ARCH-FUT-01 is **NOT executed** — filed as tracked technical debt with the Phases 13–14
> evidence chain as its trigger criteria.

The milestone produced an **honest null result**: across six-plus interventions (DEC arms
A1–A4, REPLAY arms R1–R3 + valve) no arm cleared the INST-05 falsifier bar (gpt-5-mini
commit rate ≥ 0.6 at n=5 with no anchor regression). The best signal was A2
(`FORCED_COMMIT_STEP=6`) at 0.500 pooled — but that 0.500 was **entirely model-initiated
commits**; the forced-commit synthesizer was broken (CR-01) until Phase 13-08, so the
forced path **never actually fired** in the A2 measurement.

Phase 15 therefore delivers the three PROMO requirements **plus the two front-matter items
the user added at the checkpoint**:

1. **A2 retest** — measure whether the *now-working* forced-commit path lifts
   `refinement_cheaper` above 0.000 (PROMO-adjacent experiment; the milestone's highest-ROI
   remaining lever).
2. **refinement_cheaper root-cause** — diagnose WHY gpt-5-mini commits at 1.000 on omakase
   but 0.000 on refinement across every tested arm.
3. **PROMO-01** — regenerate honest n=5 baselines for the runnable matrix cells via
   `write_baselines.py`.
4. **PROMO-02** — promote reasoning-model gates from logged-not-gated to enforced **only
   where measured data earns it**; leave the rest logged with a Phase-15 note.
5. **PROMO-03** — latency report decomposed from INST-04 telemetry vs the ~30s/turn prod
   budget, with the prod-driver recommendation (ratify gpt-4o-mini) written down.

**Scope anchor (locked, do not re-open):** gpt-4o-mini stays the prod anchor.
ARCH-FUT-01 is NOT executed this phase. No prod-default behavior flip unless explicitly
re-decided (see D-15-07). No new scorers, no provider-shopping, no LangGraph replacement
(milestone anti-scope).

</domain>

<decisions>
## Implementation Decisions

The user discussed all four HOW-to-execute gray areas (the WHAT was already locked at the
D-14-08 checkpoint) and selected the recommended option for each. D-15-01..09 are ratified
decisions; the planner may refine exact env-var names, file paths, and run-config names but
must preserve the substance.

### A2 retest role, run shape & recording
- **D-15-01: The A2 retest is a measured input, never a hard gate on Phase 15.** Phase 15
  always completes regardless of the A2 outcome. Promotion uses whichever config the data
  earns — A2-if-it-clears-0.6, else the flag-off prod default. This preserves the honest-null
  discipline from Phases 13/14: a negative result is a valid result, never a blocker. It also
  resolves the ROADMAP's "winning arm" framing — there is no winning arm; "winning config"
  means the prod default unless the A2 retest earns better.
- **D-15-02: Two separate live matrix runs.**
  - **Run #1 — A2 retest (experiment):** `FORCED_COMMIT_STEP=6`, 3 models (openai/gpt-4o-mini
    anchor, openai/gpt-5-mini, deepseek/deepseek-reasoner) × 2 scenarios
    (omakase_mission_open_ended + refinement_cheaper) × n=5, temp=1.0, sequential, via
    `configs/eval_matrix_arm.yaml`. Smoke n=1 with `arm_flags` self-description verification
    before the full n=5 spend (Phase-13/14 precedent). Feeds the verdict delta. **NOT written
    as baselines** (experiment flag set — violates the baselines=prod-config invariant).
  - **Run #2 — flag-off prod-config (baseline source):** same 3 models × 2 scenarios × n=5,
    all experiment flags UNSET — the honest baseline source for `write_baselines.py`.
  - **Anchor non-regression is mandatory on both runs** (gpt-4o-mini must hold its committed
    rate; A3's anchor regression is the cautionary precedent).
  - Total live spend stays within the prior ≤4-run cap; no billing top-ups.
- **D-15-03: New `docs/promotion_decision.md` is the single milestone-closing record.** It
  holds the A2 retest delta, the root-cause findings, the gate-promotion decisions, the
  baseline regen provenance, and the PROMO-03 latency report. It **cross-links** the two
  closed verdict docs (`docs/decisiveness_arm_verdicts.md`, `docs/replay_arm_verdicts.md`)
  as immutable inputs — it does NOT append to them. Mirrors the Phase-14 "closed records stay
  immutable, new phase = new doc" discipline. Becomes the v2.2 milestone-audit anchor.

### Baseline regen (PROMO-01)
- **D-15-04: Baselines = runnable cells only.** openai/gpt-4o-mini + openai/gpt-5-mini +
  deepseek/deepseek-reasoner across the 2 live scenarios. anthropic and gemini stay
  deferred-logged (D-12-09 billing/quota decision, locked — no top-up). late_night_closure_cascade
  stays quarantined. PROMO-01's "all matrix cells" reads as "all *eligible* cells" — which is
  exactly what `scripts/write_baselines.py` already enforces (refuses partial/quarantined,
  exit 1) and what `docs/baseline_regen.md` documents.
- **D-15-05: Regenerate baselines LAST.** After all Phase-15 code changes land (including a
  shipped root-cause fix, if any — see D-15-08). The flag-off baseline run (Run #2) executes
  against the FINAL merged code; `write_baselines.py` runs last so the committed
  `configs/eval_baselines/*.json` reflect merged behavior and the staleness gate
  (`check_baselines_fresh.py`) passes on the final commit. If no agent-code fix ships,
  baselines reflect current main. Standard `docs/baseline_regen.md` discipline.

### Gate promotion (PROMO-02)
- **D-15-06: Enforce only what is both stable AND measured.**
  - **gpt-4o-mini:** keeps its enforced 0.8 hard gate, re-ratified against the fresh n=5
    (Run #2) median.
  - **gpt-5-mini:** promoted to ENFORCED **only if the A2 retest clears 0.6** (subject to
    D-15-07 provenance). Otherwise it stays logged-not-gated with an updated Phase-15
    rationale citing the measurement. (This is the likely outcome — A2 plateaued at 0.500
    before, on model-initiated commits.)
  - **deepseek-reasoner / anthropic / gemini:** stay logged-not-gated (data doesn't earn
    enforcement, or unmeasured/deferred). A 0.0-floor "enforced" gate is meaningless and
    would misrepresent a known-failing config as passing.
  - Entries that fall short retain `status: logged` (or `aspirational`) with an explicit
    Phase-15 note in the `rationale:` field of `configs/eval_gates.yaml`.
- **D-15-07: An enforced hard gate only certifies the config CI actually runs (flag-off prod
  default).** If the A2 retest clears 0.6, that number came from a `FORCED_COMMIT_STEP=6`
  experiment config — **not** the prod default CI exercises. So: the A2 number is the
  *unlock rationale*, NOT the gate-value source. gpt-5-mini is promoted to enforced only if
  the **flag-off prod-config** rate (Run #2) also supports the floor; the eval_gates.yaml
  rationale records that the forced path is the mechanism and notes that **prod would need
  `FORCED_COMMIT_STEP` enabled to realize it**. Gate value and producing config must match —
  never set a hard gate from a config CI doesn't exercise (it would fail spuriously or pass
  meaninglessly). **A prod-default flip to `FORCED_COMMIT_STEP=6` is a SEPARATE decision** —
  flagged in promotion_decision.md and **likely deferred** (anchor is ratified as-is; flipping
  the default changes the anchor's behavior too).

### Root-cause analysis depth (refinement_cheaper = 0.000)
- **D-15-08: Diagnostic-only by default; ship a fix ONLY if it is a one-flag / one-line
  low-risk change.** The deliverable is a written root-cause section in
  `docs/promotion_decision.md` grounded in the **existing** Phase-13/14 run JSONs
  (`step_telemetry`, `viable_candidates_per_step`, `rule8_met_per_step`,
  `first_commit_call_step`, `forced_commit_step`) — **zero new diagnostic live runs**. The
  diagnostic question: does the refinement prompt structure prevent viable candidates from
  appearing at step 6, and are all slots truly viable per the gate's `all_slots_viable`
  condition? If the cause is a trivial, low-risk, clearly-isolated fix (e.g. the gate's
  viability condition is computed wrong, or `FORCED_COMMIT_STEP=6` fires after the model has
  already given up), ship it behind the A2 retest. **Anything larger** (new prompt strategy,
  new mechanism, scenario redesign) is documented as a finding and **deferred with a
  tracked-debt note** — it does NOT re-open the ARCH-FUT-01 deferral the user just ratified.
- **D-15-09: Fixed pipeline order (correctness constraint, not a planner preference).**
  1. Root-cause diagnostic from existing run JSONs (zero spend).
  2. Optional trivial fix lands now (D-15-08), else document + defer.
  3. Live runs: A2 retest (Run #1, experiment) + flag-off (Run #2, baseline source) against
     the **post-fix** code; anchor non-regression checked on both.
  4. Gate-promotion decisions written (D-15-06/07).
  5. `write_baselines.py` runs LAST against final code (D-15-05).
  6. PROMO-03 latency report aggregated from run JSONs.
  7. `docs/promotion_decision.md` finalizes; milestone-close bookkeeping
     (ROADMAP/REQUIREMENTS/STATE status flips straight to main).
  The diagnose-before-retest ordering is mandatory so the A2 retest measures FINAL code; it
  prevents a wasted live run against the ≤4-run cap.

### Claude's Discretion
The user selected the recommended option for every question and locked the substance above.
The planner has discretion over: exact env-var / run-config / file names; how to structure
the latency-report aggregation script vs inline computation; the precise smoke-verification
shape; and how to split the work across plans. The planner must preserve: the two-separate-runs
discipline (experiment vs baseline), anchor non-regression as mandatory, the diagnose→fix→
retest→promote→baseline-last→latency ordering, the new (not-appended) promotion_decision.md,
the runnable-cells-only baseline set, gate-provenance honesty (D-15-07), and the
diagnostic-only-unless-trivial fix boundary.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Ratified scope + plateau record (settled — do not re-derive)
- `docs/replay_arm_verdicts.md` — Phase-14 closing verdict; ARCH-FUT-01 Evaluation section;
  the explicit "USER CHECKPOINT RESOLVED (2026-06-12)" block that **defines Phase 15 scope**;
  the recommended-inputs list (A2 run dir, blocking pattern). **Closed/immutable — cross-link,
  do not append.**
- `docs/decisiveness_arm_verdicts.md` — Phase-13 DEC record: INST-05 falsifier definition,
  A1 0.000 / A2 0.500 / A3 anchor-regression numbers, the CR-01 forced-synthesizer-broken note
  that makes the A2 retest necessary. **Closed/immutable — cross-link, do not append.**
- `.planning/ROADMAP.md` — Phase 15 "Gate Promotion + Baseline Regen" success criteria 1–3
  (the literal PROMO-01/02/03 acceptance language).
- `.planning/REQUIREMENTS.md` — PROMO-01/02/03 definitions; D-12-09 deferral of anthropic +
  gemini n=5; D-11-14 partial/quarantined refusal rule.
- `.planning/PROJECT.md` — milestone goal, Decision 3 (gpt-4o-mini anchor + ~30s/turn budget),
  milestone anti-scope (no LangGraph replacement, no new scorers, no provider-shopping).
- `.planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md` — D-14-01..08 (run
  budget, honesty contract, verdict-doc-not-appended pattern, ≤4-run cap, user-checkpoint).
- `.planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md` — D-13-01..09 (arm run
  budget reused verbatim; A2 = FORCED_COMMIT_STEP=6; CR-01 synthesizer fix lineage).

### Gate promotion machinery (the PROMO-02 targets)
- `configs/eval_gates.yaml` — the file to edit: per-family `status` / `hard` / `advisory` /
  `rationale` schema (D-10-08); current gpt-4o-mini active@0.8, gpt-5-mini aspirational@0.6
  (FAILS), deepseek/anthropic/gemini logged, late_night quarantined.
- `scripts/check_eval_gates.py` — enforces the `hard` gates (`make eval-gates-check`);
  defines what "enforced" means operationally.
- `docs/eval_gates.md` — gate semantics; links to the yaml, never duplicates numbers.

### Baseline regen machinery (the PROMO-01 targets)
- `scripts/write_baselines.py` — the ONLY way baselines are written (D-11-14); refusal rules
  (exit 1 on partial `n_scored < n_requested` or `baseline_eligible: false`); provenance
  stamps; `_observations` carry-forward. Reads an eval-matrix `summary.json`.
- `configs/eval_baselines/` — committed baseline cells (`omakase_mission_open_ended.json`,
  `refinement_cheaper.json`, quarantined `late_night_closure_cascade.json`) + `_snapshots/`.
- `scripts/check_baselines_fresh.py` — the CI staleness gate; watch-set includes
  `app/llm_factory.py` + matrix configs; defines the regen-last ordering constraint (D-15-05).
- `docs/baseline_regen.md` — deferred-cell promotion path (anthropic/gemini stay out); the
  "baselines are the last thing you regenerate" runbook.

### Eval harness + telemetry (consume, don't duplicate)
- `scripts/eval_matrix.py` + `configs/eval_matrix_arm.yaml` — the committed 3-model × 2-scenario
  arm matrix; reuse as-is for Run #1 (A2) and Run #2 (flag-off).
- `scripts/eval_agent.py` — `arm_flags` self-description dict (smoke verification); INST-04
  per-step `step_telemetry` assembly (`llm_call_seconds`, `tool_exec_seconds`,
  `tool_calls_this_step` per step) — the PROMO-03 latency-report and D-15-08 root-cause source;
  per-query `latency_seconds`; full message traces in run JSON.
- `scripts/eval_falsifier.py` — exit-code 0/1/2 contract; run-dir mode; per-scenario breakdown
  (paste verbatim into promotion_decision.md per Phase-13/14 precedent) — judges the A2 retest.

### Agent loop (potential D-15-08 fix target only)
- `app/agent/graph.py` — forced-commit synthesizer (~314, ~632, ~655 — the CR-01 fix site,
  the path the A2 retest exercises); `all_slots_viable` / rule-8 viability condition;
  `_prune_for_llm`; build-time flag-read block. **Edit ONLY for a D-15-08 trivial fix; any
  change here trips the staleness gate and forces the regen-last ordering.**

### Process docs
- `.planning/milestones/v2.2-MILESTONE-SEED.md` — milestone framing + anti-scope.
- `.planning/STATE.md` — Decisions log (A1/A2/A3/R1/R2/R3/valve outcomes; ARCH-FUT-01 ratify line).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Per-step latency decomposition already exists** in every run JSON:
  `deterministic.step_telemetry[i]` carries `llm_call_seconds` + `tool_exec_seconds` +
  `tool_calls_this_step`. PROMO-03 is a pure **aggregation/reporting** task over committed run
  JSONs — no new live runs needed for the anchor latency report itself (Run #2's JSONs supply
  it; existing Phase-13/14 JSONs corroborate). NOTE: observed anchor `latency_seconds` on
  omakase is ~46s in the A2 run dir — already **over** the ~30s budget on the open-ended
  scenario; the report must decompose honestly, not flatter.
- **`configs/eval_matrix_arm.yaml`** two-scenario 3-model arm config — reuse verbatim for both
  runs (flags differ, config shape identical).
- **`arm_flags` self-description** in run JSON — smoke verification confirms which flags were
  actually set before the n=5 spend.
- **`write_baselines.py` honesty rules** already encode D-15-04 (refuses partial/quarantined) —
  the "no partial cells" criterion is enforced by the tool, not by manual discipline.
- **Existing Phase-13/14 run JSONs** carry `viable_candidates_per_step`, `rule8_met_per_step`,
  `first_commit_call_step`, `forced_commit_step` — the D-15-08 root-cause diagnostic needs
  ZERO new live runs to reach a conclusion.

### Established Patterns
- **Honest-null discipline** (Phases 13/14): a negative/plateau result is recorded with full
  numbers and treated as a valid outcome, never a blocker — D-15-01 carries it forward.
- **Closed-record immutability** (Phase 14): finished verdict docs get an explicit closing
  line and stay frozen; new phases cross-link, never append — D-15-03 follows it.
- **Smoke-before-spend**: n=1 `arm_flags` verification before every full n=5 matrix run.
- **Anchor non-regression mandatory**: every experiment must show gpt-4o-mini held its rate
  (A3's anchor regression is the cautionary precedent).
- **Regen-baselines-last**: agent-code changes land first, baselines written against final code
  so the staleness gate passes (`docs/baseline_regen.md`).
- **Bookkeeping commits** (ROADMAP/REQUIREMENTS/STATE status flips) go straight to main, no PR.
- **Full `make test` mandatory** before merge (DB-pool contamination risk with real-graph tests).

### Integration Points
- `configs/eval_gates.yaml` rationale/status fields — the PROMO-02 edit surface (data-driven,
  no code change beyond the yaml unless a D-15-08 fix lands).
- `scripts/write_baselines.py` ← Run #2 `summary.json` — the PROMO-01 entry point.
- `app/agent/graph.py` forced-commit + viability path — exercised by the A2 retest; edited only
  for a D-15-08 trivial fix.
- `docs/promotion_decision.md` (NEW) — the milestone-closing synthesis doc; cross-links both
  closed verdict docs.

</code_context>

<specifics>
## Specific Ideas

- The A2 retest must verify (via `arm_flags` smoke + the `forced_commit_step` / `commit_forced`
  telemetry fields) that the forced-commit path **actually fires** this time — the whole point
  of the retest is that CR-01 left it never-firing in Phase 13. A "0.500 again" result is only
  meaningful if the forced mechanism demonstrably engaged.
- The PROMO-03 latency report should decompose per-turn time into LLM-call seconds vs
  tool-exec seconds (both already in `step_telemetry`) and state the anchor's per-turn total
  against ~30s/turn explicitly — including the inconvenient ~46s-on-omakase observation —
  rather than reporting a flattering single number. Decisiveness (step count) is the dominant
  latency lever per Decision 3.
- `docs/promotion_decision.md` is the v2.2 milestone-audit anchor — write it to be the single
  doc a reviewer reads to understand how the milestone closed (null result, anchor ratified,
  ARCH-FUT-01 deferred, what's enforced vs logged, what the latency budget reality is).

</specifics>

<deferred>
## Deferred Ideas

- **gpt-5-mini decisiveness on refinement_cheaper** — the open problem the milestone could not
  close across 6+ interventions. Phase 15 diagnoses it (D-15-08) but only ships a fix if it's
  trivial; the real fix (if non-trivial) is a future milestone, with the Phases 13–14 evidence
  chain as the trigger.
- **ARCH-FUT-01 execution** (custom imperative agent loop) — ratified as deferred tracked debt
  at the D-14-08 checkpoint; Phase 15 does NOT act on it. Trigger criteria = the Phases 13–14
  evidence package.
- **Prod-default flip to `FORCED_COMMIT_STEP=6`** — even if the A2 retest clears, flipping the
  prod default is a separate decision (changes anchor behavior); flagged in promotion_decision.md,
  likely deferred (D-15-07).
- **anthropic + gemini n=5 baselines** — stay deferred (D-12-09 billing/quota, locked); the
  runnable-cells-only baseline set (D-15-04) excludes them. Promotion path documented in
  `docs/baseline_regen.md`.
- **anthropic/gemini gate enforcement** — cannot be earned without the deferred baselines;
  stay logged-not-gated.

</deferred>

---

*Phase: 15-Gate Promotion + Baseline Regen*
*Context gathered: 2026-06-14*
