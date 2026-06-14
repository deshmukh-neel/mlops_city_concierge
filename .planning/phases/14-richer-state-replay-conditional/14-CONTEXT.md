# Phase 14: Richer State Replay (CONDITIONAL) - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Entry gate is OPEN: `docs/decisiveness_arm_verdicts.md` closing verdict records that no
Phase-13 DEC arm cleared the INST-05 falsifier bar (A1 = 0.000, A2 = 0.500 best signal,
A3 anchor-regressed). Phase 14 therefore executes.

Two replay interventions are implemented behind flags, A/B-tested at n=5 temp=1.0
against the Phase-13 plateau, and judged via the Phase-12 falsifier:

- **REPLAY-01 (R1):** multi-message `_reasoning_state` replay — every in-window
  tool-calling AIMessage gets its own stashed reasoning state replayed, vs the current
  most-recent-only injection (`app/agent/graph.py:336-341`).
- **REPLAY-02 (R2):** content-block preservation through `_prune_for_llm` — pre-cutoff
  AIMessage list-content survives instead of the current `str()` collapse
  (`app/agent/graph.py:232`).

The verdict document records per-arm deltas vs both the flag-off plateau floor AND the
best DEC arm (A2, 0.500), then either (a) a replay arm clears the INST-05 bar and Phase
15 begins, or (b) a documented plateau triggers an explicit ARCH-FUT-01 evaluation
section with a recommendation — the decision itself is a user checkpoint before Phase 15
scope is finalized.

Nothing ships enabled-by-default to prod in this phase — promotion is Phase 15.

</domain>

<decisions>
## Implementation Decisions

User delegated all four discussed areas to Claude ("u got it fable") — same delegation
pattern as Phase 13. D-14-01..08 are Claude's calls; planner may refine names/details
but must preserve the substance.

### Arm structure & comparison point
- **D-14-01: Two judged arms + one conditional combo, replay arms run PURE.**
  - **R1** = `REPLAY_MULTI_MESSAGE_ENABLED` (REPLAY-01 alone).
  - **R2** = `REPLAY_CONTENT_BLOCKS_ENABLED` (REPLAY-02 alone).
  - **R3 (conditional combo, R1+R2):** run ONLY if neither clears the falsifier alone
    but both show positive signal (mirrors Phase 13's A4 rule, D-13-01).
  - All replay runs execute with the three DEC arm flags UNSET — pure replay effect,
    no attribution confounding. Stacking replay on A2 is NOT a judged arm.
  - **Discretionary 4th run (escalation valve):** if R1/R2/R3 all plateau but the best
    replay arm AND A2 each independently showed positive signal, the verdict doc may
    recommend ONE best-replay+`FORCED_COMMIT_STEP=6` stack run before declaring
    plateau. Hard cap: ≤4 full live matrix runs total this phase.
- **D-14-02: Per-arm run budget identical to D-13-02.** 3 models (gpt-5-mini,
  gpt-4o-mini anchor, deepseek-reasoner) × 2 scenarios (omakase_mission_open_ended +
  refinement_cheaper) × n=5, temp=1.0, sequential, via `configs/eval_matrix_arm.yaml`.
  Smoke n=1 with `arm_flags` self-description verification before every full n=5
  spend. The `arm_flags` run-JSON dict gains the two replay flags (extend, don't
  replace, the Phase-13 keys). No billing top-ups; partials recorded honestly and
  never written as baselines (D-11-14). Anchor non-regression is mandatory for both
  judged arms — A3's anchor regression is the cautionary precedent.

### Multi-message replay mechanics (REPLAY-01)
- **D-14-03: Per-message replay is graph-side iteration over per-message state, not a
  new captured_state plumbing scheme.** Capture is already per-message (each plan()
  turn stashes `_reasoning_state` on its own AIMessage's `additional_kwargs`, and
  D-08-07 preserves kwargs across the prune cutoff). When the flag is ON, plan()
  replays EACH in-window AIMessage's own `_reasoning_state` onto that same message,
  instead of injecting only the most recent one. Implementation shape: a new adapter
  method (e.g. `replay_reasoning_state_multi(outbound)`) with a generic default in the
  ABC that iterates messages and applies the existing single-message injection logic
  per message; per-adapter overrides only where the wire format demands. The existing
  `replay_reasoning_state` signature is UNTOUCHED so the flag-off path stays
  byte-identical and the existing 9-test conformance harness passes unchanged.
- **D-14-04: All four adapters get multi-replay coverage; only three models run.**
  The generic ABC default should cover all adapters nearly for free; anthropic/gemini
  cells stay deferred (D-12-09) and are NOT run. Conformance harness
  (`tests/unit/test_adapters.py`) gains additive multi-path tests per adapter — flag-on
  AND flag-off states both covered. Revertability guarantee from Phase 9 carries over:
  the multi path must be removable per-adapter without touching the single path.

### Content-block preservation semantics (REPLAY-02)
- **D-14-05: Evidence audit BEFORE implementation spend.** Roadmap criterion 2 demands
  "an explanation of whether `str()` collapse was causing observable loss in run
  JSONs". First plan task: a zero-cost audit of existing Phase-12/13 run-dir JSONs
  (full message traces are recorded) to determine whether gpt-5-mini / deepseek
  AIMessages EVER carry list content pre-cutoff. The audit result is written into the
  verdict doc up front. If the tested models never emit list content, R2's A/B is
  expected-null on these cells — it still RUNS (criterion 2 requires a measured delta,
  not an assumption), but the explanation writes itself from the audit, and the plan
  may downscope the R2 smoke accordingly.
- **D-14-06: Minimal-diff preservation.** When `REPLAY_CONTENT_BLOCKS_ENABLED` is ON,
  the pre-cutoff replacement at `graph.py:228-235` becomes
  `AIMessage(content=m.content, additional_kwargs=m.additional_kwargs)` — original
  content shape kept verbatim (list or str), tool_calls still stripped (the
  unanswered-tool_call contract MUST hold), ToolMessages still dropped. No
  block-type filtering in v1 — pre-cutoff AIMessage count is bounded by the prune
  window, so cost exposure is small; any token/latency movement is measured via the
  existing INST-04 `llm_call_seconds` telemetry rather than a new guardrail (no new
  scorers — anti-scope).

### Verdict document & ARCH-FUT-01 handoff
- **D-14-07: New doc `docs/replay_arm_verdicts.md`** mirroring the DEC-05 document's
  structure (falsifier definition, run-budget contract, per-arm sections with run
  dirs + smoke verification + per-model tables + pasted falsifier output + closing
  verdict, summary table, explicit closing line). It cross-links
  `docs/decisiveness_arm_verdicts.md` rather than appending to it — the Phase-13
  record is closed and stays immutable. Per-arm tables report THREE numbers per
  model: pooled commit rate, delta vs flag-off floor, delta vs A2 (0.500).
- **D-14-08: ARCH-FUT-01 evaluation is a recommendation, not a decision.** On plateau,
  the verdict doc gains an "ARCH-FUT-01 Evaluation" section that (a) states the
  cumulative evidence chain (v2.1 proved byte-correct state round-trip; Phase 13 arms
  null; Phase 14 replay deltas), (b) restates what the ARCH-FUT-01 contingency would
  entail, and (c) makes a written recommendation — but Phase 15 scope finalization is
  an explicit USER CHECKPOINT. Decision 3 (gpt-4o-mini stays anchor, ~30s/turn budget)
  bounds the recommendation space: "ratify anchor, defer ARCH-FUT-01 to a future
  milestone" is the likely shape, but the data speaks first. If a replay arm CLEARS
  the bar instead, the winning flag config + run-dir path are recorded as Phase-15
  promotion inputs, same as the DEC-05 format.

### Claude's Discretion
User delegated arm structure, replay mechanics, preservation semantics, and verdict/
handoff structure entirely. Planner may refine exact env-var names, method names, and
file names, but must preserve: pure (non-stacked) judged arms, the ≤4 live-run cap with
the conditional R3 + discretionary stack valve, the additive adapter interface
(existing conformance harness untouched), the R2 evidence audit before spend, the new
verdict doc (not appended to the Phase-13 record), and the user checkpoint before
Phase 15 on plateau.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Entry gate + plateau baseline (settled — do not re-derive)
- `docs/decisiveness_arm_verdicts.md` — DEC-05 record: INST-05 falsifier definition,
  per-arm plateau numbers (A1 0.000 / A2 0.500 / A3 anchor-regression), explicit
  "Phase 14 entry gate OPEN" closing line; the A2 0.500 number is the best-DEC-arm
  comparison point for D-14-07's delta column
- `.planning/milestones/v2.2-MILESTONE-SEED.md` — finding 5 (replay gaps: single-message
  replay at graph.py, `str()` collapse, intra-request-only state); anti-scope
- `.planning/REQUIREMENTS.md` — REPLAY-01/REPLAY-02 definitions + entry gate text
- `.planning/ROADMAP.md` — Phase 14 success criteria 1-3 + conditional entry gate.
  NOTE: line 53's `[x] ... (completed 2026-06-12)` checkbox is a stale bookkeeping
  artifact from the Phase-13 close (progress table correctly says Not started) — fix
  during planning bookkeeping
- `.planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md` — D-13-01..09
  (run budget, flag pattern, honesty contract, verdict structure — all reused here)
- `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md` —
  D-12-01..09 (telemetry capture, viability semantics, falsifier pooling, deferrals)

### Falsifier + eval harness (judging machinery — consume, don't duplicate)
- `scripts/eval_falsifier.py` — exit-code 0/1/2 contract; run-dir mode; per-scenario
  breakdown output (paste verbatim into verdict doc per Phase-13 precedent)
- `scripts/eval_agent.py` — `arm_flags` self-description dict (line ~152; replay flags
  join it), INST telemetry field assembly, full message traces in run JSON (D-14-05
  audit source)
- `scripts/eval_matrix.py` + `configs/eval_matrix_arm.yaml` — the committed two-scenario
  arm matrix config from Phase 13; reuse as-is for replay runs
- `configs/eval_gates.yaml` + `scripts/check_eval_gates.py` — Phase-10 honest gates
- `scripts/write_baselines.py` + `configs/eval_baselines/` — baselines ONLY via this
  tool (D-11-14); replay arm runs are never registered as baselines

### Agent loop + adapters (implementation targets)
- `app/agent/graph.py` — `_prune_for_llm` (~193-240; `str()` collapse at line 232 is
  the REPLAY-02 target; D-08-07 kwargs-forwarding comment at ~228); most-recent-only
  replay site in `plan()` (~336-341, REPLAY-01 target); `_RECENT_TOOL_EXCHANGES_KEPT=2`
  (line 128) defines the in-window size; Phase-13 flag-read block at graph-build time
  (~285-310) is the flag-wiring precedent
- `app/agent/adapters/__init__.py` — `ProviderAdapter` ABC (capture/replay contract),
  `NoOpAdapter`, `MockReasoningAdapter`, `ADAPTERS` registry; new multi-replay method
  lands here with a generic default
- `app/agent/adapters/openai_gpt5.py`, `deepseek.py`, `anthropic.py`, `gemini.py` —
  per-provider replay implementations (all walk-in-reverse most-recent-only today)
- `tests/unit/test_adapters.py` — 9-test conformance harness (required CI step); must
  pass UNCHANGED; multi-path tests are additive
- `tests/unit/test_agent_graph.py`, `tests/unit/test_graph_forced_commit.py`,
  `tests/unit/test_graph_parallel_tools.py` — Phase-13 flag-gated graph test patterns
  to mirror for the replay flags

### Process docs
- `docs/baseline_regen.md` — deferred-cell promotion path (anthropic/gemini stay out)
- `docs/eval_gates.md` — gate semantics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Per-message `_reasoning_state` storage already exists: capture stashes onto each
  AIMessage's `additional_kwargs` and D-08-07 preserves kwargs across the prune cutoff
  — REPLAY-01 is "use what's already stored", not new plumbing
- Phase-13 `arm_flags` self-description in run JSON — replay flags extend the same dict
- `configs/eval_matrix_arm.yaml` two-scenario arm config — reuse verbatim
- `eval_falsifier.py` run-dir mode + per-scenario breakdown — verdicts paste its output
- Full message traces in Phase-12/13 run-dir JSONs — the D-14-05 list-content audit
  needs zero new live runs

### Established Patterns
- Env-flag pattern: truthy-set parsing, resolved ONCE at `build_agent_graph` time and
  closed over inner functions (Phase-13 block at graph.py ~285-310); flags-off path
  must be byte-identical (Phase 13 verified flag-off byte-identity before merge —
  repeat that verification)
- Stale-baseline CI gate: `app/agent/` changes trip it; `[skip-baseline]` ONLY if
  default-path behavior is byte-identical with flags off
- Test layering: unit/mock + smoke + functional; full `make test` mandatory
  (DB-pool contamination risk with real-graph tests)
- Adapter revertability: each provider's changes independently removable (Phase 9
  audit pattern)
- Bookkeeping commits (ROADMAP/REQUIREMENTS status flips) go straight to main

### Integration Points
- `plan()` in `graph.py` — replay-site change (R1) reads the flag closed over from
  graph-build time
- `_prune_for_llm` — module-level function today; R2's flag needs to reach it (planner
  decides: parameter from build-time closure vs module-level read — prefer parameter,
  matching the "resolve once at build time" precedent)
- `ProviderAdapter` ABC + 4 adapter files — new multi-replay method
- `scripts/eval_agent.py` `arm_flags` — two new keys

</code_context>

<specifics>
## Specific Ideas

- Per-arm verdict tables should show three columns per model: pooled rate, Δ vs
  flag-off floor, Δ vs A2 (0.500) — roadmap criterion 1 demands the best-DEC-arm delta
  be REPORTED, not stacked
- The R2 audit (D-14-05) doubles as the criterion-2 explanation: "was `str()` collapse
  causing observable loss" is answered from existing run JSONs before any live spend
- Fix the stale ROADMAP.md line-53 `[x]`/`(completed 2026-06-12)` marker on Phase 14
  during planning bookkeeping — the progress table is correct, the checkbox is not

</specifics>

<deferred>
## Deferred Ideas

- **Replay-arm + DEC-arm stacking as a promoted prod config** — only the single
  discretionary best-replay+A2 escalation run is sanctioned this phase (D-14-01);
  full stacking matrices are Phase-15 material if promotion wants them
- **ARCH-FUT-01 execution (LangGraph replacement / custom loop)** — Phase 14 only
  produces the evaluation + recommendation on plateau; acting on it is a future
  milestone decision at the user checkpoint
- **Cross-request reasoning-state persistence** (`/chat` rebuilds history text-only,
  `app/agent/io.py:32-40`) — seed finding flags it as "likely correct by design,
  verify"; out of REPLAY-01/02 scope, note-only
- **Anthropic/gemini n=5 baselines** — remain deferred (D-12-09); replay runs exclude
  those cells entirely

</deferred>

---

*Phase: 14-Richer State Replay (CONDITIONAL)*
*Context gathered: 2026-06-12*
