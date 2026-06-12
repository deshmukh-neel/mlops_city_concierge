# Phase 12: Decisiveness Instrumentation + Comparison Floor - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 12-decisiveness-instrumentation-comparison-floor
**Areas discussed:** Telemetry capture point, Commit-consideration semantics, Falsifier report mechanics, Gemini quota contingency

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Telemetry capture point | Instrument graph/state (prod code) vs harness-side reconstruction | ✓ (delegated to Claude) |
| Commit-consideration semantics | Strict tool-call vs lenient reasoning-text signal; viability threshold source | ✓ (delegated to Claude) |
| Falsifier report mechanics | Fresh live run vs read-latest; scenario scope of the 0.6 bar; reuse vs new script | ✓ (delegated to Claude) |
| Gemini quota contingency | Block Phase 12 on quota vs conditional carve-out | ✓ (user decided) |

**User's choice (verbatim):** "u decide for me fable excpt for gemini quota, just defer
for now, also dont want to top up yet"

**Notes:** Single-turn discussion. User delegated all implementation gray areas to
Claude's discretion and made one substantive call: defer the gemini n=5 baseline
(ANCH-02) — no quota/billing top-up — giving gemini the same deferred-cell treatment as
anthropic (ANCH-01). This is a scope reduction relative to ROADMAP.md success
criterion 4; CONTEXT.md records the supersession.

---

## Claude's Discretion

Decisions made by Claude under delegation (full rationale in CONTEXT.md D-12-01..08):

- **Telemetry capture:** hybrid — in-graph `step_telemetry` (timings/counts only) +
  harness-side derived metrics; eval semantics stay out of prod code
- **INST-01 signal:** strict `first_commit_call_step` primary; best-effort
  `first_commit_mention_step` secondary (null where reasoning is opaque)
- **Viability bar:** import `LOW_SIMILARITY_THRESHOLD` from `app/agent/revision.py`,
  record the value per run JSON (`viability_threshold`)
- **Falsifier:** new thin `scripts/eval_falsifier.py` reading existing eval_reports
  artifacts (never live runs), reusing eval_matrix/check_eval_gates machinery,
  PASS/FAIL + exit code; 0.6 bar pooled across all scored scenario cells (omakase-only
  would already pass at baseline — vacuous)

## Deferred Ideas

- ANCH-02 gemini n=5 baseline — deferred (no top-up); revisit when budget allows
- ROADMAP.md/REQUIREMENTS.md docs-only amendment to reflect the gemini deferral
