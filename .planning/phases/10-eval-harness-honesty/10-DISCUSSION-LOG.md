# Phase 10: Eval Harness Honesty - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 10-eval-harness-honesty
**Areas discussed:** ERROR-run semantics, Gate derivation & storage, late_night fate, Live-probe & fixtures

---

## Delegation note

Four gray areas were presented via multi-select. The user confirmed the session model
(Claude Fable 5) and replied: *"i'll just go with whatever u recommend for these
discussions"* — delegating all four areas to Claude's recommendations. Per-area
options below show what was on the table and which option Claude selected on the
user's behalf.

---

## ERROR-run semantics (EVAL-01)

| Option | Description | Selected |
|--------|-------------|----------|
| ERROR run-JSON + invalid cell | `status: error` records, excluded from aggregation, any errored run ⇒ cell INVALID_FOR_BASELINE, no exception-type allowlist | ✓ |
| Score remaining runs | Errored runs dropped, cell scored on n_scored < 5 | |
| Summary-only errors | No per-run error records, just summary failure list | |

**Choice rationale:** auditability (the 21-14-30Z post-mortem depended on per-run
artifacts) + baselines demand full n (partial-outage cells poisoned baselines twice
already). Allowlist rejected: exceptions are infra/config by definition here; model
failure = low score on a completed run, never an exception. → D-10-01..04.

## Gate derivation & storage (EVAL-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Machine-readable YAML + provisional values now | `configs/eval_gates.yaml` consumed by executable check; commit-rate hard floors per D-09-02 shape; strict-1.0 retired; Phase 11 re-ratifies | ✓ |
| Docs + hardcoded Makefile | Numbers in docs/Makefile only | |
| Defer all numbers to Phase 11 | Phase 10 ships mechanism with empty gate file | |

**Choice rationale:** hardcoded-doc numbers are how the unsatisfiable strict-1.0 gate
fossilized; an empty gate file makes EVAL-03's "fires on synthetic regression"
untestable. `status: aspirational` keeps the failing gpt-5-mini Part A gate visible
without blocking known-gap work. → D-10-05..08.

## late_night fate (EVAL-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Quarantine now, migrate later | `baseline_eligible: false`, stays a diagnostic, excluded from regen + gates | ✓ |
| Migrate to prod threading | Honest shape but changes what the scenario measures; turn-2 scorers may stop firing | |
| Run both shapes | Adds a prod-threading variant alongside legacy | |

**Choice rationale:** migration is a scenario redesign (the closure-cascade scorers
were built against full-tool-history shape) — scope creep on a harness phase; Phase 11
regen doesn't need the scenario. Migration deferred. → D-10-09/10.

## Live-probe & fixtures (EVAL-05)

| Option | Description | Selected |
|--------|-------------|----------|
| Full-fidelity probes → checked-in fixtures | Generalized probe script, redacted AIMessage dumps in tests/fixtures/, parametrized tests AUGMENT synthetic cases; manual pre-matrix step | ✓ |
| Non-None canary probes | Cheap assertion-only probes, no fixtures | |
| Scheduled probes | CI/cron probing (needs live keys in CI) | |

**Choice rationale:** the Gemini lcgg key-shape miss and 4 live-only Anthropic bugs
were exactly "synthetic fixture ≠ real wire"; checked-in real payloads close it.
Scheduled rejected per D-09-10 (no live keys in CI). Folds in 09-REVIEW IN-04
(redaction) and IN-03 (hardcoded probe path). → D-10-11..14.

## Claude's Discretion

- EVAL-04: verification-only unless new matrix files appear (PR #104 shipped the test).
- EVAL-06 specifics: factory dispatch tests, ScriptedChatModel ainvoke approach,
  vibe_check verify-first-then-fix (D-10-15..17).
- Schemas/naming for error records, gates YAML, probe fixtures; plan decomposition.

## Deferred Ideas

- late_night prod-threading scenario redesign (Phase 11 stretch / v2.2)
- Parallel matrix execution (after EVAL-01; Phase 11 candidate)
- CI promotion of gates/conformance/live secrets (Phase 11 BASE-03)
- Advisory→hard non-regression delta promotion (Phase 11 BASE-03)
- Baseline-writer tool with n_scored enforcement (Phase 11 BASE-01)
- All decisiveness work (`.planning/v2.2-MILESTONE-SEED.md`)
