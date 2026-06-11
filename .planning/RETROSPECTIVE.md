# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v2.1 — Reasoning-Model Compat

**Shipped:** 2026-06-11
**Phases:** 5 (7–11) | **Plans:** 35 | **PRs:** #103, #105, #106

### What Was Built
- Prompt/rubric decoupling: behavioral rules moved from SYSTEM_PROMPT into the `refinement_minimal_edit` scorer, locked by a CI grep gate
- Typed `ProviderAdapter` reasoning-state contract + per-provider conformance harness (required CI step); LangGraph retained after the REASON-05 decision gate
- Four per-provider reasoning-state adapters (gpt-5 Responses API, DeepSeek reasoner, Anthropic, Gemini 3 experimental)
- Eval-harness honesty: fail-open scoring closed, error records, scenario quarantine, per-family gates from honest data
- Honest n=5 baselines via `write_baselines.py`, 3 cross-model matrix anchors, live-key-free `--baselines-mode` CI gate, baseline-regen runbook

### What Worked
- **Falsifier-first sequencing.** Phase 7 was deliberately ordered as a falsifier for the architectural diagnosis; gpt-5-mini staying flat 0/5 post-decoupling confirmed state-loss dominated and kept Phase 9 at full scope without debate.
- **Decision gates instead of upfront commitments.** Phase 8's conformance harness doubled as the LangGraph-vs-imperative-loop gate; the cheap test settled an expensive architecture question.
- **Post-execution code review caught what unit tests missed.** Phase 11's review found two execution-proven criticals (the `float(None)` crash and a fail-open CI gate) *after* 1185 tests were green — because a test re-implemented the filter inline instead of driving the real pipeline.
- **Atomic commits as crash insurance.** A network failure killed an executor mid-live-run (~2.5h session); every completed task was already committed, so a fresh continuation agent resumed without re-spending API credit.
- **Scoped temp matrix configs.** Re-measuring only contaminated cells (instead of full matrices) roughly halved the gap-closure regen wall time and spend.

### What Was Inefficient
- **Baselines were written before the measurement-semantics fixes were reviewed.** The CR-01 contamination forced a second ~2.5h live regen. Review-before-regen should be the rule: run `/gsd-code-review` after the harness-fix waves, before any wave that commits empirical numbers.
- **Live-infra preconditions discovered at checkpoint time.** Anthropic's empty credit balance surfaced 5 cells into a paid run; a cheap per-provider preflight probe (1-token call per key) before the matrix would have caught it for cents.
- **The MLflow VM ran on a manual nohup.** The VM reboot silently killed MLflow and hung app startup for an unrelated workflow mid-milestone; an hour of ops diagnosis that a systemd unit would have prevented.
- **Stale bookkeeping accumulated.** PROMPT checkboxes and traceability rows sat at Pending for a week after Phase 7 verified; the milestone audit had to reconcile them.

### Patterns Established
- Baselines are never hand-rolled: `write_baselines.py` is the only writer, and it refuses partial/quarantined cells mechanically
- Integration tests must drive the real pipeline (`score_checks → aggregate_results → exit code`), not re-implement its logic inline
- CI gates fail closed: missing/empty input directories are exit-2 errors, never silent passes
- Deferrals are coded, not just documented: logged-not-gated status in `eval_gates.yaml` + `_DEFERRED_BASELINE_CELLS` parity entries + runbook promotion paths
- 0/1/2 exit-code contract (clean / model violation / infra failure) across all eval scripts

### Key Lessons
1. A scorer-semantics change is only done when its *consumer* handles the new value — `category_compliance` returning `None` was correct in isolation and crashed at `float()` one layer up.
2. Numbers committed under broken semantics are permanent until re-measured: sequence semantic fixes strictly before empirical regen, and budget for one regen redo anyway.
3. Reasoning-state preservation was necessary but not sufficient — with state threading verified across all four providers, reasoning models still don't commit. Decisiveness is a behavior problem (v2.2), not a plumbing problem.
4. Billing/quota are first-class preconditions: probe every provider key cheaply before paid matrix runs.

### Cost Observations
- Live eval spend: 2 full matrix sessions (~25 min omakase, ~40 min refinement) + 1 scoped gap-closure regen (~2.5h incl. polling); deepseek-reasoner dominates wall time
- Sequential execution (D-11-14) is the latency floor; parallel-across-providers is the obvious v2.2+ speedup if latency metrics are excluded from parallel runs
- Anthropic cells: 10 paid-error calls burned on an empty-balance account before the checkpoint fired

## Cross-Milestone Trends

| Milestone | Shipped | Phases | Plans | Theme |
|-----------|---------|--------|-------|-------|
| v1.0 Knowledge Graph | 2026-05-14 | 1 | — | KG edge table + traversal tool |
| v2.0 Production Readiness | 2026-06-03 | 5 (2–6) | 29 | Eval harness + agent-behavior fixes |
| v2.1 Reasoning-Model Compat | 2026-06-11 | 5 (7–11) | 35 | Reasoning-state adapters + honest baselines + CI gates |

Recurring: each milestone's verification honesty work (v2.0 baselines → v2.1 fail-open closure) exposed the next milestone's real problem. v2.2's decisiveness scope came directly from v2.1's honest measurements.
