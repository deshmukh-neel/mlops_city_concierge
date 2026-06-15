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

## Milestone: v2.2 — Reasoning-Model Decisiveness

**Shipped:** 2026-06-15
**Phases:** 4 (12–15) | **Plans:** 24 | **PRs:** #110 + phase 12-14 branches

### What Was Built
- Decisiveness instrumentation + executable falsifier: per-run telemetry (first-commit step, viable-candidate counts, rule-8 precondition flag) + per-turn latency decomposition; `make eval-falsifier` as a single pass/fail report
- Four joint experiment arms (viability contract, forced-commit-at-step-N, co-tuned critique recalibration, parallel tool execution) run at n=5 against the comparison floor
- Conditional richer-state replay (multi-message `_reasoning_state` replay, content-block preservation) — entered on the DEC plateau, A/B-measured
- Gate promotion + honest baseline regen: gpt-4o-mini enforced, gpt-5-mini logged, 6 cells re-baselined flag-off with corrected provenance, latency report vs the ~30s budget

### What Worked
- **A pre-registered falsifier turned a vibes question into a decidable one.** "Make reasoning models decisive" is unfalsifiable as stated; INST-05 (`gpt-5-mini ≥ 0.6 at n=5, anchor non-regressing`) made every arm machine-checkable and killed any temptation to declare a marginal arm a "win." The milestone's value is a *clean negative*, and that was only possible because the bar was set before the runs.
- **The conditional phase gate paid for itself.** Phase 14 (replay) was gated on all DEC arms plateauing. The gate opened honestly, replay ran, and it *also* plateaued — converting "maybe richer state would help" from a standing hypothesis into closed evidence. Without the gate we'd either have skipped replay (leaving the hypothesis open) or run it speculatively (spend with no decision rule).
- **Zero-spend diagnostics before paid runs.** Phase 15's wave 1 was a pure root-cause read of existing run JSONs (refinement_cheaper) before any live API call; Phase 14's R2 evidence audit was likewise zero-spend. Cheap analysis repeatedly narrowed or eliminated expensive runs.
- **Provenance honesty caught a contaminated gate.** The prior `refinement_cheaper` 1.000 turned out to be a flag-ON arm artifact, not honest prod config. Re-baselining flag-off (to 0.000) before promoting any gate avoided repeating the v2.0 fail-open mistake — the same class of bug, caught one milestone earlier in its lifecycle.

### What Was Inefficient
- **The forced-commit mechanism shipped inoperative and was only caught at code review.** CR-01: the A2 branch built the wrong-shaped candidates (typed `PlaceHit` vs dict) so it never fired; the 0.500 result was model-initiated, not forced. A non-mocked regression test would have caught it before the live A2 run — the same "test drives the real pipeline" lesson v2.1 already logged, recurring in a new spot.
- **A reporting-tool bug pasted 0/0 into a verdict.** CR-02: the falsifier split reader read the wrong field, so a verdict table showed 0/0 until hand-computed numbers corrected it. Eval *tooling* needs the same fixture-against-real-shape rigor as the eval *scorers*.
- **Latency stayed structurally unmeasurable for the delta that motivated an arm.** The Phase-12 comparison-floor run dirs predate the INST-04 step telemetry, so the parallel-tool-execution arm's reduction-vs-floor delta couldn't be computed (criterion respecified to absolute latency for a future baseline). Instrumentation that lands *after* the baseline it needs to compare against can't retro-measure — telemetry should precede the floor it will be judged against.

### Patterns Established
- A behavior-chasing milestone needs one pre-registered, executable falsifier before any runs; "works" is defined as a number, not a judgment
- Expensive/architectural phases are gated on cheaper phases failing first (conditional entry gates), so spend follows evidence
- Zero-spend diagnostic waves (read existing run JSONs) precede any live-run wave
- An honest null result is a shippable milestone outcome — ratify the incumbent and defer the rewrite with a documented trigger, rather than forcing a weak promotion
- Empirical numbers carry provenance: a metric measured under a non-prod flag state is not a prod baseline, and promoting a gate on it is a fail-open in disguise

### Key Lessons
1. **Necessary ≠ sufficient, now proven twice.** v2.1 showed reasoning-state preservation was necessary but didn't make models commit; v2.2 showed that *no* prompt/graph/replay intervention does either, under budget. The decisiveness gap is architectural — further tuning is ruled out by evidence, not opinion.
2. **A null result is only valuable if the bar was set first.** The entire milestone's credibility rests on INST-05 being defined at the start. Retrofitting a success criterion to a plateau would have been worthless; pre-registration is what makes "nothing worked" a finding.
3. **Eval tooling is production code.** Two of the three gap-closure items (CR-01 forced-commit, CR-02 split reader) were bugs in the *experiment apparatus*, not the agent. The apparatus that decides whether an intervention works needs regression tests that drive its real path.
4. **Order instrumentation before the baseline it measures against.** The unmeasurable latency delta is a sequencing failure: telemetry added after the comparison floor can't compare to it.

### Cost Observations
- Live eval spend concentrated in three checkpointed waves (Phase 13 arms, Phase 14 R1/R2, Phase 15 A2-retest + flag-off baseline), each ≤4-run capped; the rest of the milestone was zero-spend analysis and flag-gated code
- The ≤4-run cap + conditional gates kept a six-intervention investigation inside a small live-run budget; most candidate runs were eliminated by cheaper analysis before spending
- deepseek-reasoner remains the wall-time dominator on any matrix it appears in; sequential execution (D-11-14) is still the latency floor

## Cross-Milestone Trends

| Milestone | Shipped | Phases | Plans | Theme |
|-----------|---------|--------|-------|-------|
| v1.0 Knowledge Graph | 2026-05-14 | 1 | — | KG edge table + traversal tool |
| v2.0 Production Readiness | 2026-06-03 | 5 (2–6) | 29 | Eval harness + agent-behavior fixes |
| v2.1 Reasoning-Model Compat | 2026-06-11 | 5 (7–11) | 35 | Reasoning-state adapters + honest baselines + CI gates |
| v2.2 Reasoning-Model Decisiveness | 2026-06-15 | 4 (12–15) | 24 | Decisiveness falsifier + experiment arms → honest null; anchor ratified |

Recurring: each milestone's verification honesty work (v2.0 baselines → v2.1 fail-open closure → v2.2 falsifier + provenance correction) exposed or settled the next question. v2.2's decisiveness scope came directly from v2.1's honest measurements; v2.2's honest null now scopes the open architectural question (ARCH-FUT-01) for whenever it's triggered.

Recurring failure mode across v2.1 → v2.2: **the test/eval apparatus itself ships bugs that green unit tests miss** (v2.1 `float(None)` + fail-open gate; v2.2 forced-commit no-op + split-reader 0/0). Each was caught only at post-execution code review. The standing mitigation — integration tests drive the real pipeline, eval tooling gets fixtures against real shapes — is established but not yet reflexive.
