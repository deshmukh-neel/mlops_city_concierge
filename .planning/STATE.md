---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Reasoning-Model Compat
current_phase: 11
status: "Phase 11 shipped — PR #106"
last_updated: "2026-06-11T23:10:54.904Z"
last_activity: 2026-06-11
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 35
  completed_plans: 35
  percent: 83
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.1 Reasoning-Model Compat (started 2026-06-03)
**Current phase:** 11

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03 for v2.1 milestone start)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0)
See: .planning/milestones/v2.0-{ROADMAP,REQUIREMENTS}.md for v2.0 archive

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Milestone complete

## Status

- [x] v1.0 Knowledge Graph shipped 2026-05-14
- [x] v2.0 Production Readiness shipped 2026-06-03 via PR #100 (main `14e01dd`)
- [x] v2.1 milestone formalized via `/gsd-new-milestone v2.1` (2026-06-03)
- [x] v2.1 requirements defined (REQUIREMENTS.md — 20 requirements, 4 categories)
- [x] v2.1 roadmap created (phases 7-10, 2026-06-04)
- [x] Phase 7: Prompt/rubric decoupling — completed 2026-06-04
- [x] Phase 8: Reasoning-state thread-through (contract + harness) — completed 2026-06-04
- [x] Phase 9: Per-provider state preservation impls (gpt-5 → DeepSeek → Claude → Gemini 3) — completed 2026-06-05, merged PR #103
- [ ] Phase 10: Eval harness honesty (EVAL-01..06; re-scoped 2026-06-10, original BASE scope moved to Phase 11)
- [ ] Phase 11: Cross-model baseline regen + matrix expansion (BASE-01..04)

## Notes

- **Empirical anchor gate for v2.1:** gpt-5-mini × `refinement_cheaper` × prod × flag-on commits 3 stops in median 5/5 runs at temp=1.0 (currently 0/1). Gates on PROV-01 in Phase 9.
- **Phase 7 falsifier:** if `gpt-5-mini × refinement_cheaper` is still 0/5 after Phase 7 ships, prompt-coupling was not the root cause — state-loss dominates and Phase 9 scope stays at full. If > 0, prompt-coupling contributed and Phase 9 scope may shrink.
- **Phase 8 harness-swap decision gate (REASON-05):** if conformance tests pass in isolation but fail through `graph.invoke`, Phase 8 surfaces this as an explicit blocker and v2.1 replans around a custom imperative loop before Phase 9 starts. This is a real architectural branch point.
- v2.0 closed with one accepted-with-notes gate (D-06-09 part 2 baseline regen — pre-Phase-6 1.0 baselines were Phase-4 fail-open false positives). Real fix is the reasoning-content thread-through landing in Phase 8/9 and honest regen in Phase 10.
- Agent driver remains locked to `openai/gpt-4o-mini` for prod until Phase 9 sub-phases ship per-provider.
- Flagged for separate hotfix (carried from v2.0): CLO-01 — over-aggressive closure detection on Mission queries.

## Resume

Next step: `/gsd-plan-phase 11` to plan Phase 11 (Cross-Model Baseline Regen + Matrix
Expansion). Context is in
`.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md`
(D-11-01..21, all areas decided in auto mode 2026-06-11).
Working branch: `gsd/phase-11-cross-model-baseline-regen-matrix-expansion` (off main @ cc1407c, post-PR #105).

## Current Position

Phase: 11 (cross-model-baseline-regen-matrix-expansion) — EXECUTING
Plan: Not started
Status: Phase 11 shipped — PR #106
Last activity: 2026-06-11
Last session: 2026-06-11T22:58:25.396Z

### Blockers

None active for Phase 9 completion. PROV-05 atomicity audit completed (`.planning/phases/09-per-provider-state-preservation-implementations/09-05-AUDIT.md`). Phase 9 PR-ready: all 5 plans shipped, atomicity audit done, gates documented as SHIPPED-WITH-GAP / SHIPPED-STRUCTURAL / PASS-WITH-FINDINGS per Wave 1/2/3 + D-06-09 precedent. Per `feedback_user_merges_prs`: do NOT run `gh pr merge` once CI is green.

**Pre-Phase-11 prerequisite (was pre-Phase-10 before the 2026-06-10 re-scope):** OpenAI embeddings quota topped up 2026-06-10 (repo secret rotated, CI green). Cloud SQL must be reachable before Phase 11 BASE-01 (re-measure anthropic n=5 + first-time gemini n=5). Phase 10 itself needs no live infra beyond ~$0.05 of probes.

**PROV-05 audit findings carried into PATTERNS.md / Phase 10:**

- Phase 9's additive-overlay pattern (matrix YAML + baseline JSON + cell-count test extended by every sub-phase) makes mid-stack single-PROV revert non-mechanical; cumulative reverse-pop is the realistic developer workflow.
- PROV-02 chore commit 3800737 has a latent test-vs-data atomicity gap (added YAML entry without updating co-tracked `test_eval_matrix.py` assertion); masked at commit time by PROV-03's later bump. Future phases adopt convention: when a sub-phase appends to a shared additive data file with a co-tracked cell-count test, the same commit updates both.

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 9 P09-05 | 60m | 2 tasks | 1 files |
| Phase 10 P10-01 | 60m | 3 tasks | 3 files |
| Phase 10 P04 | 45m | 2 tasks | 6 files |
| Phase 10 P06 | 5m | 2 tasks | 2 files |
| Phase 10 P02 | 30m | 2 tasks | 2 files |
| Phase 10 P05 | 5m | 2 tasks | 5 files |
| Phase 10 P03 | 25m | 2 tasks | 6 files |
| Phase 10 P07 | 10m | 2 tasks | 2 files |
| Phase 10 P08 | 15m | 2 tasks | 4 files |
| Phase 10 P09 | 5m | 2 tasks | 2 files |
| Phase 11 P11-01 | 20m | 2 tasks | 2 files |
| Phase 11 P02 | 3m | 1 tasks | 2 files |
| Phase 11 P07 | 5m | 2 tasks | 4 files |
| Phase 11 P03 | 6m | 2 tasks | 2 files |
| Phase 11 P04 | 10m | 2 tasks | 3 files |
| Phase 11 PP05 | 12m | 2 tasks | 3 files |
| Phase 11 P06 | 25m | 2 tasks | 4 files |
| Phase 11 PP08 | 90m | 4 tasks | 7 files |
| Phase 11 P09 | 150 | 4 tasks | 2 files |

## Decisions

- [Phase ?]: Phase 9 PROV-05 atomicity audit: PASS-WITH-FINDINGS — import isolation PASS; cumulative reverse-pop revert preserves v2.0 anchor; PROV-02 chore 3800737 latent test-vs-data atomicity gap documented as note (D-06-09 precedent)
- [Phase 10 P10-01]: D-10-01 QueryEvalResult gains status discriminator (default 'ok') + error dict field for error-run records per D-10-01 schema
- [Phase 10 P10-01]: D-10-02 Exceptions in both threading branches return make_error_record() not partial-state scoring — scorers never reached on exception
- [Phase 10 P10-01]: D-10-03 aggregate_results filters on status=='ok'; gains n_scored/n_errored/cell_valid — errored runs excluded from scorer means
- [Phase ?]: D-10-05: configs/eval_gates.yaml is the single source of truth for per-family merge gates
- [Phase ?]: D-10-06: strict refinement_minimal_edit == 1.0 gate formally retired; honest anchor median 0.0/max 0.5 post-Phase-7
- [Phase ?]: D-10-07: gpt-4o-mini active committed_itinerary_rate >= 0.8; gpt-5-mini aspirational >= 0.6; anthropic provisional-n1 >= 0.8; deepseek/gemini logged-not-gated
- [Phase 10 P10-06]: D-10-15: gpt-5-mini routes through OpenAIReasoningChatModel(use_responses_api=True); gpt-4o-mini stays on plain ChatOpenAI — both paths now test-locked (EVAL-06)
- [Phase 10 P10-06]: D-10-16: ScriptedChatModel ainvoke works via BaseChatModel executor fallback proven by async test (EVAL-06)
- [Phase 10 P10-06]: D-10-17: vibe_check sync invoke safe under LangGraph 1.2.0 ThreadPoolExecutor sync-node dispatch; doc-comment added, no code change (EVAL-06)
- [Phase 10 P10-02]: D-10-03 (aggregator half): aggregate_cell_jsons reads n_scored/n_errored/errors from each cell JSON; per-provider block gains n_scored/n_errored/cell_valid in summary.json
- [Phase 10 P10-02]: T-10-02-02: total_errored>0 forces non-zero exit with distinct INVALID_FOR_BASELINE stderr line; error count separate from violation count
- [Phase 10 P10-02]: structural-check Check 6: synthetic error cell validates stage in {'setup','turn0','turnN'} — error-schema contract enforced in CI without live calls
- [Phase 10 P10-05]: D-10-11: fixture output path is tests/fixtures/provider_payloads/{provider}.json (JSON not markdown)
- [Phase 10 P10-05]: D-10-12: fixture-loading adapter tests augment (never replace) synthetic cases; absent fixtures SKIP gracefully in CI
- [Phase 10 P10-05]: D-10-13: _SECRET_PATTERNS covers OpenAI sk-, Anthropic sk-ant-, Google AIzaSy...; env-var-sourced secrets substituted pre-regex; redaction unit-tested (EVAL-05)
- [Phase 10 P10-05]: D-10-14: make probe-providers is MANDATORY pre-matrix step, CI-free; fail-closed post-write guard deletes fixture on secret leak
- [Phase ?]: D-10-09: late_night_closure_cascade quarantined via baseline_eligible=False on EvalQuery
- [Phase ?]: D-10-10: late_night baseline JSON annotated with _observations; annotate-not-regenerate pattern confirmed
- [Phase ?]: CR-01 closed: _check_gate walks nested scenarios->providers shape
- [Phase 10 P10-08]: CR-03 closed: main() wires load_eval_queries(args.eval_queries) into aggregate_cell_jsons with try/except fallback to None; baseline_eligible now reaches real summary.json
- [Phase 10 P10-08]: CR-02 closed: _constraints_for_case None-guards expected_results dereference; all 30 hand_written cases including clarification cases build constraints without AttributeError
- [Phase ?]: CR-05 closed: response_metadata, usage_metadata, tool_calls all pass through _redact(json.dumps); _scan_fixture_for_secrets helper covers regex + _SECRET_ENV_VARS; EVAL-05 fail-closed claim now accurate
- [Phase ?]: CR-04 closed: REPO_ROOT = Path(__file__).resolve().parents[2] replaces hardcoded author path in test_main_help_exits_zero; test passes from any cwd/machine
- [Phase 11 P11-01]: D-11-05: _NON_TOOL_SCRATCH_KEYS frozenset excludes prior_committed_stops + prior_stops_obj from tool-call counting in eval_agent.py (WR-08)
- [Phase 11 P11-01]: D-11-06: single-turn error capture in evaluate_case wraps ainvoke in try/except, returns make_error_record(case, 'turn0', exc); scorers never invoked on failure (WR-06)
- [Phase 11 P11-01]: D-11-04: all five derived rates in aggregate_results guarded with `if n_scored > 0 else None` — None not fail-open 1.0 on all-errored cells (WR-09)
- [Phase 11 P11-01]: D-11-16: main() exit-code contract: 0=clean, 1=model-behavior violations, 2=infra failure (build_report exception or n_errored>0) (WR-07)
- [Phase ?]: D-11-03: category_compliance zero-stop guard returns None (abstain) not 1.0; zero-stop guard fires before D-03 empty-requested guard
- [Phase ?]: D-11-21: WATCH_PREFIXES extends staleness gate to cover app/llm_factory.py and configs/eval_matrix* (BASE-04)
- [Phase ?]: WR-10: probe_provider_capture.py additional_kwargs use type-faithful json.loads(_redact(json.dumps(v, default=str))) redaction; T-11-18 mitigated
- [Phase 11 P11-03]: D-11-02: committed_itinerary_rate threaded into summary.json scorers block as supplemental scalar bypassing CRITIQUE_THRESHOLDS whitelist; hard gates flip from NOT-EVALUABLE to enforced
- [Phase 11 P11-03]: D-11-16 (matrix half): run_matrix returns 3-tuple (rc, violation_cells, error_cells); rc==2 on errors, rc==1 on violations-only, rc==0 clean; error dominates violation in precedence
- [Phase 11 P11-03]: WR-11: structural-check Check 6 replaced tautological synthetic-dict with real make_error_record(EvalQuery, 'turn0', RuntimeError) schema validation
- [Phase 11 P11-04]: D-11-12: eval_matrix.yaml gains gpt-5-mini, claude-sonnet-4-6, deepseek-reasoner flag-OFF entries; gemini excluded per PROV-04; Wave-1 deferrals in _DEFERRED_BASELINE_CELLS
- [Phase 11 P11-04]: D-11-13: late_night_closure_cascade removed from default scenarios (stays runnable via SCENARIOS=); baseline JSON preserved per D-10-10
- [Phase 11 P11-06]: D-11-15: baselines-mode input-source swap reuses _check_gate unchanged; _build_summary_from_baselines derives scenario_id from filename stem for legacy baseline JSONs
- [Phase 11 P11-06]: D-11-17: advisory entries report-only WARN; refinement_minimal_edit_median resolved to refinement_minimal_edit scorer in _check_gate
- [Phase 11 P11-06]: D-11-19: reasoning_conformance marker promoted to required CI step (no continue-on-error); mock-driven, no live keys
- [Phase 11 P11-06]: D-11-20: aspirational misses (gpt-5-mini) remain non-blocking in baselines mode
- [Phase ?]: D-11-20 complete: anthropic demoted to logged (billing exhaustion); gpt-5-mini refinement n=5 median=0.0 measured; gates re-ratified with live data
- [Phase ?]: No eval_gates.yaml rationale edits needed: all gates key on committed_itinerary_rate (CR-01-clean); CR-01 contamination was confined to category_compliance; gap-closure regen delivered honest n=5 abstain-semantics baselines (D-11-09)

## Accumulated Context

### Roadmap Evolution

- Phase 10 edited: re-scoped to Eval Harness Honesty (EVAL-01..06) after post-Phase-9 harness analysis; BASE scope moved to Phase 11
- Phase 11 added: Cross-Model Baseline Regen + Matrix Expansion (carries BASE-01..04 from original Phase 10)
