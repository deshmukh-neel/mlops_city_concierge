---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Reasoning-Model Compat
current_phase: 10
status: verifying
last_updated: "2026-06-11T05:12:01.580Z"
last_activity: 2026-06-11
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 26
  completed_plans: 26
  percent: 67
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.1 Reasoning-Model Compat (started 2026-06-03)
**Current phase:** 10

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03 for v2.1 milestone start)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0)
See: .planning/milestones/v2.0-{ROADMAP,REQUIREMENTS}.md for v2.0 archive

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Phase 10 — eval-harness-honesty

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

Next step: `/gsd-plan-phase 10` to plan Phase 10 (Eval Harness Honesty). Context is in
`.planning/phases/10-eval-harness-honesty/10-CONTEXT.md` (D-10-01..17, all areas decided).
Working branch: `gsd/phase-10-eval-harness-honesty` (off main @ e3dc6c2, post-PR #104).

## Current Position

Phase: 10 (eval-harness-honesty) — EXECUTING
Plan: 6 of 6
Status: Phase complete — ready for verification
Last activity: 2026-06-11

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

## Accumulated Context

### Roadmap Evolution

- Phase 10 edited: re-scoped to Eval Harness Honesty (EVAL-01..06) after post-Phase-9 harness analysis; BASE scope moved to Phase 11
- Phase 11 added: Cross-Model Baseline Regen + Matrix Expansion (carries BASE-01..04 from original Phase 10)
