---
phase: 19-productionized-loop-metric-loop
plan: "04"
subsystem: testing
tags: [loop-runner, mlops, hit@k, recall@k, floor, operator-gate, CI-unit-tested, gap-mine, sandbox]

# Dependency graph
requires:
  - phase: 19-03
    provides: scripts/loop_runner.py full orchestrator (coercion-ordering + gap-handoff + frozen-paraphrase + v2-diff hit@k/recall@k + floor gate + MLflow)
  - phase: 19-01
    provides: app/loop/falsifier_core.py FLOOR + decide_loop_exit + compute_recall_at_k
  - phase: 16-loop-falsifier
    provides: operator-run gate / CI-unit-tested split pattern (D-06)
provides:
  - docs/loop_runner.md operator runbook (gate purpose, two demand modes, D-01/D-02 caveats, exit codes, DEFERRED calibration finding)
  - make loop synced into CLAUDE.md / AGENTS.md / .github/copilot-instructions.md
  - FLOOR=0.0 confirmed correct and green unit suite (49 tests pass)
affects:
  - any operator running make loop (reads docs/loop_runner.md)
  - v2.3 milestone close (phase 19 capstone)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Operator-run gate / CI-unit-tested split (D-06): make loop uses live keys; CI tests pure decision logic only"
    - "Plumbing-verified with deferred calibration: gate pipeline runs end-to-end; quality floor deferred until corpus blocker resolved"
    - "FLOOR=0.0 is intentional uncalibrated default — strict-positive-delta only until a citywide-absent-cuisine gap is available"

key-files:
  created:
    - docs/loop_runner.md
    - .planning/phases/19-productionized-loop-metric-loop/19-04-SUMMARY.md
  modified:
    - CLAUDE.md
    - AGENTS.md
    - .github/copilot-instructions.md

key-decisions:
  - "FLOOR stays 0.0 — calibration deferred. Per-(neighborhood,cuisine) gaps do not zero in the SF corpus (Places API leaks cuisine across 20 neighborhoods); a real calibration needs a citywide-absent cuisine construct."
  - "Two bugs fixed during operator gate run: dbf9b1a (metric target-set asymmetry — before scored against wrong IDs, giving delta=-1.0) and 387f1b3 (Gemini block-content crash in paraphrase generation)."
  - "Loop is plumbing-verified: full gap-mine→ingest→embed→score→gate pipeline runs end-to-end. PRE-FIX run 42fdf99657714f1ca17e849ddf0ce787 documents the metric bug, NOT a calibration outcome."
  - "AI-instruction sync maintained: make loop and Phase 19 note added to all three files (CLAUDE.md / AGENTS.md / copilot-instructions.md)."

patterns-established:
  - "DEFERRED calibration documented honestly: FLOOR=0.0, structural blocker recorded in docs/loop_runner.md with the corpus finding, pre-fix run id, and what a real calibration requires."
  - "Two demand modes (REAL: set DEMAND_DATABASE_URL; FIXTURE: run scripts/seed_demand_log.py before make loop) documented as EXECUTABLE steps."

requirements-completed: [LOOP-01, LOOP-02, LOOP-03, METRIC-01]

# Metrics
duration: 25min
completed: 2026-06-20
---

# Phase 19 Plan 04: Loop Runner Docs + AI-Instruction Sync Summary

**Operator runbook (docs/loop_runner.md) with DEFERRED calibration finding, three-file AI-instruction sync, and confirmed FLOOR=0.0 with 49 green unit tests**

## Performance

- **Duration:** ~25 min (Task 3 only — Tasks 1 and 2 were completed before this session)
- **Started:** 2026-06-20T~18:00Z
- **Completed:** 2026-06-20
- **Tasks:** 3 (Task 1 done in prior session, Task 2 operator checkpoint + 2 bug fixes, Task 3 done here)
- **Files modified:** 5 (docs/loop_runner.md, CLAUDE.md, AGENTS.md, .github/copilot-instructions.md, this SUMMARY)

## Accomplishments

- Task 1 (prior session): `make loop` target with three-key guard (SANDBOX_DATABASE_URL + GOOGLE_PLACES_API_KEY + OPENAI_API_KEY) + 27 zero-API-cost orchestrator decision-logic unit tests in `tests/unit/test_loop_runner_orchestrator.py`.
- Task 2 (operator, CALIBRATION DEFERRED): Operator ran `make loop`. Found and fixed two bugs (commits `dbf9b1a` + `387f1b3`). Structural corpus finding blocked calibration: per-(neighborhood,cuisine) supply gaps are NOT zeroable in the SF corpus (cuisine places leak across ~20 neighborhoods via Places API). Decision (user-ratified): ship as plumbing-verified; FLOOR stays 0.0; calibration deferred to a different gap construct.
- Task 3 (this session): `docs/loop_runner.md` written with full operator runbook (gate purpose, D-06 CI split, both demand modes with executable FIXTURE seed step, D-01/D-02 caveats, exit codes, PARAPHRASE_PROVIDER override, cleanup gotcha, and prominent DEFERRED calibration section with corpus finding + bug fix commits). All three AI-instruction files synced (`make loop` + Phase 19 note). Unit tests confirmed green (49 passed, FLOOR==0.0 assertion holds).

## Task Commits

1. **Task 1: make loop + orchestrator unit tests** — `3e2e9a3` (feat) + `4f028a6` (docs checkpoint record)
2. **Task 2: operator calibration run** — `dbf9b1a` (fix: metric target-set asymmetry), `387f1b3` (fix: Gemini block-content crash in paraphrase gen)
3. **Task 3: runbook + sync** — `6fcf827` (docs: loop_runner.md), `191a3af` (docs: CLAUDE/AGENTS/copilot sync)

**Plan metadata:** (this commit — docs: complete plan SUMMARY + STATE + ROADMAP)

## Files Created/Modified

- `docs/loop_runner.md` — operator runbook with gate purpose, two demand modes (REAL / FIXTURE with `scripts/seed_demand_log.py`), D-01/D-02 caveats, exit codes 0/1/2, PARAPHRASE_PROVIDER override, cleanup gotcha, DEFERRED calibration section
- `CLAUDE.md` — added `make loop` to Commands block + Phase 19 adaptive data loop note
- `AGENTS.md` — same edits as CLAUDE.md (sync)
- `.github/copilot-instructions.md` — same edits as CLAUDE.md (sync)
- `app/loop/falsifier_core.py` — FLOOR stays 0.0 (no change in this task; confirmed correct)
- `tests/unit/test_loop_runner_orchestrator.py` — 27 tests added in Task 1 (prior commit)
- `tests/unit/test_falsifier_core_recall.py` — FLOOR==0.0 assertion passes (no change needed)

## Decisions Made

- **FLOOR=0.0 stays uncalibrated.** The operator attempted calibration on three fixture gap buckets (Outer Sunset/vietnamese, Inner Sunset/nepalese, Outer Richmond/korean). All three failed to produce a positive delta because the SF corpus is too geographically dense — cuisine places in one neighborhood appear in ~20 other neighborhoods' queries, so `before_hit@k` is already ~1.0 for these buckets after provisioning. No positive-lift calibration is possible with per-(neighborhood,cuisine) gaps and the current Places API behavior. A real calibration requires a cuisine absent or near-absent citywide. Deferred by user decision.
- **PRE-FIX MLflow run `42fdf99657714f1ca17e849ddf0ce787`** (`loop-runner-Outer Sunset-vietnamese`, `delta=-1.000`) documents the metric target-set bug (commit `dbf9b1a`), NOT a calibration result. Explicitly documented in runbook.
- **docs/loop_runner.md is the canonical reference** for the gate; AI instruction files point to it.

## Deviations from Plan

The plan (Task 3 acceptance criteria) specified: "set `FLOOR` in `app/loop/falsifier_core.py` to the exact value approved in Task 2." Task 2's calibration was attempted but DEFERRED — no positive-lift run was possible. The user-ratified decision was to ship with FLOOR=0.0 (the initial default) and document the deferral honestly. This is a deviation from the ratchet instruction, handled under the user-ratified calibration-deferred outcome.

### Auto-fixed Issues (during Task 2 operator run, committed before this session)

**1. [Rule 1 - Bug] Fixed metric target-set asymmetry in loop_runner.py**
- **Found during:** Task 2 (operator gate run)
- **Issue:** `before_hit@k` was scored against `before_v2_ids` (ALL pre-existing embedded IDs), inflating before to ~1.0 and making delta ~-1.0. This is the exact opposite of D-03 which says before_hit@k = 0 by construction (new v2 IDs did not exist before ingest).
- **Fix:** Changed both before and after scoring to use `new_v2_ids` as the target set (the v2 DB-diff). Commit `dbf9b1a`.
- **Files modified:** `scripts/loop_runner.py`
- **Verification:** PRE-FIX run gave delta=-1.000; post-fix scoring design gives before=0 by construction.

**2. [Rule 1 - Bug] Fixed Gemini block-content crash in paraphrase generation**
- **Found during:** Task 2 (operator gate run, paraphrase-gen stage)
- **Issue:** Gemini returns paraphrase content as a list of typed blocks `[{'type':'text','text':'<json>'}]` instead of a plain string. The runner tried to JSON-parse the list directly, raising a parse error and EXIT_INFRA.
- **Fix:** Added block-content unwrapping: iterate the list, extract `text` field from any dict-with-text-key, concatenate. Commit `387f1b3`.
- **Files modified:** `scripts/loop_runner.py`
- **Verification:** Default Gemini paraphrase provider now works; openai/gpt-4o-mini also confirmed working. Noted in runbook.

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs), 1 user-ratified plan deviation (FLOOR stays 0.0, calibration deferred).
**Impact on plan:** Bug fixes were necessary for correctness. Calibration deferral is documented as explicit known debt. No scope creep.

## Issues Encountered

- **SF corpus geography blocks per-neighborhood calibration.** The Places API `"in {neighborhood}"` filter does not partition SF restaurants by neighborhood — results for `"{cuisine} restaurants in Outer Sunset"` appear in results for the same cuisine in ~20 other neighborhoods. This structural property means `before_hit@k` against the v2-diff target cannot be reduced below ~1.0 for a provisioned-populated baseline, making positive-lift impossible for any of the three tested fixture buckets. Documented in `docs/loop_runner.md` under "Floor calibration — DEFERRED".

## Known Stubs

None — all gate logic is wired; `docs/loop_runner.md` names the deferred calibration as an intentional known limitation, not a missing feature. The runbook section "What a real calibration requires" specifies what is needed to resolve it.

## Next Phase Readiness

Phase 19 (and v2.3 Adaptive Data Loop milestone) is complete. The full loop pipeline
(gap-mine → ingest → embed → score → gate → MLflow) runs end-to-end and is plumbing-verified.

Open debt (tracked, not blocking v2.3 close):
- Floor calibration: needs a citywide-absent-cuisine gap construct; see `docs/loop_runner.md`.
- Real-demand run: needs a populated prod `user_query_log`; fixture mode is the current workaround.

---
*Phase: 19-productionized-loop-metric-loop*
*Completed: 2026-06-20*
