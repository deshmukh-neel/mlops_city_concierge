---
phase: 13-decisiveness-experiment-arms
plan: "10"
subsystem: config + agent/revision + docs + planning
tags: [sc-3, wr-02, wr-09, env_flag, dry, flag-hygiene, respecification, gap-closure]
dependency_graph:
  requires: [13-09]
  provides: [env_flag DRY helper, SC-3 respecification, WR-02 split-read closed, WR-09 DRY closed]
  affects:
    - app/config.py
    - app/agent/graph.py
    - app/agent/revision.py
    - scripts/eval_agent.py
    - tests/unit/test_config.py
    - .planning/ROADMAP.md
    - docs/decisiveness_arm_verdicts.md
tech_stack:
  added: []
  patterns: [single-source-of-truth env_flag helper, live flag read per call instead of import-time constant]
key_files:
  created: []
  modified:
    - app/config.py (added env_flag helper)
    - app/agent/graph.py (VIABILITY_CONTRACT_ENABLED + PARALLEL_TOOL_EXECUTION_ENABLED via env_flag)
    - app/agent/revision.py (live env_flag read; _VIABILITY_CONTRACT_ENABLED constant removed)
    - scripts/eval_agent.py (arm_flags boolean reads + flag_enabled via env_flag)
    - tests/unit/test_config.py (env_flag truthiness tests added)
    - .planning/ROADMAP.md (SC-3 respecified to absolute-latency-for-future-baseline)
    - docs/decisiveness_arm_verdicts.md (A3 SC-3 respecification echo block added)
decisions:
  - "SC-3 closed at zero spend: criterion 3 respecified from 'measurable latency reduction' to 'absolute tool_exec_seconds recorded for future-baseline use'; discovered constraint (Phase-12 step_telemetry=None) annotated in both ROADMAP and A3 verdict"
  - "WR-09 closed: env_flag(name) in app/config.py is the single truthy-set parser; all six prior inline copies replaced; graph and arm_flags arm_flags self-description provably parse identically"
  - "WR-02 closed: _VIABILITY_CONTRACT_ENABLED module constant removed; _diagnose_last_tool_result reads flag live via env_flag so DEC-01 (graph build-time) and DEC-03 (critique scoping) cannot desync"
metrics:
  duration: "5m"
  completed: "2026-06-12"
  tasks: 3
  files: 7
---

# Phase 13 Plan 10: SC-3 Respecify and Flag Hygiene Summary

**One-liner:** Respecify ROADMAP SC-3 to absolute-latency-for-future-baseline (zero-spend, D-13-02-consistent); add a single env_flag helper that replaces six inlined truthy-set parses (WR-09 DRY); close the import-time/build-time split-read hazard on VIABILITY_CONTRACT_ENABLED (WR-02).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Respecify ROADMAP SC-3 + echo constraint in A3 verdict | edf7174 | .planning/ROADMAP.md, docs/decisiveness_arm_verdicts.md |
| 2 | Add env_flag helper; replace inlined truthy-set parses (WR-09) | 65bb002 | app/config.py, app/agent/graph.py, app/agent/revision.py, scripts/eval_agent.py, tests/unit/test_config.py |
| 3 | Live-read VIABILITY_CONTRACT_ENABLED in revision.py; remove constant (WR-02) | 7d73b04 | app/agent/revision.py |

## What Was Built

### Task 1: SC-3 Respecification

**Problem:** Phase 13 ROADMAP success criterion 3 required "measurable gpt-4o-mini latency reduction at n=5 recorded in run JSON". This is structurally unmeasurable: the Phase-12 comparison-floor run dirs predate the INST-04 step_telemetry instrumentation — all Phase-12 run files have `step_telemetry: None`, no `tool_exec_seconds` field. No valid before-point exists for computing a reduction delta.

**Path (a) rejected:** A flag-off control run would provide a valid sequential baseline, but D-13-02 hard-caps the phase at ≤4 full live matrix runs and all 4 are already consumed. A 5th run would break a locked decision. CONTEXT also forbids billing top-ups.

**Path (b) applied:** Respecify the criterion as "absolute latency recorded for future-baseline use." The absolute `tool_exec_seconds` values ARE in the A3 run JSON. A later phase that regenerates the Phase-12 floor with step_telemetry CAN compute the delta.

**Changes made:**
- `.planning/ROADMAP.md` criterion 3: rewritten to require absolute `tool_exec_seconds` recorded for future-baseline use, with a one-line parenthetical naming the discovered constraint (Phase-12 `step_telemetry=None`).
- `docs/decisiveness_arm_verdicts.md` A3 Latency Analysis: added SC-3 respecification block after the raw tool_exec_seconds values, stating: the ROADMAP criterion has been respecified, the constraint is annotated not hidden, and the A3 arm run IS the future-baseline artifact.

The A3 anchor-regression FAIL finding (refinement_cheaper 0.000 vs 1.000 baseline) and the honest null result ("No arm cleared the INST-05 falsifier bar") are unchanged.

### Task 2: env_flag Helper (WR-09 DRY)

**Problem:** Boolean env-flag parsing was inlined 6 times across graph.py (2), eval_agent.py (3), revision.py (1) using the idiom `os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}`. CLAUDE.md calls out DRY violations aggressively. Any future change to the truthy set would require 6 coordinated edits; a missed copy would desync graph behavior from the run-JSON arm_flags self-description (T-13-10-02).

**Fix:** Added `def env_flag(name: str) -> bool` to `app/config.py` as a module-level function (not a Settings field, since flags are read from `os.environ` live). The truthy set matches the canonical REFINEMENT_STRUCTURED_PLAN_ENABLED precedent in `app/main.py`.

**Sites replaced:**
- `app/agent/graph.py` lines 305-310: `VIABILITY_CONTRACT_ENABLED` and `PARALLEL_TOOL_EXECUTION_ENABLED` boolean reads (FORCED_COMMIT_STEP int read left as-is)
- `scripts/eval_agent.py` lines 929-937 (arm_flags assembly): `viability_contract` and `parallel_tool` boolean reads
- `scripts/eval_agent.py` line ~1180: `flag_enabled = flag_raw.strip().lower() in {...}` for REFINEMENT_STRUCTURED_PLAN_ENABLED
- `app/agent/revision.py` module-level constant (later removed in Task 3)

**Tests added (tests/unit/test_config.py):**
- `test_env_flag_truthy_values` (11 parametrized cases): "1", "true", "yes", "on", and case/whitespace variants all return True
- `test_env_flag_falsy_values` (9 parametrized cases): "0", "false", "off", "no", and other non-truthy values return False
- `test_env_flag_unset_returns_false`: unset env var returns False

### Task 3: WR-02 Split-Read Risk Closed

**Problem:** `revision.py` read `VIABILITY_CONTRACT_ENABLED` at **module import time** into `_VIABILITY_CONTRACT_ENABLED`. `graph.py` reads the same flag at **build time** (inside `build_agent_graph()`). These are the two co-tuned halves of the A1 arm — DEC-01 (prompt addendum) and DEC-03 (critique scoping). In the normal code path, module import and graph build happen together, so they're identical. But under `monkeypatch.setenv` after import (or any env change between module load and graph build in prod), the two halves can read different values and desync.

**Fix:** Removed the `_VIABILITY_CONTRACT_ENABLED` module-level constant entirely. The `_diagnose_last_tool_result` gate now calls `env_flag("VIABILITY_CONTRACT_ENABLED")` live per call. Since the flag is off by default and changes only at process start (arm run setup), the live read has identical semantics in practice, and the desync hazard is eliminated by construction.

**Flag-off byte-identity preserved:** The live read is still inside the existing `if hint.reason == "low_similarity" and env_flag(...):` gate — flag-off runs never reach `all_slots_viable`, same as before (T-13-10-03 mitigation).

**Tests confirmed green:**
- `test_flag_off_low_similarity_fires_as_before` — flag-off path unchanged
- `test_flag_on_all_viable_low_similarity_suppressed` — flag-on suppression works with live read
- `test_flag_on_not_all_viable_low_similarity_fires` — flag-on, not-all-viable still fires

The existing tests use `importlib.reload(rev)` + `monkeypatch.setenv` — with a live read the reload is now a no-op (env is already set before the call), but the tests remain valid and green.

## Deviations from Plan

None — plan executed exactly as written. The `app/main.py` REFINEMENT_STRUCTURED_PLAN_ENABLED read was also replaced by `env_flag` in eval_agent.py's copy (per task 2 scope: "the flag_enabled read at ~1180"), which is inside scripts/eval_agent.py (not app/main.py), so no scope creep.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or trust boundaries introduced.

## Verification Results

- `grep -qi "absolute" .planning/ROADMAP.md` — FOUND
- `grep -qi "future-baseline\|future baseline" .planning/ROADMAP.md` — FOUND
- `grep -qi "future baseline\|future-baseline" docs/decisiveness_arm_verdicts.md` — FOUND
- `grep -qi "step_telemetry" docs/decisiveness_arm_verdicts.md` — FOUND
- `grep -c "anchor regression" docs/decisiveness_arm_verdicts.md` — 6 (unchanged)
- `grep -c "No arm cleared" docs/decisiveness_arm_verdicts.md` — 2 (unchanged)
- `git status configs/eval_baselines/` — clean (no baselines written)
- `grep -rc 'strip().lower() in {"1", "true", "yes", "on"}' app/agent/graph.py scripts/eval_agent.py` — 0 (all replaced)
- `grep -rn "_VIABILITY_CONTRACT_ENABLED" app/ scripts/` — no matches (constant removed)
- `grep -q 'env_flag("VIABILITY_CONTRACT_ENABLED")' app/agent/revision.py` — FOUND (live read)
- `poetry run pytest tests/unit/test_config.py tests/unit/test_agent_revision.py tests/unit/test_agent_prompts.py tests/unit/test_eval_agent.py tests/unit/test_graph_forced_commit.py -q` — 231 passed
- Phase-7 grep gate: `poetry run pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` — 1 passed
- `make test` — 1376 passed, 53 skipped, 17 warnings (full suite clean)

## Self-Check: PASSED

Files verified:
- app/config.py — exists; `def env_flag` present; `import os` at top
- app/agent/graph.py — uses `env_flag("VIABILITY_CONTRACT_ENABLED")` and `env_flag("PARALLEL_TOOL_EXECUTION_ENABLED")` at build time; no inlined truthy-set parse
- app/agent/revision.py — `_VIABILITY_CONTRACT_ENABLED` constant absent; `env_flag("VIABILITY_CONTRACT_ENABLED")` live read in `_diagnose_last_tool_result`; `from app.config import env_flag` imported
- scripts/eval_agent.py — `env_flag` imported from app.config; used in arm_flags and flag_enabled; no inlined boolean truthy-set
- tests/unit/test_config.py — 31 tests including 3 new env_flag tests; all pass
- .planning/ROADMAP.md — criterion 3 contains "absolute" and "future-baseline" and constraint annotation
- docs/decisiveness_arm_verdicts.md — SC-3 respecification block present after raw tool_exec_seconds; anchor-regression FAIL and null result unchanged

Commits verified: edf7174, 65bb002, 7d73b04 — all present in git log.
