---
phase: 14
plan: "01"
subsystem: agent-loop
tags: [replay, adapters, flags, experiment-arms]
dependency_graph:
  requires: []
  provides: [REPLAY-01-flag, REPLAY-02-flag, replay_reasoning_state_multi-ABC]
  affects: [app/agent/graph.py, app/agent/adapters/__init__.py, scripts/eval_agent.py]
tech_stack:
  added: []
  patterns: [env-flag build-time closure, keyword-only param, ABC generic default]
key_files:
  created: []
  modified:
    - app/agent/adapters/__init__.py
    - app/agent/graph.py
    - scripts/eval_agent.py
    - tests/unit/test_agent_graph.py
    - tests/unit/test_eval_agent.py
decisions:
  - "replay_reasoning_state_multi added as non-abstract ABC default; NoOpAdapter gets explicit no-op override (D-14-03)"
  - "_prune_for_llm gains preserve_content_blocks keyword param defaulting False; flag-off path byte-identical (D-14-06)"
  - "Both replay flags resolved once at build time and closed over plan() — same precedent as Phase-13 DEC flags"
  - "arm_flags extended with replay_multi_message + replay_content_blocks alongside Phase-13 keys (D-14-02)"
metrics:
  duration: "~15min"
  completed: "2026-06-12"
  tasks_completed: 3
  files_modified: 5
---

# Phase 14 Plan 01: Replay Flag Wiring and Graph Branches Summary

Wire the two Phase-14 replay arms (REPLAY-01 multi-message reasoning-state replay, REPLAY-02 content-block preservation) into the agent loop behind env flags, extend the run-JSON arm_flags self-description, and add the generic multi-replay ABC method — with both flags defaulting OFF and the flag-off path byte-identical to the Phase-13 plateau.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add replay_reasoning_state_multi generic default to ProviderAdapter ABC | 758bbae | app/agent/adapters/__init__.py |
| 2 | Wire both replay flags into graph.py and extend eval_agent arm_flags | 31fa747 | app/agent/graph.py, scripts/eval_agent.py |
| 3 | Add flag-gated graph tests for both replay branches + greppable flag-name test | 38fc47f | tests/unit/test_agent_graph.py |

## Verification

- `poetry run pytest tests/unit/test_adapters.py tests/unit/test_agent_graph.py -q`: 93 passed, 4 skipped
- `make test` (full suite): 1380 passed, 53 skipped, 9 deselected
- `make typecheck`: Success, no issues in 40 source files
- Flag-off byte-identity: `_prune_for_llm` collapses list content to str and `plan()` uses single-message replay path when both flags unset (verified by Task 2/3 assertions)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Existing arm_flags test assertions broke after arm_flags dict was extended**
- **Found during:** Task 3 / full `make test` run
- **Issue:** `tests/unit/test_eval_agent.py::TestArmFlagsAndForcedCommitTelemetry::test_arm_flags_all_off_when_no_env_vars_set` and `test_arm_flags_reflects_env_vars_when_set` asserted the exact Phase-13 four-key arm_flags shape. Adding two Phase-14 replay keys caused both tests to fail.
- **Fix:** Updated both test assertions to include `replay_multi_message` and `replay_content_blocks` alongside the four Phase-13 keys, matching the new six-key contract.
- **Files modified:** tests/unit/test_eval_agent.py
- **Commit:** ea81c21

## Key Decisions Made

- `replay_reasoning_state_multi` is non-abstract (generic ABC default delegates to per-message `replay_reasoning_state`). No `@abstractmethod` → existing 9-test conformance harness unchanged.
- `NoOpAdapter` gets an explicit no-op override for clarity, even though the ABC default would also be a no-op via the empty-additional_kwargs path.
- `_prune_for_llm` receives the `preserve_content_blocks` flag as a keyword-only parameter (not a module-level read), matching the "resolve once at build time, pass as parameter" precedent from graph.py closures.
- The `plan()` REPLAY-01 branch structure: `if _replay_multi_message_enabled: multi_path; else: existing_path_verbatim` — the else branch is byte-identical to the Phase-13 plateau.
- `[skip-baseline]` commit tag used on all three task commits because the flag-off default path produces byte-identical behavior to Phase-13; verified by Task 2/3 inline assertions.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. Two new env-var reads (`REPLAY_MULTI_MESSAGE_ENABLED`, `REPLAY_CONTENT_BLOCKS_ENABLED`) use the same `env_flag()` truthy-set parser as Phase-13 flags (T-14-01 mitigated). No new DB access. No cross-request state leakage (T-14-02 accepted per plan).

## Self-Check: PASSED

- [x] app/agent/adapters/__init__.py exists and has 2 `replay_reasoning_state_multi` definitions
- [x] app/agent/graph.py contains `REPLAY_MULTI_MESSAGE_ENABLED`, `REPLAY_CONTENT_BLOCKS_ENABLED`, `preserve_content_blocks`, `replay_reasoning_state_multi`
- [x] scripts/eval_agent.py contains `replay_multi_message` and `replay_content_blocks` keys
- [x] tests/unit/test_agent_graph.py contains all 4 new test functions
- [x] Commits 758bbae, 31fa747, 38fc47f, ea81c21 exist in git log
- [x] `make test` passes: 1380 passed, 53 skipped
- [x] `make typecheck` passes: no issues
