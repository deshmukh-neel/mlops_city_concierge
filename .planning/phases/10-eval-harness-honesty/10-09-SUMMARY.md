---
phase: 10-eval-harness-honesty
plan: "09"
subsystem: eval-harness
tags: [security, portability, redaction, probe, gap-closure]
dependency_graph:
  requires: []
  provides: [EVAL-05-verified]
  affects: [scripts/probe_provider_capture.py, tests/unit/test_probe_provider_capture.py]
tech_stack:
  added: []
  patterns: [json.loads(_redact(json.dumps(...))), _scan_fixture_for_secrets helper]
key_files:
  created: []
  modified:
    - scripts/probe_provider_capture.py
    - tests/unit/test_probe_provider_capture.py
decisions:
  - "CR-05 closed: route response_metadata, usage_metadata, tool_calls through _redact(json.dumps); extract _scan_fixture_for_secrets covering regex + env-var values"
  - "CR-04 closed: REPO_ROOT = Path(__file__).resolve().parents[2] replaces hardcoded author path in test_main_help_exits_zero"
metrics:
  duration: "~5m"
  completed: "2026-06-11"
  tasks: 2
  files: 2
---

# Phase 10 Plan 09: Probe Redaction and Portability Summary

Gap closure fixing CR-05 (fail-closed redaction not genuinely fail-closed) and CR-04 (hardcoded absolute path breaks CI).

## What Was Built

**Task 1 (TDD — CR-05):** `scripts/probe_provider_capture.py` now routes all value-bearing fixture fields through `_redact` before write:
- `response_metadata`: `json.loads(_redact(json.dumps(_sanitize_response_metadata(...), default=str)))` — sanitize (defense in depth) then redact
- `usage_metadata`: `json.loads(_redact(json.dumps(..., default=str)))` when not None
- `tool_calls`: `json.loads(_redact(json.dumps(..., default=str)))`

A new `_scan_fixture_for_secrets(text: str) -> bool` helper is extracted from the post-write guard. It checks two independent channels (mirroring `_redact`): (1) `_SECRET_PATTERNS` regexes and (2) runtime values of `_SECRET_ENV_VARS` (len >= 10). The post-write guard in `main()` now calls this helper, so rotated keys with non-regex shapes are caught and the fixture is deleted with return 2.

The module docstring updated to accurately describe the fail-closed guarantee.

**Task 2 (CR-04):** `tests/unit/test_probe_provider_capture.py` adds `REPO_ROOT = Path(__file__).resolve().parents[2]` module constant (mirroring `tests/unit/test_check_eval_gates.py:25`). `test_main_help_exits_zero` uses `cwd=str(REPO_ROOT)` and `str(REPO_ROOT / "scripts" / "probe_provider_capture.py")` so the test passes from any working directory on any machine.

## Commits

| Task | Type | Hash | Description |
|------|------|------|-------------|
| Task 1 (RED) | test | a1f76e3 | add failing RED test for env-var post-write guard (CR-05) |
| Task 1 (GREEN) | feat | a66ab6d | fail-closed redaction + env-var-aware post-write guard (CR-05) |
| Task 2 | fix | 01bc284 | replace hardcoded absolute path with REPO_ROOT resolution (CR-04) |

## Verification

- `grep -c "_redact(json.dumps" scripts/probe_provider_capture.py` → 3 (response_metadata, usage_metadata, tool_calls)
- `grep -c "_SECRET_ENV_VARS" scripts/probe_provider_capture.py` → 5 (definition + _redact + _scan_fixture_for_secrets + docstrings)
- `grep -c "/Users/pnhek" tests/unit/test_probe_provider_capture.py` → 0
- `grep -c "Path(__file__).resolve().parents" tests/unit/test_probe_provider_capture.py` → 1
- `poetry run pytest tests/unit/test_probe_provider_capture.py -q` → 11 passed (10 existing + 1 new env-var guard test)
- `make test` → 1127 passed, 53 skipped

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Rotated-secret test value matched a regex pattern**
- **Found during:** Task 1 RED phase
- **Issue:** Initial candidate secret `ROTATED-not-sk-shaped-XYZ-9876543210` matched the `sk-[A-Za-z0-9_-]{20,}` pattern because `sk-shaped-XYZ-9876543210` starts with `sk-` and is 20+ chars. The test pre-check assertion failed before reaching the guard.
- **Fix:** Changed to `rotated-deepseek-key-9876543210` which bypasses all four `_SECRET_PATTERNS` regexes (verified by inspection).
- **Files modified:** `tests/unit/test_probe_provider_capture.py`

**2. [Rule 1 - Bug] Ruff S603 pre-commit hook rejected subprocess.run**
- **Found during:** Task 2 commit
- **Issue:** Adding `subprocess.run(...)` in the test triggered ruff S603 (subprocess call with untrusted input), causing the pre-commit hook to fail.
- **Fix:** Added `# noqa: S603` on the `subprocess.run(` line, matching the existing pattern in `scripts/check_baselines_fresh.py:75` and `scripts/eval_matrix.py:514`.
- **Files modified:** `tests/unit/test_probe_provider_capture.py`

**3. [Rule 3 - Blocking] `_redact(json.dumps` grep count was 2 not 3**
- **Found during:** Task 1 GREEN verification
- **Issue:** `response_metadata` used a multi-line form `_redact(\n    json.dumps(...)` rather than inline `_redact(json.dumps(`, so the literal grep counted only 2 occurrences.
- **Fix:** Extracted `_sanitized_meta` to a local variable so `response_metadata = json.loads(_redact(json.dumps(_sanitized_meta, default=str)))` is on one line. All 3 fields now match the grep criterion.
- **Files modified:** `scripts/probe_provider_capture.py`

## CRs Closed

| CR | Severity | Status |
|----|----------|--------|
| CR-05 | WARNING (security) | CLOSED — all three value-bearing fields redacted; env-var guard extended; _scan_fixture_for_secrets extracted |
| CR-04 | BLOCKER (portability) | CLOSED — REPO_ROOT resolution replaces hardcoded author path |

## Known Stubs

None.

## Threat Flags

None — changes are within the already-modeled trust boundary (live provider wire -> checked-in fixture). No new network endpoints, auth paths, or schema changes introduced.

## Self-Check: PASSED

- `scripts/probe_provider_capture.py` — exists and contains `_scan_fixture_for_secrets`
- `tests/unit/test_probe_provider_capture.py` — exists and contains `REPO_ROOT`
- Commits a1f76e3, a66ab6d, 01bc284 — all present in git log
- 11/11 probe tests pass; 1127/1127 full suite tests pass
