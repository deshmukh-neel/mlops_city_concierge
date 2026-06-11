---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "07"
subsystem: eval-harness
tags: [staleness-gate, probe-fixtures, tdd, base-04, wr-10]
dependency_graph:
  requires: []
  provides:
    - scripts/check_baselines_fresh.py WATCH_PREFIXES watch-set (D-11-21 / BASE-04)
    - scripts/probe_provider_capture.py type-faithful additional_kwargs redaction (WR-10)
  affects:
    - tests/unit/test_check_baselines_fresh.py
    - tests/unit/test_adapters.py
tech_stack:
  added: []
  patterns:
    - "WATCH_PREFIXES list replaces AGENT_PREFIX scalar: any(p.startswith(prefix) for prefix in WATCH_PREFIXES)"
    - "json.loads(_redact(json.dumps(v, default=str))) for type-faithful redaction"
key_files:
  created: []
  modified:
    - scripts/check_baselines_fresh.py
    - tests/unit/test_check_baselines_fresh.py
    - scripts/probe_provider_capture.py
    - tests/unit/test_adapters.py
decisions:
  - "D-11-21: WATCH_PREFIXES = ['app/agent/', 'app/llm_factory.py', 'configs/eval_matrix'] — bare prefix catches both eval_matrix.yaml and eval_matrix_refinement.yaml"
  - "WR-10: additional_kwargs values route through json.loads(_redact(json.dumps(v, default=str))) matching the response_metadata/usage_metadata/tool_calls pattern; redaction never bypassed"
metrics:
  duration: "5 minutes"
  completed: "2026-06-11"
  tasks_completed: 2
  files_modified: 4
---

# Phase 11 Plan 07: Staleness and Fixture Fidelity Summary

**One-liner:** Extended check_baselines_fresh.py watch-set to cover app/llm_factory.py + configs/eval_matrix* (D-11-21/BASE-04), and fixed probe_provider_capture.py to preserve real dict/list types in additional_kwargs fixture values instead of Python repr strings (WR-10).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for WATCH_PREFIXES extension | b100c24 | tests/unit/test_check_baselines_fresh.py |
| 1 (GREEN) | Extend staleness watch-set (D-11-21 / BASE-04) | e75e02e | scripts/check_baselines_fresh.py |
| 2 (RED) | WR-10 type-fidelity tests for additional_kwargs | 7b52d93 | tests/unit/test_adapters.py |
| 2 (GREEN) | WR-10 probe fixture type-faithful redaction fix | 3aba2b0 | scripts/probe_provider_capture.py |

## What Was Built

### Task 1: BASE-04 / D-11-21 Watch-Set Extension

`check_baselines_fresh.py` previously only watched `app/agent/` for staleness. Provider branches, thinking policies, and temperature clamps live in `app/llm_factory.py`; eval matrix entries that determine which models are measured live in `configs/eval_matrix*.yaml`. Changes to either file can change measured behavior without touching `app/agent/`, bypassing the staleness gate.

**Fix:** Replaced the scalar `AGENT_PREFIX = "app/agent/"` with:
```python
WATCH_PREFIXES = [
    "app/agent/",
    "app/llm_factory.py",
    "configs/eval_matrix",   # bare prefix matches both .yaml files
]
```

Updated `_agent_changed` body to `any(p.startswith(prefix) for prefix in WATCH_PREFIXES)`. Function name, caller in `main()`, and all exit-code semantics unchanged (minimal blast radius). Updated docstring, argparse description, `_format_stale_error`, and Branch 3 output message to use "watch-set" language. stdlib-only constraint preserved (no new imports).

**Tests added (6):**
- `test_llm_factory_change_triggers_stale_gate` — exits 1 without baseline refresh
- `test_eval_matrix_yaml_change_triggers_stale_gate` — exits 1 for eval_matrix.yaml
- `test_eval_matrix_refinement_yaml_change_triggers_stale_gate` — exits 1 for refinement yaml
- `test_agent_file_change_still_triggers_stale_gate_no_regression` — no regression on app/agent/
- `test_llm_factory_change_with_baseline_refresh_passes` — exits 0 with baseline refresh
- `test_unrelated_change_still_passes` — exits 0 for README.md / unrelated files

All 21 tests in the file pass.

### Task 2: WR-10 Probe Fixture Type Fidelity

`probe_provider_capture.py` line 220 called `_redact(message.additional_kwargs[k])` which invoked `str(value)` on the value, collapsing dict/list payloads like `{"reasoning": {"tokens": 42}}` to the Python repr string `"{'reasoning': {'tokens': 42}}"`. This made adapter fixture tests never see real-typed dict/list values.

**Fix:** Route `additional_kwargs` values through the same type-faithful pattern already used for `response_metadata`, `usage_metadata`, and `tool_calls`:
```python
add_kwargs_values = {
    k: json.loads(_redact(json.dumps(message.additional_kwargs[k], default=str)))
    for k in add_kwargs_keys
}
```
Secrets are still redacted — `_redact` is still called on the serialized form. The post-write `_scan_fixture_for_secrets` fail-closed guard is unchanged.

**Tests added (4):**
- `test_wr10_dict_additional_kwargs_value_preserved_as_dict` — dict survives as dict after redaction
- `test_wr10_list_additional_kwargs_value_preserved_as_list` — list survives as list
- `test_wr10_secrets_in_additional_kwargs_are_still_redacted` — sk-* secrets still removed
- `test_wr10_synthetic_fixture_roundtrips_dict_value` — full fixture write + reload preserves dict type

All 58 tests in test_adapters.py pass (4 skip CI-safely when live fixtures are absent).

## Verification Results

```
tests/unit/test_check_baselines_fresh.py  21 passed
tests/unit/test_adapters.py               58 passed, 4 skipped
Full suite (make test):                1169 passed, 53 skipped
ruff check scripts/check_baselines_fresh.py scripts/probe_provider_capture.py: clean
python scripts/check_baselines_fresh.py --help: exits 0
```

## Acceptance Criteria Verification

### Task 1
- `grep -n "WATCH_PREFIXES" scripts/check_baselines_fresh.py` shows list containing `app/llm_factory.py` and `configs/eval_matrix` [PASS]
- `_agent_changed` uses `any(p.startswith(prefix) for prefix in WATCH_PREFIXES)` [PASS]
- Dry-run test for `app/llm_factory.py` (no refresh) asserts exit 1 [PASS]
- Existing `app/agent/` no-refresh test still asserts exit 1 [PASS]
- `grep -n "^import\|^from" scripts/check_baselines_fresh.py` shows only stdlib imports [PASS]

### Task 2
- `grep -n "json.loads(_redact(json.dumps(" scripts/probe_provider_capture.py` shows additional_kwargs path at line 229 [PASS]
- Test asserts dict-valued additional_kwargs entry survives as dict [PASS]
- Test asserts planted sk-* secret in dict value is still redacted [PASS]
- `poetry run pytest tests/unit/test_adapters.py -k "fixture or additional_kwargs"` exits 0 [PASS]

## Deviations from Plan

None — plan executed exactly as written. TDD RED/GREEN cycle followed for both tasks.

## Known Stubs

None.

## Threat Flags

No new trust boundaries introduced. WR-10 fix routes through the existing `_redact` + `_scan_fixture_for_secrets` guards — type fidelity never bypasses the secret-scan path (T-11-18 mitigated).

## Self-Check: PASSED

Files modified:
- [x] scripts/check_baselines_fresh.py exists and contains WATCH_PREFIXES
- [x] tests/unit/test_check_baselines_fresh.py has 21 tests all passing
- [x] scripts/probe_provider_capture.py has type-faithful additional_kwargs redaction
- [x] tests/unit/test_adapters.py has WR-10 tests

Commits verified:
- [x] b100c24 test(11-07): add failing tests for WATCH_PREFIXES extension
- [x] e75e02e feat(11-07): extend staleness watch-set to llm_factory + eval_matrix
- [x] 7b52d93 test(11-07): add WR-10 type-fidelity tests for additional_kwargs in probe fixtures
- [x] 3aba2b0 fix(11-07): WR-10 type-faithful additional_kwargs redaction in probe fixtures
