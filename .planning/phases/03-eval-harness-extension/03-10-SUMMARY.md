---
phase: 03-eval-harness-extension
plan: 10
subsystem: eval-harness / ci-gating
tags: [gap-closure, wr-02, in-01, check-baselines-fresh, ci-yml, hard-gate, p9, security-hardening]
requirements: [EVAL-07]
dependency_graph:
  requires:
    - "Plan 03-07 (`scripts/check_baselines_fresh.py` + `lint-baselines` job — Plan 03-10 hardens the WR-02 / IN-01 surfaces those introduced)"
  provides:
    - "Loud-fail-on-error posture for the hard baseline-lint gate (rc=2 distinct from rc=1 stale-baseline)"
    - "GitHub-documented safe shape for BASE_SHA in the lint-baselines CI step (env block + double-quoted shell var)"
  affects:
    - "Phase 4-6 PRs that touch app/agent/ — gate behaviour for the 4 truth-table branches is unchanged; only the silent-pass-on-environmental-brokenness path now loudly fails"
tech_stack:
  added: []
  patterns:
    - "Subprocess error surfacing pattern: catch FileNotFoundError + check rc, re-raise as RuntimeError naming the failing tool + argv + stderr"
    - "Exit-code stratification: rc=0 (pass), rc=1 (rule violation), rc=2 (infrastructure failure) so CI signal disambiguates rule vs environment"
    - "GitHub Actions env-block-over-interpolation: `env: VAR: ${{ … }}` + `\"$VAR\"` inside `run:`"
key_files:
  created: []
  modified:
    - scripts/check_baselines_fresh.py
    - tests/unit/test_check_baselines_fresh.py
    - .github/workflows/ci.yml
decisions:
  - "Use rc=2 for RuntimeError (infrastructure failure), not rc=1 — CI logs need to distinguish 'fix your baseline' from 'fix your CI environment'"
  - "Empty-string positional BASE_SHA raises loudly rather than silently falling back to origin/main — accidental robustness was hiding workflow misconfiguration"
  - "WR-02 tests patch `script.subprocess.run` (one level deeper than the existing tests, which patch `_run_git` wholesale) — keeps the existing 9-test seam intact"
metrics:
  duration: "~6 minutes (TDD red/green for Task 1, single-edit verify for Task 2)"
  completed_date: "2026-05-22"
  tasks: 2
  tests_added: 4
  files_modified: 3
  commits: 3
---

# Phase 03 Plan 10: Baseline Lint Hardening Summary

Convert the EVAL-07 / P9 stale-baseline hard gate from silent-pass-on-environmental-brokenness to loud-fail-on-environmental-brokenness, and remove the GitHub Actions run-string interpolation anti-pattern from the `lint-baselines` CI step.

## What Shipped

**WR-02 (silent-pass-on-error) closed:**
- `scripts/check_baselines_fresh.py:_run_git` now raises `RuntimeError` with `rc` + argv + stderr on git non-zero exit (was: silently returned empty stdout, which caused a missing/shallow `origin/main` ref or bogus `BASE_SHA` to trivially pass the gate).
- `_run_git` catches `FileNotFoundError` and re-raises as `RuntimeError` naming `git` (was: confusing Python-internal traceback).
- `_resolve_base` rejects explicit empty-string positional `BASE_SHA` or `--merge-base` (was: empty string was falsy, silently fell back to `origin/main`).
- `main()` translates RuntimeError into exit code 2 with a stderr message, distinct from rc=1 stale-baseline failures.
- Docstring truth table extended with the new rc=2 row.

**IN-01 (GitHub Actions interpolation anti-pattern) closed:**
- `.github/workflows/ci.yml` `lint-baselines.Check baselines fresh vs PR base` step now reads `BASE_SHA` from an `env:` block and consumes it as `"$BASE_SHA"` inside the `run:` script. The `${{ github.event.pull_request.base.sha }}` interpolation no longer appears in any `run:` string.

**4 new regression tests** in `tests/unit/test_check_baselines_fresh.py` (under a clearly-labelled "WR-02 loud-fail regression tests (Plan 03-10)" section, patching `subprocess.run` one level deeper than the existing `_run_git`-wholesale-patch tests):
- `test_run_git_raises_on_nonzero_return_code` — pins the rc-surfacing contract.
- `test_run_git_raises_actionable_error_when_git_binary_missing` — pins the FileNotFoundError-to-actionable-RuntimeError conversion.
- `test_main_exits_non_zero_when_base_sha_is_empty_string` — pins the empty-string-positional rejection.
- `test_run_git_still_returns_stdout_on_success` — backward-compat pin for the rc=0 happy path (which was previously untested at the real-`subprocess.run` level because the 9 existing tests stub `_run_git` wholesale).

## Tasks

| Task | Name | Commit(s) | Files |
| ---- | ---- | --------- | ----- |
| 1 (RED) | Failing tests for loud-fail _run_git + empty BASE_SHA reject | `23236e9` | `tests/unit/test_check_baselines_fresh.py` |
| 1 (GREEN) | Loud-fail _run_git + reject empty BASE_SHA (WR-02 fix) | `f58df9c` | `scripts/check_baselines_fresh.py` |
| 2 | CI lint-baselines env block + "$BASE_SHA" (IN-01 fix) | `bedf585` | `.github/workflows/ci.yml` |

## Verification

- `poetry run pytest tests/unit/test_check_baselines_fresh.py -v` → **13 passed** (9 pre-existing truth-table + 4 new WR-02). Acceptance criterion met.
- `poetry run pytest tests/unit/ -q --no-cov` → **709 passed** (was 705 pre-plan; +4 new tests, no regressions in the broader suite).
- `poetry run ruff check` + `ruff format --check` on changed files → all clean.
- `grep -c "raise RuntimeError" scripts/check_baselines_fresh.py` → **4** (≥3 required: FileNotFoundError path, rc!=0 path, empty BASE_SHA path, empty --merge-base path).
- `grep -c "FileNotFoundError" scripts/check_baselines_fresh.py` → **2** (≥1 required).
- Stdlib-only invariant for `scripts/check_baselines_fresh.py`: `grep -E "^(import|from) " scripts/check_baselines_fresh.py | grep -vE "(argparse|subprocess|sys|collections|__future__)"` → empty.
- End-to-end empty-BASE_SHA: `poetry run python scripts/check_baselines_fresh.py ""` → stderr `BASE_SHA positional argument was the empty string; …`, exit code **2**. The rc=2 RuntimeError exit path is functional.
- YAML lint (Task 2 verify snippet): `lint-baselines` job has `env.BASE_SHA == "${{ github.event.pull_request.base.sha }}"`, `run` has no `${{`, `run` references `"$BASE_SHA"`, `if: github.event_name == 'pull_request'` preserved, all 7 jobs present (`eval-matrix`, `integration-cloud`, `lint`, `lint-baselines`, `migrations`, `test`, `typecheck`), no new `continue-on-error: true`.

## Truth-Table Behaviour Preserved

The 4 pass branches of the existing truth table (Plan 03-07) are unchanged:

| agent_changed | baselines_changed | [skip-baseline] | exit |
| ------------- | ----------------- | --------------- | ---- |
| T             | F                 | F               | 1    |
| T             | T                 | F               | 0    |
| F             | *                 | *               | 0    |
| T             | F                 | T               | 0    |

The new behaviour only fires on **infrastructure failure** (rc=2): missing git binary, git rc != 0, or explicit empty-string BASE_SHA. None of those previously had test coverage because the existing tests stub `_run_git` wholesale.

## Deviations from Plan

**None — plan executed exactly as written.**

The plan explicitly anticipated the test-seam concern (existing tests monkeypatch `_run_git`; new tests patch `subprocess.run` one level deeper) and the implementation followed that exact pattern. Both the existing 9 tests and the 4 new tests passed without modification to the existing fixtures.

One small implementation detail beyond the strict letter of the plan: `_resolve_base` also rejects an explicit empty-string `--merge-base` flag (the plan called this out in the "New control flow" section but the explicit emphasis was on the positional). This is the same loud-fail principle, applied consistently to both argv sources. No regression risk: the existing tests pass `--merge-base "deadbeef"`, never the empty string.

## Threat Flags

None — the changes are pure defense-in-depth on the existing trust boundary (`subprocess.run(["git", ...])` → Python, and `${{ … }}` → shell). No new endpoints, no new data flows, no new file access.

## Self-Check: PASSED

- `[ -f scripts/check_baselines_fresh.py ]` → FOUND
- `[ -f tests/unit/test_check_baselines_fresh.py ]` → FOUND
- `[ -f .github/workflows/ci.yml ]` → FOUND
- Commit `23236e9` (test RED) → FOUND in `git log --all`
- Commit `f58df9c` (fix GREEN) → FOUND in `git log --all`
- Commit `bedf585` (ci) → FOUND in `git log --all`
- All 4 new test functions present (grep verified): `test_run_git_raises_on_nonzero_return_code`, `test_run_git_raises_actionable_error_when_git_binary_missing`, `test_main_exits_non_zero_when_base_sha_is_empty_string`, `test_run_git_still_returns_stdout_on_success`
- `poetry run pytest tests/unit/test_check_baselines_fresh.py -v` → 13 passed
- Full unit suite `poetry run pytest tests/unit/ -q --no-cov` → 709 passed
