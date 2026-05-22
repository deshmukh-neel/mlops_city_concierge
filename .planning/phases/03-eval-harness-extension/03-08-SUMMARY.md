---
phase: 03-eval-harness-extension
plan: 08
subsystem: testing
tags: [eval-matrix, scorer-whitelist, pydantic-validator, observability, tdd, gap-closure]

# Dependency graph
requires:
  - phase: 03-eval-harness-extension
    provides: "Plan 03-05 (eval_matrix.py + aggregate_cell_jsons) and Plan 03-07 (baselines + lint) — this plan hardens those interfaces in place."
provides:
  - "Scorer-whitelist filter in _scorer_means_from_cell — only CRITIQUE_THRESHOLDS-registered names admitted into summary.json (CR-01 BLOCKER closed)"
  - "bool exclusion in the numeric isinstance check so bool-disguised-as-numeric values are dropped (IN-04 closed)"
  - "MatrixEntry.reject_double_dash after-validator on provider/model so the '--' cell-filename separator invariant cannot be violated at config-load time (WR-01 closed)"
  - "Aggregator WARNING log when _parse_cell_filename returns None so silent cell drops are observable (WR-01 second leg)"
  - "Top-level overridden_to field in summary.json when --llm-provider-override is set, plus a one-shot INFO log line in run_matrix (IN-02 closed)"
  - "8 new unit tests (705 → 713) pinning all four contracts"
affects: ["Phase 4 category compliance", "Phase 5 rationale-stop alignment", "Phase 6 minimal-edit refinement", "any future plan that touches scripts/eval_matrix.py or app/eval/config.py:MatrixEntry"]

# Tech tracking
tech-stack:
  added: []  # No new dependencies — all changes use existing Python stdlib (logging) and existing pydantic field_validator.
  patterns:
    - "Whitelist-against-canonical-source: filter using `app.agent.critique.checks.CRITIQUE_THRESHOLDS` membership, not a regex on key names. Future scorers are admitted automatically when registered in the canonical dict."
    - "Two-stage field_validator on pydantic models: mode='before' for normalization (strip/cast) + mode='after' for invariant assertions (reject '--'). Additive — preserves existing validator semantics."
    - "Observability over kill-switch: the aggregator emits a WARNING and continues when it sees an unparseable filename, instead of crashing. Operators get a log line; partial runs still produce summary.json."

key-files:
  created: []
  modified:
    - "scripts/eval_matrix.py — import CRITIQUE_THRESHOLDS, whitelist scorers in _scorer_means_from_cell, exclude bool, module-level _log, WARNING on unparseable cell filename, optional llm_provider_override kwarg on aggregate_cell_jsons, overridden_to summary field, INFO log at start of run_matrix when override active, main() threads override into aggregate_cell_jsons"
    - "app/eval/config.py — MatrixEntry.reject_double_dash field_validator (mode='after') rejects '--' in provider/model with a field-named error citing the separator collision"
    - "tests/unit/test_eval_matrix.py — _write_cell_with_aggregate helper + 5 new tests (whitelist, bool exclusion, unparseable-filename warning, overridden_to set, overridden_to omitted)"
    - "tests/unit/test_eval_config.py — 3 new MatrixEntry validator tests (rejects '--' in model, rejects '--' in provider, accepts single dash anywhere)"

key-decisions:
  - "Whitelist via `scorer_name in CRITIQUE_THRESHOLDS` (positive admission) instead of a deny-list of the 6 known polluting keys — new scorers added in future plans flow through automatically, no Phase-3.x deny-list maintenance."
  - "bool exclusion uses a paired `or isinstance(value, bool)` check rather than a try/except — explicit-over-clever, and `not isinstance(value, int | float) or isinstance(value, bool)` is one boolean expression, no exception machinery."
  - "MatrixEntry validator is mode='after' (runs on the already-stripped string) so the error message can quote the actual rejected value rather than the raw pre-strip input."
  - "Aggregator WARNING uses `_log.warning(\"... %s in %s\", filename, dir)` — structured %-format so log aggregators can index per-filename without parsing free text."
  - "overridden_to is a top-level field on the summary dict, NOT a remapping of per-provider scorer keys. PR diff tooling that ignores top-level fields will still produce a clean cross-provider scorer diff; tooling that reads it gets the rebind signal."
  - "aggregate_cell_jsons's new llm_provider_override kwarg defaults to None — backward-compatible with the 4 existing test call sites + the write_summary_json helper."

patterns-established:
  - "TDD execution on a gap-closure plan: 6 atomic commits in strict RED→GREEN pairs (test → feat), zero refactor commits needed (the code stayed simple enough)."
  - "Pre-commit ruff format reformats are re-staged and committed unchanged — the hook is the authority on whitespace, the human is the authority on logic."

requirements-completed: [EVAL-05, EVAL-07]

# Metrics
duration: 7min
completed: 2026-05-22
---

# Phase 03 Plan 08: Scorer Whitelist + Aggregator Fixes Summary

**Scorer-whitelist filter against CRITIQUE_THRESHOLDS + MatrixEntry `--` rejection + bool exclusion + overridden_to summary field — closing CR-01 (BLOCKER), IN-02, IN-04, and WR-01 in one atomic plan before the user runs `APP_ENV=eval make eval-matrix RUNS=3`.**

## Performance

- **Duration:** ~7 min (first commit 2026-05-22T08:13:40Z, last commit 2026-05-22T08:20:43Z)
- **Started:** 2026-05-22T08:13:40Z
- **Completed:** 2026-05-22T08:20:43Z
- **Tasks:** 3 (each TDD: RED + GREEN)
- **Files modified:** 4 (`scripts/eval_matrix.py`, `app/eval/config.py`, `tests/unit/test_eval_matrix.py`, `tests/unit/test_eval_config.py`)

## Accomplishments

- **CR-01 (BLOCKER) closed**: `_scorer_means_from_cell` now filters via `scorer_name in CRITIQUE_THRESHOLDS`. The 6 polluting non-scorer `_mean` keys (`tool_calls_mean`, `results_mean`, `contexts_mean`, `revision_hints_mean`, `committed_stops_mean`, `answer_retrieved_place_coverage_mean`) no longer leak into `summary.json`. The user's upcoming `make eval-matrix RUNS=3` will produce a scorer-only summary, ready for baseline post-processing without manual filtering.
- **IN-04 closed**: bool values (which `isinstance(x, int | float)` matches because `bool` is a subclass of `int`) are now explicitly excluded from the numeric check. Defensive against future regressions where a scorer accidentally emits a bool.
- **WR-01 closed**: `MatrixEntry` now rejects `--` in provider or model strings at config-load time with a field-named `ValidationError` citing the separator collision. The aggregator additionally emits a `WARNING` log when `_parse_cell_filename` returns None, so any stray file in the run dir is observable.
- **IN-02 closed**: When `--llm-provider-override` is set, `summary.json` carries a top-level `overridden_to: <provider>` field and `run_matrix` logs the rebind once at INFO. Downstream PR-diff tooling can detect the rebind without inspecting per-provider keys.
- **Full unit suite**: 713 passed (705 baseline + 8 new tests), zero regressions, 32s wall time.

## Task Commits

Each task was TDD-paired (RED test commit, then GREEN feat commit):

1. **Task 1 RED: failing tests for scorer whitelist + bool exclusion** — `23f4026` (test)
2. **Task 1 GREEN: whitelist scorers + exclude bool in `_scorer_means_from_cell`** — `17f82a4` (feat)
3. **Task 2 RED: failing tests for MatrixEntry `--` reject + aggregator warn** — `0a09f70` (test)
4. **Task 2 GREEN: reject `--` in MatrixEntry + warn on unparseable cells** — `f5d4e9a` (feat)
5. **Task 3 RED: failing tests for `overridden_to` summary field** — `96160a3` (test)
6. **Task 3 GREEN: add `overridden_to` summary field + override log line** — `de46f54` (feat)

No refactor commits — the implementations were small enough that GREEN landed clean.

## Files Created/Modified

- `scripts/eval_matrix.py` — Added `from app.agent.critique.checks import CRITIQUE_THRESHOLDS`, `import logging`, module-level `_log`. `_scorer_means_from_cell` body rewritten to whitelist via `scorer_name in CRITIQUE_THRESHOLDS` and exclude bool. `aggregate_cell_jsons` gained optional `llm_provider_override: str | None = None` kwarg and conditional `overridden_to` field. Aggregator loop emits `_log.warning(...)` on unparseable filename. `run_matrix` emits one `_log.info(...)` when override active. `main()` threads `args.llm_provider_override` into `aggregate_cell_jsons`.
- `app/eval/config.py` — Added `MatrixEntry.reject_double_dash` field validator (`mode="after"`, applies to `provider` and `model`). Raises `ValueError` quoting the rejected value and citing the `'--' is reserved` contract.
- `tests/unit/test_eval_matrix.py` — Added `_write_cell_with_aggregate` helper. Added 5 tests: `test_scorer_means_excludes_non_scorer_keys`, `test_scorer_means_rejects_bool_values_disguised_as_numeric`, `test_aggregate_warns_on_unparseable_cell_filename`, `test_aggregate_records_overridden_to_when_override_set`, `test_aggregate_omits_overridden_to_when_no_override`.
- `tests/unit/test_eval_config.py` — Added 3 tests: `test_matrix_entry_rejects_double_dash_in_model`, `test_matrix_entry_rejects_double_dash_in_provider`, `test_matrix_entry_accepts_single_dash_anywhere`.

## Decisions Made

- **Positive-admission whitelist over deny-list.** New scorers added in Phase 4-6 (e.g. `closure_aware_swap`) will be admitted automatically once registered in `CRITIQUE_THRESHOLDS`. No Phase 3.x maintenance.
- **Paired bool exclusion in the same isinstance check.** Single boolean expression, no try/except — explicit-over-clever per CLAUDE.md.
- **Two-validator pattern on `MatrixEntry`** — preserve the existing `strip_required_text` (mode=before) and add `reject_double_dash` (mode=after) so error messages can quote the actual stripped value.
- **Observability over kill-switch on unparseable filenames.** A WARNING + continue keeps partial runs producing summary.json; an exception would have made the aggregator fragile to a single stray foreign file.
- **`overridden_to` is a top-level field, not a key rebind.** PR diff tooling stays simple; cross-provider scorer diffs stay clean; the rebind signal is opt-in for tooling that reads top-level metadata.

## Deviations from Plan

**None substantive** — plan executed exactly as written. Two micro-deviations worth noting:

1. **Plan acceptance criterion specified `scorer_name in CRITIQUE_THRESHOLDS` as a grep target.** My first implementation used `scorer_name not in CRITIQUE_THRESHOLDS` (early-continue style). The substring `scorer_name in CRITIQUE_THRESHOLDS` is NOT present in `scorer_name not in CRITIQUE_THRESHOLDS` (the space-aware substring check). I refactored to the positive form `if scorer_name in CRITIQUE_THRESHOLDS: out[scorer_name] = float(value)` to satisfy the grep criterion verbatim. The cleaner form is also more idiomatic and ruff-friendly.
2. **Pre-commit `ruff format` reformatted 2 of the 6 commits** before they landed. Standard project behavior (per project memory `feedback_precommit_ruff`); re-staged and committed without further intervention.

## Issues Encountered

None. The plan's `<read_first>` and `<action>` blocks were precise enough that each task landed on the first try.

## User Setup Required

None — purely internal code changes, no environment variables, no infra, no dependencies. The user's next manual step (`APP_ENV=eval make eval-matrix RUNS=3`) is unchanged — it just now produces a clean summary.json.

## Next Phase Readiness

The blocker that gated the user's matrix run is closed:
- `summary.json` will contain only the 8 scorers from `CRITIQUE_THRESHOLDS`, ready for direct baseline post-processing.
- The Phase 4-6 `scorer median >= baseline median + delta` merge gate will diff against the right set of keys.
- The 18-cell dry-run still emits 18 cells (no signature regression).
- All 713 unit tests pass.

**Remaining Phase 3 follow-ups (per VERIFICATION.md, NOT in this plan's scope):**
- CR-02 (ScriptedChatModel singleton fallback) — lower urgency, soft-gated; user can decide Phase 3.1 vs triage to Phase 4.
- The user's manual matrix run + baseline post-process step. Unchanged by this plan.

## TDD Gate Compliance

All 3 tasks followed RED → GREEN. Per-task gate evidence:

| Task | RED commit | GREEN commit | New tests |
|------|------------|--------------|-----------|
| 1 (CR-01 + IN-04) | `23f4026` | `17f82a4` | 2 |
| 2 (WR-01) | `0a09f70` | `f5d4e9a` | 4 (3 in test_eval_config, 1 in test_eval_matrix) |
| 3 (IN-02) | `96160a3` | `de46f54` | 2 |

8 new tests total; matches the plan's "approximately 7 new tests" estimate.

## Self-Check: PASSED

Verified before sign-off:
- `[FOUND]` `.planning/phases/03-eval-harness-extension/03-08-SUMMARY.md` (this file)
- `[FOUND]` commits `23f4026`, `17f82a4`, `0a09f70`, `f5d4e9a`, `96160a3`, `de46f54` all present in `git log --oneline 10414e1..HEAD`
- `[VERIFIED]` `poetry run pytest tests/unit/ -q` → 713 passed, 9 warnings, 32.10s
- `[VERIFIED]` `grep "from app.agent.critique.checks import CRITIQUE_THRESHOLDS" scripts/eval_matrix.py` → line 43
- `[VERIFIED]` `grep "scorer_name in CRITIQUE_THRESHOLDS" scripts/eval_matrix.py` → line 165
- `[VERIFIED]` `grep "isinstance(value, bool)" scripts/eval_matrix.py` → line 161
- `[VERIFIED]` `grep "reject_double_dash" app/eval/config.py` → line 212
- `[VERIFIED]` `grep "_log = logging.getLogger" scripts/eval_matrix.py` → line 60
- `[VERIFIED]` `grep "skipping unparseable cell file" scripts/eval_matrix.py` → line 227
- `[VERIFIED]` `grep "overridden_to" scripts/eval_matrix.py` → 4 lines (docstring, return assignment, comment, etc.)
- `[VERIFIED]` `inspect.signature(aggregate_cell_jsons)` includes `llm_provider_override`
- `[VERIFIED]` `MatrixEntry(provider='openai', model='gpt-4--turbo')` raises `ValidationError`
- `[VERIFIED]` 18-cell dry-run still emits 18 cells (`APP_ENV=eval poetry run python scripts/eval_matrix.py --dry-run --runs 3 --matrix-config configs/eval_matrix.yaml | wc -l` → 18)
- `[VERIFIED]` `poetry run ruff check` + `ruff format --check` → all checks passed, 4 files already formatted
- `[VERIFIED]` Plan-07 deliverables intact: `configs/eval_baselines/*.json` (3 files), `scripts/check_baselines_fresh.py`, `.github/workflows/ci.yml` all present and unmodified

---
*Phase: 03-eval-harness-extension*
*Plan: 08*
*Completed: 2026-05-22*
