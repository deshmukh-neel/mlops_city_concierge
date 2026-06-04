---
phase: 07-prompt-rubric-decoupling
plan: 03
subsystem: eval-matrix
tags:
  - eval-matrix
  - reasoning-models
  - falsifier
  - logged-not-gated
dependency_graph:
  requires:
    - configs/eval_matrix_refinement.yaml (existing entries[] + scenarios[] shape, Phase 6 plan 06-07)
    - app/eval/config.py::MatrixEntry validators (StrictStr env, reject_double_dash, unique provider/model)
    - app/eval/config.py::EvalMatrixConfig.entries_are_unique
    - scripts/eval_matrix.py::iter_cells (per-entry env propagation, MEDIUM-1 preservation)
  provides:
    - openai/gpt-5-mini cell in refinement_cheaper matrix (run on `make eval-matrix-refinement`)
    - PROMPT-05 falsifier wiring — plan 07-07 reads gpt-5-mini median from post-Phase-7 baseline
  affects:
    - configs/eval_baselines/eval_matrix_refinement.json (will gain a gpt-5-mini cell on next regen — plan 07-07)
tech_stack:
  added: []
  patterns:
    - Logged-not-gated matrix entry (DeepSeek precedent, D-04-11 / D-06-09 / D-07-08)
    - YAML-string env value via Pydantic StrictStr (Phase 6 / D-06-10)
key_files:
  created: []
  modified:
    - configs/eval_matrix_refinement.yaml (+10 lines: comment block + openai/gpt-5-mini entry)
decisions:
  - D-07-08 applied: gpt-5-mini is logged-not-gated — merge gate stays at openai/gpt-4o-mini only
  - D-07-09 applied: reasoning stays ENABLED on gpt-5-mini (no thinking-disable env toggle) so the PROMPT-05 falsifier remains meaningful
metrics:
  duration: ~10 min
  completed: 2026-06-03
requirements_addressed:
  - PROMPT-05
---

# Phase 7 Plan 3: gpt-5-mini Matrix Entry Summary

Wired `openai/gpt-5-mini` into `configs/eval_matrix_refinement.yaml` as a logged-not-gated cell so PROMPT-05's falsifier signal (median > 0 across 5 runs at temp=1.0) can be measured by plan 07-07's re-baseline, without changing the merge-gate scope (still `openai/gpt-4o-mini` only).

## Objective

Make the cell `(openai, gpt-5-mini) × refinement_cheaper` runnable end-to-end via `make eval-matrix-refinement` so plan 07-07 can produce the empirical PROMPT-05 measurement. Without this YAML wiring the cell would never iterate; without the cell iterating the falsifier cannot be observed.

## What Was Built

### Task 1 — `configs/eval_matrix_refinement.yaml` entry append

| Block | Provider | Model | Env | Gate semantics |
|-------|----------|-------|-----|----------------|
| Existing | `openai` | `gpt-4o-mini` | `REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"` | **Gated** — strict 1.0 merge gate |
| Existing | `deepseek` | `deepseek-chat` | `REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"` | Logged-not-gated (D-04-11 / D-06-09) |
| **NEW (this plan)** | `openai` | `gpt-5-mini` | `REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"` | **Logged-not-gated** (D-07-08) |

Insertion is purely additive — the two existing entries are byte-identical in the diff (10-line insert, 0 deletions). Comment header on the new block names D-07-08 + D-07-09 so future readers know why gpt-5-mini is here and why reasoning is deliberately not disabled.

**Commit:** `43aaeb3` — `feat(07-03): add openai/gpt-5-mini to refinement matrix (logged-not-gated)`

## Verification

| Check | Command | Result |
|-------|---------|--------|
| Matrix loads (Pydantic) | `load_eval_matrix('configs/eval_matrix_refinement.yaml')` | OK, 3 entries |
| `(openai, gpt-5-mini)` present | introspection of `m.entries` | ✓ |
| `env` is `{"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"}` (str, not bool) | introspection | ✓ |
| Pre-existing entries byte-identical | `git diff configs/eval_matrix_refinement.yaml` | Insert-only, 0 lines removed |
| Structural check exit 0 | `scripts/eval_matrix.py --structural-check` (absolute path) | `OK — matrix has 3 cell(s)` |
| Dry-run lists gpt-5-mini cell | `scripts/eval_matrix.py --dry-run --runs 1 --llm-provider-override scripted` | `scripted/gpt-5-mini :: refinement_cheaper :: --run-0` printed |
| Merge-gate scope unchanged | `git grep -n gpt-5-mini Makefile scripts/check_baselines_fresh.py` | No matches (gate stays at gpt-4o-mini only) |
| `gpt-4o-mini` references | `git grep -n gpt-4o-mini Makefile scripts/check_baselines_fresh.py` | Single usage-example comment at `Makefile:102` (unchanged) |
| Validator constraints | Manual: `--` absent from "gpt-5-mini", string env values | ✓ |

### Note on `--structural-check` cell count when invoked without absolute path

When running `python scripts/eval_matrix.py --matrix-config configs/eval_matrix_refinement.yaml --structural-check` from the worktree CWD, Python resolves the editable-installed `app.eval.config` to the **main repo path** (`/Users/pnhek/.../mlops_city_concierge/app/eval/config.py` — via Poetry editable install), and `resolve_eval_matrix_path` in that module computes `REPO_ROOT` as the main repo root, so it loads the main repo's pre-edit YAML (2 entries) instead of the worktree's edited YAML (3 entries). Passing the absolute path of the worktree's YAML (`--matrix-config "$(pwd)/configs/eval_matrix_refinement.yaml"`) bypasses the resolver and confirms 3 cells. This is a worktree-environment artifact, not a defect in the change: once this commit lands on main and CI runs from the main repo's CWD, `make eval-matrix-refinement-structural-check` will see 3 cells natively.

## Deviations from Plan

None — plan executed exactly as written. No Rules 1-4 invocations needed:

- No bugs found in surrounding code (Rule 1 N/A)
- No missing critical functionality (Rule 2 N/A — the entry is the only change required)
- No blocking issues (Rule 3 N/A — Pydantic validators accepted the entry as-specified)
- No architectural changes (Rule 4 N/A)

## Acceptance Criteria

All criteria from the plan's `<acceptance_criteria>` block pass:

- [x] `load_eval_matrix('configs/eval_matrix_refinement.yaml')` succeeds with no Pydantic validation error.
- [x] Loaded matrix has exactly 3 entries: `(openai, gpt-4o-mini)`, `(deepseek, deepseek-chat)`, `(openai, gpt-5-mini)`.
- [x] `gpt-5-mini` entry has `env == {"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"}` (string value, not bool).
- [x] `scenarios` list is unchanged — still `["refinement_cheaper"]`.
- [x] Pre-existing entries (`openai/gpt-4o-mini`, `deepseek/deepseek-chat`) are byte-identical (insert-only diff).
- [x] `scripts/eval_matrix.py --structural-check` exits 0 (verified with absolute worktree path to bypass the editable-install path-resolution quirk).
- [x] Dry-run lists three provider/model cells when running `--llm-provider-override scripted`.

## Decisions Made

- **D-07-08 in practice:** the merge gate (Makefile + `scripts/check_baselines_fresh.py`) was deliberately NOT touched. PROMPT-05 will be evaluated by reading the gpt-5-mini cell from the post-Phase-7 baseline JSON (plan 07-07), not by a CI hard gate. This mirrors the existing DeepSeek logged-not-gated treatment.
- **D-07-09 in practice:** no thinking-disable env toggle added to the gpt-5-mini block. Disabling reasoning would neutralize the very signal PROMPT-05 was designed to surface — if prompt-decoupling helps a reasoning model whose `reasoning_content` is dropped by `_prune_for_llm`, that is information about prompt-coupling's contribution; disabling reasoning would erase the test.

## Known Stubs

None — this plan only adds wiring; no UI surface and no placeholder data.

## Threat Flags

None — the change is a single YAML entry in a config file already on a trust boundary (eval-only, never reaches `/chat` prod). No new network endpoint, no new auth path, no new file access pattern.

## Self-Check: PASSED

- File exists: `configs/eval_matrix_refinement.yaml` modified — verified via `git status --short` (clean post-commit) and `git diff HEAD~1 HEAD --stat`.
- Commit exists: `43aaeb3` — verified via `git log --oneline -3`.
- Acceptance criteria: all 7 boxes checked above with verified evidence.
