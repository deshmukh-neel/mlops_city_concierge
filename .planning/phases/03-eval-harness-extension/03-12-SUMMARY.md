---
phase: 03-eval-harness-extension
plan: 12
subsystem: eval-harness
tags: [gap-closure, in-03, makefile, eval-agent, eval-matrix, runs-rename]
requirements:
  - EVAL-10
dependency_graph:
  requires:
    - Makefile eval block (added by plan 03-05)
  provides:
    - Makefile target `eval-agent` driven by `QUERIES` variable (matches `--max-queries` semantics)
    - Makefile target `eval-matrix` still driven by `RUNS` (unchanged; correct for `--runs`)
  affects:
    - Operators invoking `make eval-agent` from the CLI or CI runbooks
tech_stack:
  added: []
  patterns:
    - "Per-target Makefile variables (avoid one-name-two-semantics footgun)"
key_files:
  created: []
  modified:
    - Makefile
decisions:
  - "Split `RUNS` into two variables instead of unifying both targets on one name — `RUNS` on eval-matrix is genuinely per-cell runs (matches `--runs`); `QUERIES` on eval-agent is case-count cap (matches `--max-queries`). Same variable name for both would be a misnomer on at least one target."
  - "Did NOT touch `scripts/eval_agent.py` — gap-closure scope explicitly forbids the script edit. Variable-name correction is the Makefile's job."
metrics:
  duration: ~5 minutes (single-file edit + smoke verification)
  completed: 2026-05-22
  tasks_completed: 1
  files_modified: 1
---

# Phase 03 Plan 12: Makefile Runs Rename Summary

Closed IN-03 by splitting the Makefile's shared `RUNS` variable into per-target `QUERIES` (eval-agent → `--max-queries`) and `RUNS` (eval-matrix → `--runs`), eliminating the same-name-different-semantics footgun.

## What Shipped

- `Makefile` eval block (lines 100-124):
  - Added `QUERIES ?= 1` alongside the preserved `RUNS ?= 1`.
  - `eval-agent` recipe: `--max-queries $(QUERIES)` (was `$(RUNS)`).
  - `eval-agent` help line: `(PROVIDER/MODEL/QUERIES/SCENARIOS params)` (was `RUNS`).
  - Example-comment block: `make eval-agent ... QUERIES=1 ...` (was `RUNS=1`).
  - `eval-matrix` recipe and help line: unchanged — `RUNS` is correctly per-cell here.
- `scripts/eval_agent.py` and `scripts/eval_matrix.py`: NOT touched (gap-closure scope forbids the script edit; the Makefile is the contract surface that needed fixing).

## Before / After (Makefile)

**Before:**

```make
# Parameter variables — override on the command line.
#   make eval-agent PROVIDER=openai MODEL=gpt-4o-mini RUNS=1 SCENARIOS=omakase_mission_open_ended
#   make eval-matrix RUNS=3
PROVIDER ?= scripted
MODEL ?= placeholder
RUNS ?= 1
SCENARIOS ?=
LLM_OVERRIDE ?=

.PHONY: eval-agent
eval-agent: ## Run scripts/eval_agent.py once (PROVIDER/MODEL/RUNS/SCENARIOS params)
	$(POETRY_RUN) python scripts/eval_agent.py \
	  --llm-provider $(PROVIDER) \
	  --chat-model $(MODEL) \
	  $(if $(SCENARIOS),--scenario-ids $(SCENARIOS),) \
	  --max-queries $(RUNS)
```

**After:**

```make
# Parameter variables — override on the command line.
#   make eval-agent PROVIDER=openai MODEL=gpt-4o-mini QUERIES=1 SCENARIOS=omakase_mission_open_ended
#   make eval-matrix RUNS=3
PROVIDER ?= scripted
MODEL ?= placeholder
RUNS ?= 1
QUERIES ?= 1
SCENARIOS ?=
LLM_OVERRIDE ?=

.PHONY: eval-agent
eval-agent: ## Run scripts/eval_agent.py once (PROVIDER/MODEL/QUERIES/SCENARIOS params)
	$(POETRY_RUN) python scripts/eval_agent.py \
	  --llm-provider $(PROVIDER) \
	  --chat-model $(MODEL) \
	  $(if $(SCENARIOS),--scenario-ids $(SCENARIOS),) \
	  --max-queries $(QUERIES)
```

`eval-matrix` recipe is byte-for-byte identical to the pre-state.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Rename Makefile RUNS to QUERIES on eval-agent; keep RUNS on eval-matrix; update help/comment lines (IN-03) | f588cb1 | Makefile |

## Verification Outputs

All three `make -n` smoke checks pass exactly as specified in the plan's `<verify>` block:

```
$ make -n eval-agent QUERIES=3
poetry run python scripts/eval_agent.py \
	  --llm-provider scripted \
	  --chat-model placeholder \
	   \
	  --max-queries 3                  # ← QUERIES=3 plumbs through ✓

$ make -n eval-matrix RUNS=3
poetry run python scripts/eval_matrix.py \
	  --matrix-config configs/eval_matrix.yaml \
	  --runs 3 \                       # ← RUNS=3 plumbs through ✓

$ make -n eval-agent RUNS=99
poetry run python scripts/eval_agent.py \
	  --llm-provider scripted \
	  --chat-model placeholder \
	   \
	  --max-queries 1                  # ← OLD `RUNS=N` invocation ignored on eval-agent ✓
```

The third check is the load-bearing one: it proves the rename **actually took effect** — passing `RUNS=99` to `eval-agent` no longer feeds into `--max-queries`; the new `QUERIES ?= 1` default applies instead. Callers using the old habit get default behavior, not silent acceptance of the wrong variable.

### make help (post-rename)

```
$ make help | grep eval-
eval-agent          Run scripts/eval_agent.py once (PROVIDER/MODEL/QUERIES/SCENARIOS params)
eval-matrix         Run cross-provider matrix (LLM_OVERRIDE=scripted for CI; RUNS=3 default)
```

Operator reading `make help` now sees `QUERIES` on `eval-agent` (matches `--max-queries`) and `RUNS` on `eval-matrix` (matches `--runs`). No more name-collision footgun.

### Acceptance Criteria (from plan)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `grep -nE "^QUERIES \?=" Makefile` returns exactly one line | PASS (line 108) |
| 2 | `grep -nE "^RUNS \?=" Makefile` returns exactly one line (RUNS preserved) | PASS (line 107) |
| 3 | `grep -n -- "--max-queries \$(QUERIES)" Makefile` returns ≥1 line in eval-agent recipe | PASS (line 118) |
| 4 | `grep -n -- "--max-queries \$(RUNS)" Makefile` returns ZERO lines | PASS (none) |
| 5 | `grep -n -- "--runs \$(RUNS)" Makefile` returns ≥1 line in eval-matrix recipe (preserved) | PASS (line 124) |
| 6 | `grep -n "QUERIES/SCENARIOS" Makefile` returns ≥1 line (updated help text) | PASS (line 113) |
| 7 | `make -n eval-agent QUERIES=3` exits 0 with `--max-queries 3` | PASS |
| 8 | `make -n eval-matrix RUNS=3` exits 0 with `--runs 3` | PASS |
| 9 | `make -n eval-agent RUNS=99` exits 0 with `--max-queries 1` | PASS |
| 10 | No file other than `Makefile` modified | PASS (`git status --short` = ` M Makefile`) |

## Deviations from Plan

None — plan executed exactly as written. Single-file edit, single commit, all acceptance criteria green.

## Auth Gates

None. No auth required for Makefile edits.

## Known Stubs

None. No code added; only an existing variable name was disambiguated.

## Unit-Suite Sanity Check (Plan's `<verification>` bullet #5)

Plan calls for `poetry run pytest tests/unit/ -q` as a "sanity check (no Python files changed)". Skipped in this worktree: the Claude Code worktree does not have an installed Poetry venv (Poetry tried to create a fresh `mlops-city-concierge-eFsFYN4g-py3.13` venv but no deps are installed inside it; `pytest` is not on PATH). Since this plan touches **zero Python files** (only the Makefile, by gap-closure constraint), no unit test outcome can be caused by the change. The orchestrator's main-repo CI run will provide the canonical green-suite signal on merge.

## Closure (IN-03)

IN-03 is closed:

- The Makefile variable name now matches the underlying CLI flag's semantics on both targets.
- `make help` cannot be misread to imply `RUNS` controls per-case repetition on `eval-agent` — there is no `RUNS` on that target anymore.
- The CLI scripts (`eval_agent.py`, `eval_matrix.py`) are untouched, preserving gap-closure scope.

## Self-Check: PASSED

- File `Makefile` exists and contains the edits (verified by `sed -n '100,125p'`).
- Commit `f588cb1` exists on branch `worktree-agent-a523636b23bba8d6d` (`refactor(03-12): rename Makefile RUNS to QUERIES on eval-agent (IN-03)`).
- All 10 acceptance criteria from plan pass.
- All 3 plan-verify smoke checks emit the expected `--max-queries N` / `--runs N` output.
- Only `Makefile` modified; no out-of-scope writes.
