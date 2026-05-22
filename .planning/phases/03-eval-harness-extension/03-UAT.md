---
status: complete
phase: 03-eval-harness-extension
source:
  - 03-02-SUMMARY.md
  - 03-03-SUMMARY.md
  - 03-04-SUMMARY.md
  - 03-05-SUMMARY.md
  - 03-06-SUMMARY.md
  - 03-07-SUMMARY.md
  - 03-08-SUMMARY.md
  - 03-09-SUMMARY.md
  - 03-10-SUMMARY.md
  - 03-11-SUMMARY.md
  - 03-12-SUMMARY.md
  - 03-VERIFICATION.md
  - 03-REVIEW.md
  - 03-REVIEW-FIX.md
started: 2026-05-22T18:30:00Z
updated: 2026-05-22T18:45:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Full test suite passes
expected: `make test-unit` exits 0 with ~736 passed / 49 skipped. No FAILED lines.
result: pass
evidence: `736 passed, 7 skipped, 9 warnings in 13.48s`

### 2. Eval-matrix dry-run lists 18 cells (2 providers × 3 scenarios × 3 runs)
expected: `APP_ENV=eval poetry run python scripts/eval_matrix.py --dry-run --runs 3 --matrix-config configs/eval_matrix.yaml` prints 18 lines. Exit 0.
result: pass
evidence: line count = 18, exit 0

### 3. WR-01 closed — eval_agent.py boots without sys.path shim
expected: `poetry run python scripts/eval_agent.py --help` prints argparse help, no `ModuleNotFoundError`.
result: pass
evidence: usage block printed; `grep -n sys.path scripts/eval_agent.py` returns no matches

### 4. WR-04 closed — `--llm-provider-override foo--bar` rejected cleanly
expected: argparse error citing `'--' is reserved as the cell-filename separator`. No Pydantic ValidationError traceback.
result: pass
evidence: `eval_matrix.py: error: argument --llm-provider-override: --llm-provider-override='foo--bar' contains '--'; '--' is reserved as the cell-filename separator. Use a single-dash or alphanumeric provider name.`

### 5. WR-02 closed — `check_baselines_fresh.py ""` loud-fails rc=2
expected: rc=2 with stderr `BASE_SHA positional argument was the empty string; …`.
result: pass
evidence: `BASE_SHA positional argument was the empty string; pass a real SHA or omit the argument to use origin/main` / EXIT=2

### 6. IN-03 closed — Makefile QUERIES vs RUNS split
expected: `make -n eval-agent QUERIES=3` → `--max-queries 3`; `make -n eval-agent RUNS=99` → `--max-queries 1`; `make -n eval-matrix RUNS=3` → `--runs 3`.
result: pass
evidence: All three printed commands match. Old habit (`RUNS` on eval-agent) ignored, not silently consumed.

### 7. CR-01 (re-review) closed — baseline content gate active when populated
expected: `pytest tests/unit/test_baselines_are_populated.py -v` shows tests skipped when `BASELINES_POPULATED` is unset, with actionable skip message.
result: pass
evidence: `7 skipped in 0.02s` with skip messages pointing the user at `APP_ENV=eval make eval-matrix RUNS=3` + post-processing + `BASELINES_POPULATED=1` flip.

### 8. CI scripted-mode regression — `make eval-matrix LLM_OVERRIDE=scripted RUNS=1` works without keys
expected: `summary.json` contains `overridden_to: "scripted"`, scorer block whitelisted, `[SCRIPTED CI MODE]` marker present.
result: pass
evidence: |
  - `summary.json` top-level: `overridden_to: "scripted"`, `failures: [6 cells]`, `scenarios: {...}`, `generated_at: <ts>`
  - Per-scenario scorer keys (whitelisted, 7 exactly): `['category_compliance', 'constraints_satisfied', 'geographic_coherence', 'no_hallucinated_place_ids', 'rationale_stop_alignment', 'temporal_coherence', 'walking_budget_respected']` — **zero 6-polluter keys leaked** (no `tool_calls_mean`, `results_mean`, `revision_hints_mean`, `committed_stops_mean`, `answer_retrieved_place_coverage_mean`, `contexts_mean`)
  - Per-cell `final_reply: "[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py."`
  - 6 cells logged in `failures` array with rc=1 — **documented soft-gate behavior** per `.github/workflows/ci.yml:189-195`: scripted cells fail scorer thresholds (no committed itinerary) and the CI job uses `continue-on-error: true` until live baselines flip it to a hard gate (plan 03-07 user-action item).

### 9. Lint clean
expected: `make lint` exits 0 with no findings.
result: pass
evidence: `All checks passed!`

### 10. Type-check clean
expected: `make typecheck` exits 0 — `Success: no issues found in 34 source files`.
result: pass
evidence: `Success: no issues found in 34 source files`

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0

## Gaps

[none]

## Known Human-Only Item (out of UAT scope)

The single open item from VERIFICATION.md (`status: human_needed`) is the **live baseline matrix run**:
`APP_ENV=eval make eval-matrix RUNS=3` with real `OPENAI_API_KEY` + `DEEPSEEK_API_KEY` (~15 min, real API spend), then post-process `eval_reports/<ts>/summary.json` into the three `configs/eval_baselines/*.json` files. This is intentionally outside the UAT loop — it doesn't block Phase 3 closure per plan 03-07's explicit decision. After the run, set `BASELINES_POPULATED=1` to flip Test 7 from "skipped" to "pass" as a permanent CI content gate.
