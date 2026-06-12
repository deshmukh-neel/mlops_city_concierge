---
phase: 12-decisiveness-instrumentation-comparison-floor
fixed_at: 2026-06-11T00:00:00Z
review_path: .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 10
skipped: 0
status: all_fixed
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-06-11
**Source review:** .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 10 (fix_scope: critical_warning — 1 Critical + 9 Warning; IN-01..IN-04 out of scope)
- Fixed: 10
- Skipped: 0

Verification: every fix carries new unit tests; full suite green (1271 passed, 53 skipped,
9 deselected); `mypy app/` clean; ruff check/format clean (pre-commit hooks ran on every commit).

## Fixed Issues

### CR-01: Falsifier anchor check pools mismatched scenario sets

**Files modified:** `scripts/eval_falsifier.py`, `tests/unit/test_eval_falsifier.py`
**Commit:** f6c17c8
**Applied fix:** `_pooled_commit_rate` gained an optional `scenario_ids` restriction. The
run-dir anchor check now pools BOTH sides over the intersection of scenarios with a
non-None rate in both summaries, prints any scenarios excluded for asymmetry, warns loudly
(treating as no-floor PASS, matching the prior no-baseline semantics) when the intersection
is empty, and still FAILs when the anchor has no evaluable cells at all. Six new tests
cover the false-FAIL repro from the review (run at its own omakase baseline vs a
refinement-inflated floor), the anchor-regression exit-1 path, exclusion reporting, the
empty-intersection warning, and the no-cells FAIL. This also resolves UAT item 2 in
12-HUMAN-UAT.md.

### WR-01: `nearby` hits can never be "viable"

**Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`
**Commit:** 128b210
**Applied fix:** Took the review's option (b), per phase-scope guidance (the root cause —
`0.0 AS similarity` in `app/tools/retrieval.py` — is outside the phase's changed files).
Both `viable_candidates_per_step_from_state` and `rule8_met_per_step_from_state` now scan
`semantic_search` only; docstrings document the SEMANTIC-SEARCH-ONLY scope and the known
Phase 13 limitation (nearby-driven flows undercount viable candidates because the nearby
tool carries no similarity signal). The enshrined `test_nearby_zero_similarity_contributes_zero`
was replaced with `test_nearby_entries_are_not_scanned`, which documents why and asserts
nearby entries are ignored even with a high similarity value.

### WR-02: Rule-8 coverage collapses duplicate types and double-counts duplicate venues

**Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`
**Commit:** 1888ab8
**Applied fix:** Typed path now does multiset coverage: `Counter(requested_types)` with a
per-type set of distinct viable `place_id`s — `["restaurant", "restaurant", "bar"]` needs
two distinct restaurants. No-types fallback now accumulates a cumulative SET of viable
`place_id`s against `num_stops` instead of summing per-step counts. Hits lacking a usable
place_id can't be deduplicated and count once per occurrence (documented limitation,
preserves existing fixture behavior). New tests: duplicate requested types need distinct
ids, same place_id at one step counts once, and a venue repeated across 3 steps does not
satisfy a 3-stop request (3 distinct ones do).

### WR-03: "Pooled" rate is a median-weighted average, not a pooled rate

**Files modified:** `scripts/eval_falsifier.py`
**Commit:** cb80f68
**Applied fix:** Minimum option per the plan-12-03 design constraint: all printed labels
changed from "pooled committed_itinerary_rate" to "median-weighted committed_itinerary_rate"
(and "median-weighted baseline" in baselines-mode), and `_pooled_commit_rate` carries an
explicit WR-03 HONESTY NOTE documenting the formula, why per-run means are unavailable
(summary.json stat blocks carry only {median,min,max,stdev,n}), and the divergence example
from the review. The "better" option (persisting mean in eval_matrix stat blocks) was not
taken because plan 12-03 mandates median-weighting.

### WR-04: Malformed summary shapes crash with exit code 1

**Files modified:** `scripts/eval_falsifier.py`, `tests/unit/test_eval_falsifier.py`
**Commit:** f9e2d55
**Applied fix:** `_pooled_commit_rate` now isinstance-guards every nested read: the
top-level `scenarios` mapping, scenario blocks, `providers`, cells, `scorers`, the
`committed_itinerary_rate` block, and `n` (int, not bool, >= 0 — mirroring the existing
median guard). Malformed shapes degrade to an unevaluable (None) scenario and a deliberate
N/A → FAIL verdict instead of an uncaught TypeError/AttributeError traceback (which exits
1 and is indistinguishable from a legitimate FAIL to exit-code consumers). Six new tests
cover null/string `n`, non-dict scenario/scorers/scenarios shapes, and a `main()` run over
a malformed artifact returning a deliberate exit code.

### WR-05: `--gates-config` parsed and passed but never used

**Files modified:** `scripts/eval_falsifier.py`, `Makefile`
**Commit:** 9b4ec9f
**Applied fix:** Removed the argparse entry and the `--gates-config configs/eval_gates.yaml`
line from the `eval-falsifier` Makefile target (the `eval-gates-check*` targets keep theirs —
`check_eval_gates.py` genuinely uses it). A comment at the removal site documents why the
falsifier deliberately does not read the gates YAML (bar is `_FALSIFIER_BAR`, floor comes
from `--baselines-dir`).

### WR-06: Latest-run-dir mode can silently grade the wrong matrix

**Files modified:** `scripts/eval_falsifier.py`, `tests/unit/test_eval_falsifier.py`
**Commit:** 6436026
**Applied fix:** The report header now always names its source ("source: run dir {path}" /
"source: committed baselines at {dir}"). summary.json carries no matrix-identity metadata,
so a new best-effort `_expected_matrix_scenarios()` helper reads the scenario ids from
`configs/eval_matrix.yaml` and the report prints a loud WARNING when none of them appear in
the graded summary (naming eval_matrix_refinement.yaml as the likely culprit and suggesting
an explicit `--run-dir`). Warn-only: missing/unparseable config never errors. Six new tests.

### WR-07: Telemetry step indices not unique on revision loops

**Files modified:** `app/agent/graph.py`, `app/agent/state.py`, `tests/unit/test_agent_graph.py`
**Commit:** 2968502
**Applied fix:** Took the merge option (stronger than docs-only): `plan()` now merges into
the trailing telemetry entry when it carries the same `step_count` (summing
`llm_call_seconds`), so `step_telemetry` keeps exactly one entry per step index and Phase 13
joins on `step` stay safe. The `ItineraryState.step_telemetry` field description now states
the one-entry-per-step INVARIANT and the merge semantics. New regression test drives a real
plan → critique → plan loop (finalize rejected by a retryable `geographic_coherence`
violation, then accepted) and asserts unique step indices, JSON safety, and plain-float
timing on the merged entry.

### WR-08: `first_commit_call_step_from_state` crashes on non-int step values

**Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`
**Commit:** 9f286dd
**Applied fix:** Applied the review's exact guard (`isinstance(e.get("step"), int)` and
`not isinstance(e.get("step"), bool)`) in `first_commit_call_step_from_state`, and the same
guard (plus a list-type guard on the scratch value) in the `commit_steps` set comprehension
in `query_result_from_state`. New tests: step=None returns None, mixed `[2, "3"]` returns 2,
and bool steps are rejected.

### WR-09: Threshold parameterization asymmetric in `rule8_met_per_step_from_state`

**Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`
**Commit:** 2abcaa8
**Applied fix:** `rule8_met_per_step_from_state` now takes a required
`viability_threshold: float` parameter (no internal rebinding to the module constant);
`query_result_from_state` threads the same `threshold` local into both per-step metrics so
the report's self-describing `viability_threshold` field provably binds both. All call sites
updated. New tests: a caller-supplied threshold below the module constant is honored; the
kept-searching derivation (commit step excluded, no-commit case, threshold field binding) is
now covered via `query_result_from_state`; the previously-untested anchor-regression FAIL
path was added with CR-01 (f6c17c8).

## Skipped Issues

None — all in-scope findings were fixed. IN-01 through IN-04 (Info tier) were out of scope
for `fix_scope: critical_warning` and remain open.

---

_Fixed: 2026-06-11_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
