---
phase: 12-decisiveness-instrumentation-comparison-floor
reviewed: 2026-06-11T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - app/agent/graph.py
  - app/agent/state.py
  - docs/baseline_regen.md
  - docs/eval_gates.md
  - Makefile
  - scripts/eval_agent.py
  - scripts/eval_falsifier.py
  - tests/unit/test_agent_graph.py
  - tests/unit/test_eval_agent.py
  - tests/unit/test_eval_falsifier.py
  - tests/unit/test_eval_matrix.py
findings:
  critical: 1
  warning: 8
  info: 4
  total: 13
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-11
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed the Phase 12 diff (175b32f..HEAD): in-graph step telemetry (`app/agent/graph.py`,
`app/agent/state.py`), harness decisiveness fields (`scripts/eval_agent.py`), the new
falsifier report (`scripts/eval_falsifier.py` + Makefile target), deferral docs, and the
four test files. All 281 tests in the changed test files pass; ruff check/format are clean.

The telemetry plumbing is solid (JSON-safe, pre-increment step contract respected on both
the scratch and telemetry sides, replace-reducer semantics correct). The serious problems
are concentrated in the *measurement semantics*: the falsifier's anchor non-regression
check compares pooled rates over structurally mismatched scenario sets (CR-01), and the
viability/rule-8 instrumentation has three systematic biases (nearby hits can never be
viable, duplicate requested types collapse, duplicate-venue double counting) that will
distort exactly the Phase 13 analysis this phase exists to feed.

## Critical Issues

### CR-01: Falsifier anchor check pools mismatched scenario sets — verdict can be wrong in both directions

**File:** `scripts/eval_falsifier.py:236-296` (anchor non-regression block)
**Issue:** In run-dir mode the anchor check compares
`_pooled_commit_rate(summary, anchor)` against
`_pooled_commit_rate(baselines_summary, anchor)`, but the two summaries cover
**different scenario universes by construction**:

- A `make eval-matrix` run covers only `omakase_mission_open_ended`
  (`configs/eval_matrix.yaml:39-40`; closure-cascade removed per D-11-13).
- `_build_summary_from_baselines` pools **all** committed baselines, which includes
  `refinement_cheaper.json` — a scenario produced only by the *refinement* matrix and
  therefore **never present** in an `eval_matrix.yaml` run summary. It is
  baseline-eligible (only `late_night_closure_cascade` carries
  `baseline_eligible: false`, `configs/eval_queries.yaml:428`), so it always
  contributes to the baseline floor.

Today both committed anchor medians happen to be 1.0, so the numbers coincide. The
moment either baseline is honestly regenerated below 1.0, the comparison is
apples-to-oranges: e.g. baseline omakase 0.8 / refinement 1.0 → floor 0.9; a run where
the anchor scores exactly its omakase baseline (0.8) reports `0.8 < 0.9 FAIL` — a false
anchor regression. The inverse (false PASS) occurs when the weak scenario is the one
missing from the run. `SCENARIOS=` subset runs widen the same hole. This is the
milestone go/no-go instrument giving a wrong answer on realistic data.

**Fix:** Restrict the anchor comparison to the intersection of scenarios with a
non-None rate in *both* summaries (and print any scenarios excluded for asymmetry):

```python
common = {
    sid
    for sid in set(anchor_per_scenario) & set(baseline_per_scenario)
    if anchor_per_scenario[sid] is not None and baseline_per_scenario[sid] is not None
}
# re-pool both sides over `common` only; FAIL loudly (or warn) when common == set()
```

Alternatively compare per-scenario (`run[sid] >= baseline[sid]` for every common sid),
which is stricter and immune to weighting artifacts.

## Warnings

### WR-01: `nearby` hits can never be "viable" — viability metrics silently ignore half the retrieval evidence

**File:** `scripts/eval_agent.py:563-599` (`viable_candidates_per_step_from_state`), `scripts/eval_agent.py:643-679` (`rule8_met_per_step_from_state`)
**Issue:** Both helpers read `"nearby"` scratch entries and gate viability on
`similarity >= LOW_SIMILARITY_THRESHOLD` (0.55). But the `nearby` SQL hardcodes
`0.0 AS similarity` (`app/tools/retrieval.py:152,175,201`), so **no nearby hit can ever
count as viable**, regardless of type/area match. Rule 8 in the system prompt
(`app/agent/prompts.py:151-155`) defines viable as "matches the cuisine/type and is in
roughly the right area" — a type-matching `nearby` hit is viable by that definition but
scores 0 here. Consequences: `viable_candidates_per_step` undercounts on nearby-driven
flows, `rule8_met_per_step` stays False when the model legitimately had enough material,
and the decisiveness-gap signal (`rule8_met_but_kept_searching_steps`) is biased low for
exactly the models Phase 13 wants to compare. The test
`test_nearby_zero_similarity_contributes_zero` enshrines the behavior without
documenting why nearby is read at all.
**Fix:** Either (a) treat `nearby` hits as viable on `primary_type` match alone (no
similarity gate, since the tool provides none), or (b) drop `"nearby"` from the scanned
tools and document that viability is semantic-search-only. Don't ship a metric that
reads a source it can never count.

### WR-02: Rule-8 coverage collapses duplicate requested types and double-counts duplicate venues

**File:** `scripts/eval_agent.py:632-679` (`rule8_met_per_step_from_state`)
**Issue:** Two false-positive paths in "every requested stop had >=1 viable candidate":

1. Typed path: `covered_types >= set(requested_types)` collapses duplicates. For
   `requested_primary_types == ["restaurant", "restaurant", "bar"]` (two distinct
   restaurant stops), one viable restaurant marks both restaurant slots covered.
2. No-types fallback: `cumulative >= num_stops` sums per-step counts without
   deduplication by `place_id` — the same venue returned by searches at steps 0, 1, 2
   counts as 3 viable candidates toward a 3-stop request.

Both inflate `rule8_met_per_step` → inflate `rule8_met_but_kept_searching_steps` →
overstate the decisiveness gap that Phase 13 will diagnose against.
**Fix:** For (1), require multiset coverage: count distinct viable `place_id`s per
requested type and compare against `Counter(requested_types)`. For (2), accumulate a
`set` of viable `place_id`s and compare `len(seen_ids) >= target`.

### WR-03: "Pooled committed_itinerary_rate" is a weighted average of per-scenario medians, not a pooled rate

**File:** `scripts/eval_falsifier.py:69-114` (`_pooled_commit_rate`)
**Issue:** The pooled value is `sum(median_s * n_s) / sum(n_s)` — it treats every run in
a scenario as if it scored the scenario median. summary.json scorer blocks carry only
`{median,min,max,stdev,n}` (`scripts/eval_matrix.py:198-208`), so the true per-run mean
is unavailable, but the distortion is real: scenario run-rates `[0,0,0.7,0.8,1.0]`
pool as 0.7 (median) vs true 0.5 — PASS vs FAIL at the 0.6 bar. With the current
single-scenario default and binary per-run rates at n=5 the verdict happens to be
equivalent; with multiple scenarios or fractional rates it diverges. The plan (12-03)
mandates median-weighting, so this is a design constraint — but the printed label
"pooled committed_itinerary_rate" overstates what is computed.
**Fix:** Minimum: label the output "median-weighted" and document the divergence.
Better: have `eval_matrix` also persist `mean` in scorer stat blocks (it already has
the values when computing median) and pool on mean.

### WR-04: Malformed summary shapes crash with exit code 1 — conflated with a legitimate FAIL verdict

**File:** `scripts/eval_falsifier.py:99-100, 84-96` (`_pooled_commit_rate` callers in `main`)
**Issue:** `_pooled_commit_rate` guards `median` against None/bool/non-numeric but not
`n`: `cir_block.get("n", 0)` followed by `float(median) * n` raises `TypeError` when
`n` is `null`/string. Similarly `scenario_block.get(...)`/`cell.get("scorers", {})`
raise `AttributeError` when a scenario/cell is not a dict. These calls sit **outside**
the `(OSError, ValueError)` infra handler, so a malformed artifact produces an uncaught
traceback and interpreter exit code 1 — which the documented contract defines as
"FAIL — expected; not an infra error". Tooling keyed on exit codes (the intended
consumer) will misread a corrupt summary as a true falsifier failure.
**Fix:** Guard `n` like `median` (`isinstance(n, int) and not isinstance(n, bool) and n >= 0`),
add `isinstance(..., dict)` guards on scenario/cell/scorers, and/or wrap the two check
blocks so unexpected exceptions map to exit 2.

### WR-05: `--gates-config` is parsed (and passed by the Makefile) but never used

**File:** `scripts/eval_falsifier.py:148-152` (argparse), `Makefile:218` (eval-falsifier target)
**Issue:** `args.gates_config` is never referenced in `main()` — the 0.6 bar is the
hardcoded `_FALSIFIER_BAR` and the anchor floor comes from `--baselines-dir`. The
Makefile passing `--gates-config configs/eval_gates.yaml` implies the gate YAML drives
the falsifier when it does not; an operator editing `eval_gates.yaml` expecting the
falsifier to follow will be silently ignored.
**Fix:** Remove the argument and the Makefile flag, or actually read the bar/anchor
metric from the gates config.

### WR-06: `_latest_run_dir` can silently grade the wrong matrix; resolved run dir is never printed

**File:** `scripts/eval_falsifier.py:46-62, 178` (`_latest_run_dir`, `main`)
**Issue:** Both `make eval-matrix` and `make eval-matrix-refinement` write to
`eval_reports/`. Default-mode `make eval-falsifier` grabs the lexicographically newest
subdirectory with no validation that the summary came from `configs/eval_matrix.yaml`
(the docstring's explicit D-12-08 scope). Run the refinement matrix last and the
falsifier reports against the wrong artifact — and since `main()` never prints which
run dir it resolved, the operator has no way to notice from the output.
**Fix:** Print the resolved run dir in the report header, and validate the summary's
matrix identity (eval_matrix records config metadata in summary.json; if not, check
that the expected provider keys/scenario ids are present and warn otherwise).

### WR-07: Telemetry "step" indices are not unique — revision loops produce duplicate-step entries

**File:** `app/agent/graph.py:336-345` (plan), `app/agent/state.py:278-287` (docstring)
**Issue:** `step_count` increments only in `act()`. On the
`plan → critique → plan` paths (critique rejects a finalize via
`critique_final_with_stops` hard-check failure, or
`_retry_unnecessary_stop_count_clarification`), `plan()` runs twice at the same
`step_count` and appends two telemetry entries with the same `"step"` value; a
subsequent `act()` patches only the last. The `state.py` docstring ("Per-step raw
timing") and the INST-04 framing imply one entry per step; Phase 13 consumers that
group or join on `step` (e.g. against `viable_candidates_per_step`, which is indexed
by unique step) will double-count `llm_call_seconds` or mis-join. The unit tests only
assert non-decreasing steps, so this shape ships unexercised.
**Fix:** Document entries as "one per plan() invocation; `step` may repeat" in the
state docstring and harness comment, or merge same-step entries in `plan()` (sum
`llm_call_seconds` into an existing trailing entry for the same step).

### WR-08: `first_commit_call_step_from_state` crashes on non-int step values despite promising None on malformed input

**File:** `scripts/eval_agent.py:530-534`
**Issue:** `steps = [e["step"] for e in entries if isinstance(e, dict) and "step" in e]`
collects values without the `isinstance(step, int)` guard used by both sibling helpers
(lines 571-572, 652-653). `min([None])` / mixed `[2, "3"]` raises `TypeError`, while the
docstring states "Returns None ... if the scratch value is malformed." A malformed entry
would take down the whole query row (uncaught in `query_result_from_state`).
**Fix:**
```python
steps = [
    e["step"]
    for e in entries
    if isinstance(e, dict) and isinstance(e.get("step"), int) and not isinstance(e.get("step"), bool)
]
```
(Apply the same guard to the `commit_steps` set comprehension at
`scripts/eval_agent.py:818-822`, which currently admits non-int steps.)

### WR-09: Threshold parameterization is asymmetric — `rule8_met_per_step_from_state` hardcodes the module constant

**File:** `scripts/eval_agent.py:644` vs `scripts/eval_agent.py:537-541`
**Issue:** `viable_candidates_per_step_from_state` takes `viability_threshold` as a
parameter; `rule8_met_per_step_from_state` ignores any caller intent and binds
`viability_threshold = LOW_SIMILARITY_THRESHOLD` internally. The report's
self-describing `viability_threshold` field claims to be "the threshold value used for
viability judgments" but only provably binds one of the two metrics — pass a different
threshold to the first helper and the two fields silently disagree. Untested branch:
no test covers the run-dir anchor-regression FAIL path or the
`rule8_met_but_kept_searching_steps` derivation in `query_result_from_state` either.
**Fix:** Add a `viability_threshold: float` parameter to
`rule8_met_per_step_from_state` and thread the same `threshold` local from
`query_result_from_state`. Add a unit test for the kept-searching derivation
(commit step excluded) and a falsifier test where the anchor pooled rate is below the
baseline floor (exit 1 via the anchor branch).

## Info

### IN-01: `first_commit_mention_step` comment implies conditional logic that doesn't exist

**File:** `scripts/eval_agent.py:845`
**Issue:** `first_commit_mention_step=None,  # opaque by default; set when visible (D-12-03)` —
nothing in the codebase ever sets it; the plan specifies "null by default, never used
for gating." A Phase 13 reader will interpret `null` as "reasoning was opaque" when it
actually means "not implemented."
**Fix:** Change the comment to "always None in Phase 12; reserved heuristic slot per D-12-03."

### IN-02: `tool_calls_this_step` excludes unknown-tool calls but includes their wall time

**File:** `app/agent/graph.py:378-386, 437-439`
**Issue:** The unknown-tool branch `continue`s before `tool_calls_this_step += 1`, yet
`_tool_elapsed` spans the whole loop. The "requested vs executed" semantics of the
counter are undocumented in the state-field description.
**Fix:** One-line docstring/comment clarifying the counter means "executed tool calls
(known tools + commit)".

### IN-03: Gemini promotion path duplicated across two docs and already drifting

**File:** `docs/baseline_regen.md:249-282`, `docs/eval_gates.md:92-122`
**Issue:** Two near-identical 7-step promotion blocks; step 1 already differs
("verify GEMINI_API_KEY via embeddings probe" vs "top up quota"). The anthropic
deferral set the dual-doc precedent, but each new deferral doubles the drift surface.
**Fix:** Keep the full path in one doc (eval_gates.md owns gate semantics) and link
from the other.

### IN-04: Test fixtures hardcode `viability_threshold=0.55`

**File:** `tests/unit/test_eval_agent.py:241, 565`
**Issue:** The phase's own directive is "import, never hardcode 0.55" and the file
already imports `LOW_SIMILARITY_THRESHOLD`. Fixture literals will silently desync if
the constant changes (the assertions wouldn't fail, but the fixture would misrepresent
real report rows).
**Fix:** Use `LOW_SIMILARITY_THRESHOLD` in both fixture dicts.

---

_Reviewed: 2026-06-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
