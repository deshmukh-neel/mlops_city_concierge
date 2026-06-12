---
phase: 13-decisiveness-experiment-arms
reviewed: 2026-06-12T17:54:06Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - app/agent/graph.py
  - app/agent/prompts.py
  - app/agent/revision.py
  - app/agent/state.py
  - app/agent/viability.py
  - app/config.py
  - configs/eval_matrix_arm.yaml
  - docs/decisiveness_arm_verdicts.md
  - docs/decisiveness_dec03_decision.md
  - scripts/eval_agent.py
  - scripts/eval_falsifier.py
  - tests/unit/test_agent_prompts.py
  - tests/unit/test_agent_revision.py
  - tests/unit/test_config.py
  - tests/unit/test_eval_agent.py
  - tests/unit/test_eval_falsifier.py
  - tests/unit/test_graph_forced_commit.py
  - tests/unit/test_graph_parallel_tools.py
  - tests/unit/test_viability.py
findings:
  critical: 0
  warning: 5
  info: 6
  total: 11
status: issues_found
---

# Phase 13: Code Review Report (Re-Review After Gap Closure)

**Reviewed:** 2026-06-12T17:54:06Z
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found

## Summary

This is a re-review verifying the gap-closure plans 13-08/13-09/13-10 against the prior
review's CR-01, CR-02, WR-02, and WR-09. **All four prior findings are verified fixed**
(evidence below). All 153 tests across the seven Phase-13 unit-test files pass
(`pytest -q`, 2.15s).

The re-review found **no new critical defects**, but five warnings — three of them
clustered in the CR-02 fix area of `scripts/eval_falsifier.py` (a contract-violating crash
path proven by live repro, a split-denominator semantic that diverges from the D-13-04
documented format, and a split/rate universe mismatch) — plus a prompt-placement confound
in the A1 arm and a residual flag-read timing asymmetry that contradicts an in-code claim.

### Prior-Finding Verification

| Prior finding | Verdict | Evidence |
|---|---|---|
| **CR-01** (forced-commit synthesizer: PlaceHit → `{}`; missing `rationale`) | **FIXED** | `app/agent/viability.py:182-187, 219-224` — `isinstance(hit, BaseModel)` → `model_dump(mode="json")`; unknown shapes `continue` (no `{}` placeholder). `app/agent/graph.py:637-643` synthesizes the required `rationale` string; candidates without a truthy `place_id` are skipped (graph.py:626-629). Verified end-to-end against `app/agent/commit.py` `commit_stops`: synthesized dict keys (`place_id`, `name`, `primary_type`, `source`, `rationale`) are all valid `Stop` fields, and place_ids derived from scratch hits are in the grounded set. Non-mocked regression test `test_forced_commit_synthesizer_real_placehit_shapes` exercises real `PlaceHit` objects through the real `best_viable_candidate_per_slot` + `commit_stops` path (only DB/LLM boundaries mocked) and passes. Four CR-01 white-box tests in `test_viability.py:443-569` pin the typed path, dict path, unknown-shape skip, and place_id propagation. |
| **CR-02** (`_commit_split_from_run_dir` read top-level `deterministic`, always 0/0) | **FIXED** | `scripts/eval_falsifier.py:207-225` now iterates `data.get("queries") or []` and reads `query.get("deterministic")` per entry. Regression test `test_cr02_real_shape_returns_nonzero_counts` builds fixtures via the real `EvalRunReport`/`report_to_dict` dataclasses (shape cannot drift silently) and asserts (2, 2); `test_cr02_old_top_level_shape_returns_zeros` pins rejection of the old shape. Both pass. The verdicts doc carries honest CR-01/CR-02 post-run annotations preserving the buggy historical output as record. **However, see WR-01/WR-02/WR-03 below — the fixed reader has residual defects.** |
| **WR-02 (prior)** (import-time flag read in revision.py) | **FIXED** | `app/agent/revision.py:203` reads `env_flag("VIABILITY_CONTRACT_ENABLED")` live per call; `all_slots_viable` is only reached when the flag is on (flag-off byte-identity preserved, pinned by `test_flag_off_low_similarity_fires_as_before`). **But see WR-05 — the accompanying comment overclaims, and a build-time/call-time asymmetry with graph.py remains.** |
| **WR-09 (prior)** (truthy-env parsing duplicated 6×) | **FIXED** | `app/config.py:14-23` `env_flag` is the single boolean-flag parser; grep confirms zero remaining hand-rolled truthy parses in `app/` and `scripts/` — all five boolean call sites (graph.py ×2, revision.py ×1, eval_agent.py ×2) route through it. Parametrized truthy/falsy/unset tests in `test_config.py:119-142`. (Residual: the *int* flag `FORCED_COMMIT_STEP` parse is duplicated 2× — see IN-01.) |

## Warnings

### WR-01: `_commit_split_from_run_dir` violates its "never raises" contract on valid-JSON non-dict artifacts

**File:** `scripts/eval_falsifier.py:202-225`
**Issue:** The docstring promises "T-13-05-01: malformed JSON or OSError → file silently
skipped; never raises," but only `(OSError, ValueError)` from `json.loads` are caught. A
run file whose top level is a valid JSON *array* (or string/number) raises
`AttributeError: 'list' object has no attribute 'get'` at `data.get("queries")`; a
`queries` list containing non-dict entries raises the same at `query.get("deterministic")`.
Both reproduced live:

```
array RAISED: AttributeError 'list' object has no attribute 'get'
non-dict query RAISED: AttributeError 'str' object has no attribute 'get'
```

The exception propagates out of `main()`, crashing the interpreter with **exit code 1 — the
code the documented contract reserves for a legitimate FAIL verdict** — and no VERDICT block
is printed. CI tooling keyed on exit codes would misread a corrupt artifact as a true
falsifier FAIL. This is the exact failure class the project's own WR-04 (Phase 12) fixed in
`_pooled_commit_rate`.
**Fix:**
```python
data = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    continue
for query in data.get("queries") or []:
    if not isinstance(query, dict):
        continue
    det = query.get("deterministic")
    ...
```
Add a malformed-shape test alongside `test_split_skips_malformed_json` (valid JSON array +
non-dict query entry).

### WR-02: Split line denominator is "total commits," not "total episodes" — diverges from the D-13-04 documented format

**File:** `scripts/eval_falsifier.py:370-372, 455-459`
**Issue:** `gpt5_total = gpt5_mi + gpt5_fc` — the printed denominator is the number of
*commits*, while the D-13-04 honesty contract as recorded in
`docs/decisiveness_arm_verdicts.md` uses *episodes* (e.g. A2: "model-initiated 4/10, forced
0/10" over 10 episodes). With the fixed reader on the A2 run dir the falsifier prints
"(model-initiated 4/4, forced 0/4)" — numerators match the hand-computed tables, denominators
do not. The doc's CR-02 annotation claim that re-running "reproduces the hand-computed table
numbers" is therefore only true for the numerators, and "forced 0/4" misreads as "0 of 4
episodes" when there were 10. Episodes that never committed vanish from the printed total
entirely.
**Fix:** Count episodes scanned in `_commit_split_from_run_dir` (one per `queries[i]` with a
dict `deterministic` block) and return/print `(mi, forced, episodes)` using `episodes` as the
denominator — matching the documented `M/total` format. Alternatively relabel the line
("of N commits") and correct the doc annotation; the former matches D-13-04 as written.

### WR-03: Split annotations are computed over a different scenario universe than the rates they annotate

**File:** `scripts/eval_falsifier.py:370, 455`
**Issue:** `main()` calls `_commit_split_from_run_dir(run_dir, _GPT5_KEY)` and
`(run_dir, _ANCHOR_KEY)` with **no `scenario_ids`**, so splits count every scenario present
in the run dir — including `baseline_eligible: False` scenarios that `_pooled_commit_rate`
excludes, and (for the anchor) scenarios excluded from the `common` run∩baseline
intersection that the anchor floor is computed over (lines 436-450). The split printed on
the anchor PASS/FAIL line can therefore include commits from scenarios that the comparison
itself excludes. The `scenario_ids` parameter exists and is unit-tested
(`test_cr02_scenario_filtering_reads_from_query_scenario_id`) but is never wired into
`main()` — dead capability at the only call sites.
**Fix:** Pass the eligible scenario set for the gpt-5-mini split and `common` for the anchor
split: `_commit_split_from_run_dir(run_dir, _ANCHOR_KEY, scenario_ids=common)`.

### WR-04: "Rule 8" viability addendum actually renders at the very end of the prompt, after REVISION_GUIDANCE

**File:** `app/agent/prompts.py:19-47` (with `app/agent/graph.py:309, 320`)
**Issue:** `SYSTEM_PROMPT` is `<base prompt> + REVISION_GUIDANCE` (prompts.py:221-223), and
graph.py appends the addendum *after* `SYSTEM_PROMPT.format(...)`. So the sentence framed
everywhere as an "additive extension to SYSTEM_PROMPT rule 8" (function name
`rule8_viability_addendum`, its docstring, `docs/decisiveness_dec03_decision.md`,
`docs/decisiveness_arm_verdicts.md` "viability prompt addendum to rule 8") actually lands
~120 lines away from rule 8, immediately after "...Better to ask than to lie.", indented
with three spaces as if continuing a numbered list that isn't there. The concatenation
contract (D-13-06 "ADDITIVE only") is honored, but the placement is an orphaned fragment —
a plausible confound for the A1 arm's 0.000 null result, and the naming actively misleads
anyone tuning the experiment later (Phase 14 retry candidates reference these docs).
**Fix:** Either (a) record the actual rendered placement explicitly in
`decisiveness_dec03_decision.md` / the verdicts doc so the A1 null result is interpreted
against the real prompt layout, or (b) restructure so the addendum is inserted adjacent to
rule 8 (e.g. a second format placeholder inside rule 8 defaulting to "") before any A1
re-run. Do not leave the name claiming a placement the code does not produce.

### WR-05: DEC-01/DEC-03 read the shared co-tuning flag at different effective times; revision.py comment overclaims the WR-02 fix

**File:** `app/agent/revision.py:200-203` (with `app/agent/graph.py:306-309`)
**Issue:** graph.py resolves `VIABILITY_CONTRACT_ENABLED` **once at graph-build time** (the
DEC-01 prompt addendum is baked into `_viability_prompt_addendum`), while revision.py reads
the same flag **live on every critique call**. The comment at revision.py:200-202 states the
two "pick up env changes at the same effective time — the import-time freeze hazard is
eliminated." That is false: if the env var changes after the graph is built (long-lived prod
process, test monkeypatching after `build_agent_graph`), DEC-03 critique suppression flips
while the DEC-01 prompt stays frozen — exactly the "fires in isolation" state that D-13-05
("co-tuned by construction... Neither fires in isolation") forbids. In practice the eval
harness exports flags before process start, so the arms ran consistently; the hazard is the
inaccurate invariant claim plus a real divergence window.
**Fix:** Resolve the flag once in `build_agent_graph` and thread it into the critique path
(e.g. a closure/parameter on `critique_step`/`_diagnose_last_tool_result`), or correct the
comment to state the actual semantics (build-time prompt, call-time critique) and document
the divergence window as accepted.

## Info

### IN-01: `FORCED_COMMIT_STEP` int-parse duplicated at two edit sites

**File:** `app/agent/graph.py:305`, `scripts/eval_agent.py:930`
**Issue:** `int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0")` appears verbatim in both
files — the same drift class WR-09 just closed for booleans. If one site later gains
stripping/validation, the graph's behavior and the report's `arm_flags` self-description can
disagree.
**Fix:** Add `env_int(name, default)` next to `env_flag` in `app/config.py` and use it at
both sites.

### IN-02: "Single source of truth" viability logic is still duplicated in eval_agent, and the gate/synthesizer disagree on anonymous hits

**File:** `app/agent/viability.py:82-142` vs `scripts/eval_agent.py:629-744`; `viability.py:131-140` vs `viability.py:216-218`
**Issue:** (a) `rule8_met_per_step_from_state` re-implements the full multiset coverage
algorithm rather than delegating to the viability module; agreement is pinned by only two
drift-guard tests (one True case, one False case) — multiset/anon/untyped agreement is
unguarded. (b) Within the module itself, `all_slots_viable` counts hits lacking a usable
`place_id` (`per_type_anon`) while `best_viable_candidate_per_slot` skips them — so the
forced-commit gate could pass while the synthesizer produces fewer stops than slots,
yielding a `stop_count_mismatch` revision with `commit_forced=True` already stamped.
Practically unreachable today (`PlaceHit.place_id` is a required non-empty str), but it is a
semantic split inside the module whose stated job is preventing exactly this.
**Fix:** Extend the drift-guard tests to multiset and anon-hit cases; align
`best_viable_candidate_per_slot`'s anon handling with `all_slots_viable` (or have
`all_slots_viable` ignore anon hits too, matching the synthesizer).

### IN-03: Dead `summary.json` skip in `_commit_split_from_run_dir`; its test passes vacuously

**File:** `scripts/eval_falsifier.py:197-200`; `tests/unit/test_eval_falsifier.py:1174-1187`
**Issue:** The glob `f"{provider_slug}--*.json"` can never match `summary.json`, so the
`if path.name == "summary.json": continue` branch is unreachable, and
`test_split_skips_summary_json` passes regardless of the branch's existence.
**Fix:** Delete the dead branch (the glob is the guard) or keep it with a comment
acknowledging the glob already excludes it.

### IN-04: CR-02 scenario-filter test exercises the filename *fallback*, not the query-record path its docstring claims

**File:** `tests/unit/test_eval_falsifier.py:1290-1313` (fixture at line 998: `id=f"{scenario}--run-{run_n}"`)
**Issue:** The fixture's query `id` is `"omakase_mission_open_ended--run-0"`, which fails the
`scenario_in_query not in scenario_ids` check, so the test passes via the filename-parsing
fallback (eval_falsifier.py:216-219). Real artifacts set `QueryEvalResult.id = case.id`
(the bare scenario id), which takes the *primary* path — currently untested with a positive
case — and the test docstring asserts the opposite of what is exercised.
**Fix:** Set the fixture's `id` to the bare scenario id (matching real artifacts) or add a
second fixture variant covering the primary path; correct the docstring.

### IN-05: `importlib.reload` tests mutate module-global threshold without a restoring teardown

**File:** `tests/unit/test_agent_revision.py:73-106, 112-127`
**Issue:** `test_threshold_override_applied_when_set` reloads `app.agent.revision` with
`LOW_SIMILARITY_THRESHOLD_OVERRIDE=0.45` baked into the module global. `monkeypatch` restores
the env at teardown but not the reloaded module state; full-file order self-heals (later
tests reload with the env deleted), but a filtered run (`-k override`) leaves
`revision.LOW_SIMILARITY_THRESHOLD == 0.45` in `sys.modules` for the remainder of the pytest
session — cross-test contamination for any later test reading the module attribute live.
**Fix:** A fixture that reloads the module with a clean env in teardown (try/finally), so the
post-test module state is deterministic regardless of test selection.

### IN-06: `arm_flags` self-description reads env at scoring time, not graph-build time

**File:** `scripts/eval_agent.py:928-934`
**Issue:** `query_result_from_state` snapshots the arm flags from the environment when the
report row is built, but the graph's behavior was fixed by the env at `build_agent_graph`
time (graph.py:305-307). If the env changes mid-run, the report self-describes a
configuration the graph did not use. Same family as WR-05; harmless under the current
export-then-run workflow.
**Fix:** Capture the flag snapshot once per run (e.g. in `build_report` before
`evaluate_cases`) and thread it into the row builder.

---

_Reviewed: 2026-06-12T17:54:06Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
