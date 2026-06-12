---
phase: 13-decisiveness-experiment-arms
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - app/agent/graph.py
  - app/agent/prompts.py
  - app/agent/revision.py
  - app/agent/state.py
  - app/agent/viability.py
  - configs/eval_matrix_arm.yaml
  - docs/decisiveness_arm_verdicts.md
  - docs/decisiveness_dec03_decision.md
  - scripts/eval_agent.py
  - scripts/eval_falsifier.py
  - tests/unit/test_agent_prompts.py
  - tests/unit/test_agent_revision.py
  - tests/unit/test_eval_agent.py
  - tests/unit/test_eval_falsifier.py
  - tests/unit/test_graph_forced_commit.py
  - tests/unit/test_graph_parallel_tools.py
  - tests/unit/test_viability.py
findings:
  critical: 2
  warning: 10
  info: 7
  total: 19
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-06-12
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

Phase 13 adds flag-gated decisiveness experiment arms: a shared viability predicate, an A2 forced-commit gate, an A1 prompt addendum + critique scoping, an A3 parallel tool path, an arm matrix config, and falsifier split reporting. The flag-off byte-identity requirement holds: all three arm flags are correctly gated (`_forced_commit_step > 0`, empty-string addendum, untouched sequential `act()` branch), and flag-off prompt/critique behavior is pinned by tests.

However, two of the arm mechanisms themselves are broken in ways the unit tests cannot see because the tests mock the exact functions that fail:

1. **The A2 forced-commit synthesizer can never produce a valid commit in production** (CR-01). Real `semantic_search` hits in scratch are `PlaceHit` Pydantic models; the typed path of `best_viable_candidate_per_slot` converts every non-dict hit to `{}`, and even a correct hit dict lacks the required `rationale` field, so `commit_stops` rejects every synthesized stop. The failure is silent (no log, no telemetry). The committed A2 run's "forced=0 across all 30 episodes" is exactly the signature of this bug.
2. **The falsifier's D-13-04 model-initiated/forced split always reports 0/0** (CR-02). It reads `deterministic` at the top level of per-run JSON files, but those files are full `EvalRunReport` objects with `deterministic` nested under `queries[i]`. The committed verdicts doc contains the buggy `0/0` output verbatim, side by side with manually computed `4/10` tables.

Both bugs corrupt the evidentiary record that the Phase 14 entry decision and the A2 verdict rest on. The A2 "mechanism's viability gate was not satisfied" conclusion in `docs/decisiveness_arm_verdicts.md` is not supported by the code as shipped.

## Critical Issues

### CR-01: Forced-commit synthesizer (A2) can never produce a valid commit on real data

**File:** `app/agent/viability.py:216`, `app/agent/graph.py:619-639`
**Issue:** Two independent defects, either of which alone is fatal to the DEC-02 mechanism:

(a) **Typed path discards real hits.** `best_viable_candidate_per_slot` typed path:

```python
hit_dict = hit if isinstance(hit, dict) else {}
```

In production, `semantic_search` returns `list[PlaceHit]` (Pydantic models — `app/agent/tools.py:40`, `app/tools/retrieval.py:32`), and `act()` stores those objects verbatim in scratch. So every candidate for a typed scenario (omakase, refinement — i.e., every Phase 13 arm scenario) becomes an **empty dict** with no `place_id`. The untyped path (lines 180-184) at least attempts an object→dict conversion; the typed path silently returns `{}`.

(b) **Candidates lack required `Stop` fields.** Even with a correct hit dict, `commit_stops` does `Stop(**raw)` and `Stop.rationale` is a required field with no default (`app/agent/state.py:198`). A `PlaceHit` has no `rationale`, so every synthesized stop is rejected with `"invalid stop: ..."`. The graph.py comment ("they are grounded in scratch, so commit_stops validates them") is wrong — grounding passes, model validation fails.

Net effect in `graph.py:623-625`: `raw_stops` is `[{}, {}]` → `commit_stops` rejects all (`place_id not seen via prior tool result`) → `committed_stops` is empty → `if committed_stops:` is False → **silent fall-through** with `commit_forced=False`. The A2 arm's forced mechanism is dead code in production. The live A2 run (forced=0 in all 30 episodes, per `docs/decisiveness_arm_verdicts.md:180`) is fully explained by this bug; the doc's alternative explanation ("gate conditions not satisfied") is unverifiable because the failure leaves no trace.

`tests/unit/test_graph_forced_commit.py:242-272` masks this completely: it patches `all_slots_viable`, `best_viable_candidate_per_slot`, AND `commit_stops`, so the only test of the A2 branch never exercises the synthesizer against real shapes.

**Fix:**
```python
# viability.py typed path — convert objects like the untyped path should:
if isinstance(hit, dict):
    hit_dict = dict(hit)
elif isinstance(hit, BaseModel):
    hit_dict = hit.model_dump(mode="json")
else:
    continue  # unknown shape — not a usable candidate
```
And in the graph.py synthesizer, build commit-shaped stops explicitly before calling `commit_stops`:
```python
raw_stops = [
    {
        "place_id": c["place_id"],
        "name": c.get("name") or "",
        "primary_type": c.get("primary_type"),
        "rationale": f"Best available match (cosine {c.get('similarity', 0):.2f}) "
                     f"for the requested {c.get('primary_type')} slot.",
        "source": c.get("source") or "google_places",
    }
    for c in candidates
    if c is not None and c.get("place_id")
]
```
Add a non-mocked integration test: real `best_viable_candidate_per_slot` + real `commit_stops` (DB lookup mocked at `get_details_many` only) over `PlaceHit` objects in scratch, asserting `commit_forced is True` and non-empty stops.

### CR-02: Falsifier commit split reads the wrong JSON shape — always reports "0/0, forced 0/0"

**File:** `scripts/eval_falsifier.py:205`, fixture defect at `tests/unit/test_eval_falsifier.py:951`
**Issue:** `_commit_split_from_run_dir` does:

```python
det = data.get("deterministic") or {}
```

But the per-run cell files it globs are written by `scripts/eval_agent.py write_report` via `--output` (`scripts/eval_matrix.py:430-431`) — full `EvalRunReport` JSONs whose top-level keys are `eval_queries_path / llm_provider / chat_model / query_count / aggregate / queries`. The `deterministic` block lives at `queries[i]["deterministic"]`. So `det` is always `{}`, both counters stay 0, and every split annotation prints `(model-initiated 0/0, forced 0/0)`.

This is empirically confirmed in the committed record: all three verbatim falsifier outputs in `docs/decisiveness_arm_verdicts.md` (lines 91, 97, 167, 173, 258, 264) print `0/0` while the adjacent hand-computed tables read `4/10`, `9/10`, etc. The D-13-04 honesty contract ("the verdict MUST report the model-initiated vs forced split") is silently broken in the tool that exists to enforce it.

The unit tests pass because the fixture writer (`test_eval_falsifier.py:933-965`) writes `{"deterministic": {...}}` at the top level — it encodes the bug, not the real artifact shape.

**Fix:**
```python
for query in data.get("queries") or []:
    det = query.get("deterministic") if isinstance(query, dict) else None
    if not isinstance(det, dict):
        continue
    if det.get("commit_forced"):
        forced += 1
    elif det.get("first_commit_call_step") is not None:
        model_initiated += 1
```
Update the test fixture to write the real `EvalRunReport` shape (ideally generated through `report_to_dict` / `asdict` from `scripts/eval_agent.py` so it cannot drift again), and add a contract test asserting the fixture shape matches `make_error_record`/`QueryEvalResult` serialization.

## Warnings

### WR-01: Viability addendum is documented as a "rule 8 extension" but lands after the entire prompt, orphaned

**File:** `app/agent/prompts.py:19-47`, `app/agent/graph.py:318-325`
**Issue:** `rule8_viability_addendum` returns a sentence indented with three spaces to look like a rule-8 list item, but the caller concatenates it after `SYSTEM_PROMPT.format(...)` — and `SYSTEM_PROMPT` already ends with the full `REVISION_GUIDANCE` block ("...Better to ask than to lie."). The model sees the viability sentence as a stray indented fragment at the very bottom of the prompt, ~90 lines away from rule 8 and immediately after revision guidance that includes the `low_similarity` "rephrase your query" instruction. This plausibly contributed to A1's measured zero effect.
**Fix:** Either inject the sentence into rule 8 via a `{rule8_addendum}` placeholder in `SYSTEM_PROMPT` (formatted with `""` when off — preserves byte-identity), or reframe the addendum as a clearly-labeled standalone section ("VIABILITY DEFINITION: ...") instead of a fake list item. Update the docstring to match actual placement.

### WR-02: VIABILITY_CONTRACT_ENABLED read at import time in revision.py but at graph-build time in graph.py — co-tuning invariant can split

**File:** `app/agent/revision.py:35-37`, `app/agent/graph.py:305-307`
**Issue:** D-13-05 requires DEC-01 (prompt addendum) and DEC-03 (critique scoping) to flip together. The prompt half is read when `build_agent_graph` runs; the critique half (`_VIABILITY_CONTRACT_ENABLED`) is baked at module import. Any process where the env changes between import and graph build (test harnesses using `monkeypatch.setenv` — exactly what `test_graph_forced_commit.py:232-234` does, ineffectively, for revision.py — or a runner that sets flags after importing app modules) gets the addendum without suppression or vice versa. "Co-tuned by construction" (decision doc line 104) is only true when the env is frozen before first import.
**Fix:** Read the flag in one place at one time. Simplest: have `_diagnose_last_tool_result` read the env per call (cheap), or pass the flag into revision from the graph-build closure so both halves share the graph-build read.

### WR-03: test_agent_revision.py reloads modules without restoring them — flag state leaks into the rest of the test session

**File:** `tests/unit/test_agent_revision.py:161-267`
**Issue:** The flag-on tests `importlib.reload(rev)` with `VIABILITY_CONTRACT_ENABLED=1`. `monkeypatch` restores the env at teardown, but the reloaded module keeps `_VIABILITY_CONTRACT_ENABLED = True` for the remainder of the session (reload mutates the shared module namespace in place). Every later test in the suite that exercises `_diagnose_last_tool_result` with an all-slots-viable state silently runs in flag-ON mode — order-dependent flakiness, and it specifically undermines the suite's flag-off byte-identity guarantees.
**Fix:** Add a fixture that re-reloads `app.agent.revision` (and `app.agent.viability`) with a clean env after each test in this file:
```python
@pytest.fixture(autouse=True)
def _restore_revision_module():
    yield
    os.environ.pop("VIABILITY_CONTRACT_ENABLED", None)
    os.environ.pop("LOW_SIMILARITY_THRESHOLD_OVERRIDE", None)
    importlib.reload(rev); importlib.reload(via)
```

### WR-04: Falsifier split denominator is "committed runs", not "total episodes" — contradicts the D-13-04 format

**File:** `scripts/eval_falsifier.py:366-367, 451-453`
**Issue:** `gpt5_total = gpt5_mi + gpt5_fc` — once CR-02 is fixed, a model that committed 4 of 10 episodes prints `model-initiated 4/4, forced 0/4`, implying a 100% commit rate. The D-13-04 contract and the verdicts doc both use total episodes as the denominator (`4/10`).
**Fix:** Count total run files scanned (post scenario filter) as the denominator, independent of whether they committed.

### WR-05: Forced-commit synthesis failure is completely silent

**File:** `app/agent/graph.py:623-639`
**Issue:** When `commit_stops` rejects every synthesized stop, the branch falls through with no log line and no state marker. Operationally this is indistinguishable from "gate not satisfied" — which is precisely why CR-01 survived a full live A2 matrix run and was written up as a model-behavior finding. A mechanism whose entire purpose is telemetry honesty must not have an invisible failure mode.
**Fix:** Log at warning level with the rejection payload when the gate fired but `committed_stops` is empty, e.g. `logger.warning("forced-commit gate fired at step %d but synthesis was rejected: %s", state.step_count, _payload)`. Consider a `forced_commit_attempted` telemetry field so eval reports can distinguish "gate never fired" from "fired but failed".

### WR-06: Forced commit leaves no scratch/message trace and commit_forced is sticky across subsequent model-initiated commits

**File:** `app/agent/graph.py:626-639`
**Issue:** Three related inconsistencies when the forced branch fires:
1. The synthesized commit is not recorded in `scratch["commit_itinerary"]` and no ToolMessage is appended, so `first_commit_call_step` stays None and `rule8_met_but_kept_searching_steps` lists the forced step as "kept searching" — the eval metrics contradict `commit_forced=True`.
2. If `critique_final_with_stops` requests a revision (done=False), the model receives a `CRITIQUE_ITINERARY` message instructing it to "re-call commit_itinerary" about a commit that does not exist anywhere in its message history.
3. If the model then commits on its own, `commit_forced` remains True, and the (fixed) falsifier split counts the run as forced (`commit_forced` takes precedence over `first_commit_call_step`), overstating forced influence.
**Fix:** Record the synthesized commit as a scratch entry (`{"args": {"stops": raw_stops}, "result": payload, "step": ..., "id": "forced-commit-<step>", "forced": True}`); only set `commit_forced` on the path where the forced stops survive to the final state, or have the falsifier prefer model-initiated when both signals are present.

### WR-07: Gate and synthesizer disagree on anonymous hits — gate can pass while synthesis under-fills slots

**File:** `app/agent/viability.py:114-121` vs `:179, :214-217`
**Issue:** `all_slots_viable` counts hits lacking a `place_id` toward coverage (`anon_count`), but `best_viable_candidate_per_slot` skips them (`continue` on `pid is None`). The gate can return True while the candidate list contains `None` entries; `raw_stops` is then silently shorter than the requested slot count, producing a guaranteed `stop_count_mismatch` revision loop from a "successful" forced commit.
**Fix:** Use the same admission rule in both functions — require a usable `place_id` in `all_slots_viable` (anonymous hits cannot be committed, so they should not count as viable for the commit-oriented gate).

### WR-08: Forced-commit gate overrides legitimate clarifying-question finalizations

**File:** `app/agent/graph.py:613-618`
**Issue:** The gate fires before `finalizing` is computed. With the flag on, once `step_count >= N`, a model that finalizes with a deliberate escalation question (the `neighborhood_no_match` "ask the user ONE concise question and STOP" path, or the stop-count clarification) is overridden by a synthetic commit whenever slots happen to be viable. This contradicts the prompt's own escalation contract and the critique-loop design. Arm-only behavior, but it changes the meaning of A2 results on clarification-shaped scenarios.
**Fix:** Skip the forced gate when the last message is a tool-call-free AIMessage whose path would route to `finalize_as_is` with a clarification (e.g., gate on `committed_this_step or not finalizing`), or document explicitly that A2 intentionally overrides clarifications.

### WR-09: Truthy-env-flag parsing duplicated six times across the changed files

**File:** `app/agent/graph.py:305-310`, `app/agent/revision.py:35-37`, `scripts/eval_agent.py:929-937, 1179-1180`
**Issue:** The idiom `os.environ.get(X, "").strip().lower() in {"1", "true", "yes", "on"}` is copy-pasted in six places. A future change to the truthy set (or a typo in one copy) silently desynchronizes graph behavior from the report's `arm_flags` self-description — the exact field operators use to verify which arm ran. Project guidance (CLAUDE.md) flags DRY violations as critical.
**Fix:** Add one helper (e.g., `app/config.py: def env_flag(name: str) -> bool`) and use it at every site; the `arm_flags` block in `eval_agent.py` then provably parses identically to the graph.

### WR-10: A2 verdict's causal claim is unsupported given CR-01/CR-02 — Phase 14/15 inputs rest on it

**File:** `docs/decisiveness_arm_verdicts.md:180-186, 193-201`
**Issue:** The doc asserts "the forced gate conditions were not satisfied" and treats the A2 null as informative. Given CR-01 (synthesizer cannot succeed and fails silently) and CR-02 (the split tooling reads nothing), `forced=0` is over-determined and the gate-condition claim is unverifiable from the recorded artifacts. The Phase 14 entry decision is unaffected (all arms failed regardless), but the recorded reason A2 failed — and therefore whether forced-commit deserves a re-run after the fix — is wrong as written.
**Fix:** After CR-01/CR-02 land, annotate the A2 section: mechanism was inoperative due to a synthesis bug; the 0.500 model-initiated improvement stands, but the forced mechanism is untested at n=5. Decide explicitly whether the reserved 4th run slot (or a Phase 14 side-quest) should re-test A2 with a working synthesizer.

## Info

### IN-01: best_viable_candidate_per_slot untyped path's object→dict conversion is not JSON-safe as documented

**File:** `app/agent/viability.py:180-184`
**Issue:** `{k: getattr(hit, k, None) for k in dir(hit) if not k.startswith("_")}` captures bound methods (`model_dump`, `copy`, ...) and triggers `PydanticDeprecatedSince20` warnings via `getattr` on deprecated attrs (`.dict`, `.json`). The docstring claims "all returned entries are plain dicts (JSON-safe)" — they are dicts, but not JSON-safe.
**Fix:** Use `hit.model_dump(mode="json")` for `BaseModel` instances (same fix as CR-01(a)).

### IN-02: viability hit collection ignores scratch entry "step" validity, diverging from the eval helper it claims to mirror

**File:** `app/agent/viability.py:57-74` vs `scripts/eval_agent.py:679-687`
**Issue:** `_collect_step_hits` admits hits from entries with missing/malformed `"step"`; `rule8_met_per_step_from_state` requires `isinstance(step, int)` and drops them. On malformed scratch the two "single source of truth" predicates can disagree.
**Fix:** Apply the same `isinstance(entry.get("step"), int)` guard in `_collect_step_hits`, or document the divergence.

### IN-03: Dead summary.json guard and quarantine-blind pooling in the split helper

**File:** `scripts/eval_falsifier.py:197-200`
**Issue:** The glob `f"{provider_slug}--*.json"` can never match `summary.json`, so the skip check is dead code. Separately, the split (called without `scenario_ids` at lines 365/450) pools every run file in the dir, including scenarios the rate computation excludes as `baseline_eligible: false` — numerator/denominator universes can differ from the rate they annotate.
**Fix:** Drop the dead check; pass the same scenario universe used for the pooled rate.

### IN-04: No-op step_count in forced-commit model_copy

**File:** `app/agent/graph.py:627-629`
**Issue:** `update={"stops": committed_stops, "step_count": state.step_count}` — the `step_count` entry writes the value it already has.
**Fix:** Remove `"step_count"` from the update dict.

### IN-05: Bad --matrix-config path silently disables the zero-overlap guard

**File:** `scripts/eval_falsifier.py:71-90, 341`
**Issue:** `_expected_matrix_scenarios` returns an empty set on any read failure, and the guard condition `expected_scenarios and ...` then never fires. A typo'd `--matrix-config` path means the operator-requested guard is skipped without any message.
**Fix:** When `args.matrix_config` was explicitly provided and the file cannot be read/parsed, print a warning (or exit 2) instead of silently proceeding.

### IN-06: arm_flags self-description re-reads env at scoring time, not graph-build time

**File:** `scripts/eval_agent.py:928-940`
**Issue:** The flags the graph actually closed over were read in `build_agent_graph`; `arm_flags` re-parses the env per query row. Same-process eval runs are fine, but any consumer that builds the graph and scores later (or mutates env mid-run) can record a self-description that does not match the behavior. Also, a malformed `FORCED_COMMIT_STEP` raises `ValueError` here, converting a scored run into a whole-process exit-2 after the spend.
**Fix:** Capture the flag dict once (ideally returned from/alongside `build_agent_graph`) and thread it into `query_result_from_state`; wrap the `int()` parse defensively.

### IN-07: Parallel arm (A3) still blocks the event loop on commit and is bounded by DB pool size

**File:** `app/agent/graph.py:413-422, 477-491`
**Issue:** In `_exec_one`, `commit_stops` (which performs a blocking `get_details_many` DB read) runs directly on the event loop, so a commit in a multi-tool step serializes the "parallel" gather. Separately, N concurrent `asyncio.to_thread(tool.invoke, ...)` calls each borrow from `ThreadedConnectionPool`; exhaustion (`maxconn`) surfaces as `{"error": PoolError}` tool results under high tool-call fan-out. Neither is a correctness race (the pool is thread-safe), but both bound the latency benefit A3 was built to measure.
**Fix:** Wrap `commit_stops` in `asyncio.to_thread` on the parallel path; note the pool ceiling in the A3 section of the verdicts doc.

---

_Reviewed: 2026-06-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
