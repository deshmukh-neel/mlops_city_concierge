---
phase: 13-decisiveness-experiment-arms
plan: 09
type: execute
wave: 2
gap_closure: true
depends_on: ["13-08"]
files_modified:
  - scripts/eval_falsifier.py
  - tests/unit/test_eval_falsifier.py
  - docs/decisiveness_arm_verdicts.md
autonomous: true
requirements: [DEC-05]
must_haves:
  truths:
    - "_commit_split_from_run_dir reads deterministic from queries[i] inside the EvalRunReport (not top level) and returns the correct (model_initiated, forced) split — (4, 0) for the A2 gpt-5-mini run dir, not (0, 0)"
    - "The test fixture writes the REAL EvalRunReport JSON shape (top-level aggregate/chat_model/eval_queries_path/llm_provider/queries/query_count with deterministic nested under queries[i]) so the test cannot pass while the reader is broken"
    - "At least one regression test asserts the split reader returns non-zero counts on the real nested shape and fails against the old top-level reader"
    - "The verdict doc annotates that the pasted falsifier '0/0' output was a tool bug (CR-02); the hand-computed table numbers (4/10 etc.) are correct"
  artifacts:
    - path: "scripts/eval_falsifier.py"
      provides: "Split reader iterating queries[i].deterministic"
      contains: "queries"
    - path: "tests/unit/test_eval_falsifier.py"
      provides: "Fixture writing the real EvalRunReport shape + regression test"
      contains: "queries"
    - path: "docs/decisiveness_arm_verdicts.md"
      provides: "CR-02 annotation that pasted 0/0 falsifier output is a tool bug"
      contains: "tool bug"
  key_links:
    - from: "scripts/eval_falsifier.py _commit_split_from_run_dir"
      to: "EvalRunReport queries[i].deterministic"
      via: "per-query iteration over the run JSON"
      pattern: "queries"
---

<objective>
Fix CR-02: the falsifier's D-13-04 model-initiated/forced split reader
(`_commit_split_from_run_dir`) reads `deterministic` at the top level of each per-run
JSON, but the real EvalRunReport nests `deterministic` under `queries[i]`. The reader
always returns (0, 0). Fix the reader to iterate `queries[i]`, fix the test fixture so it
writes the REAL EvalRunReport shape (it currently encodes the bug), add a regression test
that fails on the old reader, and annotate the verdict doc that the pasted "0/0" output
was a tool bug while the hand-computed tables are correct.

Purpose: CR-02 is a blocker for Phase 14 (per 13-VERIFICATION.md). The D-13-04 honesty
contract's enforcement tool (eval_falsifier.py) is silently broken — any future operator
running eval-falsifier-arm gets 0/0 in all split annotations. The verdict doc carries
correct hand-computed numbers but a contradictory pasted 0/0, a documentation-integrity
defect the Phase-14 entry decision must not inherit.

Output: a correct split reader, a fixture pinned to the real artifact shape, a regression
test, and an annotated verdict doc. No behavior change to the falsifier exit-code contract
(0/1/2). No flags shipped, no baselines written.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md
@.planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md
@.planning/phases/13-decisiveness-experiment-arms/13-VERIFICATION.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Fix _commit_split_from_run_dir to read queries[i].deterministic + regression test</name>
  <files>scripts/eval_falsifier.py, tests/unit/test_eval_falsifier.py</files>
  <read_first>
    - scripts/eval_falsifier.py lines 179-220 (_commit_split_from_run_dir: the `det = data.get("deterministic") or {}` at line 205, the scenario-from-filename filter at lines 206-215, the commit_forced/first_commit_call_step counting at 216-219, and the provider_slug glob at 197)
    - .planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md CR-02 section (the recommended fix iterates `for query in data.get("queries") or []` and reads `query.get("deterministic")`; real EvalRunReport top-level keys are eval_queries_path / llm_provider / chat_model / query_count / aggregate / queries)
    - scripts/eval_agent.py (the report serialization — find report_to_dict / asdict / write_report and the QueryEvalResult/EvalRunReport shape, so the fixture matches the actual serialized structure: each queries[i] carries scenario_id/query_id and a nested deterministic block with commit_forced + first_commit_call_step)
    - tests/unit/test_eval_falsifier.py lines 925-1000 (TestCommitSplitFromRunDir: the _write_run_file helper at 933-960 that currently writes {"deterministic": {...}} at TOP level — the fixture encoding the bug — and the existing split tests)
  </read_first>
  <behavior>
    - Test 1: a run-dir of EvalRunReport-shaped files (deterministic nested under queries[i]) with 2 forced + 2 model-initiated + 1 never-committed query returns (2, 2).
    - Test 2: the same logic on a file that has ONLY a top-level deterministic block (the OLD buggy shape) returns (0, 0) — proving the reader no longer accepts the wrong shape, so the fixture-shape regression is pinned.
    - Test 3: a query whose deterministic has commit_forced=True is counted as forced; a query with commit_forced falsey but first_commit_call_step not None is counted model-initiated; a query with both falsey/None is counted as neither.
    - Test 4: scenario filtering still works (only queries[i] whose scenario matches scenario_ids are counted) — scenario read from the query record's scenario_id when iterating queries.
  </behavior>
  <action>
    In `scripts/eval_falsifier.py`, rewrite the body of `_commit_split_from_run_dir` (lines ~201-219). The current code does `det = data.get("deterministic") or {}` then counts once per FILE — wrong on two counts: (1) the real per-run JSON is a full EvalRunReport whose `deterministic` lives under each `data["queries"][i]`, not top level; (2) a file may contain multiple queries. Replace with iteration over `data.get("queries") or []`: for each query dict, read `det = query.get("deterministic")`; skip if `det` is not a dict; if `det.get("commit_forced")` increment forced; elif `det.get("first_commit_call_step") is not None` increment model_initiated. For scenario filtering, prefer reading the scenario from the query record (e.g. `query.get("scenario_id")`) when `scenario_ids` is provided, falling back to the filename-derived scenario only if the query record lacks it; keep the existing summary.json skip and the OSError/ValueError guard (return-without-raise contract, T-13-05-01) unchanged.

    THEN fix the fixture in `tests/unit/test_eval_falsifier.py`: rewrite `_write_run_file` (lines 933-960) so the payload is a REAL EvalRunReport shape — top-level keys including `eval_queries_path`, `llm_provider`, `chat_model`, `query_count`, `aggregate`, and `queries` (a list), with the `deterministic` block nested under each `queries[i]` (alongside a `scenario_id`). REQUIRED: derive the fixture's field set from scripts/eval_agent.py's serialization (report_to_dict / asdict of the EvalRunReport/QueryEvalResult dataclasses) — do NOT hand-craft the key set — so it cannot drift from the real artifact again. Add the four behavior tests above. Add Test 2 explicitly as the fixture-shape regression: a payload with ONLY a top-level `deterministic` block returns (0, 0) under the fixed reader (the OLD fixture shape no longer "works").
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && poetry run pytest tests/unit/test_eval_falsifier.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "queries" scripts/eval_falsifier.py` shows _commit_split_from_run_dir iterating `data.get("queries")`.
    - `grep -n 'data.get("deterministic")' scripts/eval_falsifier.py` returns NO match (the top-level read is gone).
    - The fixture _write_run_file in tests/unit/test_eval_falsifier.py writes a payload whose top-level keys include `queries` and whose `deterministic` block is nested under `queries[i]` (NOT at top level).
    - The fixture's shape is derived from scripts/eval_agent.py's serialization path (report_to_dict / asdict of EvalRunReport/QueryEvalResult), not hand-crafted — the fixture builds or asserts against the dataclass field set from `scripts.eval_agent`.
    - A regression test asserts the fixed reader returns non-zero counts (e.g. (2,2)) on the real nested shape AND returns (0,0) on the old top-level-only shape — the test FAILS against the pre-fix top-level reader.
    - `poetry run pytest tests/unit/test_eval_falsifier.py -q` exits 0.
  </acceptance_criteria>
  <done>The split reader reads queries[i].deterministic and returns correct (model_initiated, forced) counts; the fixture writes the real EvalRunReport shape so the test can no longer pass while the reader reads the wrong level; a regression test pins both directions.</done>
</task>

<task type="auto">
  <name>Task 2: Annotate the verdict doc that the pasted falsifier 0/0 was a tool bug</name>
  <files>docs/decisiveness_arm_verdicts.md</files>
  <read_first>
    - docs/decisiveness_arm_verdicts.md lines 145-200 (A2 per-model table at 147-151 with the CORRECT hand-computed split "model-initiated 4/10, forced 0/10", and the pasted verbatim falsifier output at 157-178 showing the contradictory "model-initiated 0/0, forced 0/0")
    - docs/decisiveness_arm_verdicts.md lines 91, 97, 258, 264 (the other pasted falsifier 0/0 lines in A1 and A3 sections)
    - .planning/phases/13-decisiveness-experiment-arms/13-VERIFICATION.md CR-02 required_action_before_phase_14 (annotate that the pasted falsifier "0/0" output was a tool bug; hand-computed table numbers are correct) and the Data-Flow Trace row marking the pasted falsifier output HOLLOW
  </read_first>
  <action>
    Add a single clearly-labeled CR-02 annotation note near the first pasted verbatim falsifier output block in the A2 section (and reference it from A1/A3 where the same 0/0 appears). The note MUST state: (a) the pasted falsifier per-scenario output lines showing "(model-initiated 0/0, forced 0/0)" were produced by a tool bug (CR-02: `_commit_split_from_run_dir` read `deterministic` at the wrong JSON level, always returning 0/0); (b) the hand-computed split numbers in the per-model TABLES (e.g. gpt-5-mini "model-initiated 4/10, forced 0/10") are CORRECT and were verified against the actual run JSON `queries[i].deterministic` data; (c) the bug is fixed in scripts/eval_falsifier.py in this gap-closure — re-running eval-falsifier-arm on the recorded run dirs now reproduces the table numbers, not 0/0. Do NOT delete or rewrite the pasted verbatim output (it is the historical record of what the tool printed) — annotate it as historically-buggy. Do not alter the honest null result or the per-model table numbers.
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && grep -qi "tool bug" docs/decisiveness_arm_verdicts.md && grep -qi "CR-02" docs/decisiveness_arm_verdicts.md && grep -q "4/10" docs/decisiveness_arm_verdicts.md && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - The verdict doc contains a CR-02 annotation using the phrase "tool bug" and referencing CR-02.
    - The annotation states the hand-computed table numbers are correct (the "4/10" string is still present in the A2 table and referenced as correct).
    - The annotation states the bug is fixed in scripts/eval_falsifier.py.
    - The pasted verbatim falsifier output blocks are NOT deleted (they remain as historical record): `grep -c "model-initiated 0/0" docs/decisiveness_arm_verdicts.md` >= 1.
    - The honest null line "No arm cleared the INST-05 falsifier bar" is present and unaltered (`grep -c "No arm cleared" docs/decisiveness_arm_verdicts.md` >= 1).
  </acceptance_criteria>
  <done>The verdict doc honestly flags the pasted 0/0 falsifier output as a CR-02 tool bug, affirms the hand-computed table numbers as correct, and records that the tool is now fixed — resolving the documentation-integrity inconsistency without touching the null result.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | This plan repairs an offline analysis/reporting tool (`eval_falsifier.py`) and a test fixture. It reads local run-dir JSON files only — no network, no auth, no external input crosses a new boundary. The exit-code contract (0/1/2) and the OSError/ValueError swallow-and-skip behavior are preserved. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-09-01 | Repudiation | falsifier split annotation prints 0/0, corrupting the D-13-04 honesty record for future arm runs | mitigate | Reader fixed to iterate queries[i]; fixture pinned to real EvalRunReport shape; regression test fails on the old reader |
| T-13-09-02 | Information Disclosure | malformed run JSON crashes the falsifier mid-report | accept | Existing OSError/ValueError guard (T-13-05-01) preserved — malformed files are skipped, never raised |
| T-13-09-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A — no new dependencies added |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_falsifier.py -q` exits 0, including the new fixture-shape regression test.
- Full suite mandatory before merge (DB-pool contamination risk with real-graph tests): `make test` passes.
- The falsifier exit-code contract is unchanged: `make eval-matrix-refinement-structural-check` (or the existing no-subprocess smoke) stays green.
- Verdict doc annotated; pasted verbatim 0/0 output preserved as historical record; hand-computed tables affirmed correct; honest null result intact.
- No flags enabled by default; no baselines written (`git status configs/eval_baselines/` clean).
</verification>

<success_criteria>
- CR-02 closed: `_commit_split_from_run_dir` reads `queries[i].deterministic` and returns the correct split (non-zero on real run dirs).
- The fixture writes the real EvalRunReport shape — the test can no longer pass while the reader reads the wrong JSON level; a regression test pins both directions and fails on the old code.
- The verdict doc honestly records the pasted 0/0 as a CR-02 tool bug and affirms the hand-computed numbers; the D-13-04 honesty contract's enforcement tool works for future arm runs.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-09-SUMMARY.md` when done.
</output>
