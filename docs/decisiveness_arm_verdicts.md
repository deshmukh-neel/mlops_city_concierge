# Decisiveness Experiment Arm Verdicts

**Role:** DEC-05 record — the canonical per-arm verdict document for Phase 13.

**Downstream gates:**
- **Phase 14 conditional entry gate** — Phase 14 (state-replay interventions) executes ONLY if
  no arm clears the INST-05 bar. If any arm clears, Phase 14 is skipped and Phase 15
  (prod promotion) is entered directly.
- **Phase 15 promotion input** — the winning arm's flag config and run-dir path are the
  promotion inputs; no arm may be promoted without a clear INST-05 verdict recorded here.

---

## INST-05 Falsifier Definition

An arm **clears** the INST-05 bar when ALL of the following hold for the full n=5 run:

1. **gpt-5-mini pooled `committed_itinerary_rate` >= 0.6** across both scenarios
   (omakase_mission_open_ended + refinement_cheaper, 5 runs each = 10 episodes).
2. **gpt-4o-mini anchor holds >= its honest Phase-12 comparison-floor baseline** —
   the anchor must not regress.
3. **Falsifier exit code 0** (`scripts/eval_falsifier.py --run-dir ... --matrix-config
   configs/eval_matrix_arm.yaml --baselines-dir configs/eval_baselines` returns 0).

The anchor criterion applies to A1 and A3 as commit-rate non-regression. For A2, see the
anchor red-flag rule below.

---

## Run Budget Contract

Per D-13-01, the hard cap is **≤ 4 full live matrix runs total** across all arms in this phase.
This plan (13-06) consumes exactly 3 (A1 + A2 + A3). The 4th run slot is reserved for A4
(conditional combo, plan 13-07) if A4 qualifies. No billing top-ups: if quota or budget dies
mid-arm, the partial result is recorded honestly (labeled PARTIAL) in that arm's section and
the arm stops. **Partial-cell results are never written to `configs/eval_baselines/`** (D-11-14:
`scripts/write_baselines.py` is the only permitted write path, and arm runs must not be
registered as baselines under any circumstance).

Per D-13-02, each full arm run is:
- 3 models × 2 scenarios × n=5, temp=1.0, sequential
- Smoke n=1 sanity check before every full n=5 spend
- Smoke arm_flags self-description verified to match intended arm config before full spend
- Scenario universe: `omakase_mission_open_ended` + `refinement_cheaper`
- late_night_closure_cascade excluded (D-10-09 quarantine); anthropic/gemini cells deferred
  (D-12-09)

---

## A1: Viability Contract + Critique Recalibration

**Flag config:** `VIABILITY_CONTRACT_ENABLED=1`
**Override:** `LOW_SIMILARITY_THRESHOLD_OVERRIDE` — **UNSET on first A1 run** (per
`docs/decisiveness_dec03_decision.md`: isolates scoping effect; 0.45 tested only if A1
shows positive-but-short signal — commit rate improves but stays below 0.6)
**Matrix config:** `configs/eval_matrix_arm.yaml`
**Arm includes:** DEC-01 (viability prompt addendum to rule 8) + DEC-03 (low_similarity
scoped to pre-candidate steps only), both gated by the same flag (D-13-05).

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | `eval_reports/2026-06-12T06-15-29Z` |
| Full (n=5) | `eval_reports/2026-06-12T06-25-52Z` |

**Smoke arm_flags verification:** `{'forced_commit_step': 0, 'parallel_tool': False, 'viability_contract': True, 'viability_threshold_override': None}`

### Per-model results

| Model | Pooled commit rate | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|
| openai/gpt-5-mini | 0.000 (model-initiated 1/10, forced 0/10) | 0.000 | 0.000 | FAIL |
| openai/gpt-4o-mini (anchor) | 1.000 (model-initiated 8/10, forced 0/10) | 1.000 | 1.000 | PASS — non-regression (baseline 1.000) |
| deepseek/deepseek-reasoner | 0.000 (model-initiated 1/10, forced 0/10) | 0.000 | 0.000 | (informational) |

**Falsifier exit code:** `1 (FAIL)`

**Falsifier per-scenario breakdown (pasted verbatim from falsifier output):**

> **Note:** The `(model-initiated 0/0, forced 0/0)` split lines below are a tool bug (CR-02).
> See the CR-02 annotation in the A2 section for full details. The hand-computed split numbers
> in the per-model table above are correct.

```
============================================================
eval_falsifier: INST-05 Milestone Falsifier Report
============================================================
source: run dir eval_reports/2026-06-12T06-25-52Z

[openai/gpt-5-mini] committed_itinerary_rate per scenario:
  omakase_mission_open_ended: 0.000
  refinement_cheaper: 0.000

openai/gpt-5-mini: median-weighted committed_itinerary_rate = 0.000 < 0.6 (model-initiated 0/0, forced 0/0)  FAIL

[openai/gpt-4o-mini] committed_itinerary_rate per scenario (run vs baseline):
  omakase_mission_open_ended: run=1.000  baseline=1.000
  refinement_cheaper: run=1.000  baseline=1.000

openai/gpt-4o-mini: median-weighted = 1.000 >= baseline 1.000 (model-initiated 0/0, forced 0/0)  PASS

============================================================
eval_falsifier: VERDICT = FAIL
============================================================
```

### Closing verdict

A1 does NOT clear the INST-05 bar. gpt-5-mini pooled commit rate = 0.000 across both
scenarios — no improvement over the comparison floor. The viability contract (DEC-01) and
critique recalibration (DEC-03) show zero effect on gpt-5-mini decisiveness in isolation.
Anchor (gpt-4o-mini) held at 1.000 — non-regression confirmed.
LOW_SIMILARITY_THRESHOLD_OVERRIDE was UNSET on this run per plan. A1 result (0.000) is
below the 0.6 bar AND below the "positive-but-short" threshold, so the 0.45 override
variant is not warranted. Honest null result — A1 FAIL.

---

## A2: Forced Commit at Step 6

**Flag config:** `FORCED_COMMIT_STEP=6`
**Matrix config:** `configs/eval_matrix_arm.yaml`
**Mechanism:** At step 6 (env-driven, default off), if the model has not yet committed AND
every requested stop has >= 1 viable candidate (cosine >= LOW_SIMILARITY_THRESHOLD AND matching
primary_type), the graph synthesizes a `commit_itinerary` call from best-so-far candidates and
routes it through the normal commit path (place_id validation, critique_final_with_stops, finalize).

### Honesty Contract (D-13-04)

Forced commits count toward `committed_itinerary_rate` (product-honest: the user receives a
real committed plan), but the verdict MUST report the **model-initiated vs forced split** per
model. The split format is: `commit_rate X.X (model-initiated M/total, forced F/total)`.

**Anchor red-flag rule:** gpt-4o-mini commits before step 6 on its own — ANY behavior change
in the anchor (different commit rate, forced commits on anchor, or quality regression) is a
**red flag** that must be flagged explicitly here and in the A2 verdict. An A2 pass REQUIRES
the anchor to be behaviorally unchanged.

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | `eval_reports/2026-06-12T07-16-04Z` |
| Full (n=5) | `eval_reports/2026-06-12T07-27-03Z` |

**Smoke arm_flags verification:** `{'forced_commit_step': 6, 'parallel_tool': False, 'viability_contract': False, 'viability_threshold_override': None}`

### Per-model results

| Model | Pooled commit rate (with split) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|
| openai/gpt-5-mini | 0.500 (model-initiated 4/10, forced 0/10) | 1.000 | 0.000 | FAIL — below 0.6 bar |
| openai/gpt-4o-mini (anchor) | 1.000 (model-initiated 9/10, forced 0/10) | 1.000 | 1.000 | PASS — non-regression; anchor behaviorally unchanged |
| deepseek/deepseek-reasoner | 0.000 (model-initiated 0/10, forced 0/10) | 0.000 | 0.000 | (informational) |

**Falsifier exit code:** `1 (FAIL)`

**Falsifier per-scenario breakdown (pasted verbatim from falsifier output, including split lines):**

```
============================================================
eval_falsifier: INST-05 Milestone Falsifier Report
============================================================
source: run dir eval_reports/2026-06-12T07-27-03Z

[openai/gpt-5-mini] committed_itinerary_rate per scenario:
  omakase_mission_open_ended: 1.000
  refinement_cheaper: 0.000

openai/gpt-5-mini: median-weighted committed_itinerary_rate = 0.500 < 0.6 (model-initiated 0/0, forced 0/0)  FAIL

[openai/gpt-4o-mini] committed_itinerary_rate per scenario (run vs baseline):
  omakase_mission_open_ended: run=1.000  baseline=1.000
  refinement_cheaper: run=1.000  baseline=1.000

openai/gpt-4o-mini: median-weighted = 1.000 >= baseline 1.000 (model-initiated 0/0, forced 0/0)  PASS

============================================================
eval_falsifier: VERDICT = FAIL
============================================================
```

> **CR-02 POST-RUN ANNOTATION (Plan 13-09, 2026-06-12): Pasted falsifier 0/0 split lines are
> a tool bug.** The `(model-initiated 0/0, forced 0/0)` lines in the pasted verbatim falsifier
> output above — and in all other pasted falsifier output blocks in this document (A1, A3) — were
> produced by a bug in `_commit_split_from_run_dir` (`scripts/eval_falsifier.py`). The reader
> called `data.get("deterministic")` at the top level of each per-run JSON file, but the real
> `EvalRunReport` (written by `eval_agent.py`) nests `deterministic` under each `queries[i]`,
> not at the top level. Because the top-level key does not exist, the reader always returned
> `{}`, yielding 0/0 for every model and every arm. **This is a pure reporting tool bug (CR-02)
> — it did not affect the actual eval runs, the committed_itinerary_rate values, or the
> falsifier exit codes (0/1/2).**
>
> **The hand-computed split numbers in the per-model tables are CORRECT.** For example, the A2
> gpt-5-mini row "model-initiated 4/10, forced 0/10" was verified directly from the run-dir
> JSON files' `queries[i].deterministic.first_commit_call_step` and
> `queries[i].deterministic.commit_forced` fields, bypassing the buggy reader. Those numbers
> are the ground truth; the pasted falsifier 0/0 lines are the incorrect output of the broken
> tool and are preserved here as a historical record only.
>
> **The bug is fixed in `scripts/eval_falsifier.py` (Plan 13-09):** `_commit_split_from_run_dir`
> now iterates `data.get("queries") or []` and reads `query.get("deterministic")` from each
> entry. Re-running `eval-falsifier-arm` on the recorded run dirs now reproduces the
> hand-computed table numbers, not 0/0. A regression test (`tests/unit/test_eval_falsifier.py
> ::TestCommitSplitFromRunDir::test_cr02_real_shape_returns_nonzero_counts`) verifies the
> fixed reader returns non-zero counts on the real EvalRunReport shape and returns (0,0) on
> the old top-level-only shape.

**Key finding — FORCED_COMMIT_STEP=6 mechanism NEVER FIRED:** Raw per-file analysis
confirms forced=0 for ALL models across all 10 episodes each. The mechanism requires all
slots to have a viable candidate (cosine >= threshold AND matching primary_type) at step 6.
gpt-5-mini reached step 6 in scenarios where it eventually committed on omakase (4/5 runs)
but the forced gate conditions were not satisfied on refinement_cheaper (0/5 commits).
The A2 commit rate improvement over A1 (0.500 vs 0.000 for gpt-5-mini) is entirely
model-initiated, not forced — all 4 committed omakase runs were model-initiated at step <= 6.

> **CR-01 POST-RUN ANNOTATION (Plan 13-08, 2026-06-12):** The forced-commit mechanism was
> INOPERATIVE in the n=5 A2 run due to a synthesis bug (CR-01). Two independent defects made
> every synthesized stop empty or rejected before it could reach `commit_stops`:
> (a) `viability.py` typed path discarded real `PlaceHit` Pydantic models to `{}` (line 216:
> `hit if isinstance(hit, dict) else {}`), so `best_viable_candidate_per_slot` yielded empty
> dicts with no `place_id` — candidates were filtered out before reaching `commit_stops`;
> (b) even if a candidate survived with a `place_id`, it lacked a `rationale` field, and
> `Stop.rationale` is REQUIRED with no default (`app/agent/state.py`), so `Stop(**raw)` raised
> a `ValidationError` and `commit_stops` rejected the stop.
>
> **Consequence:** `forced=0` for all 10 episodes is **over-determined by the CR-01 synthesis
> bug**, not solely by gate non-satisfaction. forced=0 is explained by the bug; whether the
> viability gate would have been satisfied on the fixed synthesizer is unknown and untested.
>
> **The 0.500 gpt-5-mini result STANDS** as an entirely **model-initiated** commit rate. The
> synthesis bug only affected the forced path (`FORCED_COMMIT_STEP=6` branch); the model's own
> `commit_itinerary` tool calls were never touched by this bug. The 4 committed omakase runs
> and 0 refinement_cheaper commits are genuine model-initiated behavior.
>
> **The forced mechanism is UNTESTED at n=5.** Its effect on commit rate is unknown until a
> re-run on the fixed synthesizer. The CR-01 fixes (Plan 13-08) repair both defects and add a
> non-mocked regression test (`tests/unit/test_graph_forced_commit.py::
> test_forced_commit_synthesizer_real_placehit_shapes`) that fails on the pre-fix code.
>
> **Phase-14 A2 retry disposition:** A2 re-test with the working synthesizer is a
> **Phase-14/15 candidate, NOT a Phase-13 re-run**. The D-13-02 four-run live cap is already
> consumed (n=5 smoke + full runs exhausted the budget). Whether to retry A2 in Phase 14 should
> be decided once the Phase 13 overall verdict is known; if Phase 13 already shows that no arm
> clears the INST-05 bar, an A2 forced-path re-test is a reasonable Phase-14 entry item.

**Anchor red-flag assessment:** gpt-4o-mini behaviorally UNCHANGED — held at 1.000 across
both scenarios, no forced commits, commit behavior matches comparison floor. NO RED FLAG.

### Closing verdict

A2 does NOT clear the INST-05 bar. gpt-5-mini pooled = 0.500 — improved over A1 (0.000)
but still below 0.6. The improvement is entirely model-initiated on omakase; refinement_cheaper
remains at 0.000. The FORCED_COMMIT_STEP=6 mechanism never fired (forced=0 for all models) —
forced=0 is over-determined by the CR-01 synthesis bug; whether the gate would have been
satisfied on the fixed synthesizer is unknown. Anchor held at 1.000 — no red flag.

A2 shows POSITIVE SIGNAL (gpt-5-mini improved from 0.0 to 0.5 vs A1) without the forced
mechanism triggering. Both A1 and A2 show positive signal (A1=0.0 → A2=0.5) — qualifies
A4 conditional combo arm if A3 also fails to clear independently.

---

## A3: Parallel Tool Execution

**Flag config:** `PARALLEL_TOOL_EXECUTION_ENABLED=1`
**Matrix config:** `configs/eval_matrix_arm.yaml`
**Mechanism:** All tool calls within one `act()` step run concurrently (asyncio.gather); results
appended in original tool_call order regardless of completion order (D-13-08: order-stable).

**Judging criterion (D-13-01):** A3 is judged on **measured latency reduction + zero scorer
regression** — NOT on commit rate. Commit rate is recorded for completeness but is NOT the A3
pass criterion. A3 passes if:
1. gpt-4o-mini shows measurable latency reduction in `tool_execution_seconds` vs the Phase-12
   comparison-floor run dirs (sum of step_telemetry tool_exec_seconds per run, arm vs baseline).
2. Quality scorers hold >= baseline (zero regression on category_compliance and other active gates).

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | `eval_reports/2026-06-12T08-21-16Z` |
| Full (n=5) | `eval_reports/2026-06-12T08-30-52Z` |

**Smoke arm_flags verification:** `{'forced_commit_step': 0, 'parallel_tool': True, 'viability_contract': False, 'viability_threshold_override': None}`

**Smoke error cell (informational):** `deepseek--deepseek-reasoner--refinement_cheaper--run-0.json`
— ValueError ('dictionary update sequence element #0 has length 1; 2 is required').
Known recurring DeepSeek/refinement_cheaper issue; does not block A3 since DeepSeek is
informational and the error is provider-specific, not a parallel-execution artifact.

### Per-model results (commit rate — informational only for A3)

| Model | Pooled commit rate | omakase | refinement_cheaper | Quality scorers |
|---|---|---|---|---|
| openai/gpt-5-mini | 0.500 (model-initiated 5/10, forced 0/10) | 1.000 | 0.000 | category_compliance, constraints_satisfied, geographic_coherence: all held at 1.0 on omakase; refinement_cheaper violations on quality scorers present but pre-existing pattern |
| openai/gpt-4o-mini (anchor) | 0.500 (model-initiated 5/10, forced 0/10) | 1.000 | 0.000 | REGRESSED — refinement_cheaper commit rate 0.000 vs baseline 1.000 (anchor regression) |
| deepseek/deepseek-reasoner | 0.000 (model-initiated 0/10, forced 0/10) | 0.000 | 0.000 | (informational) |

**Falsifier exit code:** `1 (FAIL — anchor regression on refinement_cheaper)`

Note: A3 pass criterion is latency+scorer non-regression, NOT commit rate. However, the
anchor regression (gpt-4o-mini refinement_cheaper 0.000 vs baseline 1.000) is itself a
scorer regression (committed_itinerary_rate is a quality signal).

**Falsifier per-scenario breakdown (pasted verbatim, for scorer non-regression evidence):**

> **Note:** The `(model-initiated 0/0, forced 0/0)` split lines below are a tool bug (CR-02).
> See the CR-02 annotation in the A2 section for full details. The hand-computed split numbers
> in the per-model table above are correct.

```
============================================================
eval_falsifier: INST-05 Milestone Falsifier Report
============================================================
source: run dir eval_reports/2026-06-12T08-30-52Z

[openai/gpt-5-mini] committed_itinerary_rate per scenario:
  omakase_mission_open_ended: 1.000
  refinement_cheaper: 0.000

openai/gpt-5-mini: median-weighted committed_itinerary_rate = 0.500 < 0.6 (model-initiated 0/0, forced 0/0)  FAIL

[openai/gpt-4o-mini] committed_itinerary_rate per scenario (run vs baseline):
  omakase_mission_open_ended: run=1.000  baseline=1.000
  refinement_cheaper: run=0.000  baseline=1.000

openai/gpt-4o-mini: median-weighted = 0.500 < baseline 1.000 (model-initiated 0/0, forced 0/0)  FAIL (anchor regression)

============================================================
eval_falsifier: VERDICT = FAIL
============================================================
```

### Latency Analysis (gpt-4o-mini anchor — primary A3 pass criterion)

**Comparison baseline run dir (Phase-12 floor):** `eval_reports/2026-06-11T19-09-10Z` (omakase),
`eval_reports/2026-06-11T22-19-17Z` (refinement_cheaper)
**A3 arm run dir:** `eval_reports/2026-06-12T08-30-52Z`

**CRITICAL FINDING — Phase-12 floor runs have NO step_telemetry:** The Phase-12
comparison-floor run dirs (`2026-06-11T*`) predate the INST-04 step_telemetry instrumentation
added in Phase 13 plan 13-01. Their run JSON files have `step_telemetry: None` with no
`tool_exec_seconds` field. Latency comparison against the Phase-12 floor is therefore
**unmeasurable** — no valid baseline exists for tool_exec_seconds.

| Metric | Phase-12 baseline | A3 arm (gpt-4o-mini) | Delta |
|---|---|---|---|
| Mean tool_execution_seconds (omakase) | N/A — no telemetry in Phase-12 runs | 5.927s | unmeasurable |
| Mean tool_execution_seconds (refinement_cheaper) | N/A — no telemetry in Phase-12 runs | 6.471s | unmeasurable |
| Mean tool_execution_seconds (pooled) | N/A — no telemetry in Phase-12 runs | 6.199s | unmeasurable |

**A3 arm raw tool_exec_seconds (gpt-4o-mini, per run, 5 runs each scenario):**
- omakase: [4.792, 3.695, 4.604, 7.768, 8.779] → mean = 5.927s
- refinement_cheaper: [6.102, 8.901, 5.639, 5.705, 6.006] → mean = 6.471s
- gpt-5-mini omakase: [11.081, 6.844, 7.048, 8.705, 11.48] → mean = 9.032s
- gpt-5-mini refinement_cheaper: [5.644, 6.025, 6.239, 6.081, 6.084] → mean = 6.015s

**Source:** INST-04 `step_telemetry` — `tool_exec_seconds` per step, summed per run,
averaged over n=5. Values from individual run JSONs in `eval_reports/2026-06-12T08-30-52Z`.

**Scorer non-regression assessment:** FAIL — gpt-4o-mini anchor regressed on
refinement_cheaper. committed_itinerary_rate dropped from 1.000 (baseline) to 0.000 on
refinement_cheaper (5 episodes, 0 commits). 3/5 runs had committed_itinerary_rate=0.0,
median=0.0. This is a real quality regression, not a scoring artifact. The parallel tool
execution mechanism introduced a behavioral change in the anchor on refinement_cheaper.

### Closing verdict

A3 FAILS on the latency+scorer criterion:
1. **Latency comparison: UNMEASURABLE** — Phase-12 comparison-floor run dirs have no
   step_telemetry instrumentation. Cannot confirm or deny latency reduction.
2. **Scorer regression: YES** — gpt-4o-mini anchor dropped from 1.000 to 0.000 on
   refinement_cheaper (median). This is a decisive scorer regression that disqualifies A3
   independently of the latency finding.

A3 does NOT clear the INST-05 bar and fails its own judging criterion (zero scorer
regression). The parallel tool execution arm introduces a real anchor regression on the
refinement scenario — likely a race condition or state capture issue under concurrent
tool calls in refinement context. A3 FAIL.

---

## A4: Conditional Combo (A1 + A2 together)

**Status: CONDITIONAL** — this arm runs ONLY if BOTH of the following hold after A1 and A2:

1. **Neither A1 nor A2 alone clears the INST-05 bar** (i.e., neither falsifier exit code is 0).
2. **Both A1 AND A2 show positive signal** (commit rate improved vs comparison floor, even
   if below 0.6).

Per D-13-01, the run-budget hard cap is 4 full live matrix runs total for this phase. A1 + A2
+ A3 consume 3. A4 is the 4th and final slot — if A4 does not qualify (any arm clears alone,
or both arms show no positive signal), the 4th slot is preserved but not used.

**Decision recorded in:** plan 13-07 (Closing Verdict and A4 Decision).

**Flag config (if run):** `VIABILITY_CONTRACT_ENABLED=1 FORCED_COMMIT_STEP=6`

### A4 Decision: NOT RUN

**Decision:** skip-a4 (recorded 2026-06-12, plan 13-07)

**Rationale:** A1 showed zero signal (0.000 pooled — the viability contract had no measurable
effect on gpt-5-mini), so the D-13-01 "both arms show positive signal" precondition is not
satisfied. D-13-01 requires BOTH A1 AND A2 to show positive signal for A4 to be sanctioned;
combining a zero-signal flag (A1) with a mechanism that never fired (A2 forced=0) has no
expected synergy, so the 4th run slot is left unused and the phase closes on the honest null
result.

**D-13-01 precondition check:**
1. Neither A1 nor A2 alone clears — SATISFIED (A1 exit=1, A2 exit=1).
2. Both A1 AND A2 show positive signal — NOT SATISFIED. A1 = 0.000 (no improvement over
   comparison floor of 0.000); A2 = 0.500 (improvement). A1 shows NO positive signal.
   The precondition requires BOTH arms to be positive-but-short; A1 being at floor disqualifies
   the combo.

**Run budget:** 3/4 slots consumed (A1 + A2 + A3). The 4th slot is unused per this decision.

---

## Closing Verdict

**Recorded in plan 13-07 (2026-06-12). A4 decision: skip-a4 (see above).**

### Per-Arm Summary Table

| Arm | Flag Config | gpt-5-mini pooled | deepseek-reasoner pooled | gpt-4o-mini anchor | Falsifier exit code |
|-----|-------------|-------------------|--------------------------|--------------------|---------------------|
| A1 | `VIABILITY_CONTRACT_ENABLED=1` | 0.000 (model-initiated 1/10, forced 0/10) | 0.000 (informational) | 1.000 — PASS (baseline 1.000) | 1 (FAIL) |
| A2 | `FORCED_COMMIT_STEP=6` | 0.500 (model-initiated 4/10, forced 0/10) | 0.000 (informational) | 1.000 — PASS, behaviorally unchanged | 1 (FAIL) |
| A3 | `PARALLEL_TOOL_EXECUTION_ENABLED=1` | 0.500 (model-initiated 5/10, forced 0/10) | 0.000 (informational) | 0.500 — REGRESSION (refinement_cheaper 0.000 vs baseline 1.000) | 1 (FAIL) |
| A4 | NOT RUN | — | — | — | — |

**A2 split-qualification note (D-13-04(c)):** A2 gpt-5-mini rate is 0.500 with forced=0 for
all models — the improvement is entirely model-initiated, not forced-inflated. Quality scorers
(category_compliance, constraints_satisfied, geographic_coherence) held on committed omakase
episodes. Anchor (gpt-4o-mini) was behaviorally unchanged at 1.000. The A2 rate is
split-qualified; it does not clear the 0.6 bar regardless.

### Explicit Closing Line

**No arm cleared the INST-05 falsifier bar. All arms plateaued below gpt-5-mini >= 0.6.**

- A1: 0.000 (viability contract + critique recalibration — zero effect on gpt-5-mini)
- A2: 0.500 (best signal; model-initiated improvement on omakase only; refinement_cheaper = 0.000; forced mechanism never fired; split-qualified per D-13-04(c) but still below 0.6)
- A3: FAIL on anchor regression (refinement_cheaper 0.000 vs 1.000 baseline) + latency unmeasurable
- A4: NOT RUN (D-13-01 precondition not satisfied — A1 showed no positive signal)

### Phase-14 Consequence

**Phase 14 (Richer State Replay) entry gate: OPEN.**

All DEC arms (A1, A2, A3) plateaued below the INST-05 falsifier bar at n=5. No arm cleared.
Phase 14 is entered per the conditional entry gate: multi-message reasoning-state replay
(REPLAY-01) and content-block preservation (REPLAY-02) A/B experiments proceed as the next
escalation against this documented plateau baseline. Phase 14 is NOT skipped; Phase 15
(Gate Promotion + Baseline Regen) does not proceed directly.
