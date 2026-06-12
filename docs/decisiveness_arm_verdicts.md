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
| Smoke (n=1) | `eval_reports/[fill after smoke run]` |
| Full (n=5) | `eval_reports/[fill after full run]` |

**Smoke arm_flags verification:** `[fill — paste arm_flags field from one smoke run JSON]`

### Per-model results

| Model | Pooled commit rate | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|
| openai/gpt-5-mini | [fill] | [fill] | [fill] | [fill] |
| openai/gpt-4o-mini (anchor) | [fill] | [fill] | [fill] | [fill — non-regression check] |
| deepseek/deepseek-reasoner | [fill] | [fill] | [fill] | (informational) |

**Falsifier exit code:** `[fill — 0/1/2]`

**Falsifier per-scenario breakdown (pasted verbatim from falsifier output):**

```
[fill — paste complete falsifier output including per-scenario lines]
```

### Closing verdict

`[fill — which model(s) cleared, or honest null result; note override status]`

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
| Smoke (n=1) | `eval_reports/[fill after smoke run]` |
| Full (n=5) | `eval_reports/[fill after full run]` |

**Smoke arm_flags verification:** `[fill — paste arm_flags field from one smoke run JSON]`

### Per-model results

| Model | Pooled commit rate (with split) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|
| openai/gpt-5-mini | [fill: rate (model-initiated M/N, forced F/N)] | [fill] | [fill] | [fill] |
| openai/gpt-4o-mini (anchor) | [fill: rate (model-initiated M/N, forced F/N)] | [fill] | [fill] | [fill — anchor red-flag assessment] |
| deepseek/deepseek-reasoner | [fill: rate (model-initiated M/N, forced F/N)] | [fill] | [fill] | (informational) |

**Falsifier exit code:** `[fill — 0/1/2]`

**Falsifier per-scenario breakdown (pasted verbatim from falsifier output, including split lines):**

```
[fill — paste complete falsifier output including model-initiated vs forced split per model]
```

**Anchor red-flag assessment:** `[fill — explicit statement: anchor behaviorally unchanged /
or red flag description if anchor changed]`

### Closing verdict

`[fill — A2 clears only if quality scorers hold >= baseline AND anchor is behaviorally unchanged;
note the model-initiated vs forced split for gpt-5-mini]`

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
| Smoke (n=1) | `eval_reports/[fill after smoke run]` |
| Full (n=5) | `eval_reports/[fill after full run]` |

**Smoke arm_flags verification:** `[fill — paste arm_flags field from one smoke run JSON]`

### Per-model results (commit rate — informational only for A3)

| Model | Pooled commit rate | omakase | refinement_cheaper | Quality scorers |
|---|---|---|---|---|
| openai/gpt-5-mini | [fill] | [fill] | [fill] | [fill — non-regression check] |
| openai/gpt-4o-mini (anchor) | [fill] | [fill] | [fill] | [fill — non-regression check] |
| deepseek/deepseek-reasoner | [fill] | [fill] | [fill] | (informational) |

**Falsifier exit code:** `[fill — 0/1/2; note: A3 verdict does NOT use falsifier exit code for pass/fail]`

**Falsifier per-scenario breakdown (pasted verbatim, for scorer non-regression evidence):**

```
[fill — paste complete falsifier output]
```

### Latency Analysis (gpt-4o-mini anchor — primary A3 pass criterion)

**Comparison baseline run dir (Phase-12 floor):** `eval_reports/[fill — from 13-05-SUMMARY or Phase-12 run records]`
**A3 arm run dir:** `eval_reports/[fill after full run]`

| Metric | Phase-12 baseline | A3 arm | Delta |
|---|---|---|---|
| Mean tool_execution_seconds (omakase) | [fill] | [fill] | [fill] |
| Mean tool_execution_seconds (refinement_cheaper) | [fill] | [fill] | [fill] |
| Mean tool_execution_seconds (pooled) | [fill] | [fill] | [fill] |

**Source:** INST-04 `step_telemetry` — `tool_exec_seconds` per run, summed across steps and
averaged over n=5. Values read from individual run JSONs in the run dirs above.

**Scorer non-regression assessment:** `[fill — explicit statement: scorers held / or regression
described with affected metric]`

### Closing verdict

`[fill — A3 verdict: latency reduced / not reduced; scorers held / regressed; A3 passes/fails
on latency+scorer, not commit rate]`

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

### Run Dirs (fill only if A4 qualifies)

| Run | Dir |
|-----|-----|
| Smoke (n=1) | `eval_reports/[fill if A4 runs]` |
| Full (n=5) | `eval_reports/[fill if A4 runs]` |

### Per-model results (fill only if A4 runs)

| Model | Pooled commit rate | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|
| openai/gpt-5-mini | [fill or N/A] | [fill or N/A] | [fill or N/A] | [fill or N/A] |
| openai/gpt-4o-mini (anchor) | [fill or N/A] | [fill or N/A] | [fill or N/A] | [fill or N/A] |
| deepseek/deepseek-reasoner | [fill or N/A] | [fill or N/A] | [fill or N/A] | (informational) |

**Falsifier exit code:** `[fill or N/A — depends on A4 qualification decision in 13-07]`

---

## Closing Verdict

`[Fill in plan 13-07 after all judged arms are recorded. Names the winning arm (if any) that
cleared the INST-05 bar, or records the honest null result for Phase 14 conditional entry.]`
