# Replay Experiment Arm Verdicts

**Role:** REPLAY-05 record — the canonical per-arm verdict document for Phase 14.

**Cross-link:** This document closes out [`docs/decisiveness_arm_verdicts.md`](decisiveness_arm_verdicts.md)
(Phase-13 DEC-05 record, immutable). The Phase-13 record is NOT appended to or modified;
it stands as the historical plateau baseline for all Phase-14 delta comparisons.

---

## INST-05 Falsifier Definition

An arm **clears** the INST-05 bar when ALL of the following hold for the full n=5 run:

1. **gpt-5-mini pooled `committed_itinerary_rate` >= 0.6** across both scenarios
   (omakase_mission_open_ended + refinement_cheaper, 5 runs each = 10 episodes).
2. **gpt-4o-mini anchor holds >= its honest Phase-12 comparison-floor baseline** —
   the anchor must not regress.
3. **Falsifier exit code 0** (`scripts/eval_falsifier.py --run-dir ... --matrix-config
   configs/eval_matrix_arm.yaml --baselines-dir configs/eval_baselines` returns 0).

Phase-14 comparison points (D-14-07):
- **Flag-off plateau floor:** gpt-5-mini = 0.000 (A1, the zero-signal DEC arm)
- **Best DEC arm (A2):** gpt-5-mini = 0.500 (FORCED_COMMIT_STEP=6, positive signal)

Per-arm tables report THREE delta columns per model: pooled commit rate, Delta vs flag-off
floor (0.000), Delta vs A2 (0.500). Both deltas must be reported regardless of direction.

---

## Flag-Off Floor and Byte-Identity Verification (Task 1)

**Conducted:** 2026-06-12 (Plan 14-04, Wave 3, before any live arm spend)

**Byte-identity smoke run:** `eval_reports/2026-06-12T19-49-39Z` (scripted mode, n=1, no live keys, no arm flags set)

**arm_flags from scripted smoke:**
```
{'forced_commit_step': 0, 'parallel_tool': False, 'replay_content_blocks': False, 'replay_multi_message': False, 'viability_contract': False, 'viability_threshold_override': None}
```

**Verification result: PASS** — `replay_multi_message: False` and `replay_content_blocks: False` confirmed alongside the four Phase-13 keys (viability_contract, forced_commit_step, parallel_tool, viability_threshold_override). Flag-off path is byte-identical to Phase-13 plateau.

**Flag-off floor source:** The Phase-13 plateau numbers from `docs/decisiveness_arm_verdicts.md` (A1 full run dir `eval_reports/2026-06-12T06-25-52Z`) are used as the "Delta vs flag-off floor" denominator for all Phase-14 per-arm tables. The comparison floor is:
- gpt-5-mini: **0.000** pooled (A1 = 0.000 — the zero-signal DEC arm, same as pre-DEC comparison floor)
- gpt-4o-mini (anchor): **1.000** (Phase-12 comparison-floor baseline)
- deepseek-reasoner: **0.000** (informational)

No fresh n=5 control run is spent — the Phase-13 plateau numbers are reused as the flag-off floor per the run-budget contract (D-14-01). This preserves the ≤4-run hard cap for R1, R2, and conditional R3/escalation.

---

## Run Budget Contract

Per D-14-01, the hard cap is **≤ 4 full live matrix runs total** across all arms in this phase:

- **R1** — `REPLAY_MULTI_MESSAGE_ENABLED=1` (pure replay effect, all DEC flags UNSET)
- **R2** — `REPLAY_CONTENT_BLOCKS_ENABLED=1` (pure replay effect, all DEC flags UNSET)
- **R3** (conditional combo) — run ONLY if neither R1 nor R2 clears alone but both show
  positive signal (mirrors Phase-13 D-13-01 A4 rule)
- **Discretionary 4th run (escalation valve)** — if R1/R2/R3 all plateau but the best
  replay arm AND A2 each independently showed positive signal, the verdict doc may
  recommend ONE best-replay+`FORCED_COMMIT_STEP=6` stack run before declaring plateau.
  Hard cap: ≤4 full live matrix runs total this phase.

Per D-14-02, each full arm run is:
- 3 models × 2 scenarios × n=5, temp=1.0, sequential, via `configs/eval_matrix_arm.yaml`
- Smoke n=1 with `arm_flags` self-description verification before every full n=5 spend
- Scenario universe: `omakase_mission_open_ended` + `refinement_cheaper`
- late_night_closure_cascade excluded (D-10-09 quarantine); anthropic/gemini cells deferred
  (D-12-09)
- No billing top-ups; partials recorded honestly and never written as baselines (D-11-14)
- Anchor non-regression is mandatory for both judged arms (A3's anchor regression is the
  cautionary precedent)

---

## R1: Multi-Message Reasoning-State Replay

**Flag config:** `REPLAY_MULTI_MESSAGE_ENABLED=1`
**DEC arm flags:** ALL UNSET (pure replay effect, no attribution confounding)
**Mechanism:** When the flag is ON, `plan()` in `graph.py` replays EACH in-window
AIMessage's own `_reasoning_state` onto that message via `replay_reasoning_state_multi()`,
instead of the current most-recent-only injection. Per-message `_reasoning_state` is
already stashed in `additional_kwargs` by Phase 8/9; R1 uses what is already stored.

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | `eval_reports/2026-06-12T19-50-32Z` |
| Full (n=5) | `eval_reports/2026-06-12T20-00-05Z` |

**Smoke arm_flags verification:** `{'forced_commit_step': 0, 'parallel_tool': False, 'replay_content_blocks': False, 'replay_multi_message': True, 'viability_contract': False, 'viability_threshold_override': None}`

Confounded-run guard **PASS**: `replay_multi_message: True`, `replay_content_blocks: False`, all three DEC flags off. Full n=5 spend approved.

### Per-model results

| Model | Pooled commit rate | Delta vs flag-off (0.000) | Delta vs A2 (0.500) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|---|---|
| openai/gpt-5-mini | 0.500 (model-initiated 4/10, forced 0/10) | +0.500 | ±0.000 | 1.000 (median, 4/5 runs) | 0.000 (0/5) | FAIL — 0.500 < 0.6 bar |
| openai/gpt-4o-mini (anchor) | 1.000 (model-initiated 10/10, forced 0/10) | +1.000 | +0.500 | 1.000 | 1.000 | PASS — non-regression (baseline 1.000) |
| deepseek/deepseek-reasoner | 0.100 (model-initiated 1/10, forced 0/10) | +0.100 | -0.400 | 0.000 (median, 1/5 runs) | 0.000 (0/5) | (informational) |

**Falsifier exit code:** `1 (FAIL)`

**Falsifier per-scenario breakdown (pasted verbatim):**
```
============================================================
eval_falsifier: INST-05 Milestone Falsifier Report
============================================================
source: run dir eval_reports/2026-06-12T20-00-05Z

[openai/gpt-5-mini] committed_itinerary_rate per scenario:
  omakase_mission_open_ended: 1.000
  refinement_cheaper: 0.000

openai/gpt-5-mini: median-weighted committed_itinerary_rate = 0.500 < 0.6 (model-initiated 4/4, forced 0/4)  FAIL

[openai/gpt-4o-mini] committed_itinerary_rate per scenario (run vs baseline):
  omakase_mission_open_ended: run=1.000  baseline=1.000
  refinement_cheaper: run=1.000  baseline=1.000

openai/gpt-4o-mini: median-weighted = 1.000 >= baseline 1.000 (model-initiated 10/10, forced 0/10)  PASS

============================================================
eval_falsifier: VERDICT = FAIL
============================================================
```

**Note on deepseek-reasoner split:** The falsifier only reports gpt-5-mini and gpt-4o-mini (gated models). DeepSeek commit split verified from run-dir files: 1 committed omakase run (run-4, rate=1.0), 0 refinement runs → model-initiated 1/10, forced 0/10. Pooled = 0.100 (1/10 total episodes). Flagging that the falsifier "model-initiated 4/4" count reflects only the 4 episodes that had a committed result, consistent with Phase-13 CR-02 semantics (only committed episodes appear in the split).

### Closing verdict

R1 does NOT clear the INST-05 bar. gpt-5-mini pooled median-weighted commit rate = **0.500** (same as the best Phase-13 DEC arm A2, not an improvement). The R1 delta vs flag-off floor (0.000) is **+0.500** — this matches A2's positive signal exactly. The delta vs A2 (0.500) is **±0.000** — R1 brings no additional improvement over the best DEC arm.

The pattern mirrors A2 exactly: gpt-5-mini commits strongly on omakase (4/5 runs, median=1.000) but fails entirely on refinement_cheaper (0/5 runs). Multi-message reasoning-state replay does NOT resolve the asymmetry between scenarios.

**Anchor (gpt-4o-mini) non-regression: CONFIRMED.** Anchor held at 1.000 across both scenarios — no regression vs baseline 1.000. No red flag.

**DeepSeek (informational):** 1/10 pooled = 0.100 (1 committed omakase run, 0 refinement runs). Modest positive signal over baseline 0.000 but informational only.

R1 shows positive signal (matching A2's +0.500) without clearing the bar. Both R1 and R2 must be checked against the R3 qualification criteria after R2 completes.

---

## R2: Content-Block Preservation Through _prune_for_llm

**Flag config:** `REPLAY_CONTENT_BLOCKS_ENABLED=1`
**DEC arm flags:** ALL UNSET (pure replay effect, no attribution confounding)
**Mechanism:** When the flag is ON, the pre-cutoff replacement at `graph.py:228-235`
preserves the original `m.content` shape (list or str) verbatim instead of collapsing
via `str(m.content)`. Tool_calls are still stripped (the unanswered-tool_call contract
holds). Flag-off path is byte-identical to Phase-13 plateau.

### R2 Evidence Audit (D-14-05)

**Conducted:** 2026-06-12 (Plan 14-02, before any live R2 spend)
**Script:** `scripts/audit_list_content_aimessages.py`
**Run dirs examined:** Phase-13 arm run dirs
(`eval_reports/2026-06-12T06-25-52Z` A1 full,
`eval_reports/2026-06-12T07-27-03Z` A2 full,
`eval_reports/2026-06-12T08-30-52Z` A3 full)

#### Half (a) — Run-dir scan finding

**Finding:** EvalRunReport JSONs persist `queries[i].deterministic.tool_calls` as an
**integer count**. No serialized AIMessage `.content` or `.additional_kwargs` is present
in any run JSON file across the three Phase-13 arm run dirs (31 files in A2 alone, all
confirmed). The run JSONs are structurally **insufficient** to directly answer "did an
AIMessage carry list content pre-cutoff" — the message trace is not persisted.

This finding is not an error; it is the documented EvalRunReport shape. The structural
adapter analysis (half b) provides the ground truth answer.

#### Half (b) — Structural adapter analysis verdict

All four Phase-9 adapters were analysed against their source in `app/agent/adapters/`:

| Provider | Adapter | Content shape | List-content? | In run matrix? | str() collapse effect |
|---|---|---|---|---|---|
| openai | OpenAIReasoningAdapter | str | NO | YES (gpt-5-mini + anchor) | NO-OP |
| deepseek | DeepSeekReasonerAdapter | str | NO | YES (deepseek-reasoner) | NO-OP |
| anthropic | AnthropicAdapter | list[dict] — thinking + text blocks | YES | NO — DEFERRED (D-12-09) | LOSSY (unreachable in current runs) |
| gemini | GeminiAdapter | str | NO | NO — DEFERRED (D-12-09) | NO-OP (and deferred) |

**Key facts:**
- `OpenAIReasoningAdapter` and `DeepSeekReasonerAdapter` store reasoning state in
  `additional_kwargs["reasoning_content"]`. Their `AIMessage.content` is a plain
  string reply — NOT a block list.
- `AnthropicAdapter` is the ONLY adapter that uses a content block list
  (heterogeneous `list[dict]` with `{"type": "thinking", ...}` and `{"type": "text", ...}`
  blocks). This is explicitly documented in `anthropic.py`'s ASYMMETRY CALLOUT.
- Anthropic is **deferred** (D-12-09) and is NOT in the Phase-14 run matrix.

#### Audit conclusion

**VERDICT: R2 EXPECTED-NULL on all three tested cells** (gpt-5-mini, gpt-4o-mini,
deepseek-reasoner).

`str()` collapse at `graph.py:232` was a **NO-OP** for all three RUN models: `str(s) == s`
for any string, so no observable content loss could have occurred for these cells in the
Phase-12/13 arm runs. The only adapter that would have experienced a lossy collapse is
`AnthropicAdapter`, which uses a content block list — but Anthropic is deferred and was
not run.

**R2 still runs per roadmap criterion 2** — "an explanation of whether `str()` collapse
was causing observable loss in run JSONs" requires a measured A/B delta, not just a
structural assertion. The expected-null conclusion is the explanation; the R2 live run
provides the measurement confirmation. An EXPECTED-NULL result on R2 is a valid,
informative outcome that closes this criterion with evidence.

> **POST-RUN ANNOTATION (Plan 14-04, 2026-06-12): the half-(b) openai row above is
> INCORRECT for gpt-5-mini.** The audit's structural analysis examined the
> `OpenAIReasoningAdapter` and concluded "content shape: str → str() collapse NO-OP".
> That claim holds for plain Chat-Completions `ChatOpenAI` (gpt-4o-mini) but NOT for
> the gpt-5 family: since W10, gpt-5 models route through `OpenAIReasoningChatModel`
> (`app/llm_factory.py`) with `use_responses_api=True`, and Responses-API
> `AIMessage.content` IS a content-block LIST (the `_lift_reasoning_blocks` hook at
> `llm_factory.py:116` explicitly checks `isinstance(content, list)` — structural proof
> the list shape exists). The audit looked at the adapter, not the chat-model factory.
> Consequence measured below: R2 is NOT null on gpt-5-mini — it is catastrophically
> negative. The EXPECTED-NULL prediction held only for the two str-content models
> (gpt-4o-mini, deepseek-reasoner).

---

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | `eval_reports/2026-06-12T20-53-05Z` |
| Full (n=5) | `eval_reports/2026-06-12T20-58-32Z` |

**Smoke arm_flags verification:** `{'forced_commit_step': 0, 'parallel_tool': False, 'replay_content_blocks': True, 'replay_multi_message': False, 'viability_contract': False, 'viability_threshold_override': None}`

Confounded-run guard **PASS**: `replay_content_blocks: True`, `replay_multi_message: False`, all three DEC flags off. Full n=5 spend proceeded.

> **Process note (honesty contract):** the n=1 smoke ALREADY contained the gpt-5-mini
> 400 error cells (both scenarios, `status=error`). The D-14-02 smoke contract verifies
> the `arm_flags` dict only, which passed; the smoke's error cells were not inspected
> before the full spend (the A3 precedent treated a smoke error cell as informational
> and proceeded). The full run confirmed the failure is deterministic (10/10 episodes),
> so the spend produced the anchor + deepseek measurements and upgraded the gpt-5-mini
> finding from "n=1 anecdote" to "deterministic across n=5 × 2 scenarios" — but a
> future smoke contract should check error cells in addition to arm_flags.

### Per-model results

| Model | Pooled commit rate | Delta vs flag-off (0.000) | Delta vs A2 (0.500) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|---|---|
| openai/gpt-5-mini | ERRORED — 0/10 episodes evaluable (10/10 provider 400 errors) | — (no commit measurement; effective catastrophic regression) | — | N/A (5/5 errored) | N/A (5/5 errored) | FAIL — N/A (no evaluable cells) |
| openai/gpt-4o-mini (anchor) | 1.000 median-weighted (model-initiated 7/10, forced 0/10) | +1.000 | +0.500 | 1.000 (median; raw 4/5) | 1.000 (median; raw 3/5) | PASS — non-regression (median 1.000 >= baseline 1.000) |
| deepseek/deepseek-reasoner | 0.100 (model-initiated 1/10, forced 0/10) | +0.100 | -0.400 | 0.000 (median; raw 1/5) | 0.000 (0/5) | (informational) |

**gpt-5-mini error (verbatim, identical shape across all 10 episodes):**
```
Error code: 400 - {'error': {'message': 'No tool output found for function call call_gRV9GJVVMARaWorzeZzkPKaC.', 'type': 'invalid_request_error', 'param': 'input', 'code': None}}
```
(stage `turn0`, `BadRequestError`; only the `call_...` id differs per episode)

**Falsifier exit code:** `1 (FAIL)`

**Falsifier per-scenario breakdown (pasted verbatim):**
```
============================================================
eval_falsifier: INST-05 Milestone Falsifier Report
============================================================
source: run dir eval_reports/2026-06-12T20-58-32Z

[openai/gpt-5-mini] committed_itinerary_rate per scenario:
  omakase_mission_open_ended: N/A
  refinement_cheaper: N/A

openai/gpt-5-mini: median-weighted committed_itinerary_rate = N/A (no evaluable cells)  FAIL

[openai/gpt-4o-mini] committed_itinerary_rate per scenario (run vs baseline):
  omakase_mission_open_ended: run=1.000  baseline=1.000
  refinement_cheaper: run=1.000  baseline=1.000

openai/gpt-4o-mini: median-weighted = 1.000 >= baseline 1.000 (model-initiated 7/7, forced 0/7)  PASS

============================================================
eval_falsifier: VERDICT = FAIL
============================================================
```

**Anchor raw-count note (A3-precedent check):** gpt-4o-mini raw commits were 7/10
(omakase 4/5, refinement 3/5) vs 10/10 in R1. The falsifier criterion is the per-scenario
MEDIAN, which held at 1.000 on both scenarios — no regression by the INST-05 anchor
criterion, and no scenario flipped to majority-fail (the A3 red-flag condition was a
median collapse to 0.000). gpt-4o-mini is a plain Chat-Completions str-content model, so
R2's preservation branch is structurally a no-op for it; the 7/10 vs 10/10 movement is
within run-to-run noise on a no-op path. NOT flagged as a regression; recorded for
completeness.

### Closing verdict

R2 does NOT clear the INST-05 bar — and the measured result **REFUTES the EXPECTED-NULL
prediction for gpt-5-mini in the surprising direction the audit's closing caveat allowed
for** ("or reveals a surprising signal").

**Criterion-2 explanation (measured, not assumed):** "Was `str()` collapse causing
observable loss?" — NO. The opposite: for gpt-5-mini the `str()` collapse was
**load-bearing protection**. gpt-5-mini's Responses-API `AIMessage.content` is a
content-block list that embeds function-call items. The flag-off `str()` collapse
flattened those blocks into inert text; with `REPLAY_CONTENT_BLOCKS_ENABLED=1` the
preserved list re-sends `function_call` items whose paired ToolMessage outputs were
pruned (`_prune_for_llm` still drops pre-cutoff ToolMessages), violating OpenAI's
unanswered-tool_call contract → deterministic
`400 "No tool output found for function call"` on every episode (10/10). The
`AIMessage(content=m.content, ...)` constructor strips the `.tool_calls` ATTRIBUTE,
but Responses-API function-call state ALSO lives inside the content block list — the
D-14-06 "tool_calls are still stripped" assumption did not account for that second
channel.

For the two str-content models the EXPECTED-NULL prediction was CONFIRMED:
gpt-4o-mini (anchor) held at median 1.000/1.000 — non-regression PASS — and
deepseek-reasoner stayed at its plateau (raw 1/10, same raw rate as R1).

**R2 signal classification: NEGATIVE (not null, not positive).** R2 catastrophically
disables the very model the phase is trying to help. R2 therefore cannot contribute to
an R3 combo: D-14-01 requires BOTH R1 and R2 to show positive signal, and R2's signal is
strictly negative. (The R3 decision itself is recorded in the R3 section by Plan 14-05.)

**Anchor non-regression: CONFIRMED** (median criterion; raw-count movement noted above,
on a structurally no-op path for the anchor).

---

## R3: Conditional Combo (R1 + R2)

**Flag config:** `REPLAY_MULTI_MESSAGE_ENABLED=1 REPLAY_CONTENT_BLOCKS_ENABLED=1`
**DEC arm flags:** ALL UNSET

**Status: CONDITIONAL** — this arm runs ONLY if BOTH of the following hold after R1 and R2:

1. **Neither R1 nor R2 alone clears the INST-05 bar** (i.e., neither falsifier exit code is 0).
2. **Both R1 AND R2 show positive signal** (commit rate improved vs flag-off floor, even
   if below 0.6).

Per D-14-01, the run-budget hard cap is ≤4 full live matrix runs total. R1 + R2 consume 2
slots. R3 is the 3rd slot; the 4th (discretionary valve) is reserved.

**Decision recorded:** R3 SKIPPED — D-14-01 precondition NOT satisfied (see below)

### R3 Decision: NOT RUN

**D-14-01 precondition check (mirrors A4 decision format from docs/decisiveness_arm_verdicts.md):**

1. Neither R1 nor R2 alone clears the INST-05 bar — **SATISFIED**. R1 falsifier exit=1,
   R2 falsifier exit=1. Both arms failed to clear the 0.6 bar.

2. Both R1 AND R2 show positive signal (commit rate improved vs flag-off floor) — **NOT SATISFIED**.
   - R1: gpt-5-mini pooled = 0.500 vs floor 0.000 → **POSITIVE signal (+0.500)**
   - R2: gpt-5-mini ERRORED 10/10 episodes with deterministic 400 BadRequestErrors → **NEGATIVE signal**
     (no evaluable commit-rate measurement; catastrophic functional regression, not a partial miss)

   D-14-01 requires BOTH arms to be positive-but-short. R2 shows strictly negative signal —
   it catastrophically disables gpt-5-mini rather than showing any improvement. The combo would
   combine a positive-signal flag (R1) with a flag that produces deterministic provider failures
   (R2). Enabling both flags simultaneously would not resolve the R2 root cause (list-content
   function-call items re-sent to the Responses API without paired tool outputs); the 400 errors
   would occur regardless of whether R1 is also set.

**Conclusion:** R3 is NOT RUN. The precondition requires both R1 and R2 to show positive
(though sub-bar) signal — R2's negative result disqualifies the combo. Running R3 would
spend a budget slot on a configuration guaranteed to produce the same 10/10 400-error failure
as R2 alone.

**Run budget:** 2/4 slots consumed (R1 + R2). R3 slot is preserved-but-not-spent.

---

## Discretionary Stack Valve (best-replay + FORCED_COMMIT_STEP=6)

**Flag config (if run):** `REPLAY_MULTI_MESSAGE_ENABLED=1 FORCED_COMMIT_STEP=6`
**DEC arm flags:** `FORCED_COMMIT_STEP=6` SET; all other DEC flags UNSET (this is a
deliberate stack, not a pure arm — the escalation valve sanctions this cross-phase combination)

**Status: CONDITIONAL** — per D-14-01, the discretionary valve may run ONLY if:

1. R1/R2/R3 all plateau (none clears the INST-05 bar independently)
2. The best replay arm AND A2 each independently showed positive signal

**Decision recorded:** VALVE NOT RUN — D-14-01 precondition met but run not sanctioned
(see recommendation below)

### Valve Decision: NOT RUN

**D-14-01 precondition check:**

1. R1/R2/R3 all plateau (none clears independently) — **SATISFIED**.
   R1 exit=1 (0.500 < 0.6), R2 exit=1 (catastrophically negative), R3 NOT RUN.
   No replay arm cleared the INST-05 bar.

2. Best replay arm (R1) AND A2 each independently showed positive signal — **SATISFIED**.
   - R1 (best replay arm): gpt-5-mini 0.500 vs floor 0.000 → +0.500 (positive signal)
   - A2 (FORCED_COMMIT_STEP=6): gpt-5-mini 0.500 (model-initiated, positive signal)

   The D-14-01 escalation-valve precondition is formally satisfied. The valve is therefore
   within scope as an option before declaring plateau.

**Recommendation: NOT RUN.** Rationale:

R1 produced exactly the same gpt-5-mini rate as A2 (0.500 vs 0.500 — delta = ±0.000). This
means multi-message reasoning-state replay provides zero incremental signal over the forced-commit
mechanism alone. Stacking R1 on A2 (the valve configuration) would combine a mechanism with
zero marginal value (R1) on top of A2's forced-commit path. The expected additional signal is
zero by the transitivity of zero-delta evidence.

Furthermore, while the CR-01 synthesis bug fix (Phase 13-08) repaired the A2 forced-commit
synthesizer, the forced mechanism was still never confirmed to fire at n=5 on the fixed
synthesizer — A2's 0.500 result is entirely model-initiated even after the fix (a separate
A2 re-run on the fixed synthesizer would be needed to test forced-path effectiveness). A
R1+A2 valve run would test forced-path on the fixed synthesizer while also enabling R1, but
the R1 zero-delta evidence suggests R1 adds nothing; the net effect of the valve is
indistinguishable from an A2 retest alone.

An A2 retest (flag-off replay + FORCED_COMMIT_STEP=6 on the fixed synthesizer) would be a
cleaner next step, and that decision properly belongs to Phase 15 scope — not to the Phase 14
≤4-run budget. Spending the 3rd budget slot on the valve here would preempt the Phase 15
scope decision that is the explicit USER CHECKPOINT of this phase.

**Run budget:** 2/4 slots consumed (R1 + R2). The 3rd slot (and 4th) are preserved-but-unused.
Phase closes on a 2-of-4-slot budget with R3 and the valve both NOT RUN per their respective
precondition checks.

---

## Closing Verdict

### Per-Arm Summary Table

| Arm | Flag Config | gpt-5-mini pooled | Delta vs floor (0.000) | Delta vs A2 (0.500) | gpt-4o-mini anchor | Falsifier exit |
|---|---|---|---|---|---|---|
| R1 | `REPLAY_MULTI_MESSAGE_ENABLED=1` | 0.500 (model-initiated 4/10, forced 0/10) | +0.500 | ±0.000 | 1.000 — non-regression PASS | 1 (FAIL) |
| R2 | `REPLAY_CONTENT_BLOCKS_ENABLED=1` | ERRORED — 0/10 evaluable (10/10 provider 400s) | — (catastrophic; no measurement) | — | 1.000 median — non-regression PASS | 1 (FAIL) |
| R3 | `R1+R2` (conditional) | NOT RUN | — | — | — | — |
| Discretionary valve | `R1+FORCED_COMMIT_STEP=6` | NOT RUN | — | — | — | — |

**Table notes:**
- Anchor non-regression CONFIRMED for both R1 and R2 (median criterion; raw R2 anchor movement noted in R2 section as within-noise on a structurally no-op path)
- R3: precondition 2 not satisfied (R2 negative, not positive-but-short)
- Valve: precondition met but not run (R1 zero-delta vs A2 makes stack expected-marginal-signal = 0)
- DeepSeek (informational): R1 raw 1/10, R2 raw 1/10 — plateau unchanged; omitted from summary table per falsifier definition (gated on gpt-5-mini + anchor only)

---

### ARCH-FUT-01 Evaluation (on plateau)

**(a) Cumulative evidence chain:**

The following evidence chain spans v2.1 through Phase 14, establishing what we know about
the decisiveness gap and what reasoning-state interventions can and cannot contribute:

- **v2.1 (Phase 9):** Per-message `_reasoning_state` round-trip proved byte-correct through
  all four provider adapters (OpenAI, DeepSeek, Anthropic, Gemini). Each in-window
  tool-calling AIMessage has its own `_reasoning_state` stashed in `additional_kwargs`
  and D-08-07 preserves kwargs across the `_prune_for_llm` boundary. State capture and
  replay are mechanically correct at the per-adapter level. **The state round-trip is not
  the problem.**

- **Phase 13 DEC arms (A1/A2/A3 — all null below INST-05 bar):**
  - A1 (viability contract + critique recalibration): 0.000 — prompt-level framing of the
    commit precondition had zero measurable effect on gpt-5-mini decisiveness.
  - A2 (FORCED_COMMIT_STEP=6): 0.500 — best Phase-13 signal, entirely model-initiated on
    omakase (refinement_cheaper = 0.000). The forced mechanism never fired due to CR-01 bug
    (synthesizer fixed in Phase 13-08; untested at n=5 on fixed synthesizer). The 0.500 is
    a genuine model-behavioral signal, not a forced-mechanism artifact.
  - A3 (parallel tool execution): anchor regressed (refinement_cheaper 0.000 vs 1.000).
    Behaviorally unsafe for the anchor — not promotable.
  - **Cross-arm pattern:** gpt-5-mini commits decisively on omakase (1.000 median in A2/A3/R1)
    but consistently fails refinement_cheaper (0.000 median across all arms). This asymmetry
    persisted through every DEC and REPLAY intervention — it is the dominant structural finding.

- **Phase 14 REPLAY arms:**
  - R1 (multi-message reasoning-state replay): gpt-5-mini 0.500 — identical to A2; delta
    vs A2 = ±0.000. Multi-message replay of already-stored per-message reasoning state
    does not improve on single-message replay. The plateau-replication is exact.
  - R2 (content-block preservation): NEGATIVE — catastrophically breaks gpt-5-mini
    (10/10 provider 400s). Reveals that `str()` collapse was load-bearing protection for
    the Responses-API list-content path; preserving list content re-sends function-call
    items whose ToolMessage outputs were already pruned. The R2 failure is architectural:
    the prune window drops ToolMessages but the Responses-API tracks function-call state
    inside the content block list, not only in `.tool_calls`.
  - R3: NOT RUN (R2 negative disqualifies combo).
  - Valve: NOT RUN (R1 zero-delta over A2 makes stack expected-marginal-signal = 0).

- **Interpretation:** Every measurable intervention — prompt framing, forced-commit
  mechanisms, parallel tools, multi-message replay, and content-block preservation —
  has been tested against the gpt-5-mini decisiveness gap and found insufficient. The
  consistent signal is that gpt-5-mini commits on omakase but not refinement_cheaper,
  regardless of state richness or prompt framing. The problem is not state round-tripping
  (v2.1 proved it works), not reasoning state freshness (R1 = A2 exactly), and not
  content-block loss (R2 shows loss was not occurring for the tested models — and
  "fixing" it breaks the model). The evidence points toward a **behavioral gap in the
  model's handling of refinement-class tasks**, not a state-plumbing gap that replay
  interventions can bridge.

**(b) ARCH-FUT-01 contingency:**

ARCH-FUT-01 entails architectural work to replace or augment the current LangGraph-based
agent loop with a custom imperative loop that natively threads per-message reasoning state
without relying on `additional_kwargs` round-tripping across the `_prune_for_llm` boundary.
This was originally conceived as a contingency for the case where state loss was causing the
decisiveness gap — exposing reasoning content to the LLM on every in-window turn regardless
of prune cutoff, and potentially enabling cross-request persistence beyond the current
intra-request-only scope.

The Phase 14 evidence materially changes the case for ARCH-FUT-01: the state round-trip
was already byte-correct (v2.1), and delivering richer state (R1 = multi-message replay)
produced exactly zero improvement. The contingency's premise — "if all state-plumbing
interventions fail, rearchitect the loop" — is evaluated against a different finding: state
plumbing is not the bottleneck. The ARCH-FUT-01 trigger was "all DEC arms fail AND state
loss is the diagnosed cause"; the cumulative evidence no longer supports the state-loss
diagnosis for the tested models.

Decision 3 (gpt-4o-mini stays anchor, ~30s/turn budget) bounds the recommendation space:
no recommendation may require replacing gpt-4o-mini or exceeding the latency budget.

**(c) Written recommendation — USER CHECKPOINT before Phase 15:**

> **RECOMMENDATION: Ratify the gpt-4o-mini anchor and defer ARCH-FUT-01 to a documented
> future contingency. Phase 15 scope should be defined without ARCH-FUT-01 execution.**

Rationale:

1. **ARCH-FUT-01 is not the right next lever.** The cumulative evidence shows that state
   richness interventions (R1: multi-message replay) add zero signal over the best DEC arm
   (A2: 0.500). If state loss were the cause of the decisiveness gap, R1 should have
   outperformed A2 — it did not. ARCH-FUT-01 (custom imperative loop) would increase state
   richness further, but the evidence argues the decisiveness gap is behavioral (refinement
   scenario), not architectural (state round-tripping).

2. **The A2 forced-commit mechanism is untested at n=5 on the fixed synthesizer (CR-01
   fixed in Phase 13-08).** The highest-ROI next experiment is an A2 retest with the working
   synthesizer — a clean Phase-15 in-scope item that reuses the existing eval machinery at
   no additional engineering cost. If A2 on the fixed synthesizer clears the bar (forced
   commits fire on refinement_cheaper), Phase 15 can promote it directly. If it still
   plateaus, the evidence case for ARCH-FUT-01 becomes more defensible.

3. **The refinement_cheaper scenario is the blocking pattern.** gpt-5-mini commits at 1.000
   on omakase across all arms but fails at 0.000 on refinement_cheaper across all arms. This
   scenario-specific asymmetry survived every intervention. Understanding why refinement
   scenarios fail to trigger model-initiated commits is the diagnostic step before any
   architectural intervention.

4. **gpt-4o-mini anchor stays ratified.** The anchor held at 1.000 across all arms (R1 and
   R2), confirming Decision 3: the anchor is stable, committed, and within the ~30s/turn
   budget. No evidence warrants changing the anchor.

**Phase 15 scope (recommendation, not decision — USER CHECKPOINT):**

The recommended Phase 15 scope is:
- A2 retest on the fixed synthesizer (FORCED_COMMIT_STEP=6, CR-01 fixed) — tests whether
  the forced-commit path actually fires on refinement_cheaper and whether it can lift the
  pooled rate above 0.500
- Scenario-level root cause analysis of the refinement_cheaper 0.000 pattern across all arms
  (does the refinement prompt structure prevent viable candidates from appearing at step 6?
  are all slots truly viable at step 6 per the gate condition?)
- Gate promotion and baseline regen under the best-performing config (whichever of the above
  earns it, or the current A2 model-initiated result if neither improves)
- ARCH-FUT-01 explicitly deferred to a future milestone: file as tracked technical debt with
  the evidence package from Phases 13-14 as the trigger criteria

**This recommendation does not resolve Phase 15 scope — that is a USER CHECKPOINT per
D-14-08. The user must explicitly approve Phase 15 scope before any Phase 15 planning begins.**

---

### Explicit Closing Line

**No arm cleared the INST-05 falsifier bar. All Phase-14 REPLAY arms plateaued below
gpt-5-mini >= 0.6 (R1 = 0.500 — identical to best DEC arm A2, no incremental improvement;
R2 = NEGATIVE — catastrophic provider 400s on gpt-5-mini; R3 = NOT RUN — R2 disqualifies
combo; discretionary valve = NOT RUN — R1 zero-delta over A2 makes stack expected marginal
signal = 0).**

### Phase-15 Consequence

**Phase 15 scope finalization is a USER CHECKPOINT per D-14-08.** The ARCH-FUT-01
Evaluation section above provides the recommendation. Phase 15 does not proceed until the
user approves the scope.

**Recommended inputs for Phase 15 scoping decision:**
- Best performing config: A2 (`FORCED_COMMIT_STEP=6`, Phase-13 run dir `eval_reports/2026-06-12T07-27-03Z`) — gpt-5-mini 0.500 model-initiated; forced mechanism untested at n=5 on the fixed synthesizer (CR-01 fix: Phase 13-08)
- Blocking pattern: refinement_cheaper = 0.000 across all arms (gpt-5-mini never commits on this scenario under any tested intervention)
- Recommendation: A2 retest on fixed synthesizer before any architectural escalation

**USER CHECKPOINT RESOLVED (2026-06-12):** The user approved the recommended scope.
Phase 15 = A2 retest on the fixed synthesizer + refinement_cheaper root-cause analysis +
gate promotion/baseline regen. gpt-4o-mini anchor ratified. ARCH-FUT-01 is NOT executed —
filed as tracked technical debt with the Phases 13–14 evidence chain as its trigger criteria.
