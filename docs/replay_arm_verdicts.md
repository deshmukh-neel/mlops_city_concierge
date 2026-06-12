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
| Smoke (n=1) | [fill in Wave 3/4] |
| Full (n=5) | [fill in Wave 3/4] |

**Smoke arm_flags verification:** [fill — paste `arm_flags` dict from smoke run JSON;
expected shape: `{'viability_contract': False, 'forced_commit_step': 0, 'parallel_tool':
False, 'viability_threshold_override': None, 'replay_multi_message': True,
'replay_content_blocks': False}`]

### Per-model results

| Model | Pooled commit rate | Delta vs flag-off (0.000) | Delta vs A2 (0.500) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|---|---|
| openai/gpt-5-mini | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| openai/gpt-4o-mini (anchor) | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| deepseek/deepseek-reasoner | [fill] | [fill] | [fill] | [fill] | [fill] | (informational) |

**Falsifier exit code:** [fill — 0 (PASS) or 1 (FAIL) or 2 (ERROR)]

**Falsifier per-scenario breakdown (pasted verbatim):**
```
[fill — paste full eval_falsifier.py output from R1 full run dir]
```

### Closing verdict

[fill — state whether R1 clears the INST-05 bar, gpt-5-mini pooled rate, anchor result,
split (model-initiated vs forced), and the R1 contribution to the A2-positive-signal
comparison point]

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

---

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | [fill in Wave 3/4] |
| Full (n=5) | [fill in Wave 3/4] |

**Smoke arm_flags verification:** [fill — paste `arm_flags` dict from smoke run JSON;
expected shape: `{'viability_contract': False, 'forced_commit_step': 0, 'parallel_tool':
False, 'viability_threshold_override': None, 'replay_multi_message': False,
'replay_content_blocks': True}`]

### Per-model results

| Model | Pooled commit rate | Delta vs flag-off (0.000) | Delta vs A2 (0.500) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|---|---|
| openai/gpt-5-mini | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| openai/gpt-4o-mini (anchor) | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| deepseek/deepseek-reasoner | [fill] | [fill] | [fill] | [fill] | [fill] | (informational) |

**Falsifier exit code:** [fill — 0 (PASS) or 1 (FAIL) or 2 (ERROR)]

**Falsifier per-scenario breakdown (pasted verbatim):**
```
[fill — paste full eval_falsifier.py output from R2 full run dir]
```

### Closing verdict

[fill — state whether R2 clears the INST-05 bar, gpt-5-mini pooled rate, anchor result,
and whether the observed delta confirms the EXPECTED-NULL prediction from the evidence
audit or reveals a surprising signal]

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

**Decision recorded:** [fill — R3 QUALIFIED or R3 SKIPPED (with D-14-01 precondition check)]

### R3 Decision: [fill]

[fill — if QUALIFIED: proceed with run dirs and table below; if SKIPPED: record rationale
and D-14-01 precondition check]

### Run Dirs

| Run | Dir |
|-----|-----|
| Smoke (n=1) | [fill if R3 runs] |
| Full (n=5) | [fill if R3 runs] |

**Smoke arm_flags verification:** [fill if R3 runs]

### Per-model results

| Model | Pooled commit rate | Delta vs flag-off (0.000) | Delta vs A2 (0.500) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|---|---|
| openai/gpt-5-mini | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| openai/gpt-4o-mini (anchor) | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| deepseek/deepseek-reasoner | [fill] | [fill] | [fill] | [fill] | [fill] | (informational) |

**Falsifier exit code:** [fill if R3 runs]

**Falsifier per-scenario breakdown (pasted verbatim):**
```
[fill if R3 runs]
```

### Closing verdict

[fill]

---

## Closing Verdict

### Per-Arm Summary Table

| Arm | Flag Config | gpt-5-mini pooled | Delta vs floor (0.000) | Delta vs A2 (0.500) | gpt-4o-mini anchor | Falsifier exit |
|---|---|---|---|---|---|---|
| R1 | `REPLAY_MULTI_MESSAGE_ENABLED=1` | [fill] | [fill] | [fill] | [fill] | [fill] |
| R2 | `REPLAY_CONTENT_BLOCKS_ENABLED=1` | [fill] | [fill] | [fill] | [fill] | [fill] |
| R3 | `R1+R2` (conditional) | [fill or NOT RUN] | [fill or —] | [fill or —] | [fill or —] | [fill or —] |

### ARCH-FUT-01 Evaluation (on plateau)

**Status:** USER CHECKPOINT (D-14-08) — this section is filled after the live runs
complete and the per-arm table above is populated. It is a recommendation, not a decision;
Phase 15 scope finalization requires explicit user approval.

**(a) Cumulative evidence chain** [fill after live runs]:
- v2.1 (Phase 9): byte-correct per-message `_reasoning_state` round-trip verified for
  all four adapters. State capture and replay work correctly at the per-adapter level.
- Phase 13 DEC arms (A1/A2/A3): all plateaued below INST-05 bar. A2 showed positive
  signal at 0.500 (model-initiated, forced mechanism never fired). A3 triggered anchor
  regression on refinement_cheaper.
- Phase 14 REPLAY arms: [fill — R1 delta, R2 delta, R3 delta if run]
- Interpretation: [fill — does cumulative evidence suggest the block-level or multi-replay
  paths contribute to decisiveness, or does the plateau hold?]

**(b) ARCH-FUT-01 contingency** [fill after live runs]:
ARCH-FUT-01 entails architectural work to replace or augment the current LangGraph-based
agent loop with a custom loop that natively threads per-message reasoning state without
relying on `additional_kwargs` round-tripping across the `_prune_for_llm` boundary. This
would involve threading reasoning content through the graph reducer, exposing it to the LLM
on every in-window turn regardless of prune cutoff, and potentially enabling cross-request
persistence beyond the current intra-request-only scope (see deferred finding in CONTEXT.md).
Decision 3 (gpt-4o-mini anchor, ~30s/turn budget) bounds the recommendation space.

**(c) Written recommendation** [fill after live runs — USER CHECKPOINT before Phase 15]:
[One of:
  (i) "Ratify anchor and defer ARCH-FUT-01 to a future milestone" — likely if Phase 14
      also plateaus, since both replay interventions are EXPECTED-NULL on the tested cells
      and DeepSeek's decisiveness gap persists at the architecture level.
  (ii) "Promote winning replay flag and file ARCH-FUT-01 as a tracked debt item" — if a
       replay arm cleared the bar, document the promotion path and bound ARCH-FUT-01 scope.
  (iii) Other recommendation bounded by the data and Decision 3.
]

### Explicit Closing Line

[fill — one sentence: "Arm [X] cleared the INST-05 falsifier bar (gpt-5-mini >= 0.6,
anchor non-regression, exit code 0)" OR "No arm cleared the INST-05 falsifier bar.
All arms plateaued below gpt-5-mini >= 0.6 [with per-arm summary]."

Example plateau form: "No arm cleared the INST-05 falsifier bar. All arms plateaued
below gpt-5-mini >= 0.6 (R1=[fill], R2=[fill], R3=[fill or NOT RUN])."]

### Phase-15 Consequence

[fill — one of:
  (a) Winning arm: "Phase 15 (Gate Promotion + Baseline Regen) proceeds with flag config
      [X] from run dir [Y]."
  (b) Plateau: "Phase 15 scope finalization is a USER CHECKPOINT per D-14-08. The
      ARCH-FUT-01 Evaluation section above provides the recommendation. Phase 15 does not
      proceed until the user approves the scope."]
