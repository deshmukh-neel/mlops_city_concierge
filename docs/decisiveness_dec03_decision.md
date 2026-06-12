# DEC-03: Critique Recalibration — Decision Record

**Status:** Documented pre-code (roadmap criterion 4)
**Phase:** 13 — Decisiveness Experiment Arms
**Co-tuning constraint:** DEC-03 is part of Arm A1, always co-tuned with DEC-01 via `VIABILITY_CONTRACT_ENABLED` (D-13-05).
**Decision date:** 2026-06-12

---

## Background

Seed finding 2 (`.planning/milestones/v2.2-MILESTONE-SEED.md`):

> `LOW_SIMILARITY_THRESHOLD` already IS 0.55 (`app/agent/revision.py:21`, unchanged since the
> file was extracted). The "deferred fix: set it to 0.55" recorded in 09-04-SUMMARY is a
> no-op as written — if threshold tuning is attempted, the change is *lowering it below* 0.55,
> and it must be co-tuned with commit instructions.

The project memory `critique-loop-and-commit-tool-conflict` establishes that tightening the
critique-loop pressure (lower threshold → more `low_similarity` hints) and commit decisiveness
pull in opposite directions. Tuning either in isolation risks regressing the other metric.

D-13-07 requires two candidate DEC-03 changes inside the A1 arm:

- **(a)** Lower `LOW_SIMILARITY_THRESHOLD` below 0.55 via an env override (leave code default at 0.55).
- **(b)** Scope `low_similarity` hints to pre-candidate steps only — suppress the hint once every
  requested stop already has a viable candidate.

---

## Decision 1: Threshold Direction

**Code default stays 0.55.**

"Set it to 0.55" is a no-op (seed finding 2 — it is already 0.55). Changing the threshold
means *lowering* it below 0.55, which would accept weaker similarity matches as sufficient and
therefore commit sooner. This direction is directionally correct for decisiveness.

**Mechanism: `LOW_SIMILARITY_THRESHOLD_OVERRIDE` env var.**

The threshold becomes env-overridable:

```python
LOW_SIMILARITY_THRESHOLD: float = float(
    os.environ.get("LOW_SIMILARITY_THRESHOLD_OVERRIDE", "") or "0.55"
)
```

- When `LOW_SIMILARITY_THRESHOLD_OVERRIDE` is **unset or empty**, the value resolves to `0.55`
  (unchanged from the current code default).
- When set (e.g. `LOW_SIMILARITY_THRESHOLD_OVERRIDE=0.45`), the threshold is lowered to that value.

**First A1 run: override UNSET.**

The first arm run keeps `LOW_SIMILARITY_THRESHOLD_OVERRIDE` unset. This isolates the effect of
the scoping change (Decision 2 below) from any threshold change. Only if A1 shows positive-but-short
signal (e.g., commit rate improves but falls below the INST-05 bar of 0.6 at n=5) is a lower value
tested. The suggested lower value is **0.45** — one meaningful step below the current 0.55.

**Rationale for 0.45:** a 0.10-point reduction is large enough to change viability decisions for
borderline results (cosine 0.48–0.54) without collapsing the distinction between strong and weak
matches entirely. A value below 0.40 would likely degrade category_compliance.

---

## Decision 2: Low-Similarity Scoping

**Suppress `low_similarity` hints once every requested stop has a viable candidate.**

Currently `_diagnose_one` in `app/agent/revision.py` fires `low_similarity` unconditionally for
any `semantic_search` result below the threshold. This is correct when no slot has a viable
candidate yet — the model should rephrase and retry. But once **every** requested stop already has
at least one viable candidate in the scratch buffer (rule-8-met condition), the `low_similarity`
hint is counterproductive: it tells the model to rephrase when it should instead call
`commit_itinerary`.

**Scoping rule (rule8-met gate):** In `_diagnose_last_tool_result`, before emitting a
`low_similarity` hint:

1. Check if the arm flag is ON (`VIABILITY_CONTRACT_ENABLED` is truthy).
2. If ON, call `all_slots_viable(state, LOW_SIMILARITY_THRESHOLD)` (imported from
   `app.agent.viability`, the single source of truth from plan 13-01).
3. If `all_slots_viable` returns `True`, skip the `low_similarity` hint entirely — return
   `None` so the model is not told to rephrase.
4. All other hint reasons (`empty_results`, `all_closed`, `neighborhood_no_match`,
   `tool_error`) are unaffected and always fire when appropriate.

This makes `low_similarity` a **pre-candidate-only signal**: it applies exclusively to steps where
at least one slot is still without a viable candidate. This directly resolves the
`critique-loop-and-commit-tool-conflict` tension: once the viability precondition is met, critique
no longer obstructs the commit.

---

## Co-Tuning Enforcement (D-13-05)

Both DEC-03 changes ride the **same `VIABILITY_CONTRACT_ENABLED` flag as DEC-01**.

DEC-01 adds the explicit viability definition to rule 8 in the SYSTEM_PROMPT; DEC-03 scopes the
low_similarity critique once that viability condition is met. They are co-tuned by construction:
enabling the flag activates both the clearer viability contract AND the relaxed critique pressure.
Neither fires in isolation.

This is mechanically enforced by:

- `LOW_SIMILARITY_THRESHOLD_OVERRIDE` is only meaningful when `VIABILITY_CONTRACT_ENABLED=1`
  (the scoping gate uses the same threshold).
- `all_slots_viable` in `revision.py` is only called when the flag is ON.
- Flag-OFF behavior is byte-identical to current behavior: no `all_slots_viable` call is made,
  `_diagnose_last_tool_result` follows the existing code path exactly.

---

## Referenced Design Decisions

- **D-13-07:** DEC-03 candidate changes: (a) lower threshold via env override, (b) scope
  `low_similarity` to pre-candidate steps only. Direction must be documented before any code
  lands (roadmap criterion 4).
- **Roadmap criterion 4:** Decision doc recorded before threshold-touching code.
- **D-13-05:** All arms behind env flags, default OFF. `VIABILITY_CONTRACT_ENABLED` is the
  shared co-tuning flag for A1 (DEC-01 + DEC-03 together).
- **D-13-03:** Single source of truth for viability — `app/agent/viability.all_slots_viable`.

---

## Security Note

**Threat T-13-03-01:** Malformed `LOW_SIMILARITY_THRESHOLD_OVERRIDE` crashes module import.
Mitigation: `float(os.environ.get(...) or "0.55")` — empty/unset falls back to `"0.55"`; a
non-float value raises `ValueError` at import time, which is operator-visible at process start
rather than silently corrupting a runtime value. The prod path never sets this override.

**Threat T-13-03-02:** Suppressing `low_similarity` on the prod path weakens critique coverage.
Mitigation: the suppression is gated behind `VIABILITY_CONTRACT_ENABLED`; flag-OFF (prod) is
byte-identical to the current behavior, pinned by a unit test.
