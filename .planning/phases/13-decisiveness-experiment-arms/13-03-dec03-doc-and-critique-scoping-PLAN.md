---
phase: 13-decisiveness-experiment-arms
plan: 03
type: execute
wave: 2
depends_on: ["13-01"]
files_modified:
  - docs/decisiveness_dec03_decision.md
  - app/agent/revision.py
  - tests/unit/test_agent_revision.py
autonomous: true
requirements: [DEC-03]
must_haves:
  truths:
    - "A decision doc records the LOW_SIMILARITY_THRESHOLD change direction AND the low_similarity scoping decision BEFORE any threshold-touching code lands (roadmap criterion 4)"
    - "LOW_SIMILARITY_THRESHOLD is read from an env override (LOW_SIMILARITY_THRESHOLD_OVERRIDE) with code default unchanged at 0.55 (D-13-07)"
    - "When the viability-contract arm is enabled AND every requested stop already has a viable candidate, the low_similarity hint is suppressed (the model should commit, not rephrase) — gated so flag-off behavior is unchanged"
    - "DEC-03 is co-tuned with DEC-01: both ride the same VIABILITY_CONTRACT_ENABLED flag (mechanically enforced co-tuning per D-13-05) — DEC-03 never fires in isolation"
  artifacts:
    - path: "docs/decisiveness_dec03_decision.md"
      provides: "DEC-03 threshold direction + low_similarity scoping decision, documented before code"
      min_lines: 20
    - path: "app/agent/revision.py"
      provides: "Env-overridable threshold + flag-gated low_similarity scoping"
      contains: "LOW_SIMILARITY_THRESHOLD_OVERRIDE"
  key_links:
    - from: "app/agent/revision.py _diagnose_last_tool_result"
      to: "app.agent.viability.all_slots_viable"
      via: "rule8-already-met gate suppressing low_similarity when every slot is viable"
      pattern: "all_slots_viable"
---

<objective>
Land DEC-03 (critique recalibration) co-tuned with DEC-01 inside the single
VIABILITY_CONTRACT_ENABLED arm. Two parts: (1) DOCUMENT-FIRST — write the decision
doc recording the threshold-change direction and the low_similarity scoping
decision BEFORE any threshold-touching code lands (roadmap criterion 4); (2) make
LOW_SIMILARITY_THRESHOLD env-overridable (code default stays 0.55) and scope the
low_similarity hint so it stops firing once every requested stop has a viable
candidate — but ONLY when the arm flag is on, so flag-off behavior is
byte-identical.

Purpose: D-13-07 + roadmap criterion 4 require the direction + scoping decision
documented before the threshold change; the `critique-loop-and-commit-tool-conflict`
memory locks DEC-03 to DEC-01 (never tune in isolation), enforced by sharing the
VIABILITY_CONTRACT_ENABLED flag. Depends on 13-01 for the shared all_slots_viable
predicate (single source of truth for "every slot viable").

Output: `docs/decisiveness_dec03_decision.md` (written first), an env-overridable
threshold, flag-gated low_similarity scoping in revision.py, and tests.
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
@.planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md
@.planning/phases/13-decisiveness-experiment-arms/13-01-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write the DEC-03 decision doc BEFORE any threshold-touching code</name>
  <files>docs/decisiveness_dec03_decision.md</files>
  <read_first>
    - .planning/milestones/v2.2-MILESTONE-SEED.md (seed finding 2: threshold already IS 0.55; "set to 0.55" is a no-op — load-bearing, do not re-derive)
    - app/agent/revision.py lines 21-166 (LOW_SIMILARITY_THRESHOLD + _diagnose_one low_similarity firing — the surface the scoping decision describes)
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-07 (the two candidate DEC-03 changes: lower threshold via env override, and/or scope low_similarity to pre-candidate steps only)
  </read_first>
  <action>
    Create `docs/decisiveness_dec03_decision.md` documenting, BEFORE any code change, the two DEC-03 decisions per roadmap criterion 4. (a) THRESHOLD DIRECTION: record that the code default stays 0.55 (seed finding 2: it already is 0.55, so "set to 0.55" is a no-op) and the experiment knob is the `LOW_SIMILARITY_THRESHOLD_OVERRIDE` env var, used to test values BELOW 0.55 in the A1 arm only; state explicitly that the first A1 run keeps the override UNSET so the scoping change is measured alone, and a lower value (suggest 0.45) is tested only if A1 shows positive-but-short signal. (b) LOW_SIMILARITY SCOPING: record the decision to suppress the low_similarity hint once every requested stop has a viable candidate (rule8-met), so the critique loop stops pulling the model back to "rephrase" when it should commit — this makes low_similarity a PRE-candidate-only signal and resolves the `critique-loop-and-commit-tool-conflict` tension. Note the co-tuning enforcement: both DEC-03 changes ride the SAME VIABILITY_CONTRACT_ENABLED flag as DEC-01 (D-13-05), so DEC-03 can never be tuned in isolation. Reference D-13-07 and roadmap criterion 4 by ID in the doc body.
  </action>
  <verify>
    <automated>test -f docs/decisiveness_dec03_decision.md && grep -q "LOW_SIMILARITY_THRESHOLD_OVERRIDE" docs/decisiveness_dec03_decision.md && grep -q "VIABILITY_CONTRACT_ENABLED" docs/decisiveness_dec03_decision.md && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - `docs/decisiveness_dec03_decision.md` exists and is >= 20 lines.
    - The doc states the threshold direction (default stays 0.55; override knob is LOW_SIMILARITY_THRESHOLD_OVERRIDE; first run keeps it unset).
    - The doc states the low_similarity scoping decision (suppress once every slot has a viable candidate; pre-candidate-only).
    - The doc references D-13-07 and roadmap criterion 4, and names VIABILITY_CONTRACT_ENABLED as the shared co-tuning flag.
    - This task's commit lands BEFORE the Task 2 code change (doc-first ordering is enforced by task order within this plan).
  </acceptance_criteria>
  <done>The DEC-03 direction + scoping decision is documented before any threshold-touching code, satisfying roadmap criterion 4.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Make threshold env-overridable and scope low_similarity behind the arm flag</name>
  <files>app/agent/revision.py, tests/unit/test_agent_revision.py</files>
  <read_first>
    - app/agent/revision.py line 21 (LOW_SIMILARITY_THRESHOLD = 0.55 — becomes env-overridable; code default stays 0.55)
    - app/agent/revision.py lines 82-179 (_diagnose_one + _diagnose_last_tool_result — the low_similarity firing site and its caller, which has access to `state`)
    - app/agent/viability.py (all_slots_viable — the shared predicate from plan 13-01; import it to compute rule8-already-met)
    - app/main.py lines 753-763 (truthy-set env-flag parsing precedent for VIABILITY_CONTRACT_ENABLED)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "app/agent/revision.py — DEC-03 low_similarity scoping" (env-override pattern + flag-gated suppression shape)
    - tests/unit/test_critique_checks.py (existing _diagnose_one-adjacent test patterns to mirror; tests/unit/test_agent_revision.py does NOT exist yet — this task creates it new)
  </read_first>
  <behavior>
    - LOW_SIMILARITY_THRESHOLD == 0.55 when LOW_SIMILARITY_THRESHOLD_OVERRIDE is unset; equals the override float when set (e.g. 0.45).
    - With VIABILITY_CONTRACT_ENABLED unset (flag off): low_similarity fires exactly as today for a below-threshold semantic_search hit — flag-off behavior byte-identical.
    - With VIABILITY_CONTRACT_ENABLED=1 AND all_slots_viable(state)==True: _diagnose_last_tool_result returns None for what would otherwise be a low_similarity hint (suppressed — model should commit).
    - With VIABILITY_CONTRACT_ENABLED=1 AND all_slots_viable(state)==False: low_similarity still fires (pre-candidate steps keep the hint).
  </behavior>
  <action>
    In `app/agent/revision.py`: (1) Make the threshold env-overridable — `LOW_SIMILARITY_THRESHOLD = float(os.environ.get("LOW_SIMILARITY_THRESHOLD_OVERRIDE", "") or "0.55")` (import os; default literal 0.55 stays as the fallback only). (2) Add the flag-gated scoping in `_diagnose_last_tool_result` (which already has `state`): read `VIABILITY_CONTRACT_ENABLED` via the truthy-set parser; if the flag is ON and `all_slots_viable(state, LOW_SIMILARITY_THRESHOLD)` is True (imported from `app.agent.viability`), SKIP emitting any `low_similarity` hint — return None for that diagnosis so the model commits instead of rephrasing. The suppression applies ONLY to the `low_similarity` reason; `empty_results`, `all_closed`, `neighborhood_no_match`, and `tool_error` are unaffected. CRITICAL (flag-off byte-identity): when the flag is off, the code path must be exactly the current behavior — gate the new `all_slots_viable` call entirely behind the flag check so flag-off runs never call it. Do NOT lower the 0.55 code default — only the env override changes the value.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_agent_revision.py -v -k "low_similarity or override or viability_contract"</automated>
  </verify>
  <acceptance_criteria>
    - A test asserts LOW_SIMILARITY_THRESHOLD resolves to 0.55 with the override unset (re-import under monkeypatch) and to 0.45 when LOW_SIMILARITY_THRESHOLD_OVERRIDE=0.45.
    - A test asserts that with VIABILITY_CONTRACT_ENABLED unset, a below-threshold semantic_search state yields a low_similarity hint (flag-off behavior unchanged).
    - A test asserts that with VIABILITY_CONTRACT_ENABLED=1 and a state where all_slots_viable is True, _diagnose_last_tool_result returns None (low_similarity suppressed).
    - A test asserts that with VIABILITY_CONTRACT_ENABLED=1 and all_slots_viable False, low_similarity still fires.
    - `make typecheck` passes; existing tests/unit/test_critique_checks.py tests all still pass (test_agent_revision.py is created new by this task).
  </acceptance_criteria>
  <done>Threshold is env-overridable (default 0.55), low_similarity is suppressed once every slot is viable but only under the arm flag; flag-off is byte-identical; co-tuned with DEC-01 via the shared flag.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| LOW_SIMILARITY_THRESHOLD_OVERRIDE env → threshold value | Operator-controlled in eval; malformed value must not crash module import |
| state.scratch → all_slots_viable | Untrusted model output reaches the viability reader (mitigated in 13-01) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-03-01 | Denial of Service | malformed LOW_SIMILARITY_THRESHOLD_OVERRIDE crashing import | mitigate | `float(os.environ.get(...) or "0.55")` — empty/unset falls back; a non-float value raises ValueError at import which is operator-visible at run start, not a silent prod path (flag-off prod never sets the override) |
| T-13-03-02 | Tampering | low_similarity suppression weakening critique on the prod path | mitigate | suppression is gated behind VIABILITY_CONTRACT_ENABLED; flag-off (prod) byte-identity test pins current behavior |
| T-13-03-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_agent_revision.py tests/unit/test_critique_checks.py -v` passes (new file plus the pre-existing critique-check suite).
- `make typecheck` and `make lint` pass.
- Flag-off path is byte-identical to current revision behavior (no all_slots_viable call when flag off).
- docs/decisiveness_dec03_decision.md committed BEFORE the revision.py change (doc-first, roadmap criterion 4).
</verification>

<success_criteria>
- DEC-03 direction + scoping documented before code (roadmap criterion 4).
- Threshold env-overridable; default unchanged at 0.55.
- low_similarity scoped to pre-candidate steps under the arm flag; co-tuned with DEC-01 via the shared VIABILITY_CONTRACT_ENABLED flag (never isolated).
- Flag-off behavior byte-identical.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-03-SUMMARY.md` when done.
</output>
