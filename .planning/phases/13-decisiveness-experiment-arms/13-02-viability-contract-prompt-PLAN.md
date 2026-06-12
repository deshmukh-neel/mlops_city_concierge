---
phase: 13-decisiveness-experiment-arms
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - app/agent/prompts.py
  - tests/unit/test_agent_prompts.py
autonomous: true
requirements: [DEC-01]
must_haves:
  truths:
    - "A helper renders rule 8 with an explicit viability-definition sentence appended ONLY when the viability-contract arm is enabled; the sentence names the cosine threshold and matching primary_type"
    - "The addendum is purely additive — every existing phrase lock in test_agent_prompts.py passes in BOTH flag states"
    - "With the flag unset, the rendered system prompt is byte-identical to the current baseline (the viability sentence does not appear)"
    - "The Phase-7 CI grep gate (test_agent_io.py forbidden-phrase preamble gate) stays green — no behavioral-rubric phrases leak"
  artifacts:
    - path: "app/agent/prompts.py"
      provides: "Flag-gated rule-8 viability addendum builder (additive only)"
      contains: "cosine"
    - path: "tests/unit/test_agent_prompts.py"
      provides: "Both-flag-states additive-only assertions for DEC-01"
      contains: "VIABILITY_CONTRACT_ENABLED"
  key_links:
    - from: "app/agent/prompts.py viability addendum"
      to: "app.agent.revision.LOW_SIMILARITY_THRESHOLD"
      via: "threshold value interpolated into the viability sentence (import, never hardcode 0.55)"
      pattern: "LOW_SIMILARITY_THRESHOLD"
---

<objective>
Ship the DEC-01 viability-contract prompt variant: an ADDITIVE extension to
SYSTEM_PROMPT rule 8 that, when the arm flag is on, appends a single sentence
defining viability precisely ("a result with cosine >= {threshold} and matching
primary_type IS viable — do not keep searching past it"). The variant is selected
by a helper so plan 13-04 can wire it into graph-build-time prompt assembly.

Purpose: D-13-06 requires DEC-01 to be additive-only — every pinned phrase in the
Phase-7 prompt-lock gate must stay green in BOTH flag states, and the
test_agent_io.py forbidden-phrase preamble gate must not trip. Roadmap success
criterion 1 makes the grep gate staying green a merge condition.

Output: a flag-gated viability-addendum builder in prompts.py and new both-flag-
state tests. This plan touches ONLY prompts.py + its test (no graph.py wiring yet
— that is plan 13-04), so it has no file overlap with any Wave-1 sibling.
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
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add a flag-gated, additive rule-8 viability addendum builder</name>
  <files>app/agent/prompts.py</files>
  <read_first>
    - app/agent/prompts.py lines 69-191 (SYSTEM_PROMPT, esp. rule 8 lines 151-161 — the existing decisive-commit text the addendum extends; note SYSTEM_PROMPT.format takes max_steps + current_datetime placeholders)
    - app/agent/revision.py line 21 (LOW_SIMILARITY_THRESHOLD — import this; the addendum interpolates it, never hardcodes 0.55)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "app/agent/prompts.py — DEC-01 viability text additive to rule 8" (exact addendum sentence + flag-conditional pattern: build the addendum string at graph-build time and append after SYSTEM_PROMPT.format(...))
    - tests/unit/test_agent_prompts.py lines 91-108 (test_system_prompt_has_decisive_commit_contract — the existing locks that must stay green: "one viable option"/"good enough" AND "do not keep"/"don't keep"/"stop optimizing")
  </read_first>
  <behavior>
    - rule8_viability_addendum(enabled=True) returns a non-empty string containing the phrases "cosine" and "is viable" and the LOW_SIMILARITY_THRESHOLD value (e.g. "0.55").
    - rule8_viability_addendum(enabled=False) returns "" (empty string).
    - The addendum does NOT delete or alter any existing rule-8 text — it is appended.
    - The addendum text contains NONE of the forbidden behavioral-rubric phrases from the test_agent_io.py gate ("byte-for-byte", "SAME primary_type").
  </behavior>
  <action>
    Add a function `rule8_viability_addendum(enabled: bool, threshold: float | None = None) -> str` to `app/agent/prompts.py`. When `enabled` is False, return `""`. When True, return a single appended sentence: `f"   A result with cosine similarity >= {threshold} and matching primary_type IS viable — do not keep searching past it.\n"`, where `threshold` defaults to `LOW_SIMILARITY_THRESHOLD` imported from `app.agent.revision`. Keep the wording ADDITIVE — do NOT modify the existing SYSTEM_PROMPT rule-8 string; the addendum is meant to be concatenated AFTER `SYSTEM_PROMPT.format(...)` by the caller (plan 13-04). CRITICAL (D-13-06): the addendum must not contain any phrase on the test_agent_io.py forbidden list (`byte-for-byte`, `SAME primary_type`) and must not remove the existing pinned phrase `one viable option`. Import LOW_SIMILARITY_THRESHOLD — never write the literal 0.55.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_agent_prompts.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `rule8_viability_addendum(False)` returns `""`.
    - `rule8_viability_addendum(True)` contains "cosine", "is viable", and the threshold value from LOW_SIMILARITY_THRESHOLD.
    - `grep -n "0\.55" app/agent/prompts.py` shows no hardcoded literal in the addendum (threshold imported).
    - `rule8_viability_addendum(True)` contains none of: "byte-for-byte", "SAME primary_type".
  </acceptance_criteria>
  <done>An additive, flag-gated viability addendum builder exists in prompts.py, threshold imported, no behavioral-rubric phrases.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Lock both flag states with additive-only prompt tests</name>
  <files>tests/unit/test_agent_prompts.py</files>
  <read_first>
    - tests/unit/test_agent_prompts.py lines 91-108 (existing decisive-commit-contract lock — the assertions both flag states must satisfy)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "tests/unit/test_agent_prompts.py — flag-state assertions for DEC-01" (test_viability_contract_addendum_is_additive + test_viability_contract_flag_off_unchanged shapes)
    - tests/unit/test_agent_io.py lines 77-99 (the forbidden-phrase preamble gate this addendum must not violate)
  </read_first>
  <behavior>
    - Flag-ON: base rule-8 + addendum still satisfies every existing lock ("one viable option"/"good enough", "do not keep"/"don't keep"/"stop optimizing") AND contains "cosine similarity" + "is viable".
    - Flag-OFF: the rendered prompt (SYSTEM_PROMPT.format + addendum("")) contains no "cosine similarity" text and is byte-identical to SYSTEM_PROMPT.format alone.
  </behavior>
  <action>
    Add two tests to `tests/unit/test_agent_prompts.py`: (1) `test_viability_contract_addendum_is_additive` — render `SYSTEM_PROMPT.format(max_steps=8, current_datetime="...") + rule8_viability_addendum(True)` and assert ALL existing rule-8 locks hold AND `"cosine similarity"` and `"is viable"` appear (lowercased). (2) `test_viability_contract_flag_off_unchanged` — assert `SYSTEM_PROMPT.format(...) + rule8_viability_addendum(False)` equals `SYSTEM_PROMPT.format(...)` (byte-identical; the addendum is empty) and `"cosine similarity"` does NOT appear. Reference the flag name `VIABILITY_CONTRACT_ENABLED` in a comment/docstring so the gate's coverage is greppable. Do NOT weaken or delete any existing assertion in the file.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_agent_prompts.py tests/unit/test_agent_io.py -v</automated>
  </verify>
  <acceptance_criteria>
    - Both new tests pass.
    - All pre-existing tests in test_agent_prompts.py and test_agent_io.py still pass (Phase-7 grep gate green in both flag states).
    - `grep -c "VIABILITY_CONTRACT_ENABLED" tests/unit/test_agent_prompts.py` >= 1.
  </acceptance_criteria>
  <done>Both flag states are locked: addendum is provably additive, flag-off is byte-identical, Phase-7 grep gate stays green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| arm flag → prompt content | The viability-contract flag controls one appended sentence; no user input reaches the addendum |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-02-01 | Tampering | prompt-lock gate bypass via addendum | mitigate | test_agent_prompts.py asserts existing locks hold in BOTH flag states; test_agent_io.py forbidden-phrase gate runs in the same suite — a behavioral-rubric phrase leaking into the addendum fails CI |
| T-13-02-02 | Repudiation | silent prompt drift between flag states | mitigate | flag-off byte-identity test pins the baseline; any non-additive change fails |
| T-13-02-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_agent_prompts.py tests/unit/test_agent_io.py -v` passes.
- `make lint` passes (ruff, line-length 100).
- Flag-off rendered prompt is byte-identical to current baseline (no behavioral change with flag off — preserves the stale-baseline CI gate's [skip-baseline] eligibility for this file).
</verification>

<success_criteria>
- DEC-01 viability addendum is additive-only and flag-gated.
- Both flag states pass every prompt-lock assertion; Phase-7 grep gate stays green (roadmap criterion 1).
- Flag-off is byte-identical to baseline.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-02-SUMMARY.md` when done.
</output>
