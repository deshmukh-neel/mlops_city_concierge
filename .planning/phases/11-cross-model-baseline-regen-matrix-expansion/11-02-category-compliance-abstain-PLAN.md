---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - app/agent/critique/checks.py
  - tests/unit/test_critique_checks.py
autonomous: true
requirements: [BASE-01]
must_haves:
  truths:
    - "category_compliance returns None (abstains) when the committed itinerary has zero stops, instead of 1.0"
    - "A zero-stop run no longer inflates category_compliance medians in aggregation"
  artifacts:
    - path: "app/agent/critique/checks.py"
      provides: "category_compliance zero-stop abstain (WR-12 / D-11-03) + updated docstring"
    - path: "tests/unit/test_critique_checks.py"
      provides: "Zero-stop abstain unit coverage"
  key_links:
    - from: "app/agent/critique/checks.py category_compliance"
      to: "scripts/eval_agent.py aggregate_results None-score filter"
      via: "CheckResult.score float | None propagation"
      pattern: "if score is not None"
---

<objective>
Fix WR-12 / D-11-03: `category_compliance` currently returns 1.0 when there are zero committed stops, contradicting its own docstring and inflating the medians for decisiveness-failing providers (e.g. DeepSeek's zero-stop runs). Change it to ABSTAIN by returning `None`, so zero-stop runs are excluded from aggregation. This changes measurement semantics only — it does not change agent behavior — and must land before BASE-01 regen so the corrected score is what gets baked into committed baselines.

Purpose: Decisiveness failure is already captured by the hard-gated `committed_itinerary_rate` signal. `category_compliance` measures only what was committed; a zero-stop run carries no category-compliance signal and must not score as a perfect 1.0.
Output: Corrected scorer + zero-stop abstain tests.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: category_compliance abstains (None) on zero stops</name>
  <files>app/agent/critique/checks.py, tests/unit/test_critique_checks.py</files>
  <read_first>
    - app/agent/critique/checks.py — read `category_compliance` (lines 229-264, current line 254-255 returns 1.0 on `if not state.stops`). Read the docstring at lines 230-249 (the "score what we can measure" framing). Note the function's current signature is `-> float`; it must become `-> float | None`. Do NOT touch `category_compliance_strict` (lines 267+) — only `category_compliance` is in scope for D-11-03.
    - scripts/eval_agent.py — confirm `aggregate_results` (around line 1117-1123) already filters `if score is not None` over `result.deterministic.checks[name].score`, so a None return propagates cleanly with no aggregation change.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`app/agent/critique/checks.py (MODIFY)` shows the exact before/after and the abstain idiom contrast vs `refinement_minimal_edit` (which abstains with 1.0, NOT None).
    - tests/unit/test_critique_checks.py — read the existing `TestCategoryCompliance` (or equivalent) test class to mirror state-construction helpers (`_make_state` or similar) and the requested_primary_types fixture pattern.
  </read_first>
  <behavior>
    - Test: `category_compliance(state)` with `state.stops == []` and a non-empty `requested_primary_types` returns `None` (not 1.0).
    - Test: `category_compliance(state)` with `state.stops == []` and empty `requested_primary_types` also returns `None` — verify ordering: the abstain on the empty-requested path (returns 1.0) vs zero-stop path. Per D-11-03, zero stops abstains with None; confirm which guard fires first and assert the documented behavior.
    - Test: `category_compliance(state)` with committed stops and matching requested types still returns a float (no regression on the populated path).
    - Test (aggregation propagation): a results list mixing one zero-stop run (score None) and one populated run (score float) aggregates the median over the populated run only — the None is excluded.
  </behavior>
  <action>
    Change `category_compliance` return annotation from `-> float` to `-> float | None`. At the zero-stop guard (current line 254-255), replace `return 1.0` with `return None  # WR-12 / D-11-03: abstain — zero-stop runs carry no category signal; excluded from aggregation`. Update the function docstring: remove the implicit fail-open framing and state explicitly that a zero-stop itinerary abstains (returns None) because decisiveness failure is already the hard-gated committed_itinerary_rate signal. Preserve the existing empty-requested-types abstain (`if not requested: return 1.0`) — that is the D-03 contract and is unchanged. Add the four behavior tests to `tests/unit/test_critique_checks.py` following the existing test-class conventions. Do NOT modify `category_compliance_strict`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_critique_checks.py -v -k "category_compliance and (zero or abstain or none)"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "return None" app/agent/critique/checks.py` shows a None return inside `category_compliance` at the zero-stop guard
    - app/agent/critique/checks.py `def category_compliance(` signature ends in `-> float | None:`
    - `category_compliance_strict` still returns 1.0 on zero stops (unchanged — `grep -A2 "def category_compliance_strict" app/agent/critique/checks.py` followed by the zero-stop branch still shows 1.0)
    - `poetry run pytest tests/unit/test_critique_checks.py -k "category_compliance"` exits 0
  </acceptance_criteria>
  <done>category_compliance abstains with None on zero stops; strict variant untouched; aggregation excludes the None; tests pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| scorer → committed baseline JSON | category_compliance medians become committed empirical record; a fail-open 1.0 inflates a provider's apparent quality |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-04 | Tampering | category_compliance zero-stop return | mitigate | D-11-03 None-abstain prevents zero-stop runs from inflating committed category-compliance medians |
| T-11-05 | Denial of service | scorer None propagation | accept | aggregate_results already filters None scores; no new crash surface introduced |
| T-11-02-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH §Package Legitimacy Audit: none) |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_critique_checks.py -v` passes.
- `make test` full suite passes (per memory `full_suite_db_pool_contamination`).
- `poetry run mypy app/agent/critique/checks.py` — the `float | None` annotation typechecks.
</verification>

<success_criteria>
- WR-12 / D-11-03 fix present; zero-stop abstain verified; strict variant untouched; no agent-behavior change.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-02-SUMMARY.md` when done.
</output>
