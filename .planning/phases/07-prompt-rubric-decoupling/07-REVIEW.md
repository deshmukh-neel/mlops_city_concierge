---
phase: 07-prompt-rubric-decoupling
reviewed: 2026-06-03T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - app/agent/critique/checks.py
  - app/agent/io.py
  - app/agent/prompts.py
  - app/eval/config.py
  - configs/eval_matrix_refinement.yaml
  - scripts/eval_agent.py
  - tests/unit/test_agent_io.py
  - tests/unit/test_chat_functional.py
  - tests/unit/test_critique_checks.py
  - tests/unit/test_eval_agent.py
  - tests/unit/test_eval_matrix.py
findings:
  critical: 0
  warning: 2
  info: 5
  total: 7
status: warnings_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-06-03
**Depth:** standard
**Files Reviewed:** 11
**Status:** warnings_found

## Summary

Phase 7 ("Prompt/Rubric Decoupling") cleanly executes the bulk of its stated
contract: `SYSTEM_PROMPT` rule 10 is deleted, `_REFINEMENT_PREAMBLE` is rewritten,
`refinement_minimal_edit` Branch 5 grows a same-`primary_type` sub-check guarded
by a deterministic four-cell matrix, the eval-runner scratch payload now carries
`primary_type` per entry, and a logged-not-gated `openai/gpt-5-mini` matrix entry
threads the PROMPT-05 falsifier. The grep gate and integration tests pass; the
existing 8-cell injection truth-table is preserved; HIGH-4 prompt-injection
mitigation is byte-identically preserved (the model-facing JSON block is
unchanged).

No critical/security defects were found.

The findings below cluster around two themes:

1. **D-07-03 ↔ D-07-10 invariant slip.** The plan-07-07 preamble-tightening
   iteration ("Reuse the `place_id` and `slot` index of every stop you are not
   changing exactly as listed; only the slot named by the user gets a new
   `place_id`") is a behavioral prescription about HOW to satisfy the edit. The
   D-07-03 docstring still claims the preamble is "task description, no
   behavioral prescriptions." The grep gate (D-07-04) does not catch this
   phrasing because it whitelists by literal substring rather than by the
   stated invariant. This is documented in 07-07's accept-with-notes verdict
   but is not surfaced in the prompt's own comments or the scorer's docstring,
   so a future contributor reading the source will believe the prompt is
   behavior-free when it isn't.
2. **Defensive coverage gaps in the extended scorer.** The new D-07-07 abstain
   branch silently merges two distinct cases (legacy payload missing key vs
   user-named target slot not present in prior plan), and the surrounding
   `_try(...)` block still swallows TypeErrors from malformed
   `refinement_target_slot` values without surfacing them in `violations`.
   Neither is a regression introduced by Phase 7 — but Phase 7 widened the
   surface area and is the right place to tighten.

The accept-with-notes PROMPT-04 outcome from plan 07-07 is an explicit
architectural decision and is intentionally out of scope for this review.

## Warnings

### WR-01: `_REFINEMENT_PREAMBLE` re-introduces a behavioral prescription that the D-07-03 contract forbids and the grep gate misses

**File:** `app/agent/io.py:88-97`
**Issue:** The post-iteration preamble now contains the sentence:

> "Reuse the `place_id` and `slot` index of every stop you are not changing
> exactly as listed; only the slot named by the user gets a new `place_id`."

This is a behavioral prescription about HOW to satisfy the edit. CONTEXT.md
D-07-03 explicitly states the preamble "MUST NOT contain ... any other
prescription about HOW to satisfy the edit," and the docstring at `io.py:64-87`
still describes the preamble as task-only. The grep gate at
`tests/unit/test_critique_checks.py:992-1025` does not catch this phrasing
because the literal forbidden list only includes `byte-for-byte`, `same
primary_type`, `keep same stop count`, etc. — none of which appear in the new
sentence even though the semantic violation is identical.

Plan 07-07's SUMMARY documents this as the D-07-10 preamble-tightening
iteration accepted under PROMPT-04 "accept-with-notes." That is a legitimate
architectural decision, but the source code does not declare it. A future
contributor reading `io.py:64-87` (the comment block claiming the preamble is
task-only) plus the grep gate (which is green) will reasonably conclude no
behavioral prescriptions remain. The next prompt-rewrite plan could remove the
"Reuse the `place_id` ..." line on the assumption that it's redundant with the
scorer.

**Fix:**

```python
# Option A — annotate the constraint as a known tightening iteration.
# app/agent/io.py: extend the comment block above _REFINEMENT_PREAMBLE.
#
# D-07-10 tightening iteration (plan 07-07): the second sentence
# ("Reuse the `place_id` ... exactly as listed; only the slot named by
# the user gets a new `place_id`.") is a behavioral prescription
# retained against the D-07-03 task-only contract because removing it
# regressed `openai/gpt-4o-mini` × `refinement_cheaper` from 1.0 → 0.0.
# Accept-with-notes per 07-07 SUMMARY § PROMPT-04. The behavior it
# prescribes is *also* enforced by `refinement_minimal_edit` Branch 5;
# the prompt line is a recovery shim, not the source of truth.

# Option B — extend the grep gate to catch the semantic violation, not
# just the literal forbidden list. (Riskier — would currently fail.)
forbidden_semantic = [
    "reuse the `place_id`",       # rule-shaped imperative
    "exactly as listed",          # byte-equality euphemism
    "only the slot named",        # category-shaped imperative
]
```

Pick Option A. The source needs to declare the known D-07-03 violation so the
next prompt edit doesn't accidentally remove the recovery shim.

### WR-02: `refinement_minimal_edit` fails open on a malformed `refinement_target_slot` via the `_try(...)` exception swallow

**File:** `app/agent/critique/checks.py:553` (read site) + `app/agent/critique/checks.py:588-595` (`_try` fail-open in `itinerary_violations`)
**Issue:** The scorer reads `target_slot = state.scratch.get("refinement_target_slot")` and immediately uses it as an `int`: `current_target_idx = target_slot - 1` (line 553). If a future caller or a deserialized scratch payload passes `target_slot` as a string ("2"), a float (2.0), or another non-int, the subtraction at line 553 (or, for non-numeric types, at the dict lookup at line 547 inside the byte-fraction sum) raises a `TypeError`. The exception propagates out of `refinement_minimal_edit` and is silently swallowed by `_try(...)` in `itinerary_violations` (line 591), which logs a warning and **does not** add `refinement_minimal_edit` to `failed`. The merge gate (`refinement_minimal_edit == 1.0`) is therefore silently bypassed on a malformed payload — exactly the opposite of the strict-1.0 binary gate D-06-09 / D-07-05 promise.

The pre-Phase-7 Branch-5 byte-equality logic had the same exposure (line 547
`prior_by_slot[slot]` would also throw on a non-int slot key, caught by
`_try`), but Phase 7 widened the surface by adding the `target_slot - 1`
arithmetic. The current test for `target_slot is None` (Branch 2, line 497) is
not enough — it leaves the "wrong-type" path uncovered.

**Fix:**

```python
# In refinement_minimal_edit, between line 494 and 497, tighten the
# Branch 2 type check so non-int target_slot is fail-loud, not fail-open.

prior = state.scratch.get("prior_committed_stops")
target_slot = state.scratch.get("refinement_target_slot")

# Branch 2: refinement context but prior/target data missing OR malformed
# → fail-loud. A non-int target_slot is a scratch-contract violation and
# must surface as a 0.0, not silently abstain via _try's exception catch
# in itinerary_violations.
if target_slot is None or prior is None:
    return 0.0
if not isinstance(target_slot, int) or isinstance(target_slot, bool):
    # bool is a subclass of int in Python; exclude it explicitly so
    # `refinement_target_slot=True` cannot pass as slot 1.
    return 0.0
if isinstance(prior, list) and len(prior) == 0:
    return 0.0
```

Add a unit test that pins this: `state.scratch["refinement_target_slot"] =
"2"` → `refinement_minimal_edit(state) == 0.0` (not abstain, not TypeError).

## Info

### IN-01: PROMPT-02 grep gate's `forbidden` list contains a duplicate entry that's dead after `.lower()`

**File:** `tests/unit/test_critique_checks.py:1013-1020`
**Issue:** The list literal is:

```python
forbidden = [
    "keep same stop count",
    "do not ask clarifying questions",
    "preserve `place_id` byte-for-byte",
    "byte-for-byte",
    "same primary_type".lower(),       # = "same primary_type"
    "SAME primary_type".lower(),       # = "same primary_type"  ← DUPLICATE
]
```

After the `.lower()` calls fold, entries 5 and 6 are byte-equal — `len(set(...))
== 5`, not 6. Plan 07-05's SUMMARY explicitly notes this was "intentional for
clarity, matches the plan's literal list" — but the result is a list literal
where one element is silently dead. A reader who counts six forbidden phrases
in the source will think the gate enforces six distinct rules; in reality only
five are checked. A future contributor refactoring this could legitimately
delete the duplicate and weaken the documentation.

**Fix:** Either drop the duplicate (5-element list with a comment noting
case-insensitive coverage of "SAME primary_type"), or use a raw set:

```python
forbidden = {
    "keep same stop count",
    "do not ask clarifying questions",
    "preserve `place_id` byte-for-byte",
    "byte-for-byte",
    # Lowercased — covers both "same primary_type" and "SAME primary_type"
    # because `combined_lower` was passed through .lower() above.
    "same primary_type",
}
```

### IN-02: `refinement_minimal_edit` silently abstains when `target_slot` is not present in `prior_by_slot` (impossible-target-slot path)

**File:** `app/agent/critique/checks.py:552, 565-568`
**Issue:** Consider a user message "make stop 5 cheaper" when the prior committed plan only had 3 stops. The eval-runner-side YAML would have `target_slot=5`, and the scratch payload's `prior_committed_stops` would have entries for slots 1, 2, 3. Then:

1. `prior_by_slot = {1: ..., 2: ..., 3: ...}` (no key 5)
2. `prior_non_target_slots = [1, 2, 3]` (all != 5)
3. Branch 4 doesn't fire (non-empty list)
4. `byte_fraction` computed across slots 1-3
5. `prior_target_pt = prior_primary_type_by_slot.get(5)` → `None` (default, key missing)
6. Line 565: `if prior_target_pt is None: return byte_fraction` → silent abstain on the category check

This conflates two semantically distinct cases:
- **Legacy 06-06 payload** where `prior_committed_stops[i]` lacks the new
  `primary_type` key (intended abstain — migration path).
- **Impossible target slot** where the user/YAML named a slot that the prior
  plan never had (scratch-contract violation — should surface, not abstain).

The pre-Phase-7 scorer did not have this conflation because there was no
category sub-check. Phase 7's new abstain branch now masks the second case.

Note: the eval-runner side (`app/eval/config.py:142 — target_slot: int = Field(ge=1)`) only enforces `target_slot >= 1`, not `target_slot <= len(prior)`. The `/chat` injection block at `app/main.py:759-765` DOES enforce the upper bound, but the scorer does not — and the scorer is what the merge gate reads.

**Fix:** Add a defensive Branch-2.5 immediately after building `prior_by_slot`:

```python
# Branch 2.5 (fail-loud): refinement_target_slot is not in prior_by_slot.
# Distinct from D-07-07 abstain ("prior payload missing the new field"):
# this is "prior payload has no entry for the named slot at all."
if target_slot not in prior_by_slot:
    return 0.0
```

This costs one extra dict lookup and keeps the D-07-07 abstain semantic clean
(it now ONLY fires when the prior entry exists but the new `primary_type` key
is absent/None — the actual legacy-payload case).

### IN-03: `_constraints_for_case` carries dead `is not None` guards on required `ExpectedResults` fields

**File:** `scripts/eval_agent.py:549-553`
**Issue:** `ExpectedResults.min_stops` and `.max_stops` are typed `int = Field(ge=0)` (required, non-optional) at `app/eval/config.py:104-105`. The guard at line 552 `if min_s is not None and max_s is not None and min_s == max_s:` therefore checks for a condition that can never be False once `case.expected_results` itself is non-None. (The function does correctly handle `case.expected_results is None` indirectly: it doesn't dereference at all on that branch because `num_stops` is set by `explicit_num_stops_from_text` first — but if the YAML had `expected_results=None`, accessing `.min_stops` would `AttributeError`. The validator `normal_cases_have_expected_stops` blocks that at config load, so it's safe in practice.)

This is pre-existing Phase 6 code that Phase 7 did not touch — flagging for
cleanup only. Not a defect.

**Fix:**

```python
# scripts/eval_agent.py:549-553
if num_stops is None and case.expected_results is not None:
    min_s = case.expected_results.min_stops
    max_s = case.expected_results.max_stops
    if min_s == max_s:
        num_stops = min_s
```

### IN-04: `app/agent/io.py` docstring/comment text still refers to `byte-equal` and `preserve-byte-equal`, which the grep gate does not catch

**File:** `app/agent/io.py:77, 79, 107`
**Issue:** Three references appear in adjacent comments/docstrings:

- Line 77: `# byte-equal contract anchor) and \`arrival_time\` (downstream-timing`
- Line 79: `# against \`places_raw\` and are not needed for the preserve-byte-equal`
- Line 107: `    carries the byte-equal \`place_id\` anchors the model must preserve.`

These are documentation strings (not the model-facing preamble), and `byte-equal` is NOT on the D-07-04 forbidden list (only `byte-for-byte` is). So the grep gate is technically correct. However, the docs still describe the
helper as enforcing byte-equality, which contradicts the Phase 7 narrative
that the scorer (not the prompt/helper) enforces this rule. A reader looking
at the helper's docstring will think the helper is responsible for the
contract, not the scorer.

**Fix:** Update the docstring/comment block to point at the scorer as the
canonical enforcer:

```python
# io.py:73-82 comment block (existing):
# ... preserve-byte-equal contract.
# →
# ... preserve-byte-equal contract. The actual byte-equality enforcement
# lives in `app/agent/critique/checks.py::refinement_minimal_edit` (Phase 7
# / D-07-05) — this helper just surfaces the `place_id` anchors the scorer
# reads. The prompt is intentionally rule-free per D-07-03.
```

### IN-05: PROMPT-01 test name is searchable but verbose; consider extracting `_NEW_SLOT2_PLACE_ID` shape into the existing `_canonical_fixture_set` pattern

**File:** `tests/unit/test_chat_functional.py:820, 1190-1371`
**Issue:** Two minor maintainability nits in plan 07-06:

1. The new class constant `_NEW_SLOT2_PLACE_ID = "ChIJtest_fixture_NEW2_xxxxxx"` is 28 chars but `_CANON_PLACE_ID` is 26 — these don't share a length nor a shape (`NEW2_xxxxxx` vs `id_aaaaaa`). The plan-06-01 Task 3 validator only enforces `>= 20`, so both pass, but a uniform fixture-id convention (all 26 chars, all matching `ChIJtest_fixture_*_<6 lowercase>`) would make the truth-table reading easier.

2. The test method name `test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied` is 70 chars and tracks PROMPT-01 / D-07-11 — useful for grep navigation but pushes the line into PEP8 line-length warning territory in some configurations. Acceptable, just noting.

Neither is a defect. Flagging for awareness only.

---

_Reviewed: 2026-06-03_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
