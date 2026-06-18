---
phase: 18-gap-mining-gap
reviewed: 2026-06-18T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - scripts/sandbox_guard.py
  - scripts/seed_demand_log.py
  - scripts/coverage_agent.py
  - tests/unit/test_gap_miner.py
  - tests/unit/test_gap_miner_smoke.py
  - tests/integration/test_gap_miner.py
  - tests/unit/test_sandbox_guard.py
  - tests/unit/test_seed_demand_log.py
  - Makefile
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: resolved
resolution: >
  All 3 Warning findings fixed during execution (commit follows this REVIEW.md):
  WR-01 (vacuous injection-test assertion → JSON round-trip assertion),
  WR-02 (shared-reference aliasing in _extract_demand_batch empty list →
  per-element comprehension), WR-03 (__import__("os") hack → top-level import os).
  Also resolved IN-01 (dropped redundant build_seed_queries re-import) and
  clarified IN-02's comment (guard is QUERY_LIMIT-defensive, not dead). IN-03
  (unanchored substring match) and IN-04 (cold-start guard silence) accepted as
  documented residual/correct-behavior. Suite green: 115 unit, integration skips.
---

# Phase 18: Code Review Report

**Reviewed:** 2026-06-18T00:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found (0 Critical, 3 Warning, 4 Info)

## Summary

Phase 18 adds a demand-driven gap miner on top of the existing supply-side `coverage_agent.py`. The implementation correctly addresses all four cross-AI review findings (HIGH-1 pair-level supply, HIGH-2 dedup split, HIGH-3 guard, ROUND-2/3 findings). The SQL injection surface, sandbox-write guard, catalog constraint, and connection lifecycle are all handled correctly. No critical or security-blocking defects were found.

Three findings warrant fixes before the next phase: a vacuously-true test assertion that provides no injection protection guarantee, a latent mutable-default sharing pattern in `_extract_demand_batch`, and an avoidable `__import__('os')` hack. The four info findings are minor quality issues.

All previously-guarded supply-only functions (`gather_stats`, `find_gaps`, `propose_queries`, `filter_already_covered`, `existing_query_texts`) are unchanged. No regressions to the existing W5 supply path were introduced.

---

## Warnings

### WR-01: Test assertion for prompt-injection protection is vacuously true

**File:** `tests/unit/test_gap_miner.py:242`

**Issue:** The second assertion in `TestPromptInjectionSafety.test_messages_are_json_encoded_in_prompt` is logically vacuous — it always passes regardless of whether the raw injection string leaks into the prompt:

```python
assert '"]}) DROP TABLE; --' not in prompt or json.dumps([message]) in prompt
```

The right-hand operand `json.dumps([message]) in prompt` is always `True` because the prompt itself is built from `json.dumps([message])` (i.e., `encoded_messages`). The `OR` short-circuits and the test never actually validates injection protection. An implementation that f-string-interpolated the raw message directly (a bug) would still pass this assertion.

The first assertion on line 241 (`assert json.dumps(message) in prompt`) is correct and meaningful. The second adds false confidence.

**Fix:**

Replace the vacuous second assertion with one that directly verifies the raw message does NOT appear in the prompt unescaped at a structural boundary, or remove it entirely and document why the first assertion suffices:

```python
# The first assertion verifies the JSON-encoded form is present (correct).
assert json.dumps(message) in prompt
# The raw string appears as a *substring* of the JSON encoding but that is expected
# (it's inside proper JSON quoting and cannot escape the string boundary).
# No additional assertion is needed — the json.dumps call is the protection.
```

Alternatively, for real negative coverage: mock `_build_demand_batch_prompt` to inject the raw string directly (bypassing `json.dumps`) and verify the second assertion would then fail.

---

### WR-02: Mutable shared list objects returned from `_extract_demand_batch` when `llm=None`

**File:** `scripts/coverage_agent.py:278`

**Issue:** The `empty` sentinel is constructed with list multiplication:

```python
empty = [([], [])] * len(messages)
```

In Python, `([], [])` is one tuple containing two specific list objects. Multiplying it creates N references to the **same** tuple — all N slots share identical inner list objects. Mutating `empty[0][0]` (the neighborhoods list of slot 0) also corrupts `empty[1][0]`, `empty[2][0]`, etc.

The current `gather_demand` callers only rebind (`hoods = llm_hoods`, `cuiss = llm_cuiss`) rather than mutate, so the bug does not trigger today. However, the contract is fragile: any future caller that does `llm_hoods.extend([...])` or `catalog_hoods = item.get(...)` followed by `llm_hoods += ...` would silently corrupt all slots.

**Fix:**

Use a list comprehension to produce independent list objects per slot:

```python
# Before (line 278):
empty = [([], [])] * len(messages)

# After:
empty = [([], []) for _ in range(len(messages))]
```

This is a one-line change that eliminates the sharing without changing the return type or semantics.

---

### WR-03: `__import__('os')` hack instead of a module-level import

**File:** `scripts/coverage_agent.py:800`

**Issue:** `gap_mine_main` accesses `os.environ` via a dynamic import:

```python
demand_url = __import__("os").environ.get("DEMAND_DATABASE_URL", None)
```

The 18-02 SUMMARY notes that ruff flagged `os` as an unused import at some point, so it was removed from the top-level block. When `os.environ` was later needed in `gap_mine_main`, the `__import__` workaround was used to avoid re-adding the import. This:

1. Violates the project's "explicit over clever" engineering principle (CLAUDE.md).
2. Bypasses `mypy`/IDE analysis (the import is invisible to static tools).
3. Is confusing to future contributors.

Ruff will **not** flag `import os` as unused (`F401`) once `os.environ` is used in the same module — the import is only flagged when it is unreferenced.

**Fix:**

Add `import os` to the top-level import block and use it directly:

```python
# In the top-level imports (after line 24, alphabetically):
import os

# In gap_mine_main (line 800), replace:
demand_url = __import__("os").environ.get("DEMAND_DATABASE_URL", None)
# With:
demand_url = os.environ.get("DEMAND_DATABASE_URL", None)
```

---

## Info

### IN-01: Redundant local import of `build_seed_queries` inside `gap_to_seed_query`

**File:** `scripts/coverage_agent.py:465`

**Issue:** `build_seed_queries` is already imported at module level (line 37):

```python
from scripts.ingest_places_sf import CUISINES, NEIGHBORHOODS, build_seed_queries
```

The function body re-imports it:

```python
from scripts.ingest_places_sf import build_seed_queries  # line 465
```

The accompanying comment "importing build_seed_queries at module level on every startup" is factually wrong — it **is** already imported at module level. The local re-import is harmless (Python's module cache handles it as a dict lookup), but the misleading comment could confuse future readers into thinking the lazy import is avoiding some cost.

**Fix:** Remove lines 464-465 (the local import and its comment). The already-in-scope `build_seed_queries` is used correctly at line 467.

---

### IN-02: Dead code path in integration test pair selection

**File:** `tests/integration/test_gap_miner.py:158-162`

**Issue:** The pair-selection loop (lines 155-169) calls `gap_to_seed_query(n, c)` inside the loop body, then immediately checks:

```python
seed = gap_to_seed_query(n, c)
if seed not in _CATALOG_SEEDS:
    continue
```

`gap_to_seed_query` already asserts catalog membership on both axes (lines 457-460 in `coverage_agent.py`) and raises `AssertionError` for any off-catalog input. Since `n` and `c` come directly from the catalog lists `NEIGHBORHOODS` and `CUISINES`, `gap_to_seed_query` will never raise here, and the produced seed will always be in `_CATALOG_SEEDS`. The `continue` branch is unreachable. The comment on line 160 correctly identifies this as defensive, but the dead branch adds confusion.

**Fix:** Remove the four lines (158-162). The `gap_to_seed_query` contract guarantees the seed is always catalog-valid when called with catalog inputs.

---

### IN-03: `_lexical_cuisines` and `_lexical_neighborhoods` use unanchored substring matching

**File:** `scripts/coverage_agent.py:213`, `232`

**Issue:** Both functions test `c in lower_msg` / `n.lower() in lower_msg` without word-boundary anchors. This means `"thai"` matches in `"thailand"`, `"taco"` matches in `"tacos"`, `"persian"` matches in `"Persian Gulf food"`, and `"indian"` matches in any occurrence of the letter sequence. In the restaurant-app context this rarely matters (users are asking about cuisines), and the LLM and catalog-membership filter are both downstream safety nets. However, the matching is noisier than necessary and could inflate demand counts for popular short tokens like `"bbq"` or `"taco"`.

**Fix (optional):** Use `re.search(r'\b' + re.escape(c) + r'\b', lower_msg)` for proper word-boundary matching. The catalogs have no items that are valid substrings of each other (verified), so the behavior change would only affect fringe cases. Given the downstream LLM filter already handles ambiguity, this is low priority. If left unaddressed, a comment explaining the design choice prevents future churn.

---

### IN-04: `gap_mines_main` cold-start and no-gap paths skip the sandbox write guard

**File:** `scripts/coverage_agent.py:806-819`, `828-841`

**Issue:** `assert_sandbox_write_target` is called only inside the `with get_conn() as write_conn:` block (line 864), which is reached only when there are gaps to insert. On the cold-start exit (line 819) and the "all pairs meet supply floor" exit (line 841), the guard is never invoked. This is **correct by design** — no writes occur on those paths so the guard is unnecessary — but an operator who runs `make gap-mine` and sees it complete successfully on a prod-pointed `DATABASE_URL` may incorrectly believe "the guard passed." In reality, the guard simply wasn't reached.

**Fix (optional):** Add a log line on cold-start: `_log.info("cold-start: no demand — write guard not evaluated (no writes will occur)")`. This makes the no-guard-fire case explicit rather than silent. Alternatively, invoke the guard eagerly at the top of `gap_mine_main` unconditionally (before reading demand) to fail fast on misconfigured `DATABASE_URL` even on cold-start runs.

---

_Reviewed: 2026-06-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
