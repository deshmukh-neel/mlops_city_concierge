---
phase: 03-eval-harness-extension
fixed_at: 2026-05-22T18:00:00Z
review_path: .planning/phases/03-eval-harness-extension/03-REVIEW.md
iteration: 1
findings_in_scope: 11
fixed: 11
skipped: 0
status: all_fixed
---

# Phase 3: Code Review Fix Report

**Fixed at:** 2026-05-22T18:00:00Z
**Source review:** `.planning/phases/03-eval-harness-extension/03-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 11 (1 Critical, 6 Warning, 4 Info)
- Fixed: 11
- Skipped: 0

**Validation evidence:**
- `make lint` (ruff check): **All checks passed**
- `make typecheck` (mypy app/): **Success: no issues found in 34 source files**
- Full test suite (`pytest tests/`): **736 passed, 49 skipped, 0 failed**

## Fixed Issues

### WR-01: Re-introduced `sys.path.insert` in `scripts/eval_agent.py`

**Files modified:** `scripts/eval_agent.py`
**Commit:** `586288a`
**Applied fix:** Deleted the 3-line `REPO_ROOT` / `sys.path.insert` bootstrap and removed every `# noqa: E402` marker downstream. `app` is poetry editable-installed (`packages = [{ include = "app" }]`) so `from app... import ...` already works from any cwd. Verified with `poetry run python scripts/eval_agent.py --help` from the worktree root.

### WR-02: `write_summary_json` is dead code in `scripts/eval_matrix.py`

**Files modified:** `scripts/eval_matrix.py`
**Commit:** `6b787c1`
**Applied fix:** Deleted the entire `write_summary_json(output_dir)` function. Confirmed zero call sites via `grep -rn write_summary_json` before removal. `main()` already inlines an equivalent writer that correctly threads `llm_provider_override` + `failures`.

### WR-03: `child_env = os.environ.copy()` outside loop contradicts comment

**Files modified:** `scripts/eval_matrix.py`
**Commit:** `f74e587`
**Applied fix:** Option B per REVIEW.md — updated the comment to match the actual behavior ("subprocess.run snapshots `env` for the child process, so a single os.environ.copy() shared across cells is safe"). Code unchanged; narrative now matches.

### WR-04: `--llm-provider-override foo--bar` crashes with `ValidationError`

**Files modified:** `scripts/eval_matrix.py`, `tests/unit/test_eval_matrix.py`
**Commit:** `b9fbd14`
**Applied fix:** Added `_validate_override` argparse `type=` callback that rejects `--` in the override value with an `argparse.ArgumentTypeError`. Added two unit tests: a positive case asserting `parse_args` raises `SystemExit(code=2)` on `foo--bar`, and a negative-control case asserting alphanumeric/single-dash names still work.

### WR-05: `evaluate_multi_turn_case` mutates prior turn's `state.scratch` on failure

**Files modified:** `scripts/eval_agent.py`
**Commit:** `51c3482`
**Applied fix:** Replaced `partial_state = state if state is not None else ...` with `state.model_copy(deep=True) if state is not None else ...` so the synthetic `multi_turn_runner` tool-error is recorded on a deep copy. The bug is latent today (state isn't reused post-return) but the pattern would surface as soon as any debug hook kept per-turn snapshots. Existing 5 multi-turn tests still pass.

### WR-06: `_eval_context_for(case)` computed but used only for turn 0

**Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`
**Commit:** `f11813d`
**Applied fix:** Defensive option (A): explicitly strip any prior SystemMessage from `state.messages` and re-inject a fresh `SystemMessage(eval_context)` on every turn >= 1. Tightened `test_evaluate_multi_turn_threads_messages` to assert the substring `"Expected open time:"` is present on turn 2's SystemMessage — locking the invariant against a future `add_messages` refactor that might strip system messages.

### IN-01: `validate_args` only validates `max_steps`

**Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`
**Commit:** `1ac998c`
**Applied fix:** Extended `validate_args` to check `max_queries >= 1` (when provided) and `0.0 <= temperature <= 2.0`. Added 4 new unit tests covering positive/negative/boundary cases (`-0.1`, `2.1`, `5.0` reject; `0.0`, `1.0`, `2.0` accept; `None` max_queries accepted; `0` max_queries rejected). Updated the existing `max_steps` test to pass the new required attrs on the Namespace.

### IN-02: Field-validator `info` parameter is untyped

**Files modified:** `app/eval/config.py`
**Commits:** `a83de83` (annotation), `c49bfce` (mypy follow-up)
**Applied fix:** Imported `ValidationInfo` from pydantic and annotated `info: ValidationInfo` on `EvalQuery.strip_required_text`, `MatrixEntry.strip_required_text`, and `MatrixEntry.reject_double_dash`. The first commit surfaced two mypy errors in `strip_required_text` (since `info.field_name` is typed as `str | None` in pydantic but `strip_non_empty` requires `str`); the follow-up commit adds an `or "field"` fallback to satisfy the type contract.

### IN-03: `scripted: list[AIMessage] = []` is a class-level mutable default

**Files modified:** `app/llm_factory.py`, `tests/unit/test_llm_factory.py`
**Commit:** `d61da5d`
**Applied fix:** Imported `Field` from pydantic and changed to `Field(default_factory=list)`. Added `test_scripted_chat_model_default_scripted_list_is_per_instance` that asserts `a.scripted is not b.scripted` for two fresh instances. Confirmed not a bug today (Pydantic v2 deep-copies the default) — this is a convention/durability fix.

### IN-04: `[skip-baseline]` bypass token uses raw substring match

**Files modified:** `scripts/check_baselines_fresh.py`, `tests/unit/test_check_baselines_fresh.py`
**Commit:** `170ab27`
**Applied fix:** Imported `re` and added `_SKIP_BASELINE_RE = re.compile(r"(^|\n)\s*\[skip-baseline\](\s|$)")`. Replaced the `SKIP_BASELINE_TOKEN in commit_msg` substring check with `bool(_SKIP_BASELINE_RE.search(commit_msg))`. Updated the existing `test_skip_baseline_bypass_passes` to put the token on its own line (trailer-style). Added `test_skip_baseline_at_subject_line_start_bypasses` (positive) and `test_incidental_skip_baseline_mention_does_not_bypass` (negative, the "docs PR explaining [skip-baseline]" trap that the prior substring check would silently let through).

### CR-01: Baseline JSON stubs ship with `null`; lint doesn't validate contents

**Files modified:** `tests/unit/test_baselines_are_populated.py` (new)
**Commit:** `123716e`
**Applied fix:** Option (b) per REVIEW.md — added a new test file that loads every `configs/eval_baselines/*.json` and asserts `generated_at` is non-null plus every `{median, min, max}` triple under `providers.{provider/model}.scorers.{scorer}` is non-null. **Gated on `BASELINES_POPULATED=1` env var** via a module-level `pytestmark = pytest.mark.skipif(...)` so the test job stays green during Phase 3 (the baselines ship as intentional PENDING_USER_RUN stubs). Verified the gate fires correctly when activated: running `BASELINES_POPULATED=1 pytest tests/unit/test_baselines_are_populated.py` produces 6 failures (3 generated_at + 3 scorer-stats) and 1 pass (the file-existence sanity test) — exactly the contract the REVIEW.md fix called for. Baseline JSON contents intentionally unchanged (populating requires a ~15-min live API matrix run; see VERIFICATION.md handoff section). The skip-reason embedded in the test docstring tells a future operator exactly how to flip the gate to hard.

## Skipped Issues

None — all 11 findings fixed.

---

_Fixed: 2026-05-22T18:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
