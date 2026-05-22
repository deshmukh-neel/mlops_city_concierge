---
phase: 03-eval-harness-extension
plan: 02
subsystem: eval-schema
tags: [eval, schema, pydantic, backward-compat]
requires:
  - "app/agent/state.py:UserConstraints"
  - "app/eval/config.py:EvalQuery"
  - "app/eval/config.py:strip_non_empty"
  - "app/eval/config.py:strip_non_empty_list"
provides:
  - "app/agent/state.py:UserConstraints.requested_primary_types (list[str], default [])"
  - "app/eval/config.py:EvalQuery.turns (list[str] | None, default None)"
  - "app/eval/config.py:MatrixEntry (provider, model)"
  - "app/eval/config.py:EvalMatrixConfig (entries, scenarios)"
  - "app/eval/config.py:load_eval_matrix(path)"
  - "app/eval/config.py:DEFAULT_EVAL_MATRIX_PATH"
affects:
  - "app/agent/critique/checks.py:category_compliance (plan 03-03 consumer)"
  - "scripts/eval_agent.py (plan 03-04 multi-turn runner consumer)"
  - "scripts/eval_matrix.py (plan 03-05 matrix runner consumer)"
tech-stack:
  added: []
  patterns:
    - "Pydantic v2 field_validator + strip_non_empty / strip_non_empty_list"
    - "model_config = ConfigDict(extra='forbid')"
    - "Field(min_length=1) for required-non-empty lists"
    - "model_validator(mode='after') for cross-field uniqueness checks"
key-files:
  created: []
  modified:
    - "app/agent/state.py"
    - "app/eval/config.py"
    - "tests/unit/test_agent_state.py"
    - "tests/unit/test_eval_config.py"
decisions:
  - "Placed requested_primary_types at the BOTTOM of UserConstraints to preserve any positional-style constructors (no usages found, but cheap insurance per plan guidance)"
  - "turns field-validator rejects [] in the validator body (not via min_length) so the error message names the field — clearer than 'list too short' for YAML authors"
  - "strip_non_empty_list trims and rejects blanks for turns, mirroring tags exactly"
  - "EvalMatrixConfig.entries uniqueness keyed on (provider, model) tuple, not just provider — same provider can run multiple models"
  - "DEFAULT_EVAL_MATRIX_PATH defined here even though configs/eval_matrix.yaml is plan 03-05's deliverable; the loader needs a sensible default"
metrics:
  duration_minutes: 12
  tasks_completed: 2
  files_modified: 4
  tests_added: 21
  completed: "2026-05-21"
requirements:
  - EVAL-03
  - EVAL-04
---

# Phase 3 Plan 02: Schema Additions Summary

One-liner: Added `UserConstraints.requested_primary_types: list[str] = []` (D-01), `EvalQuery.turns: list[str] | None = None` (EVAL-03), and `EvalMatrixConfig` + `MatrixEntry` + `load_eval_matrix()` (EVAL-04) — three schema seeds that unblock plans 03-03, 03-04, and 03-05 to run in parallel.

## What Shipped

### Task 1 — UserConstraints.requested_primary_types (D-01)

Added one `Field(default_factory=list)` on `UserConstraints` at the bottom of the model. The description names D-01 explicitly and points downstream readers at `app.agent.critique.checks.category_compliance` (EVAL-01) and Phase 4's `primary_type_family` enforcement. Three new unit tests cover (a) the empty-list default, (b) explicit-value round-trip, and (c) per-instance isolation (no shared mutable default).

**Commits:**
- `20cbc5b` — RED: failing tests prove the field doesn't exist
- `b623939` — GREEN: field added, all 21 existing + 3 new tests pass

**No agent-graph wiring touched.** D-02 (intake LLM populating the field) is explicitly Phase 4 work; the field stays at `[]` in every existing call path, preserving full backward compat (OVR-03 pattern).

### Task 2 — EvalQuery.turns + EvalMatrixConfig + MatrixEntry + load_eval_matrix (EVAL-03, EVAL-04)

Four additions to `app/eval/config.py`, none of which touch `EvalQueriesConfig` itself (the loaders sit side-by-side):

1. **`EvalQuery.turns: list[str] | None = None`** — positioned between `tags` and the existing `model_validator`. A new `@field_validator("turns")` returns `None` unchanged, raises `ValueError` with a field-named message on `[]`, and reuses `strip_non_empty_list` for non-empty lists (trims whitespace, rejects blank strings).
2. **`MatrixEntry(BaseModel)`** — `provider: str`, `model: str`, both validated through `strip_non_empty` in `mode="before"` (matches EvalQuery's `strip_required_text` pattern). `extra="forbid"`.
3. **`EvalMatrixConfig(BaseModel)`** — `entries: list[MatrixEntry] = Field(min_length=1)`, `scenarios: list[str] = Field(min_length=1)`, plus a `model_validator(mode="after")` that requires unique `(provider, model)` tuples (mirrors `EvalQueriesConfig.ids_are_unique`). `scenarios` are validated through `strip_non_empty_list` (consistent with `tags`).
4. **`load_eval_matrix(path)`** — byte-for-byte mirror of `load_eval_queries`: `yaml.safe_load` → mapping check → `model_validate`. `DEFAULT_EVAL_MATRIX_PATH = Path("configs/eval_matrix.yaml")` ships here even though the YAML file itself is plan 03-05's deliverable, so downstream code can `import` the constant.

Fifteen new unit tests cover every behavior bullet in the plan: default `turns=None`, non-empty list accepted, `[]` rejected, blank strings rejected, whitespace stripped, MatrixEntry positive + negative paths, EvalMatrixConfig empty-entries/empty-scenarios/blank-scenarios/duplicate-entries/extra-keys rejection, `load_eval_matrix` positive + non-mapping-root negative, and `DEFAULT_EVAL_MATRIX_PATH` shape.

**Commits:**
- `810306c` — RED: 18 failing tests (collection-time ImportError on the new symbols)
- `aa5badd` — GREEN: schema additions, all 29 tests in the file pass

## Verification Run

```text
poetry run pytest tests/unit/test_agent_state.py tests/unit/test_eval_config.py -v
  50 passed in 0.15s

poetry run pytest tests/unit/    # full unit suite (make test-unit)
  621 passed, 9 warnings in 31.75s

poetry run mypy app/agent/state.py app/eval/config.py
  Success: no issues found in 2 source files

poetry run python -c "from app.eval.config import load_eval_queries, REPO_ROOT, DEFAULT_EVAL_QUERIES_PATH; cfg = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH); print(f'{len(cfg.hand_written)} cases, turns set on any: {any(c.turns is not None for c in cfg.hand_written)}')"
  30 cases, turns set on any: False
```

Full backward compat with the 30 hand-written cases in `configs/eval_queries.yaml`: every case carries `turns=None`, matching pre-change behavior.

## Deviations from Plan

None — plan executed exactly as written.

The plan called out two minor "deviations" in advance that I followed without amplification:

- The plan suggests `EvalQuery.turns=[]` should be REJECTED with a message naming the field. I implemented this in the `field_validator` body (not via `min_length=1` on the type) precisely so the error message says `"turns must be omitted (None) or contain at least one follow-up turn"` rather than the Pydantic-default `"List should have at least 1 item"`. Decision recorded in frontmatter.
- The plan does not specify HOW to enforce `MatrixEntry` uniqueness. I keyed it on the `(provider, model)` tuple, not just provider, because the same provider can legitimately run multiple models in a single matrix.

No Rule 1/2/3 auto-fixes were needed; nothing in the codebase relied on `EvalQuery` or `UserConstraints` field ORDER.

## Authentication Gates

None — schema-only changes, no external services touched.

## Known Stubs

None. The added fields and models are immediately consumable by plans 03-03, 03-04, 03-05. The matrix YAML file (`configs/eval_matrix.yaml`) is intentionally NOT created here per the plan — it ships in 03-05's deliverable. `DEFAULT_EVAL_MATRIX_PATH` is defined so that the loader's signature is complete, not as a stub.

## Threat Flags

None. Schema-only changes; no new network endpoints, auth paths, file-access patterns, or trust boundaries introduced. The new YAML loader (`load_eval_matrix`) inherits the same `yaml.safe_load` pattern as `load_eval_queries`, which is the existing safe-loading convention.

## TDD Gate Compliance

Both tasks followed RED → GREEN. No refactor commits were needed (the additions are pure-additive and small enough that GREEN-as-final-shape was acceptable).

| Task | RED         | GREEN       |
| ---- | ----------- | ----------- |
| 1    | `20cbc5b`   | `b623939`   |
| 2    | `810306c`   | `aa5badd`   |

## Self-Check: PASSED

- FOUND: `app/agent/state.py` — modified, line 90 has `requested_primary_types`
- FOUND: `app/eval/config.py` — modified, `EvalQuery.turns`, `MatrixEntry`, `EvalMatrixConfig`, `load_eval_matrix`, `DEFAULT_EVAL_MATRIX_PATH` all present
- FOUND: `tests/unit/test_agent_state.py` — modified, 3 new `requested_primary_types` tests
- FOUND: `tests/unit/test_eval_config.py` — modified, 18 new tests covering turns + matrix
- FOUND: commit `20cbc5b` (test: requested_primary_types RED)
- FOUND: commit `b623939` (feat: requested_primary_types GREEN)
- FOUND: commit `810306c` (test: turns + EvalMatrixConfig RED)
- FOUND: commit `aa5badd` (feat: turns + EvalMatrixConfig GREEN)
