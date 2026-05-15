---
phase: 01-knowledge-graph-layer-place-relations-real-kg-traverse
plan: B
slug: tool-agent-mlflow
type: execute
wave: 2
depends_on: [A]
files_modified:
  - app/tools/graph.py
  - app/agent/tools.py
  - app/agent/prompts.py
  - scripts/log_model_to_mlflow.py
  - tests/unit/test_kg_traverse.py
  - tests/unit/test_kg_traverse_smoke.py
  - tests/unit/test_kg_traverse_functional.py
  - tests/unit/test_agent_prompts.py
  - tests/unit/test_mlflow_logging.py
  - implementation_plan/james/README.md
  - implementation_plan/james/w7_knowledge_graph.md
autonomous: true
requirements: [TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05, TOOL-06, MLOPS-01, TEST-01]
must_haves:
  truths:
    - "`from app.tools.graph import kg_traverse, RelatedPlace` succeeds."
    - "`kg_traverse(place_id, 'NEAR', k=5)` returns RelatedPlace list ordered ascending by weight."
    - "`kg_traverse(place_id, 'SIMILAR_VECTOR', k=5)` returns RelatedPlace list ordered descending by weight."
    - "`kg_traverse(place_id, 'BOGUS')` raises ValueError."
    - "The SQL JOIN uses `_view_name()` (resolves to place_documents or place_documents_v2 per settings)."
    - "Rows in place_relations whose dst_place_id is not in the active embeddings view are silently dropped by the JOIN."
    - "`app/agent/tools.py` _TOOLS entry for kg_traverse delegates to `app.tools.graph.kg_traverse` (no stub return)."
    - "Agent SYSTEM_PROMPT contains a paragraph naming all five relation_type values and when to pick each."
    - "`scripts/log_model_to_mlflow.py` logs `kg_enabled: bool` in its params dict."
  artifacts:
    - path: "app/tools/graph.py"
      provides: "RelatedPlace + kg_traverse(place_id, relation_type, k)"
      exports: ["RelatedPlace", "kg_traverse", "VALID_RELATIONS"]
      min_lines: 40
    - path: "tests/unit/test_kg_traverse.py"
      provides: "TOOL-01..TOOL-04 unit coverage with fake-cursor"
      contains: "FakeCursor"
    - path: "tests/unit/test_kg_traverse_smoke.py"
      provides: "import + RelatedPlace constructs + tool registered in _TOOLS"
    - path: "tests/unit/test_kg_traverse_functional.py"
      provides: "Multi-relation end-to-end NEAR + SIMILAR_VECTOR ordering via fake-cursor"
  key_links:
    - from: "app/tools/graph.py"
      to: "app.tools.retrieval._view_name + _execute"
      via: "import + f-string view"
      pattern: "from app.tools.retrieval import .*_view_name"
    - from: "app/agent/tools.py kg_traverse"
      to: "app.tools.graph.kg_traverse"
      via: "thin wrapper, lazy import inside function body"
      pattern: "from app.tools.graph import kg_traverse"
    - from: "scripts/log_model_to_mlflow.py params dict"
      to: "MLflow run params"
      via: "mlflow.log_params"
      pattern: "kg_enabled"
---

<objective>
Ship the tool/agent/MLflow layer of the W7 knowledge graph: create `app/tools/graph.py` with the view-aware `kg_traverse` and `RelatedPlace`, replace the W2 stub in `app/agent/tools.py`, extend the agent SYSTEM_PROMPT with relation_type guidance, add `kg_enabled` to MLflow params, and ship the unit + smoke + functional test layers (the four-layer convention the project requires beyond the spec's unit + integration).

Purpose: Plan A delivered the data; Plan B exposes it to the agent and tracks it in MLflow. This is the layer the user sees end-to-end through `/chat`.

Output: Real `kg_traverse` callable from the LangGraph agent, agent prompt updated, MLflow runs include `kg_enabled`, three new unit-test files green plus two existing files extended.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-RESEARCH.md
@.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-CONTEXT.md
@.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-PLAN-A-db-and-builder.md
@.planning/REQUIREMENTS.md
@CLAUDE.md
@implementation_plan/james/w7_knowledge_graph.md
@app/tools/retrieval.py
@app/agent/tools.py
@app/agent/prompts.py
@scripts/log_model_to_mlflow.py
@tests/unit/test_tools_retrieval.py
@tests/unit/test_agent_prompts.py
@tests/unit/test_mlflow_logging.py

<interfaces>
<!-- From app/tools/retrieval.py — what graph.py imports: -->
class PlaceHit(BaseModel):
    place_id: str
    name: str
    primary_type: str | None = None
    formatted_address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    rating: float | None = None
    price_level: str | None = None
    business_status: str | None = None
    source: str
    similarity: float
    snippet: str | None = None

def _view_name() -> str:  # returns "place_documents" or "place_documents_v2"
def _execute(sql: str, params: list) -> list[dict]:  # RealDictCursor; returns list of dicts

<!-- From app/agent/tools.py — the _TOOLS list registration we replace at line 122
     and the _args_schema_for requirement: every param MUST have a type hint or
     it raises TypeError at module import. -->

<!-- From scripts/log_model_to_mlflow.py around lines 194-201, params dict shape: -->
params: dict[str, str | int | float | bool] = {
    "llm_provider": ..., "chat_model": ..., "k": ...,
    "temperature": ..., "embedding_model": ..., "vector_store": "pgvector",
    # NEW: "kg_enabled": True,
}
</interfaces>
</context>

## Requirement Coverage

| Requirement | Task |
|-------------|------|
| TOOL-01 (kg_traverse joins view, drops missing dst) | B1, B4 |
| TOOL-02 (RelatedPlace shape) | B1, B4 |
| TOOL-03 (NEAR asc, SIMILAR_VECTOR desc ordering) | B1, B4 |
| TOOL-04 (ValueError on unknown relation_type) | B1, B4 |
| TOOL-05 (agent tools.py stub replaced) | B2, B5 |
| TOOL-06 (SYSTEM_PROMPT relation_type guidance) | B3, B6 |
| MLOPS-01 (kg_enabled in MLflow params) | B7 |
| TEST-01 (unit suite for kg_traverse) | B4, B5 |

## Locked Decisions (from planning brief — surfaced for verifier)

1. **`kg_traverse` JOIN uses `_view_name()` from `app/tools/retrieval.py`** — NOT hard-coded `place_documents`. Justification: the project supports v1/v2 embedding view switching (`app/tools/retrieval.py:20-23, 56-60`); hard-coding breaks v2-mode. Spec drift acknowledged.
2. **Construct `RelatedPlace(**row)` directly** — NOT `RelatedPlace(**_row_to_hit(row))`. The `_row_to_hit` helper does not exist in `app/tools/retrieval.py` (verified by grep in RESEARCH.md). `_execute` already returns `list[dict]` from `RealDictCursor`, so `**row` works directly and matches the `semantic_search` pattern (`app/tools/retrieval.py:104`).
3. **Four-layer test convention** — unit + smoke + functional (this plan) + integration (Plan A). The W7 spec only specifies unit + integration; the project's `.planning/codebase/TESTING.md` requires all four for new modules.

<tasks>

<task type="auto" tdd="true">
  <name>Task B1: Create app/tools/graph.py (RelatedPlace + kg_traverse)</name>
  <files>app/tools/graph.py</files>
  <behavior>
    - Importable: `from app.tools.graph import RelatedPlace, kg_traverse, VALID_RELATIONS`.
    - `RelatedPlace` extends `PlaceHit` with three new fields: `relation_type: str`, `weight: Optional[float] = None`, `relation_metadata: dict = {}`.
    - `VALID_RELATIONS = {"NEAR", "SAME_NEIGHBORHOOD", "CONTAINED_IN", "NEAR_LANDMARK", "SIMILAR_VECTOR"}`.
    - `kg_traverse(place_id: str, relation_type: str = "SIMILAR_VECTOR", k: int = 5) -> list[RelatedPlace]`:
      - Raises `ValueError(f"Unknown relation_type: {relation_type}")` when `relation_type not in VALID_RELATIONS`.
      - Builds SQL JOINing `place_relations r` to `{view} pd` where `view = _view_name()`. Selects PlaceHit columns from `pd`, plus `0.0 AS similarity`, `LEFT(pd.embedding_text, 400) AS snippet`, `r.relation_type`, `r.weight`, `r.metadata AS relation_metadata` (alias so dict key matches the Pydantic field name).
      - `WHERE r.src_place_id = %s AND r.relation_type = %s`.
      - `ORDER BY CASE r.relation_type WHEN 'NEAR' THEN r.weight WHEN 'SIMILAR_VECTOR' THEN -r.weight ELSE 0 END`.
      - `LIMIT %s`.
      - Calls `_execute(sql, [place_id, relation_type, k])` and returns `[RelatedPlace(**row) for row in rows]`.
      - SQL uses f-string for `{view}` (safe — `_view_name()` returns an allowlist member) with `# noqa: S608` comment justifying the suppression.
  </behavior>
  <action>
    Create `app/tools/graph.py` per the recommended shape in RESEARCH.md "Code Examples" (`app/tools/graph.py (recommended shape — view-aware)`). Imports from `app.tools.retrieval`: `PlaceHit, _execute, _view_name`. Use `from __future__ import annotations`. Add module docstring noting "Graph-traversal tool. Returns related places by relation_type. JOINs place_relations to the active embeddings view via _view_name() so v1/v2 toggle is honored."

    Type-hint every parameter — `_args_schema_for` in `app/agent/tools.py:99-102` raises on missing annotations. The `# noqa: S608` comment must include a justification like `# noqa: S608  # view is an allowlist member from _view_name()`.
  </action>
  <verify>
    <automated>python -c "from app.tools.graph import RelatedPlace, kg_traverse, VALID_RELATIONS; assert VALID_RELATIONS == {'NEAR','SAME_NEIGHBORHOOD','CONTAINED_IN','NEAR_LANDMARK','SIMILAR_VECTOR'}; assert 'relation_type' in RelatedPlace.model_fields"</automated>
  </verify>
  <done>
    File exists, imports succeed, `VALID_RELATIONS` matches spec, `RelatedPlace` has the three new fields with correct defaults. Pre-commit ruff passes.
  </done>
</task>

<task type="auto">
  <name>Task B2: Replace W2 kg_traverse stub in app/agent/tools.py</name>
  <files>app/agent/tools.py</files>
  <action>
    Replace the stub function at `app/agent/tools.py:76-85` with the thin wrapper from RESEARCH.md "Replacement at `app/agent/tools.py:76-85`":

    - Signature: `def kg_traverse(place_id: str, relation_type: str = "SIMILAR_VECTOR", k: int = 5) -> list[RelatedPlace]:` (every param annotated — `_args_schema_for` requires it).
    - Docstring explains when to use each relation_type — same content as the prompt addition (Task B3), kept short. Mention: SIMILAR_VECTOR for "more like this"; SAME_NEIGHBORHOOD for "same area"; NEAR_LANDMARK for "near landmark X"; NEAR for "geographic neighbors without re-running nearby"; CONTAINED_IN for "inside this place" (rare).
    - Body: lazy-import inside function: `from app.tools.graph import kg_traverse as _kg_traverse`; return `_kg_traverse(place_id=place_id, relation_type=relation_type, k=k)`. Lazy import avoids a circular import risk if graph.py later imports anything from agent.

    Add `RelatedPlace` to the imports block at the top of the file alongside `PlaceHit` and `PlaceDetails`:

      from app.tools.graph import RelatedPlace

    (Top-level import is fine for the type; only the function uses lazy.)

    The `_TOOLS` entry at line 122 stays the same — `_to_lc_tool("kg_traverse", kg_traverse.__doc__ or "", kg_traverse)` — only the underlying function changes, so the registration line needs no edit.
  </action>
  <verify>
    <automated>python -c "from app.agent.tools import all_tools, kg_traverse; t = [x for x in all_tools() if x.name == 'kg_traverse'][0]; assert 'NOT YET AVAILABLE' not in (t.description or ''); import inspect; assert 'relation_type' in inspect.signature(kg_traverse).parameters"</automated>
  </verify>
  <done>
    Stub gone (no "NOT YET AVAILABLE" or "available: False" in the file). Wrapper present. `all_tools()` still returns 5 tools including kg_traverse. Pre-commit passes.
  </done>
</task>

<task type="auto">
  <name>Task B3: Extend SYSTEM_PROMPT with relation_type guidance</name>
  <files>app/agent/prompts.py</files>
  <action>
    In `app/agent/prompts.py`, the SYSTEM_PROMPT already forward-references `kg_traverse(stop_K, relation_type='NEAR')` at line 85 inside the WALKING BUDGET section. Add a new dedicated paragraph (numbered item 9, before the OUTPUT FORMAT section at line 98) titled "KNOWLEDGE GRAPH (kg_traverse)". The paragraph must:

    - Enumerate all five relation_type values and one-line guidance for each:
      - `SIMILAR_VECTOR`: "more like this" — same vibe/category as an anchor.
      - `SAME_NEIGHBORHOOD`: alternates in the same SF neighborhood.
      - `NEAR`: geographic neighbors within ~800m — cheaper than calling `nearby` again.
      - `NEAR_LANDMARK`: anchor near a known landmark (museum, park, etc.).
      - `CONTAINED_IN`: parent venue (e.g., a stall inside a food hall) — rare.
    - Note that `kg_traverse` is single-hop; for multi-hop reasoning, the agent should call it again with the new anchor.
    - Note that if `kg_traverse` returns empty, fall back to `semantic_search` or `nearby`.

    Do NOT redesign the surrounding prompt. Do NOT remove the forward reference at line 85. The new paragraph fills it in.
  </action>
  <verify>
    <automated>python -c "from app.agent.prompts import SYSTEM_PROMPT; s = SYSTEM_PROMPT.lower(); assert all(k in s for k in ['similar_vector','same_neighborhood','near_landmark','contained_in','kg_traverse'])"</automated>
  </verify>
  <done>
    All five relation_type values appear in SYSTEM_PROMPT. Existing forward reference at line 85 still present. Pre-commit passes.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task B4: Unit tests for kg_traverse (TOOL-01..TOOL-04, TEST-01)</name>
  <files>tests/unit/test_kg_traverse.py</files>
  <behavior>
    Five test cases using the fake-cursor pattern from `tests/unit/test_tools_retrieval.py:47-91` (the cleaner parent over `test_retriever.py`):

    - `test_invalid_relation_raises`: `kg_traverse('p1', 'BOGUS')` raises `ValueError` containing "Unknown relation_type". (TOOL-04)
    - `test_related_place_shape`: with a fake cursor returning one row with all PlaceHit fields + `relation_type='NEAR'`, `weight=42.0`, `relation_metadata={'k':'v'}`, the returned `RelatedPlace` exposes those three new fields. (TOOL-02)
    - `test_ordering_clause_near_ascending`: inspect the SQL string captured by `FakeCursor.executed_sql`; assert it contains `CASE r.relation_type WHEN 'NEAR' THEN r.weight` (asc-by-weight clause). (TOOL-03)
    - `test_ordering_clause_similar_vector_descending`: assert the captured SQL contains `WHEN 'SIMILAR_VECTOR' THEN -r.weight`. (TOOL-03)
    - `test_join_drops_missing_dst`: when the fake cursor returns zero rows (simulating dst_place_ids missing from place_documents and therefore dropped by the JOIN), `kg_traverse` returns `[]` (not an error). Verify the SQL uses `JOIN`, not `LEFT JOIN`, so missing dst is silently dropped. (TOOL-01)
    - `test_view_name_resolved`: with settings.embedding_table = 'place_embeddings_v2', assert the captured SQL contains `place_documents_v2` (NOT hard-coded `place_documents`). Use `monkeypatch.setenv` or directly patch `app.tools.graph._view_name`. (TOOL-01 + locked decision #1)
  </behavior>
  <action>
    Build `FakeCursor` and `FakeConnection` classes implementing `__enter__/__exit__/execute/fetchall` modeled on `tests/unit/test_tools_retrieval.py`. The `FakeCursor.execute(sql, params)` stores `self.executed_sql = sql` and `self.executed_params = list(params)`. `FakeCursor.fetchall()` returns the dict rows pre-loaded onto the fixture. Patch `app.tools.graph.get_conn` (imported into graph.py via `_execute`'s call chain) — note that `_execute` actually uses `app.db.get_conn`; patch at the module where it's resolved, which is `app.tools.retrieval.get_conn` (since `_execute` is defined there). Read `tests/unit/test_tools_retrieval.py:47-91` for the exact patch target.

    Use `asyncio_mode = "auto"` is already set; no `@pytest.mark.asyncio`. Each test is a plain `def test_*`.

    Make rows match the `RealDictCursor` shape (dicts with all PlaceHit columns + relation_type + weight + relation_metadata). Use minimal valid PlaceHit dummy data: `{"place_id":"p2","name":"Test","source":"test","similarity":0.0, "primary_type":None, "formatted_address":None, "latitude":None, "longitude":None, "rating":None, "price_level":None, "business_status":None, "snippet":None, "relation_type":"NEAR", "weight":42.0, "relation_metadata":{}}`.
  </action>
  <verify>
    <automated>pytest tests/unit/test_kg_traverse.py -v --no-cov 2>&1 | tail -20</automated>
  </verify>
  <done>
    All 6 test cases pass. Pre-commit ruff passes.
  </done>
</task>

<task type="auto">
  <name>Task B5: Smoke + functional tests (project four-layer convention)</name>
  <files>tests/unit/test_kg_traverse_smoke.py, tests/unit/test_kg_traverse_functional.py</files>
  <action>
    Add the two missing test layers per project convention (RESEARCH.md A5 + `.planning/codebase/TESTING.md:58-63`).

    **`tests/unit/test_kg_traverse_smoke.py`** — pure import + construction, no DB, no patching:
    - `test_imports_ok`: `from app.tools.graph import kg_traverse, RelatedPlace, VALID_RELATIONS` succeeds.
    - `test_related_place_constructs`: `RelatedPlace(place_id='p', name='n', source='test', similarity=0.0, relation_type='NEAR', weight=1.0)` constructs without error and the three new fields are accessible.
    - `test_kg_traverse_registered_in_tools`: `from app.agent.tools import all_tools; assert any(t.name == 'kg_traverse' for t in all_tools())` and the tool's description does NOT contain "NOT YET AVAILABLE" or "available: False" (sanity check the stub is gone). (TOOL-05)
    - `test_kg_traverse_args_schema_has_k_typed`: pull the StructuredTool, inspect `args_schema`, assert it has fields `place_id`, `relation_type`, `k` with correct types.

    **`tests/unit/test_kg_traverse_functional.py`** — multi-relation happy path via fake-cursor; one flow that exercises both NEAR ASC ordering and SIMILAR_VECTOR DESC ordering in a single test using the same FakeCursor class (lift it into a shared `conftest.py` if you want to share between this file and `test_kg_traverse.py`, OR duplicate locally — both acceptable; the project hasn't established a strong preference here, and the duplicating-locally path is simpler).
    - `test_near_returns_places_ordered_by_weight_asc`: fake-cursor returns three rows with weights [100, 50, 200]; the SQL ORDER BY clause is verified to contain the asc-for-NEAR case; the test asserts `kg_traverse(..., 'NEAR', k=3)` returns the three rows in the order the cursor provided (the fake cursor doesn't actually sort — we only assert that the SQL string contains the correct ORDER BY clause AND that the rows pass through unchanged). The "ordering correctness" is verified by SQL inspection, not by post-hoc sort.
    - `test_similar_vector_returns_places_ordered_desc_in_sql`: same shape, but for SIMILAR_VECTOR, and assert the SQL contains the desc-for-SIMILAR_VECTOR case.
    - `test_metadata_preserved_through_pipeline`: fake row has `relation_metadata={'displayName':'Ferry Building','types':['landmark']}`; assert the returned RelatedPlace's `relation_metadata` matches exactly (jsonb round-trip integrity).

    Together these three files (`test_kg_traverse.py`, `test_kg_traverse_smoke.py`, `test_kg_traverse_functional.py`) satisfy TEST-01.
  </action>
  <verify>
    <automated>pytest tests/unit/test_kg_traverse_smoke.py tests/unit/test_kg_traverse_functional.py -v --no-cov 2>&1 | tail -20</automated>
  </verify>
  <done>
    Both files pass all test cases (4 smoke + 3 functional). Pre-commit ruff passes.
  </done>
</task>

<task type="auto">
  <name>Task B6: Extend test_agent_prompts.py (TOOL-06)</name>
  <files>tests/unit/test_agent_prompts.py</files>
  <action>
    Read the existing `tests/unit/test_agent_prompts.py` to learn its existing assertion style. Add ONE new test (or extend an existing test) that asserts the SYSTEM_PROMPT contains relation_type guidance:

    - `test_system_prompt_contains_relation_type_guidance`: load SYSTEM_PROMPT, lowercase it, assert it mentions all five relation_type values (`similar_vector`, `same_neighborhood`, `near_landmark`, `contained_in`, and the word `kg_traverse`). Use one assertion per term so failures are precise.

    Do NOT rewrite or restructure the existing tests in this file.
  </action>
  <verify>
    <automated>pytest tests/unit/test_agent_prompts.py -v --no-cov 2>&1 | tail -10</automated>
  </verify>
  <done>
    All existing tests still pass. New assertion present and green. Pre-commit ruff passes.
  </done>
</task>

<task type="auto">
  <name>Task B7: MLflow kg_enabled param + extend test_mlflow_logging.py (MLOPS-01)</name>
  <files>scripts/log_model_to_mlflow.py, tests/unit/test_mlflow_logging.py</files>
  <action>
    Two edits in one commit (atomic from the user-feature perspective; the script change without the test would be incomplete):

    1. **`scripts/log_model_to_mlflow.py`** around lines 194-201, add `"kg_enabled": True` to the `params` dict (default True after this PR ships; W6 evals will A/B with False). Widen the dict's type annotation to include `bool` if not already (e.g. `dict[str, str | int | float | bool]`). Add a short inline comment: `# W7: KG on by default; W6 evals will A/B with False`.

    2. **`tests/unit/test_mlflow_logging.py`** — read the existing file to understand its existing assertion style (it likely patches `mlflow.log_params` or asserts on a built dict). Add one assertion that the params dict produced by the relevant function contains `"kg_enabled"` and that its value is a bool. Do not redesign existing tests.
  </action>
  <verify>
    <automated>grep -q '"kg_enabled"' scripts/log_model_to_mlflow.py && pytest tests/unit/test_mlflow_logging.py -v --no-cov 2>&1 | tail -10</automated>
  </verify>
  <done>
    Script grep finds `"kg_enabled"`. Test file has an explicit `kg_enabled` assertion. `pytest tests/unit/test_mlflow_logging.py` passes. Pre-commit ruff passes.
  </done>
</task>

<task type="auto">
  <name>Task B8: Update implementation_plan tracking (CLAUDE.md convention)</name>
  <files>implementation_plan/james/README.md, implementation_plan/james/w7_knowledge_graph.md</files>
  <action>
    Per CLAUDE.md "Implementation plan tracking" — this task is only run when the Phase 1 PR is merged. Document it now in PLAN.md so the executor (or the user, post-merge) knows to do both:

    1. Update the W7 row in `implementation_plan/james/README.md` status table from its current state to "✅ Merged with #<PR-number>". Replace `🚧 In progress` if present.
    2. Add a `**Status:**` footer to the bottom of `implementation_plan/james/w7_knowledge_graph.md`: a one-line stamp like `**Status:** Merged 2026-05-XX in #<PR-number>. Deferred: LLM-extracted edges, editorial-source edges, per-source NEAR cap (see DEFERRED ideas in 01-CONTEXT.md).`

    Per CLAUDE.md: "Don't update mid-workstream — only on merge." This task is a placeholder reminder; the executor should NOT run it until the PR is actually merged, at which point it becomes a 2-line commit. If the executor reaches this task before merge, **skip it and surface the skip in the SUMMARY** with a note for the user to do post-merge.
  </action>
  <verify>
    <automated>grep -q "W7" implementation_plan/james/README.md && grep -q "Status:" implementation_plan/james/w7_knowledge_graph.md || echo "DEFERRED — run after PR merge per CLAUDE.md convention"</automated>
  </verify>
  <done>
    Either: (a) README row + w7 footer updated post-merge, OR (b) explicitly deferred and noted in SUMMARY for the user.
  </done>
</task>

</tasks>

<verification>
- `make test-unit` is green (existing tests + 3 new + 2 extended).
- `make lint` is green.
- `python -c "from app.agent.tools import all_tools; [print(t.name) for t in all_tools()]"` lists 5 tools including `kg_traverse` with no stub language in its description.
- End-to-end smoke: with a populated `place_relations` table (from Plan A), `python -c "from app.tools.graph import kg_traverse; print(kg_traverse('ChIJExYUW8Z_j4AREJB4F5tJJto', 'NEAR', k=5))"` returns ordered RelatedPlace results.
- An MLflow run via `make log-mlflow` shows `kg_enabled: True` in the params tab.
</verification>

<success_criteria>
- All 18 phase requirements (KG-01..TEST-02) have a green test path.
- Agent SYSTEM_PROMPT covers all five relation_type values.
- `kg_traverse` is real (no stub) and view-aware (honors v1/v2 toggle).
- Unit + smoke + functional test layers present (project four-layer convention satisfied).
- MLflow eval-A/B-ready: `kg_enabled` param logged per agent run.
</success_criteria>

<output>
Each task is committed separately with a one-line message per user preference:
- B1: `feat(kg): add RelatedPlace + view-aware kg_traverse in app/tools/graph.py`
- B2: `feat(kg): replace W2 kg_traverse stub with real wrapper`
- B3: `docs(agent): add relation_type guidance to SYSTEM_PROMPT`
- B4: `test(kg): unit tests for kg_traverse (TOOL-01..04)`
- B5: `test(kg): add smoke + functional layers for kg_traverse`
- B6: `test(agent): assert SYSTEM_PROMPT mentions relation_type values`
- B7: `feat(mlflow): log kg_enabled param on agent runs`
- B8: (post-merge only) `docs(plan): mark W7 merged in implementation_plan tracker`

After tasks B1–B7 complete, create `.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-PLAN-B-SUMMARY.md` per the gsd summary template. B8 is a post-merge follow-up.
</output>
