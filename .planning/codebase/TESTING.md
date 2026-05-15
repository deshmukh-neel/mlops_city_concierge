# Testing Patterns

**Analysis Date:** 2026-05-14

## Test Framework

**Runner:**
- `pytest >=8.2,<9` (`pyproject.toml:45`)
- Config: `[tool.pytest.ini_options]` in `pyproject.toml:61-64`
  - `testpaths = ["tests"]`
  - `asyncio_mode = "auto"` — every `async def test_...` is auto-marked; do **not** add `@pytest.mark.asyncio` manually
  - `addopts = "-v --tb=short"` — verbose with short tracebacks

**Plugins:**
- `pytest-asyncio >=0.23.6` — async test support
- `pytest-cov >=5.0` — coverage
- `pytest-mock >=3.14` — `mocker` fixture (preferred over bare `unittest.mock`)
- `factory-boy >=3.3` — declared but currently unused; helpers in `tests/conftest.py` cover fixture-building today

**Assertion Library:**
- Plain `assert` statements (S101 ignored under `tests/**/*.py` via `pyproject.toml:75`)

**Run Commands (from `Makefile`):**
```bash
make test                 # pytest tests/ -v --cov=app --cov-report=term-missing
make test-unit            # pytest tests/unit/ -v
make test-integration     # pytest tests/integration/ -v  (needs APP_ENV=integration + live DB)
make test-integration-cloud  # integration suite against Cloud SQL via cloud-sql-proxy + IAM auth
pytest tests/unit/test_ingest.py -v                                   # single file
pytest tests/unit/test_ingest.py::TestIngestScript::test_loads_jsonl_records -v  # single test
```

## Test File Organization

**Location:**
- Separate from source. Tree:
  ```
  tests/
  ├── conftest.py              # shared fixtures + helpers
  ├── __init__.py
  ├── unit/                    # 32 unit test modules
  │   ├── __init__.py
  │   ├── test_<module>.py             # primary unit tests
  │   ├── test_<module>_smoke.py       # imports + minimal compile (e.g., test_agent_smoke.py)
  │   ├── test_<module>_functional.py  # multi-component, still mocked (e.g., test_chat_functional.py)
  │   └── ...
  └── integration/             # 9 integration test modules
      ├── __init__.py
      ├── test_db.py
      ├── test_agent_graph.py
      └── ...
  ```

**Naming:**
- Files mirror the module under test: `tests/unit/test_retriever.py` ↔ `app/retriever.py`; `tests/unit/test_agent_graph.py` ↔ `app/agent/graph.py`
- Test functions are `snake_case` and self-describing: `test_chat_endpoint_returns_503_when_agent_unavailable` (`tests/unit/test_chat_endpoint.py:92`); `test_get_db_closes_connection_when_reset_fails` (`tests/unit/test_db.py:39`)

**Test Layering Convention:**
New modules ship with **all four layers**, not just unit tests:
1. **Unit** with mocks — every public function (`tests/unit/test_<feature>.py`)
2. **Smoke** — module imports cleanly, public objects construct, graph compiles (`tests/unit/test_agent_smoke.py`)
3. **Functional** — multi-component happy paths, still mocked (`tests/unit/test_chat_functional.py`, `tests/unit/test_agent_self_correct_functional.py`)
4. **Integration** — exercises real Postgres / Langfuse / etc. (`tests/integration/test_<feature>.py`)

## Test Structure

**Suite Organization:**
- Most modules use **flat module-level functions**, not test classes (`tests/unit/test_chat_endpoint.py`, `tests/unit/test_retriever.py`)
- Test classes appear when grouping smoke/contract assertions: `class TestDatabaseConnection` in `tests/integration/test_db.py:25`, `class TestIngestScript` in `tests/unit/test_ingest.py`

**Module Header Conventions:**
```python
"""
Brief docstring describing what's being tested.
"""
from __future__ import annotations
import pytest
from app.<module> import <thing under test>
```

**Integration-test module header — required pattern (`tests/integration/test_db.py:19`, `tests/integration/test_agent_graph.py:20`):**
```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)
```
Stack multiple `pytest.mark.skipif` in a list when extra creds (Langfuse, OpenAI) are needed:
`tests/integration/test_observability_tracing_integration.py:21-30`.

## Mocking

**Framework:** `pytest-mock`'s `mocker` fixture (preferred). Bare `unittest.mock.MagicMock` only when constructing standalone fakes (`tests/unit/test_retriever.py:3`).

**Patterns:**

1. **Patch the symbol where it's looked up, not where it's defined** — patch `app.main.load_registered_rag_chain`, NOT `app.<source_module>.load_registered_rag_chain`:
   ```python
   mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock()))
   mocker.patch("app.main.build_agent_graph", return_value=fake_graph)
   ```
   (`tests/unit/test_chat_endpoint.py:59-62`)

2. **Hand-rolled fakes for context-manager protocols** — when mocking psycopg2 connections/cursors, build small classes implementing `__enter__`/`__exit__` rather than configuring nested MagicMocks:
   ```python
   class FakeCursor:
       def __enter__(self): return self
       def __exit__(self, *a): return False
       def execute(self, sql, params): self.executed_sql = sql; ...
       def fetchall(self): return self.rows
   ```
   (`tests/unit/test_retriever.py:10-42`)

3. **Async mocking** — define a real `async def` and assign it to the mock attribute (don't use `AsyncMock` for graph nodes):
   ```python
   fake_graph = mocker.Mock()
   async def _ainvoke(state, config=None):
       return _final_state_dict(reply="ok")
   fake_graph.ainvoke = _ainvoke
   ```
   (`tests/unit/test_chat_endpoint.py:42-58`)

4. **FastAPI endpoint testing** — always use `with TestClient(app) as client:` so `lifespan` runs (otherwise `app.state` is empty):
   ```python
   with TestClient(app) as client:
       response = client.post("/chat", json={...})
   ```
   (`tests/unit/test_chat_endpoint.py:64`)

5. **Stub LLMs as real `BaseChatModel` subclasses**, not `Mock()` — LangChain checks `isinstance` and reads `_llm_type`. See `_NoopLLM` (`tests/unit/test_agent_smoke.py:15-34`) and `_SemanticSearchOnceLLM` (`tests/integration/test_agent_graph.py:34-66`).

**What to Mock:**
- External services: OpenAI (`OpenAIEmbeddings`, `ChatOpenAI`), Gemini (`ChatGoogleGenerativeAI`), MLflow registry, Langfuse, psycopg2 connections in unit tests
- The DB pool when testing connection lifecycle: `mocker.patch("app.db.get_connection", ...)`, `mocker.patch("app.db.return_connection")` (`tests/unit/test_db.py:11-12`)

**What NOT to Mock:**
- Pydantic models — construct real instances via the `make_stop` / `make_hit` helpers in `tests/conftest.py`
- Pure functions under test (`build_database_url`, `vector_to_pg`, `compile_filters`)
- The DB / Langfuse in integration tests — that's the whole point of the integration layer

## Fixtures and Factories

**Shared `conftest.py` (`tests/conftest.py`):**
- `_patch_env` (`autouse=True`) — sets safe defaults for `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DATABASE_URL`, `APP_ENV`. Uses **setdefault semantics** (`if os.environ.get(key) is None`) so caller-supplied env wins — that's how integration tests with real `APP_ENV=integration` flow through unchanged
- Calls `get_settings.cache_clear()` before AND after each test so `lru_cache` doesn't leak settings between tests
- Plain helper functions exported for `from tests.conftest import make_stop, make_hit`:
  - `make_stop(place_id="p1", **kwargs) -> Stop`
  - `make_hit(place_id="p1", *, similarity=0.9, business_status="OPERATIONAL", **kwargs) -> PlaceHit`

**Test-data construction:**
- Inline dict factories like `_final_state_dict(...)` (`tests/unit/test_chat_endpoint.py:25`) and `_stub_loaded_config(...)` (`:10`) are co-located in the test module
- factory-boy is in dev deps but not used in current code — when adding new factories, prefer the `make_*` helper pattern in `conftest.py` for cross-module reuse

## Coverage

**Configuration:**
- `make test` runs `pytest tests/ -v --cov=app --cov-report=term-missing`
- No coverage threshold enforced in CI; goal is "err on the side of too many tests rather than too few"

**Excluded from coverage:**
- Use `# pragma: no cover - <reason>` for branches that are exercised by integration/startup tests but not unit tests:
  - `except Exception as exc:  # pragma: no cover - exercised via startup tests` (`app/main.py:133`)

## Test Types

**Unit Tests (`tests/unit/`):**
- Scope: a single function or class in isolation
- All external I/O mocked
- Should never start a real DB connection — `_patch_env` keeps `DATABASE_URL` pointed at a fake host

**Smoke Tests (`tests/unit/test_*_smoke.py`):**
- Verify imports, public class instantiation, and graph compilation
- Pattern: `@pytest.mark.parametrize("module", [...])` + `importlib.import_module(module)` (`tests/unit/test_agent_smoke.py:37-49`)
- Cheap signal that nothing is broken at the package boundary

**Functional Tests (`tests/unit/test_*_functional.py`):**
- Multi-node / multi-tool happy-path flows
- Still fully mocked but with realistic-looking data (e.g., `test_chat_functional.py`, `test_agent_self_correct_functional.py`)

**Integration Tests (`tests/integration/`):**
- Skipped unless `APP_ENV=integration`; some additionally require credential env vars (Langfuse keys)
- Real Postgres + pgvector required (`make db-up` then `APP_ENV=integration make test-integration`)
- LLMs typically still stubbed (cost / determinism) — see `_SemanticSearchOnceLLM` in `tests/integration/test_agent_graph.py:34`

**E2E Tests:**
- No browser/E2E framework. The frontend (`frontend/`) is tested separately

## Common Patterns

**Async Testing:**
```python
# pyproject.toml sets asyncio_mode = "auto" — no decorator needed.
async def test_build_agent_graph_compiles_and_runs_happy_path() -> None:
    graph = build_agent_graph(_NoopLLM(), max_steps=2)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
```
(`tests/unit/test_agent_smoke.py:59`)

**Error Testing:**
```python
def test_build_rag_chain_rejects_invalid_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        build_rag_chain(connection_string="postgresql://example", api_key="unused",
                        llm_provider="anthropic", chat_model="claude", k=3)
```
(`tests/unit/test_chain.py:82`)

**Generator Lifecycle Testing (FastAPI deps):**
```python
db = get_db()
next(db)
with pytest.raises(StopIteration):
    next(db)              # forces the finally block
return_connection.assert_called_once_with(conn, close=False)
```
(`tests/unit/test_db.py:14-21`) — used to verify cleanup paths in dependency generators.

**Throwing Into a Generator (caller-error simulation):**
```python
db = get_db(); next(db)
with pytest.raises(ValueError, match="boom"):
    db.throw(ValueError("boom"))
```
(`tests/unit/test_db.py:30-33`)

**Parametrized Edge Cases:**
```python
@pytest.mark.parametrize("party_size_in,expected_in_call", [(None, 2), (0, 2), (4, 4), (1, 1)])
```
(`tests/unit/test_agent_graph.py:573`) — covers None / falsy / typical / boundary in one block. Use this aggressively rather than copy-pasting test bodies.

**Asserting On Captured State:**
```python
captured: dict[str, Any] = {}
async def _ainvoke(state, config=None):
    captured["state"] = state
    return _final_state_dict(reply="ok")
fake_graph.ainvoke = _ainvoke
# ... later:
assert captured["state"].messages[2].content == "actually make it 4 stops"
```
(`tests/unit/test_chat_endpoint.py:108-141`) — preferred over `mock.call_args` when the call is async.

**Cache Hygiene:**
- Tests that exercise `lru_cache`-decorated functions (`_embed_cached`, `get_settings`) must call `.cache_clear()` at the start to avoid bleed between tests (`tests/unit/test_retriever.py:105`, `:123`)

---

*Testing analysis: 2026-05-14*
