# Coding Conventions

**Analysis Date:** 2026-05-14

## Naming Patterns

**Files:**
- Modules use `snake_case.py`: `app/db_pool.py`, `app/retriever.py`, `app/agent/graph.py`
- Test files mirror the module under test with a `test_` prefix: `tests/unit/test_retriever.py` covers `app/retriever.py`; `tests/unit/test_agent_graph.py` covers `app/agent/graph.py`
- Sub-packages keep narrow purposes (`app/agent/`, `app/tools/`, `app/observability/`)
- Layered test variants per feature: `test_<feature>.py` (unit), `test_<feature>_smoke.py` (smoke), `test_<feature>_functional.py` (functional), `tests/integration/test_<feature>.py` (integration). See `tests/unit/test_agent_self_correct.py` vs `tests/unit/test_agent_self_correct_functional.py`

**Functions:**
- Public functions are `snake_case`: `build_database_url`, `resolve_database_url`, `build_rag_chain`, `semantic_search`
- Module-private helpers prefixed with `_`: `_return_connection_safely` (`app/db.py:12`), `_embed_cached` (`app/retriever.py:20`), `_view_name` (`app/tools/retrieval.py:56`)
- LangChain protocol methods keep underscore prefix even when imported: `_get_relevant_documents` (`app/retriever.py:50`)
- Async route handlers are plain `snake_case`: `async def chat(...)` (`app/main.py:276`)

**Variables:**
- `snake_case` for locals; `UPPER_SNAKE_CASE` for module-level constants and enums-as-values
- Module constants: `RAG_UNAVAILABLE_DETAIL` (`app/main.py:27`), `LOW_SIMILARITY_THRESHOLD = 0.55` (`app/agent/graph.py:41`), `DEFAULT_STOP_DURATION_MIN_FALLBACK = 60` (`app/agent/state.py:104`), `ALLOWED_EMBEDDING_TABLES: frozenset[str]` (`app/config.py:10`)
- Private module constants also use leading underscore + UPPER: `_VIEW_FOR_TABLE`, `_OVERFETCH_FACTOR` (`app/tools/retrieval.py:20`, `:29`)

**Types / Classes:**
- `PascalCase` for Pydantic models, dataclasses, and protocol classes: `RecommendationRequest`, `LoadedConfig`, `PgVectorRetriever`, `ItineraryState`, `Stop`, `PlaceHit`, `PlaceCard`, `RevisionHint`
- Type aliases via `typing.Literal` are `PascalCase`: `RevisionReason`, `RevisionAction` (`app/agent/state.py:19`, `:33`)
- Frontend-contract field names that violate Python naming get explicit `# noqa: N815`: `ragLabel: str  # noqa: N815` (`app/main.py:89`)

## Code Style

**Formatting:**
- Tool: `ruff format` (line-length 100), pinned via `.pre-commit-config.yaml` to `v0.15.8`
- Target: `py310` (`pyproject.toml:68`)
- **Pre-commit auto-runs ruff on commit** — do not run `ruff format` manually before committing; the hook handles it
- Pre-commit ruff version MUST stay in sync with the `ruff` version in `poetry.lock` or CI's `ruff format --check` will diverge from the local hook

**Linting:**
- Tool: `ruff check`
- Rules enabled (`pyproject.toml:71`): `E` (pycodestyle), `F` (pyflakes), `I` (isort), `N` (pep8-naming), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify), `S` (bandit security)
- Ignored globally: `E501` (line-too-long handled by formatter)
- Per-file ignores (`pyproject.toml:74`):
  - `tests/**/*.py`: `S101` (allow `assert`), `S105`, `S106` (allow hardcoded "passwords"/tokens in fixtures)
  - `scripts/**/*.py`: `S101` (allow `assert`)
- Inline `# noqa: S608` is the convention for SQL strings built via f-string from validated allowlist values (`app/retriever.py:82`, `app/tools/retrieval.py:94`, `:155`, `:175`, `:200`)

**Type Checking:**
- `mypy app/` via `make typecheck`; config in `pyproject.toml:78` (`python_version = "3.10"`, `strict = false`, `ignore_missing_imports = true`)
- `sqlalchemy[mypy]` plugin enabled via dev deps

## Import Organization

**Order (enforced by ruff `I`):**
1. `from __future__ import annotations` — first line of nearly every `app/` module (`app/main.py:1`, `app/config.py:1`, `app/chain.py:1`, `app/db_pool.py:1`)
2. Standard library
3. Third-party (fastapi, pydantic, langchain, psycopg2, mlflow)
4. First-party (`app.*`, `tests.*`) — relative inside `app/` (`from .config import get_settings`), absolute from tests and cross-package (`from app.tools.booking_types import Provider`)

**Path Aliases:**
- None — Python imports only. Frontend (`frontend/src/api/chat.js`) is a separate JS app

## Error Handling

**Patterns:**
- Domain misconfiguration → `RuntimeError` with a human-readable remediation hint:
  - `raise RuntimeError("Missing OPENAI_API_KEY for query embedding generation.")` (`app/retriever.py:40`)
  - `raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")` (`app/main.py:147`, `app/db_pool.py:102`)
- Programmer errors / invalid inputs → `ValueError` with the offending value:
  - `raise ValueError(f"Unsupported llm_provider: {llm_provider}")` (`app/config.py:148`, `app/chain.py:48`)
- HTTP-layer failures → `HTTPException` with a 503 + remediation detail; never bubble RuntimeError into FastAPI:
  - `raise HTTPException(status_code=503, detail=AGENT_UNAVAILABLE_DETAIL)` (`app/main.py:279`)
- Wrap-and-re-raise to preserve cause chain on infra failures:
  - `raise RuntimeError("Unable to load the MLflow production alias for ...") from exc` (`app/main.py:134`)
- **Degraded boot pattern**: catch broad `Exception` at lifespan startup, log with `exc_info=True`, set state to `None`, and let endpoints surface the 503 — boot must succeed even if MLflow/agent are unreachable (`app/main.py:197-222`)
- **Defensive cleanup**: log + force-close on cleanup failure rather than re-raise. See `_return_connection_safely` (`app/db.py:12`) — if `conn.rollback()` raises, log warning and discard connection rather than crash the request
- Exception-handling guidance: do NOT catch broad `Exception` except in (a) lifespan startup, (b) cleanup paths that must not raise. Add `# pragma: no cover - exercised via startup tests` when the broad catch is tested elsewhere (`app/main.py:133`)

## Logging

**Framework:** stdlib `logging` only. No structured logger / no print statements.

**Patterns:**
- Module-level `logger = logging.getLogger(__name__)` is the default (`app/db.py:9`, `app/db_pool.py:11`, `app/main.py:25`, `app/agent/graph.py:39`)
- Cross-cutting subsystems use a namespaced logger: `logging.getLogger("city_concierge.cost")` (`app/observability/cost.py:20`), `logging.getLogger("city_concierge.observability")` (`app/observability/__init__.py:16`)
- Use `logger.warning(..., exc_info=True)` to attach the traceback rather than `logger.exception` — keeps the message a warning, not an error
- Log on degraded paths and silent failures; do not log on normal happy-path requests (FastAPI/uvicorn already does)

## Comments

**When to Comment:**
- Module docstrings explain *purpose and architecture*, not API surface — see `app/agent/graph.py:1-11` (node responsibilities + edge logic), `app/agent/state.py:1-5`, `app/tools/retrieval.py:1-5`
- Inline comments justify *non-obvious decisions* and link to invariants:
  - "settings.embedding_table is validated against an allowlist at config load (see ALLOWED_EMBEDDING_TABLES in app/config.py), so f-stringing is safe." (`app/retriever.py:66`)
  - "When W6 evals show recall regressing on tightly-filtered queries, bump this..." (`app/tools/retrieval.py:26`)
- Reference workstream IDs (W0..W7) when a comment explains roadmap-driven decisions

**Docstrings:**
- Triple-quoted, prose, no enforced format (no Google/Numpy style)
- Class docstrings explain *why this shape*, not field-by-field rehash: see `UserConstraints` (`app/agent/state.py:57-63`)
- No JSDoc/TSDoc-style param tables

## Function Design

**Size:** Keep functions short and single-purpose. Extract helpers when logic branches (`_grounded_place_ids`, `_commit_stops` in `app/agent/graph.py`).

**Parameters:**
- Use keyword-only args (`*`) for anything with >2 params or where call-sites benefit from named clarity: `build_database_url(*, user, password, dbname, host=..., port=...)` (`app/config.py:13`)
- Default to immutable defaults; for collection defaults, use `Field(default_factory=list)` on Pydantic models (`app/agent/state.py:71`)
- Type-annotate every parameter and return value — even private helpers and test fixtures

**Return Values:**
- Prefer Pydantic models or dataclasses over dicts for cross-module returns (`LoadedConfig`, `BuiltChain`, `PlaceHit`, `PlaceDetails`)
- Use `Optional` (`X | None`) returns rather than raising for "not found" lookups: `get_details(...) -> PlaceDetails | None` (`app/tools/retrieval.py:162`)

## Module Design

**Exports:**
- `__all__` is used selectively to publish a contract surface — see `app/tools/retrieval.py:206-215` exposes `_VIEW_FOR_TABLE` and `ALLOWED_EMBEDDING_TABLES` for the contract test in `tests/unit/test_tools_retrieval.py`
- Otherwise rely on the leading-underscore convention to mark private

**Barrel Files:**
- `app/__init__.py` and `app/agent/__init__.py` are present but kept minimal — no re-export gymnastics

## Configuration & Settings

- Single Pydantic-Settings class `Settings(BaseSettings)` in `app/config.py:75` reads from env + `.env`
- Cached accessor `@lru_cache get_settings()` (`app/config.py:130`); test conftest calls `get_settings.cache_clear()` before/after each test (`tests/conftest.py:41`)
- Validators on settings: `@field_validator("embedding_table")` enforces an allowlist (`app/config.py:103`)
- DB URL resolution is split: `resolve_database_url(env)` (`app/config.py:47`) is a pure function over an env mapping — testable without setting real env vars

## SQL Conventions

- Parameterized queries via `cur.execute(sql, params)` — never string-interpolate user input
- F-string interpolation is allowed ONLY for table/view names sourced from the validated allowlist; mark each such SQL block with `# noqa: S608` and a comment explaining why it's safe (`app/tools/retrieval.py:80`, `:94`)
- Cursors and connections always borrowed via `with get_conn() as conn, conn.cursor(...) as cur:` to guarantee return-to-pool

## Async Conventions

- `pytest` runs in `asyncio_mode = "auto"` (`pyproject.toml:63`) — `async def test_...` works without `@pytest.mark.asyncio`
- FastAPI handlers that call into LangGraph are `async def` and `await graph.ainvoke(...)` (`app/main.py:276`); sync handlers (`/health`, `/root`) stay `def`
- Lifespan via `@asynccontextmanager async def lifespan(app: FastAPI)` (`app/main.py:182`); always pair `init_db_pool` with `close_db_pool()` in a `finally`

## Dependency Injection

- Module-level FastAPI `Depends` are bound to a constant to satisfy ruff B008: `db_connection_dependency = Depends(get_db)` (`app/main.py:36`), then used as the parameter default
- App-state injection (RAG chain, agent graph) lives on `request.app.state` and is set during `lifespan`; handlers fetch with `getattr(request.app.state, "agent_graph", None)` and 503 if missing

---

*Convention analysis: 2026-05-14*
