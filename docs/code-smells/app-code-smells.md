# `app/` Code Smell Audit

Audit of `app/` (`main.py`, `config.py`, `db.py`, `chain.py`, `retriever.py`).

## Architecture / Design

### 1. Duplicated provider-dispatch logic (DRY violation — highest priority) — ✅ FIXED

The "openai vs gemini, else raise" branching previously existed in three places with slightly different shapes:

- `app/main.py:57-63` — picks default chat model
- `app/config.py:128-133` — picks API key
- `app/chain.py:31-40` — instantiates the LLM

**Resolution:** Consolidated into `app/providers.py` with a `PROVIDERS` dict keyed by name. Each entry holds `default_chat_model`, `api_key`, and `build_llm` callables. All three call sites now go through `get_provider(name)`.

### 2. Duplicated "missing DATABASE_URL" guard — ✅ FIXED

`app/main.py:100-101` and `app/db.py:11-12` both raised the same `RuntimeError` with the same message.

**Resolution:** Added `require_database_url()` helper in `config.py`. Both call sites now use it. `resolved_database_url` still exists as the optional-returning property; `require_database_url()` is the assertive variant.

### 3. `PgVectorRetriever` opens its own connection per query — ✅ FIXED (with #16)

`app/retriever.py:54` — the retriever bypassed `get_db()` and dialed Postgres directly using `connection_string`. Under load this was a fresh TCP+auth roundtrip per request.

**Resolution:** See #16. Both call sites now share a `ThreadedConnectionPool` via `borrow_connection()`. `PgVectorRetriever` no longer takes a `connection_string` kwarg.

### 4. `PgVectorRetriever` re-instantiates `OpenAIEmbeddings` every call — ✅ FIXED

`app/retriever.py:35` — previously built a fresh `OpenAIEmbeddings` client on every query.

**Resolution:** Added `_embeddings` private field + `_get_embeddings()` lazy-init method. The client is constructed once on first use and cached on the retriever instance. (`cached_property` doesn't compose cleanly with langchain's pydantic-based `BaseRetriever`, hence the manual lazy field.)

## Code Quality

### 5. `main.py` is doing too much — ✅ FIXED

`main.py` previously mixed Pydantic schemas, MLflow loading, chain bootstrapping, source serialization, app factory, and routes (195 lines).

**Resolution:** Split into:

- `app/schemas.py` — Pydantic request/response models + `ActiveModelConfig`
- `app/bootstrap.py` — `load_registered_rag_chain`, `parse_active_model_config`
- `app/routes.py` — endpoints (`APIRouter`) + `serialize_sources`
- `app/main.py` — `create_app`, `lifespan`, `app = create_app()` (45 lines)

### 6. Hard-coded CORS origins — ✅ FIXED

`app/main.py:145-146` previously baked in `http://localhost:5173`, `http://localhost:3000`, and the `*.vercel.app` regex.

**Resolution:** Added `cors_allowed_origins: list[str]` and `cors_allowed_origin_regex: str | None` to `Settings`. Defaults preserve current dev behavior. `.env.example` documents `CORS_ALLOWED_ORIGINS` (comma-separated) and `CORS_ALLOWED_ORIGIN_REGEX`. Methods/headers/credentials remain in code since they're tied to API surface, not deployment.

### 7. Hard-coded MLflow IPs in defaults — ✅ FIXED

`app/config.py:90-91` previously defaulted to `http://35.223.147.177:5000` (shared class MLflow server), leaking infra into source.

**Resolution:** Both `mlflow_tracking_uri` and `mlflow_artifacts_uri` now default to `""`. Required via `.env` (already documented in `.env.example`). `app/bootstrap.py` raises a clear `RuntimeError` at startup if `MLFLOW_TRACKING_URI` is missing. `scripts/log_model_to_mlflow.py` skips `os.environ.setdefault("MLFLOW_ARTIFACTS_URI", ...)` when the setting is empty so we don't pollute the env with an empty string.

**Note:** the `mlflow-artifacts://host:port/` form in `.env.example` may still be incorrect per the MLflow URI spec (the scheme is normally host-less), but it works empirically with the running tracking server. Out of scope here — verify with whoever owns the tracking server before changing.

### 8. `lifespan` has no error handling — ✅ FIXED

`app/main.py:128-133` — previously, if `load_registered_rag_chain` raised at startup, uvicorn exited with a stack trace and `/health` could never report why.

**Resolution:** `lifespan` now catches startup errors, logs them via `logger.exception(...)`, and stores `f"{type(exc).__name__}: {exc}"` on `app.state.startup_error`. `/health` returns **503** with `{status: "degraded", error: <reason>}` when `active_model_config` is `None`. Added `test_health_reports_degraded_when_startup_fails` to verify. (Side effect: introduces the first `logging` call in `app/`, partially addresses #14.)

### 9. `parse_active_model_config` mixes types from MLflow params — ✅ FIXED

`app/bootstrap.py` (post-#5 split) previously mixed an int default (`settings.retriever_k`) with a string default (`"0.0"`) and didn't handle the empty-string-from-MLflow edge case.

**Resolution:** Added `_parse_int` / `_parse_float` helpers that treat missing key and empty string identically by stringifying the default first (`params.get(key) or str(default)`). Both numeric fields now use them. Three new tests in `tests/unit/test_bootstrap.py` cover string params, missing-key fallback, and the empty-string edge case.

### 10. Bare `except Exception` at MLflow load

`app/main.py:87` — the `# pragma: no cover` flags this as known sketchy. Catch `mlflow.exceptions.MlflowException` / `RestException` specifically.

### 11. `serialize_sources` slices after retrieval

`app/main.py:115` — `/predict` accepts `limit` up to 20, but the retriever fetches only `k` (e.g., 5) per MLflow config. If `k=5` and the user asks for `limit=20`, they silently get 5. Either clamp `limit` to `k` and document it, or thread `limit` through to the retriever.

### 12. `del run_manager`

`app/retriever.py:29` — unusual style. Rename the parameter `_run_manager` or drop the `del`.

### 13. `vector_to_pg` truncates float precision

`app/retriever.py:14` — `f"{value:.8f}"` loses precision unnecessarily. pgvector accepts full float repr; use `repr(value)` or `str(value)`.

## Tests / Observability

### 14. No logging anywhere in `app/`

Zero `logging` calls in the package. Startup, MLflow load, and prediction errors are all silent. Add a module logger and log at minimum: model version + provider on startup, retriever errors, predict failures.

### 15. `/predict` has no error handling around `rag_chain.invoke`

`app/main.py:189` — any LLM/DB exception 500s with a stack trace to the client. Wrap and return a structured error response.

## Performance

### 16. `get_db()` opens a fresh connection per request — ✅ FIXED

`app/db.py:13` — same root cause as #3.

**Resolution:** Added a module-level `ThreadedConnectionPool` in `app/db.py` (`minconn=1, maxconn=10`) with a `borrow_connection()` context manager. Both `get_db()` and `PgVectorRetriever` borrow from it. `lifespan` calls `close_pool()` on shutdown so the pool is cleanly drained.

### 17. Duplicated embedding model default

`app/retriever.py:19` defaults to `"text-embedding-3-small"` and `app/config.py:87` does the same. The retriever default is dead code (always overridden in `app/chain.py:25`) — drop it, or read from settings inside the retriever.

---

## Suggested fix order

1. **#1** — provider registry. Eliminates ~30 lines of triplicated branching across three files.
2. **#3 / #16** — connection pooling. Actual production risk under load.
3. **#5** — split `main.py`. Unlocks easier testing of everything above.
