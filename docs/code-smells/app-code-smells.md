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

### 4. `PgVectorRetriever` re-instantiates `OpenAIEmbeddings` every call

`app/retriever.py:35` — construct the embeddings client once (in `__init__` or as a pydantic field), not per query.

## Code Quality

### 5. `main.py` is doing too much

`main.py` mixes Pydantic schemas, MLflow loading, chain bootstrapping, source serialization, app factory, and routes. Split into:

- `app/schemas.py` — request/response models
- `app/bootstrap.py` (or `app/mlflow_loader.py`) — `load_registered_rag_chain`, `parse_active_model_config`
- `app/routes.py` — endpoints

`main.py` should be ~30 lines.

### 6. Hard-coded CORS origins

`app/main.py:145-146` bakes in `http://localhost:5173`, `http://localhost:3000`, and the `*.vercel.app` regex. Move to `Settings` (`cors_allowed_origins: list[str]`) so dev/staging/prod differ via env.

### 7. Hard-coded MLflow IPs in defaults

`app/config.py:90-91` defaults to `http://35.223.147.177:5000`, leaking infra into source. Require it via env (`= ""` + validation) or at minimum comment that it's the shared class server. Also `mlflow_artifacts_uri` looks malformed — `mlflow-artifacts://` URIs normally don't include `host:port`.

### 8. `lifespan` has no error handling

`app/main.py:128-133` — if `load_registered_rag_chain` raises, the app crashes at startup with a stack trace and `/health` can never report why. Catch, log, and store the error on `app.state` so `/health` returns `"rag_chain unavailable: <reason>"`.

### 9. `parse_active_model_config` mixes types from MLflow params

`app/main.py:65-72` — `int(params.get("k", settings.retriever_k))` works because the default is already an int, but `params.get("temperature", "0.0")` mixes string and float defaults. Standardize on strings everywhere or normalize once.

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
