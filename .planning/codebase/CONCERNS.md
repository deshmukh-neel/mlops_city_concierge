# Codebase Concerns

**Analysis Date:** 2026-05-14

## Tech Debt

**Stray ad-hoc test script at repo root:**
- Issue: `test_google_api.py` is a one-off Google Places probe at the repo root, not under `tests/`. It uses an env var spelled `GOOGLE-PLACES-API-KEY` (hyphens, not underscores — invalid POSIX env var name) and is not collected by pytest. Looks like dead exploratory code mixed with real tests.
- Files: `test_google_api.py`
- Impact: Confusion for newcomers, false signal that this is part of the test suite, broken if anyone tries to run it.
- Fix approach: Either move into `scripts/` as a manual probe, or delete now that ingestion is owned by `scripts/ingest_places_sf.py`.

**v1/v2 embedding script duplication:**
- Issue: `scripts/embed_places_pgvector.py` (407 lines) and `scripts/embed_places_pgvector_v2.py` (510 lines) share ~80% of their logic intentionally during the v1→v2 migration.
- Files: `scripts/embed_places_pgvector.py`, `scripts/embed_places_pgvector_v2.py`
- Impact: Drift risk — re-running v1 may write inconsistent embeddings vs v2.
- Fix approach: Per `implementation_plan/james/FUTURE_WATCH.md:86-100`, delete v1 once W6 evals confirm `EMBEDDING_TABLE=place_embeddings_v2` wins. Do NOT refactor into a shared base class during the migration — explicit duplication is the point.

**Baseline DDL split between `init.sql` and Alembic:**
- Issue: `scripts/db/init.sql` is a baseline-only DDL that runs on fresh Docker-Compose Postgres starts. Alembic owns all subsequent changes (`alembic/versions/`). New databases pick up everything; pre-existing local databases need a manual `alembic stamp head`.
- Files: `scripts/db/init.sql`, `alembic/versions/*.py`
- Impact: Confusing dual-source-of-truth — easy to "fix" by editing `init.sql` and silently miss Cloud SQL prod and existing local DBs (called out in the file's own header comment, lines 3-7).
- Fix approach: Once everyone's local DB is on Alembic head, replace `init.sql` with a one-line `CREATE EXTENSION vector;` and let Alembic own everything else, including baseline.

**HNSW indexes default-tuned:**
- Issue: `place_embeddings`, `place_embeddings_v2`, and `city_chunks` all use `USING hnsw (embedding vector_cosine_ops)` with no `m` / `ef_construction` parameters specified.
- Files: `scripts/db/init.sql:25-28`, `scripts/db/init.sql:105-107`, `alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py:42`
- Impact: Default `m=16, ef_construction=64` is fine at current corpus size but recall degrades as corpus grows past ~50k vectors. No `SET hnsw.ef_search` is set per-query either.
- Fix approach: Once W6 evals run, profile recall@k on v2; if regressed, rebuild with tuned `m=24, ef_construction=128` and add a per-session `SET LOCAL hnsw.ef_search = 100` in `app/retriever.py` and `app/tools/retrieval.py`.

**`mlflow_tracking_uri` defaults to `http://localhost:5000`:**
- Issue: `app/config.py:93` defaults `mlflow_tracking_uri` to `http://localhost:5000`. Production and CI must set the env var; if unset, the app silently boots in degraded mode (`/predict` and `/chat` return 503).
- Files: `app/config.py:93-94`
- Impact: Misconfigured deploys don't fail loud — they boot and return 503 forever.
- Fix approach: Either fail-fast at startup if `MLFLOW_TRACKING_URI` is unset and `APP_ENV=production`, or surface this in `/health` more prominently than `status: degraded`.

**`gemini-3.1-flash-lite-preview` not priced in cost telemetry:**
- Issue: W5 flipped the default judge to Gemini 3.1 Flash Lite preview, but `app/observability/cost.py:24-32` `PRICING` dict only knows `gemini-2.5-flash`. Judge calls log `est_cost_usd = 0.0`.
- Files: `app/observability/cost.py:24-32`
- Impact: Cost dashboards understate Gemini judge spend; A/B comparisons across judge models are biased.
- Fix approach: Add a `gemini-3.1-flash-lite-preview` row with current per-MTok rates from the Gemini docs (per `implementation_plan/james/FUTURE_WATCH.md:158-167`).

**`f"…SELECT … FROM {settings.embedding_table}"` repeated:**
- Issue: `# noqa: S608` (raw SQL injection bypass) appears 15 times across `app/retriever.py`, `app/tools/retrieval.py`, `scripts/embed_places_pgvector_v2.py`, and `scripts/ingest_places_sf.py`. Each is justified by an allowlist (`ALLOWED_EMBEDDING_TABLES` in `app/config.py:10`) or hardcoded literal.
- Files: `app/retriever.py:68-82`, `app/tools/retrieval.py:82-200`, `scripts/embed_places_pgvector_v2.py:305-429`, `scripts/ingest_places_sf.py:596`
- Impact: Justified today, but if any new contributor adds an f-string SQL with user input by copy-paste, the `noqa` precedent makes it easy to miss.
- Fix approach: Centralize the table-name → view mapping in one helper (`_view_for_embedding_table()` already exists in `app/tools/retrieval.py:20`), and document the rule in `CLAUDE.md`.

## Known Bugs / Recent Fixes (likely fragile areas)

**Alembic IAM-token URL escaping (recently fixed):**
- Symptoms: Alembic migrations failed in CI when the IAM auth token (used as DB password) contained `%` characters, because configparser interpolated them.
- Files: `alembic/env.py`, `app/db_url.py:17-33` (commit `9146a49`)
- Trigger: GitHub Actions `integration-cloud` job using `gcloud sql generate-login-token` as the password in `DATABASE_URL`.
- Workaround: The fix escapes `%` before handing to configparser. Watch for regressions when `alembic/env.py` is touched.

**Proposals table writes blocked in CI by missing GRANT (recently fixed):**
- Symptoms: W5 coverage-agent integration tests failed with permission denied on INSERT/DELETE.
- Files: `alembic/versions/2026_05_08_1100-a1b2c3d4e5f6_grant_ci_sa_proposals.py`, CI workflow integration job
- Trigger: New table created without GRANT to the `github-actions-deployer@mlops-491820.iam` SA.
- Workaround: An idempotent `GRANT INSERT, DELETE` migration plus a CI skip-gate (`01cf104`) that no-ops when the table isn't deployed yet. Both are legitimate but signal a gap: **new tables that CI must write to need an explicit GRANT migration; this isn't enforced anywhere except convention.**
- Fix approach: Add a CI lint that diffs `op.create_table(...)` migrations against a list of known SA-writable tables and fails if a GRANT migration isn't present.

**`PlaceCard` missing `address`/`rating`/`price_level`:**
- Symptoms: Frontend place cards render with `address: null, rating: null, price_level: null` even though `places_raw` has all three.
- Files: `app/agent/state.py` (`Stop`), `app/agent/graph.py` (`_commit_stops`), `app/agent/io.py:38` (`state_to_cards`)
- Trigger: Every committed itinerary.
- Workaround: None today; documented in `implementation_plan/james/FUTURE_WATCH.md:46-68`.
- Fix approach: Extend `Stop` with the three fields and populate from the grounded `PlaceHit`/`PlaceDetails` already in `state.scratch` — no extra DB calls.

## Security Considerations

**No authentication on `/chat`, `/predict`:**
- Risk: The FastAPI app exposes `/chat`, `/predict`, `/health`, `/health/db`, `/root` with **no authentication**. Once Cloud Run is publicly reachable, anyone can issue arbitrary LLM queries on your OpenAI/Gemini bill.
- Files: `app/main.py:249-315` (no `Depends(verify_token)` anywhere)
- Current mitigation: CORS allowlist (`http://localhost:5173`, `http://localhost:3000`, `https://*.vercel.app`) blocks browser cross-origin abuse but does nothing against direct `curl`/scripted clients.
- Recommendations: Add either (a) a shared-secret API-key header, (b) Cloud Run IAM auth + Identity-Aware Proxy, or (c) per-user rate limiting (already flagged as deferred in `implementation_plan/james/w0_infra.md:19`). Without this the LLM endpoints are an open ATM.

**MLflow server hardcoded public IP, no auth:**
- Risk: `35.223.147.177:5000` is a public GCP VM with no TLS / no auth ("acceptable for class, open registry for productionization story" — `implementation_plan/james/w0_infra.md:147`). Anyone scanning that IP can read or modify model versions, including flipping the production alias.
- Files: `infra/compute.tf` (`mlflow-server` instance), historical `docker-compose.yml`/CI references
- Current mitigation: The firewall `allow-mlflow` (`infra/compute.tf:60-72`) is *now* locked to `source_ranges = ["10.128.0.0/9"]` (internal GCP range only), which is good — but the planning doc warns the public-IP era is recent.
- Recommendations: Verify no path still resolves the public IP from outside GCP. Add MLflow basic auth + TLS proxy (W0 §3 still deferred per `README.md:22`).

**SSL mode `ALLOW_UNENCRYPTED_AND_ENCRYPTED` on Cloud SQL:**
- Risk: `infra/sql.tf:46` sets `ssl_mode = "ALLOW_UNENCRYPTED_AND_ENCRYPTED"`, meaning unencrypted connections are accepted. Combined with `authorized_networks { value = "149.36.48.76" }` (a developer's home IP, line 51-53), there's a path for plaintext Postgres traffic over the public internet.
- Files: `infra/sql.tf:46`, `infra/sql.tf:50-53`
- Current mitigation: Cloud SQL Proxy is the recommended access path; IAM auth is enabled (`cloudsql.iam_authentication=on`).
- Recommendations: Tighten to `ssl_mode = "ENCRYPTED_ONLY"` and remove the personal authorized network in favor of the proxy + IAM.

**Cloud SQL backups disabled:**
- Risk: `infra/sql.tf:31-41` has `backup_configuration { enabled = false, point_in_time_recovery_enabled = false }`. A bad migration or `DROP TABLE` is unrecoverable.
- Files: `infra/sql.tf:31-41`
- Current mitigation: `deletion_protection = true` on the instance prevents accidental destroy, but doesn't help against schema/data corruption.
- Recommendations: Enable daily backups and PITR. The retention block is already configured; just flip `enabled = true`.

**`CVE-2025-*` / `CVE-2026-*` waivers in `.trivyignore`:**
- Risk: 13 CVEs ignored, mostly MLflow-server-side bugs (path traversal, command injection, auth bypass) marked "not applicable to our client container."
- Files: `.trivyignore`
- Current mitigation: Justified per-CVE in comments; client container does not run the MLflow server. But the **shared MLflow VM does** and is unpatched (TODO comment on line 38).
- Recommendations: Upgrade MLflow on the GCP VM to 3.x as the comment instructs, then re-evaluate the waiver list. Add a calendar reminder for the "Re-evaluate quarterly" line at the top.

**Broad `except Exception` patterns:**
- Risk: 9+ locations swallow `Exception` (ruff `BLE001` suppressed). Hides bugs in critique nodes and observability paths.
- Files: `app/agent/graph.py:89,535`, `app/agent/critique/vibe.py:121`, `app/agent/critique/checks.py:194`, `app/observability/__init__.py:65,94,103`, `app/main.py:197,215`, `app/db_pool.py:73`
- Current mitigation: Each one logs with `exc_info=True`, so traces aren't fully lost.
- Recommendations: The startup handlers in `app/main.py:197,215` and pool teardown in `app/db_pool.py:73` are legitimately broad (degraded-mode boot). The agent-graph and critique catches should narrow to `LangChainException`/`OutputParserException`/`ValueError` — anything else is a real bug masquerading as a "judge unavailable" string.

## Performance Bottlenecks

**No embedding cache shared across processes:**
- Problem: `app/retriever.py:19-25` uses `@lru_cache(maxsize=4096)` per-process for query embeddings. On Cloud Run with multiple replicas (and after each cold start), the cache is empty.
- Files: `app/retriever.py:19-41`
- Cause: In-memory cache, not Redis or Cloud SQL-backed.
- Improvement path: Already flagged as deferred in `implementation_plan/james/w0_infra.md:19` ("embedding cache"). Add a Redis-backed cache keyed on `(query, model)` with a 7-day TTL.

**`/chat` and `/predict` not streaming:**
- Problem: Frontend waits for the full LangGraph trajectory before any token is shown. With 3-stop itineraries calling 6+ tools this can be 8-15s.
- Files: `app/main.py:275-301` (`/chat`)
- Cause: `await graph.ainvoke(...)` blocks until `END`.
- Improvement path: Switch to `graph.astream_events(...)` and a streaming response (per `implementation_plan/james/FUTURE_WATCH.md:218-225`). Trigger this when real-user latency feedback arrives.

**Per-row INSERT in legacy v1 embed script:**
- Problem: Capped throughput at ~50 rows/min through Cloud SQL proxy.
- Files: `scripts/embed_places_pgvector.py`
- Cause: Per-row `upsert_embedding`, no `execute_values` batching.
- Improvement path: Already fixed in v2 (`implementation_plan/james/w0a_embeddings_v2.md:520`); v1 left unchanged because it's being deprecated. If v1 is somehow promoted again, port the batching first.

**MLflow registry hit on every cold start:**
- Problem: `app/main.py:121-156` (`load_registered_rag_chain`) calls MLflow `get_model_version_by_alias` + `get_run` synchronously during FastAPI lifespan. If MLflow VM is slow or down, cold starts take seconds or fail outright.
- Files: `app/main.py:121-223`
- Cause: No timeout, no local cache of last-known-good config.
- Improvement path: Wrap the registry lookup in a `tenacity` retry with a short total budget (~5s) and persist the resolved config to disk so subsequent cold starts can boot from cache while refreshing in the background.

## Fragile Areas

**LangGraph ↔ Pydantic AI tool adapter:**
- Files: `app/agent/tools.py` (`_to_lc_tool` helper), `app/agent/graph.py:1-50`
- Why fragile: Two fast-moving libraries. Either may break the adapter on minor-version bumps. `implementation_plan/james/FUTURE_WATCH.md:10-29` flags this explicitly.
- Safe modification: Pin both `langgraph` and `pydantic-ai` to exact minor versions before any agent-graph change. Run `tests/unit/test_agent_graph.py` (646 lines) and the smoke script (`scripts/smoke_w3.py`) before merging.
- Test coverage: Strong — 646-line unit suite plus `tests/integration/test_agent_graph.py`.

**`app/agent/graph.py` size and density:**
- Files: `app/agent/graph.py` (596 lines, approaching the 400-line trigger from `FUTURE_WATCH.md:80-84`)
- Why fragile: Single file owning plan/act/critique nodes plus revision-counting logic. Touching one node risks breaking another's edge case.
- Safe modification: Per FUTURE_WATCH guidance, do NOT pre-split. Once it crosses ~700 lines OR one of the helper sections crosses ~150 lines, split into `app/agent/nodes/{plan,act,critique}.py`.
- Test coverage: Strong — 646-line unit + 807-line self-correct functional + integration.

**Integration-test skip gate hides real failures:**
- Files: `tests/integration/` (recent commit `01cf104` "skip integration writes when proposals table isn't deployed")
- Why fragile: Skip-on-missing-table prevents red CI when migrations aren't applied first, but also masks real "we forgot to migrate prod" mistakes.
- Safe modification: Keep the alembic-upgrade-then-test ordering in CI (`integration-cloud` job runs `alembic upgrade head` before pytest, lines 150 / 449-452). Don't relax the skip gate further.
- Test coverage: Mixed. Unit covers the agent code; the skip gate itself isn't tested.

**`places_raw.source_json` is JSONB free-for-all:**
- Files: `scripts/db/init.sql:49`, `scripts/ingest_places_sf.py`
- Why fragile: Raw Google Places API responses dumped without schema validation. Downstream code (`scripts/embed_places_pgvector_v2.py`, W7 KG builders) reaches into nested keys (`addressDescriptor.landmarks[]`, `containingPlaces[]`) that Google can rename/remove without warning.
- Safe modification: When Google changes a field name, the embed script crashes silently for affected rows. Add a Pydantic model for the subset of `source_json` we actually consume.
- Test coverage: `tests/unit/test_embed_v2_compose.py` (202 lines) covers known shapes; no coverage of "Google added a new key."

## Scaling Limits

**Single Cloud SQL instance, ZONAL availability:**
- Current capacity: `db-perf-optimized-N-8`, ZONAL (single zone), 100GB SSD (`infra/sql.tf:14-19`).
- Limit: Zone outage = total downtime. No read replicas.
- Scaling path: Promote to `REGIONAL` availability_type for HA failover; add a read replica for embedding-script bulk reads.

**MLflow VM is `e2-small`:**
- Current capacity: `e2-small` single VM (`infra/compute.tf:3`), no auto-scaling, no backups configured.
- Limit: One concurrent MLflow user comfortably; tracking server bogs down with simultaneous run-logging from multiple W6 eval jobs.
- Scaling path: Move to a managed MLflow (Databricks-hosted or self-hosted on GKE with HA).

**DB connection pool max=10 by default:**
- Current capacity: `db_pool_max_connections = 10` (`app/config.py:99`).
- Limit: ~10 concurrent `/chat` requests before a request blocks waiting for a connection.
- Scaling path: Tune up once Cloud Run concurrency exceeds 10 per instance, but Cloud SQL proxy connection limits dominate at higher fan-out. Profile before raising.

## Dependencies at Risk

**LangChain 0.2 + LangGraph 0.2:**
- Risk: Both pinned with permissive ranges (`langchain >= 0.2.0,<1.0.0`, `langgraph >= 0.2.0,<1.0.0` in `pyproject.toml:25-29`). Minor-version bumps can break the LangGraph ↔ Pydantic AI adapter.
- Impact: Agent graph stops loading on `pip install -U`.
- Migration plan: Pin to exact minor versions (`langchain ~= 0.2`, `langgraph ~= 0.2`); upgrade in a dedicated PR with the smoke scripts.

**MLflow 2.x with 3.x migration looming:**
- Risk: Most `.trivyignore` entries are MLflow CVEs awaiting "MLflow 3.x migration" (file comment line 38).
- Impact: Stuck on a vulnerable major until the migration happens.
- Migration plan: Schedule MLflow 3.x upgrade as a dedicated workstream; tracking-URI compat and run-format changes need testing against the registered RAG model.

**`google-genai >= 0.7.0,<1.0.0`:**
- Risk: Pre-1.0 SDK; breaking changes are normal.
- Files: `pyproject.toml:21`
- Impact: Gemini provider path (`app/main.py:104-110`) breaks on minor bump.
- Migration plan: Pin to exact version; upgrade in lockstep with `langchain-google-genai`.

## Missing Critical Features

**No request authentication or rate limiting:**
- Problem: See "Security Considerations" above. Open LLM endpoints on public Cloud Run.
- Blocks: Production launch beyond classroom demo.

**No backups on Cloud SQL:**
- Problem: See "Security Considerations." Single bad migration = data loss.
- Blocks: Any "real users" scenario.

**No streaming responses:**
- Problem: 8-15s wait for `/chat` results.
- Blocks: User experience parity with ChatGPT-tier products.

**MLflow tracking server unauthenticated:**
- Problem: Anyone on the internal GCP network can flip the production alias.
- Blocks: Multi-team / multi-project use of the same MLflow server.

## Test Coverage Gaps

**No frontend tests detected:**
- What's not tested: `frontend/` Vue/React app — no `*.test.js` / `*.spec.ts` files visible.
- Files: `frontend/`
- Risk: `chat.js` API contract drift goes unnoticed until manual smoke.
- Priority: Medium (small surface area today; grows with W4 booking UI).

**Skip-gate-protected integration tests:**
- What's not tested: When the proposals table isn't deployed, the W5 integration tests skip silently. There's no "regression-detect" test that fails when a required migration is missing.
- Files: `tests/integration/test_coverage_agent.py`
- Risk: A future contributor removes a migration; CI passes; prod breaks.
- Priority: Medium.

**No load / soak tests:**
- What's not tested: `/chat` under concurrency, embedding-cache hit rate, DB-pool exhaustion.
- Files: (none)
- Risk: Pool max=10 and per-process LRU cache will surface as latency cliffs in production.
- Priority: Low until launch traffic appears.

**`source_json` schema-drift coverage:**
- What's not tested: Behavior when Google Places adds/removes/renames a key in `source_json`.
- Files: `tests/unit/test_embed_v2_compose.py`
- Risk: Silent embed-script failures on a subset of rows.
- Priority: Medium.

**`gemini-3.1-flash-lite-preview` cost-estimation gap is untested:**
- What's not tested: Cost telemetry asserts `est_cost_usd > 0` for known models, but no test catches the `= 0.0` fallback for the new judge model.
- Files: `tests/unit/test_observability_cost.py`
- Risk: Cost dashboards show $0 forever and no alarm fires.
- Priority: Low until cost dashboards are user-visible.

---

*Concerns audit: 2026-05-14*
