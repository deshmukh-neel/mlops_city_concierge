# Sandbox Database: Provisioning and Reset Runbook

The sandbox database (`city_concierge_sandbox`) is the isolated Postgres database used
exclusively by the **loop falsifier** (Phase 16, FALSIFY-01 / LOOP-00). It is a real,
fully-provisioned copy of the schema running inside the same local Docker pgvector
container — but it is **a completely separate database**, never shared with `places_raw`
in the production or development `city_concierge` database.

## Why a Separate Database?

The loop falsifier must write real Google Places data (via `ingest_places_sf.py`) and
refresh embeddings (via `embed_places_pgvector_v2.py`) as part of its before/after
snapshot cycle. Running this against the shared development or production database would:

1. Pollute the production `places_raw` table with sandbox test data.
2. Make the before/after DB-diff meaningless (you would not know which rows were added
   by the falsifier versus pre-existing).

A dedicated separate database is required. Using a different Postgres schema or
`search_path` trick is **not sufficient**: table names are hard-coded across the ingest,
embed, and retrieval scripts, and those scripts read `DATABASE_URL` directly. The falsifier
injects `DATABASE_URL=$SANDBOX_DATABASE_URL` into the subprocess environment (D-10) to
retarget them at the sandbox — zero changes to the reused scripts.

## Required Environment Variable

```bash
SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox
```

Add this to your `.env` or export it before running any sandbox-related targets. The
variable is documented (commented-out) in `.env.example`. It must **never** point at the
production database; the provisioning script enforces this with a prod-safety guard.

**GOTCHA 2 — Port 5433:** Postgres.app (the native macOS postgres install) squats
`127.0.0.1:5432` and shadows the Docker container. The project exposes the Docker
container on port `5433` (`POSTGRES_PORT=5433` in `.env`). Both the development database
and the sandbox URL use port `5433` on this machine.

## Provisioning

```bash
# 1. Set the sandbox URL (once per shell session, or add to .env)
export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox

# 2. Provision (idempotent — safe to re-run)
make sandbox-provision
```

The `make sandbox-provision` target runs `bash scripts/provision_sandbox.sh`, which
executes the following steps in order:

1. **Prod-safety guard** — validates the URL before any DDL: the dbname must contain
   `_sandbox`, the `(host, dbname)` must differ from the resolved prod `DATABASE_URL`
   (read from the environment with `.env` merged via `dotenv_values`), and Cloud SQL
   socket paths are rejected unless `SANDBOX_ALLOW_REMOTE=1` is set. The guard exits
   non-zero and runs no DDL if any check fails or cannot be evaluated (fail-closed).
2. **db-up** — starts the pgvector Docker container (`city_concierge_db`) if it is not
   already running.
3. **CREATE DATABASE** — creates `city_concierge_sandbox` if it does not already exist
   (pre-checked via `pg_database`, so idempotent under `set -euo pipefail`).
4. **init.sql** — pipes `scripts/db/init.sql` into the new database, creating the
   extensions (`vector`, `uuid-ossp`) and the baseline tables (`places_raw`,
   `place_embeddings`, `place_query_hits`, `places_ingest_query_checkpoints`).
5. **Alembic migrations** — runs `DATABASE_URL="$SANDBOX_DATABASE_URL" poetry run alembic upgrade head`
   to layer the v2 tables and views on top (`place_embeddings_v2`, `place_documents_v2`,
   `places_ingest_query_proposals`, `places_ingest_query_checkpoints` enhancements,
   `place_relations`).

### GOTCHA 1 — init.sql before alembic upgrade head (MUST preserve this order)

The Alembic baseline migration (`b932216bf431_baseline.py`) is a **no-op stamp** — it
assumes `scripts/db/init.sql` already created `places_raw`, `place_embeddings`, and the
other baseline tables. The very next migration (`create_place_embeddings_v2`) adds a
`REFERENCES places_raw(place_id)` foreign key.

**If you run a bare `alembic upgrade head` against an empty database, it will fail with:**

```
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable)
relation "places_raw" does not exist
```

The `make sandbox-provision` target resolves this by always running `init.sql` before
invoking alembic. **Never** run `poetry run alembic upgrade head` alone against an empty
sandbox database.

## Expected Schema After Provisioning

```
Tables:
  places_raw
  place_embeddings
  place_embeddings_v2
  place_query_hits
  places_ingest_query_proposals
  places_ingest_query_checkpoints
  place_relations

Views:
  place_documents_v2

Extension:
  vector (pgvector)
  uuid-ossp
```

Row counts will all be 0 — the sandbox starts empty, which is the intended falsifier
baseline. The loop falsifier's before-snapshot asserts `places_raw` and
`place_embeddings_v2` are both empty in-process before probing — `hit@k = 0/N` is a
verified result, not assumed from construction.

## Verifying the Provisioned Sandbox

```bash
# List all tables (expect 7 tables + 1 view)
docker exec city_concierge_db psql -U postgres -d city_concierge_sandbox -c "\dt"
docker exec city_concierge_db psql -U postgres -d city_concierge_sandbox -c "\dv"

# Confirm empty (0 rows)
docker exec city_concierge_db psql -U postgres -d city_concierge_sandbox \
  -c "SELECT count(*) FROM places_raw;"

# Confirm alembic head
DATABASE_URL="$SANDBOX_DATABASE_URL" poetry run alembic current
# Expected output: e0cd7069bc8f (head)
```

## Reset and Reprovision

To wipe the sandbox and start fresh:

```bash
# Drop the database (from inside the container, always connect via the postgres maintenance DB)
docker exec city_concierge_db psql -U postgres -d postgres \
  -c 'DROP DATABASE city_concierge_sandbox;'

# Re-provision from scratch
make sandbox-provision
```

## Prod-Safety Guarantee

The provisioning script, and the loop falsifier itself (via D-12), enforce that the
sandbox database is the **only write target**:

- `scripts/provision_sandbox.sh` runs the prod-safety guard before any DDL. If
  `SANDBOX_DATABASE_URL` is missing, has a non-`_sandbox` dbname, matches the resolved
  prod `DATABASE_URL`, or points at a Cloud SQL socket path, the script exits non-zero
  with a clear error message and runs no DDL.
- `scripts/loop_falsifier.py` hard-asserts `SANDBOX_DATABASE_URL` is set and its
  `(host, dbname)` differs from any resolvable prod `DATABASE_URL` — exits 2 if not.
- **Prod `places_raw` is NEVER written** by any Phase 16 component.

To test the guard manually (expect a non-zero exit, no DDL):

```bash
# Should fail: prod URL pointed at sandbox target
SANDBOX_DATABASE_URL="$DATABASE_URL" make sandbox-provision

# Should fail: non-_sandbox dbname
SANDBOX_DATABASE_URL="postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_test" make sandbox-provision
```
