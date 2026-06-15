#!/usr/bin/env bash
# scripts/provision_sandbox.sh
#
# Provision the isolated sandbox Postgres DB for the loop falsifier (LOOP-00).
#
# Idempotent recipe:
#   1. Assert SANDBOX_DATABASE_URL is set and safe (prod-safety guard).
#   2. Start the pgvector Docker container (db-up).
#   3. CREATE DATABASE idempotently (pg_database pre-check).
#   4. Pipe scripts/db/init.sql (extensions + baseline tables; GOTCHA 1 — MUST precede alembic).
#   5. Layer Alembic migrations (place_embeddings_v2, place_documents_v2, proposals,
#      checkpoints, relations) via DATABASE_URL=$SANDBOX_DATABASE_URL poetry run alembic.
#
# Usage:
#   export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox
#   bash scripts/provision_sandbox.sh
#
# To reset and reprovision:
#   docker exec city_concierge_db psql -U postgres -d postgres -c 'DROP DATABASE city_concierge_sandbox;'
#   bash scripts/provision_sandbox.sh

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

# Docker container name for the pgvector Postgres instance.
CONTAINER_NAME="city_concierge_db"

# The expected sandbox dbname — parsed from SANDBOX_DATABASE_URL below.
# This variable is the canonical one used for CREATE/init/alembic.
# The guard asserts that the URL-parsed name matches this at the bottom.
EXPECTED_SUFFIX="_sandbox"

# ─── Step 0: Assert SANDBOX_DATABASE_URL is set ───────────────────────────────

if [[ -z "${SANDBOX_DATABASE_URL:-}" ]]; then
  echo "ERROR: SANDBOX_DATABASE_URL is not set." >&2
  echo "  Export it before running this script:" >&2
  echo "  export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox" >&2
  exit 1
fi

# ─── Parse dbname and host from SANDBOX_DATABASE_URL ─────────────────────────

# Strip any query string, then extract everything after the last '/'.
_url_no_query="${SANDBOX_DATABASE_URL%%\?*}"
_parsed_dbname="${_url_no_query##*/}"

if [[ -z "${_parsed_dbname}" ]]; then
  echo "ERROR: Could not parse dbname from SANDBOX_DATABASE_URL='${SANDBOX_DATABASE_URL}'." >&2
  echo "  Expected format: postgresql://user:pass@host:port/dbname" >&2
  exit 1
fi

# Extract host (between @ and the next : or /)
_after_at="${SANDBOX_DATABASE_URL#*@}"
_host_port="${_after_at%%/*}"
_parsed_host="${_host_port%%:*}"

# DB_NAME is the canonical variable for this script — every DDL step uses this.
DB_NAME="${_parsed_dbname}"

# ─── Codex MEDIUM: assert URL-parsed dbname == DB_NAME ────────────────────────
# (They are always equal here since DB_NAME is derived from the URL, but if
# someone edits this script and introduces a separate DB_NAME constant, the
# guard would catch the mismatch. Belt-and-suspenders check.)

if [[ "${_parsed_dbname}" != "${DB_NAME}" ]]; then
  echo "ERROR: URL-parsed dbname '${_parsed_dbname}' does not match DB_NAME '${DB_NAME}'." >&2
  echo "  The script would CREATE/init/migrate different databases. Aborting." >&2
  exit 1
fi

# ─── Codex HIGH: Prod-safety guard ────────────────────────────────────────────
# Must run BEFORE any CREATE DATABASE / alembic DDL.
# FAILS CLOSED — if any check cannot be evaluated, exit non-zero.

echo "Running prod-safety guard..."

# (a) dbname must contain/suffix '_sandbox'
if [[ "${DB_NAME}" != *"${EXPECTED_SUFFIX}"* ]]; then
  echo "ERROR: Prod-safety guard rejected SANDBOX_DATABASE_URL." >&2
  echo "  Reason: dbname '${DB_NAME}' does not contain '${EXPECTED_SUFFIX}'." >&2
  echo "  The sandbox database must have '_sandbox' in its name to prevent" >&2
  echo "  accidental provisioning against prod. Aborting." >&2
  exit 1
fi

# (b) Resolve the prod DATABASE_URL (with .env merged) and compare (host, dbname).
#     We use dotenv_values(".env") so prod sitting UNEXPORTED in .env is still compared.
#     If the resolver is unavailable, fail closed.
PROD_URL=""
if command -v poetry &>/dev/null; then
  PROD_URL=$(
    poetry run python -c "
import sys
try:
    from dotenv import dotenv_values
    import os
    from app.config import resolve_database_url
    env = {**dotenv_values('.env'), **os.environ}
    env.pop('SANDBOX_DATABASE_URL', None)
    result = resolve_database_url(env) or ''
    print(result)
except Exception as e:
    print('', end='')  # empty = fail-closed will trigger
    import sys
    print(f'WARN: prod URL resolver raised: {e}', file=sys.stderr)
" 2>/dev/null || echo ""
  ) || true
fi

if [[ -n "${PROD_URL}" ]]; then
  # Parse host and dbname from prod URL
  _prod_no_query="${PROD_URL%%\?*}"
  _prod_dbname="${_prod_no_query##*/}"
  _prod_after_at="${PROD_URL#*@}"
  _prod_host_port="${_prod_after_at%%/*}"
  _prod_host="${_prod_host_port%%:*}"

  if [[ "${_parsed_host}" == "${_prod_host}" && "${DB_NAME}" == "${_prod_dbname}" ]]; then
    echo "ERROR: Prod-safety guard rejected SANDBOX_DATABASE_URL." >&2
    echo "  Reason: (host='${_parsed_host}', dbname='${DB_NAME}') matches the resolved prod URL." >&2
    echo "  SANDBOX_DATABASE_URL must not point at the production database. Aborting." >&2
    exit 1
  fi
else
  # Resolver unavailable or returned empty — fall back to string-only checks.
  # Check (a) already passed. Proceed but note the fallback.
  echo "WARN: Could not resolve prod DATABASE_URL (python/poetry resolver unavailable or .env absent)."
  echo "      Prod-safety relies on the _sandbox suffix check only. Proceed with caution."
fi

# (c) Reject obvious Cloud SQL / prod hostnames unless SANDBOX_ALLOW_REMOTE=1.
#     A '/cloudsql/' socket path or a known GCP prod host pattern indicates prod.
if [[ -z "${SANDBOX_ALLOW_REMOTE:-}" ]]; then
  if [[ "${SANDBOX_DATABASE_URL}" == */cloudsql/* ]]; then
    echo "ERROR: Prod-safety guard rejected SANDBOX_DATABASE_URL." >&2
    echo "  Reason: URL contains '/cloudsql/' (Cloud SQL socket path)." >&2
    echo "  The sandbox target must be a local Docker container." >&2
    echo "  Set SANDBOX_ALLOW_REMOTE=1 only if you are intentionally targeting a remote sandbox." >&2
    exit 1
  fi
  # Reject GCP project-style hostnames (e.g. mlops-491820:us-central1:instance)
  if [[ "${_parsed_host}" == *":"*":"* ]]; then
    echo "ERROR: Prod-safety guard rejected SANDBOX_DATABASE_URL." >&2
    echo "  Reason: host '${_parsed_host}' looks like a Cloud SQL instance connection name." >&2
    echo "  Set SANDBOX_ALLOW_REMOTE=1 only if you are intentionally targeting a remote sandbox." >&2
    exit 1
  fi
fi

echo "Prod-safety guard PASSED: dbname='${DB_NAME}', host='${_parsed_host}' (local Docker)."

# ─── Step 1: Start the pgvector Docker container ──────────────────────────────

echo ""
echo "Step 1: Starting pgvector Docker container (${CONTAINER_NAME})..."
# Use make db-up which delegates to docker compose up -d db.
# Fall back to direct docker compose if make is not available.
if command -v make &>/dev/null; then
  make db-up
else
  docker compose up -d db
fi

# Wait briefly for the container to be ready to accept connections.
echo "Waiting for Postgres to be ready..."
for i in $(seq 1 20); do
  if docker exec "${CONTAINER_NAME}" pg_isready -U postgres -q 2>/dev/null; then
    break
  fi
  sleep 1
done
if ! docker exec "${CONTAINER_NAME}" pg_isready -U postgres -q 2>/dev/null; then
  echo "ERROR: Postgres container '${CONTAINER_NAME}' did not become ready within 20 seconds." >&2
  exit 1
fi
echo "Postgres is ready."

# ─── Step 2: Create the sandbox DB idempotently ───────────────────────────────
# Pre-check rather than swallowing an error under set -euo pipefail (Codex LOW).

echo ""
echo "Step 2: Creating database '${DB_NAME}' (idempotent)..."
_db_exists=$(
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" 2>/dev/null || echo ""
)

if [[ "${_db_exists}" == "1" ]]; then
  echo "Database '${DB_NAME}' already exists — skipping CREATE DATABASE."
else
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -c "CREATE DATABASE \"${DB_NAME}\";"
  echo "Database '${DB_NAME}' created."
fi

# ─── Step 3: Pipe init.sql (GOTCHA 1 — MUST precede alembic upgrade head) ─────
# init.sql uses CREATE TABLE IF NOT EXISTS throughout, so this is idempotent.

echo ""
echo "Step 3: Applying baseline schema (scripts/db/init.sql)..."
echo "  (GOTCHA 1: init.sql MUST run before alembic — the baseline migration is a no-op stamp)"

# Resolve the repo root (where Makefile lives) so the script is portable.
_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker exec -i "${CONTAINER_NAME}" psql -U postgres -d "${DB_NAME}" \
  < "${_repo_root}/scripts/db/init.sql"

echo "init.sql applied."

# ─── Step 4: Layer Alembic migrations ─────────────────────────────────────────
# Retarget alembic at the sandbox via DATABASE_URL= inline (D-10 / D-11).
# alembic is not on PATH — must be invoked via poetry run.

echo ""
echo "Step 4: Applying Alembic migrations (alembic upgrade head)..."
echo "  DATABASE_URL=${SANDBOX_DATABASE_URL}"

cd "${_repo_root}"
DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run alembic upgrade head

echo "Alembic migrations applied."

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "SUCCESS: Sandbox database '${DB_NAME}' provisioned at ${SANDBOX_DATABASE_URL}"
echo "  Schema: places_raw, place_embeddings, place_embeddings_v2, place_documents_v2"
echo "          places_ingest_query_proposals, places_ingest_query_checkpoints, place_relations"
echo "  Rows:   0 (empty — the intended loop-falsifier baseline)"
echo "  Prod:   NEVER touched (all DDL ran against '${DB_NAME}' only)"
echo ""
echo "To reset: docker exec ${CONTAINER_NAME} psql -U postgres -d postgres -c 'DROP DATABASE ${DB_NAME};'"
echo "          bash scripts/provision_sandbox.sh"
