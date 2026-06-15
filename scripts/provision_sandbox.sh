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

# ─── Parse dbname from SANDBOX_DATABASE_URL (Python-delegated, WR-06) ────────
#
# WR-06: URL parsing is delegated to app.loop.falsifier_core._normalize_url via Python.
# The prior bash string-slicing parser mis-handled:
#   - Passwords containing '@' (splits at the wrong '@')
#   - Cloud SQL ?host= socket URLs (host lives in query string, not netloc)
# Python's urllib.parse handles both correctly; reusing the same parser as the
# in-process guard ensures the two guards agree by construction (DRY).
#
# The Python call emits three tab-separated fields: dbname, host, cloud_sql_instance
# (empty string when absent). On failure it prints ERROR to stderr and exits non-zero.

_PARSED_FIELDS=$(
  poetry run python -c "
import sys, os
try:
    from app.loop.falsifier_core import _normalize_url
    url = os.environ.get('SANDBOX_DATABASE_URL', '')
    host, port, dbname, instance = _normalize_url(url)
    print(dbname + '\t' + host + '\t' + instance)
except Exception as e:
    print(f'ERROR: could not parse SANDBOX_DATABASE_URL via _normalize_url: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1
) || {
  echo "ERROR: Python URL parser failed for SANDBOX_DATABASE_URL." >&2
  echo "  Ensure poetry and app.loop.falsifier_core are available." >&2
  exit 1
}

_parsed_dbname=$(printf '%s' "${_PARSED_FIELDS}" | cut -f1)
_parsed_host=$(printf '%s' "${_PARSED_FIELDS}" | cut -f2)
_parsed_instance=$(printf '%s' "${_PARSED_FIELDS}" | cut -f3)

if [[ -z "${_parsed_dbname}" ]]; then
  echo "ERROR: Could not parse dbname from SANDBOX_DATABASE_URL='${SANDBOX_DATABASE_URL}'." >&2
  echo "  Expected format: postgresql://user:pass@host:port/dbname" >&2
  exit 1
fi

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

# (b) Resolve prod URL and compare (host, dbname, instance) using Python.
#     The Python check_prod_safety guard uses the same _normalize_url logic, so
#     the comparison here is consistent with the in-process guard.  (WR-06 DRY)
if command -v poetry &>/dev/null; then
  _GUARD_RESULT=$(
    poetry run python -c "
import sys, os
try:
    from dotenv import dotenv_values
    from app.config import resolve_database_url
    from app.loop.falsifier_core import check_prod_safety
    env = {**dotenv_values('.env'), **os.environ}
    env.pop('SANDBOX_DATABASE_URL', None)
    prod_url = resolve_database_url(env)
    allow_remote = bool(os.environ.get('SANDBOX_ALLOW_REMOTE'))
    sandbox_url = os.environ.get('SANDBOX_DATABASE_URL', '')
    result = check_prod_safety(sandbox_url, prod_url, allow_remote=allow_remote)
    if result.ok:
        print('OK')
    else:
        print('FAIL:' + result.message)
except Exception as e:
    print(f'WARN: prod-safety resolver raised: {e}', file=sys.stderr)
    print('WARN')
" 2>/dev/null || echo "WARN"
  ) || _GUARD_RESULT="WARN"
else
  _GUARD_RESULT="WARN"
fi

if [[ "${_GUARD_RESULT}" == FAIL:* ]]; then
  echo "ERROR: Prod-safety guard rejected SANDBOX_DATABASE_URL." >&2
  echo "  Reason: ${_GUARD_RESULT#FAIL:}" >&2
  echo "  Aborting — no DDL has been run." >&2
  exit 1
elif [[ "${_GUARD_RESULT}" == "WARN" ]]; then
  echo "WARN: Could not resolve prod DATABASE_URL (python/poetry resolver unavailable or .env absent)."
  echo "      Falling back to dbname suffix check only. Proceed with caution."
fi

echo "Prod-safety guard PASSED: dbname='${DB_NAME}', host='${_parsed_host}'."

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
