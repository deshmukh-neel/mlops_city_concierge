#!/usr/bin/env bash
# scripts/provision_sandbox.sh
#
# Provision the isolated sandbox Postgres DB for the loop falsifier (LOOP-00).
#
# Idempotent recipe (no flags — empty-sandbox baseline):
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
# To reset (schema-only, no data):
#   bash scripts/provision_sandbox.sh --reset
#   (DROP+recreate the DB, rebuild schema — no ingest or embed)
#
# To provision or idempotently reset to the populated baseline (D-01, D-02):
#   export LOOP_GAP_NEIGHBORHOOD="Outer Sunset"
#   export LOOP_GAP_CUISINE="vietnamese"
#   bash scripts/provision_sandbox.sh --populated
#   (DROP+recreate, init.sql, alembic, mark ONLY the gap-bucket queries 'completed',
#    ingest the broad non-gap catalog, fully embed BEFORE returning — D-02 no-backlog)
#   NOTE: --populated IS the idempotent populated reset — it restores the EXACT baseline
#   data AND embeddings every run (DROP+re-provision). It does NOT merely clear proposals.

set -euo pipefail

# ─── Argument parsing ─────────────────────────────────────────────────────────
# --reset:     SCHEMA-ONLY alias (DROP+recreate+schema, no ingest/embed)
# --populated: Idempotent populated reset: DROP+recreate+schema → mark ONLY the
#              gap-bucket exclusion set 'completed' → ingest non-gap catalog → embed.
#              INVERSION from Phase 16: we mark ONLY the gap-bucket queries completed
#              (so ingest skips the gap), NOT the non-gap catalog (which is the baseline).

RESET_MODE=""
POPULATE_BASELINE=""

for _arg in "$@"; do
  case "${_arg}" in
    --reset)
      RESET_MODE="1"
      ;;
    --populated)
      RESET_MODE="1"
      POPULATE_BASELINE="1"
      ;;
    *)
      echo "ERROR: Unknown argument '${_arg}'. Use --reset or --populated." >&2
      exit 1
      ;;
  esac
done

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
# WR-06: URL parsing is delegated to app.loop.falsifier_core.normalize_url via Python.
# The prior bash string-slicing parser mis-handled:
#   - Passwords containing '@' (splits at the wrong '@')
#   - Cloud SQL ?host= socket URLs (host lives in query string, not netloc)
# Python's urllib.parse handles both correctly; reusing the same parser as the
# in-process guard ensures the two guards agree by construction (DRY).
#
# The Python call emits three tab-separated fields: dbname, host, cloud_sql_instance
# (empty string when absent). On failure it prints ERROR to stderr and exits non-zero.

PARSED_FIELDS=$(
  poetry run python -c "
import sys, os
try:
    from app.loop.falsifier_core import normalize_url
    url = os.environ.get('SANDBOX_DATABASE_URL', '')
    host, port, dbname, instance = normalize_url(url)
    print(dbname + '\t' + host + '\t' + instance)
except Exception as e:
    print(f'ERROR: could not parse SANDBOX_DATABASE_URL via normalize_url: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
) || {
  echo "ERROR: Python URL parser failed for SANDBOX_DATABASE_URL." >&2
  echo "  Ensure poetry and app.loop.falsifier_core are available." >&2
  exit 1
}

parsed_dbname=$(printf '%s' "${PARSED_FIELDS}" | cut -f1)
parsed_host=$(printf '%s' "${PARSED_FIELDS}" | cut -f2)
parsed_instance=$(printf '%s' "${PARSED_FIELDS}" | cut -f3)

if [[ -z "${parsed_dbname}" ]]; then
  echo "ERROR: Could not parse dbname from SANDBOX_DATABASE_URL='${SANDBOX_DATABASE_URL}'." >&2
  echo "  Expected format: postgresql://user:pass@host:port/dbname" >&2
  exit 1
fi

# DB_NAME is the canonical variable for this script — every DDL step uses this.
DB_NAME="${parsed_dbname}"

# ─── Codex MEDIUM: assert URL-parsed dbname == DB_NAME ────────────────────────
# (They are always equal here since DB_NAME is derived from the URL, but if
# someone edits this script and introduces a separate DB_NAME constant, the
# guard would catch the mismatch. Belt-and-suspenders check.)

if [[ "${parsed_dbname}" != "${DB_NAME}" ]]; then
  echo "ERROR: URL-parsed dbname '${parsed_dbname}' does not match DB_NAME '${DB_NAME}'." >&2
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
#     The Python check_prod_safety guard uses the same normalize_url logic, so
#     the comparison here is consistent with the in-process guard.  (WR-06 DRY)
if command -v poetry &>/dev/null; then
  GUARD_RESULT=$(
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
  ) || GUARD_RESULT="WARN"
else
  GUARD_RESULT="WARN"
fi

if [[ "${GUARD_RESULT}" == FAIL:* ]]; then
  echo "ERROR: Prod-safety guard rejected SANDBOX_DATABASE_URL." >&2
  echo "  Reason: ${GUARD_RESULT#FAIL:}" >&2
  echo "  Aborting — no DDL has been run." >&2
  exit 1
elif [[ "${GUARD_RESULT}" == "WARN" ]]; then
  echo "WARN: Could not resolve prod DATABASE_URL (python/poetry resolver unavailable or .env absent)."
  echo "      Falling back to dbname suffix check only. Proceed with caution."
fi

echo "Prod-safety guard PASSED: dbname='${DB_NAME}', host='${parsed_host}'."

# ─── DROP+recreate (idempotent reset for --reset and --populated) ─────────────
# Both --reset (schema-only) and --populated (schema+data) start with a clean DB.
# This block runs AFTER the prod-safety guard — guard-before-destroy ordering is
# asserted by the unit test (Prod-safety guard PASSED appears before DROP DATABASE).

if [[ "${RESET_MODE:-}" == "1" ]]; then
  echo ""
  echo "RESET: Dropping and recreating '${DB_NAME}' for a clean baseline..."
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -c "DROP DATABASE IF EXISTS \"${DB_NAME}\";"
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -c "CREATE DATABASE \"${DB_NAME}\";"
  echo "Database '${DB_NAME}' dropped and recreated."
fi

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
db_exists=$(
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" 2>/dev/null || echo ""
)

if [[ "${db_exists}" == "1" ]]; then
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
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker exec -i "${CONTAINER_NAME}" psql -U postgres -d "${DB_NAME}" \
  < "${repo_root}/scripts/db/init.sql"

echo "init.sql applied."

# ─── Step 4: Layer Alembic migrations ─────────────────────────────────────────
# Retarget alembic at the sandbox via DATABASE_URL= inline (D-10 / D-11).
# alembic is not on PATH — must be invoked via poetry run.

echo ""
echo "Step 4: Applying Alembic migrations (alembic upgrade head)..."
echo "  DATABASE_URL=${SANDBOX_DATABASE_URL}"

cd "${repo_root}"
DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run alembic upgrade head

echo "Alembic migrations applied."

# ─── Step 5 & 6: Populate baseline (for --populated; LOOP-01 D-01/D-02) ──────
# INVERSION: Phase 16 marked the non-gap catalog 'completed' so ingest ran ONLY
# the gap. Phase 19 is the OPPOSITE: mark ONLY the gap-bucket exclusion set
# 'completed' (ingest SKIPS those), then ingest everything else (the broad non-gap
# catalog). The gap bucket stays un-ingested so the loop can add it later.
#
# The gap bucket is identified by LOOP_GAP_NEIGHBORHOOD + LOOP_GAP_CUISINE env vars.
# Exclusion set = every build_seed_queries() entry that surfaces that bucket:
#   - per-neighborhood: '{cuisine} restaurants in {neighborhood} San Francisco'
#   - citywide: '{cuisine} restaurants in San Francisco'
#   - per-neighborhood eatery overlaps for that neighborhood (generic food queries)
# (D-02: naive "minus one query" leaves gaps already-covered via other queries)

if [[ "${POPULATE_BASELINE:-}" == "1" ]]; then
  _gap_neighborhood="${LOOP_GAP_NEIGHBORHOOD:-Outer Sunset}"
  _gap_cuisine="${LOOP_GAP_CUISINE:-vietnamese}"

  echo ""
  echo "Step 5: Marking gap-bucket exclusion set 'completed' (ingest will skip these)..."
  echo "  Gap bucket: '${_gap_cuisine}' in '${_gap_neighborhood}'"
  echo "  (INVERSION: marking ONLY the gap queries completed — non-gap catalog will be ingested)"

  DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python - <<PYEOF
import os, sys
# SANDBOX_DATABASE_URL is already set as DATABASE_URL by the inline export above,
# so ingest_places_sf resolves to the sandbox at import time.
import psycopg2
from scripts.ingest_places_sf import build_seed_queries, checkpoint_key

gap_neighborhood = os.environ.get('LOOP_GAP_NEIGHBORHOOD', 'Outer Sunset')
gap_cuisine = os.environ.get('LOOP_GAP_CUISINE', 'vietnamese')
sandbox_url = os.environ['DATABASE_URL']

# Build the gap-bucket exclusion set: all catalog queries that surface the gap bucket.
# Per D-02: include per-neighborhood cuisine query, citywide cuisine query, AND the
# per-neighborhood generic eatery/food queries for that neighborhood (overlap coverage).
catalog = build_seed_queries()

exclusion_set = []
for q in catalog:
    q_lower = q.lower()
    neighborhood_lower = gap_neighborhood.lower()
    cuisine_lower = gap_cuisine.lower()
    # Rule 1: per-neighborhood cuisine query: '{cuisine} restaurants in {neighborhood} San Francisco'
    if cuisine_lower in q_lower and neighborhood_lower in q_lower:
        exclusion_set.append(q)
    # Rule 2: citywide cuisine query: '{cuisine} restaurants in San Francisco'
    #   (but NOT neighborhood-scoped — already covered by Rule 1)
    elif cuisine_lower in q_lower and 'san francisco' in q_lower and neighborhood_lower not in q_lower:
        exclusion_set.append(q)
    # Rule 3: per-neighborhood generic eatery/food overlaps (any eatery type in the gap neighborhood)
    # These are queries like 'restaurants in Outer Sunset San Francisco', 'cafes in Outer Sunset SF'
    elif neighborhood_lower in q_lower and 'san francisco' in q_lower:
        exclusion_set.append(q)

if not exclusion_set:
    print(f"ERROR: No gap-bucket exclusion queries found for '{gap_cuisine}' / '{gap_neighborhood}'.", file=sys.stderr)
    print("       Check that LOOP_GAP_NEIGHBORHOOD and LOOP_GAP_CUISINE match catalog entries.", file=sys.stderr)
    sys.exit(1)

# Upsert ONLY the gap-bucket exclusion set as 'completed' in checkpoints.
# This makes ingest SKIP the gap (skips 'completed' queries) and ingest everything else.
# DO NOT mark the non-gap catalog completed — that would be the Phase-16 direction
# (which would skip the entire baseline and ingest only the gap — the exact opposite).
upsert_sql = """
    INSERT INTO places_ingest_query_checkpoints (query_text, status)
    VALUES (%s, 'completed')
    ON CONFLICT (query_text) DO UPDATE SET status = 'completed'
"""
conn = psycopg2.connect(sandbox_url)
try:
    with conn.cursor() as cur:
        for q in exclusion_set:
            cur.execute(upsert_sql, [checkpoint_key(q)])
    conn.commit()
finally:
    conn.close()

print(f"Marked {len(exclusion_set)} gap-bucket queries 'completed' (ingest will skip these).")
print(f"Non-gap catalog queries (~{len(catalog) - len(exclusion_set)}) remain pending — will be ingested.")
PYEOF

  echo ""
  echo "Step 6: Ingesting the broad non-gap catalog (gap bucket is skipped via completed checkpoints)..."
  DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python scripts/ingest_places_sf.py

  echo ""
  echo "Step 7: Embedding baseline rows (embed-v2; fully embed BEFORE returning — D-02 no-backlog)..."
  DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python -m scripts.embed_places_pgvector_v2

  echo ""
  echo "Populated baseline complete."
  echo "  Gap bucket '${_gap_cuisine}' in '${_gap_neighborhood}' is un-ingested (under-served at baseline)."
  echo "  Non-gap catalog is fully ingested and embedded — ready for before-snapshot."
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
if [[ "${POPULATE_BASELINE:-}" == "1" ]]; then
  echo "SUCCESS: Populated sandbox '${DB_NAME}' provisioned at ${SANDBOX_DATABASE_URL}"
  echo "  Schema: places_raw, place_embeddings, place_embeddings_v2, place_documents_v2"
  echo "          places_ingest_query_proposals, places_ingest_query_checkpoints, place_relations"
  echo "  Rows:   Populated baseline (non-gap catalog ingested + embedded)"
  echo "  Gap:    '${_gap_cuisine:-}' in '${_gap_neighborhood:-}' is EXCLUDED (under-served)"
  echo "  Prod:   NEVER touched (all DDL ran against '${DB_NAME}' only)"
  echo ""
  echo "To idempotently reset to this populated baseline:"
  echo "  export LOOP_GAP_NEIGHBORHOOD='${_gap_neighborhood:-Outer Sunset}'"
  echo "  export LOOP_GAP_CUISINE='${_gap_cuisine:-vietnamese}'"
  echo "  bash scripts/provision_sandbox.sh --populated"
  echo ""
  echo "To reset schema only (no data):"
  echo "  bash scripts/provision_sandbox.sh --reset"
elif [[ "${RESET_MODE:-}" == "1" ]]; then
  echo "SUCCESS: Sandbox database '${DB_NAME}' reset (schema-only) at ${SANDBOX_DATABASE_URL}"
  echo "  Schema: places_raw, place_embeddings, place_embeddings_v2, place_documents_v2"
  echo "          places_ingest_query_proposals, places_ingest_query_checkpoints, place_relations"
  echo "  Rows:   0 (schema-only reset — no ingest or embed)"
  echo "  Prod:   NEVER touched (all DDL ran against '${DB_NAME}' only)"
  echo ""
  echo "To provision a populated baseline:"
  echo "  bash scripts/provision_sandbox.sh --populated"
else
  echo "SUCCESS: Sandbox database '${DB_NAME}' provisioned at ${SANDBOX_DATABASE_URL}"
  echo "  Schema: places_raw, place_embeddings, place_embeddings_v2, place_documents_v2"
  echo "          places_ingest_query_proposals, places_ingest_query_checkpoints, place_relations"
  echo "  Rows:   0 (empty — the intended loop-falsifier baseline)"
  echo "  Prod:   NEVER touched (all DDL ran against '${DB_NAME}' only)"
  echo ""
  echo "To reset and reprovision (schema-only):"
  echo "  bash scripts/provision_sandbox.sh --reset"
  echo ""
  echo "To provision a populated baseline (D-01, D-02):"
  echo "  export LOOP_GAP_NEIGHBORHOOD='Outer Sunset'"
  echo "  export LOOP_GAP_CUISINE='vietnamese'"
  echo "  bash scripts/provision_sandbox.sh --populated"
fi
