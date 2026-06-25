---
phase: 19-productionized-loop-metric-loop
plan: "02"
subsystem: sandbox-provisioner
tags: [loop, sandbox, provision, populated-baseline, ingest, embed]
dependency_graph:
  requires: [19-01]
  provides: [sandbox-provision-populated, make-sandbox-provision-populated]
  affects: [scripts/provision_sandbox.sh, Makefile]
tech_stack:
  added: []
  patterns:
    - "bash argument parsing with for-loop over $@"
    - "POPULATE_BASELINE gate wrapping ingest + embed subprocesses"
    - "Python heredoc inline script for gap-bucket exclusion upsert"
    - "guard-before-DROP ordering (asserted by unit test)"
key_files:
  created:
    - tests/unit/test_provision_sandbox_populated.py
  modified:
    - scripts/provision_sandbox.sh
    - Makefile
decisions:
  - "Gap-bucket exclusion set covers 3 query dimensions: per-neighborhood cuisine, citywide cuisine, and per-neighborhood generic eatery overlaps (D-02 full coverage)"
  - "Exclusion implemented via checkpoint upsert (status='completed') in Python heredoc — reuses the ingest SKIP_COMPLETED_QUERIES mechanism without touching ingest code"
  - "LOOP_GAP_NEIGHBORHOOD and LOOP_GAP_CUISINE default to 'Outer Sunset' / 'vietnamese' (the Phase 16 calibrated bucket) but are fully parameterised"
  - "--reset is schema-only (no ingest/embed); --populated is the full idempotent populated reset (D-02 LOCKED CONSTRAINT)"
metrics:
  duration: 216
  completed: "2026-06-21"
  tasks: 2
  files: 3
---

# Phase 19 Plan 02: Populated Sandbox Provisioner Summary

**One-liner:** Extend `provision_sandbox.sh` with `--populated` (idempotent populated baseline: DROP+recreate, mark only gap-bucket exclusion queries completed, ingest non-gap catalog, fully embed) and `--reset` (schema-only alias), plus `make sandbox-provision-populated` target and 21 zero-cost source-assertion unit tests.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add --populated / --reset modes to provision_sandbox.sh | 75f7b08 | scripts/provision_sandbox.sh |
| 2 | Add make sandbox-provision-populated target + flag-routing unit test | 5880aae | Makefile, tests/unit/test_provision_sandbox_populated.py |

## What Was Built

### Task 1: `scripts/provision_sandbox.sh`

Added argument parsing at the top of the script (after `set -euo pipefail`) that handles:

- `--reset`: sets `RESET_MODE="1"` only — SCHEMA-ONLY alias (DROP+recreate+schema, no ingest or embed)
- `--populated`: sets both `RESET_MODE="1"` and `POPULATE_BASELINE="1"` — the full idempotent populated reset

**The INVERSION (Phase 19 vs Phase 16):**
Phase 16 marked the NON-GAP catalog 'completed' so ingest ran ONLY the gap query.
Phase 19 marks ONLY the GAP-BUCKET exclusion set 'completed' (ingest skips those), then ingests everything else (the broad non-gap catalog). The gap bucket stays un-ingested at baseline so the loop can add it later.

**Populate step (after alembic, gated on `POPULATE_BASELINE=1`):**
1. A Python heredoc computes the gap-bucket exclusion set from `LOOP_GAP_NEIGHBORHOOD` / `LOOP_GAP_CUISINE` env vars (defaulting to 'Outer Sunset' / 'vietnamese'). The exclusion set covers 3 query dimensions per D-02:
   - Per-neighborhood cuisine: `'{cuisine} restaurants in {neighborhood} San Francisco'`
   - Citywide cuisine: `'{cuisine} restaurants in San Francisco'` (no neighborhood)
   - Per-neighborhood eatery overlaps: any eatery-type query in that neighborhood
2. Upserts ONLY those queries as `status='completed'` checkpoints (ingest SKIPS them)
3. `DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python scripts/ingest_places_sf.py` — ingests the broad non-gap catalog
4. `DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python -m scripts.embed_places_pgvector_v2` — fully embeds BEFORE returning (D-02 no-backlog)

**Guard ordering:** All existing prod-safety guards (dbname-suffix `_sandbox` check, Python `check_prod_safety`) run BEFORE the DROP block, asserted by both grep and unit test (guard marker index < DROP marker index).

**Bare invocation unchanged:** Default `RESET_MODE=""` / `POPULATE_BASELINE=""` means no-flag path is exactly the existing empty-sandbox behavior.

### Task 2: `Makefile` + `tests/unit/test_provision_sandbox_populated.py`

**Makefile:** Added `.PHONY: sandbox-provision-populated` target directly after `sandbox-migrate` with the standard `SANDBOX_DATABASE_URL` env-guard, calling `bash scripts/provision_sandbox.sh --populated`.

**Unit tests (21 tests, zero cost):**
- `TestFlagRoutingProvisionScript` (6 tests): `--populated` sets both RESET_MODE and POPULATE_BASELINE; `--reset` exists; both variables initialised to empty
- `TestGuardBeforeDropOrdering` (1 test): string-index assertion that guard marker index < DROP marker index
- `TestIngestEmbedUnderPopulateBaselineGate` (5 tests): ingest + embed under POPULATE_BASELINE gate; DROP under RESET_MODE gate; SANDBOX_DATABASE_URL used for both ingest and embed
- `TestInversionGapBucketExclusionKeys` (5 tests): LOOP_GAP_NEIGHBORHOOD / LOOP_GAP_CUISINE present; INVERSION comment present; citywide + neighborhood coverage; 'completed' status upserted
- `TestMakefileTarget` (4 tests): target exists with `.PHONY`, env-guard, `--populated` call

## Verification Results

- `bash -n scripts/provision_sandbox.sh`: exits 0 (syntax OK)
- `poetry run pytest tests/unit/test_provision_sandbox_populated.py -x -q`: 21 passed, 0 failed
- `make -n sandbox-provision-populated` (with dummy `SANDBOX_DATABASE_URL`): prints bash command without executing
- Full unit suite: 1637 passed, 9 skipped (no regressions)
- Guard-before-DROP ordering: guard at index 187 in file, DROP at index 198 (source line ordering)

## Deviations from Plan

None — plan executed exactly as written.

The populate step uses an inline Python heredoc for the gap-bucket exclusion upsert (rather than a separate helper script), which is consistent with the existing script's pattern of delegating URL parsing to inline Python. The exclusion logic covers all 3 query dimensions specified in D-02.

## Threat Coverage

| Threat ID | Mitigation Implemented |
|-----------|------------------------|
| T-19-02-01 | `Prod-safety guard PASSED` echo appears before `DROP DATABASE IF EXISTS` in source; unit test asserts index ordering; `set -euo pipefail` aborts on guard failure |
| T-19-02-02 | Both ingest and embed calls use `DATABASE_URL="${SANDBOX_DATABASE_URL}"` inline; unit tests assert this literally |
| T-19-02-03 | CI never runs this target; unit test is source-only with zero API calls |
| T-19-02-SC | No new package installs; reuses existing poetry env |

## Self-Check: PASSED

- `scripts/provision_sandbox.sh` exists: FOUND
- `tests/unit/test_provision_sandbox_populated.py` exists: FOUND
- `Makefile` contains `sandbox-provision-populated:`: FOUND
- Commit 75f7b08 exists: FOUND
- Commit 5880aae exists: FOUND
