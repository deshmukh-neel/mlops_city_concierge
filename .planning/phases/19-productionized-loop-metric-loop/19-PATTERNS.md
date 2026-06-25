# Phase 19: Productionized Loop + Metric - Pattern Map

**Mapped:** 2026-06-20
**Files analyzed:** 4 net-new units (loop_runner.py, compute_recall_at_k, make loop target, populated-baseline provisioning/reset)
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/loop_runner.py` | orchestrator/service | event-driven (stage chain) | `scripts/loop_falsifier.py` | exact |
| `app/loop/falsifier_core.py` (add `compute_recall_at_k`) | utility/pure-function | transform | `app/loop/falsifier_core.py::compute_hit_rate` | exact (same file) |
| `Makefile` (add `loop` + `sandbox-provision-populated` targets) | config | request-response | `Makefile` L244-246 (`loop-falsifier`), L53-60 (`sandbox-provision`) | exact |
| `scripts/provision_sandbox.sh` (extend for populated baseline + DROP reset) | utility/config | batch / file-I/O | `scripts/provision_sandbox.sh` (existing) | exact |

---

## Pattern Assignments

---

### `scripts/loop_runner.py` (orchestrator, event-driven stage chain)

**Analog:** `scripts/loop_falsifier.py`

**Imports pattern** (`loop_falsifier.py` lines 36-61):
```python
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from typing import Any

import mlflow
from dotenv import dotenv_values

from app.loop.falsifier_core import (
    EXIT_FAIL,
    EXIT_INFRA,
    EXIT_PASS,
    GuardResult,
    K,
    N,
    check_non_circularity,
    check_prod_safety,
    compute_hit_rate,
    db_diff,
    is_strictly_positive_delta,
)
```
Phase 19 also imports `compute_recall_at_k` from the same module, and imports
`gap_mine_main` from `scripts.coverage_agent` — but ONLY after the sandbox
coercion/cache-clear sequence (see coercion-ordering pattern below).

---

**Deferred-import pattern — ALL settings-touching app.* imports INSIDE main()**
(`loop_falsifier.py` docstring lines 25-33; main() lines 405-469):
```python
# At module scope: ONLY stdlib-only imports + falsifier_core (which is also stdlib-only).
# ALL other app.* imports are deferred to inside main(), AFTER the DATABASE_URL coercion.

def main() -> None:
    # ── Step 1: Read SANDBOX_DATABASE_URL BEFORE any app.* touch ──
    sandbox_url = os.environ.get("SANDBOX_DATABASE_URL")
    if not sandbox_url:
        print("[INFRA] SANDBOX_DATABASE_URL is not set. Cannot proceed.", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)

    # ── Step 2: Resolve prod URL BEFORE coercion ──
    prod_url = resolve_prod_url(sandbox_url=sandbox_url)

    # ── Step 3: Coerce DATABASE_URL = sandbox ──
    os.environ["DATABASE_URL"] = sandbox_url

    # Invalidate lru_cache immediately after coercion.
    from app.config import get_settings        # noqa: PLC0415
    from app.db_pool import close_db_pool      # noqa: PLC0415
    get_settings.cache_clear()
    close_db_pool()

    # ── Step 4: W1 resolved-target assertion ──
    from app.config import resolve_database_url  # noqa: PLC0415
    resolved = resolve_database_url(os.environ)
    assert_resolved_target(sandbox_url=sandbox_url, resolved_url=resolved)

    settings_resolved = get_settings().resolved_database_url
    if settings_resolved != sandbox_url:
        print(
            f"[INFRA] get_settings().resolved_database_url ({settings_resolved!r}) does not match ...",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    # ── Step 5: Deferred app.* imports ──
    from app.tools.retrieval import semantic_search   # noqa: PLC0415
    # Phase 19 also imports gap_mine_main HERE (not at module scope):
    from scripts.coverage_agent import gap_mine_main  # noqa: PLC0415
```
**Critical rule (D-07 LOCKED CONSTRAINT):** `gap_mine_main` (and any call that
touches `get_conn()`) MUST be imported AFTER Steps 3–4. Running gap-mine before
the coercion/cache-clear is unsafe — it could hit prod.

---

**resolve_prod_url helper** (`loop_falsifier.py` lines 152-171):
```python
def resolve_prod_url(sandbox_url: str | None) -> str | None:
    """Resolve the prod DATABASE_URL BEFORE os.environ is coerced."""
    merged: dict[str, str | None] = {**dotenv_values(".env"), **os.environ}
    merged.pop("DATABASE_URL", None)
    merged.pop("SANDBOX_DATABASE_URL", None)
    from app.config import resolve_database_url  # noqa: PLC0415
    return resolve_database_url(merged)
```

---

**assert_resolved_target helper** (`loop_falsifier.py` lines 133-149):
```python
def assert_resolved_target(sandbox_url: str, resolved_url: str | None) -> None:
    if resolved_url != sandbox_url:
        print(
            f"[INFRA] Resolved in-process DATABASE_URL ({resolved_url!r}) does not match "
            f"SANDBOX_DATABASE_URL ({sandbox_url!r}). ...",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)
```

---

**`_snapshot_ids_from_url` — pool-independent snapshot** (`loop_falsifier.py` lines 650-665):
```python
def _snapshot_ids_from_url(db_url: str, table: str) -> set[str]:
    """Direct psycopg2 connection — NOT the pool — to avoid cached-settings dependency."""
    import psycopg2  # noqa: PLC0415

    with contextlib.closing(psycopg2.connect(db_url)) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT place_id FROM {table}")  # noqa: S608
        return {row[0] for row in cur.fetchall()}
```
Phase 19 reuses this pattern unchanged for before/after snapshots of
`place_embeddings_v2` (the v2 DB-diff target set per D-03).

---

**`run_subprocess_or_infra` — subprocess driver** (`loop_falsifier.py` lines 285-299):
```python
def run_subprocess_or_infra(argv: list[str], env: dict[str, str]) -> None:
    """Non-zero subprocess exit → EXIT_INFRA(2), not silent FAIL."""
    try:
        subprocess.run(argv, env=env, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        print(
            f"[INFRA] Subprocess {argv!r} failed with exit code {exc.returncode}. ...",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA) from exc
```
Phase 19 uses the same pattern for ingest and embed-v2 subprocesses, passing
`child_env = {**os.environ, "DATABASE_URL": sandbox_url}`.

---

**Before/after snapshot + DB-diff sequence** (`loop_falsifier.py` lines 488-598):
```python
# Before: capture v2 ID set BEFORE ingest
before_v2_ids_result = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")

# Run ingest subprocess
child_env = {**os.environ, "DATABASE_URL": sandbox_url}
run_subprocess_or_infra(
    argv=[sys.executable, "scripts/ingest_places_sf.py"],
    env=child_env,
)
# Run embed subprocess
run_subprocess_or_infra(
    argv=[sys.executable, "-m", "scripts.embed_places_pgvector_v2"],
    env=child_env,
)

# After: capture v2 ID set AFTER ingest+embed
after_v2_ids_result = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")
new_v2_ids = db_diff(before_v2_ids_result, after_v2_ids_result)
embed_added_count = len(new_v2_ids)

# Guard: 0 new embeddings → EXIT_INFRA (D-02 loud-fail)
if embed_added_count == 0:
    print("[INFRA] embed-v2 produced 0 new embeddings. ...", file=sys.stderr)
    raise SystemExit(EXIT_INFRA)
```
Phase 19 change: the target set is always `new_v2_ids` (not `new_raw_ids`),
consistent with `loop_falsifier`'s `compute_hit_rate(after_topk, new_v2_ids)`.

---

**Semantic search snapshot loop** (`loop_falsifier.py` lines 509-518 and 583-591):
```python
# Before snapshot
before_topk: list[list[str]] = []
for paraphrase in paraphrases:
    hits = semantic_search(paraphrase, k=K)
    before_topk.append([h.place_id for h in hits])

# After snapshot
after_topk: list[list[str]] = []
for paraphrase in paraphrases:
    hits = semantic_search(paraphrase, k=K)
    after_topk.append([h.place_id for h in hits])
```
**D-07 assertion must precede this:** after `get_settings.cache_clear()`,
assert `get_settings().embedding_table == 'place_embeddings_v2'` — `semantic_search`
picks its view from settings; a mismatch scores against the wrong view.

---

**`log_to_mlflow` pattern — artifacts BEFORE metrics** (`loop_falsifier.py` lines 335-385):
```python
def log_to_mlflow(*, gap, seed_query, paraphrases, before_snapshot, after_snapshot,
                  db_diff_ids, before_hit_rate, after_hit_rate, hit_rate_delta,
                  new_place_count, embed_added_count) -> None:
    try:
        mlflow.set_experiment("coverage_agent")
        with mlflow.start_run(run_name=f"loop-falsifier-{gap[0]}-{gap[1]}"):
            # params first
            mlflow.log_param("gap_neighborhood", gap[0])
            mlflow.log_param("gap_cuisine", gap[1])
            mlflow.log_param("seed_query", seed_query)
            mlflow.log_param("k", K)
            mlflow.log_param("n", N)

            # IN-05: artifacts BEFORE metrics — a partial failure leaves no orphan metrics
            mlflow.log_dict({"paraphrases": paraphrases, "seed_query": seed_query},
                            "frozen_paraphrases.json")
            mlflow.log_dict(before_snapshot, "before_snapshot.json")
            mlflow.log_dict(after_snapshot, "after_snapshot.json")
            mlflow.log_dict({"place_ids": db_diff_ids}, "db_diff_place_ids.json")

            # metrics last
            mlflow.log_metric("before_hit_rate", before_hit_rate)
            mlflow.log_metric("after_hit_rate", after_hit_rate)
            mlflow.log_metric("hit_rate_delta", hit_rate_delta)
            mlflow.log_metric("new_place_count", new_place_count)
            mlflow.log_metric("embed_added_count", embed_added_count)
    except Exception as exc:
        print(f"[INFRA] MLflow logging failed: {exc}. ...", file=sys.stderr)
        raise SystemExit(EXIT_INFRA) from exc
```
Phase 19 `loop_runner.log_to_mlflow` adds: `before_hit_at_k`, `after_hit_at_k`,
`recall_at_k`, `hit_rate_delta`, `floor_value`, `fixture_mode` (bool param),
and artifacts for `frozen_paraphrases_runner.json` (per-run generated, not the
committed `falsifier_paraphrases.json`) and `demand_gaps.json`.
Run name convention (D-01 Claude's Discretion): `loop-runner-{neighborhood}-{cuisine}`.

---

**Gap handoff from proposals table (D-08 deterministic one-gap contract)**
Phase 19 specific — no exact analog in `loop_falsifier.py` (it hardcodes `GAP`).
Pattern assembles from: `coverage_agent.py::gap_mine_main` (lines 799-913),
`loop_falsifier.py::premark_seed_isolation` (lines 173-282), and D-08 decisions.

Key SQL seam to copy from `coverage_agent.py` (lines 883-895):
```python
# Snapshot pending query_text BEFORE gap-mine
with get_conn() as write_conn:
    with write_conn.cursor() as cur:
        cur.execute(
            "SELECT query_text FROM places_ingest_query_proposals WHERE status = 'pending'"
        )
        pending_before: set[str] = {row[0] for row in cur.fetchall()}
```
Then call `gap_mine_main(["--top-n", "1"])`, re-query pending, compute
`new = pending_after - pending_before` (D-08 set-diff, NOT `created_at` ordering).

Clear-stale-pending pattern (from `premark_seed_isolation` step 5, lines 243-251):
```python
reject_stale_sql = """
    UPDATE places_ingest_query_proposals
    SET status = 'rejected'
    WHERE status = 'pending' AND query_text != %s
"""
with conn.cursor() as cur:
    cur.execute(reject_stale_sql, [chosen_seed_query])
conn.commit()
```

---

**`decide_exit` gate function — pure, unit-testable** (`loop_falsifier.py` lines 302-332):
```python
def decide_exit(
    before_rate: float,
    after_rate: float,
    guard_violation: GuardResult | None,
    embed_added_count: int,
) -> int:
    if guard_violation is not None and not guard_violation.ok:
        return EXIT_INFRA
    if embed_added_count == 0:
        return EXIT_INFRA
    if before_rate != 0.0:
        return EXIT_INFRA
    if is_strictly_positive_delta(before_rate, after_rate):
        return EXIT_PASS
    return EXIT_FAIL
```
Phase 19 equivalent adds `floor` and `recall_at_k` parameters. The populated
baseline changes the `before_rate != 0.0` branch to a floor-check
(`after_rate < floor → EXIT_FAIL`) combined with `is_strictly_positive_delta`.
All gate logic stays in `falsifier_core` pure functions so unit tests have
zero API cost (D-06).

---

**Verdict print pattern** (`loop_falsifier.py` lines 636-647):
```python
print(f"\n{'=' * 60}")
if exit_code == EXIT_PASS:
    print(f"loop_falsifier: VERDICT = PASS (hit@{K} delta={hit_rate_delta:+.3f})")
elif exit_code == EXIT_FAIL:
    print(f"loop_falsifier: VERDICT = FAIL ...")
else:
    print(f"loop_falsifier: VERDICT = INFRA ERROR ...")
print(f"{'=' * 60}\n")
raise SystemExit(exit_code)
```

---

### `app/loop/falsifier_core.py` — add `compute_recall_at_k` (utility, transform)

**Analog:** `app/loop/falsifier_core.py::compute_hit_rate` (lines 71-106)

**Existing `compute_hit_rate` signature and body to mirror** (lines 71-106):
```python
def compute_hit_rate(
    per_paraphrase_topk: list[list[str]],
    newly_ingested_ids: set[str],
) -> HitRateResult:
    """Compute hit@K over N held-out paraphrases.

    A paraphrase "hits" iff at least one place_id in its top-K results appears
    in *newly_ingested_ids*.
    """
    n = len(per_paraphrase_topk)
    if n == 0:
        return HitRateResult(hit_count=0, n=0, hit_rate=0.0)

    for topk in per_paraphrase_topk:
        assert len(topk) <= K, (  # noqa: S101
            f"compute_hit_rate received a top-k list of length {len(topk)} > K={K}. ..."
        )

    hit_count = sum(1 for topk in per_paraphrase_topk if set(topk) & newly_ingested_ids)
    return HitRateResult(hit_count=hit_count, n=n, hit_rate=hit_count / n)
```

**New `compute_recall_at_k` to add** — same file, same section, pure + stdlib-only:
```python
@dataclass(frozen=True)
class RecallAtKResult:
    """Result of compute_recall_at_k."""
    found_count: int       # distinct new IDs retrieved across all paraphrases' top-k
    total_count: int       # len(newly_ingested_ids)
    recall: float          # found_count / total_count  (0.0 when total_count == 0)


def compute_recall_at_k(
    per_paraphrase_topk: list[list[str]],
    newly_ingested_ids: set[str],
) -> RecallAtKResult:
    """Compute recall@K: what fraction of new embedded places are found across paraphrases.

    recall@K = #distinct newly_ingested_ids that appear in ANY paraphrase's top-K
               / total len(newly_ingested_ids).

    Args:
        per_paraphrase_topk: One list of place_id strings per paraphrase.
            Each inner list must have at most K entries (same IN-02 assertion as
            compute_hit_rate — single source of truth for retrieval window).
        newly_ingested_ids: Set of place_ids that are newly ingested (DB-diff, D-03).

    Returns:
        RecallAtKResult with found_count, total_count, recall.
        Empty newly_ingested_ids returns found_count=0, total_count=0, recall=0.0.
    """
    total_count = len(newly_ingested_ids)
    if total_count == 0:
        return RecallAtKResult(found_count=0, total_count=0, recall=0.0)

    for topk in per_paraphrase_topk:
        assert len(topk) <= K, (  # noqa: S101
            f"compute_recall_at_k received a top-k list of length {len(topk)} > K={K}. "
            "The caller must pass semantic_search(k=K) results. (IN-02)"
        )

    found: set[str] = set()
    for topk in per_paraphrase_topk:
        found |= set(topk) & newly_ingested_ids
    found_count = len(found)
    return RecallAtKResult(found_count=found_count, total_count=total_count, recall=found_count / total_count)
```

**Module constants to reuse** (`falsifier_core.py` lines 27-41):
```python
K = 5  # Retrieval window for hit@k and recall@k
N = 5  # Number of held-out paraphrases per runner run

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INFRA = 2
```
Phase 19 adds a `FLOOR` constant (runtime-tunable default, env/CLI override per D-05):
```python
#: Default quality floor for the populated-corpus gate (D-05).
#: Runtime-tunable via env LOOP_HIT_RATE_FLOOR or CLI --floor.
#: First run uses strict-positive-delta only; ratchet after observing after_hit@k.
FLOOR: float = 0.0  # updated after calibration run
```

**Existing `is_strictly_positive_delta`** (lines 119-126) — reused unchanged:
```python
def is_strictly_positive_delta(before_hit_rate: float, after_hit_rate: float) -> bool:
    return (after_hit_rate - before_hit_rate) > 0.0
```

**`db_diff`** (lines 134-140) — reused unchanged:
```python
def db_diff(before_ids: set[str], after_ids: set[str]) -> set[str]:
    return after_ids - before_ids
```

**`check_non_circularity`** (lines 298-329) — reused unchanged for per-run
frozen paraphrases. The per-run artifact is the moving-target analog of
`configs/falsifier_paraphrases.json`.

**`check_prod_safety`** (lines 209-295) — reused unchanged.

---

### `Makefile` — `make loop` target (config, request-response)

**Analog:** `Makefile` lines 244-246 (`loop-falsifier`) and lines 53-60 (`sandbox-provision`)

**`loop-falsifier` target pattern** (`Makefile` lines 237-246):
```makefile
# Phase 16 / FALSIFY-01 / D-08/D-09: loop falsifier gate.
# ...
# Requires: SANDBOX_DATABASE_URL, GOOGLE_PLACES_API_KEY, OPENAI_API_KEY exported.
# Exit 0 = PASS; 1 = FAIL (re-scopes milestone); 2 = INFRA error.
.PHONY: loop-falsifier
loop-falsifier: ## FALSIFY-01: loop falsifier gate — ...
	$(POETRY_RUN) python scripts/loop_falsifier.py
```

**`sandbox-provision` env-guard pattern** (`Makefile` lines 53-60):
```makefile
.PHONY: sandbox-provision
sandbox-provision: ## Provision the isolated falsifier sandbox DB (LOOP-00; never prod)
	@[ -n "$${SANDBOX_DATABASE_URL:-}" ] || { \
	  echo "ERROR: SANDBOX_DATABASE_URL is not set."; \
	  echo "  Export it first: export SANDBOX_DATABASE_URL=..."; \
	  exit 1; \
	}
	bash scripts/provision_sandbox.sh
```

**Phase 19 `make loop` target to add** — mirror `loop-falsifier` style, add three-key guard:
```makefile
# Phase 19 / LOOP-01..03 + METRIC: productionized loop runner.
# Stages: clear-stale → gap-mine → paraphrase → before-snapshot →
#         ingest → embed-v2 → DB-diff → after-snapshot → hit@k/recall@k → MLflow.
# Requires: SANDBOX_DATABASE_URL, GOOGLE_PLACES_API_KEY, OPENAI_API_KEY exported.
# Exit 0 = PASS; 1 = FAIL; 2 = INFRA error.
# See docs/loop_runner.md for the full runbook.
.PHONY: loop
loop: ## LOOP-01..03+METRIC: productionized loop runner — full gap-mine→ingest→embed→score cycle (requires SANDBOX_DATABASE_URL + GOOGLE_PLACES_API_KEY + OPENAI_API_KEY)
	@[ -n "$${SANDBOX_DATABASE_URL:-}" ] || { \
	  echo "ERROR: SANDBOX_DATABASE_URL is not set."; \
	  exit 1; \
	}
	@[ -n "$${GOOGLE_PLACES_API_KEY:-}" ] || { \
	  echo "ERROR: GOOGLE_PLACES_API_KEY is not set."; \
	  exit 1; \
	}
	@[ -n "$${OPENAI_API_KEY:-}" ] || { \
	  echo "ERROR: OPENAI_API_KEY is not set."; \
	  exit 1; \
	}
	$(POETRY_RUN) python scripts/loop_runner.py
```

**Phase 19 `sandbox-provision-populated` target** — mirrors `sandbox-provision` but
calls the extended provision script with `--populated` or a separate script:
```makefile
.PHONY: sandbox-provision-populated
sandbox-provision-populated: ## Provision populated-baseline sandbox (D-02; drops+recreates for idempotent reset)
	@[ -n "$${SANDBOX_DATABASE_URL:-}" ] || { \
	  echo "ERROR: SANDBOX_DATABASE_URL is not set."; \
	  exit 1; \
	}
	bash scripts/provision_sandbox.sh --populated
```

---

### `scripts/provision_sandbox.sh` — extend for populated baseline + DROP reset

**Analog:** `scripts/provision_sandbox.sh` (full file, 237 lines)

**Existing DROP+recreate reset comment** (`provision_sandbox.sh` lines 17-20):
```bash
# To reset and reprovision:
#   docker exec city_concierge_db psql -U postgres -d postgres -c 'DROP DATABASE city_concierge_sandbox;'
#   bash scripts/provision_sandbox.sh
```

**Existing prod-safety guard pattern** (`provision_sandbox.sh` lines 36-150):
The script already enforces:
1. `SANDBOX_DATABASE_URL` must be set (`lines 36-41`).
2. `DB_NAME` must contain `_sandbox` suffix (`lines 103-110`).
3. Python `check_prod_safety` guard via `app.loop.falsifier_core` (`lines 115-150`).

**Existing idempotent CREATE DATABASE pattern** (`provision_sandbox.sh` lines 184-196):
```bash
_db_exists=$(
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" 2>/dev/null || echo ""
)

if [[ "${_db_exists}" == "1" ]]; then
  echo "Database '${DB_NAME}' already exists — skipping CREATE DATABASE."
else
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -c "CREATE DATABASE \"${DB_NAME}\";"
fi
```

**Phase 19 DROP+restore reset pattern to add** — controlled by `--populated` flag
or a standalone `--reset` flag. D-02 LOCKED CONSTRAINT: reset MUST drop the DB
and re-run ALL steps (init.sql + alembic + baseline ingest + embed). Clearing
only `places_ingest_query_proposals` is insufficient because `embed_places_pgvector_v2`
skips already-current rows (embed backlog contaminates the after-snapshot diff).

New section to add after the prod-safety guard, when `--populated` or `--reset`:
```bash
# ── DROP+recreate (idempotent reset for the populated baseline) ────────────
if [[ "${RESET_MODE:-}" == "1" ]]; then
  echo "RESET: Dropping and recreating '${DB_NAME}' for a clean baseline..."
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -c "DROP DATABASE IF EXISTS \"${DB_NAME}\";"
  docker exec "${CONTAINER_NAME}" psql -U postgres -d postgres \
    -c "CREATE DATABASE \"${DB_NAME}\";"
  # Re-run init.sql + alembic on the fresh DB (fall through to Steps 3+4 below)
fi
```

**Baseline ingest step to add** (after alembic, for populated mode):
```bash
# ── Step 5: Populate baseline (for populated sandbox — LOOP-01 D-02) ──────
# Ingests the static catalog MINUS all gap-bucket queries for the target gap.
# The premark set (catalog minus seed) is handled by loop_runner.py at run time,
# NOT here — this step only ingest the broad non-gap catalog.
if [[ "${POPULATE_BASELINE:-}" == "1" ]]; then
  echo "Step 5: Ingesting populated baseline (non-gap catalog)..."
  DATABASE_URL="${SANDBOX_DATABASE_URL}" $(POETRY_RUN) python scripts/ingest_places_sf.py
  echo "Step 6: Embedding baseline rows (embed-v2)..."
  DATABASE_URL="${SANDBOX_DATABASE_URL}" $(POETRY_RUN) python -m scripts.embed_places_pgvector_v2
  echo "Populated baseline complete."
fi
```

---

## Shared Patterns

### DATABASE_URL Coercion + lru_cache/Pool Footgun
**Source:** `scripts/loop_falsifier.py` lines 405-466
**Apply to:** `scripts/loop_runner.py` (mandatory, D-07 LOCKED CONSTRAINT)

The exact 4-step sequence MUST be followed in this exact order before ANY
settings-touching code (including `gap_mine_main`, `get_conn`, `semantic_search`):
1. Read `SANDBOX_DATABASE_URL`.
2. Resolve prod URL via `resolve_prod_url()` BEFORE coercion.
3. `os.environ["DATABASE_URL"] = sandbox_url` → `get_settings.cache_clear()` → `close_db_pool()`.
4. Assert `resolve_database_url(os.environ) == sandbox_url` AND
   `get_settings().resolved_database_url == sandbox_url`.

```python
# CORRECT — from loop_falsifier.py lines 426-464
os.environ["DATABASE_URL"] = sandbox_url
from app.config import get_settings        # noqa: PLC0415
from app.db_pool import close_db_pool      # noqa: PLC0415
get_settings.cache_clear()
close_db_pool()

from app.config import resolve_database_url  # noqa: PLC0415
resolved = resolve_database_url(os.environ)
assert_resolved_target(sandbox_url=sandbox_url, resolved_url=resolved)

settings_resolved = get_settings().resolved_database_url
if settings_resolved != sandbox_url:
    print(f"[INFRA] get_settings().resolved_database_url ({settings_resolved!r}) ...",
          file=sys.stderr)
    raise SystemExit(EXIT_INFRA)
```

### Embedding-Table Assertion
**Source:** D-07 LOCKED CONSTRAINT (no existing analog in loop_falsifier.py — Phase 19 net-new)
**Apply to:** `scripts/loop_runner.py`, after cache-clear step

```python
# After get_settings.cache_clear(), assert the embedding table is correct
# BEFORE semantic_search is called — it reads settings.embedding_table at call time.
settings = get_settings()
if settings.embedding_table != "place_embeddings_v2":
    print(
        f"[INFRA] settings.embedding_table={settings.embedding_table!r} != "
        "'place_embeddings_v2'. semantic_search would score against the wrong view.",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_INFRA)
```

### prod-safety Guard (check_prod_safety)
**Source:** `scripts/loop_falsifier.py` lines 415-423; `app/loop/falsifier_core.py` lines 209-295
**Apply to:** `scripts/loop_runner.py`

```python
allow_remote = bool(os.environ.get("SANDBOX_ALLOW_REMOTE"))
safety_result = check_prod_safety(sandbox_url, prod_url, allow_remote=allow_remote)
if not safety_result.ok:
    print(f"[INFRA] Prod-safety guard FAILED: {safety_result.message}", file=sys.stderr)
    raise SystemExit(EXIT_INFRA)
print("[prod-safety] PASS")
```

### assert_sandbox_write_target (proposals-table writes)
**Source:** `scripts/sandbox_guard.py` lines 33-55; `scripts/coverage_agent.py` line 894
**Apply to:** `scripts/loop_runner.py` before any proposal-table mutation (stale-clear step)

```python
from app.db import get_conn                                     # noqa: PLC0415  (get_conn lives in app.db, NOT app.db_pool)
from scripts.sandbox_guard import assert_sandbox_write_target   # noqa: PLC0415
# ...
with get_conn() as write_conn:
    assert_sandbox_write_target(write_conn)
    # ... SQL writes only happen after guard passes
```

### Exit Code Convention (0/1/2)
**Source:** `app/loop/falsifier_core.py` lines 34-41; `scripts/eval_falsifier.py`
**Apply to:** `scripts/loop_runner.py`

```python
EXIT_PASS  = 0  # PASS
EXIT_FAIL  = 1  # FAIL (non-positive delta OR below floor)
EXIT_INFRA = 2  # INFRA (precondition, guard, subprocess, MLflow, empty diff)
```

### MLflow Artifacts-Before-Metrics (IN-05)
**Source:** `scripts/loop_falsifier.py` lines 365-378
**Apply to:** `scripts/loop_runner.py::log_to_mlflow`

Always `mlflow.log_dict(...)` ALL artifact calls BEFORE any `mlflow.log_metric(...)`.
A partial failure on an artifact call then exits EXIT_INFRA without orphaned metrics.

### Subprocess Driver Pattern
**Source:** `scripts/loop_falsifier.py` lines 285-299 (`run_subprocess_or_infra`)
**Apply to:** `scripts/loop_runner.py` for ingest and embed-v2 subprocess calls

```python
def run_subprocess_or_infra(argv: list[str], env: dict[str, str]) -> None:
    try:
        subprocess.run(argv, env=env, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        print(f"[INFRA] Subprocess {argv!r} failed ...", file=sys.stderr)
        raise SystemExit(EXIT_INFRA) from exc
```

### Pure-Core / Operator-Run Split (D-06)
**Source:** Phase 16 convention; `tests/unit/test_loop_falsifier_core.py` (entire file)
**Apply to:** `tests/unit/test_loop_runner_core.py` (new unit test file)

All gate/diff/floor/recall logic lives in `falsifier_core` pure functions.
Unit tests import only from `app.loop.falsifier_core` — zero DB, network, LLM calls.
Orchestrator tests (`test_loop_runner_orchestrator.py`) mock everything external
exactly as `tests/unit/test_loop_falsifier_orchestrator.py` does.

Test structure to copy from `test_loop_falsifier_core.py`:
```python
"""Unit tests for app.loop.falsifier_core — new Phase 19 additions.

Zero DB, network, or LLM calls.
"""
from __future__ import annotations

import pytest
from app.loop.falsifier_core import (
    EXIT_FAIL, EXIT_INFRA, EXIT_PASS,
    K, N,
    RecallAtKResult,      # new dataclass
    compute_recall_at_k,  # new function
    # ... existing imports unchanged
)

class TestComputeRecallAtK:
    def test_all_new_found_across_paraphrases(self) -> None:
        ...
    def test_partial_recall(self) -> None:
        ...
    def test_empty_new_ids_returns_zero(self) -> None:
        ...
    def test_topk_exceeding_k_raises_assertion(self) -> None:
        ...
```

### Frozen Paraphrase Artifact Shape
**Source:** `configs/falsifier_paraphrases.json` (entire file)
**Apply to:** `scripts/loop_runner.py` per-run artifact (D-04)

```json
{
  "seed_query": "...",
  "generation_prompt": "...",
  "non_circularity_note": "...",
  "gap_neighborhood": "...",
  "gap_cuisine": "...",
  "generation_timestamp": "...",
  "generation_model": "...",
  "paraphrases": ["...", "...", "...", "...", "..."]
}
```
The per-run artifact extends the committed `falsifier_paraphrases.json` shape
with `gap_neighborhood`, `gap_cuisine`, `generation_timestamp`, `generation_model`
(all fields that make the per-run gap traceable). It is logged via
`mlflow.log_dict(artifact_data, "frozen_paraphrases_runner.json")` BEFORE ingest.

---

## Read Seams (caller contracts)

| Seam | File | Key contract for loop_runner.py |
|---|---|---|
| `gap_mine_main(argv)` | `scripts/coverage_agent.py` L799 | Returns `int` exit code. Uses `get_conn()` pool. MUST call AFTER sandbox coercion. Accepts `["--top-n", "1"]`. |
| `ingest_places_sf.py` | `scripts/ingest_places_sf.py` | Subprocess only; needs `DATABASE_URL=sandbox` + `GOOGLE_PLACES_API_KEY` in env. Consumes ALL `'pending'` proposals. |
| `embed_places_pgvector_v2.py` | `scripts/embed_places_pgvector_v2.py` | Subprocess as `python -m scripts.embed_places_pgvector_v2`. Raises on error (check=True catches). Skips already-current rows. |
| `assert_sandbox_write_target(conn)` | `scripts/sandbox_guard.py` L33 | Must pass the SAME conn used for writes. Raises RuntimeError (not SystemExit) on violation. |
| `semantic_search(query, k=K)` | `app/tools/retrieval.py` L72 | Sync, returns `list[PlaceHit]`. Reads view chosen by `settings.embedding_table`. Must call AFTER embedding-table assertion. |
| `get_conn()` / `close_db_pool()` | `app/db.py` L37 / `app/db_pool.py` L83 | `get_conn()` (a contextmanager in `app.db`) lazily inits the pool from cached settings; `close_db_pool()` (in `app.db_pool`) is no-op when pool is None — safe unconditionally. NB: `app/db_pool.py` exposes `get_connection()` (L55), NOT `get_conn`. Import `get_conn` from `app.db`, `close_db_pool` from `app.db_pool`. |
| `resolve_database_url(env)` | `app/config.py` L60 | Precedence: `DATABASE_URL` > `POSTGRES_*`. Returns `str | None`. |
| `get_settings()` | `app/config.py` L155 | `@lru_cache` — MUST `cache_clear()` after `os.environ["DATABASE_URL"] = sandbox_url`. |

---

## No Analog Found

No files in this phase lack an analog. All patterns are drawn from the live codebase.

---

## Metadata

**Analog search scope:** `scripts/`, `app/loop/`, `app/config.py`, `app/db_pool.py`,
  `app/tools/retrieval.py`, `scripts/sandbox_guard.py`, `Makefile`,
  `tests/unit/test_loop_falsifier_core.py`, `tests/unit/test_loop_falsifier_orchestrator.py`
**Files scanned:** 12
**Pattern extraction date:** 2026-06-20
