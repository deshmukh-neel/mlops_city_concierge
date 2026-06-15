#!/usr/bin/env python3
"""Loop falsifier orchestrator (FALSIFY-01 / D-08).

Takes ONE real coverage gap end-to-end:
  freeze-check -> prod-safety guard -> seed-isolation pre-mark ->
  before-snapshot -> real Google Places ingest -> embed-v2 ->
  DB-diff -> after-snapshot -> strictly-positive hit@k gate ->
  MLflow log -> exit 0/1/2

Exit codes (mirrors scripts/eval_falsifier.py, D-09):
  0 = PASS  — strictly-positive before->after hit@k delta
  1 = FAIL  — non-positive delta (expected falsifier outcome; re-scopes milestone)
  2 = INFRA — precondition error (sandbox unset/unsafe, guard violation,
              embed produced nothing, non-circularity, paraphrase file issue,
              subprocess failure, MLflow failure)

Usage:
  SANDBOX_DATABASE_URL=... GOOGLE_PLACES_API_KEY=... OPENAI_API_KEY=... \\
    python scripts/loop_falsifier.py
  # or:
  make loop-falsifier

Design notes (D-08 through D-12):
- All in-process semantic_search / app.* imports are DEFERRED inside main()
  (after os.environ["DATABASE_URL"] = sandbox_url) to avoid the lru_cache
  footgun in get_settings() which fires at module scope in app/config.py.
- Prod URL is resolved BEFORE the DATABASE_URL coercion, using
  {**dotenv_values(".env"), **os.environ} so prod sitting unexported in .env
  is still compared (Codex MEDIUM fail-open fix).
- Ingest + embed run as subprocesses with DATABASE_URL=sandbox via child_env
  (D-10: zero changes to ingest/embed/retrieval scripts).
- Only app.loop.falsifier_core (stdlib-only, settings-free) is imported at
  module scope; all other app.* imports stay inside main()/helpers.
"""

from __future__ import annotations

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
    build_premark_set,
    check_non_circularity,
    check_prod_safety,
    compute_hit_rate,
    db_diff,
    is_strictly_positive_delta,
)

# ---------------------------------------------------------------------------
# Gap constant — ONE swappable place (D-01/D-02)
# Concrete value chosen after a place-count smoke test; swap in one line.
# ---------------------------------------------------------------------------

GAP = ("Outer Sunset", "vietnamese")  # (neighborhood, cuisine)
SEED_QUERY = f"{GAP[1]} restaurants in {GAP[0]} San Francisco"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PARAPHRASE_FILE = "configs/falsifier_paraphrases.json"

# ---------------------------------------------------------------------------
# Public helpers (also called in unit tests)
# ---------------------------------------------------------------------------


def load_paraphrases(path: str = PARAPHRASE_FILE) -> tuple[list[str], str]:
    """Read the frozen paraphrase file.

    Returns (paraphrases, seed_query).  Exits EXIT_INFRA if the file is
    missing, unparseable, or does not contain exactly N paraphrases.
    The falsifier NEVER regenerates paraphrases at gate time (D-06).
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        print(f"[INFRA] Paraphrase file not found: {path!r}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA) from exc
    except json.JSONDecodeError as exc:
        print(f"[INFRA] Paraphrase file is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA) from exc

    paraphrases = data.get("paraphrases", [])
    if len(paraphrases) != N:
        print(
            f"[INFRA] Paraphrase file must contain exactly N={N} entries; "
            f"found {len(paraphrases)} in {path!r}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    seed_query = data.get("seed_query", "")
    return list(paraphrases), seed_query


def run_guards(
    sandbox_url: str | None,
    prod_url: str | None,
    paraphrases: list[str],
    seed_query: str,
) -> GuardResult:
    """Run prod-safety + non-circularity guards.

    Returns the first violation found, or ok=True.
    """
    safety = check_prod_safety(sandbox_url, prod_url)
    if not safety.ok:
        return safety

    circularity = check_non_circularity(paraphrases, [seed_query])
    if not circularity.ok:
        return circularity

    return GuardResult(ok=True, message="ok")


def assert_resolved_target(sandbox_url: str, resolved_url: str | None) -> None:
    """Assert the in-process resolved DB target equals the sandbox URL.

    This is the belt-and-suspenders W1 check: after setting
    os.environ["DATABASE_URL"] = sandbox_url and importing settings,
    confirm the lru_cache resolved to the sandbox — not a stale prod target.
    Exits EXIT_INFRA(2) if they differ.
    """
    if resolved_url != sandbox_url:
        print(
            f"[INFRA] Resolved in-process DATABASE_URL ({resolved_url!r}) does not match "
            f"SANDBOX_DATABASE_URL ({sandbox_url!r}). "
            "The settings lru_cache may have been populated before the sandbox injection. "
            "Ensure all app.* imports that touch settings are deferred inside main().",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)


def resolve_prod_url(sandbox_url: str | None) -> str | None:
    """Resolve the prod DATABASE_URL from {**dotenv_values(".env"), **os.environ}.

    The sandbox key(s) are popped from the merged copy so the result is
    the REAL prod target even when it lives unexported in .env (Codex MEDIUM).

    This MUST be called BEFORE os.environ["DATABASE_URL"] = sandbox_url.
    """
    # Merge: .env file first, then os.environ overrides (same order as load_dotenv)
    merged: dict[str, str | None] = {**dotenv_values(".env"), **os.environ}

    # Pop the sandbox override keys so we resolve the real prod URL
    merged.pop("DATABASE_URL", None)
    merged.pop("SANDBOX_DATABASE_URL", None)

    # Use the same resolver as app/config.py
    from app.config import resolve_database_url  # noqa: PLC0415 — deferred (settings-touching)

    return resolve_database_url(merged)


def premark_seed_isolation(
    conn: Any,
    chosen_seed_query: str,
) -> None:
    """Pre-mark the static catalog so ingest runs ONLY the chosen seed gap.

    Implements the BLOCKING Codex HIGH seed-isolation fix:
    1. Import build_seed_queries from scripts.ingest_places_sf.
    2. Validate chosen_seed_query IS in the catalog (else EXIT_INFRA).
    3. UPSERT every catalog entry EXCEPT chosen_seed_query as 'completed' in
       places_ingest_query_checkpoints — the SKIP_COMPLETED_QUERIES path then
       skips them, leaving only the gap query (~1 paid call).
    4. Insert SEED_QUERY as a pending proposal in places_ingest_query_proposals
       (ON CONFLICT DO NOTHING — idempotent).
    5. UPDATE every OTHER pending row in proposals to status='rejected'
       (Codex round-2 NEW #2 — clears stale proposals from reused sandboxes
       so fetch_pending_proposals does not re-pollute the DB-diff).
    6. ASSERT zero non-SEED_QUERY pending rows remain (EXIT_INFRA if any).

    All writes target the sandbox conn — never prod.
    """
    # Deferred import to avoid module-scope side-effect (ingest resolves DATABASE_URL
    # at import time; the sandbox injection must have happened BEFORE this call)
    from scripts.ingest_places_sf import build_seed_queries  # noqa: PLC0415

    catalog = build_seed_queries()
    if chosen_seed_query not in set(catalog):
        print(
            f"[INFRA] SEED_QUERY {chosen_seed_query!r} is not in build_seed_queries() catalog. "
            "A typo'd seed would silently let the whole catalog through — aborting.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    to_mark = build_premark_set(catalog, chosen_seed_query)

    # Step 3: UPSERT catalog-minus-seed as 'completed' in checkpoints table
    upsert_sql = """
        INSERT INTO places_ingest_query_checkpoints (query_text, status)
        VALUES (%s, 'completed')
        ON CONFLICT (query_text) DO UPDATE SET status = 'completed'
    """
    with conn.cursor() as cur:
        for query_text in to_mark:
            cur.execute(upsert_sql, [query_text])
    conn.commit()

    # Step 4: Insert SEED_QUERY as pending proposal (idempotent)
    insert_pending_sql = """
        INSERT INTO places_ingest_query_proposals (query_text, status, rationale)
        VALUES (%s, 'pending', 'loop falsifier seed isolation — gap query')
        ON CONFLICT (query_text) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(insert_pending_sql, [chosen_seed_query])
    conn.commit()

    # Step 5: Clear stale pending proposals != SEED_QUERY (Codex round-2 NEW #2)
    # NOTE: 'rejected' is the valid CHECK constraint value (not 'skipped')
    reject_stale_sql = """
        UPDATE places_ingest_query_proposals
        SET status = 'rejected'
        WHERE status = 'pending' AND query_text != %s
    """
    with conn.cursor() as cur:
        cur.execute(reject_stale_sql, [chosen_seed_query])
    conn.commit()

    # Step 6: Assert zero stale pending rows remain
    count_stale_sql = """
        SELECT count(*) FROM places_ingest_query_proposals
        WHERE status = 'pending' AND query_text != %s
    """
    with conn.cursor() as cur:
        cur.execute(count_stale_sql, [chosen_seed_query])
        stale_count = cur.fetchone()[0]

    if stale_count != 0:
        # Fetch offending rows for the error message
        with conn.cursor() as cur:
            cur.execute(
                "SELECT query_text FROM places_ingest_query_proposals "
                "WHERE status = 'pending' AND query_text != %s "
                "LIMIT 5",
                [chosen_seed_query],
            )
            offenders = [row[0] for row in cur.fetchall()]
        print(
            f"[INFRA] {stale_count} stale pending proposals remain after clearing; "
            f"first offenders: {offenders!r}. Cannot guarantee gap-specific DB-diff.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    print(
        f"[seed-isolation] Pre-marked {len(to_mark)} catalog queries as completed; "
        f"inserted {chosen_seed_query!r} as the only pending proposal."
    )


def run_subprocess_or_infra(argv: list[str], env: dict[str, str]) -> None:
    """Run a subprocess. A non-zero exit maps to EXIT_INFRA(2), not a silent FAIL.

    Uses sys.executable for the python interpreter (Codex LOW — not bare 'python').
    check=True raises CalledProcessError on non-zero exit.
    """
    try:
        subprocess.run(argv, env=env, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        print(
            f"[INFRA] Subprocess {argv!r} failed with exit code {exc.returncode}. "
            "Ingest/embed failure is an infrastructure error, not a gate FAIL.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA) from exc


def decide_exit(
    before_rate: float,
    after_rate: float,
    guard_violation: GuardResult | None,
    embed_added_count: int,
) -> int:
    """Return the appropriate exit code for the gate result.

    Priority order (highest to lowest):
    1. Guard violation -> EXIT_INFRA
    2. embed_added_count == 0 -> EXIT_INFRA (LOOP-02 loud-fail)
    3. before_rate != 0.0 -> EXIT_INFRA (sandbox was not clean)
    4. is_strictly_positive_delta(before, after) -> EXIT_PASS
    5. else -> EXIT_FAIL
    """
    if guard_violation is not None and not guard_violation.ok:
        return EXIT_INFRA
    if embed_added_count == 0:
        return EXIT_INFRA
    if before_rate != 0.0:
        return EXIT_INFRA
    if is_strictly_positive_delta(before_rate, after_rate):
        return EXIT_PASS
    return EXIT_FAIL


def _snapshot_ids(conn: Any, table: str, id_col: str = "place_id") -> set[str]:
    """Capture a set of place_ids from the given table via a direct psycopg2 conn."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {id_col} FROM {table}")  # noqa: S608
        return {row[0] for row in cur.fetchall()}


def log_to_mlflow(
    *,
    gap: tuple[str, str],
    seed_query: str,
    paraphrases: list[str],
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    db_diff_ids: list[str],
    before_hit_rate: float,
    after_hit_rate: float,
    hit_rate_delta: float,
    new_place_count: int,
    embed_added_count: int,
) -> None:
    """Log all artifacts + metrics to MLflow under 'coverage_agent' experiment.

    A logging failure exits EXIT_INFRA(2) — durable artifacts are part of
    FALSIFY-01(e) and a silent pass would hide the missing evidence.
    """
    try:
        mlflow.set_experiment("coverage_agent")
        with mlflow.start_run(run_name=f"loop-falsifier-{gap[0]}-{gap[1]}"):
            mlflow.log_param("gap_neighborhood", gap[0])
            mlflow.log_param("gap_cuisine", gap[1])
            mlflow.log_param("seed_query", seed_query)
            mlflow.log_param("k", K)
            mlflow.log_param("n", N)

            mlflow.log_metric("before_hit_rate", before_hit_rate)
            mlflow.log_metric("after_hit_rate", after_hit_rate)
            mlflow.log_metric("hit_rate_delta", hit_rate_delta)
            mlflow.log_metric("new_place_count", new_place_count)
            mlflow.log_metric("embed_added_count", embed_added_count)

            mlflow.log_dict(
                {"paraphrases": paraphrases, "seed_query": seed_query}, "frozen_paraphrases.json"
            )
            mlflow.log_dict(before_snapshot, "before_snapshot.json")
            mlflow.log_dict(after_snapshot, "after_snapshot.json")
            mlflow.log_dict({"place_ids": db_diff_ids}, "db_diff_place_ids.json")
    except Exception as exc:
        print(
            f"[INFRA] MLflow logging failed: {exc}. "
            "Durable artifacts are required by FALSIFY-01(e).",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA) from exc


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    """Run the full loop falsifier gate sequence.

    CANONICAL ORDER (interdependent — do not reorder):
    1. Read SANDBOX_DATABASE_URL; resolve prod_url BEFORE coercing DATABASE_URL.
    2. Run prod-safety guard — EXIT_INFRA if violated.
    3. Coerce os.environ["DATABASE_URL"] = sandbox_url (routes in-process probes).
    4. W1 resolved-target assertion: confirm in-process target == sandbox.
    5. Deferred imports of app.tools.retrieval / any settings-touching app.*.
    6. premark_seed_isolation on sandbox conn.
    7. before-snapshot -> ingest -> embed -> DB-diff -> after-snapshot -> gate.
    """
    # ── Step 1: Resolve sandbox URL + prod URL BEFORE coercion ──────────────
    sandbox_url = os.environ.get("SANDBOX_DATABASE_URL")
    if not sandbox_url:
        print("[INFRA] SANDBOX_DATABASE_URL is not set. Cannot proceed.", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)

    # Resolve prod URL with .env MERGED (Codex MEDIUM — prod may be unexported)
    # resolve_prod_url pops DATABASE_URL + SANDBOX_DATABASE_URL from the merged copy
    # so it returns the real prod target.
    prod_url = resolve_prod_url(sandbox_url=sandbox_url)

    # ── Step 2: Prod-safety guard BEFORE any coercion or destructive op ─────
    safety_result = check_prod_safety(sandbox_url, prod_url)
    if not safety_result.ok:
        print(f"[INFRA] Prod-safety guard FAILED: {safety_result.message}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    print("[prod-safety] PASS")

    # ── Step 3: Coerce DATABASE_URL to sandbox ──────────────────────────────
    os.environ["DATABASE_URL"] = sandbox_url

    # ── Step 4: W1 resolved-target assertion (belt-and-suspenders) ──────────
    # Import AFTER coercion to avoid lru_cache footgun
    from app.config import resolve_database_url  # noqa: PLC0415

    resolved = resolve_database_url(os.environ)
    assert_resolved_target(sandbox_url=sandbox_url, resolved_url=resolved)
    print(f"[resolved-target] in-process target confirmed: {resolved!r}")

    # ── Step 5: Deferred imports (all settings-touching app.* imports) ──────
    from app.tools.retrieval import semantic_search  # noqa: PLC0415

    # ── Load + validate paraphrases ─────────────────────────────────────────
    paraphrases, frozen_seed_query = load_paraphrases()

    # ── Step 2b: Non-circularity guard ──────────────────────────────────────
    non_circ = check_non_circularity(paraphrases, [SEED_QUERY, frozen_seed_query])
    if not non_circ.ok:
        print(f"[INFRA] Non-circularity guard FAILED: {non_circ.message}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    print("[non-circularity] PASS")

    # ── Step 6: Seed-isolation pre-mark ─────────────────────────────────────
    import psycopg2  # noqa: PLC0415

    print("[seed-isolation] Pre-marking static catalog in sandbox ...")
    with psycopg2.connect(sandbox_url) as sandbox_conn:
        premark_seed_isolation(sandbox_conn, SEED_QUERY)

    # ── Step 7a: Capture before-snapshot (empty sandbox -> hit_rate must be 0) ──
    print(f"[before-snapshot] Probing {N} paraphrases with k={K} ...")
    before_raw_ids_result = _snapshot_ids_from_url(sandbox_url, "places_raw")
    before_v2_ids_result = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")

    before_topk: list[list[str]] = []
    for paraphrase in paraphrases:
        hits = semantic_search(paraphrase, k=K)
        before_topk.append([h.place_id for h in hits])

    # With empty sandbox, target set is empty -> hit_rate is 0.0
    before_hit_result = compute_hit_rate(before_topk, set())
    before_hit_rate = before_hit_result.hit_rate

    if before_hit_rate != 0.0:
        print(
            f"[INFRA] Before-snapshot hit_rate={before_hit_rate:.3f} != 0.0. "
            "Sandbox is not clean — use `make sandbox-provision` to reset.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)
    print(
        f"[before-snapshot] hit@{K} = {before_hit_result.hit_count}/{before_hit_result.n} = {before_hit_rate:.3f} (expected 0.0 for empty sandbox)"
    )

    # ── Step 7b: Run ingest subprocess ──────────────────────────────────────
    child_env = {**os.environ, "DATABASE_URL": sandbox_url}
    print(f"[ingest] Running ingest for SEED_QUERY={SEED_QUERY!r} ...")
    run_subprocess_or_infra(
        argv=[sys.executable, "scripts/ingest_places_sf.py"],
        env=child_env,
    )
    print("[ingest] Done.")

    # ── Step 7c: Run embed-v2 subprocess ─────────────────────────────────────
    print("[embed-v2] Running embed-v2 ...")
    run_subprocess_or_infra(
        argv=[sys.executable, "-m", "scripts.embed_places_pgvector_v2"],
        env=child_env,
    )
    print("[embed-v2] Done.")

    # ── Step 7d: Compute DB-diff ─────────────────────────────────────────────
    after_raw_ids_result = _snapshot_ids_from_url(sandbox_url, "places_raw")
    after_v2_ids_result = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")

    new_place_ids = db_diff(before_raw_ids_result, after_raw_ids_result)
    new_v2_ids = db_diff(before_v2_ids_result, after_v2_ids_result)

    new_place_count = len(new_place_ids)
    embed_added_count = len(new_v2_ids)

    print(f"[db-diff] new places_raw rows: {new_place_count}")
    print(f"[db-diff] new place_embeddings_v2 rows: {embed_added_count}")

    if embed_added_count == 0:
        print(
            "[INFRA] embed-v2 produced 0 new embeddings. "
            "The ingest may have found no new places, or embed-v2 silently skipped all. "
            "Exit EXIT_INFRA — this is the LOOP-02 loud-fail.",
            file=sys.stderr,
        )
        # Will be caught by decide_exit below, but print early for clarity

    # ── Step 7e: After-snapshot ───────────────────────────────────────────────
    print(
        f"[after-snapshot] Probing {N} paraphrases with k={K} against {new_place_count} new places ..."
    )
    after_topk: list[list[str]] = []
    for paraphrase in paraphrases:
        hits = semantic_search(paraphrase, k=K)
        after_topk.append([h.place_id for h in hits])

    # Target set = newly-ingested + embedded place_ids (from DB-diff on v2 table)
    after_hit_result = compute_hit_rate(after_topk, new_v2_ids)
    after_hit_rate = after_hit_result.hit_rate

    hit_rate_delta = after_hit_rate - before_hit_rate
    print(
        f"[after-snapshot] hit@{K} = {after_hit_result.hit_count}/{after_hit_result.n} = {after_hit_rate:.3f}"
    )
    print(
        f"[gate] delta = {hit_rate_delta:+.3f} (before={before_hit_rate:.3f}, after={after_hit_rate:.3f})"
    )

    # ── Step 7f: Gate decision ────────────────────────────────────────────────
    exit_code = decide_exit(
        before_rate=before_hit_rate,
        after_rate=after_hit_rate,
        guard_violation=None,
        embed_added_count=embed_added_count,
    )

    # ── Step 7g: MLflow logging ───────────────────────────────────────────────
    before_snapshot = {
        "paraphrase_topk": before_topk,
        "hit_rate": before_hit_rate,
        "hit_count": before_hit_result.hit_count,
        "n": before_hit_result.n,
    }
    after_snapshot = {
        "paraphrase_topk": after_topk,
        "hit_rate": after_hit_rate,
        "hit_count": after_hit_result.hit_count,
        "n": after_hit_result.n,
    }
    log_to_mlflow(
        gap=GAP,
        seed_query=SEED_QUERY,
        paraphrases=paraphrases,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        db_diff_ids=sorted(new_place_ids),
        before_hit_rate=before_hit_rate,
        after_hit_rate=after_hit_rate,
        hit_rate_delta=hit_rate_delta,
        new_place_count=new_place_count,
        embed_added_count=embed_added_count,
    )

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if exit_code == EXIT_PASS:
        print(f"loop_falsifier: VERDICT = PASS (hit@{K} delta={hit_rate_delta:+.3f})")
    elif exit_code == EXIT_FAIL:
        print(
            f"loop_falsifier: VERDICT = FAIL (hit@{K} delta={hit_rate_delta:+.3f}; re-scopes milestone)"
        )
    else:
        print(f"loop_falsifier: VERDICT = INFRA ERROR (embed_added_count={embed_added_count})")
    print(f"{'=' * 60}\n")

    raise SystemExit(exit_code)


def _snapshot_ids_from_url(db_url: str, table: str) -> set[str]:
    """Capture a set of place_ids from a table using a direct psycopg2 connection.

    Uses a direct connection (not the pool) to avoid dependency on the
    pool's cached settings (which may point to a different target).
    """
    import psycopg2  # noqa: PLC0415

    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT place_id FROM {table}")  # noqa: S608
        return {row[0] for row in cur.fetchall()}


if __name__ == "__main__":
    main()
