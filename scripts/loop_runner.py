#!/usr/bin/env python3
"""Productionized loop runner orchestrator (Phase 19 / LOOP-01..03 + METRIC).

Takes ONE demand-driven gap end-to-end:
  coerce sandbox → cache_clear → close_pool → assert-sandbox →
  embedding-table assert → clear-stale-pending → gap-mine → set-diff →
  LLM-generate + freeze paraphrases → before-snapshot → ingest →
  embed-v2 → DB-diff (places_raw guard + v2 guard) → after-snapshot →
  hit@k / recall@k → floor gate → MLflow → exit 0/1/2

Exit codes (mirrors loop_falsifier.py, D-09):
  0 = PASS  — strictly-positive before→after hit@k delta AND after >= floor
  1 = FAIL  — delta non-positive or after below floor
  2 = INFRA — precondition error (sandbox unset/unsafe, guard violation,
              empty v2-diff, zero new raw rows, non-circularity, MLflow failure,
              subprocess failure, gap-handoff mismatch)

Usage:
  SANDBOX_DATABASE_URL=... GOOGLE_PLACES_API_KEY=... OPENAI_API_KEY=... \\
    python scripts/loop_runner.py
  # or:
  make loop

Design notes (D-07, D-08 LOCKED CONSTRAINTS):
- Module scope imports ONLY stdlib + mlflow + dotenv + falsifier_core (which is
  stdlib-only). ALL settings-touching app.* imports are deferred inside main()
  AFTER the sandbox coercion/cache_clear sequence (D-07 coercion-ordering).
- gap_mine_main is imported AFTER the sandbox coercion step (D-07 ordering).
- Target set = place_embeddings_v2 DB-diff (new embedded IDs), NOT places_raw.
  before_hit@k = 0.0 by construction since new IDs did not exist before (D-03).
- N paraphrases are LLM-generated and durably frozen to disk BEFORE ingest (D-04).
- loop_falsifier.py and its hardcoded GAP are untouched (D-07).
- fixture_mode flag: True when DEMAND_DATABASE_URL is unset (seeded-sandbox
  fixture demand), False when DEMAND_DATABASE_URL is set (real demand) (D-01).
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import subprocess
import sys
from typing import Any

import mlflow
from dotenv import dotenv_values

from app.loop.falsifier_core import (
    EXIT_FAIL,  # noqa: F401
    EXIT_INFRA,
    EXIT_PASS,
    FLOOR,
    K,
    N,
    check_non_circularity,
    check_prod_safety,
    compute_hit_rate,
    compute_recall_at_k,
    db_diff,
    decide_loop_exit,
    is_strictly_positive_delta,  # noqa: F401
)

# ---------------------------------------------------------------------------
# Artifact directory for per-run frozen files
# ---------------------------------------------------------------------------

RUNNER_ARTIFACT_DIR = "loop_runner_artifacts"

# ---------------------------------------------------------------------------
# Paraphrase generation prompt template (matches falsifier_paraphrases.json shape)
# ---------------------------------------------------------------------------

_PARAPHRASE_GENERATION_PROMPT = (
    "Generate exactly {n} distinct rephrasings of the search intent: {seed_query!r}. "
    "Each rephrasing must express the same intent (finding {cuisine} food in the "
    "{neighborhood} neighborhood of SF) but use different words and phrasing. "
    "Do NOT repeat the original string. "
    "Return only the {n} strings as a JSON array."
)

_NON_CIRCULARITY_NOTE = (
    "Paraphrases differ from the literal seed query string (exact-string check "
    "enforced at gate time). Expected place_ids are post-ingest DB-diff rows — "
    "they did not exist before the ingest, so the metric cannot be gamed by "
    "pre-existing data."
)

# ---------------------------------------------------------------------------
# Helpers copied verbatim from loop_falsifier.py (D-07: do NOT import from it)
# ---------------------------------------------------------------------------


def resolve_prod_url(sandbox_url: str | None) -> str | None:
    """Resolve the prod DATABASE_URL from {**dotenv_values(".env"), **os.environ}.

    The sandbox key(s) are popped from the merged copy so the result is
    the REAL prod target even when it lives unexported in .env (Codex MEDIUM).

    This MUST be called BEFORE os.environ["DATABASE_URL"] = sandbox_url.
    """
    merged: dict[str, str | None] = {**dotenv_values(".env"), **os.environ}
    merged.pop("DATABASE_URL", None)
    merged.pop("SANDBOX_DATABASE_URL", None)

    from app.config import resolve_database_url  # noqa: PLC0415 — deferred

    return resolve_database_url(merged)


def assert_resolved_target(sandbox_url: str, resolved_url: str | None) -> None:
    """Assert the in-process resolved DB target equals the sandbox URL.

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


def run_subprocess_or_infra(argv: list[str], env: dict[str, str]) -> None:
    """Run a subprocess. A non-zero exit maps to EXIT_INFRA(2), not a silent FAIL."""
    try:
        subprocess.run(argv, env=env, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        print(
            f"[INFRA] Subprocess {argv!r} failed with exit code {exc.returncode}. "
            "Ingest/embed failure is an infrastructure error, not a gate FAIL.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA) from exc


def _snapshot_ids_from_url(db_url: str, table: str) -> set[str]:
    """Capture a set of place_ids from a table using a direct psycopg2 connection.

    Uses a direct connection (not the pool) to avoid dependency on the
    pool's cached settings (which may point to a different target).
    """
    import psycopg2  # noqa: PLC0415

    with contextlib.closing(psycopg2.connect(db_url)) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT place_id FROM {table}")  # noqa: S608
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# LLM paraphrase generation
# ---------------------------------------------------------------------------


def _generate_paraphrases(
    seed_query: str,
    neighborhood: str,
    cuisine: str,
    n: int = N,
) -> tuple[list[str], str, str]:
    """Generate N paraphrases of the gap intent using make_judge / build_chat_model.

    Returns (paraphrases, generation_prompt, generation_model).
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    from app.agent.critique.vibe import (  # noqa: PLC0415
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
    )
    from app.llm_factory import build_chat_model  # noqa: PLC0415

    provider = os.environ.get("PARAPHRASE_PROVIDER", DEFAULT_JUDGE_PROVIDER)
    model_name = os.environ.get("PARAPHRASE_MODEL", DEFAULT_JUDGE_MODEL)

    generation_prompt = _PARAPHRASE_GENERATION_PROMPT.format(
        n=n,
        seed_query=seed_query,
        cuisine=cuisine,
        neighborhood=neighborhood,
    )

    llm = build_chat_model(provider, model_name, temperature=1.0)
    raw = llm.invoke([HumanMessage(content=generation_prompt)]).content
    if not isinstance(raw, str):
        print(
            f"[INFRA] Paraphrase generation returned non-string content: {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    # Strip markdown code fences if present
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first and last fence lines
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        stripped = "\n".join(inner).strip()

    try:
        paraphrases: list[str] = json.loads(stripped)
    except json.JSONDecodeError as exc:
        print(
            f"[INFRA] Paraphrase generation returned unparseable JSON: {exc}\nRaw: {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA) from exc

    if not isinstance(paraphrases, list) or len(paraphrases) != n:
        print(
            f"[INFRA] Paraphrase generation must return exactly {n} strings; "
            f"got {len(paraphrases) if isinstance(paraphrases, list) else type(paraphrases)}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    return paraphrases, generation_prompt, model_name


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_to_mlflow(
    *,
    neighborhood: str,
    cuisine: str,
    seed_query: str,
    paraphrases: list[str],
    frozen_artifact: dict[str, Any],
    frozen_artifact_path: str,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    new_v2_ids: set[str],
    before_hit_at_k: float,
    after_hit_at_k: float,
    hit_rate_delta: float,
    recall_at_k: float,
    new_place_count: int,
    embed_added_count: int,
    floor: float,
    fixture_mode: bool,
) -> None:
    """Log all artifacts + metrics to MLflow under 'coverage_agent' experiment.

    IN-05: ALL log_dict artifact calls BEFORE any log_metric calls.
    A logging failure exits EXIT_INFRA — durable artifacts are required.
    """
    try:
        mlflow.set_experiment("coverage_agent")
        run_name = f"loop-runner-{neighborhood}-{cuisine}"
        with mlflow.start_run(run_name=run_name):
            # params first
            mlflow.log_param("gap_neighborhood", neighborhood)
            mlflow.log_param("gap_cuisine", cuisine)
            mlflow.log_param("seed_query", seed_query)
            mlflow.log_param("k", K)
            mlflow.log_param("n", N)
            mlflow.log_param("floor", floor)
            mlflow.log_param("fixture_mode", fixture_mode)

            # IN-05: artifacts BEFORE metrics — partial failure leaves no orphan metrics
            mlflow.log_dict(frozen_artifact, "frozen_paraphrases_runner.json")
            mlflow.log_dict(before_snapshot, "before_snapshot.json")
            mlflow.log_dict(after_snapshot, "after_snapshot.json")
            mlflow.log_dict(
                {"place_ids": sorted(new_v2_ids)},
                "db_diff_v2_place_ids.json",
            )

            # metrics last
            mlflow.log_metric("before_hit_at_k", before_hit_at_k)
            mlflow.log_metric("after_hit_at_k", after_hit_at_k)
            mlflow.log_metric("hit_rate_delta", hit_rate_delta)
            mlflow.log_metric("recall_at_k", recall_at_k)
            mlflow.log_metric("new_place_count", new_place_count)
            mlflow.log_metric("embed_added_count", embed_added_count)
    except Exception as exc:
        print(
            f"[INFRA] MLflow logging failed: {exc}. Durable artifacts are required.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA) from exc


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    """Run the full productionized loop runner gate sequence.

    CANONICAL ORDER (D-07 LOCKED CONSTRAINT — do not reorder):
    1. Read SANDBOX_DATABASE_URL; EXIT_INFRA if unset.
    2. Resolve prod URL BEFORE coercing DATABASE_URL.
    3. Prod-safety guard BEFORE coercion.
    4. Coerce os.environ["DATABASE_URL"] = sandbox_url.
       get_settings.cache_clear() + close_db_pool() immediately after.
    5. W1 resolved-target assertion (both free-function AND cached-settings path).
    4b. Embedding-table assertion: settings.embedding_table == 'place_embeddings_v2'.
    6. Deferred imports of gap_mine_main + get_conn + assert_sandbox_write_target
       (ALL deferred to AFTER sandbox coercion — D-07).
    7. Gap handoff: clear-stale-pending → snapshot → gap_mine_main → set-diff.
    8. Paraphrase generation + non-circularity check + durable freeze to disk (D-04).
    9. Before-snapshot (places_raw + v2).
    10. Ingest + embed subprocesses.
    11. DB-diffs: places_raw guard (no new rows → EXIT_INFRA) + v2 guard (no embeds).
    12. After-snapshot + score (hit@k + recall@k).
    13. Floor gate via decide_loop_exit.
    14. MLflow artifacts-before-metrics.
    15. Verdict banner + SystemExit.
    """
    # ── Step 1: Resolve sandbox URL BEFORE any app.* touch ──────────────────
    sandbox_url = os.environ.get("SANDBOX_DATABASE_URL")
    if not sandbox_url:
        print("[INFRA] SANDBOX_DATABASE_URL is not set. Cannot proceed.", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)

    # fixture_mode flag: True = seeded-sandbox fixture demand (no DEMAND_DATABASE_URL)
    fixture_mode = not bool(os.environ.get("DEMAND_DATABASE_URL"))

    # ── Step 2: Resolve prod URL BEFORE coercion ─────────────────────────────
    prod_url = resolve_prod_url(sandbox_url=sandbox_url)

    # ── Step 3: Prod-safety guard BEFORE any coercion or destructive op ──────
    allow_remote = bool(os.environ.get("SANDBOX_ALLOW_REMOTE"))
    safety_result = check_prod_safety(sandbox_url, prod_url, allow_remote=allow_remote)
    if not safety_result.ok:
        print(f"[INFRA] Prod-safety guard FAILED: {safety_result.message}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    print("[prod-safety] PASS")

    # ── Step 4: Coerce DATABASE_URL to sandbox + cache_clear + close_pool ────
    os.environ["DATABASE_URL"] = sandbox_url

    from app.config import get_settings  # noqa: PLC0415
    from app.db_pool import close_db_pool  # noqa: PLC0415

    get_settings.cache_clear()
    close_db_pool()

    # ── Step 5: W1 resolved-target assertion ─────────────────────────────────
    from app.config import resolve_database_url  # noqa: PLC0415

    resolved = resolve_database_url(os.environ)
    assert_resolved_target(sandbox_url=sandbox_url, resolved_url=resolved)

    settings_resolved = get_settings().resolved_database_url
    if settings_resolved != sandbox_url:
        print(
            f"[INFRA] get_settings().resolved_database_url ({settings_resolved!r}) does not match "
            f"SANDBOX_DATABASE_URL ({sandbox_url!r}). "
            "The settings lru_cache was not cleared properly after DATABASE_URL injection. "
            "This is the path the DB pool uses — mismatch means the pool would target prod.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    print(f"[resolved-target] in-process target confirmed: {resolved!r}")

    # ── Step 4b: Embedding-table assertion (D-07 LOCKED CONSTRAINT) ──────────
    # semantic_search reads settings.embedding_table at call time; a stale settings
    # object pointing at the wrong view would score against the wrong index.
    settings = get_settings()
    if settings.embedding_table != "place_embeddings_v2":
        print(
            f"[INFRA] settings.embedding_table={settings.embedding_table!r} != "
            "'place_embeddings_v2'. semantic_search would score against the wrong view. "
            "Set EMBEDDING_TABLE=place_embeddings_v2 in your environment.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    print(f"[embedding-table] confirmed: {settings.embedding_table!r}")

    # ── Step 6: Deferred imports (ALL after coercion — D-07 ordering) ────────
    # IMPORTANT: gap_mine_main MUST be imported here (not at module scope) because
    # importing coverage_agent triggers get_conn() pool initialization from cached
    # settings. Importing AFTER cache_clear ensures the pool targets sandbox.
    from app.db import get_conn  # noqa: PLC0415,I001
    from app.tools.retrieval import semantic_search  # noqa: PLC0415
    from scripts.coverage_agent import gap_mine_main  # noqa: PLC0415
    from scripts.sandbox_guard import assert_sandbox_write_target  # noqa: PLC0415

    # ── Step 7: Gap handoff (D-08 deterministic one-gap set-diff) ────────────
    print("[gap-handoff] Clearing stale pending proposals and running gap-mine ...")

    # The stale-clear runs on the SAME conn as the write guard (D-07 / sandbox_guard)
    with get_conn() as write_conn:
        assert_sandbox_write_target(write_conn)

        # Clear ALL stale pending rows so the set-diff is clean
        reject_stale_sql = """
            UPDATE places_ingest_query_proposals
            SET status = 'rejected'
            WHERE status = 'pending'
        """
        with write_conn.cursor() as cur:
            cur.execute(reject_stale_sql)
        write_conn.commit()

        # Snapshot pending query_text BEFORE gap-mine
        with write_conn.cursor() as cur:
            cur.execute(
                "SELECT query_text FROM places_ingest_query_proposals WHERE status = 'pending'"
            )
            pending_before: set[str] = {row[0] for row in cur.fetchall()}

    # Run gap miner (deferred import above — after coercion)
    gap_mine_main(["--top-n", "1"])

    # Snapshot pending query_text AFTER gap-mine (new connection — pool now sandbox)
    with get_conn() as read_conn, read_conn.cursor() as cur:
        cur.execute("SELECT query_text FROM places_ingest_query_proposals WHERE status = 'pending'")
        pending_after: set[str] = {row[0] for row in cur.fetchall()}

    # Deterministic set-diff on query_text (D-08)
    new = pending_after - pending_before

    if len(new) == 0:
        print(
            "[loop-runner] No new pending proposals after gap-mine. "
            "Cold start or no demand gaps found — honest exit 0 (no-op).",
        )
        raise SystemExit(EXIT_PASS)

    if len(new) > 1:
        print(
            f"[INFRA] gap-mine with --top-n 1 produced {len(new)} new proposals: {new!r}. "
            "Expected exactly 1. This is an infrastructure error — aborting.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    # Exactly one new gap: parse (neighborhood, cuisine) from seed query
    gap_seed_query = next(iter(new))
    print(f"[gap-handoff] Gap seed query: {gap_seed_query!r}")

    # Reverse "{cuisine} restaurants in {neighborhood} San Francisco"
    suffix = " San Francisco"
    midfix = " restaurants in "
    if not gap_seed_query.endswith(suffix) or midfix not in gap_seed_query:
        print(
            f"[INFRA] Cannot parse gap seed query {gap_seed_query!r} — "
            f"expected format: '{{cuisine}} restaurants in {{neighborhood}} San Francisco'",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    without_suffix = gap_seed_query[: -len(suffix)]
    idx = without_suffix.index(midfix)
    cuisine = without_suffix[:idx]
    neighborhood = without_suffix[idx + len(midfix) :]

    print(f"[gap-handoff] Parsed: neighborhood={neighborhood!r}, cuisine={cuisine!r}")

    # ── Step 6b: BASELINE/MINER GAP CONTRACT (WARNING 2) ─────────────────────
    # If LOOP_GAP_NEIGHBORHOOD / LOOP_GAP_CUISINE are set, the miner-parsed pair
    # MUST match them — otherwise the metric would silently measure the wrong gap
    # (the baseline excluded that bucket, but the miner chose a different one).
    env_neighborhood = os.environ.get("LOOP_GAP_NEIGHBORHOOD")
    env_cuisine = os.environ.get("LOOP_GAP_CUISINE")
    if env_neighborhood is not None and env_cuisine is not None:
        if (neighborhood, cuisine) != (env_neighborhood, env_cuisine):
            print(
                f"[INFRA] BASELINE/MINER GAP MISMATCH: "
                f"baseline excluded bucket ({env_neighborhood!r}, {env_cuisine!r}) "
                f"but miner chose ({neighborhood!r}, {cuisine!r}). "
                "The hit@k metric would silently measure the wrong gap. "
                "Re-provision the sandbox with the bucket the miner chose, or "
                "set LOOP_GAP_NEIGHBORHOOD/LOOP_GAP_CUISINE to match the miner's choice.",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_INFRA)
        print("[gap-contract] PASS — miner-parsed pair matches LOOP_GAP_* env")

    # ── Step 7 (D-04 paraphrase freeze): LLM-generate + non-circularity + disk ──
    # MUST happen BEFORE before-snapshot/ingest so the freeze survives a crash
    # and post-ingest data cannot game the paraphrases.
    print(f"[paraphrase-gen] Generating {N} paraphrases for {gap_seed_query!r} ...")
    paraphrases, generation_prompt, generation_model = _generate_paraphrases(
        seed_query=gap_seed_query,
        neighborhood=neighborhood,
        cuisine=cuisine,
        n=N,
    )

    # Non-circularity check: none of the paraphrases may equal the seed (exact-string)
    non_circ = check_non_circularity(paraphrases, [gap_seed_query])
    if not non_circ.ok:
        print(f"[INFRA] Non-circularity guard FAILED: {non_circ.message}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    print("[non-circularity] PASS")

    # Build frozen artifact dict (extends falsifier_paraphrases.json shape)
    frozen_artifact: dict[str, Any] = {
        "seed_query": gap_seed_query,
        "generation_prompt": generation_prompt,
        "non_circularity_note": _NON_CIRCULARITY_NOTE,
        "gap_neighborhood": neighborhood,
        "gap_cuisine": cuisine,
        "generation_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "generation_model": generation_model,
        "paraphrases": paraphrases,
    }

    # Durably freeze to disk BEFORE ingest (D-04 BLOCKER 4)
    # This write MUST happen here, before the ingest subprocess in Step 9 below.
    os.makedirs(RUNNER_ARTIFACT_DIR, exist_ok=True)
    frozen_artifact_path = os.path.join(RUNNER_ARTIFACT_DIR, "frozen_paraphrases_runner.json")
    with open(frozen_artifact_path, "w") as fh:
        json.dump(frozen_artifact, fh, indent=2)
    print(f"[paraphrase-freeze] Frozen to disk: {frozen_artifact_path!r}")

    # ── Step 8: Before-snapshot (places_raw + v2) ────────────────────────────
    print(f"[before-snapshot] Probing {N} paraphrases with k={K} ...")
    before_raw_ids = _snapshot_ids_from_url(sandbox_url, "places_raw")
    before_v2_ids = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")

    before_topk: list[list[str]] = [
        [h.place_id for h in semantic_search(p, k=K)] for p in paraphrases
    ]
    # before_hit@k = 0 by construction (new IDs didn't exist before — D-03)
    before_hit_result = compute_hit_rate(before_topk, before_v2_ids)

    # ── Step 9: Ingest + embed subprocesses ──────────────────────────────────
    child_env = {**os.environ, "DATABASE_URL": sandbox_url}

    print(f"[ingest] Running ingest for gap query={gap_seed_query!r} ...")
    run_subprocess_or_infra(
        argv=[sys.executable, "scripts/ingest_places_sf.py"],
        env=child_env,
    )
    print("[ingest] Done.")

    print("[embed-v2] Running embed-v2 ...")
    run_subprocess_or_infra(
        argv=[sys.executable, "-m", "scripts.embed_places_pgvector_v2"],
        env=child_env,
    )
    print("[embed-v2] Done.")

    # ── Step 10: DB-diffs + provisioning guards ───────────────────────────────
    # GUARD A: places_raw diff — "no new rows ingested" provisioning guard (D-02)
    after_raw_ids = _snapshot_ids_from_url(sandbox_url, "places_raw")
    new_raw_ids = db_diff(before_raw_ids, after_raw_ids)
    new_place_count = len(new_raw_ids)

    print(f"[db-diff] new places_raw rows: {new_place_count}")

    if new_place_count == 0:
        print(
            "[INFRA] loop ran but ingested ZERO new places_raw rows — "
            "this is a provisioning/INFRA error, not a metric FAIL. "
            "Check that the gap query is valid, GOOGLE_PLACES_API_KEY is set, "
            "and the gap bucket was not already fully ingested.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    # GUARD B: v2 diff — "rows ingested but none embedded" infra guard (D-02)
    after_v2_ids = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")
    new_v2_ids = db_diff(before_v2_ids, after_v2_ids)
    embed_added_count = len(new_v2_ids)

    print(f"[db-diff] new place_embeddings_v2 rows: {embed_added_count}")

    if embed_added_count == 0:
        print(
            "[INFRA] rows were ingested to places_raw but NONE were embedded to "
            "place_embeddings_v2 — this is an INFRA error (embed-v2 silently skipped all). "
            "Exit EXIT_INFRA — do NOT burn after-snapshot API calls.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_INFRA)

    # ── Step 11: After-snapshot + score ──────────────────────────────────────
    print(
        f"[after-snapshot] Probing {N} paraphrases with k={K} against "
        f"{embed_added_count} new v2 embeddings ..."
    )
    after_topk: list[list[str]] = [
        [h.place_id for h in semantic_search(p, k=K)] for p in paraphrases
    ]

    # Target set = v2 DB-diff (new embedded IDs) — NOT new_raw_ids (D-03)
    after_hit_result = compute_hit_rate(after_topk, new_v2_ids)
    after_hit_at_k = after_hit_result.hit_rate
    before_hit_at_k = before_hit_result.hit_rate

    recall_result = compute_recall_at_k(after_topk, new_v2_ids)
    delta = after_hit_at_k - before_hit_at_k

    print(
        f"[after-snapshot] hit@{K} = {after_hit_result.hit_count}/{after_hit_result.n} = "
        f"{after_hit_at_k:.3f}"
    )
    print(f"[gate] delta = {delta:+.3f} (before={before_hit_at_k:.3f}, after={after_hit_at_k:.3f})")
    print(
        f"[recall] recall@{K} = {recall_result.recall:.3f} ({recall_result.found_count}/{recall_result.total_count})"
    )

    # ── Step 12: Floor resolution ─────────────────────────────────────────────
    floor = float(os.environ.get("LOOP_HIT_RATE_FLOOR", FLOOR))
    print(f"[floor] hit@k floor = {floor}")

    # ── Step 13: Gate ─────────────────────────────────────────────────────────
    exit_code = decide_loop_exit(
        before_rate=before_hit_at_k,
        after_rate=after_hit_at_k,
        floor=floor,
        guard_violation=None,
        embed_added_count=embed_added_count,
    )

    # ── Step 14: MLflow (artifacts-before-metrics, IN-05) ────────────────────
    before_snapshot: dict[str, Any] = {
        "paraphrase_topk": before_topk,
        "hit_rate": before_hit_at_k,
        "hit_count": before_hit_result.hit_count,
        "n": before_hit_result.n,
    }
    after_snapshot: dict[str, Any] = {
        "paraphrase_topk": after_topk,
        "hit_rate": after_hit_at_k,
        "hit_count": after_hit_result.hit_count,
        "n": after_hit_result.n,
        "recall": recall_result.recall,
        "found_count": recall_result.found_count,
        "total_count": recall_result.total_count,
    }
    log_to_mlflow(
        neighborhood=neighborhood,
        cuisine=cuisine,
        seed_query=gap_seed_query,
        paraphrases=paraphrases,
        frozen_artifact=frozen_artifact,
        frozen_artifact_path=frozen_artifact_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        new_v2_ids=new_v2_ids,
        before_hit_at_k=before_hit_at_k,
        after_hit_at_k=after_hit_at_k,
        hit_rate_delta=delta,
        recall_at_k=recall_result.recall,
        new_place_count=new_place_count,
        embed_added_count=embed_added_count,
        floor=floor,
        fixture_mode=fixture_mode,
    )

    # ── Step 15: Verdict banner ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if exit_code == EXIT_PASS:
        print(
            f"loop_runner: VERDICT = PASS "
            f"(hit@{K} delta={delta:+.3f}, after={after_hit_at_k:.3f}, "
            f"recall@{K}={recall_result.recall:.3f}, floor={floor}, fixture_mode={fixture_mode})"
        )
    elif exit_code == EXIT_FAIL:
        print(
            f"loop_runner: VERDICT = FAIL "
            f"(hit@{K} delta={delta:+.3f}, after={after_hit_at_k:.3f}, "
            f"floor={floor}; re-scopes milestone, fixture_mode={fixture_mode})"
        )
    else:
        print(
            f"loop_runner: VERDICT = INFRA ERROR "
            f"(embed_added_count={embed_added_count}, fixture_mode={fixture_mode})"
        )
    print(f"{'=' * 60}\n")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
