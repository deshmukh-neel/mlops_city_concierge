"""Coverage-gap ingestion agent (W5).

Reads place_query_hits + places_raw to find under-covered neighborhoods and
cuisines, asks the judge LLM to propose new seed queries, and writes them
into places_ingest_query_proposals(status='pending'). The next ingest run
prepends pending proposals to its seed list.

Design notes
------------
- LLM is `app.agent.critique.vibe.make_judge()` so the model swap mechanism
  matches the rest of the codebase (env: EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL).
- Dedup is DB-side via PRIMARY KEY (query_text) + ON CONFLICT DO NOTHING;
  no prompt-side exclusion list (flagged as future work).
- Neighborhood extraction is a naive regex against formatted_address; some
  buckets will be garbage. Acceptable v1 — see Risks in the plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
from langchain_core.messages import HumanMessage

from app.agent.critique import vibe
from app.db import get_conn
from scripts.ingest_places_sf import CUISINES, build_seed_queries

_log = logging.getLogger("coverage_agent")


@dataclass
class CoverageStat:
    bucket: str  # "neighborhood:Outer Sunset" | "cuisine:vietnamese" | "recent_query"
    place_count: int
    distinct_queries: int
    last_ingest: datetime | None


@dataclass
class ProposedQuery:
    query_text: str
    field_mode: str
    rationale: str


def gather_stats(days: int) -> list[CoverageStat]:
    """Pull coverage buckets from the DB.

    Three union'd CTEs: per-neighborhood place counts, per-cuisine place
    counts, and a single 'recent_query' row reporting query diversity in
    the last `days`.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    sql = """
        WITH neighborhoods AS (
            SELECT
                regexp_replace(
                    formatted_address,
                    '.*, ([^,]+), San Francisco.*',
                    E'\\\\1'
                ) AS bucket,
                COUNT(*) AS place_count,
                MAX(source_updated_at) AS last_ingest
            FROM places_raw
            WHERE source_city ILIKE '%san francisco%'
              AND formatted_address ~ '.*, [^,]+, San Francisco.*'
            GROUP BY 1
        ),
        cuisines AS (
            SELECT bucket, COUNT(*) AS place_count, MAX(source_updated_at) AS last_ingest
            FROM (
                SELECT LOWER(unnest(types)) AS bucket, source_updated_at
                FROM places_raw
                WHERE source_city ILIKE '%san francisco%'
            ) t
            WHERE bucket = ANY(%s)
            GROUP BY 1
        ),
        recent_query_diversity AS (
            SELECT
                'recent_query' AS bucket,
                COUNT(DISTINCT query_text) AS distinct_queries,
                COUNT(*) AS place_count,
                MAX(seen_at) AS last_ingest
            FROM place_query_hits
            WHERE seen_at >= %s
        )
        SELECT 'neighborhood:' || bucket, place_count, 0, last_ingest FROM neighborhoods
        UNION ALL
        SELECT 'cuisine:' || bucket, place_count, 0, last_ingest FROM cuisines
        UNION ALL
        SELECT bucket, place_count, distinct_queries, last_ingest
        FROM recent_query_diversity;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [CUISINES, cutoff])
        stats = [CoverageStat(*row) for row in cur.fetchall()]
    return _fill_missing_cuisines(stats)


def _fill_missing_cuisines(stats: list[CoverageStat]) -> list[CoverageStat]:
    """Synthesize zero-count rows for cuisines absent from the result set.

    Cuisines with 0 places are silently dropped by the SQL (no rows to group),
    but those are exactly the gaps we most need the LLM to fill.
    """
    seen = {s.bucket.removeprefix("cuisine:") for s in stats if s.bucket.startswith("cuisine:")}
    return stats + [CoverageStat(f"cuisine:{c}", 0, 0, None) for c in CUISINES if c not in seen]


def find_gaps(stats: list[CoverageStat], min_place_count: int = 5) -> list[CoverageStat]:
    return [
        s
        for s in stats
        if s.bucket.startswith(("neighborhood:", "cuisine:")) and s.place_count < min_place_count
    ]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_REQUIRED_KEYS = {"query_text", "field_mode", "rationale"}


def _parse_proposals(raw: str) -> list[ProposedQuery]:
    """Parse LLM JSON output into ProposedQuery objects.

    Tolerates ```json fences. Skips items missing required keys or with
    non-string values rather than failing the whole run.
    """
    cleaned = _FENCE_RE.sub("", raw or "").strip()
    if not cleaned:
        return []
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as e:
        _log.warning("LLM returned non-JSON output (%s); ignoring", e)
        return []
    if not isinstance(items, list):
        _log.warning("LLM returned non-list payload (%r); ignoring", type(items).__name__)
        return []

    proposals: list[ProposedQuery] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not _REQUIRED_KEYS.issubset(item.keys()):
            continue
        if not all(isinstance(item[k], str) for k in _REQUIRED_KEYS):
            continue
        proposals.append(
            ProposedQuery(
                query_text=item["query_text"].strip(),
                field_mode=item["field_mode"].strip() or "enriched",
                rationale=item["rationale"].strip(),
            )
        )
    return proposals


def _format_gap_line(g: CoverageStat) -> str:
    axis, _, name = g.bucket.partition(":")
    return f"- type={axis} name='{name}' place_count={g.place_count}"


def _build_proposal_prompt(gaps: list[CoverageStat]) -> str:
    gap_lines = "\n".join(_format_gap_line(g) for g in gaps)
    return f"""You are a data coverage planner for a SF restaurant database.

Coverage gaps (buckets with under-represented place counts):
{gap_lines}

Propose up to 8 new seed queries to fill these gaps. Output a JSON list with
this schema:

  [
    {{"query_text": "vietnamese restaurants in Outer Sunset San Francisco",
      "field_mode": "enriched",
      "rationale": "Outer Sunset has only 2 vietnamese places"}},
    ...
  ]

Rules:
- For type=neighborhood gaps: "<thing> in <name> San Francisco".
- For type=cuisine gaps: "<name> restaurants in San Francisco".
- Don't propose generic queries already covered (e.g. "restaurants in San Francisco").
- field_mode must be "enriched".
- One JSON list, no prose, no markdown.
"""


def propose_queries(gaps: list[CoverageStat], llm: Any | None) -> list[ProposedQuery]:
    """Ask the LLM to fill the given gaps. Returns [] if no gaps or no LLM."""
    if not gaps or llm is None:
        return []
    prompt = _build_proposal_prompt(gaps)
    raw = llm.invoke([HumanMessage(content=prompt)]).content
    if not isinstance(raw, str):
        _log.warning("LLM returned non-string content type %s; ignoring", type(raw).__name__)
        return []
    return _parse_proposals(raw)


def existing_query_texts(conn: Any) -> set[str]:
    """Queries the ingester already knows about, from any source.

    Union of: static seed list, prior checkpoints (any status), prior
    proposals (any status). Used to filter agent output before insert so
    the proposals table reflects actually-new coverage.
    """
    existing = set(build_seed_queries())
    with conn.cursor() as cur:
        cur.execute("SELECT query_text FROM places_ingest_query_checkpoints")
        existing.update(row[0] for row in cur.fetchall())
        cur.execute("SELECT query_text FROM places_ingest_query_proposals")
        existing.update(row[0] for row in cur.fetchall())
    return existing


def filter_already_covered(
    proposals: list[ProposedQuery], existing: set[str]
) -> tuple[list[ProposedQuery], list[ProposedQuery]]:
    """Split proposals into (kept, dropped) based on the existing-query set."""
    kept: list[ProposedQuery] = []
    dropped: list[ProposedQuery] = []
    for p in proposals:
        (dropped if p.query_text in existing else kept).append(p)
    return kept, dropped


def insert_pending(proposals: list[ProposedQuery], dry_run: bool) -> int:
    """Insert proposals as 'pending' rows. Returns the number of rows
    actually inserted (0 if dry_run, or if every proposal collided)."""
    if dry_run:
        for p in proposals:
            print(f"[dry-run] would insert: {p.query_text!r} ({p.rationale})")
        return 0
    if not proposals:
        return 0

    sql = """
        INSERT INTO places_ingest_query_proposals (query_text, status, rationale)
        VALUES (%s, 'pending', %s)
        ON CONFLICT (query_text) DO NOTHING;
    """
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for p in proposals:
            cur.execute(sql, [p.query_text, p.rationale])
            inserted += cur.rowcount
        conn.commit()
    return inserted


def log_to_mlflow(
    stats: list[CoverageStat],
    gaps: list[CoverageStat],
    proposals: list[ProposedQuery],
    dropped: list[ProposedQuery],
    inserted: int,
    dry_run: bool,
) -> None:
    mlflow.set_experiment("coverage_agent")
    run_name = f"coverage-{datetime.now(UTC).isoformat(timespec='seconds')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dry_run", dry_run)
        mlflow.log_metric("gaps_found", len(gaps))
        mlflow.log_metric("proposals_made", len(proposals))
        mlflow.log_metric("dropped_already_covered", len(dropped))
        mlflow.log_metric("inserted", inserted)
        mlflow.log_dict({"stats": [_stat_to_dict(s) for s in stats]}, "stats.json")
        mlflow.log_dict({"gaps": [_stat_to_dict(g) for g in gaps]}, "gaps.json")
        mlflow.log_dict({"proposals": [asdict(p) for p in proposals]}, "proposals.json")
        mlflow.log_dict({"dropped": [asdict(p) for p in dropped]}, "dropped.json")


def _stat_to_dict(stat: CoverageStat) -> dict[str, Any]:
    out = asdict(stat)
    if out["last_ingest"] is not None:
        out["last_ingest"] = out["last_ingest"].isoformat()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--days",
        type=int,
        default=14,
        help="Look at hits from the last N days for diversity stats.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed queries; don't insert.",
    )
    p.add_argument(
        "--min-places",
        type=int,
        default=5,
        help="Bucket with fewer places than this is a gap.",
    )
    args = p.parse_args(argv)

    stats = gather_stats(args.days)
    gaps = find_gaps(stats, args.min_places)
    llm = vibe.make_judge()
    if llm is None and gaps:
        _log.warning("judge unavailable (missing creds?); skipping LLM proposal step")
    proposals = propose_queries(gaps, llm)
    with get_conn() as conn:
        kept, dropped = filter_already_covered(proposals, existing_query_texts(conn))
    inserted = insert_pending(kept, args.dry_run)

    log_to_mlflow(stats, gaps, kept, dropped, inserted, args.dry_run)
    print(f"gaps={len(gaps)} proposals={len(proposals)} dropped={len(dropped)} inserted={inserted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
