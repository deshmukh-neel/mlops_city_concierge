# W5 — Coverage-gap ingestion agent

**Branch:** `feature/agent-w5-coverage-agent`
**Depends on:** nothing (independent of W1–W4)
**Unblocks:** —

## Goal

Turn the `place_query_hits` audit table (`scripts/db/init.sql:60-69`) — currently a write-only log — into a feedback loop that improves the dataset over time. An agent reads recent low-quality interactions, identifies under-covered neighborhoods or cuisines, decides what new seed queries to add, and inserts them into `places_ingest_query_checkpoints` so the existing ingestion script picks them up on next run.

This is the MLOps story for the class: an agent in the **data pipeline**, not just the serving path. It pairs with the existing MLflow logging and gives the system a measurable data flywheel.

After this PR:
- `python scripts/coverage_agent.py [--dry-run] [--days N]` runs the analysis.
- New seed queries land in `places_ingest_query_checkpoints` with `status='pending'`.
- The next `make ingest` run (or scheduled job) executes them.
- An MLflow run captures the analysis + proposed queries for traceability.

## Files

### New: `scripts/coverage_agent.py`

```python
"""Coverage-gap ingestion agent.

Reads place_query_hits + places_raw to find:
  - Neighborhoods with low place counts
  - Cuisines with low representation
  - Queries with consistently bad rank-of-first-relevant
Then asks the LLM to propose new seed queries, and inserts them into
places_ingest_query_checkpoints with status='pending'.

Reuses:
  - DB connection from app/retriever.py (get_conn)
  - Settings + LLM resolution from app/config.py
  - Checkpoint mechanism from scripts/ingest_places_sf.py:624-663
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import mlflow

from app.config import get_settings, resolve_llm_api_key
from app.retriever import get_conn
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class CoverageStat:
    bucket: str          # e.g. "neighborhood:Outer Sunset" or "cuisine:vietnamese"
    place_count: int
    distinct_queries: int
    last_ingest: Optional[datetime]


@dataclass
class ProposedQuery:
    query_text: str
    field_mode: str         # 'enriched' (default for new queries)
    rationale: str          # for the MLflow artifact + audit trail


def gather_stats(days: int) -> list[CoverageStat]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    sql = """
        WITH neighborhoods AS (
            SELECT
                regexp_replace(formatted_address,
                    '.*, ([^,]+), San Francisco.*', E'\\\\1') AS bucket,
                COUNT(*)             AS place_count,
                MAX(source_updated_at) AS last_ingest
            FROM places_raw
            WHERE source_city ILIKE '%san francisco%'
            GROUP BY 1
        ),
        cuisines AS (
            SELECT
                LOWER(unnest(types)) AS bucket,
                COUNT(*)             AS place_count,
                MAX(source_updated_at) AS last_ingest
            FROM places_raw
            WHERE source_city ILIKE '%san francisco%'
            GROUP BY 1
        ),
        recent_query_diversity AS (
            SELECT
                'recent_query'                     AS bucket,
                COUNT(DISTINCT query_text)         AS distinct_queries,
                COUNT(*)                           AS place_count,
                MAX(seen_at)                       AS last_ingest
            FROM place_query_hits
            WHERE seen_at >= %s
        )
        SELECT 'neighborhood:' || bucket, place_count, 0, last_ingest FROM neighborhoods
        UNION ALL
        SELECT 'cuisine:' || bucket, place_count, 0, last_ingest FROM cuisines
        UNION ALL
        SELECT bucket, place_count, distinct_queries, last_ingest FROM recent_query_diversity;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [cutoff])
        return [CoverageStat(*row) for row in cur.fetchall()]


def find_gaps(stats: list[CoverageStat], min_place_count: int = 5) -> list[CoverageStat]:
    return [s for s in stats
            if s.bucket.startswith(("neighborhood:", "cuisine:"))
            and s.place_count < min_place_count]


def propose_queries(gaps: list[CoverageStat], llm) -> list[ProposedQuery]:
    """Ask the LLM to propose new seed queries for each gap.

    Constraints applied to outputs:
      - format: '<thing> in <neighborhood> San Francisco' or
                '<cuisine> restaurants in San Francisco'
      - dedupe against existing checkpoint queries (DB-side check below)
    """
    if not gaps:
        return []

    prompt = _build_proposal_prompt(gaps)
    resp = llm.invoke(prompt).content
    raw = json.loads(resp)  # LLM is instructed to emit a JSON list
    return [ProposedQuery(**item) for item in raw]


def insert_pending(proposals: list[ProposedQuery], dry_run: bool) -> int:
    if dry_run or not proposals:
        for p in proposals:
            print(f"[dry-run] would insert: {p.query_text!r} ({p.rationale})")
        return 0

    sql = """
        INSERT INTO places_ingest_query_checkpoints
            (query_text, status, pages_processed, api_calls, rows_seen, rows_changed)
        VALUES (%s, 'pending', 0, 0, 0, 0)
        ON CONFLICT (query_text) DO NOTHING;
    """
    with get_conn() as conn, conn.cursor() as cur:
        for p in proposals:
            cur.execute(sql, [f"{p.field_mode}::{p.query_text}"])
        inserted = cur.rowcount
        conn.commit()
    return inserted


def log_to_mlflow(stats, gaps, proposals, inserted, dry_run):
    mlflow.set_experiment("coverage_agent")
    with mlflow.start_run(run_name=f"coverage-{datetime.utcnow().isoformat(timespec='seconds')}"):
        mlflow.log_param("dry_run", dry_run)
        mlflow.log_metric("gaps_found", len(gaps))
        mlflow.log_metric("proposals_made", len(proposals))
        mlflow.log_metric("inserted", inserted)
        mlflow.log_dict({"stats": [asdict(s) for s in stats]}, "stats.json")
        mlflow.log_dict({"gaps": [asdict(g) for g in gaps]}, "gaps.json")
        mlflow.log_dict({"proposals": [asdict(p) for p in proposals]}, "proposals.json")


def _build_proposal_prompt(gaps: list[CoverageStat]) -> str:
    gap_lines = "\n".join(f"- {g.bucket} (place_count={g.place_count})" for g in gaps)
    return f"""You are a data coverage planner for a SF restaurant database.

Here are coverage gaps (buckets with under-represented place counts):
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
- Format: "<thing> in <neighborhood> San Francisco" or "<cuisine> restaurants in San Francisco"
- Don't propose generic queries already covered (e.g. "restaurants in San Francisco")
- field_mode = "enriched" for new exploratory queries
- One JSON list, no prose, no markdown.
"""


def _make_llm():
    settings = get_settings()
    if settings.llm_provider == "openai":
        return ChatOpenAI(model=settings.openai_chat_model, temperature=0.3,
                          api_key=resolve_llm_api_key("openai"))
    return ChatGoogleGenerativeAI(model=settings.gemini_chat_model, temperature=0.3,
                                  google_api_key=resolve_llm_api_key("gemini"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=14,
                   help="Look at hits from the last N days for diversity stats.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print proposed queries; don't insert.")
    p.add_argument("--min-places", type=int, default=5,
                   help="Bucket with fewer places than this is a gap.")
    args = p.parse_args(argv)

    stats = gather_stats(args.days)
    gaps = find_gaps(stats, args.min_places)
    proposals = propose_queries(gaps, _make_llm())
    inserted = insert_pending(proposals, args.dry_run)

    log_to_mlflow(stats, gaps, proposals, inserted, args.dry_run)
    print(f"gaps={len(gaps)} proposals={len(proposals)} inserted={inserted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Modify: `Makefile`

Add a target:

```makefile
coverage-agent:
	python scripts/coverage_agent.py --dry-run

coverage-agent-apply:
	python scripts/coverage_agent.py
```

### Modify: `scripts/ingest_places_sf.py` (small)

Confirm the existing `run()` entrypoint already pulls `pending` checkpoints first. If it doesn't, add a path that reads `places_ingest_query_checkpoints WHERE status='pending'` and processes those before the static seed list — without duplicating any of the existing logic. (Reference: `scripts/ingest_places_sf.py:624-663` for the existing checkpoint flow.)

## Tests

### New: `tests/unit/test_coverage_agent.py`

```python
from unittest.mock import MagicMock, patch

from scripts.coverage_agent import (
    CoverageStat, ProposedQuery, find_gaps, insert_pending, propose_queries,
)


def test_find_gaps_filters_by_min_places():
    stats = [
        CoverageStat("neighborhood:Mission", 200, 0, None),
        CoverageStat("neighborhood:Outer Sunset", 2, 0, None),
        CoverageStat("cuisine:italian", 100, 0, None),
        CoverageStat("cuisine:burmese", 1, 0, None),
        CoverageStat("recent_query", 50, 30, None),  # not a bucket prefix
    ]
    gaps = find_gaps(stats, min_place_count=5)
    assert {g.bucket for g in gaps} == {"neighborhood:Outer Sunset", "cuisine:burmese"}


def test_propose_queries_parses_llm_json():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = json.dumps([{
        "query_text": "burmese restaurants in San Francisco",
        "field_mode": "enriched",
        "rationale": "burmese only has 1 place",
    }])
    gaps = [CoverageStat("cuisine:burmese", 1, 0, None)]
    proposals = propose_queries(gaps, fake_llm)
    assert len(proposals) == 1
    assert proposals[0].query_text == "burmese restaurants in San Francisco"


def test_insert_pending_dry_run_inserts_nothing(capsys):
    proposals = [ProposedQuery("x", "enriched", "y")]
    n = insert_pending(proposals, dry_run=True)
    assert n == 0
    captured = capsys.readouterr().out
    assert "[dry-run]" in captured
```

Integration test (gated): seed `place_query_hits` with a known coverage gap, run `main(["--dry-run"])`, assert MLflow run was created and the proposals JSON artifact contains a query for that gap.

## Manual verification

```bash
make ingest                                            # populate some data
psql ... -c "INSERT INTO place_query_hits ..."         # simulate a sparse bucket
python scripts/coverage_agent.py --dry-run             # see proposals
python scripts/coverage_agent.py                       # apply
psql ... -c "SELECT query_text, status FROM places_ingest_query_checkpoints
             WHERE status='pending';"                  # verify
make ingest                                            # ingestion picks up pending
```

Confirm new places appear in `places_raw` from the proposed queries within one ingest cycle.

## Risks / open questions

- **LLM proposing duplicates of existing queries.** Mitigate by passing the top-N existing queries into the prompt as exclusions, OR rely on the `ON CONFLICT DO NOTHING` insert. Latter is simpler; ship it and add the prompt-side exclusion only if duplication is high in practice.
- **Neighborhood regex is brittle.** SF addresses don't always follow `..., {neighborhood}, San Francisco`. The regex extracts a naive bucket; some buckets may be garbage. Acceptable v1; add a real neighborhood lookup table later if it matters for accuracy.
- **Cost.** Each run hits the LLM once. At a daily cadence, negligible. If we ever schedule it hourly, cap proposals at 8 (already done).
- **Race with running ingestion.** If `make ingest` is running while we insert pending rows, the script's existing checkpoint logic should handle it (the table is the synchronization point). Confirm during manual verification.
