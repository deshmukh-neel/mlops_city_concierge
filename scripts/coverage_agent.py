"""Coverage-gap ingestion agent (W5).

Reads place_query_hits + places_raw to find under-covered neighborhoods and
cuisines, asks the judge LLM to propose new seed queries, and writes them
into places_ingest_query_proposals(status='pending'). The next ingest run
prepends pending proposals to its seed list.

Design notes
------------
- LLM is `app.agent.critique.vibe.make_judge()` so the model swap mechanism
  matches the rest of the codebase (env: EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL).
- Proposals are filtered against build_seed_queries() + existing checkpoint
  and proposal rows before insert, so the proposals table reflects only
  actually-new coverage.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
import psycopg2
from langchain_core.messages import HumanMessage

from app.agent.critique import vibe
from app.db import get_conn
from scripts.ingest_places_sf import CUISINES, NEIGHBORHOODS, build_seed_queries

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
    cleaned = _FENCE_RE.sub("", raw).strip()
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


# ---------------------------------------------------------------------------
# Demand-extraction catalog sets (module-level, built once on import)
# ---------------------------------------------------------------------------

_CUISINES_SET: frozenset[str] = frozenset(CUISINES)
_NEIGHBORHOODS_SET: frozenset[str] = frozenset(NEIGHBORHOODS)

# ---------------------------------------------------------------------------
# Demand-extraction helpers (Task 1)
# ---------------------------------------------------------------------------


def _types_to_cuisines(primary_types: list[str]) -> list[str]:
    """Tier-1 cuisine extraction: map requested_primary_types to catalog cuisines.

    Strips the " restaurant" suffix (case-insensitive), lowercases, and checks
    membership in _CUISINES_SET.  No LLM call — pure lexical mapping.

    Examples::

        _types_to_cuisines(["Vietnamese Restaurant"])  # → ["vietnamese"]
        _types_to_cuisines(["Bar"])                    # → []
    """
    result: list[str] = []
    for pt in primary_types:
        candidate = pt.lower().removesuffix(" restaurant")
        if candidate in _CUISINES_SET:
            result.append(candidate)
    return result


def _lexical_cuisines(message: str) -> list[str]:
    """Tier-2a cuisine extraction: scan message text for catalog cuisine names.

    Case-insensitive substring/word scan of *message* against every CUISINES
    catalog member.  Returns a list (possibly empty) of all catalog cuisines
    named in the message.  No LLM call — symmetric to _lexical_neighborhoods.

    This closes the ROUND-3 HIGH: app/main.py returns requested_primary_types=[]
    for free-text messages, so the cuisine must be recovered from the message itself.
    """
    lower_msg = message.lower()
    return [c for c in CUISINES if c in lower_msg]


def _lexical_neighborhoods(message: str) -> list[str]:
    """Tier-1 neighborhood extraction: scan message text for catalog neighborhood names.

    Case-insensitive substring scan of *message* against every NEIGHBORHOODS
    catalog member (Title-Case strings).  Returns a list of all catalog
    neighborhoods named — no LLM call (REVIEW MEDIUM: lexical-before-LLM).

    Examples::

        _lexical_neighborhoods("dinner in Outer Sunset")
        # → ["Outer Sunset"]

        _lexical_neighborhoods("dinner in the Mission District and drinks in North Beach")
        # → ["Mission District", "North Beach"]
    """
    lower_msg = message.lower()
    return [n for n in NEIGHBORHOODS if n.lower() in lower_msg]


def _build_demand_batch_prompt(messages: list[str]) -> str:
    """Build a batch extraction prompt that embeds messages as a JSON array.

    Using json.dumps prevents a message containing ``"]}`` or other JSON-special
    characters from escaping the prompt format boundary (T-18-02-INJ mitigation).
    Residual risk: the model can still FOLLOW instructions embedded in the text;
    however, catalog-membership filtering on BOTH axes limits worst-case impact.
    """
    encoded_messages = json.dumps(messages)
    neighborhoods_list = json.dumps(NEIGHBORHOODS)
    cuisines_list = json.dumps(CUISINES)
    return (
        "You are a demand-extraction assistant for a San Francisco city guide.\n\n"
        "For each user message in the JSON array below, identify:\n"
        "  - Which San Francisco neighborhoods (from the allowed list) the user mentioned\n"
        "  - Which cuisines (from the allowed list) the user is interested in\n\n"
        f"Allowed neighborhoods: {neighborhoods_list}\n"
        f"Allowed cuisines: {cuisines_list}\n\n"
        "Messages (JSON array):\n"
        f"{encoded_messages}\n\n"
        "Output a JSON array with one object per message, in the same order:\n"
        '[\n  {"neighborhoods": [...], "cuisines": [...]},\n  ...\n]\n\n'
        "Rules:\n"
        "- Include ONLY values from the allowed lists above.\n"
        "- If a message implies no neighborhood or no cuisine, use an empty list [].\n"
        "- Output only the JSON array, no prose, no markdown fences.\n"
    )


def _extract_demand_batch(
    messages: list[str], llm: Any | None
) -> list[tuple[list[str], list[str]]]:
    """Call the LLM once to extract (neighborhoods, cuisines) for each message.

    Returns a parallel list of ``(neighborhoods, cuisines)`` pairs — one per
    input message.  When ``llm is None``, returns all-empty pairs (the caller's
    lexical pre-passes have already resolved any lexically-mappable rows; rows
    that reach this function with ``llm=None`` are counted as ``unmapped``).

    Catalog filtering is applied to BOTH axes so off-catalog LLM answers are
    silently dropped (T-18-02-INJ residual + ROUND-3 catalog-constraint).
    Fence-tolerant JSON parsing mirrors ``_parse_proposals``/``_FENCE_RE``.
    """
    empty = [([], [])] * len(messages)
    if not messages:
        return []
    if llm is None:
        return empty

    prompt = _build_demand_batch_prompt(messages)
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        if not isinstance(raw, str):
            _log.warning("LLM returned non-string in demand batch; using empty pairs")
            return empty
        cleaned = _FENCE_RE.sub("", raw).strip()
        items = json.loads(cleaned)
        if not isinstance(items, list):
            _log.warning("LLM demand batch returned non-list; using empty pairs")
            return empty
    except (json.JSONDecodeError, Exception) as exc:
        _log.warning("LLM demand batch parse error (%s); using empty pairs", exc)
        return empty

    results: list[tuple[list[str], list[str]]] = []
    for i, _msg in enumerate(messages):
        if i >= len(items) or not isinstance(items[i], dict):
            results.append(([], []))
            continue
        item = items[i]
        raw_neighborhoods = item.get("neighborhoods", [])
        raw_cuisines = item.get("cuisines", [])
        catalog_hoods = [n for n in raw_neighborhoods if n in _NEIGHBORHOODS_SET]
        catalog_cuiss = [c for c in raw_cuisines if c in _CUISINES_SET]
        results.append((catalog_hoods, catalog_cuiss))
    return results


# ---------------------------------------------------------------------------
# Two-DB connection helper (Task 2, D-05 prod-read / sandbox-write split)
# ---------------------------------------------------------------------------


@contextmanager
def get_demand_conn(url: str):  # type: ignore[return]
    """Direct non-pooled read-only connection to DEMAND_DATABASE_URL.

    Opens a psycopg2 connection (not the pool) so it targets the prod DB
    without touching the shared pool's DATABASE_URL or its lru_cache settings.
    Set to read-only so it can NEVER be used for writes (T-18-02-PROD).

    Mirrors the ``_snapshot_ids_from_url`` pattern in loop_falsifier.py.
    """
    with contextlib.closing(psycopg2.connect(url)) as conn:
        conn.set_session(readonly=True, autocommit=True)
        yield conn


# ---------------------------------------------------------------------------
# Demand reader (Task 2)
# ---------------------------------------------------------------------------


def gather_demand(
    days: int,
    url: str | None = None,
) -> tuple[dict[tuple[str, str], int], int, int]:
    """Read user_query_log and return catalog-constrained demand counts.

    Two-tier extraction on BOTH axes:

    Cuisine axis (ROUND-3 HIGH — symmetric to neighborhood):
      1. _types_to_cuisines(requested_primary_types)  [no LLM]
      2. _lexical_cuisines(message)                   [no LLM — tier 2a]
      3. _extract_demand_batch (combined with neighborhood misses) [LLM — tier 2b]

    Neighborhood axis:
      1. _lexical_neighborhoods(message)  [no LLM]
      2. _extract_demand_batch            [LLM — combined single call with cuisine misses]

    Returns:
        (demand_counts, rows_scanned, unmapped_count)

        demand_counts: dict[(neighborhood, cuisine) → count] — only catalog pairs
        rows_scanned: total rows fetched from user_query_log
        unmapped_count: rows that yielded no catalog bucket on either axis
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    sql = """
        SELECT message, COALESCE(requested_primary_types, '{}')
        FROM user_query_log
        WHERE created_at >= %s
        ORDER BY created_at DESC
    """

    ctx = get_demand_conn(url) if url is not None else get_conn()
    with ctx as conn, conn.cursor() as cur:
        cur.execute(sql, [cutoff])
        rows = cur.fetchall()

    llm = vibe.make_judge()

    # --- Per-row two-tier pre-pass (no LLM) ---
    # Each entry: (message, resolved_neighborhoods, resolved_cuisines, needs_llm_hood, needs_llm_cuisine)
    pre_resolved: list[tuple[str, list[str], list[str], bool, bool]] = []

    for message, primary_types in rows:
        # Neighborhood tier-1: lexical
        hoods = _lexical_neighborhoods(message)
        needs_llm_hood = len(hoods) == 0

        # Cuisine tier-1: types lookup
        cuiss = _types_to_cuisines(list(primary_types))
        if not cuiss:
            # Cuisine tier-2a: lexical message scan
            cuiss = _lexical_cuisines(message)
        needs_llm_cuisine = len(cuiss) == 0

        pre_resolved.append((message, hoods, cuiss, needs_llm_hood, needs_llm_cuisine))

    # --- Single combined LLM call for all misses (D-01 + ROUND-3) ---
    # Collect rows that still need the LLM on at least one axis
    miss_indices: list[int] = []
    miss_messages: list[str] = []
    for idx, (message, _hoods, _cuiss, needs_llm_hood, needs_llm_cuisine) in enumerate(
        pre_resolved
    ):
        if needs_llm_hood or needs_llm_cuisine:
            miss_indices.append(idx)
            miss_messages.append(message)

    llm_results: list[tuple[list[str], list[str]]] = []
    if miss_messages:
        llm_results = _extract_demand_batch(miss_messages, llm)

    # Merge LLM results back into pre_resolved
    for batch_pos, orig_idx in enumerate(miss_indices):
        message, hoods, cuiss, needs_llm_hood, needs_llm_cuisine = pre_resolved[orig_idx]
        if batch_pos < len(llm_results):
            llm_hoods, llm_cuiss = llm_results[batch_pos]
        else:
            llm_hoods, llm_cuiss = [], []

        if needs_llm_hood:
            hoods = llm_hoods
        if needs_llm_cuisine:
            cuiss = llm_cuiss
        pre_resolved[orig_idx] = (message, hoods, cuiss, needs_llm_hood, needs_llm_cuisine)

    # --- Accumulate demand counts ---
    demand_counts: dict[tuple[str, str], int] = defaultdict(int)
    unmapped_count = 0

    for _message, hoods, cuiss, _nh, _nc in pre_resolved:
        # Cartesian product within the row (REVIEW MEDIUM + ROUND-3 cuisine cross-product)
        pairs = [(n, c) for n in hoods for c in cuiss]
        if pairs:
            for pair in pairs:
                demand_counts[pair] += 1
        else:
            unmapped_count += 1

    return dict(demand_counts), len(rows), unmapped_count


@dataclass
class DemandGap:
    neighborhood: str
    cuisine: str
    place_count: int  # pair-level supply (TRUE pair count from place_query_hits)
    demand_count: int


def gap_to_seed_query(neighborhood: str, cuisine: str) -> str:
    """Return the exact loop-consumable seed string for a (neighborhood, cuisine) pair.

    Format matches build_seed_queries():
        "{cuisine} restaurants in {neighborhood} San Francisco"

    Raises AssertionError when either argument is not a catalog member —
    off-catalog inputs would produce un-ingestable seeds (D-03, T-18-03-SEED).
    """
    assert neighborhood in _NEIGHBORHOODS_SET, (
        f"gap_to_seed_query: {neighborhood!r} is not a catalog neighborhood"
    )
    assert cuisine in _CUISINES_SET, f"gap_to_seed_query: {cuisine!r} is not a catalog cuisine"
    seed = f"{cuisine} restaurants in {neighborhood} San Francisco"
    # Catalog-membership assertion so every emitted seed is loop-consumable
    # (premark_seed_isolation membership — D-03).  Evaluated lazily to avoid
    # importing build_seed_queries at module level on every startup.
    from scripts.ingest_places_sf import build_seed_queries

    assert seed in set(build_seed_queries()), (
        f"gap_to_seed_query: emitted seed {seed!r} is not in build_seed_queries() — "
        "this should never happen for valid catalog inputs"
    )
    return seed


def gather_pair_supply(pairs: list[tuple[str, str]], conn=None) -> dict[tuple[str, str], int]:
    """Count DISTINCT places for each (neighborhood, cuisine) pair via place_query_hits.

    For each pair we compute the exact seed query_text (via gap_to_seed_query) and
    issue ONE parameterised SELECT that counts DISTINCT place_id grouped by query_text.
    This is TRUE pair-level supply (REVIEW HIGH-1) — it counts places that matched
    the neighborhood-AND-cuisine seed, so a cuisine present city-wide but absent in
    the demanded neighborhood's seed returns 0 for that pair.

    Seed strings are passed as a %s param list (NEVER interpolated — T-18-03-SQLi).
    Never-ingested pairs yield 0 (no rows in place_query_hits for that seed).

    Args:
        pairs: List of (neighborhood, cuisine) tuples to score.
        conn: Optional already-open connection; if None, opens a pooled connection.

    Returns:
        Dict mapping each (neighborhood, cuisine) pair to its place count (0 if absent).
    """
    if not pairs:
        return {}

    # Build seed→pair mapping (catalog-validated)
    seed_to_pair: dict[str, tuple[str, str]] = {}
    for n, c in pairs:
        seed = gap_to_seed_query(n, c)
        seed_to_pair[seed] = (n, c)

    seed_strings = list(seed_to_pair.keys())

    sql = """
        SELECT query_text, COUNT(DISTINCT place_id) AS cnt
        FROM place_query_hits
        WHERE query_text = ANY(%s)
        GROUP BY query_text
    """

    ctx = conn if conn is not None else get_conn()
    # When conn is provided directly (not a context manager), we need different handling
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, [seed_strings])
            rows = cur.fetchall()
    else:
        with ctx as c2, c2.cursor() as cur:
            cur.execute(sql, [seed_strings])
            rows = cur.fetchall()

    # Initialise all pairs to 0 (never-ingested → 0)
    result: dict[tuple[str, str], int] = {pair: 0 for pair in pairs}
    for query_text, cnt in rows:
        pair = seed_to_pair.get(query_text)
        if pair is not None:
            result[pair] = int(cnt)
    return result


def find_demand_gaps(
    demand_counts: dict[tuple[str, str], int],
    pair_supply: dict[tuple[str, str], int],
    min_place_count: int = 5,
) -> list[DemandGap]:
    """Apply the D-02 filter at TRUE pair level: demand > 0 AND pair supply < floor.

    Returns a list of DemandGap instances sorted by demand_count descending.
    Pairs with demand == 0 are excluded regardless of supply.
    Pairs with pair supply >= min_place_count are excluded regardless of demand.

    This implements REVIEW HIGH-1: the supply check is against the PAIR-level count
    from place_query_hits, NOT the city-wide per-cuisine count — so "Vietnamese
    everywhere in SF but zero in Outer Sunset" correctly flags (Outer Sunset, vietnamese)
    as a gap.
    """
    gaps: list[DemandGap] = []
    for pair, demand in demand_counts.items():
        if demand <= 0:
            continue
        supply = pair_supply.get(pair, 0)
        if supply < min_place_count:
            neighborhood, cuisine = pair
            gaps.append(DemandGap(neighborhood, cuisine, supply, demand))
    gaps.sort(key=lambda g: g.demand_count, reverse=True)
    return gaps


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
        cur.execute(
            "SELECT query_text FROM places_ingest_query_checkpoints"
            " UNION ALL SELECT query_text FROM places_ingest_query_proposals"
        )
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
