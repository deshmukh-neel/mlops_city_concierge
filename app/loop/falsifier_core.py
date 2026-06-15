"""Pure falsifier core logic — no DB, no MLflow, no LLM, no network.

This module owns:
- hit@k metric (k=5, N=5 as module constants, per D-05)
- DB-diff target-set computation (D-03 / FALSIFY-01c)
- Prod-safety guard: hard-assert SANDBOX_DATABASE_URL is set and differs from prod (D-12)
- Non-circularity disjointness assertion (D-07)
- Exit-code constants mirroring scripts/eval_falsifier.py (D-09)

All functions are pure (no side effects, no I/O). The orchestrator (16-03) supplies
live inputs and calls these functions; this module stays fully unit-testable at zero
API cost.

Import contract: stdlib only + dataclasses. No mlflow, openai, psycopg2, app.db,
or semantic_search imports here.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit

# ---------------------------------------------------------------------------
# Module constants (D-05, D-09)
# ---------------------------------------------------------------------------

#: Retrieval window used for hit@k — mirrors a realistic itinerary candidate window.
K = 5  # int

#: Number of held-out paraphrases per falsifier run (D-05).
N = 5  # int

#: Exit code: PASS — strictly-positive before→after delta confirmed.
EXIT_PASS = 0  # int

#: Exit code: FAIL — delta not strictly positive (expected falsifier outcome).
EXIT_FAIL = 1  # int

#: Exit code: infrastructure / precondition error (sandbox URL unset/unsafe,
#: malformed state, embed produced nothing, non-circularity violation).
EXIT_INFRA = 2  # int


# ---------------------------------------------------------------------------
# Result types (typed returns so the orchestrator gets structured data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HitRateResult:
    """Result of compute_hit_rate."""

    hit_count: int
    n: int
    hit_rate: float


@dataclass(frozen=True)
class GuardResult:
    """Result of a guard check (prod-safety or non-circularity)."""

    ok: bool
    message: str


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def compute_hit_rate(
    per_paraphrase_topk: list[list[str]],
    newly_ingested_ids: set[str],
) -> HitRateResult:
    """Compute hit@K over N held-out paraphrases.

    Each element of *per_paraphrase_topk* is an ordered list of place_id strings
    returned by semantic_search for one paraphrase. A paraphrase "hits" iff at least
    one place_id in its top-K results (defensively truncated to K entries inside this
    function, even if the caller forgot) appears in *newly_ingested_ids*.

    Args:
        per_paraphrase_topk: One list of place_id strings per paraphrase.
        newly_ingested_ids: Set of place_ids that are newly ingested (DB-diff, D-03).

    Returns:
        HitRateResult with hit_count, n (= len(per_paraphrase_topk)), and hit_rate.
        Empty input returns hit_count=0, n=0, hit_rate=0.0 (no ZeroDivisionError).
    """
    n = len(per_paraphrase_topk)
    if n == 0:
        return HitRateResult(hit_count=0, n=0, hit_rate=0.0)

    hit_count = sum(
        1
        for topk in per_paraphrase_topk
        if set(topk[:K]) & newly_ingested_ids  # defensive truncation to K
    )
    return HitRateResult(hit_count=hit_count, n=n, hit_rate=hit_count / n)


def is_pass(after_hit_rate: float) -> bool:
    """Return True iff after_hit_rate is strictly positive (D-04).

    PASS iff at least one paraphrase retrieved a newly-ingested place in top-K.
    This is the literal FALSIFY-01 'strictly positive delta' rule when the sandbox
    starts empty (before hit@k = 0 by construction).
    """
    return after_hit_rate > 0.0


def is_strictly_positive_delta(before_hit_rate: float, after_hit_rate: float) -> bool:
    """Return True iff (after - before) > 0.0 — the literal FALSIFY-01(f) rule (D-04).

    Distinct from is_pass: this handles the general case where before may be non-zero
    (e.g. a non-empty sandbox). The orchestrator may call either depending on context;
    the gate definition is always strictly-positive delta.
    """
    return (after_hit_rate - before_hit_rate) > 0.0


# ---------------------------------------------------------------------------
# DB-diff helper
# ---------------------------------------------------------------------------


def db_diff(before_ids: set[str], after_ids: set[str]) -> set[str]:
    """Return the set of place_ids present after ingest but absent before (D-03 / FALSIFY-01c).

    This is the expected target set: newly-ingested places that the embed step should
    have made retrievable. NOT proposal `applied` status — this is a raw DB diff.
    """
    return after_ids - before_ids


# ---------------------------------------------------------------------------
# Seed-isolation helper
# ---------------------------------------------------------------------------


def build_premark_set(all_seed_queries: list[str], chosen_seed_query: str) -> set[str]:
    """Return catalog minus the chosen seed — the pre-mark set for the orchestrator.

    The orchestrator pre-marks the returned set as 'completed' in the proposals
    table so the ingest script processes ONLY the chosen seed gap (Codex HIGH
    seed-isolation fix). Pure set logic; the orchestrator supplies the catalog.

    If *chosen_seed_query* is not in the catalog, the full catalog is returned
    (the orchestrator must treat this as a precondition error — see plan 16-03).

    Duplicate entries in *all_seed_queries* collapse to a set.
    """
    return set(all_seed_queries) - {chosen_seed_query}


# ---------------------------------------------------------------------------
# URL normalization helpers (Cloud SQL-aware, per D-12 / Codex MEDIUM)
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> tuple[str, str, str, str]:
    """Normalize a PostgreSQL URL to (host, port, dbname, cloud_sql_instance).

    Handles both TCP URLs (postgresql://user:pw@host:port/dbname) and Cloud SQL
    socket URLs (postgresql://user:pw@/dbname?host=/cloudsql/<instance>) where
    urlsplit().hostname is None (empty netloc) and the socket path lives in the
    `host` query-string parameter.

    Returns a 4-tuple:
        host: hostname or socket path segment (empty string if unresolvable)
        port: port as string (empty string if absent)
        dbname: URL-decoded path stripped of leading '/'
        cloud_sql_instance: instance connection name when present in ?host= param,
            e.g. 'proj:region:inst'; empty string otherwise.
    """
    parsed = urlsplit(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)

    cloud_sql_instance = ""
    host = parsed.hostname or ""
    port = str(parsed.port) if parsed.port else ""
    dbname = unquote(parsed.path.lstrip("/"))

    if not host:
        # Cloud SQL socket URL: ?host=/cloudsql/<instance>
        host_params = qs.get("host", [])
        if host_params:
            raw_host = host_params[0]
            # Extract the instance connection name from /cloudsql/<instance>
            if "/cloudsql/" in raw_host:
                cloud_sql_instance = raw_host.split("/cloudsql/", 1)[1]
            host = raw_host

    return host, port, dbname, cloud_sql_instance


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def check_prod_safety(
    sandbox_url: str | None,
    resolvable_prod_url: str | None,
) -> GuardResult:
    """Assert that sandbox_url is safe to use (D-12).

    Returns a violation GuardResult when:
    - sandbox_url is None or empty (SANDBOX_DATABASE_URL unset)
    - sandbox_url's normalized target equals resolvable_prod_url's normalized target
    - both share the same non-empty cloud_sql_instance connection name

    Returns ok=True when resolvable_prod_url is None (prod URL unresolvable means no
    known collision; the provisioning script is the belt-and-suspenders).

    Args:
        sandbox_url: The SANDBOX_DATABASE_URL value (may be None/empty).
        resolvable_prod_url: The resolved prod DATABASE_URL (may be None — means the
            orchestrator could not resolve a prod URL; treated as no collision).
    """
    if not sandbox_url:
        return GuardResult(
            ok=False,
            message="SANDBOX_DATABASE_URL is unset or empty — refusing to run against an unknown database.",
        )

    if resolvable_prod_url is None:
        return GuardResult(
            ok=True, message="ok (prod URL not resolvable; no collision check possible)"
        )

    sb_host, sb_port, sb_dbname, sb_instance = _normalize_url(sandbox_url)
    prod_host, prod_port, prod_dbname, prod_instance = _normalize_url(resolvable_prod_url)

    # Cloud SQL instance collision (same instance = same server regardless of dbname)
    if sb_instance and prod_instance and sb_instance == prod_instance:
        return GuardResult(
            ok=False,
            message=(
                f"SANDBOX_DATABASE_URL targets the same Cloud SQL instance as prod "
                f"({sb_instance!r}). Refusing to proceed."
            ),
        )

    # TCP collision: same host AND same dbname
    if sb_host == prod_host and sb_dbname == prod_dbname:
        return GuardResult(
            ok=False,
            message=(
                f"SANDBOX_DATABASE_URL appears to target the same host+database as prod "
                f"(host={sb_host!r}, dbname={sb_dbname!r}). Refusing to proceed."
            ),
        )

    return GuardResult(ok=True, message="ok")


def check_non_circularity(
    paraphrases: list[str],
    forbidden_queries: list[str],
) -> GuardResult:
    """Assert that no paraphrase is an exact-string match of any forbidden query (D-07).

    Non-circularity is enforced by exact-string, case- and whitespace-sensitive
    comparison. Semantic overlap is inherent and expected (same intent); this guard
    only catches literal string identity.

    The violation message names BOTH the offending paraphrase AND the forbidden source
    query it collided with (Codex LOW — auditability).

    Args:
        paraphrases: LLM-generated paraphrase strings (from the frozen paraphrase file).
        forbidden_queries: The ingest seed query + any probe queries that defined the gap.
    """
    forbidden_set = set(forbidden_queries)
    for paraphrase in paraphrases:
        if paraphrase in forbidden_set:
            # Find the matching forbidden query for the auditability message
            matched = next(q for q in forbidden_queries if q == paraphrase)
            return GuardResult(
                ok=False,
                message=(
                    f"Non-circularity violation: paraphrase {paraphrase!r} is an exact "
                    f"match of forbidden query {matched!r}. Frozen paraphrase file must "
                    "be regenerated."
                ),
            )

    return GuardResult(ok=True, message="ok")
