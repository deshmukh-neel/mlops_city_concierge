"""Pure falsifier core logic: no DB, MLflow, LLM, or network calls.

This module owns:
- hit@k metric calculation
- newly-ingested target-set computation
- sandbox-vs-production safety checks
- non-circular paraphrase checks
- exit-code constants mirrored by the orchestrator

All functions are pure. The orchestrator supplies live inputs and handles I/O,
keeping this module cheap to test and safe to import.

Import contract: stdlib only + dataclasses. No mlflow, openai, psycopg2, app.db,
or semantic_search imports here.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Retrieval window used for hit@k — mirrors a realistic itinerary candidate window.
K = 5  # int

#: Number of held-out paraphrases per falsifier run.
N = 5  # int

#: Exit code: PASS — strictly-positive before→after delta confirmed.
EXIT_PASS = 0  # int

#: Exit code: FAIL — delta not strictly positive (expected falsifier outcome).
EXIT_FAIL = 1  # int

#: Exit code: infrastructure / precondition error (sandbox URL unset/unsafe,
#: malformed state, embed produced nothing, non-circularity violation).
EXIT_INFRA = 2  # int

#: Default quality floor for the populated-corpus gate (D-05).
#: Runtime-tunable via env LOOP_HIT_RATE_FLOOR or CLI --floor.
#: First run uses strict-positive-delta only (floor==0.0 reduces to delta check).
#: The orchestrator (19-04) overrides this via env/CLI; ratchet committed default
#: after calibration run.
FLOOR: float = 0.0  # updated after calibration run


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
class RecallAtKResult:
    """Result of compute_recall_at_k."""

    found_count: int  # distinct new IDs retrieved across all paraphrases' top-k
    total_count: int  # len(newly_ingested_ids)
    recall: float  # found_count / total_count  (0.0 when total_count == 0)


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
    one place_id in its top-K results appears in *newly_ingested_ids*.

    The caller is responsible for passing lists of at most K elements. An
    AssertionError is raised if any list exceeds K — this enforces a single source
    of truth: the retrieval window is set once (semantic_search(k=K)) and this
    function does not silently clamp it. (IN-02)

    Args:
        per_paraphrase_topk: One list of place_id strings per paraphrase.
            Each inner list must have at most K entries.
        newly_ingested_ids: Set of place_ids newly added by the ingest step.

    Returns:
        HitRateResult with hit_count, n (= len(per_paraphrase_topk)), and hit_rate.
        Empty input returns hit_count=0, n=0, hit_rate=0.0 (no ZeroDivisionError).
    """
    n = len(per_paraphrase_topk)
    if n == 0:
        return HitRateResult(hit_count=0, n=0, hit_rate=0.0)

    for topk in per_paraphrase_topk:
        assert len(topk) <= K, (  # noqa: S101
            f"compute_hit_rate received a top-k list of length {len(topk)} > K={K}. "
            "The caller must pass semantic_search(k=K) results — do not exceed K. (IN-02)"
        )

    hit_count = sum(1 for topk in per_paraphrase_topk if set(topk) & newly_ingested_ids)
    return HitRateResult(hit_count=hit_count, n=n, hit_rate=hit_count / n)


def compute_recall_at_k(
    per_paraphrase_topk: list[list[str]],
    newly_ingested_ids: set[str],
) -> RecallAtKResult:
    """Compute recall@K: what fraction of new embedded places are found across paraphrases.

    recall@K = #distinct newly_ingested_ids that appear in ANY paraphrase's top-K
               / total len(newly_ingested_ids).

    Each element of *per_paraphrase_topk* is an ordered list of place_id strings
    returned by semantic_search for one paraphrase. Distinct counting: the same new
    ID found in multiple paraphrases counts ONCE toward found_count (set union, D-03).

    The caller is responsible for passing lists of at most K elements. An
    AssertionError is raised if any list exceeds K — single source of truth for the
    retrieval window (IN-02), mirroring compute_hit_rate.

    Args:
        per_paraphrase_topk: One list of place_id strings per paraphrase.
            Each inner list must have at most K entries.
        newly_ingested_ids: Set of place_ids that are newly ingested (DB-diff, D-03).

    Returns:
        RecallAtKResult with found_count, total_count, recall.
        Empty newly_ingested_ids returns found_count=0, total_count=0, recall=0.0
        (no ZeroDivisionError).
    """
    total_count = len(newly_ingested_ids)
    if total_count == 0:
        return RecallAtKResult(found_count=0, total_count=0, recall=0.0)

    for topk in per_paraphrase_topk:
        assert len(topk) <= K, (  # noqa: S101
            f"compute_recall_at_k received a top-k list of length {len(topk)} > K={K}. "
            "The caller must pass semantic_search(k=K) results — do not exceed K. (IN-02)"
        )

    found: set[str] = set()
    for topk in per_paraphrase_topk:
        found |= set(topk) & newly_ingested_ids
    found_count = len(found)
    return RecallAtKResult(
        found_count=found_count,
        total_count=total_count,
        recall=found_count / total_count,
    )


def is_pass(after_hit_rate: float) -> bool:
    """Return True when after_hit_rate is strictly positive.

    PASS iff at least one paraphrase retrieved a newly-ingested place in top-K.
    This is the literal FALSIFY-01 'strictly positive delta' rule when the sandbox
    starts empty (before hit@k = 0 by construction).
    """
    return after_hit_rate > 0.0


def is_strictly_positive_delta(before_hit_rate: float, after_hit_rate: float) -> bool:
    """Return True when (after - before) > 0.0.

    Distinct from is_pass: this handles the general case where before may be non-zero
    (e.g. a non-empty sandbox). The orchestrator may call either depending on context;
    the gate definition is always strictly-positive delta.
    """
    return (after_hit_rate - before_hit_rate) > 0.0


def decide_loop_exit(
    before_rate: float,
    after_rate: float,
    floor: float,
    guard_violation: GuardResult | None,
    embed_added_count: int,
) -> int:
    """Pure gate function for the Phase 19 populated-baseline loop exit decision (D-05/D-06).

    Priority order (highest first):
    1. guard_violation present with ok=False → EXIT_INFRA
    2. embed_added_count == 0 → EXIT_INFRA  (D-02 empty-diff loud-fail)
    3. strictly-positive delta AND after_rate >= floor → EXIT_PASS
    4. else → EXIT_FAIL

    When floor == 0.0 (first-run default), condition 3 reduces to strict-positive-delta
    only — any improvement above zero passes (D-05). The orchestrator overrides floor
    via env LOOP_HIT_RATE_FLOOR or CLI --floor; plan 19-04 ratchets the committed default
    after observing a calibration run.

    NOTE: Unlike loop_falsifier.decide_exit, this function does NOT include the
    `before_rate != 0.0 → EXIT_INFRA` branch. The populated baseline legitimately scores
    before_hit@k = 0 against the v2-diff target set (D-03), so a non-zero before is
    caught upstream, not here. This is a NEW sibling function; loop_falsifier.py and its
    decide_exit are left UNTOUCHED (D-07).

    Args:
        before_rate: hit@k rate measured BEFORE ingest on the populated baseline.
        after_rate: hit@k rate measured AFTER ingest.
        floor: minimum after_rate required for EXIT_PASS (D-05 calibrated bar).
        guard_violation: GuardResult from check_prod_safety or check_non_circularity,
            or None if all guards passed.
        embed_added_count: number of embeddings added by the embed step (D-02).

    Returns:
        EXIT_PASS (0), EXIT_FAIL (1), or EXIT_INFRA (2).
    """
    if guard_violation is not None and not guard_violation.ok:
        return EXIT_INFRA
    if embed_added_count == 0:
        return EXIT_INFRA
    if is_strictly_positive_delta(before_rate, after_rate) and after_rate >= floor:
        return EXIT_PASS
    return EXIT_FAIL


# ---------------------------------------------------------------------------
# DB-diff helper
# ---------------------------------------------------------------------------


def db_diff(before_ids: set[str], after_ids: set[str]) -> set[str]:
    """Return place_ids present after ingest but absent before.

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
    table so the ingest script processes only the chosen seed gap. Pure set
    logic; the orchestrator supplies the catalog.

    If *chosen_seed_query* is not in the catalog, the full catalog is returned
    (the orchestrator must treat this as a precondition error).

    Duplicate entries in *all_seed_queries* collapse to a set.
    """
    return set(all_seed_queries) - {chosen_seed_query}


# ---------------------------------------------------------------------------
# URL normalization helpers
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> tuple[str, str, str, str]:
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
    allow_remote: bool = False,
) -> GuardResult:
    """Assert that sandbox_url is safe to use.

    Returns a violation GuardResult when:
    - sandbox_url is None or empty (SANDBOX_DATABASE_URL unset)
    - sandbox_url targets a Cloud SQL instance and allow_remote is False
    - sandbox_url's normalized target equals resolvable_prod_url's normalized target
    - both share the same non-empty cloud_sql_instance connection name

    Returns ok=True when resolvable_prod_url is None (prod URL unresolvable means no
    known collision; the provisioning script is the belt-and-suspenders).

    Collision check semantics:
    - TCP path: compares (host, dbname) — port is intentionally NOT included because the
      legitimate local workflow uses prod on :5432 (Postgres.app) and sandbox on :5433
      (Docker). Including port would cause false-negatives on that setup. Same-host +
      same-dbname is the meaningful signal; the _sandbox dbname suffix convention is the
      primary guard against accidental collisions that differ only by port.
    - Cloud SQL path: compares instance connection names (same instance = same server
      regardless of dbname or port), which is a stricter check than (host, dbname).
    - Different-dbname on the same host is considered safe by design: the _sandbox suffix
      enforced by the provisioner is the real guard, and the TCP collision check only
      fires when BOTH host AND dbname match exactly.

    Args:
        sandbox_url: The SANDBOX_DATABASE_URL value (may be None/empty).
        resolvable_prod_url: The resolved prod DATABASE_URL (may be None — means the
            orchestrator could not resolve a prod URL; treated as no collision).
        allow_remote: When True, suppress the Cloud SQL sandbox URL rejection. Set via
            bool(os.environ.get("SANDBOX_ALLOW_REMOTE")) in the orchestrator — never read
            os.environ here to keep this module import-pure and unit-testable.
    """
    if not sandbox_url:
        return GuardResult(
            ok=False,
            message="SANDBOX_DATABASE_URL is unset or empty — refusing to run against an unknown database.",
        )

    sb_host, sb_port, sb_dbname, sb_instance = normalize_url(sandbox_url)

    # Reject Cloud SQL sandbox URLs unless explicitly allowed.
    # Mirrors the shell provisioner guard.
    # A Cloud SQL socket sandbox on a DIFFERENT instance than prod would pass the
    # instance-collision check below but still be a remote write target — reject it.
    if sb_instance and not allow_remote:
        return GuardResult(
            ok=False,
            message=(
                f"SANDBOX_DATABASE_URL targets a Cloud SQL instance ({sb_instance!r}); "
                "the falsifier sandbox must be a local Docker container. "
                "Set SANDBOX_ALLOW_REMOTE=1 to override."
            ),
        )

    if resolvable_prod_url is None:
        return GuardResult(
            ok=True, message="ok (prod URL not resolvable; no collision check possible)"
        )

    prod_host, prod_port, prod_dbname, prod_instance = normalize_url(resolvable_prod_url)

    # Cloud SQL instance collision (same instance = same server regardless of dbname)
    if sb_instance and prod_instance and sb_instance == prod_instance:
        return GuardResult(
            ok=False,
            message=(
                f"SANDBOX_DATABASE_URL targets the same Cloud SQL instance as prod "
                f"({sb_instance!r}). Refusing to proceed."
            ),
        )

    # TCP collision: same host AND same dbname (port intentionally excluded — see docstring)
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
    """Assert that no paraphrase is an exact-string match of any forbidden query.

    Non-circularity is enforced by exact-string, case- and whitespace-sensitive
    comparison. Semantic overlap is inherent and expected (same intent); this guard
    only catches literal string identity.

    The violation message names BOTH the offending paraphrase AND the forbidden source
    query it collided with.

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
