"""Integration tests for the place_relations builder (W7, TEST-02).

Gated on ``APP_ENV=integration``. Requires the W7 migration applied
(``make migrate``) against a real Postgres + pgvector. Seeds a deterministic
10-place fixture into ``places_raw`` + ``place_embeddings_v2``, runs the
builder, and asserts the KG-01/KG-02 schema contract plus the BLD-01..BLD-06
behaviours. All fixture rows use a ``KGT_`` place_id prefix and are deleted on
teardown so the suite is safe to run against a shared DB.

Run with::

    make migrate
    APP_ENV=integration poetry run pytest \
        tests/integration/test_build_place_relations.py -v --no-cov
"""

from __future__ import annotations

import json
import math
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)

from app.db import get_conn  # noqa: E402
from scripts.build_place_relations import (  # noqa: E402
    SIMILAR_MIN_COS,
    build_near,
    build_same_neighborhood,
    main,
)

_PREFIX = "KGT_"
_EMBED_DIM = 1536

# Five places clustered tightly downtown (within ~800m of each other) plus
# five spread out. Three share the "Mission" neighborhood. Two carry
# containingPlaces; three carry addressDescriptor.landmarks.
_CLUSTER = [
    ("KGT_a", 37.7880, -122.4074, "Mission"),
    ("KGT_b", 37.7883, -122.4078, "Mission"),
    ("KGT_c", 37.7886, -122.4071, "Mission"),
    ("KGT_d", 37.7879, -122.4069, "SoMa"),
    ("KGT_e", 37.7884, -122.4080, "SoMa"),
]
_FAR = [
    ("KGT_f", 37.8100, -122.4770, "Presidio"),
    ("KGT_g", 37.7600, -122.4350, "Noe Valley"),
    ("KGT_h", 37.7350, -122.5050, "Sunset"),
    ("KGT_i", 37.8050, -122.4100, "North Beach"),
    ("KGT_j", 37.7700, -122.3900, "Dogpatch"),
]
_ALL = _CLUSTER + _FAR


def _source_json(place_id: str, neighborhood: str) -> dict:
    sj: dict = {"addressComponents": [{"types": ["neighborhood"], "longText": neighborhood}]}
    # Two places get containingPlaces edges.
    if place_id == "KGT_a":
        sj["containingPlaces"] = [{"id": "KGT_PARENT_1"}, {"id": "KGT_b"}]
    if place_id == "KGT_c":
        sj["containingPlaces"] = [{"id": "KGT_PARENT_2"}]
    # Three places get addressDescriptor.landmarks edges.
    if place_id in ("KGT_a", "KGT_d", "KGT_f"):
        sj["addressDescriptor"] = {
            "landmarks": [
                {
                    "placeId": f"KGT_LM_{place_id}",
                    "displayName": {"text": f"Landmark for {place_id}"},
                    "types": ["tourist_attraction"],
                    "travelDistanceMeters": 123.5,
                }
            ]
        }
    return sj


@pytest.fixture
def _writable_or_skip() -> None:
    """Skip if the current DB role can't write the fixture tables.

    Mirrors the guard in ``test_coverage_agent.py``: integration runs against
    the shared Cloud SQL instance authenticate as an IAM role that may have
    read-only access. Existence of the tables isn't enough — without the
    GRANTs the seed fixture hard-errors instead of skipping.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT bool_and(has_table_privilege(current_user, t, 'INSERT, DELETE')) "
            "FROM unnest(ARRAY['places_raw', 'place_embeddings_v2', "
            "'place_relations']) AS t"
        )
        if not cur.fetchone()[0]:
            pytest.skip(
                "current DB role lacks INSERT/DELETE on KG fixture tables "
                "(places_raw / place_embeddings_v2 / place_relations)"
            )


@pytest.fixture
def seed_10_places(_writable_or_skip):
    rows = [
        (
            pid,
            f"Place {pid}",
            "restaurant",
            lat,
            lng,
            "OPERATIONAL",
            json.dumps(_source_json(pid, nbhd)),
        )
        for pid, lat, lng, nbhd in _ALL
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO places_raw
                    (place_id, name, primary_type, latitude, longitude,
                     business_status, source_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (place_id) DO NOTHING
                """,
                rows,
            )
        conn.commit()
    yield
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM place_relations WHERE src_place_id LIKE %s",
                [_PREFIX + "%"],
            )
            cur.execute(
                "DELETE FROM place_embeddings_v2 WHERE place_id LIKE %s",
                [_PREFIX + "%"],
            )
            cur.execute(
                "DELETE FROM places_raw WHERE place_id LIKE %s",
                [_PREFIX + "%"],
            )
        conn.commit()


def _deterministic_vector(seed: int) -> list[float]:
    """Reproducible, dependency-free pseudo-vector keyed by ``seed``.

    A sinusoid over the dimension index — fully deterministic, no RNG (ruff
    S311), and cosine similarity between two such vectors is a smooth function
    of their phase offset, which is exactly what the threshold test needs.
    """
    return [math.sin((i + 1) * 0.001 + seed) for i in range(_EMBED_DIM)]


@pytest.fixture
def seed_embeddings_v2(seed_10_places):
    # KGT_a / KGT_b get near-identical phases so their cosine exceeds the 0.65
    # threshold; every other place gets a well-separated phase.
    vectors: dict[str, list[float]] = {}
    for idx, (pid, _, _, _) in enumerate(_ALL):
        if pid == "KGT_a":
            vectors[pid] = _deterministic_vector(0.0)
        elif pid == "KGT_b":
            vectors[pid] = _deterministic_vector(0.0005)
        else:
            vectors[pid] = _deterministic_vector(10.0 + idx * 5.0)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO place_embeddings_v2
                    (place_id, embedding, embedding_model, embedding_text,
                     source_updated_at)
                VALUES (%s, %s, 'test-model', 'text', NOW())
                ON CONFLICT (place_id) DO NOTHING
                """,
                [(pid, "[" + ",".join(str(x) for x in vec) + "]") for pid, vec in vectors.items()],
            )
        conn.commit()
    yield


def _count(relation_type: str) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM place_relations "
            "WHERE relation_type = %s AND src_place_id LIKE %s",
            [relation_type, _PREFIX + "%"],
        )
        row = cur.fetchone()
        return row[0] if row else 0


def test_full_build(seed_embeddings_v2) -> None:
    """BLD-02..BLD-06: every relation type populates on the fixture."""
    assert main([]) == 0
    for rt in (
        "NEAR",
        "SAME_NEIGHBORHOOD",
        "CONTAINED_IN",
        "NEAR_LANDMARK",
        "SIMILAR_VECTOR",
    ):
        assert _count(rt) > 0, f"{rt} produced no rows"


def test_near_symmetric_within_tolerance(seed_embeddings_v2) -> None:
    assert main(["--only", "NEAR"]) == 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT src_place_id, dst_place_id, weight FROM place_relations "
            "WHERE relation_type = 'NEAR' AND src_place_id LIKE %s",
            [_PREFIX + "%"],
        )
        edges = {(s, d): w for s, d, w in cur.fetchall()}
    assert edges
    for (s, d), w in edges.items():
        assert (d, s) in edges, f"missing reverse edge for ({s}, {d})"
        assert abs(edges[(d, s)] - w) < 1e-6


def test_contained_in_populates_from_source_json(seed_embeddings_v2) -> None:
    assert main(["--only", "CONTAINED_IN"]) == 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT src_place_id, dst_place_id FROM place_relations "
            "WHERE relation_type = 'CONTAINED_IN' AND src_place_id LIKE %s",
            [_PREFIX + "%"],
        )
        edges = set(cur.fetchall())
    assert ("KGT_a", "KGT_PARENT_1") in edges
    assert ("KGT_a", "KGT_b") in edges
    assert ("KGT_c", "KGT_PARENT_2") in edges


def test_similar_vector_threshold(seed_embeddings_v2) -> None:
    assert main(["--only", "SIMILAR_VECTOR"]) == 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT weight FROM place_relations "
            "WHERE relation_type = 'SIMILAR_VECTOR' AND src_place_id LIKE %s",
            [_PREFIX + "%"],
        )
        weights = [w for (w,) in cur.fetchall()]
    assert weights, "expected at least one SIMILAR_VECTOR edge (KGT_a/KGT_b)"
    for w in weights:
        assert SIMILAR_MIN_COS <= w <= 1.0 + 1e-9


def test_idempotent(seed_embeddings_v2) -> None:
    """BLD-01: re-running yields zero PK growth (assert on COUNT, not built_at)."""
    assert main([]) == 0
    first = {
        rt: _count(rt)
        for rt in ("NEAR", "SAME_NEIGHBORHOOD", "CONTAINED_IN", "NEAR_LANDMARK", "SIMILAR_VECTOR")
    }
    assert main([]) == 0
    second = {rt: _count(rt) for rt in first}
    assert first == second


def test_only_flag_subset(seed_embeddings_v2) -> None:
    """BLD-01: --only NEAR rebuilds only NEAR, leaving others untouched."""
    assert main([]) == 0
    before = {
        rt: _count(rt)
        for rt in ("SAME_NEIGHBORHOOD", "CONTAINED_IN", "NEAR_LANDMARK", "SIMILAR_VECTOR")
    }
    assert main(["--only", "NEAR"]) == 0
    after = {rt: _count(rt) for rt in before}
    assert before == after


def test_neighborhood_of_helper(seed_10_places) -> None:
    """KG-02: neighborhood_of returns the SF neighborhood text."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT neighborhood_of("
            '\'{"addressComponents":[{"types":["neighborhood"],'
            '"longText":"Mission"}]}\'::jsonb)'
        )
        row = cur.fetchone()
    assert row is not None and row[0] == "Mission"


def test_place_relations_table_shape() -> None:
    """KG-01: place_relations has exactly the spec'd column set."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'place_relations'"
        )
        cols = {r[0] for r in cur.fetchall()}
    assert cols == {
        "src_place_id",
        "dst_place_id",
        "relation_type",
        "weight",
        "metadata",
        "source",
        "built_at",
    }


def test_sub_builders_are_importable() -> None:
    """Smoke: the sub-builder functions referenced by the suite import."""
    assert callable(build_near)
    assert callable(build_same_neighborhood)
