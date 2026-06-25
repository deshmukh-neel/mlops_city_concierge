"""Idempotent builder for the W7 knowledge-graph edge table.

Seeds five edge types into ``place_relations`` from data already on hand
(no LLM calls):

  - ``NEAR``              — haversine <= 800m, both directions, weight = metres.
  - ``SAME_NEIGHBORHOOD`` — same ``neighborhood_of(source_json)`` bucket.
  - ``CONTAINED_IN``      — ``source_json -> containingPlaces``.
  - ``NEAR_LANDMARK``     — ``source_json -> addressDescriptor -> landmarks``.
  - ``SIMILAR_VECTOR``    — top-K cosine neighbours over ``place_embeddings_v2``.

Idempotency contract: re-running produces **zero PK growth**. Relations whose
weight can change (``NEAR``, ``NEAR_LANDMARK``, ``SIMILAR_VECTOR``) use
``ON CONFLICT ... DO UPDATE``; the rest use ``ON CONFLICT ... DO NOTHING``.
``built_at`` is rewritten on every UPDATE by design (it tracks "last touched"),
so idempotency must be asserted on ``COUNT(*)``, not on ``built_at``.

``SIMILAR_VECTOR`` reads ``place_embeddings_v2`` **only** — never the v1
``place_embeddings`` table. If the runtime retriever is in v1 mode the KG's
similarity edges will not agree with ``semantic_search``; this is an
intentional W7 design decision (the KG is tied to v2).

``get_conn()`` (app/db.py) borrows a pooled connection that does **not**
autocommit and is rolled back when returned to the pool, so this builder
calls ``conn.commit()`` explicitly after each sub-builder to scope failure
isolation and actually persist rows.

Usage::

    python scripts/build_place_relations.py                 # all five
    python scripts/build_place_relations.py --only NEAR
    python scripts/build_place_relations.py --only NEAR,SIMILAR_VECTOR
"""

from __future__ import annotations

import argparse
import sys

from psycopg2.extensions import connection

from app.db import get_conn

NEAR_RADIUS_M = 800
SIMILAR_TOPK = 10
SIMILAR_MIN_COS = 0.65

RELATION_TYPES = (
    "NEAR",
    "SAME_NEIGHBORHOOD",
    "CONTAINED_IN",
    "NEAR_LANDMARK",
    "SIMILAR_VECTOR",
)


def build_near(conn: connection) -> int:
    """NEAR: places within ``NEAR_RADIUS_M`` metres of each other.

    Edges are directed; the symmetric pair is written by ``UNION ALL``-ing the
    reverse direction. Haversine is identical to the formula used by
    ``app.tools.retrieval.nearby`` (Earth radius 6_371_000 m).
    """
    sql = """
        INSERT INTO place_relations
            (src_place_id, dst_place_id, relation_type, weight, metadata, source)
        WITH pairs AS (
            SELECT
                a.place_id AS src,
                b.place_id AS dst,
                6371000 * 2 * ASIN(SQRT(
                    POWER(SIN(RADIANS(b.latitude  - a.latitude)  / 2), 2) +
                    COS(RADIANS(a.latitude)) * COS(RADIANS(b.latitude)) *
                    POWER(SIN(RADIANS(b.longitude - a.longitude) / 2), 2)
                )) AS dist_m
            FROM places_raw a
            JOIN places_raw b
              ON a.place_id < b.place_id
            WHERE a.business_status = 'OPERATIONAL'
              AND b.business_status = 'OPERATIONAL'
              AND a.latitude  IS NOT NULL AND a.longitude IS NOT NULL
              AND b.latitude  IS NOT NULL AND b.longitude IS NOT NULL
        ),
        edges AS (
            SELECT src, dst, dist_m FROM pairs WHERE dist_m <= %(radius)s
            UNION ALL
            SELECT dst, src, dist_m FROM pairs WHERE dist_m <= %(radius)s
        )
        SELECT src, dst, 'NEAR', dist_m, '{}'::jsonb, 'haversine'
        FROM edges
        ON CONFLICT (src_place_id, dst_place_id, relation_type)
        DO UPDATE SET weight = EXCLUDED.weight, built_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"radius": NEAR_RADIUS_M})
        return cur.rowcount


def build_same_neighborhood(conn: connection) -> int:
    """SAME_NEIGHBORHOOD: places sharing a non-empty ``neighborhood_of`` bucket."""
    sql = """
        INSERT INTO place_relations
            (src_place_id, dst_place_id, relation_type, weight, metadata, source)
        SELECT a.place_id, b.place_id, 'SAME_NEIGHBORHOOD',
               NULL, '{}'::jsonb, 'address_components'
        FROM places_raw a
        JOIN places_raw b
          ON neighborhood_of(a.source_json) = neighborhood_of(b.source_json)
         AND neighborhood_of(a.source_json) <> ''
         AND a.place_id <> b.place_id
        ON CONFLICT (src_place_id, dst_place_id, relation_type) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount


def build_contained_in(conn: connection) -> int:
    """CONTAINED_IN: ``source_json -> containingPlaces[].id`` edges."""
    sql = """
        INSERT INTO place_relations
            (src_place_id, dst_place_id, relation_type, weight, metadata, source)
        SELECT p.place_id, elem->>'id', 'CONTAINED_IN',
               NULL, '{}'::jsonb, 'source_json'
        FROM places_raw p,
             jsonb_array_elements(p.source_json->'containingPlaces') AS elem
        WHERE elem->>'id' IS NOT NULL
          AND elem->>'id' <> p.place_id
        ON CONFLICT (src_place_id, dst_place_id, relation_type) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount


def build_near_landmark(conn: connection) -> int:
    """NEAR_LANDMARK: ``source_json -> addressDescriptor -> landmarks[]`` edges.

    Landmark ``placeId`` values may not exist in ``places_raw`` — that is the
    reason ``dst_place_id`` has no FK.
    """
    sql = """
        INSERT INTO place_relations
            (src_place_id, dst_place_id, relation_type, weight, metadata, source)
        SELECT
            p.place_id,
            lm->>'placeId',
            'NEAR_LANDMARK',
            (lm->>'travelDistanceMeters')::double precision,
            jsonb_build_object(
                'displayName', lm->'displayName'->>'text',
                'types', lm->'types'
            ),
            'source_json'
        FROM places_raw p,
             jsonb_array_elements(
                 p.source_json->'addressDescriptor'->'landmarks'
             ) AS lm
        WHERE lm->>'placeId' IS NOT NULL
          AND lm->>'placeId' <> p.place_id
        ON CONFLICT (src_place_id, dst_place_id, relation_type)
        DO UPDATE SET weight = EXCLUDED.weight,
                      metadata = EXCLUDED.metadata,
                      built_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount


def build_similar_vector(conn: connection) -> int:
    """SIMILAR_VECTOR: top-K cosine neighbours per row over ``place_embeddings_v2``.

    Reads ``place_embeddings_v2`` ONLY (never the v1 ``place_embeddings``
    table). ``a.embedding <=> b.embedding`` is pgvector cosine *distance*, so
    similarity is ``1 - distance`` and ordering ascending by distance == top
    similarity first.
    """
    sql = """
        INSERT INTO place_relations
            (src_place_id, dst_place_id, relation_type, weight, metadata, source)
        WITH ranked AS (
            SELECT
                a.place_id AS src,
                b.place_id AS dst,
                1 - (a.embedding <=> b.embedding) AS cos,
                ROW_NUMBER() OVER (
                    PARTITION BY a.place_id
                    ORDER BY a.embedding <=> b.embedding
                ) AS rn
            FROM place_embeddings_v2 a
            JOIN place_embeddings_v2 b
              ON a.place_id <> b.place_id
        )
        SELECT src, dst, 'SIMILAR_VECTOR', cos, '{}'::jsonb, 'vector_topk'
        FROM ranked
        WHERE rn <= %(topk)s AND cos >= %(min_cos)s
        ON CONFLICT (src_place_id, dst_place_id, relation_type)
        DO UPDATE SET weight = EXCLUDED.weight, built_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"topk": SIMILAR_TOPK, "min_cos": SIMILAR_MIN_COS})
        return cur.rowcount


SUB_BUILDERS = {
    "NEAR": build_near,
    "SAME_NEIGHBORHOOD": build_same_neighborhood,
    "CONTAINED_IN": build_contained_in,
    "NEAR_LANDMARK": build_near_landmark,
    "SIMILAR_VECTOR": build_similar_vector,
}


def parse_only(value: str) -> list[str]:
    selected = [v.strip().upper() for v in value.split(",") if v.strip()]
    unknown = [v for v in selected if v not in SUB_BUILDERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown relation type(s): {', '.join(unknown)}; valid: {', '.join(RELATION_TYPES)}"
        )
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the place_relations knowledge-graph edges.")
    parser.add_argument(
        "--only",
        type=parse_only,
        default=list(RELATION_TYPES),
        metavar="RELATION_TYPE[,...]",
        help=(
            "Comma-separated subset of relation types to (re)build. "
            f"Default: all of {', '.join(RELATION_TYPES)}."
        ),
    )
    args = parser.parse_args(argv)
    selected: list[str] = args.only

    with get_conn() as conn:
        if "SIMILAR_VECTOR" in selected:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM place_embeddings_v2")
                row = cur.fetchone()
                v2_count = row[0] if row else 0
            if v2_count == 0:
                print(
                    "WARNING: place_embeddings_v2 is empty; SIMILAR_VECTOR "
                    "will produce 0 rows. Run `make embed-v2` first."
                )

        for relation_type in selected:
            builder = SUB_BUILDERS[relation_type]
            try:
                n = builder(conn)
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - report and fail the run
                conn.rollback()
                print(f"{relation_type}: FAILED ({exc})", file=sys.stderr)
                return 1
            print(f"{relation_type}: {n} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
