"""Graph-traversal tool. Returns related places by relation_type.

JOINs place_relations to the active embeddings view via _view_name() so the
v1/v2 toggle is honored. Destinations missing from the active view (e.g.
landmark dst_place_ids outside places_raw) are silently dropped by the inner
JOIN, matching the W7 design contract.
"""

from __future__ import annotations

from app.tools.retrieval import PlaceHit, _execute, _view_name


class RelatedPlace(PlaceHit):
    relation_type: str
    weight: float | None = None
    relation_metadata: dict = {}


VALID_RELATIONS = {
    "NEAR",
    "SAME_NEIGHBORHOOD",
    "CONTAINED_IN",
    "NEAR_LANDMARK",
    "SIMILAR_VECTOR",
}


def kg_traverse(
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
) -> list[RelatedPlace]:
    """Return up to ``k`` places related to ``place_id`` by ``relation_type``.

    NEAR is ordered ascending by weight (closest first); SIMILAR_VECTOR is
    ordered descending by weight (most similar first); other relations use a
    stable LIMIT. Unknown relation_type raises ValueError.
    """
    if relation_type not in VALID_RELATIONS:
        raise ValueError(f"Unknown relation_type: {relation_type}")
    view = _view_name()  # allowlist member — interpolated into SQL below
    sql = f"""
        SELECT pd.place_id, pd.name, pd.primary_type, pd.formatted_address,
               pd.latitude, pd.longitude, pd.rating, pd.price_level,
               pd.business_status, pd.source,
               0.0 AS similarity,
               LEFT(pd.embedding_text, 400) AS snippet,
               r.relation_type, r.weight,
               r.metadata AS relation_metadata
        FROM place_relations r
        JOIN {view} pd ON pd.place_id = r.dst_place_id
        WHERE r.src_place_id = %s AND r.relation_type = %s
        ORDER BY
          CASE r.relation_type
            WHEN 'NEAR'           THEN  r.weight
            WHEN 'SIMILAR_VECTOR' THEN -r.weight
            ELSE 0
          END
        LIMIT %s
    """  # noqa: S608
    rows = _execute(sql, [place_id, relation_type, k])
    return [RelatedPlace(**row) for row in rows]


__all__ = ["RelatedPlace", "kg_traverse", "VALID_RELATIONS"]
