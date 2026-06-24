"""Graph-traversal tool. Returns related places by relation_type.

JOINs place_relations to the active embeddings view via view_name() so the
v1/v2 toggle is honored. Destinations missing from the active view (e.g.
landmark dst_place_ids outside places_raw) are silently dropped by the inner
JOIN, matching the W7 design contract.
"""

from __future__ import annotations

from app.tools.retrieval import PlaceHit, execute_query, view_name


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
    excluded_place_ids: list[str] | None = None,
) -> list[RelatedPlace]:
    """Return up to ``k`` places related to ``place_id`` by ``relation_type``.

    NEAR is ordered ascending by weight (closest first); SIMILAR_VECTOR is
    ordered descending by weight (most similar first); other relations use a
    stable LIMIT. Unknown relation_type raises ValueError.

    ``excluded_place_ids`` lets callers (the closure-swap node) suppress
    place_ids known to be closed in the current conversation. Filtered at the
    SQL layer so behavior matches ``nearby`` and ``semantic_search`` exclusion.
    """
    if relation_type not in VALID_RELATIONS:
        raise ValueError(f"Unknown relation_type: {relation_type}")
    view = view_name()  # allowlist member — interpolated into SQL below
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
        WHERE r.src_place_id = %s
          AND r.relation_type = %s
          AND (%s::text[] IS NULL OR pd.place_id != ALL(%s::text[]))
        ORDER BY
          CASE r.relation_type
            WHEN 'NEAR'           THEN  r.weight
            WHEN 'SIMILAR_VECTOR' THEN -r.weight
            ELSE 0
          END
        LIMIT %s
    """  # noqa: S608
    # Pass excluded twice: once for the NULL guard, once for the comparison
    # — keeps the no-exclusion call site backward-compatible.
    exclude = excluded_place_ids or None
    rows = execute_query(sql, [place_id, relation_type, exclude, exclude, k])
    return [RelatedPlace(**row) for row in rows]


__all__ = ["RelatedPlace", "kg_traverse", "VALID_RELATIONS"]
