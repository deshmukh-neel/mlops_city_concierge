from __future__ import annotations


def test_imports_ok() -> None:
    from app.tools.graph import (  # noqa: F401
        VALID_RELATIONS,
        RelatedPlace,
        kg_traverse,
    )

    assert {
        "NEAR",
        "SAME_NEIGHBORHOOD",
        "CONTAINED_IN",
        "NEAR_LANDMARK",
        "SIMILAR_VECTOR",
    } == VALID_RELATIONS


def test_related_place_constructs() -> None:
    from app.tools.graph import RelatedPlace

    rp = RelatedPlace(
        place_id="p",
        name="n",
        source="test",
        similarity=0.0,
        relation_type="NEAR",
        weight=1.0,
    )
    assert rp.relation_type == "NEAR"
    assert rp.weight == 1.0
    assert rp.relation_metadata == {}


def test_kg_traverse_registered_in_tools() -> None:
    from app.agent.tools import all_tools

    tool = next(t for t in all_tools() if t.name == "kg_traverse")
    desc = tool.description or ""
    assert "NOT YET AVAILABLE" not in desc
    assert "available: False" not in desc


def test_kg_traverse_args_schema_has_k_typed() -> None:
    from app.agent.tools import all_tools

    tool = next(t for t in all_tools() if t.name == "kg_traverse")
    fields = tool.args_schema.model_fields
    assert set(fields) >= {"place_id", "relation_type", "k"}
    assert fields["place_id"].annotation is str
    assert fields["relation_type"].annotation is str
    assert fields["k"].annotation is int
