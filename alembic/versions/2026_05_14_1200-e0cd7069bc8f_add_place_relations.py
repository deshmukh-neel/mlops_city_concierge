"""add place_relations

Creates the W7 knowledge-graph edge table plus its three indexes, and
re-issues neighborhood_of(jsonb) via CREATE OR REPLACE so the W7 migration
is self-contained per the W7 spec.

  1. place_relations — one row per (src_place_id, dst_place_id, relation_type).
     src_place_id has an FK to places_raw(place_id) ON DELETE CASCADE;
     dst_place_id has NO FK so landmark targets outside places_raw can be
     stored (W7 spec — NEAR_LANDMARK destinations are Google placeIds that
     are not necessarily ingested).
  2. Three indexes: (src_place_id, relation_type), (dst_place_id,
     relation_type), and (relation_type) alone.
  3. neighborhood_of(jsonb) — already created by W1's migration
     4c4789a14f8f. The body below is byte-for-byte identical to W1's
     (alembic/versions/2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py:35-50);
     keep these two in sync. downgrade() does NOT drop it — it belongs to W1.

Revision ID: e0cd7069bc8f
Revises: a1b2c3d4e5f6
Create Date: 2026-05-14 12:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "e0cd7069bc8f"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# duplicates W1's neighborhood_of body
# (4c4789a14f8f:35-50); keep in sync.
_NEIGHBORHOOD_OF_FN = """
CREATE OR REPLACE FUNCTION neighborhood_of(source_json JSONB)
RETURNS TEXT AS $$
DECLARE component JSONB;
BEGIN
  IF source_json IS NULL THEN RETURN ''; END IF;
  FOR component IN
    SELECT * FROM jsonb_array_elements(source_json->'addressComponents')
  LOOP
    IF (component->'types') ? 'neighborhood' THEN
      RETURN COALESCE(component->>'longText', component->>'shortText', '');
    END IF;
  END LOOP;
  RETURN '';
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS place_relations (
            src_place_id    TEXT NOT NULL REFERENCES places_raw(place_id) ON DELETE CASCADE,
            dst_place_id    TEXT NOT NULL,
            relation_type   TEXT NOT NULL,
            weight          DOUBLE PRECISION,
            metadata        JSONB DEFAULT '{}',
            source          TEXT NOT NULL,
            built_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (src_place_id, dst_place_id, relation_type)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_place_relations_src "
        "ON place_relations(src_place_id, relation_type);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_place_relations_dst "
        "ON place_relations(dst_place_id, relation_type);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_place_relations_type ON place_relations(relation_type);"
    )
    op.execute(_NEIGHBORHOOD_OF_FN)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS place_relations CASCADE")
    # Do NOT drop neighborhood_of — it belongs to W1.
