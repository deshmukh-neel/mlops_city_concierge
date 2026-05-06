"""create place_documents view

Creates four SQL objects in one atomic upgrade:

  1. neighborhood_of(jsonb) — extracts the structured neighborhood from
     places_raw.source_json.addressComponents. W1 owns this; W7's migration
     uses CREATE OR REPLACE FUNCTION with an identical body so order doesn't
     matter.
  2. place_is_open(jsonb, timestamptz) — returns true if the regular_opening_hours
     JSONB indicates the place is open at the given local time. Handles
     overnight periods (close < open the next day) so bars don't get
     incorrectly excluded.
  3. place_documents — view joining places_raw + place_embeddings (v1).
  4. place_documents_v2 — same projection joined against place_embeddings_v2.

The two views share their projection via a Python template; only the view name
and the joined embedding table differ. Both substitutions are hardcoded
literals — no SQL injection surface.

Revision ID: 4c4789a14f8f
Revises: 5187c6b09b25
Create Date: 2026-05-06 15:21:25.561093
"""

from collections.abc import Sequence

from alembic import op

revision: str = "4c4789a14f8f"
down_revision: str | Sequence[str] | None = "5187c6b09b25"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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


_PLACE_IS_OPEN_FN = """
CREATE OR REPLACE FUNCTION place_is_open(hours JSONB, at_ts TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
DECLARE
  -- Convert at_ts into America/Los_Angeles before extracting DOW/HOUR.
  -- Postgres's session timezone is UTC on Cloud SQL and on the local docker
  -- container, so without AT TIME ZONE a Friday-22:00-SF call resolves to
  -- Saturday 05:00 UTC and DOW comes back wrong. The app is SF-only today
  -- (places_raw.source_city = 'San Francisco'); when it expands beyond SF
  -- this function gets a 3rd `tz TEXT` argument.
  local_ts       TIMESTAMP := at_ts AT TIME ZONE 'America/Los_Angeles';
  dow            INT := EXTRACT(DOW FROM local_ts);
  hh             INT := EXTRACT(HOUR FROM local_ts);
  mm             INT := EXTRACT(MINUTE FROM local_ts);
  minutes_of_day INT := hh * 60 + mm;
  period         JSONB;
  open_dow       INT;
  close_dow      INT;
  open_minutes   INT;
  close_minutes  INT;
BEGIN
  IF hours IS NULL OR hours = '{}'::jsonb THEN
    RETURN TRUE;
  END IF;
  FOR period IN SELECT * FROM jsonb_array_elements(hours->'periods') LOOP
    open_dow      := (period->'open'->>'day')::int;
    close_dow     := COALESCE((period->'close'->>'day')::int, open_dow);
    open_minutes  := (period->'open'->>'hour')::int * 60
                   + COALESCE((period->'open'->>'minute')::int, 0);
    close_minutes := (period->'close'->>'hour')::int * 60
                   + COALESCE((period->'close'->>'minute')::int, 0);

    IF open_dow = close_dow AND open_dow = dow THEN
      IF minutes_of_day BETWEEN open_minutes AND close_minutes THEN
        RETURN TRUE;
      END IF;
    ELSIF open_dow <> close_dow THEN
      IF dow = open_dow  AND minutes_of_day >= open_minutes  THEN
        RETURN TRUE;
      END IF;
      IF dow = close_dow AND minutes_of_day <= close_minutes THEN
        RETURN TRUE;
      END IF;
    END IF;
  END LOOP;
  RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


# Single template; substituted twice — once per embedding table. Both
# substitutions are hardcoded literals from this file (the W1 migration),
# so there is no SQL injection surface.
_VIEW_SQL_TEMPLATE = """
CREATE OR REPLACE VIEW {view_name} AS
SELECT
    p.place_id,
    p.name,
    p.primary_type,
    p.types,
    p.formatted_address,
    neighborhood_of(p.source_json) AS neighborhood,
    p.latitude,
    p.longitude,
    p.rating,
    p.user_rating_count,
    p.price_level,
    p.business_status,
    p.website_uri,
    p.maps_uri,
    p.editorial_summary,
    p.regular_opening_hours,
    p.source_city,
    COALESCE((p.source_json->>'servesCocktails')::boolean,       FALSE) AS serves_cocktails,
    COALESCE((p.source_json->>'servesBeer')::boolean,            FALSE) AS serves_beer,
    COALESCE((p.source_json->>'servesWine')::boolean,            FALSE) AS serves_wine,
    COALESCE((p.source_json->>'servesCoffee')::boolean,          FALSE) AS serves_coffee,
    COALESCE((p.source_json->>'servesBreakfast')::boolean,       FALSE) AS serves_breakfast,
    COALESCE((p.source_json->>'servesBrunch')::boolean,          FALSE) AS serves_brunch,
    COALESCE((p.source_json->>'servesLunch')::boolean,           FALSE) AS serves_lunch,
    COALESCE((p.source_json->>'servesDinner')::boolean,          FALSE) AS serves_dinner,
    COALESCE((p.source_json->>'servesVegetarianFood')::boolean,  FALSE) AS serves_vegetarian,
    COALESCE((p.source_json->>'outdoorSeating')::boolean,        FALSE) AS outdoor_seating,
    COALESCE((p.source_json->>'reservable')::boolean,            FALSE) AS reservable,
    COALESCE((p.source_json->>'allowsDogs')::boolean,            FALSE) AS allows_dogs,
    COALESCE((p.source_json->>'liveMusic')::boolean,             FALSE) AS live_music,
    COALESCE((p.source_json->>'goodForGroups')::boolean,         FALSE) AS good_for_groups,
    COALESCE((p.source_json->>'goodForChildren')::boolean,       FALSE) AS good_for_children,
    COALESCE((p.source_json->>'goodForWatchingSports')::boolean, FALSE) AS good_for_sports,
    COALESCE(p.source_json->'parkingOptions', '{{}}'::jsonb)            AS parking_options,
    'google_places'::text AS source,
    e.embedding,
    e.embedding_model,
    e.embedding_text,
    e.source_updated_at AS embedded_source_updated_at
FROM places_raw p
JOIN {embedding_table} e ON e.place_id = p.place_id
"""


_VIEW_COMMENT = """
COMMENT ON VIEW place_documents IS
  'Unified retrieval surface. When editorial places table lands, redefine as UNION ALL with source = ''editorial''.';
"""


def upgrade() -> None:
    op.execute(_NEIGHBORHOOD_OF_FN)
    op.execute(_PLACE_IS_OPEN_FN)
    op.execute(
        _VIEW_SQL_TEMPLATE.format(
            view_name="place_documents",
            embedding_table="place_embeddings",
        )
    )
    op.execute(
        _VIEW_SQL_TEMPLATE.format(
            view_name="place_documents_v2",
            embedding_table="place_embeddings_v2",
        )
    )
    op.execute(_VIEW_COMMENT)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS place_documents_v2")
    op.execute("DROP VIEW IF EXISTS place_documents")
    op.execute("DROP FUNCTION IF EXISTS place_is_open(JSONB, TIMESTAMPTZ)")
    op.execute("DROP FUNCTION IF EXISTS neighborhood_of(JSONB)")
