"""add price_level_rank function

places_raw.price_level stores Google Places v1 enum strings
('PRICE_LEVEL_FREE', 'PRICE_LEVEL_INEXPENSIVE', ...). Lexical sort doesn't
match price order, so `price_level <= %s` was returning wrong results in W1's
SearchFilters. price_level_rank() maps the enum to 0..4 (matching Google's
documented integer convention) so SearchFilters.price_level_max can do an
integer comparison.

Revision ID: c428add573d7
Revises: 4c4789a14f8f
Create Date: 2026-05-06 16:09:14.614150
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c428add573d7"
down_revision: str | Sequence[str] | None = "4c4789a14f8f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PRICE_LEVEL_RANK_FN = """
CREATE OR REPLACE FUNCTION price_level_rank(price_level TEXT)
RETURNS INT AS $$
BEGIN
  RETURN CASE price_level
    WHEN 'PRICE_LEVEL_FREE'           THEN 0
    WHEN 'PRICE_LEVEL_INEXPENSIVE'    THEN 1
    WHEN 'PRICE_LEVEL_MODERATE'       THEN 2
    WHEN 'PRICE_LEVEL_EXPENSIVE'      THEN 3
    WHEN 'PRICE_LEVEL_VERY_EXPENSIVE' THEN 4
    ELSE NULL
  END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


def upgrade() -> None:
    op.execute(_PRICE_LEVEL_RANK_FN)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS price_level_rank(TEXT)")
