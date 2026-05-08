"""add places_ingest_query_proposals

W5 coverage agent writes proposed seed queries here. The ingest script
prepends rows with status='pending' to its seed list and marks them
'applied' once the run completes. Kept separate from
places_ingest_query_checkpoints so checkpoint semantics (run-progress log,
keyed by FIELD_MODE::query) stay clean.

Revision ID: 34679f77f726
Revises: c428add573d7
Create Date: 2026-05-08 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "34679f77f726"
down_revision: str | Sequence[str] | None = "c428add573d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "places_ingest_query_proposals",
        sa.Column("query_text", sa.Text, primary_key=True),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("applied_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'applied', 'rejected')",
            name="places_ingest_query_proposals_status_check",
        ),
    )
    op.create_index(
        "idx_places_ingest_query_proposals_status",
        "places_ingest_query_proposals",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_places_ingest_query_proposals_status",
        table_name="places_ingest_query_proposals",
    )
    op.drop_table("places_ingest_query_proposals")
