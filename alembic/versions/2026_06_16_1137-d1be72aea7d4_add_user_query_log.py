"""add user_query_log

Records every real /chat user query as the demand-side learning signal for
the v2.3 adaptive data loop. Phase 18 (GAP) mines this table for
under-served neighborhood/cuisine demand.

NOTE: raw message text is stored verbatim — no PII scrubbing — because the
entire value of this table is mining real demand text. This is a private
capstone database, not a public service (see Phase 17 CONTEXT D-04).

Revision ID: d1be72aea7d4
Revises: e0cd7069bc8f
Create Date: 2026-06-16 11:37:31.069789

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d1be72aea7d4"
down_revision: str | Sequence[str] | None = "e0cd7069bc8f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_query_log",
        sa.Column(
            "id",
            sa.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("requested_primary_types", sa.ARRAY(sa.Text), nullable=True),
        sa.Column("num_stops", sa.Integer, nullable=True),
        sa.Column("rag_label", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("session_id", sa.Text, nullable=True),
    )
    # Index on created_at for the Phase 18 miner's time-window queries
    # (mirrors idx_place_query_hits_* style from init.sql:76-77).
    op.create_index(
        "idx_user_query_log_created_at",
        "user_query_log",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_user_query_log_created_at", table_name="user_query_log")
    op.drop_table("user_query_log")
