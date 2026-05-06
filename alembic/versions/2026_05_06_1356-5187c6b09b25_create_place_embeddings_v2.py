"""create place_embeddings_v2

Brings the W0a v2 embeddings table under Alembic management. The SQL here is
identical to scripts/db/migrations/000_place_embeddings_v2.sql, which was
applied manually to Cloud SQL on 2026-05-04 before this PR existed. Existing
databases should `alembic stamp head` after merging this PR so the migration
is recorded as already-applied (the IF NOT EXISTS guards make it idempotent
either way).

Revision ID: 5187c6b09b25
Revises: b932216bf431
Create Date: 2026-05-06 13:56:28.637563
"""

from collections.abc import Sequence

from alembic import op

revision: str = "5187c6b09b25"
down_revision: str | Sequence[str] | None = "b932216bf431"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS place_embeddings_v2 (
            place_id              TEXT PRIMARY KEY REFERENCES places_raw(place_id) ON DELETE CASCADE,
            embedding             vector(1536) NOT NULL,
            embedding_model       TEXT NOT NULL,
            embedding_text        TEXT NOT NULL,
            embedded_at           TIMESTAMPTZ DEFAULT NOW(),
            source_updated_at     TIMESTAMPTZ NOT NULL
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_place_embeddings_v2_vector
            ON place_embeddings_v2
            USING hnsw (embedding vector_cosine_ops);
        """
    )
    op.execute(
        """
        COMMENT ON TABLE place_embeddings_v2 IS
          'Cleaned embeddings (no URLs, no structured facts, with neighborhood + landmark names). Drives retrieval when EMBEDDING_TABLE=place_embeddings_v2.';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS place_embeddings_v2 CASCADE")
