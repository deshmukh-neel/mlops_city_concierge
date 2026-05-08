"""grant CI service account write access to places_ingest_query_proposals

The CI integration job authenticates as the github-actions-deployer SA via
Cloud SQL IAM. After W5's CREATE TABLE the SA could SELECT but not INSERT/
DELETE, so the integration tests for the coverage agent could not run.

GRANT is idempotent. The DO block silently no-ops on DBs where the role
isn't provisioned (local dev, ephemeral CI Postgres) so this migration
applies cleanly everywhere.

Revision ID: a1b2c3d4e5f6
Revises: 34679f77f726
Create Date: 2026-05-08 11:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "34679f77f726"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles
                WHERE rolname = 'github-actions-deployer@mlops-491820.iam'
            ) THEN
                EXECUTE 'GRANT INSERT, DELETE ON places_ingest_query_proposals '
                     || 'TO "github-actions-deployer@mlops-491820.iam"';
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles
                WHERE rolname = 'github-actions-deployer@mlops-491820.iam'
            ) THEN
                EXECUTE 'REVOKE INSERT, DELETE ON places_ingest_query_proposals '
                     || 'FROM "github-actions-deployer@mlops-491820.iam"';
            END IF;
        END
        $$;
        """
    )
