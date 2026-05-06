"""baseline

Marks the schema in scripts/db/init.sql as the starting point for Alembic
tracking. This migration intentionally does nothing — the schema it represents
already exists in every environment via init.sql (local docker auto-runs it,
Cloud SQL was bootstrapped from it during initial provisioning).

After this PR merges, run `alembic stamp head` ONCE per pre-existing database
(Cloud SQL prod, any teammate's local DB that already has data) to record that
this baseline is satisfied. Fresh databases pick it up automatically the first
time `alembic upgrade head` runs.

Revision ID: b932216bf431
Revises:
Create Date: 2026-05-06 13:56:04.699331
"""

from collections.abc import Sequence

revision: str = "b932216bf431"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: baseline schema is owned by scripts/db/init.sql."""
    pass


def downgrade() -> None:
    """No-op: nothing to undo."""
    pass
