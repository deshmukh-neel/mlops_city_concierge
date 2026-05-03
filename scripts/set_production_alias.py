#!/usr/bin/env python3
"""Promote a registered MLflow model version to the `production` alias.

Reads MLFLOW_TRACKING_URI from the environment (or falls back to the
project default in app/config.py). The app loads `production` at startup,
so after running this you must restart the app for the new chain to take
effect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402

DEFAULT_MODEL_NAME = "city-concierge-rag"
DEFAULT_ALIAS = "production"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Model version to promote")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    args = parser.parse_args()

    tracking_uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(
        f"Set alias '{args.alias}' on '{args.model_name}' "
        f"to version {args.version} (tracking: {tracking_uri})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
