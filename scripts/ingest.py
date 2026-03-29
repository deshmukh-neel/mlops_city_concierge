#!/usr/bin/env python3
"""
ingest.py — Data ingestion pipeline for the City Concierge.

Reads city documents from DATA_SOURCE_PATH, chunks the text, generates
OpenAI embeddings, and upserts them into the city_chunks table in Postgres.

Usage:
    python scripts/ingest.py
    make ingest
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Allow imports from the project root when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    source_path = Path(os.environ.get("DATA_SOURCE_PATH", "data/cities.jsonl"))

    if not source_path.exists():
        log.error("Data source not found: %s", source_path)
        sys.exit(1)

    log.info("Starting ingestion from %s", source_path)

    records: list[dict] = []
    with source_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info("Loaded %d records", len(records))

    # TODO: chunk text, generate embeddings, upsert to Postgres
    # Placeholder — replace with real implementation once app/ package exists.
    for i, record in enumerate(records, 1):
        log.info("  [%d/%d] %s", i, len(records), record.get("city", "<unknown>"))

    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
