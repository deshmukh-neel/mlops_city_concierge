#!/usr/bin/env python3
"""
seed.py — Seed the database with sample city data for local development.

Usage:
    python scripts/seed.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SAMPLE_CITIES: list[dict] = [
    {
        "city": "New York",
        "content": (
            "New York City is the most populous city in the United States. "
            "Known for Times Square, Central Park, and the Statue of Liberty."
        ),
        "metadata": {"country": "US", "population": 8336817},
    },
    {
        "city": "London",
        "content": (
            "London is the capital and largest city of England and the United Kingdom. "
            "Famous for the Tower of London, Buckingham Palace, and the River Thames."
        ),
        "metadata": {"country": "GB", "population": 8799800},
    },
    {
        "city": "Tokyo",
        "content": (
            "Tokyo is the capital and most populous city of Japan. "
            "Renowned for its blend of traditional culture and cutting-edge technology."
        ),
        "metadata": {"country": "JP", "population": 13960000},
    },
]


def main() -> None:
    output_path = Path(os.environ.get("DATA_SOURCE_PATH", "data/cities.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fh:
        for record in SAMPLE_CITIES:
            fh.write(json.dumps(record) + "\n")

    log.info("Seeded %d sample cities to %s", len(SAMPLE_CITIES), output_path)


if __name__ == "__main__":
    main()
