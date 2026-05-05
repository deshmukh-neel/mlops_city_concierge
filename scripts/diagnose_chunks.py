#!/usr/bin/env python3
"""Print v1 vs v2 chunks for the same place_ids. Used to eyeball quality
before/after the cleanup. Read-only; never writes."""

from __future__ import annotations

import argparse
import os

import psycopg2
from dotenv import load_dotenv

from app.config import resolve_database_url

load_dotenv()

QUERY = """
SELECT p.name, v1.embedding_text AS v1_text, v2.embedding_text AS v2_text,
       LENGTH(v1.embedding_text) AS v1_len, LENGTH(v2.embedding_text) AS v2_len
FROM places_raw p
LEFT JOIN place_embeddings    v1 ON v1.place_id = p.place_id
LEFT JOIN place_embeddings_v2 v2 ON v2.place_id = p.place_id
WHERE v1.place_id IS NOT NULL AND v2.place_id IS NOT NULL
  AND p.user_rating_count > 100
ORDER BY random()
LIMIT %s
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=5, help="number of places to sample")
    args = parser.parse_args()

    database_url = resolve_database_url(os.environ)
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL in environment.")

    with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(QUERY, (args.n,))
        for name, v1, v2, v1_len, v2_len in cur.fetchall():
            print("=" * 80)
            print(f"NAME: {name}    v1={v1_len} chars   v2={v2_len} chars")
            print("--- V1 ---\n" + v1)
            print("--- V2 ---\n" + v2)


if __name__ == "__main__":
    main()
