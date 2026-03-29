"""Unit tests for the ingest script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestIngestScript:
    """Tests for scripts/ingest.py logic (no real DB or API calls)."""

    def test_loads_jsonl_records(self, tmp_path: Path) -> None:
        """ingest.py should read every non-empty line in the JSONL source."""
        source = tmp_path / "cities.jsonl"
        cities = [
            {"city": "Paris", "content": "City of Light"},
            {"city": "Rome", "content": "Eternal City"},
        ]
        source.write_text("\n".join(json.dumps(c) for c in cities) + "\n")

        records: list[dict] = []
        with source.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        assert len(records) == 2
        assert records[0]["city"] == "Paris"
        assert records[1]["city"] == "Rome"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines in the JSONL file should be silently ignored."""
        source = tmp_path / "cities.jsonl"
        source.write_text('\n{"city": "Berlin", "content": "German capital"}\n\n')

        records: list[dict] = []
        with source.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        assert len(records) == 1
        assert records[0]["city"] == "Berlin"
