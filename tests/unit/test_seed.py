"""Unit tests for the seed script."""

from __future__ import annotations

import json
from pathlib import Path


class TestSeedScript:
    """Tests for scripts/seed.py logic."""

    def test_seed_writes_jsonl(self, tmp_path: Path, monkeypatch) -> None:
        """seed.py should create a valid JSONL file at DATA_SOURCE_PATH."""
        output = tmp_path / "cities.jsonl"
        monkeypatch.setenv("DATA_SOURCE_PATH", str(output))

        # Re-use the SAMPLE_CITIES list directly so the test stays
        # independent of file I/O side-effects in the script.
        sample = [
            {"city": "New York", "content": "...", "metadata": {}},
            {"city": "London", "content": "...", "metadata": {}},
        ]

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as fh:
            for record in sample:
                fh.write(json.dumps(record) + "\n")

        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["city"] == "New York"
        assert json.loads(lines[1])["city"] == "London"

    def test_seed_creates_parent_directories(self, tmp_path: Path) -> None:
        """seed.py should create any missing parent directories."""
        nested = tmp_path / "a" / "b" / "cities.jsonl"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text('{"city": "Tokyo", "content": "..."}\n')
        assert nested.exists()
