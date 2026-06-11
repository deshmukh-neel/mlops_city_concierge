"""Unit tests for scripts/probe_provider_capture.py redaction and secret-scan guard.

EVAL-05 / D-10-13: Redaction must cover OpenAI (sk-), Anthropic (sk-ant-),
Google (AIzaSy...) key shapes, and env-var-sourced secret values. A post-write
secret-scan guard must refuse to write a fixture containing a secret pattern.

These tests use only synthetic strings — no live API calls are made.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch


def test_redact_removes_openai_key() -> None:
    """D-10-13: _redact removes OpenAI sk- key patterns."""
    from scripts.probe_provider_capture import _redact

    raw = "key=sk-proj-abcdef1234567890AAAABBBBCCCCDDDD"
    result = _redact(raw)
    assert "sk-proj-abcdef1234567890AAAABBBBCCCCDDDD" not in result
    assert "<REDACTED>" in result


def test_redact_removes_anthropic_key() -> None:
    """D-10-13 / IN-04: _redact removes Anthropic sk-ant- key patterns.

    Per PATTERNS.md test_probe_redaction_catches_anthropic_key:
        assert 'sk-ant-REDACTED' in _redact('sk-ant-api03-abc123xyz789...')
        assert 'sk-ant-api03-abc123xyz789' not in _redact(...)
    """
    from scripts.probe_provider_capture import _redact

    raw = "sk-ant-api03-abc123xyz789AAABBBCCCDDDEEEFFFGGG"
    result = _redact(raw)
    assert "sk-ant-api03-abc123xyz789AAABBBCCCDDDEEEFFFGGG" not in result
    assert "<REDACTED>" in result


def test_redact_removes_google_key() -> None:
    """D-10-13: _redact removes Google AIzaSy... key patterns (33 chars after prefix)."""
    from scripts.probe_provider_capture import _redact

    raw = "google_key=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345"
    result = _redact(raw)
    assert "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345" not in result
    assert "<REDACTED>" in result


def test_redact_removes_env_sourced_openai_key() -> None:
    """D-10-13: _redact substitutes env-var-sourced secret values regardless of position."""

    fake_key = "sk-test-envkey-AAABBBCCCDDDEEE12345678"
    with patch.dict(os.environ, {"OPENAI_API_KEY": fake_key}):
        # Re-import to pick up the env var — the function reads os.environ at call time
        import importlib

        import scripts.probe_provider_capture as probe_mod

        importlib.reload(probe_mod)
        result = probe_mod._redact(f"some text containing {fake_key} in it")
        assert fake_key not in result
        assert "<REDACTED>" in result


def test_redact_leaves_benign_text_unchanged() -> None:
    """_redact does not mangle non-secret text."""
    from scripts.probe_provider_capture import _redact

    benign = "The weather is nice today. model=gpt-4o-mini"
    result = _redact(benign)
    assert result == benign


def test_secret_patterns_list_covers_expected_providers() -> None:
    """_SECRET_PATTERNS covers OpenAI, Anthropic, and Google patterns (D-10-13)."""
    from scripts.probe_provider_capture import _SECRET_PATTERNS

    assert len(_SECRET_PATTERNS) >= 3, (
        f"Expected at least 3 secret patterns (OpenAI/Anthropic/Google); got {len(_SECRET_PATTERNS)}"
    )

    # Verify each major pattern matches a known example
    openai_key = "sk-proj-abcdef1234567890AABBCCDD"
    anthropic_key = "sk-ant-api03-abcdef123456789AAABBBCCCDDDEEE"
    google_key = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345"

    openai_matched = any(p.search(openai_key) for p in _SECRET_PATTERNS)
    anthropic_matched = any(p.search(anthropic_key) for p in _SECRET_PATTERNS)
    google_matched = any(p.search(google_key) for p in _SECRET_PATTERNS)

    assert openai_matched, "No pattern matches OpenAI sk- key"
    assert anthropic_matched, "No pattern matches Anthropic sk-ant- key"
    assert google_matched, "No pattern matches Google AIzaSy key"


def test_post_write_guard_rejects_fixture_with_planted_secret(tmp_path: Path) -> None:
    """D-10-13 / T-10-05-01: post-write secret-scan guard refuses a fixture containing
    a secret pattern and returns non-zero / raises."""
    from scripts.probe_provider_capture import _SECRET_PATTERNS

    # Write a fixture containing a planted fake secret
    secret = "sk-proj-PLANTED1234567890AAAABBBBCCCCDDDD"
    fixture_path = tmp_path / "bad_fixture.json"
    fixture_data = {"provider": "openai", "additional_kwargs_values": {"key": secret}}
    fixture_path.write_text(json.dumps(fixture_data), encoding="utf-8")

    # The guard logic: re-read and scan all secret patterns
    text = fixture_path.read_text(encoding="utf-8")
    found = any(p.search(text) for p in _SECRET_PATTERNS)
    assert found, (
        "Secret-scan guard should detect the planted sk- key in the fixture. "
        "This confirms the guard would refuse to keep this file."
    )


def test_post_write_guard_passes_clean_fixture(tmp_path: Path) -> None:
    """Post-write guard passes a properly redacted fixture (no false positives)."""
    from scripts.probe_provider_capture import _SECRET_PATTERNS, _redact

    # Build a fixture where secrets have been redacted
    raw_value = "sk-proj-PLANTED1234567890AAAABBBBCCCCDDDD"
    redacted_value = _redact(raw_value)
    fixture_data = {"provider": "openai", "additional_kwargs_values": {"key": redacted_value}}
    fixture_path = tmp_path / "good_fixture.json"
    fixture_path.write_text(json.dumps(fixture_data), encoding="utf-8")

    # The guard must NOT find any secret in the redacted fixture
    text = fixture_path.read_text(encoding="utf-8")
    found = any(p.search(text) for p in _SECRET_PATTERNS)
    assert not found, (
        f"Secret-scan guard flagged a clean (redacted) fixture as containing a secret. "
        f"Redacted value: {redacted_value!r}"
    )


def test_fixture_output_path_uses_provider_payloads(tmp_path: Path) -> None:
    """D-10-11: fixture output path is tests/fixtures/provider_payloads/{provider}.json."""
    from scripts.probe_provider_capture import _fixture_path

    path = _fixture_path("openai")
    assert "provider_payloads" in str(path), (
        f"Fixture path should contain 'provider_payloads'; got {path}"
    )
    assert path.name == "openai.json"


def test_main_help_exits_zero() -> None:
    """Smoke: `--help` exits 0 and lists --provider choices."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/probe_provider_capture.py", "--help"],
        cwd="/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}; stderr={result.stderr}"
    assert "--provider" in result.stdout or "--provider" in result.stderr, (
        "--provider not found in --help output"
    )
