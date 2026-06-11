"""EVAL-05 probe — capture real AIMessage shape per provider (D-10-11).

Generalizes scripts/probe_gpt5_capture.py from a gpt-5-only hardcoded target
to a `--provider {openai|deepseek|anthropic|gemini}` CLI. Makes one
tool-call-shaped request per provider and writes a redacted JSON fixture to
`tests/fixtures/provider_payloads/{provider}.json`.

Redaction is genuinely fail-closed (CR-05 / T-10-09-01..03):
  - All value-bearing fixture fields — response_metadata, usage_metadata, and
    tool_calls — are routed through `_redact` (env-var substitution + regex).
  - The post-write secret-scan guard (_scan_fixture_for_secrets) re-reads the
    written fixture and checks BOTH _SECRET_PATTERNS regexes AND the actual
    runtime values of _SECRET_ENV_VARS, so a rotated key that doesn't match any
    regex is still caught and the fixture is deleted (return 2).

Per `project_app_editable_install`: `from app.llm_factory import build_chat_model`
works as-is via poetry editable install; no sys.path bootstrap is needed.

Per D-10-14: this probe is MANUAL ONLY — no CI/cron. Live keys are NOT present
in CI. Adapter tests in tests/unit/test_adapters.py SKIP when fixtures are absent
so CI never needs to run this probe. See `make probe-providers`.

Usage:
    poetry run python scripts/probe_provider_capture.py --provider openai
    poetry run python scripts/probe_provider_capture.py --provider anthropic --model claude-sonnet-4-6
    make probe-providers  # runs all four providers in sequence
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

from app.llm_factory import build_chat_model  # noqa: E402

# Where the redacted JSON fixtures live (D-10-11: checked in, reviewed before commit).
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "provider_payloads"


def _fixture_path(provider: str) -> Path:
    """Return the path for the provider's JSON fixture."""
    return FIXTURE_DIR / f"{provider}.json"


# Short, tool-call-prone query mimicking the agent's own first-turn shape.
# Keep it short so the probe is fast and the AIMessage shape is dominated by
# the model's reasoning_content (if any) rather than long generated text.
PROBE_QUERY = "search for a bar in mission"

# D-10-13: extend beyond sk- prefix to cover all common API-key shapes
# AND env-var-sourced secret values (T-10-05-01 / T-10-05-02).
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),  # Anthropic
    re.compile(r"AIzaSy[A-Za-z0-9_-]{10,}"),  # Google/Gemini
    re.compile(r"[A-Za-z0-9]{32,}-[A-Za-z0-9]{4,}"),  # generic long token
]

# Known secret env-var names whose values are redacted regardless of where
# they appear in the message (T-10-05-02).
_SECRET_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
]


def _redact(value: object) -> str:
    """Stringify `value` and redact anything that matches a secret pattern.

    D-10-13: covers OpenAI sk-, Anthropic sk-ant-, Google AIzaSy..., and any
    env-var-sourced secret values. Defensive only — the fields we dump (keys,
    sanitized metadata, content shape) should not contain secrets, but a model
    could echo one back inside content and we never want it written to disk.
    """
    s = str(value)
    # Redact env-var-sourced secret values first (before regex patterns, in case
    # the actual key value doesn't match a regex but is still a secret).
    for env_var in _SECRET_ENV_VARS:
        secret_val = os.environ.get(env_var, "")
        # Only redact non-trivially short values to avoid false positives.
        if secret_val and len(secret_val) >= 10:
            s = s.replace(secret_val, "<REDACTED>")
    # Apply all regex patterns.
    for pat in _SECRET_PATTERNS:
        s = pat.sub("<REDACTED>", s)
    return s


def _scan_fixture_for_secrets(text: str) -> bool:
    """Return True if `text` contains any detectable secret.

    Checks two independent channels (mirroring `_redact`):
      1. _SECRET_PATTERNS regexes (sk-, sk-ant-, AIzaSy..., generic long token).
      2. Runtime values of _SECRET_ENV_VARS (catches rotated/non-regex-shaped keys).

    Called by main() as the post-write guard. Extracted as a standalone helper so
    tests can exercise the production guard logic directly without re-implementing it.
    """
    # Channel 1: regex-shaped secrets.
    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            return True
    # Channel 2: env-var-sourced secret values (mirrors _redact lines 90-94).
    for env_var in _SECRET_ENV_VARS:
        secret_val = os.environ.get(env_var, "")
        if secret_val and len(secret_val) >= 10 and secret_val in text:
            return True
    return False


def _content_shape(content: object) -> str:
    """Return a short human description of an AIMessage.content shape."""
    if isinstance(content, str):
        return f"str (len={len(content)})"
    if isinstance(content, list):
        types = sorted({type(item).__name__ for item in content})
        return f"list (len={len(content)}, item-types={types})"
    return f"{type(content).__name__}"


def _sanitize_response_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Strip well-known token-bearing fields from `response_metadata` before dump.

    We keep the keys and structural shape — that's what the probe is for — but
    drop anything that could contain a session token or echo of the request
    auth headers.
    """
    blocked_keys = {"system_fingerprint", "id"}
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if key in blocked_keys:
            sanitized[key] = "<redacted-for-probe>"
        else:
            sanitized[key] = value
    return sanitized


# Per-provider default models (overridable via --model).
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5-mini",
    "deepseek": "deepseek-reasoner",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-pro-preview",
}

# Per-provider library package name for importlib.metadata.version().
_LIBRARY_PACKAGES: dict[str, str] = {
    "openai": "langchain-openai",
    "deepseek": "langchain-deepseek",
    "anthropic": "langchain-anthropic",
    "gemini": "langchain-google-genai",
}


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="probe_provider_capture",
        description=(
            "Capture real AIMessage shape per provider and write a redacted JSON fixture "
            "to tests/fixtures/provider_payloads/{provider}.json (EVAL-05 / D-10-11). "
            "MANUAL ONLY — no CI/cron (D-10-14). See make probe-providers."
        ),
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "deepseek", "anthropic", "gemini"],
        help="LLM provider to probe.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the default model for the selected provider. "
            "Defaults: openai=gpt-5-mini, deepseek=deepseek-reasoner, "
            "anthropic=claude-sonnet-4-6, gemini=gemini-3.1-pro-preview."
        ),
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe for a single provider. Returns 0 on success, non-zero on failure."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    provider = args.provider
    model_name = args.model or _DEFAULT_MODELS[provider]
    library_pkg = _LIBRARY_PACKAGES[provider]

    try:
        library_version = importlib.metadata.version(library_pkg)
    except importlib.metadata.PackageNotFoundError:
        library_version = "unknown"

    print(
        f"probe_provider_capture: building {provider}/{model_name} via build_chat_model (temp=1.0)"
    )
    llm = build_chat_model(provider, model_name, temperature=1.0)

    print(f"probe_provider_capture: invoking {provider}/{model_name} with one HumanMessage")
    message = llm.invoke([HumanMessage(content=PROBE_QUERY)])

    # Extract fields for the fixture. Redact everything defensively.
    add_kwargs_keys = sorted(message.additional_kwargs.keys())
    # WR-10: route additional_kwargs values through the same type-faithful redaction
    # pattern used for response_metadata/usage_metadata/tool_calls.  The prior
    # `_redact(message.additional_kwargs[k])` called str() on the value, collapsing
    # dict/list/number to a Python repr string like "{'k': 'v'}".  Instead, serialize
    # then redact then re-parse so dict/list/number values round-trip as real JSON
    # types and only genuinely non-JSON-serializable leaves fall back to str() via
    # `default=str`.  Redaction still routes through _redact — type fidelity never
    # bypasses the secret-scan path (T-11-18).
    add_kwargs_values = {
        k: json.loads(_redact(json.dumps(message.additional_kwargs[k], default=str)))
        for k in add_kwargs_keys
    }

    # CR-05: all value-bearing fixture fields must pass through _redact.
    # response_metadata: first sanitize known token-bearing keys (defense in depth),
    # then redact the serialized form so env-var secrets and regex patterns are removed.
    _sanitized_meta = _sanitize_response_metadata(dict(message.response_metadata or {}))
    response_metadata = json.loads(_redact(json.dumps(_sanitized_meta, default=str)))
    content_shape = _content_shape(message.content)
    raw_usage_metadata = getattr(message, "usage_metadata", None)
    usage_metadata = (
        json.loads(_redact(json.dumps(raw_usage_metadata, default=str)))
        if raw_usage_metadata is not None
        else None
    )
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    tool_calls = json.loads(_redact(json.dumps(raw_tool_calls, default=str)))

    # D-10-11 fixture shape: provider, model, library_version, probe_query,
    # additional_kwargs_keys, type-faithful redacted additional_kwargs_values (WR-10),
    # sanitized+redacted response_metadata, content_shape, redacted usage_metadata,
    # redacted tool_calls.
    fixture: dict[str, object] = {
        "provider": provider,
        "model": model_name,
        "library_version": library_version,
        "probe_query": PROBE_QUERY,
        "additional_kwargs_keys": add_kwargs_keys,
        "additional_kwargs_values": add_kwargs_values,
        "response_metadata": response_metadata,
        "content_shape": content_shape,
        "usage_metadata": usage_metadata,
        "tool_calls": tool_calls,
    }

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_file = _fixture_path(provider)
    fixture_file.write_text(json.dumps(fixture, indent=2, default=str), encoding="utf-8")

    # T-10-05-01 / T-10-09-02: post-write secret-scan guard — re-read and refuse if
    # any secret survived (regex OR env-var value). Fail-closed: delete + return 2.
    text = fixture_file.read_text(encoding="utf-8")
    if _scan_fixture_for_secrets(text):
        fixture_file.unlink(missing_ok=True)
        print(
            f"FATAL: fixture at {fixture_file} contains a secret "
            "(T-10-05-01 / T-10-09-02 secret-redaction check failed — "
            "regex pattern or env-var-sourced value detected). "
            "Fixture deleted — NOT committing.",
            file=sys.stderr,
        )
        return 2

    print(f"probe_provider_capture: wrote fixture to {fixture_file}")
    print(f"probe_provider_capture: additional_kwargs_keys = {add_kwargs_keys}")
    print(f"probe_provider_capture: content_shape = {content_shape!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
