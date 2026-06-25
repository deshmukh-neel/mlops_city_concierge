"""PROV-01 probe — capture real gpt-5-mini AIMessage shape (D-09-03).

One-shot script that hits `gpt-5-mini` via `app.llm_factory.build_chat_model`
and dumps the returned `AIMessage` shape (additional_kwargs, response_metadata,
content shape, usage_metadata) to a committed markdown artifact at
`.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md`.

The verdict line ("kwarg path works" / "subclass required" / "neither — escalate")
dictates whether Plan 09-01 Task 2 takes Path A (read additional_kwargs from
AIMessage directly) or Path B (introduce a ChatOpenAI subclass that lifts the
field BEFORE LangChain's normalizer drops it). See D-09-03 + memory
`project_agent_loses_reasoning_state_all_providers`.

Per CONTEXT.md threat T-09-01-T1: this script writes ONLY the additional_kwargs
KEY NAMES and SANITIZED response_metadata. It NEVER logs the raw OPENAI_API_KEY,
Authorization headers, or any SecretStr contents.

Per `project_app_editable_install`: `from app.llm_factory import build_chat_model`
works as-is via poetry editable install; no sys.path bootstrap is needed.

Per `feedback_temp1_reasoning_off_all_models`: temp=1.0 always. Reasoning is left
ON in this probe because we are explicitly checking whether reasoning_content
surfaces on the AIMessage — disabling thinking would defeat the probe.

Usage:
    poetry run python scripts/probe_gpt5_capture.py
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

from app.llm_factory import build_chat_model  # noqa: E402

# Where the probe artifact lives. Committed with `git add -f` because
# `.planning/` is gitignored but SUMMARY/PROBE/VERIFICATION are tracked per
# established phases-07/08 convention.
PROBE_ARTIFACT = (
    REPO_ROOT
    / ".planning"
    / "phases"
    / "09-per-provider-state-preservation-implementations"
    / "09-PROV-01-PROBE.md"
)

# Short, tool-call-prone query mimicking the agent's own first-turn shape (a
# representative semantic_search-style request). Keep it short so the probe is
# fast and the AIMessage shape is dominated by the model's reasoning_content
# (if any) rather than long generated text.
PROBE_QUERY = "search for a bar in mission"

# Match anything that LOOKS like an OpenAI secret. Defensive scan against
# accidentally writing a key into the artifact (T-09-01-T1).
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


def redact(value: object) -> str:
    """Stringify `value` and redact anything that matches an OpenAI secret pattern.

    Defensive only — the fields we dump here (additional_kwargs.keys(),
    response_metadata sanitized, content shape, usage_metadata) should not
    contain keys, but a model could echo one back inside content and we never
    want it written to disk.
    """
    s = str(value)
    return SECRET_PATTERN.sub("sk-REDACTED", s)


def content_shape(content: object) -> str:
    """Return a short human description of an AIMessage.content shape."""
    if isinstance(content, str):
        return f"str (len={len(content)})"
    if isinstance(content, list):
        types = sorted({type(item).__name__ for item in content})
        return f"list (len={len(content)}, item-types={types})"
    return f"{type(content).__name__}"


def sanitize_response_metadata(metadata: dict[str, object]) -> dict[str, object]:
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


def main() -> int:
    chat_model_name = "gpt-5-mini"
    lc_openai_version = importlib.metadata.version("langchain-openai")

    print(f"PROV-01 probe: building {chat_model_name} via build_chat_model (temp=1.0)")
    llm = build_chat_model("openai", chat_model_name, temperature=1.0)

    print(f"PROV-01 probe: invoking {chat_model_name} with one HumanMessage")
    message = llm.invoke([HumanMessage(content=PROBE_QUERY)])

    # Extract the fields the artifact records. None of these should contain
    # raw secrets, but we redact defensively before write.
    add_kwargs_keys = sorted(message.additional_kwargs.keys())
    add_kwargs_repr = {key: redact(message.additional_kwargs[key]) for key in add_kwargs_keys}
    response_metadata = sanitize_response_metadata(dict(message.response_metadata or {}))
    message_content_shape = content_shape(message.content)
    usage_metadata = getattr(message, "usage_metadata", None)
    tool_calls = getattr(message, "tool_calls", None) or []
    raw_dict_dump = redact(dict(message))

    # Verdict heuristic. The executor / user can override this in review, but a
    # data-driven first guess helps:
    #   - reasoning_content key present on additional_kwargs OR a "reasoning"
    #     key on response_metadata -> Path A (read-the-kwarg works).
    #   - neither present -> Path B (subclass required to lift before normalizer
    #     drops it).
    has_reasoning_kwarg = "reasoning_content" in message.additional_kwargs
    has_reasoning_metadata = (
        isinstance(message.response_metadata, dict) and "reasoning" in message.response_metadata
    )
    if has_reasoning_kwarg or has_reasoning_metadata:
        verdict = "kwarg path works"
    else:
        verdict = "subclass required"

    artifact = f"""# Phase 9 / PROV-01 — gpt-5-mini AIMessage shape probe

**Probed:** 2026-06-04
**Plan:** 09-01 (openai-gpt5-adapter)
**Decision:** D-09-03 (probe-then-build)

## langchain-openai version

`{lc_openai_version}`

## gpt-5-mini chat_model used

- provider: `openai`
- model: `{chat_model_name}`
- temperature: `1.0`
- built via: `app.llm_factory.build_chat_model("openai", "{chat_model_name}", 1.0)`
- probe query: `{PROBE_QUERY!r}`

## AIMessage additional_kwargs keys

```python
{add_kwargs_keys}
```

Values (redacted for sk- patterns):

```python
{add_kwargs_repr}
```

## AIMessage response_metadata

```python
{response_metadata}
```

## AIMessage content shape

{message_content_shape}

## AIMessage usage_metadata

```python
{usage_metadata}
```

## AIMessage tool_calls

```python
{tool_calls}
```

## Raw dict(message) dump

```python
{raw_dict_dump}
```

## Verdict

{verdict}

### Interpretation

- `reasoning_content` in additional_kwargs: **{has_reasoning_kwarg}**
- `reasoning` in response_metadata: **{has_reasoning_metadata}**

If the verdict is `kwarg path works`, Plan 09-01 Task 2 takes **Path A** — the
adapter's `capture_reasoning_state` reads `message.additional_kwargs.get("reasoning_content")`
directly and `replay_reasoning_state` writes the same key back on the most-recent
outbound `AIMessage`.

If the verdict is `subclass required`, Plan 09-01 Task 2 takes **Path B** — we
introduce an `OpenAIReasoningChatModel(ChatOpenAI)` in `app/llm_factory.py` that
overrides `_generate` to lift the raw response's `reasoning_content` field into
`AIMessage.additional_kwargs["reasoning_content"]` BEFORE LangChain's normalizer
drops it. The adapter then reads from the subclass-enriched message.

If the verdict is `neither — escalate`, PROV-01 is library-blocked and the
Phase 9 PR cannot ship per D-09-02; we open `09-PROV-01-BLOCKER.md`.
"""

    PROBE_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_ARTIFACT.write_text(artifact, encoding="utf-8")

    # Final defensive scan — if a secret pattern slipped through anywhere,
    # blow up before the user commits the artifact.
    if SECRET_PATTERN.search(PROBE_ARTIFACT.read_text(encoding="utf-8")):
        print(
            f"FATAL: probe artifact at {PROBE_ARTIFACT} contains an sk- pattern "
            "(T-09-01-T1 secret-redaction check failed). NOT committing.",
            file=sys.stderr,
        )
        return 2

    print(f"PROV-01 probe: wrote artifact to {PROBE_ARTIFACT}")
    print(f"PROV-01 probe: verdict = {verdict!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
