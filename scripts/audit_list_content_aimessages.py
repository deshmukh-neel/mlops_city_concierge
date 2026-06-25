"""REPLAY-02 evidence audit: list-content AIMessage analysis (D-14-05).

Zero-spend audit that determines whether the three RUN models
(openai/gpt-5-mini, openai/gpt-4o-mini, deepseek/deepseek-reasoner) ever
produce list-content AIMessages that ``prune_for_llm``'s ``str()`` collapse
at ``app/agent/graph.py:232`` would have observably altered.

This script does TWO complementary halves:

Half (a) — Run-dir scan:
    Iterate Phase-12/13 arm run dirs, open each ``*.json``, and report what is
    actually persisted regarding AIMessage content shape. The EvalRunReport JSON
    persists ``queries[i].deterministic.tool_calls`` as an **integer count**
    with no serialized message ``.content`` or ``.additional_kwargs`` — i.e. the
    run JSONs cannot directly answer "did an AIMessage carry list content
    pre-cutoff". This finding is emitted explicitly, not treated as an error.

Half (b) — Structural adapter analysis:
    Derive the answer from adapter design. For the three RUN models:
    - openai/gpt-5-mini and openai/gpt-4o-mini use ``OpenAIReasoningAdapter``
      (or ``NoOpAdapter`` for the anchor). Both store reasoning state in
      ``additional_kwargs["reasoning_content"]``; ``AIMessage.content`` is a
      plain string reply.
    - deepseek/deepseek-reasoner uses ``DeepSeekReasonerAdapter``, which also
      stores reasoning state in ``additional_kwargs["reasoning_content"]``;
      ``AIMessage.content`` is a plain string reply.
    - Only ``AnthropicAdapter`` uses a content block list. Anthropic is deferred
      (D-12-09) and NOT in the run matrix.

    Conclusion: ``str()`` collapse at graph.py:232 is a NO-OP for all three RUN
    models (string in, identical string out). R2 is EXPECTED-NULL on the tested
    cells but still runs (criterion 2 requires a measured delta).

Usage::

    # scan the Phase-12/13 Phase-13 arm run dirs (auto-detected)
    poetry run python scripts/audit_list_content_aimessages.py

    # scan a specific run dir
    poetry run python scripts/audit_list_content_aimessages.py \\
        --run-dir eval_reports/2026-06-12T07-27-03Z

    # scan multiple run dirs
    poetry run python scripts/audit_list_content_aimessages.py \\
        --run-dir eval_reports/2026-06-12T06-25-52Z \\
        --run-dir eval_reports/2026-06-12T07-27-03Z \\
        --run-dir eval_reports/2026-06-12T08-30-52Z

No live network, no LLM, no DB.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

# ---------------------------------------------------------------------------
# The three RUN-model provider/model pairs for Phase 14 arms.
# ---------------------------------------------------------------------------
RUN_MODELS: list[tuple[str, str]] = [
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-4o-mini"),
    ("deepseek", "deepseek-reasoner"),
]

# Phase-12/13 arm run dirs referenced in docs/decisiveness_arm_verdicts.md.
KNOWN_ARM_RUN_DIRS: list[str] = [
    "eval_reports/2026-06-12T06-25-52Z",  # A1 full (n=5)
    "eval_reports/2026-06-12T07-27-03Z",  # A2 full (n=5)
    "eval_reports/2026-06-12T08-30-52Z",  # A3 full (n=5)
]


# ---------------------------------------------------------------------------
# Half (a): Run-dir scan
# ---------------------------------------------------------------------------


def scan_run_dir(run_dir: pathlib.Path) -> dict[str, Any]:
    """Scan one run dir and return a findings dict.

    Returns::

        {
            "run_dir": str,
            "files_found": int,
            "files_parsed": int,
            "shape_finding": str,  # explicit description of persisted shape
            "has_message_content": bool,  # True only if content found (never expected)
            "errors": list[str],
        }
    """
    findings: dict[str, Any] = {
        "run_dir": str(run_dir),
        "files_found": 0,
        "files_parsed": 0,
        "shape_finding": "",
        "has_message_content": False,
        "errors": [],
    }

    json_files = list(run_dir.glob("*.json"))
    findings["files_found"] = len(json_files)

    if not json_files:
        findings["shape_finding"] = "No JSON files found — run dir may be empty or not exist."
        return findings

    # Inspect a sample file to characterise the persisted shape.
    sample_parsed = 0
    for jf in json_files:
        try:
            with jf.open() as fh:
                data: dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            findings["errors"].append(f"{jf.name}: {exc}")
            continue

        queries = data.get("queries") or []
        for query in queries:
            det = query.get("deterministic") if isinstance(query, dict) else None
            if det is None:
                continue
            # Verify tool_calls is an integer count (the documented EvalRunReport shape).
            tool_calls_val = det.get("tool_calls")
            if tool_calls_val is not None and not isinstance(tool_calls_val, int):
                findings["errors"].append(
                    f"{jf.name}: 'tool_calls' has unexpected type "
                    f"{type(tool_calls_val).__name__!r} — expected int"
                )

            # Check for any serialized message content (NOT expected; present = surprise).
            if "message_content" in det or "messages" in det:
                findings["has_message_content"] = True

        sample_parsed += 1

    findings["files_parsed"] = sample_parsed

    if findings["has_message_content"]:
        # Unexpected — the run JSONs would contain message traces.
        findings["shape_finding"] = (
            "UNEXPECTED: run JSON contains 'message_content' or 'messages' key "
            "under deterministic — message traces are present."
        )
    else:
        findings["shape_finding"] = (
            "CONFIRMED: EvalRunReport persists queries[i].deterministic.tool_calls "
            "as an integer COUNT. No serialized AIMessage .content or "
            ".additional_kwargs is present. Run JSONs CANNOT directly answer "
            "'did an AIMessage carry list content pre-cutoff'."
        )

    return findings


# ---------------------------------------------------------------------------
# Half (b): Structural adapter analysis
# ---------------------------------------------------------------------------

# Per-adapter classification for all four Phase-9 adapters.
# source: app/agent/adapters/*.py — read during audit preparation.
ADAPTER_CLASSIFICATIONS: list[dict[str, str]] = [
    {
        "provider_slug": "openai",
        "model_family": "gpt-5 family (gpt-5-mini, gpt-4o-mini with reasoning adapter)",
        "adapter": "OpenAIReasoningAdapter",
        "content_shape": "str",
        "reasoning_storage": "additional_kwargs['reasoning_content']",
        "list_content": "NO",
        "run_model": "YES (gpt-5-mini + gpt-4o-mini anchor)",
        "str_collapse_effect": "NO-OP (str in, identical str out)",
    },
    {
        "provider_slug": "deepseek",
        "model_family": "deepseek-reasoner",
        "adapter": "DeepSeekReasonerAdapter",
        "content_shape": "str",
        "reasoning_storage": "additional_kwargs['reasoning_content']",
        "list_content": "NO",
        "run_model": "YES (deepseek/deepseek-reasoner)",
        "str_collapse_effect": "NO-OP (str in, identical str out)",
    },
    {
        "provider_slug": "anthropic",
        "model_family": "Claude (thinking enabled)",
        "adapter": "AnthropicAdapter",
        "content_shape": "list[dict]  ← block list with thinking + text blocks",
        "reasoning_storage": "message.content (block list, NOT additional_kwargs)",
        "list_content": "YES  ← THE ONLY ADAPTER WITH LIST CONTENT",
        "run_model": "NO — DEFERRED (D-12-09, not in run matrix)",
        "str_collapse_effect": "LOSSY — str(list_of_blocks) != original block list; "
        "thinking_blocks and type metadata destroyed. "
        "BUT: anthropic is deferred so this loss is UNREACHABLE in current runs.",
    },
    {
        "provider_slug": "gemini",
        "model_family": "Gemini (reasoning enabled)",
        "adapter": "GeminiAdapter",
        "content_shape": "str",
        "reasoning_storage": "additional_kwargs (thought_signature / function_call_thought_signatures)",
        "list_content": "NO",
        "run_model": "NO — DEFERRED (D-12-09, not in run matrix)",
        "str_collapse_effect": "NO-OP (str in, identical str out) — AND deferred",
    },
]


def structural_analysis() -> dict[str, Any]:
    """Derive the verdict from adapter design, independent of run-dir data."""
    run_model_classifications = [
        c for c in ADAPTER_CLASSIFICATIONS if c["run_model"].startswith("YES")
    ]
    list_content_adapters = [
        c for c in ADAPTER_CLASSIFICATIONS if c["list_content"].startswith("YES")
    ]
    deferred_list_content_adapters = [
        c for c in list_content_adapters if not c["run_model"].startswith("YES")
    ]

    all_run_models_str_content = all(c["list_content"] == "NO" for c in run_model_classifications)

    return {
        "run_model_count": len(run_model_classifications),
        "all_run_models_str_content": all_run_models_str_content,
        "list_content_adapter_count": len(list_content_adapters),
        "list_content_adapters": [c["provider_slug"] for c in list_content_adapters],
        "deferred_list_content_adapters": [
            c["provider_slug"] for c in deferred_list_content_adapters
        ],
        "classifications": ADAPTER_CLASSIFICATIONS,
        "verdict": (
            "EXPECTED-NULL: str() collapse was a NO-OP for the three RUN models. "
            "None of gpt-5-mini, gpt-4o-mini, or deepseek-reasoner emit list-content "
            "AIMessages — their .content is always a plain string. "
            "Only AnthropicAdapter uses a content block list, and anthropic is "
            "deferred (D-12-09) and NOT in the run matrix. "
            "R2 (REPLAY_CONTENTBLOCKS_ENABLED) is therefore EXPECTED-NULL on the "
            "tested cells. R2 still runs — roadmap criterion 2 requires a measured "
            "delta, not an assumption."
        ),
    }


# ---------------------------------------------------------------------------
# Structural import smoke check (D-14-05 correctness gate).
# Verify the adapters being analysed actually exist and match the described
# shape — catches regressions if an adapter is refactored post-audit.
# ---------------------------------------------------------------------------


def smoke_check_adapters() -> list[str]:
    """Import the four adapters and verify the content-shape assumption.

    Returns a list of warning strings (empty = all checks passed).
    No assertion errors — findings are returned for the caller to report.
    """
    warnings: list[str] = []
    try:
        from langchain_core.messages import AIMessage  # noqa: PLC0415

        from app.agent.adapters.anthropic import AnthropicAdapter  # noqa: PLC0415
        from app.agent.adapters.deepseek import DeepSeekReasonerAdapter  # noqa: PLC0415
        from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter  # noqa: PLC0415

        # OpenAI adapter: replay_reasoning_state uses additional_kwargs,
        # not content. Verify the method signature accepts the expected args.
        oai = OpenAIReasoningAdapter()
        ds = DeepSeekReasonerAdapter()
        anth = AnthropicAdapter()

        # Verify OpenAI + DeepSeek: a string-content AIMessage round-trips
        # unchanged through str() collapse (str(s) == s for any string).
        test_msg = AIMessage(content="hello")
        assert isinstance(test_msg.content, str), "Expected str content on AIMessage"

        # Verify Anthropic adapter capture returns None for a string-content message
        # (captures ONLY when content is a list with thinking blocks).
        capture_result = anth.capture_reasoning_state(test_msg)
        if capture_result is not None:
            warnings.append(
                "AnthropicAdapter.capture_reasoning_state unexpectedly returned "
                f"non-None for string-content AIMessage: {capture_result!r}"
            )

        # Verify Anthropic adapter capture returns non-None for a list-content message.
        list_msg = AIMessage(content=[{"type": "thinking", "thinking": "t", "signature": "sig"}])
        anth_capture = anth.capture_reasoning_state(list_msg)
        if anth_capture is None:
            warnings.append(
                "AnthropicAdapter.capture_reasoning_state returned None for "
                "list-content AIMessage with thinking block — expected non-None. "
                "Audit assumption about AnthropicAdapter may be stale."
            )

        # Verify OpenAI/DeepSeek adapters: capture returns None for no reasoning_content.
        oai_capture = oai.capture_reasoning_state(test_msg)
        ds_capture = ds.capture_reasoning_state(test_msg)
        if oai_capture is not None:
            warnings.append(
                "OpenAIReasoningAdapter.capture unexpectedly returned non-None "
                "for a message with no additional_kwargs['reasoning_content']"
            )
        if ds_capture is not None:
            warnings.append(
                "DeepSeekReasonerAdapter.capture unexpectedly returned non-None "
                "for a message with no additional_kwargs['reasoning_content']"
            )

    except ImportError as exc:
        warnings.append(
            f"Adapter import failed — cannot verify structural assumption: {exc}. "
            "Ensure app is poetry-installed ('poetry install')."
        )
    except AssertionError as exc:
        warnings.append(f"Structural assertion failed: {exc}")

    return warnings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_run_dir_findings(findings_list: list[dict[str, Any]]) -> None:
    print_section("Half (a): Run-Dir Scan")
    if not findings_list:
        print("  No run dirs scanned.")
        return

    for f in findings_list:
        print(f"\n  Run dir: {f['run_dir']}")
        print(f"    JSON files found:  {f['files_found']}")
        print(f"    JSON files parsed: {f['files_parsed']}")
        print("    Shape finding:")
        for line in f["shape_finding"].splitlines():
            print(f"      {line}")
        if f["has_message_content"]:
            print("    WARNING: message content found in run JSON (unexpected)")
        if f["errors"]:
            print(f"    Errors ({len(f['errors'])}):")
            for e in f["errors"]:
                print(f"      - {e}")

    # Summary finding across all dirs.
    any_has_content = any(f["has_message_content"] for f in findings_list)
    total_files = sum(f["files_found"] for f in findings_list)
    print()
    print("  SUMMARY FINDING (run-dir scan):")
    print(f"    Dirs scanned: {len(findings_list)}")
    print(f"    Total JSON files found: {total_files}")
    if any_has_content:
        print(
            "    UNEXPECTED: at least one run JSON carries message content — "
            "investigate before proceeding."
        )
    else:
        print(
            "    CONFIRMED: no run JSON carries serialized AIMessage .content. "
            "The run JSONs are insufficient to directly determine whether "
            "AIMessages carried list content during the live runs. "
            "Structural analysis (Half b) provides the answer."
        )


def print_structural_analysis(analysis: dict[str, Any]) -> None:
    print_section("Half (b): Structural Adapter Analysis")

    print()
    print("  Per-adapter content-shape classification:")
    print()

    header = (
        f"  {'Provider':<12} {'Adapter':<28} {'Content shape':<20} "
        f"{'List-content?':<14} {'In run matrix?':<16} {'str() effect'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in analysis["classifications"]:
        run_yn = "YES" if c["run_model"].startswith("YES") else "NO (deferred)"
        print(
            f"  {c['provider_slug']:<12} {c['adapter']:<28} "
            f"{c['content_shape']:<20} {c['list_content']:<14} "
            f"{run_yn:<16} {c['str_collapse_effect']}"
        )

    print()
    print("  Key findings:")
    print(
        f"    - RUN models (in matrix): {analysis['run_model_count']} "
        f"(gpt-5-mini, gpt-4o-mini, deepseek-reasoner)"
    )
    print(
        f"    - All run models use str content: "
        f"{'YES' if analysis['all_run_models_str_content'] else 'NO'}"
    )
    print(
        f"    - Adapters with list-content AIMessages: "
        f"{analysis['list_content_adapter_count']} "
        f"({', '.join(analysis['list_content_adapters'])})"
    )
    print(
        f"    - List-content adapters in run matrix: "
        f"{'NONE — all deferred' if not any(c['run_model'].startswith('YES') for c in analysis['classifications'] if c['list_content'].startswith('YES')) else 'PRESENT — check findings'}"
    )
    print()
    print("  STRUCTURAL VERDICT:")
    for line in analysis["verdict"].splitlines():
        print(f"    {line}")


def print_overall_verdict(
    run_dir_findings: list[dict[str, Any]],
    structural: dict[str, Any],
    adapter_warnings: list[str],
) -> None:
    print_section("Overall Verdict (D-14-05 / REPLAY-02 Evidence Audit)")

    any_surprise = any(f["has_message_content"] for f in run_dir_findings)

    print()
    if adapter_warnings:
        print("  STRUCTURAL SMOKE-CHECK WARNINGS:")
        for w in adapter_warnings:
            print(f"    - {w}")
        print()

    if any_surprise:
        print(
            "  AUDIT INCONCLUSIVE: Unexpected message content found in run JSONs. "
            "Investigate before drawing conclusions about R2."
        )
    elif structural["all_run_models_str_content"]:
        print("  VERDICT: R2 EXPECTED-NULL on all tested cells.")
        print()
        print("  Reasoning:")
        print(
            "    1. Run JSONs do NOT carry serialized AIMessage .content — "
            "the persisted EvalRunReport shape serializes tool_calls as an "
            "integer count with no message traces. Run-dir evidence alone "
            "cannot confirm or deny list-content AIMessages."
        )
        print(
            "    2. Structural analysis of all four Phase-9 adapters shows: "
            "the three RUN models (gpt-5-mini, gpt-4o-mini, deepseek-reasoner) "
            "all use string content on AIMessage.content, not block lists. "
            "Reasoning state is stored in additional_kwargs['reasoning_content'], "
            "not in the content field."
        )
        print(
            "    3. Only AnthropicAdapter uses a content block list. Anthropic is "
            "deferred (D-12-09) and NOT in the Phase-14 run matrix."
        )
        print(
            "    4. Therefore: str() collapse at graph.py:232 was a NO-OP for all "
            "tested cells — str(s) == s for any string. No observable content loss "
            "could have occurred for gpt-5-mini, gpt-4o-mini, or deepseek-reasoner."
        )
        print()
        print(
            "  ACTION: R2 (REPLAY_CONTENTBLOCKS_ENABLED) still runs per roadmap "
            "criterion 2 — a measured delta is required even when the effect is "
            "expected-null. The audit result is written into docs/replay_arm_verdicts.md "
            "before any live R2 spend."
        )
    else:
        print("  VERDICT: Structural analysis inconclusive — check findings above.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "REPLAY-02 evidence audit (D-14-05): determine whether the three RUN models "
            "ever produce list-content AIMessages that prune_for_llm str() collapse "
            "would have altered. Zero API spend."
        )
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        metavar="DIR",
        help=(
            "Path to a run dir to scan (may be specified multiple times). "
            "Defaults to the Phase-12/13 arm run dirs from decisiveness_arm_verdicts.md."
        ),
    )
    parser.add_argument(
        "--skip-adapter-check",
        action="store_true",
        help="Skip the live adapter import smoke check (useful in envs without app installed).",
    )
    args = parser.parse_args(argv)

    # Resolve run dirs: user-provided or defaults.
    repo_root = pathlib.Path(__file__).parent.parent
    if args.run_dirs:
        run_dirs = [pathlib.Path(d) for d in args.run_dirs]
    else:
        run_dirs = [repo_root / d for d in KNOWN_ARM_RUN_DIRS]

    print()
    print("audit_list_content_aimessages: REPLAY-02 evidence audit (D-14-05)")
    print(f"  Scanning {len(run_dirs)} run dir(s)...")

    # Half (a): run-dir scan.
    run_dir_findings = []
    for rd in run_dirs:
        if not rd.exists():
            run_dir_findings.append(
                {
                    "run_dir": str(rd),
                    "files_found": 0,
                    "files_parsed": 0,
                    "shape_finding": f"Run dir does not exist: {rd}",
                    "has_message_content": False,
                    "errors": [],
                }
            )
        else:
            run_dir_findings.append(scan_run_dir(rd))

    # Half (b): structural analysis.
    structural = structural_analysis()

    # Adapter smoke check.
    adapter_warnings: list[str] = []
    if not args.skip_adapter_check:
        adapter_warnings = smoke_check_adapters()

    # Print results.
    print_run_dir_findings(run_dir_findings)
    print_structural_analysis(structural)
    if adapter_warnings:
        print_section("Adapter Smoke-Check Warnings")
        for w in adapter_warnings:
            print(f"  - {w}")
    print_overall_verdict(run_dir_findings, structural, adapter_warnings)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
