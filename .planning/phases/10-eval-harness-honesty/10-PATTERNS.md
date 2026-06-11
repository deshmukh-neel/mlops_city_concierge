# Phase 10: Eval Harness Honesty - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 14 (new/modified)
**Analogs found:** 13 / 14

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/eval_agent.py` | service | request-response | self (EVAL-01 modifies lines 839-863 + 652-730 + 1112) | exact |
| `scripts/eval_matrix.py` | service | batch | self (EVAL-01/02 modifies lines 360-441 + 671-677 + 547-637) | exact |
| `app/agent/critique/checks.py` | utility | transform | self (EVAL-01 modifies Branch-1 guard at :488-491) | exact |
| `configs/eval_gates.yaml` | config | — | `configs/eval_matrix_refinement.yaml` | role-match |
| `configs/eval_matrix.yaml` | config | — | self (adds `baseline_eligible: false` to `late_night_closure_cascade`) | exact |
| `configs/eval_matrix_refinement.yaml` | config | — | self (gate-number comments point at gates YAML) | exact |
| `configs/eval_baselines/late_night_closure_cascade.json` | config | — | `configs/eval_baselines/omakase_mission_open_ended.json` | exact |
| `scripts/check_eval_gates.py` | utility | batch | `scripts/check_baselines_fresh.py` | exact |
| `scripts/probe_provider_capture.py` | utility | request-response | `scripts/probe_gpt5_capture.py` | exact |
| `docs/eval_gates.md` | config | — | no close analog (narrative doc) | none |
| `tests/unit/test_eval_agent.py` | test | transform | self (EVAL-01 error-path cases extend existing file) | exact |
| `tests/unit/test_eval_matrix.py` | test | batch | self (EVAL-04 verification/extension) | exact |
| `tests/unit/test_adapters.py` | test | transform | self (EVAL-05 parametrized fixture-loading tests extend existing file) | exact |
| `tests/unit/test_llm_factory.py` | test | transform | self (EVAL-06 gpt-5 dispatch + ScriptedChatModel ainvoke tests extend existing file) | exact |

---

## Pattern Assignments

### `scripts/eval_agent.py` — EVAL-01 error-path rewrite

**Analogs:** self; `scripts/check_baselines_fresh.py` (exit-code convention)

**Current fail-open exception handler to REMOVE** (`scripts/eval_agent.py:839-863`):

```python
# THIS BLOCK IS REMOVED (D-10-02). Exceptions never reach scorers.
except Exception as exc:  # noqa: BLE001
    total_latency += time.monotonic() - start_time
    partial_state = (
        state.model_copy(deep=True)
        if state is not None
        else ItineraryState(
            messages=messages_in,
            constraints=_constraints_for_case(case),
        )
    )
    partial_state.scratch.update(prior_scratch)
    partial_state.scratch.setdefault("multi_turn_runner", []).append(...)
    return (
        query_result_from_state(case, partial_state, latency_seconds=total_latency),
        partial_state,
    )
```

**Replacement pattern — ERROR record on exception** (new logic for BOTH `_run_prod_threading` and `_run_legacy_threading`):

The existing `CheckResult` dataclass (`eval_agent.py:84-92`) is the model for the new `RunErrorRecord`. Error runs write a per-run JSON with `"status": "error"` and an `"error"` block; scored fields are absent. Mirror the `write_report` pattern (`eval_agent.py:1090-1096`) but with a distinct schema:

```python
# New dataclass shape (D-10-01 schema):
@dataclass
class RunErrorRecord:
    status: Literal["error"] = "error"
    error: dict[str, str]  # keys: stage, type, message
    # stage in {"setup", "turn0", "turnN"} where N is the 1-based turn index
    # type is type(exc).__name__
    # message is str(exc)[:500]  # truncated
```

**Exit-code pattern** (copy from `check_baselines_fresh.py:215-278` + `eval_agent.py:1099-1112`):

```python
# check_baselines_fresh.py exit-code convention — three distinct exit codes:
#   0 = all clear
#   1 = gate failed (score violation)
#   2 = infrastructure failure (runtime error)
# eval_matrix.py MUST distinguish error count from violation count in output:
#   non-zero exit when errors OR violations; stderr line per category.
def main(argv: Sequence[str] | None = None) -> int:
    ...
    return 1 if report_has_errors(report) or report_has_violations(report) else 0
```

**Summary JSON threading** — new fields on `summary.json` aggregation (`eval_matrix.py:671-677`). Per-cell, the existing `_scorer_means_from_cell(payload)` reads the `aggregate` block; extend `aggregate_cell_jsons` to also read `n_scored` / `n_errored` fields from each cell JSON and surface them per-provider-key in the summary:

```python
# Shape to add inside aggregate_cell_jsons (after scorer stats block):
providers_out[provider_key] = {
    "scorers": {...},          # existing
    "n_scored": <int>,         # NEW: runs that produced a scored record
    "n_errored": <int>,        # NEW: runs that produced an error record
    "cell_valid": <bool>,      # NEW: n_scored == n_requested
}
# Top-level summary.json also gains:
summary["errors"] = [
    {"cell": "<filename>", "stage": "turn0", "type": "RateLimitError", ...},
    ...
]
```

**`score_checks` error path** (`eval_agent.py:475-495`) — RETAINED as-is. Individual check exceptions yield `CheckResult(score=None, passed=False, error=str(exc))`. This is distinct from a whole-run exception: a completed run with a failing individual check is still a scored record (`status: "ok"`), not an error record.

---

### `scripts/eval_matrix.py` — EVAL-01/02 aggregation + quarantine

**Analog:** self (structural-check mode at `:547-637`)

**Structural-check extension pattern** — add error-schema field validation inside the existing `if args.structural_check:` block (`eval_matrix.py:547-637`). Copy the five-check pattern precisely:

```python
# eval_matrix.py:547-637 — existing structural_check block pattern:
if args.structural_check:
    # Check N: validate new error-schema fields exist in a synthetic cell JSON
    synthetic_error_cell = {
        "status": "error",
        "error": {"stage": "turn0", "type": "RateLimitError", "message": "quota"},
    }
    assert "status" in synthetic_error_cell
    assert "error" in synthetic_error_cell
    assert synthetic_error_cell["error"].get("stage") in {"setup", "turn0", "turnN"}
    # ... print OK + return 0 on success
```

**`baseline_eligible` quarantine flag** — extend `MatrixEntry` (in `app/eval/config.py`) and the `iter_cells` / aggregation logic to skip cells with `baseline_eligible: false` from summary stats and gate checks. Config shape follows the existing `env` field pattern on `MatrixEntry`:

```python
# configs/eval_matrix.yaml — quarantine annotation (D-10-09/10):
scenarios:
  - omakase_mission_open_ended
  - late_night_closure_cascade
# late_night scenario block gains:
# baseline_eligible: false  # D-10-09: quarantined — legacy threading shape
# threading_mode: legacy   # already present in eval_queries.yaml
```

---

### `app/agent/critique/checks.py` — EVAL-01 Branch-1 guard

**Analog:** self (`:488-491`)

**Branch-1 abstain — RETAINED for completed non-refinement runs only.** The only change is that this branch is now unreachable from error-run paths (because exceptions never reach `score_checks`). No code change to `checks.py` may be needed beyond a clarifying comment:

```python
# checks.py:488-491 — RETAINED verbatim (D-10-04):
# Branch 1: abstain when not in refinement context.
refinement_context = bool(state.scratch.get("refinement_context", False))
if not refinement_context:
    return 1.0  # Legitimate abstain for completed non-refinement runs.
```

The key invariant: `score_checks` is only called on completed runs (`status: "ok"`). Branch-1 abstain fires when `refinement_context` is genuinely absent on a completed run — not from an exception-corrupted partial state.

---

### `configs/eval_gates.yaml` — EVAL-03 machine-readable gates

**Analog:** `configs/eval_matrix_refinement.yaml` (YAML structure with comments as rationale breadcrumbs)

**YAML schema pattern** (D-10-08 fields; copy comment style from `eval_matrix_refinement.yaml`):

```yaml
# configs/eval_gates.yaml — single source of truth for per-family merge gates.
# Consumed by scripts/check_eval_gates.py (make eval-gates-check).
# docs/eval_gates.md explains semantics and links here; it never duplicates numbers.

gates:
  - family: openai/gpt-4o-mini
    status: active                   # active | aspirational | provisional-n1 | logged | quarantined-legacy-threading
    rationale: "D-10-07: anchor; 0.8 absorbs one stochastic miss at n=5"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.8
    advisory:
      - metric: refinement_minimal_edit_median
        op: ">="
        value: 0.0        # logged-not-gated until v2.2 decisiveness work

  - family: openai/gpt-5-mini
    status: aspirational             # FAILS at 0.4; reported but not hard-failing
    rationale: "D-10-07 + D-09-02 Part A: v2.2 decisiveness target"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.6
    advisory: []

  - family: anthropic/claude-sonnet-4-6
    status: provisional-n1
    rationale: "D-10-07: single-run baseline; Phase 11 re-ratifies at n=5"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.8
    advisory: []

  - family: deepseek/deepseek-reasoner
    status: logged
    rationale: "D-10-07: logged-not-gated"
    hard: null
    advisory: []

  - family: deepseek/deepseek-chat
    status: logged
    rationale: "D-10-07: logged-not-gated"
    hard: null
    advisory: []

  - family: gemini/gemini-3.1-pro-preview
    status: logged
    rationale: "D-10-07: logged-not-gated"
    hard: null
    advisory: []

  - family: late_night_closure_cascade
    status: quarantined-legacy-threading   # D-10-09/10
    rationale: "D-10-09: legacy threading shape; excluded from baselines/gates"
    hard: null
    advisory: []
```

---

### `scripts/check_eval_gates.py` — EVAL-03 gate-check script

**Analog:** `scripts/check_baselines_fresh.py` (exact exit-code pattern, argparse style, `_run_git`-style helper, `main() -> int` with `raise SystemExit(main())`)

**Imports pattern** (copy from `check_baselines_fresh.py:1-57`):

```python
#!/usr/bin/env python3
"""Gate-check script for eval_gates.yaml (EVAL-03 / D-10-05).

Reads a matrix summary.json and exits non-zero on any hard-gate violation.

Usage:
    poetry run python scripts/check_eval_gates.py eval_reports/{ts}/summary.json
    make eval-gates-check SUMMARY=eval_reports/{ts}/summary.json

Exit codes:
    0 = all hard gates passed
    1 = one or more hard-gate violations (includes aspirational gates reported distinctly)
    2 = infrastructure failure (missing YAML, bad summary.json shape)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# No top-level LLM SDK imports — this script is the checker, not the caller.
```

**Argument parsing pattern** (copy from `check_baselines_fresh.py:163-185`):

```python
def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_eval_gates",
        description="Gate-check script for eval_gates.yaml (EVAL-03).",
    )
    parser.add_argument("summary", help="Path to summary.json from eval_matrix run.")
    parser.add_argument(
        "--gates-config",
        default="configs/eval_gates.yaml",
        help="Path to eval_gates.yaml (default: configs/eval_gates.yaml).",
    )
    return parser.parse_args(list(argv))
```

**Main + exit-code pattern** (copy from `check_baselines_fresh.py:215-279`):

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        gates_cfg = _load_gates(args.gates_config)
        summary = _load_summary(args.summary)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"check_eval_gates: {exc}\n")
        return 2

    violations: list[str] = []
    aspirational_misses: list[str] = []

    for gate in gates_cfg["gates"]:
        result = _check_gate(gate, summary)
        if result == "violation":
            violations.append(gate["family"])
        elif result == "aspirational_miss":
            aspirational_misses.append(gate["family"])

    if aspirational_misses:
        print(f"check_eval_gates: ASPIRATIONAL miss (not blocking): {aspirational_misses}")
    if violations:
        sys.stderr.write(f"check_eval_gates: HARD GATE VIOLATION: {sorted(violations)}\n")
        return 1

    print(f"check_eval_gates: OK — {len(gates_cfg['gates'])} gates checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

### `scripts/probe_provider_capture.py` — EVAL-05 generalized probe

**Analog:** `scripts/probe_gpt5_capture.py` (exact structure — generalize from gpt-5-only to `--provider {openai|deepseek|anthropic|gemini}`)

**Imports and repo-root pattern** (copy from `probe_gpt5_capture.py:29-42`):

```python
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
```

**Redaction pattern** (copy from `probe_gpt5_capture.py:63-75` and EXTEND per D-10-13):

```python
# D-10-13: extend beyond sk- prefix to cover all common API-key shapes
# AND env-var-sourced secret values.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),        # OpenAI
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),     # Anthropic
    re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),       # Google/Gemini
    re.compile(r"[A-Za-z0-9]{32,}-[A-Za-z0-9]{4,}"),  # generic long token
]

def _redact(value: object) -> str:
    s = str(value)
    for pat in _SECRET_PATTERNS:
        s = pat.sub("<REDACTED>", s)
    return s
```

**Fixture output path** (NEW — differs from probe_gpt5_capture.py's `.planning/` artifact path):

```python
# D-10-11: fixtures go to tests/fixtures/provider_payloads/{provider}.json
# (checked-in, reviewed before commit like any code).
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "provider_payloads"

def _fixture_path(provider: str) -> Path:
    return FIXTURE_DIR / f"{provider}.json"
```

**Fixture JSON shape** (D-10-11 — structured dict, not a markdown artifact):

```python
# Write as JSON, not markdown (unlike probe_gpt5_capture.py's .md artifact).
# Parametrized loader tests in test_adapters.py consume this shape.
fixture = {
    "provider": provider,
    "model": model_name,
    "library_version": importlib.metadata.version(library_pkg),
    "probe_query": PROBE_QUERY,
    "additional_kwargs_keys": sorted(message.additional_kwargs.keys()),
    "additional_kwargs_values": {k: _redact(v) for k, v in message.additional_kwargs.items()},
    "response_metadata": _sanitize_response_metadata(dict(message.response_metadata or {})),
    "content_shape": _content_shape(message.content),
    "usage_metadata": getattr(message, "usage_metadata", None),
    "tool_calls": getattr(message, "tool_calls", None) or [],
}
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
_fixture_path(provider).write_text(json.dumps(fixture, indent=2), encoding="utf-8")
```

**Argparse pattern** (NEW — generalize from hardcoded to `--provider`):

```python
def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe_provider_capture")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "deepseek", "anthropic", "gemini"],
    )
    parser.add_argument("--model", default=None, help="Override default model for provider.")
    return parser.parse_args(list(argv))
```

**Final secret-scan guard** (copy from `probe_gpt5_capture.py:226-233`):

```python
# Defensive post-write scan — blow up before user commits if a secret leaked.
text = _fixture_path(provider).read_text(encoding="utf-8")
for pat in _SECRET_PATTERNS:
    if pat.search(text):
        print(f"FATAL: fixture contains secret pattern. NOT committing.", file=sys.stderr)
        return 2
```

---

### `tests/unit/test_eval_agent.py` — EVAL-01 error-path tests

**Analog:** `tests/unit/test_eval_agent.py` (self), `tests/unit/test_critique_checks.py` (FakeConn/state construction pattern), `tests/_helpers/scripted_llm.py` (raising-on-demand stub)

**Test structure pattern** (copy from `test_eval_agent.py` — `eval_case()` builder + parametrize pattern at `:47-59` and `:149-159`):

```python
def eval_case(**overrides: object) -> EvalQuery:
    """Build a minimal EvalQuery for eval-agent unit tests."""
    payload = {
        "id": "case_one",
        "query": "coffee in soma",
        "reference": "Recommend a cafe in SOMA.",
        "expected_results": {"min_stops": 1, "max_stops": 3},
    }
    payload.update(overrides)
    return EvalQuery.model_validate(payload)
```

**Scripted-LLM raising stub** (use `tests/_helpers/scripted_llm.py` `ScriptedLLM` for EVAL-01 acceptance test; it raises on exhaustion, enabling turn-0 and turn-N exception simulation):

```python
# From tests/_helpers/scripted_llm.py — raises IndexError on exhaustion.
# For EVAL-01 acceptance: give the stub a RAISING variant instead:
class RaisingChatModel(BaseChatModel):
    """Raises on every invoke — simulates 429 / 400 / DB-down for EVAL-01."""
    raise_type: type[Exception] = Exception
    raise_msg: str = "simulated infra failure"

    def _generate(self, messages, **kwargs):
        raise self.raise_type(self.raise_msg)

    @property
    def _llm_type(self) -> str:
        return "raising"
```

**EVAL-01 acceptance test assertions** (D-10-04 spec):

```python
# Assert all three fail-open paths produce ERROR records:
# 1. turn-0 LLM exception → ERROR record, stage="turn0"
# 2. turn-1 LLM exception → ERROR record, stage="turn1"
# 3. retrieval-only exception (tool error, NOT an LLM exception) → still a scored record
#    but scored 0.0, NOT producing the asymmetric 1.0 abstain
#
# Assert former bad outcomes are GONE:
# - Branch-1 abstain-1.0 path with partial state: must not appear on error runs
# - prior-vs-itself 1.0: must not appear on error runs
# - retrieval-0.0 asymmetry: documented via score, not via ERROR record
```

---

### `tests/unit/test_eval_matrix.py` — EVAL-04 parity verification

**Analog:** self (existing file, extend only)

**Existing EVAL-04 anchor** (from `test_eval_matrix.py:116-149` — `test_baseline_provider_cells_match_matrix_entries`). This test is already the EVAL-04 anchor per PR #104. The EVAL-04 task is verification, not rewrite.

**Extension pattern for EVAL-02 quarantine flag** — add a test asserting `baseline_eligible: false` is honored in `MatrixEntry` parsing:

```python
def test_late_night_scenario_has_baseline_eligible_false() -> None:
    """D-10-09: late_night_closure_cascade must be quarantined from baseline
    aggregation. Verify the flag parses from the matrix YAML."""
    from app.eval.config import load_eval_matrix, REPO_ROOT
    # After EVAL-02 edit, MatrixEntry or scenario config gains baseline_eligible.
    # Test shape mirrors test_repo_eval_matrix_yaml_loads_via_load_eval_matrix.
    ...
```

**Deferred-cells pattern** (copy `_DEFERRED_BASELINE_CELLS` dict at `:101-104` for any new deferred cells from EVAL-02/03):

```python
_DEFERRED_BASELINE_CELLS: dict[str, set[str]] = {
    "eval_matrix_refinement.yaml": {"gemini/gemini-3.1-pro-preview"},
    "eval_matrix.yaml": set(),  # no new deferrals; quarantine ≠ deferral
}
```

---

### `tests/unit/test_adapters.py` — EVAL-05 parametrized fixture-loading tests

**Analog:** self (existing file — extend with parametrized fixture cases)

**Existing parametrized pattern** (`test_adapters.py:1-119` — one test per adapter method, using synthesized `AIMessage` dicts). EVAL-05 adds a parametrized variant that loads the real-wire JSON fixture:

```python
import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "provider_payloads"

@pytest.mark.parametrize("provider", ["openai", "deepseek", "anthropic", "gemini"])
def test_adapter_capture_on_real_wire_fixture(provider: str) -> None:
    """EVAL-05 / D-10-12: Load checked-in real-wire fixture and verify the
    provider's adapter capture_reasoning_state does not crash.
    
    Existing synthetic dict tests document the contract and run without files;
    this test closes the live-shape gap (D-09-09 Gemini lcgg key miss, 4 live
    Anthropic bugs).
    """
    fixture_path = FIXTURE_DIR / f"{provider}.json"
    if not fixture_path.exists():
        pytest.skip(f"No fixture for {provider} — run make probe-providers first")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    msg = AIMessage(
        content="x",
        additional_kwargs=payload.get("additional_kwargs_values", {}),
        response_metadata=payload.get("response_metadata", {}),
    )
    adapter = _adapter_for(provider)  # dispatches to the right adapter class
    # Must not raise; result shape is either None or a dict with "provider" key.
    result = adapter.capture_reasoning_state(msg)
    if result is not None:
        assert "provider" in result
```

---

### `tests/unit/test_llm_factory.py` — EVAL-06 gpt-5 dispatch + ScriptedChatModel ainvoke

**Analog:** self (existing file — copy dispatch test pattern at `:8-35`)

**gpt-5 dispatch test pattern** (D-10-15 — copy from `test_build_chat_model_dispatches_per_provider` pattern):

```python
def test_build_chat_model_gpt5_returns_openai_reasoning_chat_model(
    mocker, monkeypatch
) -> None:
    """D-10-15: gpt-5-* routes through OpenAIReasoningChatModel with
    use_responses_api=True. Currently ZERO tests reference use_responses_api.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from app.config import get_settings
    get_settings.cache_clear()
    cls = mocker.patch(
        "app.llm_factory.OpenAIReasoningChatModel",
        return_value="gpt5-reasoning-llm",
    )
    out = build_chat_model("openai", "gpt-5-mini", temperature=1.0)
    assert out == "gpt5-reasoning-llm"
    cls.assert_called_once()
    _, kwargs = cls.call_args
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["use_responses_api"] is True


def test_build_chat_model_gpt4o_mini_stays_plain_chat_openai(
    mocker, monkeypatch
) -> None:
    """D-10-15 regression guard: gpt-4o-mini MUST NOT be routed through
    OpenAIReasoningChatModel — it must stay on plain ChatOpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from app.config import get_settings
    get_settings.cache_clear()
    reasoning_cls = mocker.patch("app.llm_factory.OpenAIReasoningChatModel")
    plain_cls = mocker.patch("app.llm_factory.ChatOpenAI", return_value="plain-llm")
    out = build_chat_model("openai", "gpt-4o-mini", temperature=1.0)
    assert out == "plain-llm"
    reasoning_cls.assert_not_called()
```

**ScriptedChatModel ainvoke pattern** (D-10-16 — use `asyncio.run` or `pytest.mark.asyncio`):

```python
import asyncio

async def test_scripted_chat_model_ainvoke_works() -> None:
    """D-10-16: The graph only ever calls ainvoke; verify BaseChatModel
    executor fallback works for ScriptedChatModel (no _agenerate override).
    
    Copy from tests/unit/test_helpers_scripted_llm.py pattern — no network.
    """
    from app.llm_factory import ScriptedChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    llm = ScriptedChatModel(scripted=[AIMessage(content="hello")])
    result = await llm.ainvoke([HumanMessage(content="go")])
    assert result.content == "hello"
```

---

### `configs/eval_matrix.yaml` — EVAL-02 quarantine flag

**Analog:** `configs/eval_matrix_refinement.yaml` (YAML comment style documenting decision IDs)

**Quarantine annotation** (D-10-09/10 — add `baseline_eligible: false` comment block adjacent to `late_night_closure_cascade`):

```yaml
scenarios:
  - omakase_mission_open_ended
  - late_night_closure_cascade
    # D-10-09: quarantined from baselines/gates — legacy threading shape.
    # The closure-cascade turn-2 scorers were designed against full-tool-history
    # shape (project_eval_multi_turn_threading_bug); migrating to prod threading
    # redesigns the scenario (deferred). baseline_eligible: false honored by
    # matrix runner and baseline tooling.
    # baseline_eligible: false
```

---

### `configs/eval_baselines/late_night_closure_cascade.json` — EVAL-02 annotation only

**Analog:** `configs/eval_baselines/omakase_mission_open_ended.json` (existing shape)

**Annotation pattern** (D-10-10 — add `_observations` key, do NOT regen):

```json
{
  "_observations": "D-10-10: legacy-threading-shaped measurement. Not comparable to prod. Baseline NOT regenerated in Phase 10; annotated only. See eval_gates.yaml cell status: quarantined-legacy-threading.",
  "scenario_id": "late_night_closure_cascade",
  ...existing content unchanged...
}
```

---

### `docs/eval_gates.md` — EVAL-03 narrative doc

**Analog:** none (no existing narrative eval docs). Follow project README conventions — plain markdown, links to `configs/eval_gates.yaml`, no duplicated numbers.

**Structure pattern:**

```markdown
# Eval Gates

This document explains the merge-gate semantics for the eval matrix.
Numbers live in `configs/eval_gates.yaml`; this file explains only the semantics.

## Gate statuses
- `active` — hard gate enforced by `make eval-gates-check`
- `aspirational` — reported but not blocking (known v2.2 gap)
- `provisional-n1` — hard gate from a single run; Phase 11 re-ratifies at n=5
- `logged` — no gate; empirical median captured for reference
- `quarantined-legacy-threading` — excluded from baselines/gates (D-10-09)

## Running the gate check
    make eval-gates-check SUMMARY=eval_reports/{ts}/summary.json

## Adding a new gate
Edit `configs/eval_gates.yaml`. The planner must provide a D-ID rationale.
```

---

## Shared Patterns

### Error-record schema threading

**Source:** `scripts/eval_agent.py` + `scripts/eval_matrix.py`
**Apply to:** `eval_agent.py` (per-run JSON writer), `eval_matrix.py` (aggregation), `check_eval_gates.py` (gate reader)

The `status` field must thread through all four layers. Per-run JSON gains `status: "ok" | "error"`. Aggregation gains `n_scored` / `n_errored` per cell. Gate checker reads `n_scored` to determine cell validity (D-10-03: cells with `n_scored < n_requested` are `INVALID_FOR_BASELINE`).

### Exit-code conventions

**Source:** `scripts/check_baselines_fresh.py:215-279`
**Apply to:** `scripts/check_eval_gates.py`, `scripts/eval_matrix.py` (non-zero exit when errors AND when violations, counted separately in stderr output)

```python
#   0 = all clear
#   1 = gate failed (score violation OR error count > 0)
#   2 = infrastructure failure (missing YAML, bad summary.json)
# Exit code convention from check_baselines_fresh.py — copy exactly.
if __name__ == "__main__":
    raise SystemExit(main())
```

### Redaction pattern

**Source:** `scripts/probe_gpt5_capture.py:63-75`
**Apply to:** `scripts/probe_provider_capture.py` (D-10-13 extension), `tests/unit/test_adapters.py` (redaction unit test)

The existing `_SECRET_PATTERN` covers only `sk-` prefixes. D-10-13 extends to cover Anthropic (`sk-ant-`), Google API keys (`AIzaSy`), and env-var-sourced secrets. Unit test feeds a fake leaked key through the probe writer and asserts redaction:

```python
def test_probe_redaction_catches_anthropic_key() -> None:
    from scripts.probe_provider_capture import _redact
    assert "sk-ant-REDACTED" in _redact("sk-ant-api03-abc123xyz789")
    assert "sk-ant-api03-abc123xyz789" not in _redact("sk-ant-api03-abc123xyz789")
```

### Config-consumed-by-script pattern

**Source:** `scripts/check_baselines_fresh.py` + `Makefile` eval targets
**Apply to:** `configs/eval_gates.yaml` + `scripts/check_eval_gates.py` + Makefile `eval-gates-check` target

The established pattern: a machine-readable `configs/` YAML is consumed by a `scripts/` Python file with a `make` target wrapper. The target takes a required path arg for the runtime artifact (summary.json here) and a `--gates-config` default:

```makefile
# Makefile pattern (copy from eval-matrix-refinement-structural-check target):
.PHONY: eval-gates-check
eval-gates-check: ## Check summary.json against configs/eval_gates.yaml (EVAL-03)
	$(POETRY_RUN) python scripts/check_eval_gates.py \
	  $(SUMMARY) \
	  --gates-config configs/eval_gates.yaml
```

### Test layering (`feedback_test_layering`)

**Source:** project memory + `tests/unit/test_eval_agent.py` + `tests/unit/test_critique_checks.py`
**Apply to:** All EVAL-01 tests

Per the memory: unit (scorer branch isolation), functional (scripted-LLM end-to-end error run without live calls), and acceptance (21-14-30Z replay simulation). The acceptance test uses `ScriptedChatModel` / `RaisingChatModel` stubs — no real API calls. Never use `asyncio_mode = "auto"` inconsistently; tests already use `pytest` with `asyncio_mode = "auto"` (pyproject.toml).

### `monkeypatch` + `get_settings.cache_clear()` pattern

**Source:** `tests/unit/test_llm_factory.py:8-35`
**Apply to:** All new `test_llm_factory.py` tests

```python
def test_...(mocker, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from app.config import get_settings
    get_settings.cache_clear()  # MANDATORY — cached settings survive monkeypatch
    cls = mocker.patch("app.llm_factory.SomeClass", return_value="mock-llm")
    out = build_chat_model(...)
    cls.assert_called_once()
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `docs/eval_gates.md` | config/doc | — | No existing narrative eval-semantics doc in `docs/`; closest is inline YAML comments in `configs/`. |

---

## Metadata

**Analog search scope:** `scripts/`, `app/agent/critique/`, `app/llm_factory.py`, `tests/unit/`, `configs/`, `Makefile`
**Files scanned:** 20
**Pattern extraction date:** 2026-06-10
