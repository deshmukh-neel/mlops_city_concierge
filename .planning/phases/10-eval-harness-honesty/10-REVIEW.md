---
phase: 10-eval-harness-honesty
reviewed: 2026-06-10T00:00:00Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - app/agent/critique/checks.py
  - app/agent/critique/vibe.py
  - app/eval/config.py
  - configs/eval_baselines/late_night_closure_cascade.json
  - configs/eval_gates.yaml
  - configs/eval_matrix_refinement.yaml
  - configs/eval_matrix.yaml
  - configs/eval_queries.yaml
  - docs/eval_gates.md
  - scripts/check_eval_gates.py
  - scripts/eval_agent.py
  - scripts/eval_matrix.py
  - scripts/probe_provider_capture.py
  - tests/unit/test_adapters.py
  - tests/unit/test_check_eval_gates.py
  - tests/unit/test_eval_agent.py
  - tests/unit/test_eval_matrix.py
  - tests/unit/test_llm_factory.py
  - tests/unit/test_probe_provider_capture.py
findings:
  critical: 0
  warning: 12
  info: 6
  total: 18
status: issues_found
---

# Phase 10: Code Review Report (Re-Review after Gap Closure 10-07..10-09)

**Reviewed:** 2026-06-10
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found

## Summary

This is a re-review after gap-closure plans 10-07..10-09 shipped fixes for the prior
review's five Critical findings. **All five Critical fixes are verified sound:**

- **CR-01 (gate checker schema) — FIXED.** `_check_gate` now walks the real
  `summary['scenarios'][*]['providers'][family]` shape (`scripts/check_eval_gates.py:161-179`),
  skips `baseline_eligible: false` scenarios, and reports NOT-EVALUABLE on absent
  cells. The test suite was rewritten to the nested shape and gained a true integration
  test (`test_integration_real_aggregate_output_fires_hard_gate`) that feeds the checker
  actual `aggregate_cell_jsons()` output and asserts exit 1 — exactly the test class
  whose absence let CR-01 ship. Verified by direct execution.
- **CR-02 (`_constraints_for_case` crash) — FIXED.** The YAML fallback at
  `scripts/eval_agent.py:649-653` now guards `case.expected_results is not None`.
  Regression tests cover the specific clarification case and a loop over every
  checked-in case (`TestConstraintsForCaseClarificationGuard`).
- **CR-03 (quarantine flag never wired) — FIXED, with one residual gap (WR-01 below).**
  `main()` now loads the eval-queries config and threads it into `aggregate_cell_jsons`
  (`scripts/eval_matrix.py:795-809`), with an `(OSError, ValueError)` fallback. New tests
  invoke `main()` end-to-end and assert `baseline_eligible` appears in the written
  summary.json for both quarantined and eligible scenarios.
- **CR-04 (hardcoded absolute path) — FIXED.** `tests/unit/test_probe_provider_capture.py:19`
  derives `REPO_ROOT` from `Path(__file__).resolve().parents[2]`; the subprocess smoke
  test is now portable.
- **CR-05 (probe redaction not fail-closed) — FIXED.** `response_metadata`,
  `usage_metadata`, and `tool_calls` are all routed through
  `json.dumps → _redact → json.loads` (structure-preserving redaction,
  `scripts/probe_provider_capture.py:225-235`), and the post-write guard is now the
  extracted production helper `_scan_fixture_for_secrets` which checks BOTH the regex
  patterns AND the runtime values of `_SECRET_ENV_VARS`. A new test plants a
  non-regex-shaped rotated key and proves the env-var channel catches it.

No new Critical issues were introduced. However, one residual gap exists inside the
CR-03 fix itself (WR-01), the CR-01 fix introduces an order-dependent cell-selection
ambiguity (WR-02), and nine of the prior review's Warnings/Info items remain open —
the gap-closure plans scoped only the five CRs. All carry-overs were re-verified by
execution against the current code, not assumed.

## Warnings

### WR-01: CR-03 fallback does not catch malformed YAML — `yaml.YAMLError` escapes and kills summary.json writing

**File:** `scripts/eval_matrix.py:795-804`
**Issue:** The CR-03 comment states "A missing or malformed file must NOT block summary
writing," but the `except (OSError, ValueError)` clause only covers missing files
(`FileNotFoundError`) and Pydantic validation failures (`ValidationError ⊂ ValueError`).
Syntactically malformed YAML raises `yaml.parser.ParserError` (a `yaml.YAMLError`,
which subclasses neither `OSError` nor `ValueError` — verified by execution). The
exception then propagates out of `main()` AFTER `run_matrix` has completed, so the
`failures` list and all cell aggregation are lost and `summary.json` is never written —
the operator gets a raw traceback instead of the partial-results report the comment
promises. The new test only covers the missing-file path
(`test_main_aggregation_survives_missing_eval_queries_file`), so the malformed-YAML
branch is untested and broken.
**Fix:**
```python
import yaml
...
except (OSError, ValueError, yaml.YAMLError) as _exc:
```
(or mirror `check_eval_gates._load_gates`, which wraps the parse in
`except Exception → ValueError`). Add a test that writes `hand_written: [unclosed`
to the `--eval-queries` path and asserts summary.json is still written.

### WR-02: Gate cell lookup is first-match-wins across scenarios — same data can yield opposite verdicts depending on key order

**File:** `scripts/check_eval_gates.py:166-173`
**Issue:** When a family has cells under more than one eligible scenario, `_check_gate`
takes the first scenario (dict iteration order — i.e., alphabetical, since summary.json
is written with `sort_keys=True`) and `break`s. Verified by execution: a summary where
the family passes in scenario `a_first` and fails in `b_second` returns `"pass"`, while
the same cells with the failing scenario first returns `"violation"`. Today's configs
have at most one eligible scenario per summary, so no live verdict is wrong yet — but
Phase 11 wires `committed_itinerary_rate` into all scenarios, at which point a multi-
scenario summary makes the hard gate's verdict depend on scenario-id spelling. A merge
gate must not have order-dependent semantics.
**Fix:** Evaluate the gate against EVERY eligible scenario that carries the family and
return `"violation"` if any fails (fail-closed), or add an explicit `scenario` field to
the gate entry schema and look up exactly that cell.

### WR-03: Unknown gate status silently converts a failing hard gate into a pass (carry-over, verified still present)

**File:** `scripts/check_eval_gates.py:199-206`
**Issue:** When a gate fails and its status is not in `_HARD_STATUSES` or
`"aspirational"`, `_check_gate` returns `"pass"` with no output. Verified by execution:
`status: activ` (typo) with a cell at 0.0 against a `>= 0.8` gate exits 0 and prints
"OK — 1 gates checked". A one-character typo disables a hard gate with zero diagnostics
— fail-open in the tool whose entire purpose is fail-closed enforcement. There is no
status-vocabulary validation at load time.
**Fix:** Validate `status` against
`_HARD_STATUSES | _SKIP_STATUSES | {"aspirational"}` in `_load_gates` and raise
`ValueError` (→ exit 2) on anything else.

### WR-04: Malformed gates YAML or summary values crash the checker with exit 1 — misreported as "hard-gate violation" (carry-over, verified)

**File:** `scripts/check_eval_gates.py:66-67, 99-103, 156-159, 250`
**Issue:** Verified by execution: `gates:` with a null value raises `TypeError` (iterating
`None`); a gate entry missing `family` raises `KeyError` — both escape `main()` and the
process exits 1, which the documented exit-code contract (and any Makefile/CI caller)
reads as "hard-gate violation," not "infrastructure failure" (exit 2). Same class:
`hard` missing `metric`/`op`/`value` (KeyError), unknown `op` (`_evaluate_op` ValueError),
`"median": null` in summary (`float(None)` TypeError at line 102), and a non-dict
scenario block (AttributeError at line 168).
**Fix:** Validate `isinstance(data["gates"], list)` in `_load_gates`, and wrap the
gate-iteration loop in `main()` with
`except (KeyError, TypeError, ValueError, AttributeError) → stderr + return 2`.

### WR-05: `advisory` gate entries are never evaluated — dead config with an unresolvable metric name (carry-over)

**File:** `scripts/check_eval_gates.py` (whole file), `configs/eval_gates.yaml:29-32`, `docs/eval_gates.md:10-21`
**Issue:** The YAML schema comment and gates doc say advisory entries are "reported but
never blocking." The checker never reads the `advisory` key — the gpt-4o-mini advisory
on `refinement_minimal_edit_median` is dead config. The metric name would also not
resolve via `_get_metric_value` even if implemented (the summary stores
`refinement_minimal_edit` with a `median` subkey, not `refinement_minimal_edit_median`).
**Fix:** Implement advisory evaluation (print `ADVISORY miss` lines, never affect exit
code) and rename the YAML metric to `refinement_minimal_edit`, or delete the `advisory`
blocks and the doc claim until implemented.

### WR-06: Single-turn eval path still has no D-10-01 error capture — one transient failure aborts the whole run (carry-over)

**File:** `scripts/eval_agent.py:672-687`
**Issue:** `evaluate_case`'s single-turn branch wraps `graph.ainvoke` in `try/finally`
only — no `except` → no `make_error_record`. Both multi-turn branches honor the D-10-01
ERROR-record contract, but a 429/DB-down on any single-turn case still propagates
through `evaluate_cases → build_report → main` and aborts the entire multi-case run.
The `"setup"` stage documented in `make_error_record` is still never used anywhere. The
`evaluate_multi_turn_case` docstring's "Fail-open semantics mirror `evaluate_cases` in
BOTH branches" claim remains stale (the helper now returns error records, and the
single-turn path doesn't fail open at all — it crashes).
**Fix:** Wrap the single-turn `graph.ainvoke` in the same
`except Exception → make_error_record(case, "turn0", exc)` pattern; use stage `"setup"`
for pre-invocation failures (e.g., constraint building). Update the stale docstring.

### WR-07: eval_agent exit code still conflates model-behavior violations with infra failures (carry-over)

**File:** `scripts/eval_agent.py:1201`, `scripts/eval_matrix.py:521-528`
**Issue:** `main` returns 1 when `report_has_violations(report)` — i.e., when the model
merely scored below a threshold (normal: refinement medians sit at 0.0). The matrix
runner records ANY nonzero subprocess returncode as a cell failure (`failures` list,
matrix rc=1), indistinguishable from a crash. Every live matrix run with a
sub-threshold score therefore exits 1 and reports "N cell(s) failed" — the
infra-vs-model conflation Phase 10 exists to eliminate.
**Fix:** Make `eval_agent.main` return distinct exit codes (0 ok, 1 violations,
2 errors) and have `run_matrix` classify rc==1 as "violation (expected)" vs rc>=2 as
"failure"; or have `run_matrix` read the written cell JSON's `n_errored`/`cell_valid`
to classify.

### WR-08: Prod-threading scratch keys still pollute tool-call metrics (carry-over)

**File:** `scripts/eval_agent.py:394-405` (vs `:902-944`)
**Issue:** `count_tool_calls` sums `len(entries)` for every list-valued `state.scratch`
entry, and `tool_names_from_state` reports every non-empty list key as a tool name.
`_run_prod_threading` stamps `prior_committed_stops` (list of dicts) and
`prior_stops_obj` (list of Stops) into `state.scratch` on every turn. A 3-stop prod
refinement run reports ~6 phantom tool calls and lists those scratch keys as "tools" in
`tool_names`, skewing `tool_calls_mean` for exactly the prod-shaped cells this phase
wants measured honestly.
**Fix:** Exclude the known Phase-6/7 scratch keys in both helpers:
```python
_NON_TOOL_SCRATCH_KEYS = {"prior_committed_stops", "prior_stops_obj"}
```

### WR-09: All-errored cell still reports `deterministic_pass_rate: 1.0` and `tool_success_rate: 1.0` (carry-over, verified)

**File:** `scripts/eval_agent.py:1063, 1072` (with `rate()` at `:984`)
**Issue:** Verified by execution: with `n_scored == 0` (every run errored),
`1.0 - rate(x, 0)` yields **1.0** — an all-failure cell publishes perfect pass/success
headline rates. `cell_valid: false` is present, but a human or Phase 11 tooling scanning
the rates reads "100% pass" on a cell where nothing was measured — the same fail-open
shape D-10-04 killed for scorer means.
**Fix:** Emit `None` (or 0.0) for the derived rates when `n_scored == 0`, mirroring for
`deterministic_violation_rate`, `tool_error_rate`, and `tool_success_rate`.

### WR-10: `additional_kwargs_values` still stringified — the adapter fixture test's primary input remains vacuous (carry-over, partially addressed)

**File:** `scripts/probe_provider_capture.py:219-220`; `tests/unit/test_adapters.py:930-949`
**Issue:** The CR-05 fix made `response_metadata`/`usage_metadata`/`tool_calls`
structure-preserving, but `additional_kwargs_values` — the ONE field
`test_adapter_capture_on_real_wire_fixture` reconstructs into `AIMessage.additional_kwargs`
and the field every adapter actually reads — still goes through `_redact(value)`, whose
first line is `str(value)`. Bytes become `"b'\\x00...'"` and the Gemini
`__gemini_function_call_thought_signatures__` dict becomes a repr string, so every
adapter's `isinstance(bytes)`/`isinstance(dict)` guard short-circuits to `None` and the
"must not crash against the real wire shape" assertion passes without exercising the
real-typed capture path. D-10-12's live-shape gap closure is still mostly undelivered.
**Fix:** Apply the same recursive structure-preserving redaction to
`additional_kwargs_values` (bytes → `{"__bytes_b64__": ...}` marker), and have the test
decode markers back to bytes when reconstructing the message.

### WR-11: Structural-check "Check 6" is still a tautology (carry-over)

**File:** `scripts/eval_matrix.py:726-750`
**Issue:** Check 6 builds a synthetic literal dict
`{"status": "error", "error": {"stage": "turn0", ...}}` and asserts the literal contains
the keys it was just given. It cannot fail and validates nothing about the real
`make_error_record` schema, while claiming to be a T-10-02-01 CI hard-gate mitigation —
false confidence in a hard gate.
**Fix:** Exercise the real schema: call `scripts.eval_agent.make_error_record` on a
probe case and assert `rec.status == "error"` and
`rec.error["stage"] in {"setup", "turn0", "turnN"}`.

### WR-12: `category_compliance` returns 1.0 on zero committed stops — code contradicts its own docstring's stated design

**File:** `app/agent/critique/checks.py:253-255` (and `:280-281` for the strict variant)
**Issue:** The docstring explicitly documents that the abstain-on-length-mismatch
alternative was "rejected because it would let an agent commit zero stops or extra
stops and still score 1.0, defeating the scorer's purpose" — yet line 254
(`if not state.stops: return 1.0`) implements exactly that outcome for the zero-stop
case. Verified by execution: with 3 requested category slots and zero committed stops,
both `category_compliance` and `category_compliance_strict` return 1.0. A model that
commits nothing scores perfect category compliance, inflating the scorer mean for
exactly the non-committing failure mode (DeepSeek decisiveness gap) this harness is
supposed to measure honestly. Without the early return, the existing denominator math
(`max(len(requested), len(stops))`) would naturally yield 0.0.
**Fix:** Either remove the `if not state.stops: return 1.0` early return (the denominator
already handles it; confirm baseline impact since this changes scorer semantics) or
rewrite the docstring to document the zero-stop fail-open honestly and note that
`stop_count_satisfied` / `expected_results` carry that signal instead. The code and the
docstring cannot both stand as written.

## Info

### IN-01: `make_error_record` discards case metadata (carry-over)

**File:** `scripts/eval_agent.py:190, 202`
**Issue:** `expects_clarification_or_relaxation` is hardcoded `False` instead of
`case.expects_clarification_or_relaxation`, and every check's `threshold` is written as
`0.0` instead of `CRITIQUE_THRESHOLDS[name]` — error records misreport expected-behavior
metadata in the audit trail.
**Fix:** Use the case's real flag and `CheckResult(score=None, threshold=CRITIQUE_THRESHOLDS[name], passed=False)`.

### IN-02: Redundant `sk-ant-` pattern (carry-over)

**File:** `scripts/probe_provider_capture.py:65-66`
**Issue:** `sk-[A-Za-z0-9_-]{20,}` already matches `sk-ant-...` (the char class includes
`-`), so the dedicated Anthropic pattern is dead. Harmless.
**Fix:** Add a comment noting it is subsumed (keep for documentation value) or drop it.

### IN-03: Corrupt/unreadable cell JSON silently skipped during aggregation (carry-over)

**File:** `scripts/eval_matrix.py:283-286`
**Issue:** `except (OSError, json.JSONDecodeError): continue` drops the cell from both
scorers and `n_errored` accounting with no log line — inconsistent with the WR-01
warning emitted for unparseable filenames; a quiet undercount path in a harness-honesty
phase.
**Fix:** Log a warning naming the file (mirror the filename-parse warning).

### IN-04: Error records always report `latency_seconds = 0.0` (carry-over)

**File:** `scripts/eval_agent.py:170-227, 771-773, 890-895`
**Issue:** Both threading branches accumulate `total_latency` up to the failing turn,
then call `make_error_record`, which hardcodes `latency_seconds=0.0` — the time burned
before the failure (useful for diagnosing timeout-class errors) is dropped.
**Fix:** Add a `latency_seconds: float = 0.0` parameter to `make_error_record` and pass
`total_latency`.

### IN-05: `make_judge` key map omits `anthropic` despite it being a supported provider (carry-over)

**File:** `app/agent/critique/vibe.py:112-117`
**Issue:** The provider→key-attr map covers openai/gemini/deepseek/kimi only. Setting
`EVAL_JUDGE_PROVIDER=anthropic` (a member of `SUPPORTED_PROVIDERS` since Phase 9) logs
"unknown vibe judge provider" and disables the judge — graceful, but misleading and
undocumented.
**Fix:** Add `"anthropic": "anthropic_api_key"` to the map, or document the exclusion.

### IN-06: Secrets containing `"` or `\` evade both the env-var redaction and the post-write guard

**File:** `scripts/probe_provider_capture.py:93-97, 119-122, 226-235`
**Issue:** The redaction and the guard both operate on JSON-dumped text, where `"` and
`\` in a secret value appear JSON-escaped — so a raw `os.environ` secret containing
either character never matches `s.replace(secret_val, ...)` in `_redact` nor
`secret_val in text` in `_scan_fixture_for_secrets`. Real API keys are URL-safe and do
not contain these characters today, so practical risk is low, but the fail-closed claim
has this boundary.
**Fix:** Compare against `json.dumps(secret_val)[1:-1]` (the escaped form) in addition
to the raw value in both helpers, or note the limitation in the module docstring.

---

_Reviewed: 2026-06-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Re-review of gap-closure plans 10-07..10-09 (CR-01..CR-05)_
