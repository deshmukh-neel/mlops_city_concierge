---
phase: 10-eval-harness-honesty
reviewed: 2026-06-10T00:00:00Z
depth: standard
files_reviewed: 20
files_reviewed_list:
  - Makefile
  - app/agent/critique/checks.py
  - app/agent/critique/vibe.py
  - app/eval/config.py
  - configs/eval_baselines/late_night_closure_cascade.json
  - configs/eval_gates.yaml
  - configs/eval_matrix.yaml
  - configs/eval_matrix_refinement.yaml
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
  critical: 5
  warning: 9
  info: 5
  total: 19
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-06-10
**Depth:** standard
**Files Reviewed:** 20
**Status:** issues_found

## Summary

Phase 10's stated contract is harness honesty: ERROR-status records excluded from
aggregation, late_night quarantine honored, satisfiable gates enforced via
`eval_gates.yaml` + `check_eval_gates.py`, and fail-closed API-key redaction in the
provider probe. The error-record plumbing (D-10-01..04) is well built and well tested.
However, three of the phase's headline deliverables do not actually work end-to-end:

1. The gate checker reads a **summary.json schema that the matrix runner never
   produces** — every gate is permanently NOT-EVALUABLE and the checker exits 0
   (verified empirically). The unit tests bake in the same wrong schema, so CI is green.
2. The **D-10-09 quarantine flag is inert in the real pipeline** — `main()` never passes
   `eval_queries_config` to `aggregate_cell_jsons`, so no real summary.json ever carries
   `baseline_eligible`.
3. The **probe's fail-closed redaction claim does not hold** — three of the written
   fixture fields bypass `_redact` entirely, and the post-write guard does not check
   env-var-sourced secret values.

Additionally, the full eval suite **crashes on 4 of 30 checked-in scenarios** (verified
by execution), and a hardcoded user-specific absolute path in a test guarantees CI
failure on any other machine.

## Critical Issues

### CR-01: check_eval_gates.py reads a summary.json shape eval_matrix.py never produces — all gates permanently NOT-EVALUABLE

**File:** `scripts/check_eval_gates.py:155` (vs `scripts/eval_matrix.py:376-384`)
**Issue:** `_check_gate` looks up cells via `summary.get("providers", {})` — a
**top-level** `providers` key. But `aggregate_cell_jsons` (the function that writes
`summary.json`, the checker's documented input per `docs/eval_gates.md` and the
Makefile target) produces `{"generated_at", "scenarios": {<scenario_id>: {"providers":
{...}}}}` — providers are nested **under each scenario**, never at top level.
Empirically verified: running `check_eval_gates.py` against a freshly generated
`summary.json` prints `NOT-EVALUABLE — family '<x>' has no cell in summary` for every
hard-gated family and exits 0. This means:
- The active gate on `openai/gpt-4o-mini` and the provisional-n1 gate on
  `anthropic/claude-sonnet-4-6` can **never fire**, even after Phase 11 wires
  `committed_itinerary_rate` into the scorers block — the checker will still find no cell.
- `make eval-gates-check` (EVAL-03 / D-10-05) is structurally a no-op against its
  documented input.

The unit tests in `tests/unit/test_check_eval_gates.py:84-99` (`_make_summary`) construct
summaries with a top-level `providers` key — the same wrong schema — so all tests pass
while the integration is broken. The only artifact with top-level `providers` is the
baseline JSON (`configs/eval_baselines/*.json`), which is not what the Makefile target
or docs feed the checker.
**Fix:**
```python
# In _check_gate, walk the real summary shape. A family may appear under
# multiple scenarios; restrict to baseline-eligible scenarios (or take an
# explicit scenario from the gate entry):
cell = None
for scenario_id, scenario_block in summary.get("scenarios", {}).items():
    if not scenario_block.get("baseline_eligible", True):
        continue
    candidate = scenario_block.get("providers", {}).get(family)
    if candidate is not None:
        cell = candidate
        break
```
Then rewrite `_make_summary` in the tests to produce the real
`{"scenarios": {...: {"providers": {...}}}}` shape, and add one integration test that
feeds the checker an actual `aggregate_cell_jsons(...)` output (that test would have
caught this).

### CR-02: `_constraints_for_case` crashes on every clarification case without an explicit stop count — full eval suite cannot run

**File:** `scripts/eval_agent.py:648-653`
**Issue:** When `explicit_num_stops_from_text(case.query)` returns `None`, the fallback
unconditionally dereferences `case.expected_results.min_stops`. `expected_results` is
`None` by design for `expects_clarification_or_relaxation: true` cases. Verified by
execution against the checked-in `configs/eval_queries.yaml`: **4 of 30 cases crash**
(`impossible_four_am_five_star`, `impossible_cheap_michelin`,
`impossible_north_beach_sushi_4am`, `closed_monday_brunch`) with
`AttributeError: 'NoneType' object has no attribute 'min_stops'`.

In the single-turn path this exception escapes `evaluate_case` (the `try` at line 674
has only `finally`, no `except`), propagates through `evaluate_cases` → `build_report`
→ `main`, and **aborts the entire run** ("eval_agent failed: ..."), losing all other
cases' results. Running `python scripts/eval_agent.py` over the default full suite (the
documented default: "run all hand_written cases") is currently impossible. The matrix
runner dodges it only because it always passes `--scenario-ids` + `--max-queries 1`
scoped to cases that have `expected_results`.
**Fix:**
```python
if num_stops is None and case.expected_results is not None:
    min_s = case.expected_results.min_stops
    max_s = case.expected_results.max_stops
    if min_s == max_s:
        num_stops = min_s
```
Add a regression test that runs `_constraints_for_case` over every case in the
checked-in YAML (the 5-line loop used to verify this finding).

### CR-03: D-10-09 quarantine flag never reaches real summary.json — `main()` omits `eval_queries_config`

**File:** `scripts/eval_matrix.py:791`
**Issue:** `aggregate_cell_jsons` only emits `baseline_eligible` per scenario when its
`eval_queries_config` parameter is provided. The single production callsite —
`main()` — calls `aggregate_cell_jsons(output_dir, llm_provider_override=...)` with no
config, so **every real summary.json omits `baseline_eligible` entirely**. The
quarantine of `late_night_closure_cascade` (D-10-09, EVAL-02) is therefore inert in the
actual pipeline; Phase 11 baseline tooling reading summary.json will see no flag and
default the scenario to eligible. The comments in `configs/eval_matrix.yaml:35-36` and
`configs/eval_queries.yaml` claim the flag "is honored by aggregate_cell_jsons" — the
capability exists but is never wired. The unit tests
(`test_aggregate_cell_jsons_surfaces_baseline_ineligible_in_scenario_block`) pass the
config explicitly, so they pass while `main()` does not.
**Fix:**
```python
from app.eval.config import load_eval_queries
...
summary = aggregate_cell_jsons(
    output_dir,
    llm_provider_override=args.llm_provider_override,
    eval_queries_config=load_eval_queries(args.eval_queries),
)
```
(Wrap the load in try/except if a missing queries file should not block summary
writing.) Add a test that invokes `main()` (with mocked `subprocess.run`) and asserts
`baseline_eligible` appears in the written summary.json.

### CR-04: Hardcoded user-specific absolute path in test — guaranteed failure on CI and every other machine

**File:** `tests/unit/test_probe_provider_capture.py:158`
**Issue:** `test_main_help_exits_zero` runs `subprocess.run(..., cwd="/Users/pnhek/usf
msds/msds-603-mlops/mlops_city_concierge", ...)`. On CI (Ubuntu) or any teammate's
machine that directory does not exist, so `subprocess.run` raises
`FileNotFoundError` and the unit suite goes red. This breaks `make test` everywhere
except the author's laptop.
**Fix:**
```python
REPO_ROOT = Path(__file__).resolve().parents[2]
...
result = subprocess.run(
    [sys.executable, str(REPO_ROOT / "scripts" / "probe_provider_capture.py"), "--help"],
    cwd=str(REPO_ROOT),
    capture_output=True,
    text=True,
)
```

### CR-05: Probe redaction is not fail-closed — three fixture fields bypass `_redact`, and the post-write guard skips env-var secret values

**File:** `scripts/probe_provider_capture.py:196-233`
**Issue:** The phase's security mandate is "API-key redaction must be fail-closed."
Two gaps break that:

1. **Unredacted write paths.** Only `additional_kwargs_values` go through `_redact`
   (line 195). `response_metadata` (line 196), `usage_metadata` (line 198), and
   `tool_calls` (line 199) are written raw. `_sanitize_response_metadata` blanks only
   two fixed keys (`system_fingerprint`, `id`) and does not recurse into nested dicts.
   `tool_calls` args are model-generated content — exactly the "model could echo one
   back" scenario the `_redact` docstring warns about — yet they never pass through
   redaction.
2. **Post-write guard omits T-10-05-02.** The guard (lines 224-233) scans only the four
   `_SECRET_PATTERNS` regexes. The env-var-sourced secret check (the `_SECRET_ENV_VARS`
   substitution that D-10-13 added precisely because "the actual key value doesn't
   match a regex but is still a secret") is **absent from the post-write scan**. A
   non-pattern-shaped key (e.g., a `GOOGLE_API_KEY` not in `AIzaSy` form, a rotated
   DeepSeek key format) appearing in `response_metadata` or `tool_calls` would be
   written to a checked-in fixture and survive the guard. The module docstring's claim
   ("refuses to keep it if any secret pattern is found — fail-closed") is only true for
   regex-shaped secrets.

The unit tests reimplement the guard logic inline (`any(p.search(text) ...)`) instead of
invoking `main`'s actual guard, so neither gap is test-visible.
**Fix:**
```python
# 1. Route every value-bearing field through _redact before writing:
fixture = {
    ...
    "response_metadata": json.loads(_redact(json.dumps(response_metadata, default=str))),
    "usage_metadata": _redact(usage_metadata) if usage_metadata is not None else None,
    "tool_calls": json.loads(_redact(json.dumps(tool_calls, default=str))),
}
# 2. Extend the post-write guard with the env-var check:
for env_var in _SECRET_ENV_VARS:
    secret_val = os.environ.get(env_var, "")
    if secret_val and len(secret_val) >= 10 and secret_val in text:
        fixture_file.unlink(missing_ok=True)
        ...
        return 2
```

## Warnings

### WR-01: Single-turn eval path has no D-10-01 error capture — one transient failure aborts the whole run

**File:** `scripts/eval_agent.py:674-687`
**Issue:** `evaluate_case`'s single-turn branch wraps `graph.ainvoke` in `try/finally`
only — no `except` → no `make_error_record`. Both multi-turn branches were converted to
the D-10-01 ERROR-record contract, but a 429/DB-down on any single-turn case still
propagates and kills the entire multi-case run (and CR-02's crash rides this same
path). The `evaluate_multi_turn_case` docstring still claims "Fail-open semantics
mirror `evaluate_cases` in BOTH branches" — stale. The `"setup"` stage documented in
`make_error_record` is never used anywhere.
**Fix:** Wrap the single-turn `graph.ainvoke` in the same `except Exception →
make_error_record(case, "turn0", exc)` pattern (and use stage `"setup"` for failures
before invocation, e.g. constraint building). Update the stale docstring.

### WR-02: eval_agent exit code conflates model-behavior violations with infra failures — matrix "failures" list polluted

**File:** `scripts/eval_agent.py:1201`, `scripts/eval_matrix.py:520-527`
**Issue:** `main` returns 1 when `report_has_violations(report)` — i.e., when the model
merely scored below a threshold (normal: refinement medians sit at 0.0). The matrix
runner records **any** nonzero subprocess returncode as a cell failure (`failures` list,
matrix rc=1), with empty stderr, indistinguishable from a crash. So every live matrix
run with a sub-threshold score exits 1 and reports "N cell(s) failed" — exactly the
infra-vs-model conflation Phase 10 exists to eliminate (and it undermines T-10-02-02's
"distinct stderr lines" intent, since the violation-driven rc and the error-driven rc
collapse into the same `failures` channel).
**Fix:** Either (a) make `eval_agent.main` return distinct exit codes (0 ok, 1
violations, 2 errors) and have `run_matrix` classify rc==1 as "violation (expected)"
vs rc>=2 as "failure", or (b) have `run_matrix` read the written cell JSON's
`n_errored`/`cell_valid` to classify, recording violations separately from failures.

### WR-03: check_eval_gates fail-open on unknown gate status — a typo silently disables a hard gate

**File:** `scripts/check_eval_gates.py:183-189`
**Issue:** When a gate fails and its status is not in `_HARD_STATUSES` or
`"aspirational"`, `_check_gate` returns `"pass"` with no output. A typo'd status
(`activ`, `Active`, trailing whitespace) converts a hard gate into a silent pass —
fail-open in the one tool whose job is fail-closed enforcement. There is no
status-vocabulary validation at load time either.
**Fix:** Validate `status` against the known set when loading gates; treat unknown
statuses as an infrastructure failure (exit 2) or at minimum print a loud
`UNKNOWN-STATUS` line and count it as a violation.

### WR-04: `advisory` gate entries are never evaluated or reported — dead config contradicting docs

**File:** `scripts/check_eval_gates.py` (whole file), `configs/eval_gates.yaml:29-32`, `docs/eval_gates.md:10-21`
**Issue:** The YAML schema comment and docs say `advisory: list of {metric, op, value}
— reported but never blocking`. The checker never reads the `advisory` key at all — the
gpt-4o-mini advisory on `refinement_minimal_edit_median` is dead config. Additionally,
the advisory metric name `refinement_minimal_edit_median` would not resolve through
`_get_metric_value` even if implemented (summary stores the metric as
`refinement_minimal_edit` with a `median` subkey).
**Fix:** Implement advisory evaluation (print `ADVISORY miss` lines, never affect exit
code) and change the YAML metric name to `refinement_minimal_edit`, or delete the
`advisory` blocks and the doc claim until implemented.

### WR-05: Malformed gates YAML or summary values crash the checker with exit 1 — misreported as "hard-gate violation"

**File:** `scripts/check_eval_gates.py:149-152, 92-97, 233`
**Issue:** The documented exit-code contract is 2 = infra failure. But several malformed
inputs raise uncaught exceptions, and an uncaught Python exception exits with code 1 —
which the contract (and the Makefile caller) reads as "hard-gate violation":
`gate["family"]` / `hard["metric"]` / `hard["op"]` / `hard["value"]` KeyError on a
malformed entry; `gates: null` → `TypeError` iterating `None` (the loader checks only
`"gates" in data`, not that it's a list); `float(metric_block["median"])` TypeError when
`median` is `null` in the JSON.
**Fix:** Wrap the gate-iteration loop in `try/except (KeyError, TypeError, ValueError)`
→ stderr + `return 2`, and validate `isinstance(data["gates"], list)` in `_load_gates`.

### WR-06: Prod-threading scratch keys pollute tool-call metrics

**File:** `scripts/eval_agent.py:394-405` (vs `:903-944`)
**Issue:** `count_tool_calls` sums `len(entries)` for **every** list-valued
`state.scratch` entry, and `tool_names_from_state` reports every non-empty list key as a
tool name. `_run_prod_threading` stamps `prior_committed_stops` (list of dicts) and
`prior_stops_obj` (list of Stops) into `state.scratch` on every turn. A 3-stop prod
refinement run therefore reports ~6 phantom tool calls and lists
`prior_committed_stops` / `prior_stops_obj` as "tools" in `tool_names`, skewing
`tool_calls_mean` for exactly the prod-shaped cells the phase wants to measure honestly.
**Fix:** Exclude the known Phase-6/7 scratch keys in both helpers, e.g.:
```python
_NON_TOOL_SCRATCH_KEYS = {"prior_committed_stops", "prior_stops_obj"}
... if isinstance(entries, list) and tool_name not in _NON_TOOL_SCRATCH_KEYS
```

### WR-07: All-errored cell reports `deterministic_pass_rate: 1.0` and `tool_success_rate: 1.0`

**File:** `scripts/eval_agent.py:1063, 1072` (with `rate()` at `:984`)
**Issue:** With `n_scored == 0` (every run errored), `rate(x, 0)` returns 0.0 so
`1.0 - rate(...)` yields **1.0** — an all-failure cell publishes perfect pass/success
headline rates in its aggregate. `cell_valid: false` is present, but a human or a
future tool scanning the rates reads "100% pass" on a cell where nothing was measured.
This is the same fail-open shape D-10-04 was written to kill for scorer means.
**Fix:** Emit `None` (or 0.0) for the derived rates when `n_scored == 0`:
```python
"deterministic_pass_rate": None if n_scored == 0 else 1.0 - rate(queries_with_violations, n_scored),
```
(Mirror for `deterministic_violation_rate`, `tool_error_rate`, `tool_success_rate`.)

### WR-08: Fixture pipeline stringifies all wire values — adapter fixture tests pass vacuously

**File:** `scripts/probe_provider_capture.py:79-98, 195`; `tests/unit/test_adapters.py:912-954`
**Issue:** `_redact` begins with `s = str(value)`, so every `additional_kwargs` value in
the fixture is a Python-repr **string** — bytes become `"b'\\x00...'"`, the Gemini
`__gemini_function_call_thought_signatures__` dict becomes `"{'tc_1': '...'}"`. The
EVAL-05 test `test_adapter_capture_on_real_wire_fixture` reconstructs an `AIMessage`
from those stringified values; every adapter's `isinstance(bytes)`/`isinstance(dict)`
guard then short-circuits to `None`, and the "must not crash against the real wire
shape" assertion passes without ever exercising the real-typed capture path. D-10-12's
claim of closing the live-shape gap (the D-09-09 lcgg key miss class) is mostly not
delivered: a wire-shape change that crashes on real types would not be caught.
**Fix:** Preserve JSON-representable structure in the fixture — redact recursively
instead of stringifying:
```python
def _redact_obj(value):
    if isinstance(value, str): return _redact_str(value)
    if isinstance(value, bytes): return {"__bytes_b64__": base64.b64encode(value).decode()}
    if isinstance(value, dict): return {k: _redact_obj(v) for k, v in value.items()}
    if isinstance(value, list): return [_redact_obj(v) for v in value]
    return value
```
and have the test decode `__bytes_b64__` markers back to bytes when reconstructing the
message.

### WR-09: Structural-check "Check 6" is a tautology — asserts hardcoded literals about hardcoded literals

**File:** `scripts/eval_matrix.py:731-749`
**Issue:** Check 6 builds a synthetic dict
`{"status": "error", "error": {"stage": "turn0", ...}}` in-line and then checks that
this literal contains the keys it was just given and that its literal stage is in a
literal set. It cannot fail and validates nothing about the real code; the comment
claims it "ensures CI catches schema drift before any live run is attempted
(T-10-02-01 mitigation)" — false confidence in a CI **hard gate**. The companion test
`test_structural_check_error_schema_check_is_present_in_source` only greps the source
for the literal, compounding the illusion.
**Fix:** Exercise the real schema:
```python
from scripts.eval_agent import make_error_record
from app.eval.config import EvalQuery
probe_case = EvalQuery.model_validate({"id": "x", "query": "q", "reference": "r",
                                       "expected_results": {"min_stops": 1, "max_stops": 1}})
rec = make_error_record(probe_case, "turn0", RuntimeError("probe"))
if rec.status != "error" or rec.error.get("stage") not in {"setup", "turn0", "turnN"}:
    ... return 1
```

## Info

### IN-01: `make_error_record` discards case metadata

**File:** `scripts/eval_agent.py:190, 202`
**Issue:** `expects_clarification_or_relaxation` is hardcoded `False` instead of
`case.expects_clarification_or_relaxation`, and every check's `threshold` is written as
`0.0` instead of `CRITIQUE_THRESHOLDS[name]` — error records misreport expected-behavior
metadata in the audit trail.
**Fix:** Use `case.expects_clarification_or_relaxation` and
`CheckResult(score=None, threshold=CRITIQUE_THRESHOLDS[name], passed=False)`.

### IN-02: Redundant `sk-ant-` pattern

**File:** `scripts/probe_provider_capture.py:62-63`
**Issue:** `sk-[A-Za-z0-9_-]{20,}` already matches `sk-ant-...` (the char class includes
`-`), so the dedicated Anthropic pattern is dead. Harmless, but it overstates coverage
when reading the list.
**Fix:** Drop it or add a comment noting it is subsumed (keep for documentation value).

### IN-03: Corrupt/unreadable cell JSON silently skipped during aggregation

**File:** `scripts/eval_matrix.py:282-285`
**Issue:** `except (OSError, json.JSONDecodeError): continue` drops the cell from both
scorers and the `n_errored` accounting with no log line — inconsistent with the WR-01
warning emitted for unparseable filenames, and a quiet undercount path for a
harness-honesty phase.
**Fix:** Log a warning naming the file (mirror the filename-parse warning).

### IN-04: Error records always report `latency_seconds = 0.0`

**File:** `scripts/eval_agent.py:170-227, 771-773, 890-895`
**Issue:** Both threading branches accumulate `total_latency` up to the failing turn,
then call `make_error_record`, which hardcodes `latency_seconds=0.0` — the time burned
before the failure (useful for diagnosing timeout-class errors) is dropped.
**Fix:** Add a `latency_seconds: float = 0.0` parameter to `make_error_record` and pass
`total_latency`.

### IN-05: `make_judge` key map omits `anthropic` despite it being a supported provider

**File:** `app/agent/critique/vibe.py:112-117`
**Issue:** The provider→key-attr map covers openai/gemini/deepseek/kimi only. Setting
`EVAL_JUDGE_PROVIDER=anthropic` (a member of `SUPPORTED_PROVIDERS` since Phase 9) logs
"unknown vibe judge provider" and disables the judge. Graceful degradation, but the
message is misleading and the gap is undocumented.
**Fix:** Add `"anthropic": "anthropic_api_key"` to the map, or document the exclusion.

---

_Reviewed: 2026-06-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
