---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 07
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/check_baselines_fresh.py
  - tests/unit/test_check_baselines_fresh.py
  - scripts/probe_provider_capture.py
  - tests/unit/test_adapters.py
autonomous: true
requirements: [BASE-04]
must_haves:
  truths:
    - "a change to app/llm_factory.py without a baseline refresh fails the staleness lint (exit 1)"
    - "a change to configs/eval_matrix.yaml or configs/eval_matrix_refinement.yaml without a baseline refresh fails the staleness lint (exit 1)"
    - "a change under app/agent/ still fails the staleness lint without a refresh (no regression)"
    - "probe fixtures preserve real-typed additional_kwargs values (bytes/dict) through redaction so adapter fixture tests exercise real-wire types, not stringified copies"
  artifacts:
    - path: "scripts/check_baselines_fresh.py"
      provides: "Watch-set extended to app/llm_factory.py + configs/eval_matrix* (D-11-21)"
    - path: "tests/unit/test_check_baselines_fresh.py"
      provides: "Dry-run staleness tests for llm_factory + eval_matrix changes"
    - path: "scripts/probe_provider_capture.py"
      provides: "WR-10 type-fidelity fix for additional_kwargs_values redaction"
    - path: "tests/unit/test_adapters.py"
      provides: "Fixture-loading test exercising real-typed reconstruction"
  key_links:
    - from: "scripts/check_baselines_fresh.py _agent_changed"
      to: "WATCH_PREFIXES (app/agent/, app/llm_factory.py, configs/eval_matrix)"
      via: "any-prefix membership"
      pattern: "WATCH_PREFIXES"
    - from: "scripts/probe_provider_capture.py additional_kwargs_values"
      to: "tests/unit/test_adapters.py fixture reconstruction"
      via: "real-typed redacted values, not str()"
      pattern: "additional_kwargs_values"
---

<objective>
Two independent, non-blocking fixes (both no live calls). (1) BASE-04 / D-11-21: extend the `check_baselines_fresh.py` staleness watch-set beyond `app/agent/` to also cover `app/llm_factory.py` (provider branches, thinking policies, temperature clamps) and `configs/eval_matrix*.yaml` — files that directly change measured behavior but are unwatched today — with dry-run tests asserting exit 1. (2) WR-10: the probe stringifies `additional_kwargs` values via `_redact`, so adapter fixture tests never see real-typed bytes/dict payloads; fix the redaction to preserve type fidelity.

Purpose: BASE-04 requires the staleness gate to catch agent-loop changes that bypass `app/agent/` (the verified gap is `app/llm_factory.py`). WR-10 improves test fidelity so the live-probe fixtures actually exercise the real-wire types the adapters parse.
Output: Extended staleness watch-set + dry-run tests; type-faithful probe fixtures + an adapter test that exercises them.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Extend staleness watch-set to llm_factory + eval_matrix (BASE-04 / D-11-21)</name>
  <files>scripts/check_baselines_fresh.py, tests/unit/test_check_baselines_fresh.py</files>
  <read_first>
    - scripts/check_baselines_fresh.py — read the whole file. Key: `AGENT_PREFIX = "app/agent/"` (line 48), `_agent_changed` (lines 116-118, currently `p.startswith(AGENT_PREFIX)`), `_baselines_changed` (line 121), `main()` (line 215) and its branch logic (Branch 1 hard-fail at line 272 returns 1). Note the script is stdlib-only by design (so the `lint-baselines` CI job runs before installing deps) — do NOT add non-stdlib imports.
    - tests/unit/test_check_baselines_fresh.py — read the existing truth-table tests that monkeypatch `_run_git` (or `_changed_paths` / `_last_commit_message`) to drive the four branches; mirror that monkeypatch idiom for the new dry-run tests.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`scripts/check_baselines_fresh.py (MODIFY)` shows the `WATCH_PREFIXES` list and the updated `_agent_changed` body; §`tests/unit/test_check_baselines_fresh.py (MODIFY)` shows the two dry-run tests (`test_llm_factory_change_triggers_stale_gate`, `test_eval_matrix_yaml_change_triggers_stale_gate`).
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md — D-11-21 (the verified gap: only `app/agent/` is watched, but provider branches and temp clamps live in `app/llm_factory.py`; scorers under `app/agent/critique/` are already covered).
  </read_first>
  <behavior>
    - Test: a diff touching `app/llm_factory.py` with NO baseline refresh and no `[skip-baseline]` exits 1.
    - Test: a diff touching `configs/eval_matrix.yaml` with no baseline refresh exits 1.
    - Test: a diff touching `configs/eval_matrix_refinement.yaml` with no baseline refresh exits 1.
    - Test: a diff touching `app/agent/agent.py` (existing prefix) still exits 1 with no refresh (no regression).
    - Test: a diff touching `app/llm_factory.py` WITH a baseline refresh exits 0.
    - Test: an unrelated diff (e.g. `README.md` only) still exits 0.
  </behavior>
  <action>
    Replace the scalar `AGENT_PREFIX` with a `WATCH_PREFIXES` list: `["app/agent/", "app/llm_factory.py", "configs/eval_matrix"]` (the last is a bare prefix that matches BOTH `configs/eval_matrix.yaml` and `configs/eval_matrix_refinement.yaml`) per D-11-21. Update `_agent_changed` to return `sorted(p for p in paths if any(p.startswith(prefix) for prefix in WATCH_PREFIXES))`. Keep the function name `_agent_changed` and the `main()` caller unchanged (only the body changes) to minimize blast radius; update its docstring to say "watch-set" rather than "app/agent/". Update `_format_stale_error` and any user-facing string that names `app/agent/` so the error message reflects the broader watch-set (BASE-04 traceability). Do NOT add non-stdlib imports. Add the six behavior tests, mirroring the existing monkeypatch idiom.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_check_baselines_fresh.py -v -k "llm_factory or eval_matrix or watch or stale or agent_change"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "WATCH_PREFIXES" scripts/check_baselines_fresh.py` shows the list containing `app/llm_factory.py` and `configs/eval_matrix`
    - `_agent_changed` uses `any(p.startswith(prefix) for prefix in WATCH_PREFIXES)`
    - The dry-run test for `app/llm_factory.py` (no refresh) asserts exit 1; the existing `app/agent/` no-refresh test still asserts exit 1
    - `scripts/check_baselines_fresh.py` imports remain stdlib-only (`grep -n "^import\|^from" scripts/check_baselines_fresh.py` shows only argparse/re/subprocess/sys/collections.abc)
    - `poetry run pytest tests/unit/test_check_baselines_fresh.py -k "llm_factory or eval_matrix"` exits 0
  </acceptance_criteria>
  <done>The staleness gate watches app/llm_factory.py and configs/eval_matrix*.yaml in addition to app/agent/; dry-run tests prove exit 1; stdlib-only constraint preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: WR-10 probe fixture type fidelity for additional_kwargs values</name>
  <files>scripts/probe_provider_capture.py, tests/unit/test_adapters.py</files>
  <read_first>
    - scripts/probe_provider_capture.py — read `_redact` (line 82, note it does `s = str(value)` at line 90 — this is why values are stringified), `_scan_fixture_for_secrets` (line 104), the fixture-build block (lines 218-255) where `add_kwargs_values = {k: _redact(message.additional_kwargs[k]) for k in add_kwargs_keys}` (line 220) collapses bytes/dict to strings, and the `response_metadata`/`usage_metadata`/`tool_calls` handling (lines 225-235) which already preserves type via `json.loads(_redact(json.dumps(..., default=str)))` — that is the type-faithful pattern to imitate for additional_kwargs. Read the post-write secret-scan guard (lines 257+).
    - tests/unit/test_adapters.py — read `test_adapter_capture_on_real_wire_fixture` (lines 912-955); it reconstructs `additional_kwargs = payload.get("additional_kwargs_values", {})` and builds an `AIMessage`. After the WR-10 fix the values will be real-typed (dict/list/number where the wire had them), so the adapter parses real shapes.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md — WR-10 scope note (non-blocking, any wave; affects test fidelity, not recorded baselines).
    - .planning/phases/10-eval-harness-honesty/10-REVIEW.md — WR-10 finding text (probe stringifies additional_kwargs values so adapter fixture tests never see real-typed bytes/dict paths; D-10-12 residual).
  </read_first>
  <behavior>
    - Test: build a probe-fixture dict where `additional_kwargs` has a dict value (e.g. a structured reasoning payload) and a bytes value (e.g. a thought_signature); after the WR-10 redaction path, the JSON-serialized fixture's `additional_kwargs_values` preserves the dict as a dict (and the bytes encoded faithfully, e.g. base64 or a documented encoding), NOT a Python `str()` repr like `"{'k': 'v'}"`.
    - Test: secrets in `additional_kwargs` values are still redacted (the secret-scan guard still passes / a planted env-var secret is removed) — type fidelity must not weaken redaction.
    - Test: `test_adapter_capture_on_real_wire_fixture` still passes when a fixture is present and SKIPS when absent (no regression to the CI-safe skip).
  </behavior>
  <action>
    Fix WR-10 by routing `additional_kwargs` values through the same type-faithful redaction pattern already used for `response_metadata`/`usage_metadata`/`tool_calls`: instead of `_redact(message.additional_kwargs[k])` (which calls `str()` and collapses types), serialize then redact then re-parse — `json.loads(_redact(json.dumps(message.additional_kwargs[k], default=str)))` — so dict/list/number values round-trip as real JSON types and only genuinely non-JSON-serializable leaves (e.g. raw bytes) fall back to a `default=str` encoding. Keep `additional_kwargs_keys` as-is. Preserve the post-write secret-scan fail-closed guard unchanged (it must still delete + return 2 on a surviving secret). Update the inline comment at lines 237-239 to document that additional_kwargs_values is now type-faithful (matching the response_metadata pattern). Add the three behavior tests; for the adapter fixture test, add a focused unit test that constructs a synthetic type-faithful fixture in `tmp_path`, writes it, and asserts the dict/bytes value reconstructs to the real type (do not require live keys).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_adapters.py -v -k "fixture or wr10 or type_fidel or additional_kwargs" && poetry run python -c "import scripts.probe_provider_capture as p; import json; print(json.loads(p._redact(json.dumps({'a': {'b': 1}}))))"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "json.loads(_redact(json.dumps(" scripts/probe_provider_capture.py` shows the additional_kwargs path using the type-faithful round-trip (not bare `_redact(message.additional_kwargs[k])`)
    - A test asserts a dict-valued additional_kwargs entry survives as a dict (not a `str()` repr) in the written fixture
    - A test asserts a planted secret in an additional_kwargs value is still redacted
    - `poetry run pytest tests/unit/test_adapters.py -k "fixture or additional_kwargs"` exits 0
  </acceptance_criteria>
  <done>Probe fixtures carry real-typed additional_kwargs values through redaction; redaction still fail-closed; adapter fixture tests exercise real-wire types; tests pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| agent-loop source change → committed baselines | the staleness gate is the only automated guard that a behavior-changing file (now incl. llm_factory) forces a baseline refresh |
| live provider response → checked-in fixture | probe fixtures are committed to the repo; redaction must remove secrets while preserving type shape |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-17 | Tampering | unwatched llm_factory change | mitigate | D-11-21 watch-set extension fails CI when app/llm_factory.py or eval_matrix*.yaml change without a baseline refresh; dry-run tests prove exit 1 |
| T-11-18 | Information disclosure | probe fixture additional_kwargs | mitigate | WR-10 fix preserves the existing `_redact` + fail-closed post-write secret-scan guard; type fidelity routes through the same redaction, never around it |
| T-11-19 | Tampering | stdlib-only lint constraint | mitigate | the staleness lint stays stdlib-only so the zero-dependency `lint-baselines` CI job is unaffected (verified by import grep) |
| T-11-07-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH §Package Legitimacy Audit: none) |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_check_baselines_fresh.py tests/unit/test_adapters.py -v` passes.
- `python scripts/check_baselines_fresh.py --help` exits 0 (stdlib-only, runs without deps installed).
- `poetry run ruff check scripts/check_baselines_fresh.py scripts/probe_provider_capture.py` clean.
- `make test` full suite passes.
</verification>

<success_criteria>
- BASE-04 watch-set covers llm_factory + eval_matrix configs with dry-run proof; WR-10 fixture type-fidelity fixed without weakening redaction.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-07-SUMMARY.md` when done.
</output>
