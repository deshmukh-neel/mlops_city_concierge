---
phase: 10-eval-harness-honesty
plan: 05
type: execute
wave: 2
depends_on: ["10-04"]
files_modified:
  - scripts/probe_provider_capture.py
  - tests/fixtures/provider_payloads/.gitkeep
  - tests/unit/test_adapters.py
  - tests/unit/test_probe_provider_capture.py
  - Makefile
autonomous: true
requirements: [EVAL-05]
must_haves:
  truths:
    - "One generalized probe script makes a tool-call-shaped request per provider and writes a redacted fixture"
    - "Redaction covers OpenAI, Anthropic, and Google key shapes plus env-var-sourced secrets, and is unit-tested"
    - "A post-write secret-scan guard refuses to write a fixture containing a secret pattern"
    - "Adapter unit tests gain parametrized cases that load real-wire fixtures and assert capture does not crash"
    - "The synthetic hand-written adapter cases are retained, not replaced"
    - "make probe-providers is the documented mandatory pre-matrix step"
  artifacts:
    - path: "scripts/probe_provider_capture.py"
      provides: "generalized --provider probe writing redacted JSON fixtures"
      contains: "--provider"
    - path: "tests/unit/test_probe_provider_capture.py"
      provides: "redaction unit tests (fake leaked key -> redacted)"
      contains: "_redact"
    - path: "tests/unit/test_adapters.py"
      provides: "parametrized fixture-loading adapter tests augmenting synthetic cases"
      contains: "provider_payloads"
    - path: "Makefile"
      provides: "probe-providers target (manual, documented mandatory pre-matrix step)"
      contains: "probe-providers"
  key_links:
    - from: "scripts/probe_provider_capture.py"
      to: "tests/fixtures/provider_payloads/{provider}.json"
      via: "redacted AIMessage dump written per provider"
      pattern: "provider_payloads"
    - from: "tests/fixtures/provider_payloads/{provider}.json"
      to: "tests/unit/test_adapters.py"
      via: "parametrized loader feeds the fixture into the provider's adapter"
      pattern: "provider_payloads"
---

<objective>
Close the synthetic-vs-live test gap with a standing mitigation (EVAL-05). Phase 9 shipped
adapters whose unit tests used hand-written synthetic AIMessage dicts; the real wire shapes
differed (the Gemini lcgg key-shape miss D-09-09, four live-only Anthropic bugs). This plan
generalizes `scripts/probe_gpt5_capture.py` into `scripts/probe_provider_capture.py
--provider {openai|deepseek|anthropic|gemini}` which makes one ~$0.01 tool-call-shaped request
per provider and writes a redacted AIMessage dump to
`tests/fixtures/provider_payloads/{provider}.json` (D-10-11). Adapter unit tests gain
parametrized cases that LOAD those real-wire fixtures and assert capture does not crash — the
hand-written synthetic cases are AUGMENTED, never replaced (D-10-12).

Redaction is mandatory and tested: extend beyond the `sk-` prefix to cover Anthropic
(`sk-ant-`), Google (`AIzaSy`), and env-var-sourced secret values, with a unit test that feeds a
fake leaked key through the probe writer and asserts redaction (D-10-13 — folds in Phase 9
review finding IN-04). A post-write secret-scan guard refuses to write a fixture containing a
secret pattern. `make probe-providers` is the documented mandatory pre-matrix step; no CI/cron
(no live keys in CI per D-09-10/D-10-14).

Purpose: future provider changes are caught against real wire shapes, not synthetic guesses.
Output: a generalized probe, checked-in redacted fixtures, and fixture-backed adapter tests.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/10-eval-harness-honesty/10-CONTEXT.md
@.planning/phases/10-eval-harness-honesty/10-PATTERNS.md
@scripts/probe_gpt5_capture.py
@app/agent/adapters/__init__.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Generalize the probe into probe_provider_capture.py with extended redaction + secret-scan guard</name>
  <files>scripts/probe_provider_capture.py, tests/fixtures/provider_payloads/.gitkeep, tests/unit/test_probe_provider_capture.py, Makefile</files>
  <read_first>
    - scripts/probe_gpt5_capture.py (read the whole file: imports/repo-root :29-42, _redact :63-75, _content_shape :78-onward, the AIMessage dump assembly, the final secret-scan guard around :226-233 — generalize ALL of it from gpt-5-only to --provider)
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (the probe_provider_capture.py sections: extended _SECRET_PATTERNS list, FIXTURE_DIR path, fixture JSON shape, argparse --provider, final secret-scan guard)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-11 full-fidelity probes -> checked-in fixtures; D-10-13 mandatory tested redaction beyond sk-; D-10-14 manual + documented, no CI)
    - app/llm_factory.py (read SUPPORTED_PROVIDERS and build_chat_model to pick the right default model per provider; the probe uses build_chat_model exactly like the gpt5 probe does)
    - Makefile (read POETRY_RUN :6 and a target style; probe-providers must NOT depend on CI / must be documented as the mandatory pre-matrix step)
  </read_first>
  <behavior>
    - probe_provider_capture.py accepts --provider in {openai, deepseek, anthropic, gemini} and an optional --model override; it builds the chat model via app.llm_factory.build_chat_model, makes one tool-call-shaped request, and writes a redacted JSON fixture to tests/fixtures/provider_payloads/{provider}.json.
    - _redact replaces OpenAI (sk-), Anthropic (sk-ant-), and Google (AIzaSy...) key shapes and any env-var-sourced secret value with a redaction marker; the original secret substring never survives in the output.
    - A post-write secret-scan re-reads the written fixture and, if any secret pattern is found, deletes/refuses the write and returns a non-zero exit (the fixture is never committed with a leaked secret).
    - The fixture JSON contains provider, model, library_version, additional_kwargs_keys, redacted additional_kwargs_values, sanitized response_metadata, content_shape, and tool_calls.
  </behavior>
  <action>
    Create scripts/probe_provider_capture.py by generalizing scripts/probe_gpt5_capture.py: keep the `from app.llm_factory import build_chat_model` import (poetry editable install — NO sys.path bootstrap per project memory `project_app_editable_install`). Replace the hardcoded gpt-5 target with an argparse `--provider` (choices: openai, deepseek, anthropic, gemini) and optional `--model`; map each provider to a sensible default model (openai -> gpt-5-mini, deepseek -> deepseek-reasoner, anthropic -> claude-sonnet-4-6, gemini -> gemini-3.1-pro-preview) overridable via --model. Replace the single `_SECRET_PATTERN` with a `_SECRET_PATTERNS` list covering: OpenAI `sk-[A-Za-z0-9_-]{20,}`, Anthropic `sk-ant-[A-Za-z0-9_-]{20,}`, Google `AIzaSy[A-Za-z0-9_-]{33}`, and a generic long-token pattern; also redact any value found in os.environ for known secret env-var names (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, etc.) per D-10-13. `_redact` applies every pattern AND the env-sourced-secret substitution. Change the output target from the .planning .md artifact to a JSON fixture at `tests/fixtures/provider_payloads/{provider}.json` (D-10-11 shape from PATTERNS.md: provider, model, library_version, probe_query, additional_kwargs_keys, redacted additional_kwargs_values, sanitized response_metadata, content_shape, usage_metadata, tool_calls). Keep and generalize the final post-write secret-scan guard: re-read the written file, scan with every pattern, and if any matches print FATAL to stderr and return non-zero (refuse the leaked write). Create tests/fixtures/provider_payloads/.gitkeep so the dir is tracked. Create tests/unit/test_probe_provider_capture.py with redaction unit tests: feed a fake Anthropic key (sk-ant-api03-...), a fake OpenAI key, a fake Google key, and a fake env-sourced secret through _redact and assert the raw secret substring is absent and the marker is present; add a test that the post-write guard rejects a fixture containing a planted secret. Add a Makefile `.PHONY: probe-providers` target that runs the probe for all four providers in sequence with a help comment marking it the MANDATORY pre-matrix step (D-10-14); the comment states no CI/cron (no live keys in CI). Do NOT run the live probe in this task — only the script + redaction tests (which use synthetic strings, no live calls).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_probe_provider_capture.py -q && poetry run python scripts/probe_provider_capture.py --help</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/probe_provider_capture.py --help` exits 0 and lists `--provider` with choices openai/deepseek/anthropic/gemini.
    - `grep -n "sk-ant-\|AIzaSy" scripts/probe_provider_capture.py` shows the extended key patterns.
    - A redaction test asserts `"sk-ant-api03-..." not in _redact("sk-ant-api03-...")` and a redaction marker IS present (per PATTERNS.md test_probe_redaction_catches_anthropic_key).
    - A test asserts the post-write secret-scan guard returns non-zero / refuses when the fixture contains a planted secret.
    - The fixture output path is `tests/fixtures/provider_payloads/{provider}.json` (source assertion: `grep -n "provider_payloads" scripts/probe_provider_capture.py`).
    - `Makefile` contains `.PHONY: probe-providers` and the help comment marks it the mandatory pre-matrix step with no CI/cron.
    - `poetry run ruff check scripts/probe_provider_capture.py tests/unit/test_probe_provider_capture.py` passes.
  </acceptance_criteria>
  <done>A generalized, redaction-hardened probe writes JSON fixtures with a fail-closed secret guard; redaction is unit-tested; make probe-providers is documented as mandatory pre-matrix and CI-free.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add parametrized real-wire fixture-loading tests to test_adapters.py (augment, not replace)</name>
  <files>tests/unit/test_adapters.py</files>
  <read_first>
    - tests/unit/test_adapters.py (read the existing synthetic per-adapter tests :1-119 and the adapter imports :18-21; the new tests AUGMENT these — the synthetic cases stay)
    - app/agent/adapters/__init__.py (read the ADAPTERS registry :168-175 — openai/gemini/deepseek/anthropic adapter instances — to dispatch a provider string to its adapter; note "openai" -> OpenAIReasoningAdapter)
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (the parametrized test_adapter_capture_on_real_wire_fixture example: skip when fixture absent, build an AIMessage from the fixture's additional_kwargs_values/response_metadata, call capture_reasoning_state, assert no crash and result shape)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-12 fixtures augment never replace; closes D-09-09 Gemini + 4 Anthropic live-only bugs)
  </read_first>
  <behavior>
    - For each provider in {openai, deepseek, anthropic, gemini}, a parametrized test loads tests/fixtures/provider_payloads/{provider}.json if present, reconstructs an AIMessage from the redacted fixture (additional_kwargs + response_metadata + content shape), calls that provider's adapter.capture_reasoning_state, and asserts it does not raise; if a result is returned it has the expected shape (e.g. contains a provider key).
    - When the fixture file is absent the test SKIPS with a message instructing `make probe-providers` (fixtures are produced by a live probe, so CI without keys skips rather than fails).
    - All existing synthetic adapter tests still pass unchanged.
  </behavior>
  <action>
    In tests/unit/test_adapters.py, add a parametrized test `test_adapter_capture_on_real_wire_fixture` over provider in {openai, deepseek, anthropic, gemini} following the PATTERNS.md skeleton: define FIXTURE_DIR = repo_root/tests/fixtures/provider_payloads; skip with a clear message ("run make probe-providers first") when the fixture is absent; load the JSON, build an AIMessage from `additional_kwargs_values` (reconstruct content per `content_shape`; for Anthropic the adapter reads message.content thinking blocks so reconstruct list-content when content_shape indicates blocks), dispatch to the provider's adapter via a small `_adapter_for(provider)` helper that maps the provider string to the ADAPTERS instance (openai -> ADAPTERS["openai"], etc.), call `capture_reasoning_state(msg)` and assert it does not raise; if the result is not None assert it contains a "provider" key. Do NOT delete or weaken any existing synthetic test — these are additive (D-10-12). Keep redacted fixture values as opaque strings (the adapter is tested for crash-safety against the real KEY SHAPE, not against real secret content).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_adapters.py -q</automated>
  </verify>
  <acceptance_criteria>
    - `tests/unit/test_adapters.py` contains `test_adapter_capture_on_real_wire_fixture` parametrized over the four providers (source assertion: `grep -n "test_adapter_capture_on_real_wire_fixture\|provider_payloads" tests/unit/test_adapters.py`).
    - The new test SKIPS (not fails) when a fixture is absent — `poetry run pytest tests/unit/test_adapters.py -q` exits 0 even with no fixtures checked in.
    - All pre-existing synthetic adapter tests still pass (the count of pre-existing test functions is unchanged; new ones are additive).
    - `poetry run ruff check tests/unit/test_adapters.py` passes.
  </acceptance_criteria>
  <done>Adapter tests now run against real-wire fixtures (skipping gracefully when absent) while retaining every synthetic case, closing the live-vs-synthetic gap class.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| provider live response → checked-in fixture | the probe captures a real provider response and commits it; a leaked API key or PII in the fixture would be published to the repo |
| os.environ secrets → fixture content | env-var-sourced secrets could be echoed into additional_kwargs/response_metadata and written to disk |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-05-01 | Information disclosure | fixture write in probe_provider_capture.py | mitigate | _redact applies OpenAI/Anthropic/Google key patterns AND env-sourced secret substitution before write (D-10-13); a post-write secret-scan guard re-reads and refuses to keep a fixture containing any secret pattern (fail-closed); redaction is unit-tested with planted fake keys |
| T-10-05-02 | Information disclosure | env-var secret echo | mitigate | Known secret env-var values (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY) are substituted in _redact regardless of where they appear in the message |
| T-10-05-03 | Tampering | live probe in CI | mitigate | probe-providers is manual-only and documented as CI-free (D-10-14); no live keys in CI (D-09-10); adapter tests SKIP without fixtures so CI never needs to run the probe |
| T-10-05-04 | Spoofing | fixture review | accept | Fixtures are reviewed before commit like any code (D-10-13); the post-write guard is defense-in-depth, human review is the primary control |
| T-10-05-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs; probe uses existing langchain provider SDKs already in pyproject.toml |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_probe_provider_capture.py tests/unit/test_adapters.py -q` exits 0 (no live calls; adapter fixture tests skip when fixtures absent).
- `poetry run python scripts/probe_provider_capture.py --help` exits 0.
- `poetry run ruff check scripts/probe_provider_capture.py tests/unit/test_probe_provider_capture.py tests/unit/test_adapters.py` passes.
</verification>

<success_criteria>
- A generalized live-probe Make target produces redacted checked-in fixtures consumed by adapter tests; redaction is mandatory and tested (EVAL-05).
- Real-wire fixtures augment (never replace) the synthetic adapter cases; absent fixtures skip rather than fail.
- make probe-providers is documented as the mandatory CI-free pre-matrix step.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-05-SUMMARY.md` when done.
</output>
