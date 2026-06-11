---
phase: 10-eval-harness-honesty
plan: 09
type: execute
wave: 1
gap_closure: true
closes_crs: [CR-05, CR-04]
depends_on: []
files_modified:
  - scripts/probe_provider_capture.py
  - tests/unit/test_probe_provider_capture.py
autonomous: true
requirements: [EVAL-05]
must_haves:
  truths:
    - "response_metadata, usage_metadata, and tool_calls are routed through _redact before being written to the fixture"
    - "The post-write secret-scan guard also checks env-var-sourced secret values, not only the _SECRET_PATTERNS regexes"
    - "A planted env-var-sourced secret (non-regex-shaped) in a fixture field is caught and the fixture is deleted with a non-zero return"
    - "The hardcoded absolute path in test_main_help_exits_zero is replaced with a repo-root-relative resolution so the test passes on CI/Ubuntu"
  artifacts:
    - path: "scripts/probe_provider_capture.py"
      provides: "fail-closed redaction across all value-bearing fixture fields + env-var-aware post-write guard"
      contains: "_SECRET_ENV_VARS"
    - path: "tests/unit/test_probe_provider_capture.py"
      provides: "portable subprocess test + env-var post-write-guard test"
      contains: "Path(__file__)"
  key_links:
    - from: "scripts/probe_provider_capture.py response_metadata/usage_metadata/tool_calls"
      to: "_redact"
      via: "json.loads(_redact(json.dumps(...))) before write"
      pattern: "_redact"
    - from: "scripts/probe_provider_capture.py post-write guard"
      to: "_SECRET_ENV_VARS env values"
      via: "env-var substitution check mirroring _redact"
      pattern: "_SECRET_ENV_VARS"
---

<objective>
Close CR-05 (WARNING, security) and CR-04 (BLOCKER, portability), both in the EVAL-05 probe area:

CR-05: `scripts/probe_provider_capture.py` claims fail-closed redaction, but three fixture fields
bypass `_redact`: `response_metadata` (lines 196, 211) goes only through `_sanitize_response_metadata`
which blanks 2 fixed keys non-recursively, while `usage_metadata` (line 213) and `tool_calls`
(line 214) are written RAW. Additionally the post-write guard (lines 224-233) scans only the 4
`_SECRET_PATTERNS` regexes and omits the env-var-sourced secret check that `_redact` applies — so a
non-regex-shaped key (e.g. a rotated DeepSeek key, a non-`AIzaSy` Google key) appearing in
`response_metadata` or `tool_calls` would be written to a checked-in fixture and survive the guard.
The SUMMARY/docs "fail-closed" claim is currently false.

CR-04: `tests/unit/test_probe_provider_capture.py:158` hardcodes
`cwd="/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge"` — guaranteed `FileNotFoundError`
on CI (Ubuntu) and every other machine, turning `make test` red everywhere but the author's laptop.

Purpose: Make the probe's redaction genuinely fail-closed (no checked-in fixture can carry a secret),
and make the probe test pass on CI so the EVAL-05 harness is trustworthy and portable.
Output: Redaction routed through every value-bearing fixture field; env-var-aware post-write guard;
repo-root-relative test path; a test that plants an env-var-sourced secret and proves the guard deletes the fixture.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-eval-harness-honesty/10-VERIFICATION.md
@.planning/phases/10-eval-harness-honesty/10-05-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Route all value-bearing fixture fields through _redact and make the post-write guard env-var-aware (CR-05)</name>
  <files>scripts/probe_provider_capture.py, tests/unit/test_probe_provider_capture.py</files>
  <read_first>
    - scripts/probe_provider_capture.py lines 79-98 (`_redact` — stringifies its input then substitutes env-var secret values from `_SECRET_ENV_VARS` and applies `_SECRET_PATTERNS`; returns a str), lines 61-76 (`_SECRET_PATTERNS` + `_SECRET_ENV_VARS`), lines 111-125 (`_sanitize_response_metadata` — blanks `system_fingerprint`/`id` only, non-recursive), lines 193-219 (the fixture dict build: `response_metadata = _sanitize_response_metadata(...)` at 196, `usage_metadata = getattr(...)` at 198, `tool_calls = getattr(...)` at 199; fixture fields written at 211/213/214; `default=str` json.dump at 219), lines 221-238 (post-write guard: re-reads `text`, loops `_SECRET_PATTERNS` only, deletes file + returns 2 on match).
    - tests/unit/test_probe_provider_capture.py lines 100-137 (existing post-write-guard tests that re-implement the guard inline with `_SECRET_PATTERNS`), lines 53-66 (env-var redaction test pattern using `patch.dict(os.environ, {...})` + `importlib.reload`).
    - NOTE (scope guard): WR-08 (stringification flattens wire types) is OUT OF SCOPE for this gap-closure run — do NOT rewrite `_redact` into a recursive `_redact_obj`. Keep `_redact`'s stringify-then-substitute contract; apply it to the JSON-serialized form of dict/list fields so structure is preserved for the fixture (`json.loads(_redact(json.dumps(field, default=str)))`).
  </read_first>
  <behavior>
    - After the fix, `response_metadata`, `usage_metadata`, and `tool_calls` written to the fixture have passed through `_redact` (env-var secret values and regex-shaped secrets removed).
    - The post-write guard, given a fixture text containing a value equal to an env-var-sourced secret (length >= 10) that does NOT match any `_SECRET_PATTERNS` regex, detects it, deletes the fixture, and `main` returns 2.
    - A clean (fully redacted) fixture still passes the guard and `main` returns 0 (no false positives).
  </behavior>
  <action>
    In scripts/probe_provider_capture.py: (1) Route the three bypassing fields through `_redact` before they enter the fixture dict. For the dict/list-shaped fields, preserve JSON structure by redacting the serialized form: build `response_metadata` value as `json.loads(_redact(json.dumps(_sanitize_response_metadata(dict(message.response_metadata or {})), default=str)))`, `usage_metadata` as `json.loads(_redact(json.dumps(usage_metadata, default=str)))` when `usage_metadata is not None` else `None`, and `tool_calls` as `json.loads(_redact(json.dumps(tool_calls, default=str)))`. Keep `_sanitize_response_metadata` as the first pass on response_metadata (defense in depth), then redact. (2) Extend the post-write guard (lines 224-233): before/after the `_SECRET_PATTERNS` loop, add a loop over `_SECRET_ENV_VARS` that reads `secret_val = os.environ.get(env_var, "")` and, if `secret_val and len(secret_val) >= 10 and secret_val in text`, deletes the fixture (`fixture_file.unlink(missing_ok=True)`), prints the same FATAL stderr message naming the env-var class, and `return 2` — mirroring the substitution logic already in `_redact` lines 90-94. (3) Update the module docstring (lines 8-11) so the "fail-closed" claim is accurate: state that response_metadata, usage_metadata, and tool_calls are all redacted and the post-write guard checks both regex patterns and env-var secret values. In tests/unit/test_probe_provider_capture.py: add `test_post_write_guard_catches_env_var_sourced_secret` that patches `os.environ` with a non-regex-shaped secret (e.g. `{"DEEPSEEK_API_KEY": "ROTATED-not-sk-shaped-XYZ-9876543210"}`), reloads the module, writes a fixture whose `tool_calls` or `response_metadata` text contains that exact value, runs the actual `main`-guard path (or a dedicated extracted guard helper if one is introduced) against it, and asserts the fixture is deleted and the return is 2. Prefer extracting the post-write scan into a small testable helper (e.g. `_scan_fixture_for_secrets(text) -> bool`) called by `main`, so the test exercises production code instead of re-implementing the guard inline — but keep `main`'s behavior byte-identical.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_probe_provider_capture.py -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "_redact(json.dumps" scripts/probe_provider_capture.py` returns >= 3 (response_metadata, usage_metadata, tool_calls all routed through _redact).
    - `grep -c "_SECRET_ENV_VARS" scripts/probe_provider_capture.py` returns >= 2 (used in both _redact AND the post-write guard).
    - The new test plants a non-regex-shaped env-var secret into a fixture field, runs the production guard path, and asserts the fixture is deleted and the scan signals a secret / return code 2.
    - `poetry run pytest tests/unit/test_probe_provider_capture.py -q` exits 0 (existing 10 tests plus the new env-var-guard test).
    - The module docstring no longer claims fail-closed redaction while leaving fields unredacted (the three fields are demonstrably redacted).
  </acceptance_criteria>
  <done>response_metadata, usage_metadata, and tool_calls pass through _redact before write; the post-write guard checks _SECRET_ENV_VARS values in addition to _SECRET_PATTERNS; a planted non-regex env-var secret is caught and the fixture deleted; docstring made accurate.</done>
</task>

<task type="auto">
  <name>Task 2: Replace the hardcoded absolute path in test_main_help_exits_zero with a repo-root-relative resolution (CR-04)</name>
  <files>tests/unit/test_probe_provider_capture.py</files>
  <read_first>
    - tests/unit/test_probe_provider_capture.py lines 10-15 (imports: `from pathlib import Path` is already present), lines 151-166 (`test_main_help_exits_zero` — the subprocess call hardcodes `cwd="/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge"` at line 158 and uses the relative script path `"scripts/probe_provider_capture.py"` which only resolves because of that cwd).
    - tests/unit/test_check_eval_gates.py line 25 — the in-repo idiom this codebase uses: `REPO_ROOT = Path(__file__).resolve().parents[2]` (test files live at `tests/unit/`, so `parents[2]` is the repo root). Match this exactly.
  </read_first>
  <behavior>
    - `test_main_help_exits_zero` resolves the repo root and the script path from `Path(__file__)` so it runs from any working directory and on any machine (CI/Ubuntu included).
    - The subprocess `--help` invocation still exits 0 and lists `--provider`.
  </behavior>
  <action>
    In tests/unit/test_probe_provider_capture.py, add a module-level `REPO_ROOT = Path(__file__).resolve().parents[2]` constant (mirroring tests/unit/test_check_eval_gates.py:25) near the top of the file after the imports. In `test_main_help_exits_zero`, replace the hardcoded `cwd="/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge"` with `cwd=str(REPO_ROOT)`, and replace the relative script argument `"scripts/probe_provider_capture.py"` with `str(REPO_ROOT / "scripts" / "probe_provider_capture.py")` so the path is absolute and cwd-independent. Leave the assertions (returncode == 0, `--provider` in output) unchanged.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_probe_provider_capture.py -q -k "main_help"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "/Users/pnhek" tests/unit/test_probe_provider_capture.py` returns 0 (no hardcoded author path remains).
    - `grep -c "Path(__file__).resolve().parents" tests/unit/test_probe_provider_capture.py` returns >= 1 (repo-root-relative resolution present).
    - `poetry run pytest tests/unit/test_probe_provider_capture.py -q -k "main_help"` exits 0.
    - Running the test from a different cwd still passes: `cd /tmp && poetry --directory "<repo>" run pytest "<repo>/tests/unit/test_probe_provider_capture.py" -q -k main_help` exits 0 (cwd-independence proven).
  </acceptance_criteria>
  <done>test_main_help_exits_zero derives cwd and the script path from Path(__file__).resolve().parents[2]; no absolute author-specific path remains; the test passes from any cwd.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| live provider wire response -> checked-in fixture | UNTRUSTED: model-generated content (tool_calls args, response_metadata) can echo back a secret; this is the primary security boundary for this phase |
| os.environ secret values -> fixture file on disk | A leaked env-var secret written to a committed fixture is a real credential-exposure path |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-09-01 | Information Disclosure | secret in response_metadata/usage_metadata/tool_calls written to committed fixture | mitigate | Route all three value-bearing fields through `_redact` (env-var substitution + regex) before write — Task 1 |
| T-10-09-02 | Information Disclosure | non-regex-shaped env-var secret surviving the post-write guard | mitigate | Extend post-write guard to substitute-check `_SECRET_ENV_VARS` values, deleting the fixture + return 2 on a hit — Task 1 |
| T-10-09-03 | Tampering | redaction silently regressing in future | mitigate | Test plants a non-regex env-var secret and asserts the production guard deletes the fixture — Task 1 |
| T-10-09-SC | Tampering | npm/pip/cargo installs | accept | No package installs in this plan; no new dependencies added |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_probe_provider_capture.py -q` exits 0.
- Manual: `grep -n "_redact(json.dumps" scripts/probe_provider_capture.py` shows all three fields redacted; `grep -n "_SECRET_ENV_VARS" scripts/probe_provider_capture.py` shows the guard usage; `grep -n "/Users/pnhek" tests/unit/test_probe_provider_capture.py` returns nothing.
- Full suite (per project memory — real-graph tests leak a DB pool unless run together): `make test` exits 0.
</verification>

<success_criteria>
- All value-bearing fixture fields are redacted and the post-write guard checks env-var secrets (CR-05 closed; EVAL-05 PARTIAL -> VERIFIED; the "fail-closed" SUMMARY/docs claim is now true).
- The probe test no longer hardcodes an author-specific path (CR-04 closed; `make test` passes on CI/Ubuntu).
- The verification truth "The post-write secret-scan guard in probe_provider_capture.py is fail-closed (EVAL-05)" flips to VERIFIED.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-09-SUMMARY.md` when done.
</output>
