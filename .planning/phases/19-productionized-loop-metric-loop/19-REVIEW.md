---
phase: 19-productionized-loop-metric-loop
reviewed: 2026-06-21T04:30:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - app/loop/falsifier_core.py
  - scripts/loop_runner.py
  - scripts/provision_sandbox.sh
  - Makefile
  - tests/unit/test_falsifier_core_recall.py
  - tests/unit/test_loop_runner_orchestrator.py
  - tests/unit/test_provision_sandbox_populated.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 19: Code Review Report

**Reviewed:** 2026-06-21T04:30:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 19 productionizes the adaptive data loop with a new `loop_runner.py` orchestrator, a `compute_recall_at_k` primitive in `falsifier_core.py`, a populated-baseline provisioner (`provision_sandbox.sh --populated`), and three Makefile targets. The two bugs already found and fixed this phase (metric target-set asymmetry, Gemini block-content crash) are confirmed fixed. The overall architecture is sound, the D-07 coercion-ordering is implemented correctly, and the D-08 gap-handoff set-diff is correct.

Two CRITICAL issues remain:

1. **SQL injection in `_snapshot_ids_from_url`** — the `table` parameter is interpolated directly into a SQL string using an f-string with no parameterization or allowlist check. All four call sites pass hardcoded string literals today, so there is currently no exploitable path, but the function's signature accepts any string.

2. **`provision_sandbox.sh` URL parser stderr bleed** — the Python URL-parsing subshell redirects `2>&1`, which merges Poetry's own stderr output (deprecation warnings, update notices) into the captured `_PARSED_FIELDS` variable. If Poetry emits one or more lines to stderr before the Python script runs, `cut -f1` on the mixed output will set `_parsed_dbname` to those warning lines instead of the database name, causing every subsequent guard and DDL step to operate against an incorrect `DB_NAME` value — potentially silently bypassing the `_sandbox` suffix guard.

---

## Critical Issues

### CR-01: SQL injection via f-string table interpolation in `_snapshot_ids_from_url`

**File:** `scripts/loop_runner.py:151`

**Issue:** `_snapshot_ids_from_url` builds its query as
```python
cur.execute(f"SELECT place_id FROM {table}")  # noqa: S608
```
The `table` parameter is a caller-supplied string. While every call site today passes a hardcoded literal (`"places_raw"`, `"place_embeddings_v2"`), the function is a module-level helper with no type-level or runtime allowlist guard. Any future caller that passes a non-literal — including a value read from the environment or from an upstream return value — can inject arbitrary SQL. The S608 suppression documents that ruff sees this, but the pattern itself is not safe. The identical vulnerability exists in the copied `loop_falsifier.py` original (L664), but this review scope is Phase 19 only.

**Fix:** Add an allowlist guard before the execute:

```python
_ALLOWED_SNAPSHOT_TABLES = frozenset({"places_raw", "place_embeddings_v2"})

def _snapshot_ids_from_url(db_url: str, table: str) -> set[str]:
    if table not in _ALLOWED_SNAPSHOT_TABLES:
        raise ValueError(
            f"_snapshot_ids_from_url: table {table!r} is not in the allowed set "
            f"{_ALLOWED_SNAPSHOT_TABLES}. Pass a hardcoded table name."
        )
    import psycopg2  # noqa: PLC0415
    with contextlib.closing(psycopg2.connect(db_url)) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT place_id FROM {table}")  # noqa: S608
        return {row[0] for row in cur.fetchall()}
```

This keeps the fast path unchanged for the four existing call sites and closes the injection surface for future callers without requiring a prepared-statement refactor (not possible for table names via `%s` in psycopg2 anyway).

---

### CR-02: `provision_sandbox.sh` URL-parser stderr bleed corrupts `_parsed_dbname`

**File:** `scripts/provision_sandbox.sh:90-116`

**Issue:** The Python URL-parser subshell is captured with `2>&1`:

```bash
_PARSED_FIELDS=$(
  poetry run python -c "..." 2>&1
) || { exit 1; }
```

The `|| { exit 1; }` guard correctly catches a non-zero Python exit. However it does **not** guard against Poetry itself printing lines to stderr while Python exits zero. Poetry routinely emits lines such as `"Warning: poetry.lock is not consistent..."` or update notices to stderr. With `2>&1`, these lines appear in `_PARSED_FIELDS` before the real `dbname\thost\tinstance` line. When `cut -f1` runs on the multi-line result, `_parsed_dbname` receives the first non-empty line — which is the Poetry warning — not the dbname. The subsequent empty-string guard (`if [[ -z "${_parsed_dbname}" ]]`) does not fire because the warning text is non-empty. As a result:

- `DB_NAME` is set to, e.g., `"Warning: poetry.lock is not consistent with pyproject.toml"`.
- The `_sandbox` suffix check (`[[ "${DB_NAME}" != *"_sandbox"* ]]`) fires, but the error message names the mangled string, not the real dbname — confusing diagnosis.
- Worse: the check *could* spuriously pass if the warning text happened to contain `_sandbox`.

In contrast, the `_GUARD_RESULT` block immediately below (line 151) correctly uses `2>/dev/null`, suppressing Poetry stderr entirely.

**Fix:** Match the guard-result block's stderr handling:

```bash
_PARSED_FIELDS=$(
  poetry run python -c "
import sys, os
try:
    from app.loop.falsifier_core import _normalize_url
    url = os.environ.get('SANDBOX_DATABASE_URL', '')
    host, port, dbname, instance = _normalize_url(url)
    print(dbname + '\t' + host + '\t' + instance)
except Exception as e:
    print(f'ERROR: could not parse SANDBOX_DATABASE_URL via _normalize_url: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
) || {
  echo "ERROR: Python URL parser failed for SANDBOX_DATABASE_URL." >&2
  echo "  Ensure poetry and app.loop.falsifier_core are available." >&2
  exit 1
}
```

This drops Poetry's own stderr (identical to what the guard-result block does) while Python's explicit `sys.stderr` output still triggers the non-zero exit and the `|| { exit 1; }` branch.

---

## Warnings

### WR-01: `decide_loop_exit` always called with `guard_violation=None` — dead parameter path

**File:** `scripts/loop_runner.py:640-645`

**Issue:** The `decide_loop_exit` function accepts a `guard_violation: GuardResult | None` parameter and handles `guard_violation.ok == False → EXIT_INFRA` as its highest-priority branch. In `loop_runner.main()`, this parameter is always hard-coded to `None`:

```python
exit_code = decide_loop_exit(
    before_rate=before_hit_at_k,
    after_rate=after_hit_at_k,
    floor=floor,
    guard_violation=None,   # always None
    embed_added_count=embed_added_count,
)
```

The actual guard results (`safety_result`, `non_circ`) are checked inline earlier with `raise SystemExit(EXIT_INFRA)` and never flow through `decide_loop_exit`. The function's guard-violation path is therefore untested in the live code path (only tested in unit tests that pass a synthetic violation). This is not a bug today — the early exits guarantee the gate is never reached with a violation — but it means the `guard_violation` parameter in `decide_loop_exit` is dead code in the orchestrator, creating a misleading contract and a maintenance hazard (a future refactor could remove the early exits while passing `None` and silently lose the guard).

**Fix:** Either remove `guard_violation` from `decide_loop_exit` (it was designed for the case where guards are threaded through rather than exit-on-fail) and update its unit tests, or actually pass the guard results through. The simpler and lower-risk option:

```python
# Compute guard result explicitly to pass to decide_loop_exit
guard_result: GuardResult | None = None  # all guards passed (early-exit above)
exit_code = decide_loop_exit(
    before_rate=before_hit_at_k,
    after_rate=after_hit_at_k,
    floor=floor,
    guard_violation=guard_result,
    embed_added_count=embed_added_count,
)
```

With an explanatory comment this documents the intent. The real fix is removing `guard_violation` from `decide_loop_exit` if guards always exit inline.

---

### WR-02: `make loop` OPENAI_API_KEY guard is misleading when the default paraphrase provider is Gemini

**File:** `Makefile:273-276`

**Issue:** The `make loop` target guards `OPENAI_API_KEY` as a hard requirement:

```makefile
@[ -n "$${OPENAI_API_KEY:-}" ] || { \
  echo "ERROR: OPENAI_API_KEY is not set."; \
  exit 1; \
}
```

However, the default paraphrase provider resolved at runtime is `DEFAULT_JUDGE_PROVIDER = "gemini"` (from `app/agent/critique/vibe.py`). When `PARAPHRASE_PROVIDER` is unset (the typical case), paraphrase generation calls Gemini, not OpenAI. The `OPENAI_API_KEY` guard therefore fails the operator with a misleading error when they set `GEMINI_API_KEY` (or equivalent) but not `OPENAI_API_KEY`. The runbook (`docs/loop_runner.md`) and Makefile comment both say the key is required, but it is only required if OpenAI is the paraphrase provider.

**Fix:** Either change the guard to check `GEMINI_API_KEY` (or the actual key the default provider needs), or make the guard provider-aware with a note in the help text:

```makefile
# Paraphrase provider defaults to gemini — GEMINI_API_KEY is required unless
# PARAPHRASE_PROVIDER=openai is set (in which case OPENAI_API_KEY is required).
# For simplicity, the guard checks both; only the active provider's key is used.
```

Or simply update the guard to check `GEMINI_API_KEY` and document that `OPENAI_API_KEY` is only needed when `PARAPHRASE_PROVIDER=openai`.

---

### WR-03: `_generate_paraphrases` does not validate that each element of the JSON array is a string

**File:** `scripts/loop_runner.py:224-239`

**Issue:** After parsing the LLM's JSON response, the code validates `isinstance(paraphrases, list) and len(paraphrases) == n` but does not check that each element is a `str`. If the model returns a JSON array of arrays, objects, or numbers — `[[1,2,3],[4,5,6],...]` passes the length check — the list is returned and subsequently passed to `semantic_search(p, k=K)` where `p` is a non-string. This would raise a runtime `TypeError` deep inside `semantic_search` rather than the clean `EXIT_INFRA` the orchestrator promises for malformed LLM output.

```python
# Current — insufficient:
if not isinstance(paraphrases, list) or len(paraphrases) != n:
    raise SystemExit(EXIT_INFRA)
return paraphrases, generation_prompt, model_name
```

**Fix:**

```python
if not isinstance(paraphrases, list) or len(paraphrases) != n:
    raise SystemExit(EXIT_INFRA)

# Validate each element is a non-empty string
invalid = [p for p in paraphrases if not isinstance(p, str) or not p.strip()]
if invalid:
    print(
        f"[INFRA] Paraphrase generation returned {len(invalid)} non-string or empty elements: "
        f"{invalid[:3]!r} ...",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_INFRA)

return paraphrases, generation_prompt, model_name
```

---

### WR-04: `make sandbox-provision-populated` does not guard `LOOP_GAP_NEIGHBORHOOD` / `LOOP_GAP_CUISINE`

**File:** `Makefile:71-79`

**Issue:** The `sandbox-provision-populated` target guards only `SANDBOX_DATABASE_URL`:

```makefile
sandbox-provision-populated:
    @[ -n "$${SANDBOX_DATABASE_URL:-}" ] || { echo "ERROR: ..."; exit 1; }
    bash scripts/provision_sandbox.sh --populated
```

When `LOOP_GAP_NEIGHBORHOOD` and `LOOP_GAP_CUISINE` are not set, `provision_sandbox.sh` falls back to the defaults `"Outer Sunset"` and `"vietnamese"` silently. This is a silent footgun: an operator who omits these variables gets a baseline provisioned for the wrong gap bucket, and `make loop` will subsequently fail the BASELINE/MINER GAP CONTRACT check (EXIT_INFRA) if the miner picks a different bucket — but only after full ingest+embed has already run. The Makefile is the right place to surface this early.

**Fix:** Add guards for the gap-bucket variables in the Makefile target:

```makefile
sandbox-provision-populated:
    @[ -n "$${SANDBOX_DATABASE_URL:-}" ] || { \
      echo "ERROR: SANDBOX_DATABASE_URL is not set."; exit 1; }
    @[ -n "$${LOOP_GAP_NEIGHBORHOOD:-}" ] || { \
      echo "ERROR: LOOP_GAP_NEIGHBORHOOD is not set. Export before provisioning."; exit 1; }
    @[ -n "$${LOOP_GAP_CUISINE:-}" ] || { \
      echo "ERROR: LOOP_GAP_CUISINE is not set. Export before provisioning."; exit 1; }
    bash scripts/provision_sandbox.sh --populated
```

The `provision_sandbox.sh` script already documents the defaults but not having Makefile-level guards causes silent misconfiguration.

---

## Info

### IN-01: `datetime.datetime.utcnow()` is deprecated in Python 3.12+

**File:** `scripts/loop_runner.py:527`

**Issue:** `datetime.datetime.utcnow().isoformat() + "Z"` produces a DeprecationWarning in Python 3.12+ and is scheduled for removal. The project targets Python 3.10+, so this is not a breaking bug today, but the warning will surface in logs on 3.12.

**Fix:**
```python
"generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
```
This produces a timezone-aware ISO 8601 string with `+00:00` suffix (equivalent meaning; omit the manual `"Z"` append since `isoformat()` includes the offset).

---

### IN-02: `loop_runner_artifacts/` is not gitignored — future runs will leave untracked files

**File:** `scripts/loop_runner.py:70`, `scripts/loop_runner.py:534-537`

**Issue:** `RUNNER_ARTIFACT_DIR = "loop_runner_artifacts"` creates a directory at the repo root that is not listed in `.gitignore`. The directory itself is gitignored by implication (`.json` files are globally ignored), but the directory entry is not, so `git status` will show `loop_runner_artifacts/` as an untracked directory after every `make loop` run. This is minor noise but degrades the working-tree-clean discipline the project values.

**Fix:** Add to `.gitignore`:
```
loop_runner_artifacts/
```

---

### IN-03: `TestGapHandoffColdStart` does not assert the cold-start log message

**File:** `tests/unit/test_loop_runner_orchestrator.py:166-202`

**Issue:** The cold-start test asserts `exc_info.value.code == EXIT_PASS` (correct) but does not verify that the print message is the expected cold-start message (distinct from other EXIT_PASS paths). This is not a correctness bug, but the test would pass even if `main()` exited 0 for an entirely different reason — weakening its discriminating power. The stale-proposal ordering test uses source inspection as a workaround; a `capsys` check on the cold-start output would make this behavioral test more precise without adding API cost.

**Fix (optional, low priority):** Use `capsys` to assert the cold-start message:
```python
captured = capsys.readouterr()
assert "Cold start or no demand gaps found" in captured.out
```

---

_Reviewed: 2026-06-21T04:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
