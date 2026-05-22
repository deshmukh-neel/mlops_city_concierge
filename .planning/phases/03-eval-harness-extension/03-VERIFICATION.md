---
phase: 03-eval-harness-extension
verified: 2026-05-22T16:35:00Z
status: human_needed
score: 10/10 must-haves verified (CR-01 + CR-02 + WR-01..WR-04 + IN-01..IN-05 all CLOSED by plans 03-08..03-12; only remaining human-action item is the user-run baseline matrix)
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: "10/10 must-haves verified (with 2 material warnings on baseline-diff correctness)"
  gaps_closed:
    - "CR-01 BLOCKER: `_scorer_means_from_cell` whitelisted via `CRITIQUE_THRESHOLDS.keys()`; the 6 polluting non-scorer `_mean` keys (tool_calls, results, contexts, revision_hints, committed_stops, answer_retrieved_place_coverage) no longer leak into summary.json. Plan 03-08, commit 17f82a4."
    - "CR-02 BLOCKER: `ScriptedChatModel._generate` constructs a fresh AIMessage on every empty-script call; the module-level `_DEFAULT_SCRIPTED_FALLBACK` singleton is removed. LangGraph `add_messages` no longer identity-dedupes consecutive fallbacks; revision/replan loops in CI scripted mode will progress instead of spinning until max_steps. Plan 03-09, commit 8c02af9. Empirical regression `a is not b` confirmed: PASS."
    - "WR-01: `MatrixEntry` rejects `--` in provider/model with a field-named ValidationError; aggregator additionally emits a WARNING log on unparseable cell filenames. Plan 03-08, commit f5d4e9a."
    - "WR-02: `_run_git` raises `RuntimeError` on rc != 0 and converts `FileNotFoundError` to actionable RuntimeError; `_resolve_base` rejects empty-string BASE_SHA/--merge-base; `main()` translates RuntimeError into rc=2 (distinct from rc=1 stale-baseline). Plan 03-10, commit f58df9c. Empirical regression `python scripts/check_baselines_fresh.py \"\"` rc=2: PASS."
    - "WR-03: Shared `tests/_helpers/scripted_llm.py` created with `ScriptedLLM` + `RecordingScriptedLLM`; consumer test files import the helper; production `ScriptedChatModel` intentionally NOT folded into the helper (scenario-registry semantics differ). Plan 03-11, commit 35d3e2b."
    - "WR-04: 5 dead outer-scope `seen: list[list[BaseMessage]] = []` variables removed from `tests/unit/test_eval_agent.py`; `RecordingScriptedLLM.seen` now uses `Field(default_factory=list)` so each instance owns its list. Plan 03-11, commit 35d3e2b. Empirical regression `a.seen is not b.seen`: PASS."
    - "IN-01: `.github/workflows/ci.yml:lint-baselines` step threads BASE_SHA via an `env:` block + `\"$BASE_SHA\"` inside `run:`; the unsafe `${{ … }}` interpolation no longer appears in any `run:` string. Plan 03-10, commit bedf585."
    - "IN-02: `summary.json` now carries top-level `overridden_to: <provider>` when `--llm-provider-override` is set; `run_matrix` logs the rebind once at INFO. Plan 03-08, commit de46f54. Empirical check: PASS."
    - "IN-03: Makefile split shared `RUNS` variable into per-target `QUERIES` (eval-agent → `--max-queries`) and `RUNS` (eval-matrix → `--runs`). `make -n eval-agent RUNS=99` now ignores the old habit and uses the QUERIES default. Plan 03-12, commit f588cb1."
    - "IN-04: `_scorer_means_from_cell` excludes bool values disguised as numeric (`isinstance(value, int | float) or isinstance(value, bool)`). Closed alongside CR-01 in plan 03-08, commit 17f82a4."
    - "IN-05: Fallback AIMessage content is now `[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py.` — PR reviewers reading CI summary.json files immediately recognize it as deterministic placeholder output, not a real model failure. Closed alongside CR-02 in plan 03-09, commit 8c02af9."
  gaps_remaining:
    - "Baseline JSON numeric content still requires the user-run `APP_ENV=eval make eval-matrix RUNS=3` + post-process step. The three files in `configs/eval_baselines/` retain their `_status: PENDING_USER_RUN` sentinel and `median: null` placeholders. This is the documented closing-handoff per plan 03-07's deviation block (no executor `.env` access; ~15 min wall time; real API spend). It does NOT block Phase 3 closure per the explicit plan-03-07 decision — the structural lint gate is live and Phase 4 can begin while the baselines are filled."
  regressions: []
human_verification:
  - test: "Run the live matrix to overwrite the PENDING_USER_RUN baseline stubs. With `OPENAI_API_KEY` + `DEEPSEEK_API_KEY` in `.env`, run `APP_ENV=eval make eval-matrix RUNS=3` (~15 min, real API spend). Then post-process `eval_reports/{ts}/summary.json` into the three `configs/eval_baselines/*.json` files: copy `summary.scenarios.{scenario_id}.providers.{provider/model}.scorers.{scorer}` into each baseline's scorer block; set `generated_at` to the ISO timestamp; set `generated_by` to `make eval-matrix RUNS=3`; remove the top-level `_status` field. Plan 03-08 closed CR-01 BEFORE this step, so the summary.json the user will read no longer carries the 6 polluting non-scorer keys — direct copy-paste works without manual whitelisting."
    expected: "Each `configs/eval_baselines/*.json` has numeric (non-null) median/min/max/stdev/n values per scorer per provider; `generated_at` is a real ISO timestamp; `generated_by` is `make eval-matrix RUNS=3`; the top-level `_status` field is removed. The three files become the diff target for Phase 4-6 merge gates ('scorer median >= baseline median + delta')."
    why_human: "Requires real API keys + real money spend; project memory + plan-03-07's stub-path decision both route this step to the user. Not testable in CI."
---

# Phase 3: Eval Harness Extension — Verification Report (RE-VERIFICATION)

**Phase Goal (ROADMAP.md):** "Every subsequent agent-behavior fix can be scored against a committed baseline across multiple providers, with multi-turn and cross-model coverage."

**Verified:** 2026-05-22 (re-verification after gap-closure plans 03-08 through 03-12 shipped)
**Status:** human_needed (single remaining item: user-run baseline matrix)
**Re-verification:** YES — initial verification (2026-05-22 at HEAD `10414e1`) flagged 2 BLOCKERs (CR-01, CR-02) and 9 lower-severity items routed to human decision; all 11 issues are now closed by plans 03-08, 03-09, 03-10, 03-11, 03-12 (HEAD now `4a16eeb`).
**Branch:** `gsd/phase-03-eval-harness-extension` (HEAD `4a16eeb`)

---

## Re-Verification Summary

### What Changed Since the Previous Verification

| Item | Previous Status | Current Status | Closing Plan / Commit |
|------|----------------|----------------|-----------------------|
| CR-01 (BLOCKER) — `_scorer_means_from_cell` admits non-scorer `_mean` keys | OPEN (routed to human) | **CLOSED** — whitelist via `CRITIQUE_THRESHOLDS` | 03-08 / `17f82a4` |
| CR-02 (BLOCKER) — `ScriptedChatModel` returns module-level singleton AIMessage | OPEN (routed to human) | **CLOSED** — fresh AIMessage per call | 03-09 / `8c02af9` |
| WR-01 — `--` in MatrixEntry provider/model silently drops cells | OPEN | **CLOSED** — `reject_double_dash` validator + aggregator WARN log | 03-08 / `f5d4e9a` |
| WR-02 — `_run_git` silently passes when git is unreachable / BASE_SHA empty | OPEN | **CLOSED** — loud-fail RuntimeError → rc=2 | 03-10 / `f58df9c` |
| WR-03 — `_RecordingScriptedLLM` / `_ScriptedLLM` triplicated across test files | OPEN | **CLOSED** — hoisted into `tests/_helpers/scripted_llm.py` | 03-11 / `35d3e2b` |
| WR-04 — Dead outer-scope `seen` vars in test_eval_agent.py | OPEN | **CLOSED** — removed; `Field(default_factory=list)` | 03-11 / `35d3e2b` |
| IN-01 — GitHub Actions `${{ }}` interpolation into `run:` string | OPEN | **CLOSED** — env block + `"$BASE_SHA"` | 03-10 / `bedf585` |
| IN-02 — `--llm-provider-override scripted` rebinds keys silently | OPEN | **CLOSED** — top-level `overridden_to` summary field | 03-08 / `de46f54` |
| IN-03 — Makefile `RUNS` has split semantics on eval-agent vs eval-matrix | OPEN | **CLOSED** — `QUERIES` on eval-agent; `RUNS` only on eval-matrix | 03-12 / `f588cb1` |
| IN-04 — `aggregate_cell_jsons` admits `bool` as numeric | OPEN | **CLOSED** — explicit `isinstance(value, bool)` exclusion | 03-08 / `17f82a4` |
| IN-05 — `_DEFAULT_SCRIPTED_FALLBACK` content is ambiguous in summary.json | OPEN | **CLOSED** — `[SCRIPTED CI MODE]` marker + script reference | 03-09 / `8c02af9` |
| Baseline numeric content (user-run matrix + post-process) | OPEN (routed to human) | **STILL OPEN** — single remaining human-action item | — (user action; not gap-closable in-repo) |

**Net result:** 11 of 12 previously-open items closed; 1 remaining (baseline matrix run) is a user-action item that does not block Phase 3 closure per plan 03-07's explicit deviation block.

### Verdict

**Phase 3 has shipped a fully-correct measurement instrument.** The two BLOCKERs that materially affected Phase 4-6 baseline-diff correctness (CR-01 pollution, CR-02 identity-dedup) are closed and have empirical regression coverage. The lower-severity items (4 warnings, 5 info) are all closed as well.

The only remaining work is the user's manual matrix run + baseline post-processing — which the plan-03-07 executor explicitly routed to the user (no executor `.env` access; ~15 min wall time; real API spend), and which the stale-baseline CI lint correctly treats as structurally-sufficient-by-presence (the gate operates on the file diff, not numeric content). Phase 4 can begin in parallel with the user's baseline run; the structural commitment is in place and the user-run is a clean handoff.

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria + Plan must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `category_compliance` and `rationale_stop_alignment` scorers exist in `checks.py` and are callable on any `ItineraryState`; both appear in `DETERMINISTIC_CHECKS` / `CRITIQUE_THRESHOLDS` | VERIFIED | `app/agent/critique/checks.py:218` (`def category_compliance`) and `:256` (`def rationale_stop_alignment`); both keys in `CRITIQUE_THRESHOLDS` (lines 30-31); both called via `_try` in `itinerary_violations` (lines 317, 322); empirical: `python -c "from app.agent.critique.checks import CRITIQUE_THRESHOLDS; print(sorted(CRITIQUE_THRESHOLDS.keys()))"` returns 8 scorers including both new ones |
| 2 | `scripts/eval_matrix.py` runs a cross-model matrix and emits per-row JSON + aggregated diff-friendly summary; `make eval-matrix` target works; **summary contains only registered scorers (no pollution)** | VERIFIED | Module exists with subprocess fan-out; `make eval-matrix` target invokes it; `APP_ENV=eval poetry run python scripts/eval_matrix.py --dry-run --runs 3 --matrix-config configs/eval_matrix.yaml \| wc -l` returns 18 cells. **CR-01 closed**: empirical check confirms `_scorer_means_from_cell({...8 keys incl. 6 polluters and 1 bool})` returns only `['category_compliance', 'rationale_stop_alignment']`. |
| 3 | Multi-turn scenario threading: when `EvalQuery.turns` is non-None, the runner feeds each turn in sequence, threading `conversation_state` between calls | VERIFIED | `scripts/eval_agent.py:evaluate_multi_turn_case` is async; `evaluate_case` has the `if case.turns:` branch; threading invariant pinned by tests in `tests/unit/test_eval_agent.py` using the now-shared `RecordingScriptedLLM` from `tests/_helpers/scripted_llm.py` (WR-03 hoist preserved the original test semantics) |
| 4 | Three committed baselines exist in `configs/eval_baselines/` (open-ended omakase, refinement-turn cheaper, late-night closure-cascade); CI lint fails if `app/agent/` changes without updating them | VERIFIED (structural); PARTIAL (numeric) | `ls configs/eval_baselines/` returns exactly the 3 files; each has `closure_check_confirmed: "2026-05-21"`, the correct provider keys (`openai/gpt-4o-mini`, `deepseek/deepseek-chat`), full scorer shape with `n: 3` per scorer. Numeric medians are still `null` (PENDING_USER_RUN). `scripts/check_baselines_fresh.py` runs cleanly; **WR-02 hardening**: now rc=2 on infrastructure failure, rc=1 on stale-baseline (distinct CI signal); CI `lint-baselines` job hard-gated with no `continue-on-error`. |
| 5 | CI runs eval matrix in `--llm scripted` mode; real-provider runs gated behind `APP_ENV=eval`; **scripted mode is correct (no singleton AIMessage identity-dedup)** | VERIFIED | `.github/workflows/ci.yml:eval-matrix` job exists; `make eval-matrix LLM_OVERRIDE=scripted RUNS=1` is the invocation; `APP_ENV` not set anywhere in eval-matrix job region; `_gate_blocks` enforces APP_ENV=eval for non-scripted (empirical: `APP_ENV=dev … --runs 1` exits rc=2). **CR-02 closed**: empirical check confirms `m._generate(messages=[]).generations[0].message is m._generate(messages=[]).generations[0].message` returns `False` (fresh instance per call); marker `[SCRIPTED CI MODE]` present in fallback content. |

**Truth score:** 5/5 VERIFIED structurally. Truth #4 has a numeric-content PARTIAL handled by the single remaining human_verification item.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/agent/critique/checks.py` | Two new deterministic scorers + threshold registration + itinerary_violations wiring | VERIFIED | Both function defs + thresholds + `_try` calls present (lines 218, 256, 30, 31, 317, 322) |
| `app/agent/state.py` | `UserConstraints.requested_primary_types: list[str] = []` (D-01) | VERIFIED | Field default `[]` |
| `app/eval/config.py` | `EvalQuery.turns`, `EvalMatrixConfig`, `MatrixEntry`, `load_eval_matrix` (EVAL-03, EVAL-04); **MatrixEntry rejects `--` (WR-01 closed)** | VERIFIED | All four symbols importable; YAML round-trip works; `MatrixEntry(provider='openai', model='gpt-4--turbo')` raises ValidationError citing `'--' is reserved as the cell-filename separator`; `MatrixEntry(provider='ope--nai', ...)` also rejected |
| `scripts/eval_agent.py` | Multi-turn helper + `--llm-provider scripted` + `--scenario-ids` filter | VERIFIED | All present; tests pass |
| `scripts/eval_matrix.py` | Subprocess fan-out runner + summary.json aggregator + APP_ENV gate; **scorer whitelist + bool exclude + overridden_to + WARN on unparseable** | VERIFIED | `from app.agent.critique.checks import CRITIQUE_THRESHOLDS` import present (line 44); `scorer_name in CRITIQUE_THRESHOLDS` filter (line 168); `isinstance(value, bool)` exclusion (line 164); `overridden_to` field conditional (line 268); `_log.warning(...)` on unparseable cell filename (line 237); `_log.info(...)` on override active (line 362) |
| `app/llm_factory.py` | `ScriptedChatModel` + `'scripted'` SUPPORTED_PROVIDERS; **fresh AIMessage per call + `[SCRIPTED CI MODE]` marker** | VERIFIED | Class exists, no-network, no-keys; module-level `_DEFAULT_SCRIPTED_FALLBACK` REMOVED (grep returns 0 matches); `_generate` constructs inline `AIMessage(content="[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py.", tool_calls=[])` on each empty-script call (lines 147-153); empirical `a is not b` confirms fresh-construction |
| `configs/eval_matrix.yaml` | D-06 anchors locked | VERIFIED | `[(openai, gpt-4o-mini), (deepseek, deepseek-chat)]` × 3 scenarios; 18 cells at runs=3 |
| `configs/eval_queries.yaml` | Three new EvalQuery cases | VERIFIED | All 3 case IDs present (omakase_mission_open_ended, refinement_cheaper, late_night_closure_cascade) |
| `configs/eval_baselines/*.json` | 3 baseline files, correct shape, closure_check_confirmed populated | VERIFIED (shape); PARTIAL (numeric content stubs) | All 3 files exist with full shape, ISO date, both D-06 provider keys, 7 scorer entries each; `_status: PENDING_USER_RUN` sentinel present; medians are null pending user matrix run |
| `scripts/check_baselines_fresh.py` | Stale-baseline lint with `[skip-baseline]` bypass; **loud-fail on infrastructure failure (rc=2)** | VERIFIED | Runs cleanly against `origin/main`; 13 unit tests pass (9 truth-table + 4 WR-02); empirical `python scripts/check_baselines_fresh.py ""` exits rc=2 with actionable stderr; `_run_git` raises `RuntimeError` on rc != 0 and `FileNotFoundError`; `_resolve_base` rejects empty-string BASE_SHA |
| `Makefile` | `eval-agent` + `eval-matrix` targets; **QUERIES vs RUNS split (IN-03)** | VERIFIED | Both `.PHONY` targets present with `## help` lines; `make -n eval-agent QUERIES=3` plumbs `--max-queries 3`; `make -n eval-agent RUNS=99` plumbs `--max-queries 1` (the old habit is ignored, not silently accepted); `make -n eval-matrix RUNS=3` plumbs `--runs 3` |
| `.gitignore` | `eval_reports/` excluded, `configs/eval_baselines/*.json` tracked | VERIFIED | Unchanged from previous verification |
| `.github/workflows/ci.yml` | `eval-matrix` job (scripted, soft gate) + `lint-baselines` job (hard gate); **IN-01 env-block-not-interpolation** | VERIFIED | Both jobs present; `eval-matrix` has `continue-on-error: true` (soft); `lint-baselines` has no continue-on-error (hard); BASE_SHA threaded via `env:` block (line 155: `BASE_SHA: ${{ github.event.pull_request.base.sha }}`); `run:` consumes `"$BASE_SHA"` (line 156); no `${{ … }}` interpolation in any `run:` string |
| `tests/_helpers/scripted_llm.py` (NEW) | Hoisted `ScriptedLLM` + `RecordingScriptedLLM` for DRY | VERIFIED | Module exists with `ScriptedLLM` (raises `IndexError` on exhaustion — loud-fail) and `RecordingScriptedLLM(ScriptedLLM)` with `seen: list[list[BaseMessage]] = Field(default_factory=list)`; 5 dedicated tests pass; consumer files in `tests/unit/` import from this module; production `ScriptedChatModel` intentionally NOT folded in (scenario-registry semantics differ) |

### Key Link Verification

| From | To | Via | Status |
|------|-----|-----|--------|
| `configs/eval_baselines/{scenario}.json` | `03-CLOSURE-PRECHECK.md` | `closure_check_confirmed` ISO date copied verbatim | VERIFIED — all 3 baselines stamp `2026-05-21` matching D-07 verdict |
| `scripts/check_baselines_fresh.py` | `git diff --name-only` | subprocess call | VERIFIED — runs against `origin/main` cleanly; now loud-fails on rc != 0 |
| `.github/workflows/ci.yml:lint-baselines` | `scripts/check_baselines_fresh.py` | `poetry run python` step | VERIFIED — step present, BASE_SHA threaded via env block (IN-01 closed) |
| `.github/workflows/ci.yml:eval-matrix` | `scripts/eval_matrix.py` | `make eval-matrix LLM_OVERRIDE=scripted` | VERIFIED — invocation present, no APP_ENV in job |
| `scripts/eval_matrix.py:run_matrix` | `scripts/eval_agent.py:main` | `subprocess.run([sys.executable, …])` | VERIFIED — fan-out shape locked |
| `app/agent/critique/checks.py:category_compliance` | `app/agent/state.py:UserConstraints.requested_primary_types` | `state.constraints.requested_primary_types` read | VERIFIED — D-03 abstain on `[]` |
| `scripts/eval_agent.py:DETERMINISTIC_CHECKS` | `app/agent/critique/checks.py` | both new scorer names registered | VERIFIED |
| `scripts/eval_matrix.py:_scorer_means_from_cell` | `app/agent/critique/checks.py:CRITIQUE_THRESHOLDS` | whitelist filter | **NEW — VERIFIED (CR-01 closure)** — empirical: polluted input filtered to only `category_compliance` + `rationale_stop_alignment` |
| `tests/unit/test_eval_agent.py` + `test_chat_functional.py` | `tests/_helpers/scripted_llm.py` | `from tests._helpers.scripted_llm import …` | **NEW — VERIFIED (WR-03 closure)** — consumer files import the helper; no local `_ScriptedLLM` / `_RecordingScriptedLLM` definitions remain |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `configs/eval_baselines/*.json` | `scorers.{name}.median` | `summary.json` from `make eval-matrix RUNS=3` (user-run step) | NO — currently `null` everywhere | DISCONNECTED until user runs the matrix (single remaining human_verification item) |
| `summary.json` (when produced by user) | scorer means | `aggregate_cell_jsons` → `_scorer_means_from_cell` | YES — **and now CLEAN** (CR-01 closed; polluting non-scorer `_mean` keys filtered out) | FLOWING (will produce clean output when user runs matrix) |
| `category_compliance` scorer output | `state.constraints.requested_primary_types` | Agent intake LLM (D-02, deferred to Phase 4 per CONTEXT.md) | NO — field stays `[]` in current scenarios → scorer abstains (returns 1.0 by D-03) | EXPECTED PHASE-3 STATE; will flow once Phase 4 wires intake |
| `ScriptedChatModel._generate` empty-script return | `msg` (AIMessage) | Fresh `AIMessage(...)` constructed inline per call | YES (and unique per call — CR-02 closed) | FLOWING (no identity-dedup, no max_steps spin) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Unit suite green | `poetry run pytest tests/unit/ -q --no-cov` | **725 passed**, 9 warnings, 11.57s | PASS (was 705 pre-Wave; +20 new tests from gap-closure plans) |
| Matrix dry-run lists 18 cells | `APP_ENV=eval poetry run python scripts/eval_matrix.py --dry-run --runs 3 --matrix-config configs/eval_matrix.yaml \| wc -l` | 18 | PASS |
| Stale-baseline lint runs cleanly | `poetry run python scripts/check_baselines_fresh.py origin/main` | rc=0, "OK — app/agent/ changed and 3 baseline file(s) refreshed" | PASS |
| **WR-02 loud-fail on empty BASE_SHA** | `poetry run python scripts/check_baselines_fresh.py ""` | rc=2, "BASE_SHA positional argument was the empty string; …" | **PASS (new)** |
| CR-02 fresh-AIMessage regression | `python -c "from app.llm_factory import ScriptedChatModel; m = ScriptedChatModel(); a = m._generate(messages=[]).generations[0].message; b = m._generate(messages=[]).generations[0].message; assert a is not b; assert '[SCRIPTED CI MODE]' in a.content"` | exits 0; `a is not b` → True; marker present | **PASS (new)** |
| CR-01 scorer-whitelist regression | `_scorer_means_from_cell` on a polluted payload with 6 fake `_mean` keys + 1 bool + 2 real scorers | output keys: `['category_compliance', 'rationale_stop_alignment']` (6 polluters + 1 bool excluded) | **PASS (new)** |
| WR-01 MatrixEntry `--` reject | `MatrixEntry(provider='openai', model='gpt-4--turbo')` | raises ValidationError with field-named message citing `'--' is reserved` | **PASS (new)** |
| WR-04 `seen` default_factory | `a = RecordingScriptedLLM(scripted=[]); b = RecordingScriptedLLM(scripted=[]); a.seen is not b.seen` | True (each instance has its own list) | **PASS (new)** |
| IN-02 `overridden_to` field | `aggregate_cell_jsons(empty_dir, llm_provider_override='scripted')['overridden_to']` | `'scripted'`; with `override=None`, key is absent | **PASS (new)** |
| IN-03 Makefile rename | `make -n eval-agent RUNS=99` | plumbs `--max-queries 1` (the QUERIES default; old habit ignored) | **PASS (new)** |
| IN-01 env-block usage | `grep -E "^\s*BASE_SHA: \\$\\{\\{" .github/workflows/ci.yml` + `grep "\"\\$BASE_SHA\"" .github/workflows/ci.yml` | env block defines BASE_SHA; run consumes `"$BASE_SHA"`; no `${{` inside any `run:` string | **PASS (new)** |
| Gitignore re-include keeps baselines tracked | `git check-ignore configs/eval_baselines/omakase_mission_open_ended.json` | rc=1 (NOT ignored) | PASS |
| Gitignore excludes eval_reports/ | `git check-ignore eval_reports/foo.json` | rc=0 (ignored) | PASS |
| Baseline numeric medians populated | `python -c` walking the 3 JSONs and checking medians are non-null | all 3 medians are `null` (stub state) | **EXPECTED FAIL** — handled by single remaining `human_verification` item |

### Probe Execution

No project-conventional `scripts/*/tests/probe-*.sh` files exist in this repo, and the phase did not declare any probes. Probe execution: SKIPPED (no probes defined). The behavioral spot-checks + unit suite are the operative verification surface for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| **EVAL-01** | 03-03 | `category_compliance` scorer | SATISFIED | `app/agent/critique/checks.py:218`; in `CRITIQUE_THRESHOLDS`, `_try`, and `DETERMINISTIC_CHECKS` |
| **EVAL-02** | 03-03 | `rationale_stop_alignment` scorer | SATISFIED | `app/agent/critique/checks.py:256`; closure-swap placeholder regression test |
| **EVAL-03** | 03-02 | `EvalQuery.turns` field | SATISFIED | Field in `app/eval/config.py`; backward-compat round-trip verified |
| **EVAL-04** | 03-02 | `EvalMatrixConfig` + `MatrixEntry` | SATISFIED | Both models; `MatrixEntry.reject_double_dash` validator added (WR-01) |
| **EVAL-05** | 03-05 + 03-08 | Cross-(provider, model) matrix runner | **SATISFIED (CR-01 closed)** | Module + subprocess fan-out + `_scorer_means_from_cell` whitelist via `CRITIQUE_THRESHOLDS`. No more pollution. |
| **EVAL-06** | 03-04 + 03-11 | Multi-turn threading | SATISFIED | `evaluate_multi_turn_case`; threading invariant pinned by tests using the now-shared `RecordingScriptedLLM` |
| **EVAL-07** | 03-07 + 03-10 | Baselines + stale-baseline lint | **SATISFIED structurally; PARTIAL numerically** | 3 baseline files committed with full shape; closure_check_confirmed stamped; `check_baselines_fresh.py` now loud-fails on infra failure (WR-02); CI `lint-baselines` hard-gated with IN-01 env-block. Numeric medians remain `null` pending user matrix run. |
| **EVAL-08** | 03-03 + 03-04 | json.dumps(args) safety on every eval test | SATISFIED | 12+ `json.dumps` assertions in `tests/unit/test_eval_agent.py`; `test_multi_turn_tool_calls_are_json_safe` cites PR #94 commit `be541a3` |
| **EVAL-09** | 03-06 + 03-09 | CI scripted-mode eval gate + APP_ENV=eval gate for real providers | **SATISFIED (CR-02 closed)** | `eval-matrix` job + `_gate_blocks` enforcement + 3 CI-drift tests + **fresh-AIMessage contract** in `ScriptedChatModel._generate`. No more identity-dedup risk; CI scripted-mode revision loops will progress. |
| **EVAL-10** | 03-05 + 03-12 | Makefile `eval-agent` + `eval-matrix` targets | SATISFIED | Both `.PHONY` targets present; **IN-03 closed** — `QUERIES` on eval-agent (matches `--max-queries`), `RUNS` on eval-matrix (matches `--runs`); old habit (`RUNS=99` to eval-agent) ignored not silently consumed |

**Coverage:** 10/10 requirements SATISFIED structurally; EVAL-07 numeric content remains in PARTIAL pending the user-run matrix step (handled by the single `human_verification` item).

No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (None remaining) | — | The 6 anti-patterns flagged in the previous verification (CR-01, CR-02, WR-01, WR-02, WR-03/04, IN-01..IN-05) are all closed by plans 03-08 through 03-12. | — | — |

**Debt-marker gate:** Clean. `grep -rn 'TBD\|FIXME\|XXX' app/agent/critique/checks.py scripts/eval_matrix.py scripts/eval_agent.py scripts/check_baselines_fresh.py app/llm_factory.py tests/_helpers/scripted_llm.py` returns no matches in any Phase-3 modified file (including the new `tests/_helpers/scripted_llm.py`).

### Observed Latent Issue (INFO — does not block phase closure)

**Baseline scorer-list / CRITIQUE_THRESHOLDS mismatch.** The 3 committed baseline JSONs each enumerate **7** scorers (`category_compliance`, `rationale_stop_alignment`, `constraints_satisfied`, `geographic_coherence`, `temporal_coherence`, `walking_budget_respected`, `no_hallucinated_place_ids`). `CRITIQUE_THRESHOLDS` registers **8** scorers (the above 7 plus `stop_count_satisfied`). When the user runs `APP_ENV=eval make eval-matrix RUNS=3`, the resulting summary.json will include a `stop_count_satisfied` scorer column that has no baseline entry to compare against.

This is a **post-processing concern**, not a Phase-3 blocker:
- The baselines are stubs being overwritten anyway (PENDING_USER_RUN).
- The user can include `stop_count_satisfied` in the post-processed baselines without code changes.
- Phase 4-6 merge gate ("scorer median ≥ baseline median + delta") only diffs scorers that appear in BOTH baseline and current run — a missing baseline scorer is silently ignored, not a hard failure.

Recommended action: when the user post-processes summary.json into the baseline files, add a `stop_count_satisfied` block alongside the existing 7. Alternatively, a follow-up plan can update the baseline stubs to enumerate all 8 scorers before the user's matrix run — but neither path blocks Phase 3 closure.

### Human Verification Required

See the `human_verification:` block in the frontmatter for the structured form. Summarized here:

1. **Run the live matrix + post-process to overwrite the baseline stubs.** This is the closing-handoff step plan-03-07's executor explicitly routed to the user (no executor `.env` access; ~15 min wall time; real API spend). Plan-03-07 SUMMARY's Handoff section has the verbatim command sequence. **CR-01 is closed BEFORE this step**, so the summary.json the user reads no longer carries the 6 polluting non-scorer keys — direct copy-paste into the baseline JSONs works without manual whitelisting. (Optional: include the 8th scorer `stop_count_satisfied` per the INFO note above.)

### Gaps Summary

**Zero code-level gaps remain.** All 2 BLOCKERs and 9 lower-severity items flagged in the previous verification are closed by plans 03-08 through 03-12 with empirical regression coverage:

- **CR-01 closed** (plan 03-08 / `17f82a4`): summary.json scorer pollution eliminated.
- **CR-02 closed** (plan 03-09 / `8c02af9`): ScriptedChatModel singleton identity-dedup eliminated.
- **WR-01..WR-04 closed** (plans 03-08, 03-10, 03-11): `--` collision, silent-pass-on-error, test-helper triplication, dead `seen` vars all addressed.
- **IN-01..IN-05 closed** (plans 03-08, 03-09, 03-10, 03-12): GitHub Actions interpolation, key rebind invisibility, Makefile variable footgun, bool-as-numeric, ambiguous fallback content all addressed.

**The single remaining item** is the user-action baseline matrix run (real API keys + real spend), which the stale-baseline CI lint correctly treats as structurally-sufficient-by-presence. This is the documented closing handoff per plan-03-07's deviation block; it does not block Phase 4 from beginning.

Phase 3 has delivered both readings of the goal:
- **"Measurement instrument shipped"** — VERIFIED (all 10 requirements satisfied; all anti-patterns closed; empirical regression coverage in place).
- **"Three committed baselines exist with numeric content"** — STRUCTURALLY SATISFIED (3 files with full shape, closure_check_confirmed, D-06 anchors); NUMERICALLY PENDING (user-run handoff documented).

---

*Re-verified: 2026-05-22 by gsd-verifier*
*Branch: `gsd/phase-03-eval-harness-extension` @ `4a16eeb`*
*Unit suite: 725 passed (+20 from previous verification's 705)*
*Coverage: 10/10 EVAL-* requirements satisfied (with 1 partial on numeric baseline content — single remaining human_verification item)*
