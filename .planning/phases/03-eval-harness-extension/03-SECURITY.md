---
phase: 03
slug: eval-harness-extension
status: secured
threats_open: 0
asvs_level: 1
created: 2026-05-22
---

# Phase 3 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> All 13 plan-time threats verified against the implemented code on branch
> `gsd/phase-03-eval-harness-extension` @ `4a16eeb` (post REVIEW-FIX).

---

## Trust Boundaries

The eval-harness extension introduced seven concrete trust boundaries across
plans 03-08 through 03-12. Consolidated by data-flow direction:

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Cell-JSON → aggregator | Per-cell `*.json` files in `eval_reports/{ts}/` are scanned by `aggregate_cell_jsons` to produce `summary.json` (the Phase 4-6 baseline-diff target). | Untrusted scorer name strings + numeric `aggregate.*_mean` values (potentially `bool`-as-numeric, polluting non-scorer keys, or unparseable cell filenames). |
| Override-CLI → summary.json | `--llm-provider-override` on `scripts/eval_matrix.py` rebinds `MatrixEntry.provider` at runtime and is recorded in `summary.json` for downstream operator visibility. | CLI-supplied provider string; values containing `--` collide with the cell-filename separator. |
| GitHub Actions context → shell | `lint-baselines` step reads `github.event.pull_request.base.sha` and forwards it to `scripts/check_baselines_fresh.py`. | GitHub-computed 40-char hex SHA via env block, never via `${{ }}` interpolation into a `run:` string. |
| Subprocess → Python (`_run_git`) | `scripts/check_baselines_fresh.py` shells out to `git diff` / `git log` and parses stdout. | git rc + stderr; missing-binary `FileNotFoundError`; empty-string BASE_SHA. |
| CLI positional → `_resolve_base` | `check_baselines_fresh.py` accepts an optional positional BASE_SHA / `--merge-base` flag. | Possibly empty-string from a malformed workflow env. |
| Test-helper → consumer tests | `tests/_helpers/scripted_llm.py` is imported by `tests/unit/test_chat_functional.py`, `test_eval_agent.py`, and `test_helpers_scripted_llm.py` (DRY hoist of 2 near-identical classes). | `ScriptedLLM` exhaustion behavior (loud `IndexError`); `RecordingScriptedLLM.seen` per-instance list. |
| Makefile variable → CLI flag | `make eval-agent` plumbs `QUERIES` → `--max-queries`; `make eval-matrix` plumbs `RUNS` → `--runs`. Old habit `make eval-agent RUNS=99` is ignored, not silently consumed. | Operator-supplied integer; previously shared `RUNS` had split semantics across the two targets. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-03-08-01 | Tampering | `scripts/eval_matrix.py::_scorer_means_from_cell` | mitigate | Whitelist via `if scorer_name in CRITIQUE_THRESHOLDS:` — `scripts/eval_matrix.py:168`; import at `:44`. The 6 polluting non-scorer `_mean` keys (tool_calls, results, contexts, revision_hints, committed_stops, answer_retrieved_place_coverage) are rejected. | closed |
| T-03-08-02 | Information Disclosure | `app/eval/config.py::MatrixEntry` + `scripts/eval_matrix.py` | mitigate | `reject_double_dash` `field_validator(mode="after")` at `app/eval/config.py:210-227`; parse-time argparse `type=_validate_override` at `scripts/eval_matrix.py:393-412`; WARN log on unparseable cell filename at `scripts/eval_matrix.py:237-242`. | closed |
| T-03-08-03 | Tampering | `scripts/eval_matrix.py::_scorer_means_from_cell` | accept | Defensive `isinstance(value, bool)` exclusion at `scripts/eval_matrix.py:164` already in place; no production path emits bools today. See Accepted Risks Log. | closed |
| T-03-08-04 | Repudiation | `scripts/eval_matrix.py::aggregate_cell_jsons` + `run_matrix` | mitigate | `overridden_to: llm_provider_override` written into summary at `scripts/eval_matrix.py:268-269`; INFO log of rebind at `scripts/eval_matrix.py:351-354`. | closed |
| T-03-09-01 | Denial of Service | `app/llm_factory.py::ScriptedChatModel._generate` | mitigate | Fresh `AIMessage(...)` constructed inline at `app/llm_factory.py:147-153`; module-level `_DEFAULT_SCRIPTED_FALLBACK` removed (grep returns 0 matches). LangGraph `add_messages` identity-dedupe no longer absorbs the fallback. | closed |
| T-03-09-02 | Repudiation | `app/llm_factory.py::ScriptedChatModel._generate` fallback content | mitigate | Self-documenting marker `[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py.` at `app/llm_factory.py:149-150`. | closed |
| T-03-10-01 | Elevation of Privilege | `.github/workflows/ci.yml::lint-baselines` | mitigate | `env:` block at `ci.yml:154-155` sets `BASE_SHA: ${{ github.event.pull_request.base.sha }}`; `run:` consumes `"$BASE_SHA"` at `ci.yml:156`. No `${{ }}` interpolation inside any `run:` string in the lint-baselines job. | closed |
| T-03-10-02 | Denial of Service / Repudiation | `scripts/check_baselines_fresh.py` | mitigate | `_run_git` raises `RuntimeError` on rc != 0 (`scripts/check_baselines_fresh.py:89-95`) and on `FileNotFoundError` (`:81-87`); `_resolve_base` rejects empty-string BASE_SHA (`:206-211`) and empty `--merge-base` (`:198-203`); `main()` translates `RuntimeError` to rc=2 distinct from rc=1 (`:232-234`). | closed |
| T-03-10-03 | Tampering (shell-injection theoretical) | `scripts/check_baselines_fresh.py` BASE_SHA flow | accept | `base.sha` is GitHub-computed 40-char hex; double-quoted `"$BASE_SHA"` in the `run:` line prevents word-splitting/metacharacter expansion. See Accepted Risks Log. | closed |
| T-03-11-01 | Tampering (DRY drift) | `tests/_helpers/scripted_llm.py` | mitigate | `ScriptedLLM` (`:34-67`) + `RecordingScriptedLLM` (`:70-93`) hoisted; consumer files import via `from tests._helpers.scripted_llm import …` in `test_chat_functional.py:19`, `test_eval_agent.py:567`, `test_helpers_scripted_llm.py:18`. Production `ScriptedChatModel` intentionally NOT folded in (documented at `:8-12`). | closed |
| T-03-11-02 | Repudiation (dead variable) | `tests/_helpers/scripted_llm.py::RecordingScriptedLLM.seen` | mitigate | `seen: list[list[BaseMessage]] = Field(default_factory=list)` at `tests/_helpers/scripted_llm.py:81`; 5 dead outer-scope `seen` vars removed from `tests/unit/test_eval_agent.py`. | closed |
| T-03-11-03 | Denial of Service (helper exhaustion) | `tests/_helpers/scripted_llm.py::ScriptedLLM._generate` | mitigate | `IndexError` raised on empty `scripted` list at `tests/_helpers/scripted_llm.py:56-60`. Stricter than production `ScriptedChatModel` (which retains the `[SCRIPTED CI MODE]` fallback per T-03-09-01) — intentional asymmetry documented at `:14-17`. | closed |
| T-03-12-01 | Repudiation | `Makefile` `eval-agent` and `eval-matrix` targets | mitigate | `QUERIES ?= 1` plumbs `--max-queries $(QUERIES)` on eval-agent (`Makefile:108`, `:118`); `RUNS ?= 1` plumbs `--runs $(RUNS)` on eval-matrix (`Makefile:107`, `:124`). Old habit `make -n eval-agent RUNS=99` ignored — UAT Test 6 confirmed `--max-queries 1` is plumbed (the QUERIES default), not `--max-queries 99`. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-03-01 | T-03-08-03 | `bool` is a subclass of `int` in Python, so a stray `True`/`False` value in `aggregate.*_mean` would otherwise score as 1.0/0.0. No production code path (`app/agent/critique/checks.py` scorers, `scripts/eval_agent.py` aggregation) emits boolean values into `aggregate.*_mean`; all known scorers emit `float` in `[0.0, 1.0]`. The defensive `isinstance(value, bool)` guard at `scripts/eval_matrix.py:164` is retained as belt-and-suspenders against a future scorer regressing to bool, but the residual risk that a bool slips past the guard is accepted as negligible given the whitelist also requires the key name to be registered in `CRITIQUE_THRESHOLDS`. | pjnhek (orchestrator gsd-secure-phase 3) | 2026-05-22 |
| AR-03-02 | T-03-10-03 | The `BASE_SHA` value comes from `github.event.pull_request.base.sha`, which is a GitHub-computed 40-character lowercase hexadecimal commit SHA — by GitHub's API contract it cannot contain shell metacharacters. The `"$BASE_SHA"` double-quoting in `.github/workflows/ci.yml:156` provides defense-in-depth against any future GitHub API change. The residual theoretical shell-injection risk (GitHub returns a maliciously crafted base.sha) is accepted as outside the threat surface this phase commits to mitigate. | pjnhek (orchestrator gsd-secure-phase 3) | 2026-05-22 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-22 | 13 | 13 | 0 | gsd-security-auditor (orchestrator gsd-secure-phase 3) |

---

## Verification Notes

This audit anchors on three upstream artifacts, all converging on a green
state at HEAD `4a16eeb`:

1. **`03-REVIEW.md`** — adversarial re-review at `2026-05-22T17:22:10Z`
   surfaced 11 findings (1 critical, 6 warnings, 4 info) including
   `WR-01..WR-06` and `IN-01..IN-05`. All 11 findings are now closed per
   `03-REVIEW-FIX.md` (status `all_fixed`, fix iteration 1, commits
   `586288a`, `6b787c1`, `f74e587`, `b9fbd14`, `51c3482`, `f11813d`,
   `1ac998c`, `a83de83`+`c49bfce`, `d61da5d`, `170ab27`, `123716e`).

2. **`03-VERIFICATION.md`** — `status: human_needed`, score 10/10
   must-haves verified. The single remaining `human_verification` item is
   the live baseline matrix run (`APP_ENV=eval make eval-matrix RUNS=3`,
   ~15 min wall time, real API spend) — explicitly **out of security
   scope**: it concerns numeric content of `configs/eval_baselines/*.json`,
   not a threat mitigation. The structural lint gate is live; Phase 4 can
   begin in parallel.

3. **`03-UAT.md`** — 10/10 tests passed, including UAT-4 (`--llm-provider-override
   foo--bar` rejected with actionable argparse error citing `'--' is
   reserved`), UAT-5 (`check_baselines_fresh.py ""` loud-fails rc=2), and
   UAT-8 (CI scripted-mode run produces clean `summary.json` with zero
   polluter keys and `overridden_to: "scripted"` recorded).

The 13 threats in this register were authored at plan time across the five
gap-closure plans (03-08 through 03-12); plans 03-01..03-07 are pre-formal
threat-modeling and have no `<threat_model>` blocks. The single remaining
human-action item (live baseline matrix) is **out of security scope** — it
does not block the security verdict.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log (AR-03-01, AR-03-02)
- [x] `threats_open: 0` confirmed
- [x] `status: secured` set in frontmatter

**Approval:** verified 2026-05-22
