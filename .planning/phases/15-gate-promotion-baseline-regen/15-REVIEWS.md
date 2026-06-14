---
phase: 15
reviewers: [codex]
reviewed_at: 2026-06-14T22:21:05Z
re_reviewed_at: 2026-06-14T23:14:52Z
rounds: 3
plans_reviewed: [15-01-PLAN.md, 15-02-PLAN.md, 15-03-PLAN.md, 15-04-PLAN.md]
---

# Cross-AI Plan Review — Phase 15

> Reviewer note: only Codex was requested (`--codex`). The `claude` CLI is skipped
> for independence because this review was orchestrated from within Claude Code.
> A first Codex invocation was quota-blocked; a retry succeeded and produced the
> review below.

## Codex Review

**Overall Findings**
The plan set is strong and mostly preserves the Phase 15 honesty constraints: diagnose before retest, separate experiment vs flag-off baseline runs, baseline regeneration last, and no gate promotion unless earned. The main risks are provenance leaks from inherited experiment env vars, a few places where expected results are pre-baked instead of computed, incomplete recovery paths for partial live runs, and final closure language that should remain data-dependent after the A2 retest.

## 15-01-PLAN.md

**Summary**  
Good zero-spend diagnostic plan with the right ordering and a disciplined trivial-fix boundary. It correctly makes `docs/promotion_decision.md` the new record and avoids appending to closed verdict docs. The main weakness is that the action text partially assumes the diagnostic conclusion before reading the data.

**Strengths**
- Correctly runs before live spend, preserving D-15-09.
- Uses existing telemetry fields instead of new diagnostic runs.
- Keeps the D-15-08 fix boundary narrow and explicit.
- Separates diagnostic outcome from ARCH-FUT-01, preventing scope creep.

**Concerns**
- **MEDIUM:** The plan says to “establish the contrast” with expected arrays. That risks confirmation bias if actual run JSONs differ.
- **MEDIUM:** `git status --porcelain eval_reports/` is not a reliable zero-spend check if there were pre-existing untracked or modified eval artifacts.
- **LOW:** `files_modified` lists `app/agent/graph.py` but not tests, even though a shipped fix requires a regression test.
- **LOW:** It focuses heavily on the A2 run dir; if the claim is “across every arm,” the doc should either inspect Phase 14 artifacts too or narrow the claim.

**Suggestions**
- Make the diagnostic explicitly data-driven: “tabulate all runs and report actual values, including contradictions.”
- Capture an eval_reports before/after snapshot for the zero-spend assertion.
- Add test files to optional `files_modified` if a graph fix ships.
- Require the doc to name exactly which run dirs were inspected and which were not.

**Risk Assessment: MEDIUM**  
The implementation risk is low, but provenance risk is medium because the plan currently nudges the executor toward a predetermined finding.

## 15-02-PLAN.md

**Summary**  
This is the core measurement plan and it correctly separates the A2 experiment from the flag-off baseline source. The human checkpoint, smoke-before-spend pattern, and anchor non-regression checks are well placed. The largest issue is that “all flags unset” is stated but not mechanically enforced in the commands.

**Strengths**
- Preserves the two-run discipline: experiment run is never used for baselines.
- Uses smoke runs with `arm_flags` verification before n=5 spend.
- Explicitly checks forced-path telemetry, not just commit rate.
- Treats A2 as measured input, not a hard phase gate.

**Concerns**
- **HIGH:** Run #2 can inherit `FORCED_COMMIT_STEP` or other experiment env vars from the shell. `make eval-matrix-arm RUNS=5` does not itself prove flags are unset.
- **MEDIUM:** Run-cap accounting is ambiguous. Two n=1 smokes plus two n=5 runs are four live matrix invocations, while the plan describes only two runs against the cap.
- **MEDIUM:** Partial run handling is under-specified. A partial cannot become a baseline source, but the retry/stop path within the <=4-run cap is not clear.
- **MEDIUM:** “A2 retest is only meaningful if the forced path fired” needs nuance: if `all_slots_viable` never becomes true on refinement, non-fire is itself a meaningful result.
- **LOW:** “Median holds >= prior baseline” should name the exact metric and threshold per scenario or family.

**Suggestions**
- Use explicit clean-env commands for Run #2, e.g. unset every experiment flag in the invocation.
- For Run #1, explicitly unset all non-A2 flags while setting only `FORCED_COMMIT_STEP=6`.
- Define cap accounting: whether smokes count as live matrix runs or only full n=5 runs.
- Add a partial-run decision tree: retry within cap, mark blocked, or record partial and skip regen.
- Define anchor non-regression using the same metric `check_eval_gates.py` enforces.

**Risk Assessment: HIGH**  
The measurement design is sound, but a single leaked env var would invalidate the baseline provenance and gate decisions.

## 15-03-PLAN.md

**Summary**  
This plan correctly translates measurements into durable artifacts and is the strongest on D-15-04/05/06/07 provenance. Baseline regeneration through `write_baselines.py` and the latency decomposition are well scoped. The main issue is some confusing gpt-5 gate rationale wording if A2 clears but flag-off does or does not support enforcement.

**Strengths**
- Correctly uses Run #2 flag-off `summary.json` as the only baseline source.
- Keeps experiment data out of baseline generation.
- Avoids meaningless 0.0 enforced gates.
- Requires both gate checks and baseline freshness checks.
- Latency reporting is honest and decomposed by LLM/tool time.

**Concerns**
- **MEDIUM:** The gpt-5 enforcement branch says the forced path is the mechanism even though the hard gate must be sourced from flag-off CI behavior. That can blur D-15-07 provenance.
- **MEDIUM:** `check_baselines_fresh.py origin/main` assumes `origin/main` is the right comparison point and locally available.
- **MEDIUM:** Latency aggregation should reconcile `latency_seconds` against summed `step_telemetry`; otherwise overhead may be silently lost.
- **LOW:** Gate editing happens before baseline regeneration; acceptable, but the final gate check after regen is the one that matters most.

**Suggestions**
- Rewrite gpt-5 rationale rules: if flag-off earns the floor, enforce on flag-off data; if only A2 earns it, do not enforce and document forced-commit as experimental.
- Include run count, median/p95 or min/max, and step count in the latency report.
- Report telemetry sum plus observed `latency_seconds` so orchestration overhead is visible.
- Confirm the exact baseline freshness base ref before running the check.

**Risk Assessment: MEDIUM**  
The core mechanics are solid; risk is mainly in wording/provenance clarity and final freshness assumptions.

## 15-04-PLAN.md

**Summary**  
The closure plan correctly makes `docs/promotion_decision.md` the milestone audit anchor and protects closed verdict docs. It also handles planning bookkeeping in the established style. The risky part is that it hardcodes “honest null result” language even though the A2 retest outcome is not known until Plan 02 completes.

**Strengths**
- Cross-links immutable verdict docs instead of modifying them.
- Consolidates all Phase 15 outputs into one reviewer-facing record.
- Updates ROADMAP, REQUIREMENTS, and STATE after artifacts are produced.
- Explicitly records anchor ratification, ARCH-FUT-01 deferral, and prod-default non-flip.

**Concerns**
- **HIGH:** Final summary language must be data-dependent. If the A2 retest clears, “no arm cleared INST-05” becomes false or at least needs qualification.
- **MEDIUM:** “Gate numbers are NOT duplicated” conflicts with prior sections that necessarily include measured rates and deltas. Clarify that `eval_gates.yaml` remains source of truth for enforced thresholds.
- **MEDIUM:** Committing straight to main needs a clean-worktree/focused-commit preflight so unrelated changes are not swept in.
- **LOW:** Verification greps are broad; they do not prove PROMO-01/02/03 were updated correctly.

**Suggestions**
- Make closing language conditional: “If A2 did not clear…” vs “If A2 cleared experimentally but not flag-off…”
- Add final preflight: `git status --short`, verify only intended files changed, and confirm verdict docs unchanged.
- Add a final consolidated verification step: gate check, baseline freshness check, and full test status if any code changed in Plan 01.
- Clarify that measured rates may appear in the doc, but enforced gate thresholds live in `configs/eval_gates.yaml`.

**Risk Assessment: MEDIUM**  
Mostly documentation/bookkeeping risk, but the hardcoded null-result language could undermine the phase’s provenance honesty.

---

## Consensus Summary

Only one reviewer (Codex) was invoked, so there is no cross-reviewer agreement to
compute. The synthesis below ranks Codex's findings by severity and impact so the
planner can act on the highest-priority items first.

### Highest-Priority Concerns (HIGH)

1. **Run #2 env-var provenance leak (15-02).** `make eval-matrix-arm RUNS=5` does not
   itself prove experiment flags are unset — an inherited `FORCED_COMMIT_STEP` (or other
   arm flag) from the shell would silently contaminate the flag-off baseline source and
   invalidate D-15-07 provenance. The smoke `arm_flags` check catches it *if read*, but
   the invocation should explicitly clear the environment. Likewise Run #1 should set
   ONLY `FORCED_COMMIT_STEP=6` and unset every other arm flag.
2. **Hardcoded null-result language (15-04).** The closure plan pre-writes "no arm cleared
   INST-05 / honest null result," but that statement is only true if the Plan 02 A2 retest
   does NOT clear 0.6. The closing language must be data-dependent on the Plan 02 outcome.

### Notable Medium Concerns

- **Run-cap accounting (15-02):** two n=1 smokes + two n=5 runs = four live matrix
  invocations, yet the plan frames it as "two runs against the ≤4-run cap." Define
  whether smokes count.
- **Partial-run handling (15-02):** the retry/stop/record-partial decision tree within
  the cap is under-specified.
- **Non-fire is meaningful (15-02):** "A2 only meaningful if forced path fired" needs
  nuance — if `all_slots_viable` never holds on refinement, a non-fire is itself the result.
- **gpt-5 gate-rationale wording (15-03):** the "forced path is the mechanism" rationale
  can blur D-15-07 (hard gate value must come from flag-off CI behavior, not the experiment).
- **`check_baselines_fresh.py origin/main` base ref (15-03):** assumes `origin/main` is the
  right, locally-available comparison point — confirm before running.
- **Latency reconciliation (15-03):** reconcile observed `latency_seconds` against summed
  `step_telemetry` so orchestration overhead isn't silently lost.
- **Straight-to-main preflight (15-04):** bookkeeping commits need a clean-worktree /
  focused-commit preflight so unrelated changes aren't swept in.

### Lower-Priority / Polish (LOW)

- Diagnostic should report ACTUAL telemetry values (including contradictions), not pre-baked
  expected arrays — confirmation-bias guard (15-01).
- Zero-spend check should use a before/after `eval_reports/` snapshot, not just
  `git status --porcelain` (15-01).
- Add test files to optional `files_modified` when a graph fix ships (15-01).
- Verification greps are broad; they don't prove PROMO-01/02/03 were updated correctly (15-04).

### Reviewer Risk Verdicts

| Plan  | Codex Risk |
|-------|------------|
| 15-01 | MEDIUM (provenance/confirmation-bias, not implementation) |
| 15-02 | **HIGH** (single leaked env var invalidates baseline provenance) |
| 15-03 | MEDIUM (wording/provenance clarity + freshness base ref) |
| 15-04 | MEDIUM (hardcoded null-result language is the sharp edge) |

### Divergent Views

None — single reviewer.

---

## Codex Re-Review (Round 2 — after revision 2026-06-14T22:53:53Z)

> Second adversarial pass against the REVISED plans (commit c51fdf3) to confirm the
> round-1 findings were genuinely closed and to catch new problems before live spend.
> Verdict: HIGH-2 (null-result language) closed; **HIGH-1 (env-leak) NOT fully closed** —
> a sixth experiment knob was missed. All claims below were verified against the repo by
> the orchestrator (see round-2 verification note at the end).

## 15-01-PLAN.md

**Resolved?**

- LOW finding: **RESOLVED.**
  Evidence: the plan now requires actual telemetry, not expected arrays: `"reports the ACTUAL per-step telemetry values..."` and `"do NOT pre-fill expected arrays"`.
  It also fixes the zero-spend proof: `"eval_reports/ is gitignored, so git status ... is NOT a valid zero-spend check — use the directory-listing snapshot."`
  The conditional test-file issue is handled by the frontmatter comment requiring `tests/unit/test_graph_forced_commit.py` only if `graph.py` changes.

**New concerns**

- No new blocking concerns. The plan is appropriately zero-spend and diagnose-before-retest.
- LOW: `files_modified` still lists `app/agent/graph.py` and the test file even though the likely path is deferred/no edit. The inline comment makes this tolerable, but executors should not treat that list as permission to touch them casually.

**Risk Assessment:** LOW.

## 15-02-PLAN.md

**Resolved?**

- HIGH-1 env-var provenance leak: **PARTIAL / NOT GENUINELY CLOSED.**
  The plan does fix the five graph/replay flags:
  `env -u FORCED_COMMIT_STEP -u VIABILITY_CONTRACT_ENABLED -u PARALLEL_TOOL_EXECUTION_ENABLED -u REPLAY_MULTI_MESSAGE_ENABLED -u REPLAY_CONTENT_BLOCKS_ENABLED ...`
  But the repo has a sixth arm-relevant knob: `LOW_SIMILARITY_THRESHOLD_OVERRIDE`, recorded in `arm_flags` as `viability_threshold_override`. The revised plan even mentions this key in `scripts/eval_agent.py`, but neither unsets it nor requires the smoke dict to show `viability_threshold_override: null`. This can contaminate both Run #1 and especially the Run #2 baseline source.

- MEDIUM run-cap / partial-run / non-fire handling: **RESOLVED.**
  Evidence: `"RUN-CAP ACCOUNTING..."`, `"PARTIAL-RUN DECISION PATH..."`, and `"A NON-FIRE is a valid, meaningful result..."`.
  This is much better: a forced-path non-fire is now interpreted as evidence about `all_slots_viable`, not as a void retest.

**New concerns**

- HIGH: add `-u LOW_SIMILARITY_THRESHOLD_OVERRIDE` to both Run #1 and Run #2 commands, and require the smoke `arm_flags` dict to show `"viability_threshold_override": null`.
- MEDIUM: the live commands omit `APP_ENV=eval`, but `scripts/eval_matrix.py` enforces it for real-provider runs. As written, the commands fail unless the operator has already exported it. Make the command explicit: `APP_ENV=eval env -u ... make eval-matrix-arm RUNS=...`.
- MEDIUM: `make probe-providers` probes OpenAI, DeepSeek, Anthropic, and Gemini, but `user_setup` only requires OpenAI and DeepSeek and the phase says Anthropic/Gemini are deferred/no top-up. This can block execution or spend outside scope. Use provider-specific probes or update setup/scope honestly.

**Risk Assessment:** HIGH as written, because Run #2 provenance is still not clean by construction.

## 15-03-PLAN.md

**Resolved?**

- MEDIUM gate-value provenance: **RESOLVED, contingent on fixing 15-02.**
  Evidence: `"gpt-5-mini is promoted to enforced ONLY if the flag-off Run #2 committed_itinerary_rate median itself supports the >= 0.6 floor"` and `"never from the FORCED_COMMIT_STEP=6 experiment"`.
- MEDIUM baseline freshness base-ref: **RESOLVED.**
  Evidence: `"run git fetch origin main so origin/main exists locally"` and then `python scripts/check_baselines_fresh.py origin/main`.
- MEDIUM latency reconciliation: **RESOLVED.**
  Evidence: `"RECONCILE the summed step_telemetry ... against the observed per-query latency_seconds"` and explicitly calls out the `~46s` over-budget observation.

**New concerns**

- HIGH inherited: if 15-02 does not unset/assert `LOW_SIMILARITY_THRESHOLD_OVERRIDE`, this plan can regenerate and gate against contaminated Run #2 baselines.
- LOW/MEDIUM: the gpt-5 promotion text says that if flag-off clears, the rationale should still name forced-commit as the “unlock mechanism.” If flag-off cleared with flags off, forced commit was not the mechanism. Keep the rationale data-dependent too: mechanism should come from the measured telemetry, not the expected story.

**Risk Assessment:** MEDIUM after 15-02 is fixed; HIGH if executed after 15-02 as currently written.

## 15-04-PLAN.md

**Resolved?**

- HIGH-2 hardcoded null-result language: **RESOLVED.**
  Evidence: the plan now requires a data-dependent INST-05 verdict with three cases: `(a) A2 did NOT clear`, `(b) A2 cleared experimentally but flag-off did NOT`, `(c) flag-off cleared`.
- MEDIUM straight-to-main preflight: **PARTIAL.**
  Evidence: it now requires `git status --short` and only intended planning files staged. But Task 1 modifies `docs/promotion_decision.md`, while Task 2’s commit preflight only allows the three planning files. The plan does not clearly say when/how the finalized promotion doc itself is committed.
- LOW PROMO row greps: **RESOLVED.**
  Evidence: the regex checks now assert full rows like `| PROMO-01 | Phase 15 | Complete |`.

**New concerns**

- MEDIUM: the immutable-doc verification command prints `FAIL` but still exits 0:
  `grep -q . && echo "FAIL: verdict doc changed" || echo "OK..."`
  That is not a failing automated check. Use `git diff --quiet -- docs/decisiveness_arm_verdicts.md docs/replay_arm_verdicts.md`.
- MEDIUM: clarify commit sequencing for `docs/promotion_decision.md`. It is part of Plan 04, but the bookkeeping commit preflight excludes it.

**Risk Assessment:** MEDIUM.

## Overall Verdict

The two prior HIGH findings are **not both genuinely closed**.

- HIGH-2 is closed: the INST-05 closing verdict is now data-dependent.
- HIGH-1 is still open in substance: the clean-env commands omit `LOW_SIMILARITY_THRESHOLD_OVERRIDE`, even though the harness records it as `arm_flags.viability_threshold_override` and the code treats it as an experiment knob.

I would not execute Plan 02 live spend yet. Minimum fixes before spend:

```bash
APP_ENV=eval env \
  -u LOW_SIMILARITY_THRESHOLD_OVERRIDE \
  -u VIABILITY_CONTRACT_ENABLED \
  -u PARALLEL_TOOL_EXECUTION_ENABLED \
  -u REPLAY_MULTI_MESSAGE_ENABLED \
  -u REPLAY_CONTENT_BLOCKS_ENABLED \
  FORCED_COMMIT_STEP=6 make eval-matrix-arm RUNS=1
```

and for Run #2:

```bash
APP_ENV=eval env \
  -u FORCED_COMMIT_STEP \
  -u LOW_SIMILARITY_THRESHOLD_OVERRIDE \
  -u VIABILITY_CONTRACT_ENABLED \
  -u PARALLEL_TOOL_EXECUTION_ENABLED \
  -u REPLAY_MULTI_MESSAGE_ENABLED \
  -u REPLAY_CONTENT_BLOCKS_ENABLED \
  make eval-matrix-arm RUNS=1
```

Then require smoke inspection to show `viability_threshold_override: null`. Also replace `make probe-providers` with probes matching the actual 3-provider arm config, or explicitly add Anthropic/Gemini keys/spend back into scope.

---

## Round-2 Verification (orchestrator-confirmed against repo)

Codex's three substantive claims were checked against the actual source:

1. **HIGH — `LOW_SIMILARITY_THRESHOLD_OVERRIDE` is a real sixth experiment knob.** CONFIRMED:
   read in `app/agent/revision.py:29` (default `0.55`, the A1 arm knob) and recorded in
   `arm_flags.viability_threshold_override` at `scripts/eval_agent.py:933`. The round-1
   clean-env commands unset only 5 flags and miss this one — a genuine provenance hole on
   BOTH runs, worst on the Run #2 baseline source. Fix: add `-u LOW_SIMILARITY_THRESHOLD_OVERRIDE`
   to both commands and require the smoke `arm_flags` dict to show `viability_threshold_override: null`.

2. **MEDIUM — `APP_ENV=eval` required for real-provider runs.** CONFIRMED:
   `scripts/eval_matrix.py:827` hard-errors ("APP_ENV=eval required for real-provider matrix
   runs") without it. `env -u` preserves an already-exported `APP_ENV`, so this only bites if
   the operator hasn't exported it — but making the live commands explicit (`APP_ENV=eval env -u ...`)
   removes the footgun.

3. **MEDIUM — `make probe-providers` probes 4 providers, only 2 in scope.** CONFIRMED:
   the `probe-providers` Makefile target (line 161) runs openai + deepseek + **anthropic + gemini**,
   but anthropic/gemini are deferred (D-12-09, no keys/top-up) for this milestone. The plan should
   use provider-specific probes for the 3-model arm (openai + deepseek only — gpt-4o-mini and
   gpt-5-mini share the OpenAI key) rather than the all-four target, or the probe will fail/spend
   out of scope.

4. **MEDIUM (15-04) — immutable-doc check exits 0 even on failure.** CONFIRMED by inspection:
   `git diff --stat ... | grep -q . && echo FAIL || echo OK` always exits 0. Use
   `git diff --quiet -- docs/decisiveness_arm_verdicts.md docs/replay_arm_verdicts.md` so a
   changed verdict doc actually fails the automated check. Also clarify when `docs/promotion_decision.md`
   itself is committed (Task 1 edits it; Task 2's preflight only allows the 3 planning files).

5. **LOW/MEDIUM (15-03) — forced-commit "unlock mechanism" rationale should be data-dependent.**
   If the flag-off Run #2 clears 0.6 with all flags OFF, then forced-commit was NOT the mechanism;
   the rationale must reflect the measured telemetry, not the expected forced-commit story.

**Round-2 verdict:** one new HIGH (env knob) + three verified MEDIUMs + one LOW. A second
targeted replan is warranted before any live spend.

---

## Codex Review (Round 3 — verdict: SAFE TO EXECUTE — 2026-06-14T23:14:52Z)

> Third pass against the round-2 plans (commits cba4785 + 8f9b24b). Confirms all six
> round-2 fixes (A–F) are genuinely CLOSED. Only three LOW, non-blocking cleanup items
> remain, all explicitly after the money-sensitive path. Overall verdict: SAFE TO EXECUTE.

## 15-01-PLAN.md

**Round-2 Fix Status**

No listed Round-2 fix directly targets 15-01. The diagnose-before-retest ordering is solid: the objective says Plan 01 “**MUST land before any live run** so the A2 retest (Plan 02) measures the FINAL post-fix code.”

**Remaining Or New Concerns**

None blocking. The plan is appropriately zero-spend and requires a before/after `eval_reports/` directory snapshot rather than relying on git status.

**Risk Assessment:** LOW

## 15-02-PLAN.md

**Round-2 Fix Status**

**FIX A: CLOSED.** The clean env now names all six knobs. Evidence: Run #1 uses “`APP_ENV=eval env -u VIABILITY_CONTRACT_ENABLED ... -u LOW_SIMILARITY_THRESHOLD_OVERRIDE FORCED_COMMIT_STEP=6`,” and Run #2 uses “`env -u FORCED_COMMIT_STEP ... -u LOW_SIMILARITY_THRESHOLD_OVERRIDE`.” The smoke requirement also explicitly requires `viability_threshold_override: null` on both runs.

**FIX B: CLOSED.** All four matrix invocations are explicitly `APP_ENV=eval`: both Run #1 smoke/full and Run #2 smoke/full commands include the prefix.

**FIX C: CLOSED.** The out-of-scope probe is removed. Evidence: “`poetry run python scripts/probe_provider_capture.py --provider openai` and `--provider deepseek`,” plus “NOT `make probe-providers`.”

**Remaining Or New Concerns**

LOW: the smoke inspection references a singular `queries[0].deterministic.arm_flags`. Since env flags are process-global this is probably fine, but inspecting all smoke cell JSONs would be stricter before live spend. This is not blocking because the command construction itself is now correct.

**Risk Assessment:** LOW. The previous HIGH env-leak risk is genuinely closed.

## 15-03-PLAN.md

**Round-2 Fix Status**

**FIX F: CLOSED.** The gpt-5 promotion logic is now data-dependent and flag-off sourced. Evidence: “gpt-5-mini is promoted to enforced ONLY if the flag-off Run #2 committed_itinerary_rate median itself supports the >= 0.6 floor,” and “if flag-off cleared with all experiment flags OFF, the mechanism is ... model-initiated commits — NOT forced-commit.”

**Remaining Or New Concerns**

LOW: Task 1 requires `make eval-gates-check-baselines` immediately after editing `configs/eval_gates.yaml`, before Task 2 regenerates baselines. If Run #2 unexpectedly earns a gpt-5 active gate, old committed baselines could fail the check before the fresh Run #2 baselines are written. Minimum cleanup: make the Task 1 check advisory until Task 2, or move the required exit-0 gate check after `write_baselines.py`.

**Risk Assessment:** LOW to LOW-MED. This is a post-spend execution-order rough edge, not a provenance or live-run safety issue.

## 15-04-PLAN.md

**Round-2 Fix Status**

**FIX D: CLOSED.** The immutable-doc check now uses a real exit code: “`git diff --quiet -- docs/decisiveness_arm_verdicts.md docs/replay_arm_verdicts.md`,” with the plan explicitly noting exit 0 unchanged / exit 1 changed.

**FIX E: CLOSED.** The commit scopes are now non-overlapping. Evidence: “docs/promotion_decision.md is committed as a normal tracked-file change in/after Task 1,” and Task 2 is “scoped ONLY to .planning/ROADMAP.md + .planning/REQUIREMENTS.md + .planning/STATE.md.”

**Remaining Or New Concerns**

LOW: `git status --short` is a broad working-tree view, not a precise staged-files assertion. The intent is clear, but a stricter preflight would add `git diff --cached --name-only` and confirm exactly the three planning files are staged.

**Risk Assessment:** LOW

## Overall Verdict

All prior HIGH/MED findings are genuinely closed. The Plan 02 live-spend path now has explicit `APP_ENV=eval`, complete six-flag env hygiene including `LOW_SIMILARITY_THRESHOLD_OVERRIDE`, smoke inspection of `viability_threshold_override: null`, scoped provider probes, and a valid <=4 full-run cap.

The remaining concerns are non-blocking cleanup items after the money-sensitive path.

**SAFE TO EXECUTE**
