---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 09
type: execute
wave: 1
depends_on: []
gap_closure: true
autonomous: true
requirements: [BASE-01, BASE-03]
files_modified:
  - configs/eval_baselines/omakase_mission_open_ended.json
  - configs/eval_baselines/refinement_cheaper.json
  - configs/eval_gates.yaml
  - eval_reports/   # new live-run report dirs (timestamped; committed with baselines)

must_haves:
  truths:
    - "configs/eval_baselines/*.json contain no category_compliance medians produced via the CR-01 error path — every zero-stop cell is re-measured under the now-fixed abstain semantics"
    - "make eval-gates-check-baselines exits 0 (only the documented non-blocking gpt-5-mini aspirational miss prints)"
    - "Every committed eval_gates.yaml rationale line citing a category_compliance or zero-stop-derived number is accurate against the re-measured baselines, or is corrected with a D-11 ID"
    - "Full make test passes green; the test_eval_matrix.py parity test and the CR-01/CR-02 integration tests pass against the regenerated baselines"
  artifacts:
    - path: "configs/eval_baselines/omakase_mission_open_ended.json"
      provides: "Re-measured omakase cells for gpt-5-mini, deepseek-chat, deepseek-reasoner with abstain-clean category_compliance"
      contains: "generated_by"
    - path: "configs/eval_baselines/refinement_cheaper.json"
      provides: "Re-measured refinement cells for gpt-4o-mini, gpt-5-mini, deepseek-chat, deepseek-reasoner with abstain-clean category_compliance"
      contains: "generated_by"
    - path: "configs/eval_gates.yaml"
      provides: "Gate rationales re-verified against fresh n=5 data"
      contains: "committed_itinerary_rate"
  key_links:
    - from: "scripts/eval_matrix.py (temp scoped matrix-config)"
      to: "scripts/write_baselines.py"
      via: "summary.json scorers block per accepted cell"
      pattern: "write_baselines"
    - from: "configs/eval_baselines/*.json"
      to: "make eval-gates-check-baselines"
      via: "check_eval_gates.py --baselines-mode"
      pattern: "baselines-mode"
---

<objective>
Close the three remaining Phase 11 verification gaps by re-measuring ONLY the
category_compliance-contaminated baseline cells live at n=5 under the now-fixed
(CR-01) abstain semantics, then re-verifying gate rationales and running closing
verification.

The CR-01 and CR-02 code fixes are ALREADY committed (fbd1174, 3d0da9e, plus
WR-01..05 in fa518ce/b3c54cf/b89cc69/c5463c9/054a20c) and the full suite is green
(1204 passed). This plan does NOT touch scorer/gate/checker code. It only replaces
the contaminated empirical numbers those fixes made it possible to measure honestly.

The contamination is precise and proven from per-run JSONs (`check_err=1` co-occurring
with `committed_itinerary_rate=0.0` marks a zero-stop run whose category_compliance was
baked to 0.0 via the OLD float(None) TypeError path). The affected cell set is bounded
(see Task 1 table). The gpt-4o-mini omakase anchor (5/5 committed, zero check errors) is
genuinely clean and is NOT re-run. Anthropic (billing-exhausted) and gemini (D-11-11
quota deferral) stay documented deferrals and are NOT re-run.

Purpose: BASE-01 demands "honest measurements." CR-01 replaced fail-open 1.0 artifacts
with error-path 0.0 artifacts for the same zero-stop cells — both are wrong. This plan
delivers real measurements (abstain → cell drops out of aggregation) for those cells.
Output: regenerated omakase + refinement baseline cells, re-verified gates, green suite.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-VERIFICATION.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-REVIEW-FIX.md
@docs/baseline_regen.md
@configs/eval_matrix.yaml
@configs/eval_matrix_refinement.yaml
@configs/eval_gates.yaml
@scripts/write_baselines.py
@configs/eval_baselines/omakase_mission_open_ended.json
@configs/eval_baselines/refinement_cheaper.json
</context>

<live_infra_preconditions>
Verified at planning time (re-verify before Task 2):
- cloud-sql-proxy LISTENING on 127.0.0.1:5433 (confirmed at plan time; re-check with
  `lsof -iTCP:5433 -sTCP:LISTEN -P` before each live run).
- OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI/GOOGLE_API_KEY live.
- ANTHROPIC_API_KEY has NO credits (commit 2941807 demoted anthropic to logged-not-gated).
  Anthropic cells MUST NOT be in any temp matrix-config — they would error and burn time.
- D-11-14: execution stays SEQUENTIAL. No parallel/ProcessPool runs. Use RUNS=5, APP_ENV=eval.
- Embeddings sanity: the 21-14-30Z disaster was an exhausted OpenAI embeddings quota
  poisoning every cell. Run `make probe-providers` (or a single semantic_search) and
  confirm no 429s before the matrix run.
</live_infra_preconditions>

<tasks>

<task type="auto">
  <name>Task 1: Confirm the exact contaminated-cell set from per-run JSONs (no live calls)</name>
  <files>(read-only verification; no files modified)</files>
  <read_first>
    - eval_reports/2026-06-11T19-09-10Z/summary.json (omakase regen run)
    - eval_reports/2026-06-11T19-38-23Z/summary.json (refinement regen run)
    - eval_reports/2026-06-11T19-09-10Z/*--run-*.json (omakase per-run)
    - eval_reports/2026-06-11T19-38-23Z/*--run-*.json (refinement per-run)
    - configs/eval_baselines/omakase_mission_open_ended.json (committed contaminated cells)
    - configs/eval_baselines/refinement_cheaper.json (committed contaminated cells)
  </read_first>
  <action>
    Re-derive the contaminated-cell set programmatically rather than trusting the table
    below verbatim. For each per-run JSON, a run is contaminated iff
    `aggregate.committed_itinerary_rate == 0.0` AND `aggregate.check_error_count >= 1`
    (zero-stop run whose category_compliance was computed via the OLD float(None) path).
    A cell needs regen if ANY of its 5 runs is contaminated. Distinguish from infra
    errors: runs with `aggregate.n_errored >= 1` are infra failures (anthropic billing),
    NOT CR-01 contamination — those cells are deferrals, not regen targets.

    The planning-verified target set (confirm, do not assume):

    | Scenario | Provider | Contaminated runs | Action |
    |----------|----------|-------------------|--------|
    | omakase | openai/gpt-4o-mini | 0/5 (all cir=1.0, check_err=0) | KEEP — clean anchor, do NOT re-run |
    | omakase | openai/gpt-5-mini | runs 0,1 (zero-stop) | REGEN |
    | omakase | deepseek/deepseek-chat | runs 0,2,3,4 | REGEN |
    | omakase | deepseek/deepseek-reasoner | runs 0-4 (all) | REGEN |
    | omakase | anthropic/claude-sonnet-4-6 | n_errored=5 (billing) | DEFER (D-11-20) — do NOT re-run |
    | refinement | openai/gpt-4o-mini | run 0 (zero-stop) | REGEN (logged-not-gated, but BASE-01 honesty applies) |
    | refinement | openai/gpt-5-mini | runs 0-4 (all) | REGEN |
    | refinement | deepseek/deepseek-chat | runs 2,3,4 contaminated; run 1 infra-errored → cell stale Phase-9 (generated_at=null) | REGEN (clean n=5) |
    | refinement | deepseek/deepseek-reasoner | runs 0-4 (all) | REGEN |
    | refinement | anthropic/claude-sonnet-4-6 | n_errored=5 (billing) | DEFER (D-11-20) — do NOT re-run |
    | refinement | gemini/gemini-3.1-pro-preview | only run-0 present; 1-4 absent | DEFER (D-11-11) — opportunistic only (see Task 2) |

    Write the confirmed regen provider sets to two scratch lists:
      - OMAKASE_REGEN = {openai/gpt-5-mini, deepseek/deepseek-chat, deepseek/deepseek-reasoner}
      - REFINE_REGEN  = {openai/gpt-4o-mini, openai/gpt-5-mini, deepseek/deepseek-chat, deepseek/deepseek-reasoner}
    If the programmatic derivation disagrees with this table, the per-run data is the source
    of truth — record the discrepancy in the SUMMARY and use the data-derived set.
  </action>
  <acceptance_criteria>
    - A reproducible command (jq/python over the two eval_reports dirs) prints, for every
      cell, the count of runs where `committed_itinerary_rate==0.0 AND check_error_count>=1`.
    - The derived OMAKASE_REGEN and REFINE_REGEN provider sets are recorded in the SUMMARY.
    - omakase/openai/gpt-4o-mini is confirmed to have ZERO contaminated runs (not in OMAKASE_REGEN).
    - anthropic cells in both scenarios are confirmed to be infra-errored (n_errored>=1), NOT
      contaminated, and are therefore excluded from both regen sets.
  </acceptance_criteria>
  <verify>
    <automated>cd "$REPO" && python -c "import json,glob; [print(p.split('/')[-1], json.load(open(p))['aggregate'].get('committed_itinerary_rate'), json.load(open(p))['aggregate'].get('check_error_count'), json.load(open(p))['aggregate'].get('n_errored')) for p in sorted(glob.glob('eval_reports/2026-06-11T19-09-10Z/*--run-*.json'))]" | head</automated>
  </verify>
  <done>OMAKASE_REGEN and REFINE_REGEN provider sets are confirmed from per-run data and recorded; clean anchor and deferred cells are excluded.</done>
</task>

<task type="auto">
  <name>Task 2: Live re-measure the contaminated cells at n=5 via temporary scoped matrix-configs (SEQUENTIAL, DB-up)</name>
  <files>eval_reports/&lt;new-timestamp&gt;/ (live-run output; committed in Task 4)</files>
  <read_first>
    - docs/baseline_regen.md (the BASE-01 runbook — follow its precondition + step order)
    - configs/eval_matrix.yaml (omakase entries + scenarios shape to clone)
    - configs/eval_matrix_refinement.yaml (refinement entries + REFINEMENT_STRUCTURED_PLAN_ENABLED env block to clone)
    - Makefile (eval-matrix / eval-matrix-refinement targets — note both hardcode --matrix-config, so temp configs need a direct scripts/eval_matrix.py invocation)
  </read_first>
  <action>
    Per the user-approved scoped-regen guidance: eval_matrix.py has NO provider filter,
    only --matrix-config. To avoid re-spending on the clean gpt-4o-mini omakase anchor and
    the deferred anthropic/gemini cells, build two TEMPORARY scoped matrix-config YAMLs in
    /tmp (NEVER committed):

    1. /tmp/eval_matrix_regen_omakase.yaml — clone configs/eval_matrix.yaml's structure but
       list ONLY the OMAKASE_REGEN providers as entries (openai/gpt-5-mini,
       deepseek/deepseek-chat, deepseek/deepseek-reasoner), flag-OFF (no env block), scenarios
       = [omakase_mission_open_ended]. Do NOT include openai/gpt-4o-mini (clean) or
       anthropic/claude-sonnet-4-6 (billing).
    2. /tmp/eval_matrix_regen_refine.yaml — clone configs/eval_matrix_refinement.yaml's
       structure but list ONLY the REFINE_REGEN providers, each with the
       `env: { REFINEMENT_STRUCTURED_PLAN_ENABLED: "true" }` block exactly as the source file
       sets it (flag-ON for refinement). Do NOT include anthropic (billing) or gemini (defer).
       OPTIONAL D-11-11 opportunistic gemini: you MAY add gemini/gemini-3.1-pro-preview to
       this temp config for a full n=5 attempt; if any gemini run errors, it stays a documented
       D-11-11 deferral (already in _DEFERRED_BASELINE_CELLS) and NEVER blocks this plan.

    Before running: verify cloud-sql-proxy on :5433 (lsof), run `make probe-providers` (or a
    single semantic_search) to confirm no embeddings 429s (the 21-14-30Z poison-mode guard).

    Run each temp matrix SEQUENTIALLY (D-11-14), identical measurement conditions to the
    original regen — APP_ENV=eval, RUNS=5, temp/thinking policies as-shipped in
    app/llm_factory.py (no tuning, per feedback_temp1_reasoning_off_all_models):
      APP_ENV=eval poetry run python scripts/eval_matrix.py --matrix-config /tmp/eval_matrix_regen_omakase.yaml --runs 5
      APP_ENV=eval poetry run python scripts/eval_matrix.py --matrix-config /tmp/eval_matrix_regen_refine.yaml --runs 5
    (or via the same harness the Makefile uses; the direct invocation is required because the
    Make targets hardcode the canonical config path.)

    STOP-and-checkpoint rule (per planning guidance): If the omakase gpt-4o-mini anchor were
    re-run and its committed_itinerary_rate dropped below 0.8 — it is NOT being re-run here, so
    this should not recur. If ANY gated cell (none in this scoped set — gpt-4o-mini omakase
    excluded, gpt-5-mini is aspirational/non-blocking) fails repeatedly for non-billing reasons,
    or if multiple cells error with 429/RateLimitError (embeddings poison signature), STOP, do
    NOT write baselines, and surface the run for human review. A documented anthropic/gemini
    billing/quota error is NEVER a blocker.

    Record each new eval_reports/<timestamp>/summary.json path for Task 3.
  </action>
  <acceptance_criteria>
    - Two temp matrix-config YAMLs exist under /tmp (NOT under configs/, NOT git-added) listing
      only the confirmed regen providers; omakase temp config has no env block, refinement temp
      config sets REFINEMENT_STRUCTURED_PLAN_ENABLED=true on every entry.
    - cloud-sql-proxy :5433 confirmed LISTENING and an embeddings sanity check returned non-429
      before the runs.
    - Each scoped matrix run produced a new eval_reports/<timestamp>/summary.json with
      n_errored==0 and n_scored==5 for every NON-deferred regen cell (gated families clean).
    - No 429/RateLimitError storm; if one occurred, the run was aborted and NOT used downstream.
  </acceptance_criteria>
  <verify>
    <automated>cd "$REPO" && test -f /tmp/eval_matrix_regen_omakase.yaml && test -f /tmp/eval_matrix_regen_refine.yaml && ! git ls-files --error-unmatch /tmp/eval_matrix_regen_omakase.yaml 2>/dev/null && echo "temp configs present and untracked"</automated>
  </verify>
  <done>Both scoped matrices ran sequentially under identical conditions; new summary.json files show n_scored=5/n_errored=0 for all non-deferred regen cells.</done>
</task>

<task type="auto">
  <name>Task 3: Write the re-measured cells into baselines via write_baselines.py + re-verify gate rationales</name>
  <files>configs/eval_baselines/omakase_mission_open_ended.json, configs/eval_baselines/refinement_cheaper.json, configs/eval_gates.yaml</files>
  <read_first>
    - scripts/write_baselines.py (confirm: starts updated_providers from prior file, so a
      partial-matrix summary updates ONLY the cells it contains and carries everything else
      forward verbatim, including _observations; refuses n_scored<n_requested and
      baseline_eligible=false)
    - configs/eval_baselines/omakase_mission_open_ended.json (current contaminated cells)
    - configs/eval_baselines/refinement_cheaper.json (current contaminated + stale deepseek-chat cell)
    - configs/eval_gates.yaml (rationale prose to re-verify; lines 23, 35, 49, 56, 62)
  </read_first>
  <action>
    For each new scoped summary.json from Task 2, run the committed writer:
      poetry run python scripts/write_baselines.py eval_reports/<ts-omakase>/summary.json --n-requested 5
      poetry run python scripts/write_baselines.py eval_reports/<ts-refine>/summary.json --n-requested 5
    (or `make write-baselines SUMMARY=... RUNS=5`). Because write_baselines starts from the
    prior on-disk baseline and only overwrites cells present in the summary, the clean
    omakase gpt-4o-mini cell, the deferred anthropic cells, and any deferred gemini cell stay
    byte-identical. Expect exit 0 if all regen cells scored 5/5; exit 1 only if a cell was
    refused (n_scored<5) — investigate any refusal before proceeding.

    After writing, confirm the contaminated category_compliance medians are GONE: for each
    regenerated zero-stop-heavy cell, category_compliance should now reflect abstain semantics
    — either absent from scorers, or present with n < 5 (only the runs that committed stops
    contribute), NOT an n=5 all-0.0 block. committed_itinerary_rate is unaffected by CR-01 and
    should match the original measurement within stochastic noise.

    Gate re-ratification (D-11-20 / BASE-03) — NARROW scope, confirmed at plan time:
    NO gate keys on category_compliance; all gates key on committed_itinerary_rate, which CR-01
    does NOT contaminate. So gate VALUES and statuses do not change. The work is auditing every
    rationale line in configs/eval_gates.yaml for prose that cites a category_compliance or
    zero-stop-derived number, and correcting any that the fresh data invalidates. In particular:
      - openai/gpt-4o-mini (line ~23): cites omakase median 1.0 — unchanged (anchor not re-run); verify.
      - openai/gpt-5-mini (line ~35): cites omakase commit median 1.0 + refinement commit median 0.0
        — both are committed_itinerary_rate (CR-01-clean); re-confirm against fresh refinement run;
        update the run timestamp reference if the number shifted within noise; keep status aspirational.
      - deepseek/* (lines ~49, ~56): cite omakase commit median 0.0 (decisiveness gap) — re-confirm.
    Every changed line keeps a D-11 ID; add `D-11-09` (gap-closure regen) where a number/timestamp
    is refreshed. If no rationale line referenced contaminated data, record "no rationale edits
    required — gates key on commit-rate only" in the SUMMARY and leave eval_gates.yaml untouched.
  </action>
  <acceptance_criteria>
    - `jq` on omakase deepseek-reasoner category_compliance shows it is NO LONGER an n=5 all-0.0
      block (it abstains: absent, or n<5 reflecting only committed runs).
    - The omakase gpt-4o-mini cell and all anthropic/gemini deferred cells are byte-identical to
      their pre-Task-3 state (write_baselines carry-forward).
    - refinement deepseek-chat cell now has a non-null generated_at and generated_by=write_baselines.py
      (no longer the stale Phase-9 cell), OR is documented as still-refused if it scored <5.
    - Every eval_gates.yaml rationale citing category/zero-stop numbers is accurate vs fresh data
      or corrected with a D-11 ID; status/value fields are unchanged (commit-rate gates unaffected).
  </acceptance_criteria>
  <verify>
    <automated>cd "$REPO" && jq -e '.providers["deepseek/deepseek-reasoner"].scorers.category_compliance.n != 5 or (.providers["deepseek/deepseek-reasoner"].scorers.category_compliance == null)' configs/eval_baselines/omakase_mission_open_ended.json</automated>
  </verify>
  <done>Contaminated cells re-written with abstain-clean category_compliance; deferred/clean cells preserved; gate rationales re-verified.</done>
</task>

<task type="auto">
  <name>Task 4: Closing verification — baselines gate green, parity + integration tests pass, full suite green, commit</name>
  <files>(commits the regenerated baselines + any eval_gates.yaml rationale edits + new eval_reports dirs)</files>
  <read_first>
    - tests/unit/test_eval_matrix.py (parity test + _DEFERRED_BASELINE_CELLS — confirm anthropic/gemini still listed; no new deferrals needed since regen targets are non-deferred)
    - tests/unit/test_eval_agent.py (TestZeroStopAbstainPipeline — the CR-01 integration test must still pass)
    - tests/unit/test_check_eval_gates.py (CR-02 fail-closed tests)
    - Makefile (eval-gates-check-baselines target)
  </read_first>
  <action>
    Run the three closing checks the VERIFICATION report demanded:
    1. `make eval-gates-check-baselines` → must exit 0 (the only allowed stdout miss is the
       documented non-blocking "ASPIRATIONAL miss: openai/gpt-5-mini"). Any hard-gate failure or
       exit !=0 is a STOP.
    2. The parity test: `poetry run pytest tests/unit/test_eval_matrix.py -q` — every committed
       baseline provider cell must map to a matrix entry and vice versa, modulo
       _DEFERRED_BASELINE_CELLS (anthropic omakase, gemini refinement). The regen targets are all
       non-deferred providers that already had cells, so NO _DEFERRED_BASELINE_CELLS edits should
       be needed — if the parity test demands one, investigate before editing.
    3. Full suite: `make test` (full DB-pool-safe run per project_full_suite_db_pool_contamination)
       → 0 failures. Confirms the regenerated numbers don't break any baseline-reading test and the
       CR-01/CR-02 integration tests still hold.

    Then commit per feedback_small_focused_commits (small, single-line messages). Suggested seam:
      - one commit: regenerated baselines + new eval_reports/<ts> dirs (chore(11-09): regen
        contaminated category_compliance cells under fixed abstain semantics)
      - one commit (only if edited): eval_gates.yaml rationale refresh (chore(11-09): refresh gate
        rationales against gap-closure regen data (D-11-09/D-11-20))
    Do NOT git-add the /tmp temp matrix-configs. Do NOT run `gh pr merge` (feedback_user_merges_prs).
  </action>
  <acceptance_criteria>
    - `make eval-gates-check-baselines` exits 0; stdout shows only the non-blocking gpt-5-mini
      aspirational miss.
    - `pytest tests/unit/test_eval_matrix.py` passes (parity holds; no orphan cells).
    - `make test` reports 0 failures.
    - Regenerated baselines + eval_reports committed; /tmp configs are NOT tracked
      (`git ls-files | grep -c eval_matrix_regen` == 0).
  </acceptance_criteria>
  <verify>
    <automated>cd "$REPO" && make eval-gates-check-baselines; rc=$?; [ "$rc" -eq 0 ] && poetry run pytest tests/unit/test_eval_matrix.py tests/unit/test_eval_agent.py::TestZeroStopAbstainPipeline -q && [ "$(git ls-files | grep -c 'eval_matrix_regen')" -eq 0 ] && echo OK</automated>
  </verify>
  <done>Baselines gate exits 0, parity + CR-01 integration tests pass, full suite green, baselines committed, temp configs untracked.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| live provider APIs → eval harness | Untrusted: rate limits, billing exhaustion, transient 5xx can poison a run |
| temp matrix-config (/tmp) → committed baselines | A mis-scoped temp config could overwrite a clean or deferred cell |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-09-01 | Tampering | temp matrix-config scope error | mitigate | Task 1 derives the cell set from per-run data; Task 3 relies on write_baselines carry-forward so unrelated cells stay byte-identical; Task 4 parity test catches orphan/missing cells |
| T-11-09-02 | Denial of Service | OpenAI embeddings quota (21-14-30Z poison mode) | mitigate | Task 2 mandatory embeddings sanity probe + STOP rule on 429 storm before any write |
| T-11-09-03 | Information Disclosure | live provider keys in temp configs / reports | accept | temp configs hold no secrets (only provider/model strings); eval_reports already redact per Phase 10 probe redaction; /tmp configs never committed |
| T-11-09-SC | Tampering | npm/pip/cargo installs | accept | No package installs in this plan — stdlib + existing toolchain only (RESEARCH.md: no new packages) |
</threat_model>

<verification>
- CR-01 / CR-02 code fixes are present and committed (verified at plan time: score_checks
  guards None before float(); _build_summary_from_baselines raises on missing/empty dir). This
  plan must NOT re-edit them.
- No gate keys on category_compliance (verified at plan time: `grep category_compliance
  configs/eval_gates.yaml` returns nothing). Gate VALUES are CR-01-clean.
- `make eval-gates-check-baselines` currently exits 0 with the contaminated data (commit-rate
  gates pass); it must STILL exit 0 after regen.
- The contaminated cells are precisely the ones with per-run `check_err>=1 AND cir==0.0`.
</verification>

<success_criteria>
- Three failed VERIFICATION truths closed: (1) D-11-03 None-abstain produces clean
  category_compliance in committed baselines (no n=5 all-0.0 error-path blocks); (2)+(3)
  configs/eval_baselines/*.json contain honest n=5 numbers for the re-measured cells.
- make eval-gates-check-baselines exits 0; parity test passes; full make test green.
- gpt-4o-mini omakase anchor, anthropic (billing) and gemini (D-11-11) deferrals untouched.
- Temp scoped matrix-configs never committed; D-11-14 sequential execution honored.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-09-SUMMARY.md` when done.
Record: the data-derived regen cell set, the new eval_reports timestamps, before/after
category_compliance for each regenerated cell, whether any eval_gates.yaml rationale was edited,
and the final closing-verification exit codes.
</output>
