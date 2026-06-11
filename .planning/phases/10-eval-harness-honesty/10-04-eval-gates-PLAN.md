---
phase: 10-eval-harness-honesty
plan: 04
type: execute
wave: 1
depends_on: []
files_modified:
  - configs/eval_gates.yaml
  - scripts/check_eval_gates.py
  - docs/eval_gates.md
  - Makefile
  - configs/eval_matrix_refinement.yaml
  - tests/unit/test_check_eval_gates.py
autonomous: true
requirements: [EVAL-03]
must_haves:
  truths:
    - "Per-family merge gates live in one machine-readable source of truth, configs/eval_gates.yaml"
    - "The unsatisfiable strict refinement_minimal_edit == 1.0 anchor gate is formally retired"
    - "Hard gates ride on committed_itinerary_rate floors; refinement_minimal_edit medians are advisory"
    - "make eval-gates-check exits non-zero when a summary's hard-gated cell drops below its gate value"
    - "Aspirational gates (gpt-5-mini commit_rate >= 0.6) are reported distinctly, not hard-failing the build"
    - "docs/eval_gates.md explains gate semantics and links to the YAML without duplicating numbers"
  artifacts:
    - path: "configs/eval_gates.yaml"
      provides: "per-family gates with hard/advisory/status/rationale fields (D-10-08)"
      contains: "committed_itinerary_rate"
    - path: "scripts/check_eval_gates.py"
      provides: "summary.json gate-checker with check_baselines_fresh-style exit codes"
      exports: ["main"]
    - path: "docs/eval_gates.md"
      provides: "narrative gate semantics linking to the YAML"
      contains: "aspirational"
    - path: "Makefile"
      provides: "eval-gates-check target wrapping the checker"
      contains: "eval-gates-check"
  key_links:
    - from: "configs/eval_gates.yaml"
      to: "scripts/check_eval_gates.py"
      via: "checker loads the YAML and compares each gate against summary.json"
      pattern: "eval_gates.yaml"
    - from: "Makefile eval-gates-check"
      to: "scripts/check_eval_gates.py"
      via: "POETRY_RUN python scripts/check_eval_gates.py $(SUMMARY)"
      pattern: "check_eval_gates"
---

<objective>
Re-derive the per-family merge gates from honest anchor data and make them executable (EVAL-03).
The documented strict `refinement_minimal_edit == 1.0` anchor gate is unsatisfiable — the honest
anchor (`openai/gpt-4o-mini`) sits at median 0.0 / max 0.5 after the Phase-7 scorer tightening
(D-07-05/D-07-07). That gate fossilized because the number was hardcoded in a Makefile, not in a
single inspectable source of truth (D-10-05). This plan creates `configs/eval_gates.yaml` (the
one machine-readable source), `scripts/check_eval_gates.py` (consumes a matrix summary.json and
exits non-zero on hard-gate violations, imitating `check_baselines_fresh.py` exit codes),
`docs/eval_gates.md` (explains semantics, links to the YAML, never duplicates numbers), and the
`eval-gates-check` Makefile target.

Gate shape extends the D-09-02 two-part pattern to all families (D-10-06): hard gates ride on
`committed_itinerary_rate` floors; `refinement_minimal_edit` medians are advisory everywhere
until v2.2. The strict-1.0 anchor gate is formally retired. Provisional hard values come from
D-10-07: gpt-4o-mini commit_rate >= 0.8 (active); gpt-5-mini >= 0.6 (aspirational — currently
fails at 0.4, reported but non-blocking); anthropic >= 0.8 (provisional-n1); deepseek + gemini
logged-not-gated; late_night quarantined-legacy-threading.

Purpose: Phase 11 BASE-03 promotes these gates to CI; they must be satisfiable and inspectable
first.
Output: a runnable gate-check that distinguishes hard violations from aspirational misses.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/10-eval-harness-honesty/10-CONTEXT.md
@.planning/phases/10-eval-harness-honesty/10-PATTERNS.md
@scripts/check_baselines_fresh.py
@configs/eval_matrix_refinement.yaml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Author configs/eval_gates.yaml and docs/eval_gates.md; retire the strict-1.0 gate breadcrumb</name>
  <files>configs/eval_gates.yaml, docs/eval_gates.md, configs/eval_matrix_refinement.yaml</files>
  <read_first>
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (the full eval_gates.yaml schema example with all seven family entries and the docs/eval_gates.md structure — copy the field shape: family, status, rationale, hard{metric,op,value}, advisory[])
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-05 single machine-readable source; D-10-06 retire strict-1.0, hard gates on committed_itinerary_rate; D-10-07 provisional values per family; D-10-08 required fields)
    - configs/eval_matrix_refinement.yaml (read its existing gate-status comments — migrate them to POINT AT the new gates YAML; comments stay as breadcrumbs, numbers live only in the YAML per D-10-05)
  </read_first>
  <action>
    Create configs/eval_gates.yaml with a top-level `gates:` list. Each entry has fields per D-10-08: `family` (provider/model string), `status` (one of: active | aspirational | provisional-n1 | logged | quarantined-legacy-threading), `rationale` (one-liner with the D-ID), `hard` (either null or {metric, op, value}), `advisory` (list of {metric, op, value}). Author exactly these seven entries with values from D-10-07: openai/gpt-4o-mini status=active hard={committed_itinerary_rate, ">=", 0.8} advisory=[{refinement_minimal_edit_median, ">=", 0.0}]; openai/gpt-5-mini status=aspirational hard={committed_itinerary_rate, ">=", 0.6}; anthropic/claude-sonnet-4-6 status=provisional-n1 hard={committed_itinerary_rate, ">=", 0.8}; deepseek/deepseek-reasoner status=logged hard=null; deepseek/deepseek-chat status=logged hard=null; gemini/gemini-3.1-pro-preview status=logged hard=null; late_night_closure_cascade status=quarantined-legacy-threading hard=null. Each rationale cites the D-ID (D-10-07/D-09-02/D-10-09 as appropriate). Add a header comment: single source of truth, consumed by scripts/check_eval_gates.py (make eval-gates-check), docs explain semantics. Create docs/eval_gates.md: explain each status value (active enforced; aspirational reported-not-blocking; provisional-n1 single-run; logged no-gate; quarantined-legacy-threading excluded), show `make eval-gates-check SUMMARY=...`, state the strict refinement_minimal_edit==1.0 gate is formally retired (D-10-06) and why (honest anchor median 0.0/max 0.5 post-Phase-7), and instruct adding a gate by editing the YAML with a D-ID rationale. Do NOT duplicate any numeric gate value in the doc — link to the YAML. In configs/eval_matrix_refinement.yaml, migrate any inline gate-number comments to point at configs/eval_gates.yaml (keep a breadcrumb comment, remove hardcoded gate numbers from the comment body).
  </action>
  <verify>
    <automated>poetry run python -c "import yaml; g=yaml.safe_load(open('configs/eval_gates.yaml')); fams={e['family'] for e in g['gates']}; assert 'openai/gpt-4o-mini' in fams and 'openai/gpt-5-mini' in fams; anchor=[e for e in g['gates'] if e['family']=='openai/gpt-4o-mini'][0]; assert anchor['status']=='active' and anchor['hard']['metric']=='committed_itinerary_rate' and anchor['hard']['value']==0.8"</automated>
  </verify>
  <acceptance_criteria>
    - `configs/eval_gates.yaml` parses as YAML and contains all seven families with the D-10-07 statuses and values (verify one-liner exits 0).
    - The gpt-5-mini entry has `status: aspirational` and `committed_itinerary_rate >= 0.6`.
    - `grep -c "refinement_minimal_edit.*== *1.0\|refinement_minimal_edit.*1\\.0" configs/eval_gates.yaml` returns 0 (no strict-1.0 hard gate authored).
    - `docs/eval_gates.md` contains the words `aspirational`, `quarantined-legacy-threading`, and `retired` (or `formally retired`) and references `configs/eval_gates.yaml`.
    - `grep -nE '0\.8|0\.6|>= *1' configs/eval_matrix_refinement.yaml` returns 0 for gate numbers inside comments (numbers migrated out; breadcrumb pointer remains).
  </acceptance_criteria>
  <done>The single machine-readable gate source exists with honest per-family values; the strict-1.0 gate is retired; the doc explains semantics without duplicating numbers; the refinement matrix comments point at the YAML.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement scripts/check_eval_gates.py + Makefile target + unit tests</name>
  <files>scripts/check_eval_gates.py, Makefile, tests/unit/test_check_eval_gates.py</files>
  <read_first>
    - scripts/check_baselines_fresh.py (read imports :1-57, _parse_args :163-185, main + exit codes :215-279 — copy the argparse style, main() -> int, raise SystemExit(main()), and the 0/1/2 exit convention exactly)
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (check_eval_gates.py imports / argparse / main+exit-code skeleton; the ASPIRATIONAL-miss reporting vs HARD-VIOLATION distinction)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-05 Make target eval-gates-check; D-10-06 hard on committed_itinerary_rate, advisory logged; status: aspirational reported distinctly)
    - tests/unit/test_check_baselines_fresh.py (test style for a check_*-script: tmp_path fixtures, asserting exit codes)
    - Makefile (read POETRY_RUN var at :6 and the eval-matrix-refinement-structural-check target at :145-150 for the target style)
    - configs/eval_gates.yaml (the schema authored in Task 1 — the checker reads it)
  </read_first>
  <behavior>
    - check_eval_gates(summary, gates_config) returns exit 0 when every active/provisional-n1 hard gate is satisfied and no aspirational gate is checked-as-blocking.
    - A summary where the active gpt-4o-mini cell has committed_itinerary_rate below 0.8 returns exit 1 with a HARD GATE VIOLATION stderr line naming the family.
    - A summary where the aspirational gpt-5-mini cell has commit_rate 0.4 (below 0.6) returns exit 0 but prints an ASPIRATIONAL miss line (reported, not blocking).
    - A missing gates YAML or malformed summary.json returns exit 2.
    - logged and quarantined-legacy-threading families are never hard-failed regardless of their values.
  </behavior>
  <action>
    Create scripts/check_eval_gates.py per the PATTERNS.md skeleton: argparse with a required positional `summary` (path to a matrix summary.json) and `--gates-config` defaulting to configs/eval_gates.yaml. Load the gates YAML and the summary.json; for each gate with a non-null `hard` block, locate the matching family cell in the summary (the summary's per-provider keys are `<provider>/<model>` strings produced by aggregate_cell_jsons) and read the gate metric (committed_itinerary_rate). Compare via the gate's op/value. Classify each result: `violation` if status in {active, provisional-n1} and the hard gate fails; `aspirational_miss` if status==aspirational and the hard gate fails; pass otherwise; `logged`/`quarantined-legacy-threading` families are skipped entirely. Exit codes (copy check_baselines_fresh.py exactly): 0 = all hard gates passed (aspirational misses printed but non-blocking); 1 = one or more hard-gate violations; 2 = infrastructure failure (missing YAML / unreadable summary). Print aspirational misses to stdout, hard violations to stderr. Use `main(argv) -> int` and `raise SystemExit(main())`. Where the metric `committed_itinerary_rate` is not yet present in the summary scorers block, treat its absence as a gate-not-evaluable condition reported (not a silent pass) — document this as a known limitation until Phase 11 wires committed_itinerary_rate into the matrix scorers; for cells where only refinement_minimal_edit exists, the hard gate on committed_itinerary_rate is reported as not-evaluable rather than passing. Add the Makefile target `eval-gates-check` (PHONY) wrapping `$(POETRY_RUN) python scripts/check_eval_gates.py $(SUMMARY) --gates-config configs/eval_gates.yaml` with a help comment. In tests/unit/test_check_eval_gates.py, add tests using tmp_path fixtures that write a synthetic gates YAML + synthetic summary.json and assert the four exit-code outcomes (pass, hard-violation, aspirational-miss-still-0, infra-2) and that logged families never block.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_check_eval_gates.py -q</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/check_eval_gates.py` defines `main(argv=None) -> int` and ends with `raise SystemExit(main())`.
    - Exit codes match check_baselines_fresh.py: 0 = clean, 1 = hard violation, 2 = infra failure (asserted by tests).
    - A test asserts an aspirational gpt-5-mini miss returns exit 0 with an ASPIRATIONAL line printed (not a hard fail).
    - A test asserts an active gpt-4o-mini commit_rate below its gate returns exit 1 with the family name in stderr.
    - `make eval-gates-check SUMMARY=<a passing synthetic summary>` exits 0; the Makefile contains a `.PHONY: eval-gates-check` target invoking the script.
    - `poetry run ruff check scripts/check_eval_gates.py tests/unit/test_check_eval_gates.py` passes.
  </acceptance_criteria>
  <done>An executable gate-check consumes summary.json against the YAML, blocks on active/provisional hard violations, reports aspirational misses without blocking, and is wired to a Makefile target with full unit coverage.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| summary.json (untrusted on-disk artifact) → check_eval_gates.py | a malformed or hand-edited summary could spoof passing gates; the checker must fail closed (exit 2) on bad shape, never silently pass |
| configs/eval_gates.yaml → CI gate decision (Phase 11) | the YAML is the source of truth for merge-blocking; a wrong value silently weakens the gate |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-04-01 | Spoofing | summary.json shape | mitigate | check_eval_gates returns exit 2 (infra) on missing/malformed YAML or summary; a not-evaluable metric is reported, not treated as a silent pass |
| T-10-04-02 | Tampering | gate values in eval_gates.yaml | mitigate | Single source of truth with per-gate D-ID rationale; docs forbid duplicating numbers (eliminates the fossilized-Makefile-number failure mode that produced the unsatisfiable strict-1.0 gate) |
| T-10-04-03 | Elevation of privilege | aspirational gate non-blocking | accept | gpt-5-mini's known-failing 0.6 gate is status=aspirational by design (D-10-07) so Phase-10/11 work ships on a known v2.2 gap; the gate stays visible (printed) and Phase 11 BASE-03 decides promotion |
| T-10-04-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs; check_eval_gates uses only stdlib + PyYAML already in the dependency tree |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_check_eval_gates.py -q` exits 0.
- `poetry run python -c "import yaml; yaml.safe_load(open('configs/eval_gates.yaml'))"` parses.
- `make eval-gates-check SUMMARY=<passing synthetic summary>` exits 0.
- `poetry run ruff check scripts/check_eval_gates.py configs/eval_gates.yaml tests/unit/test_check_eval_gates.py 2>/dev/null; poetry run ruff check scripts/check_eval_gates.py tests/unit/test_check_eval_gates.py` passes.
</verification>

<success_criteria>
- Per-family gates are in one machine-readable source with honest, satisfiable values; the strict-1.0 gate is retired (EVAL-03).
- make eval-gates-check exits non-zero on a hard-gate regression and reports aspirational misses distinctly without blocking.
- docs/eval_gates.md explains semantics and links to the YAML without duplicating numbers.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-04-SUMMARY.md` when done.
</output>
