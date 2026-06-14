---
phase: 14-richer-state-replay-conditional
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/audit_list_content_aimessages.py
  - docs/replay_arm_verdicts.md
  - tests/unit/test_audit_list_content.py
autonomous: true
requirements: [REPLAY-02]
must_haves:
  truths:
    - "A zero-spend audit determines whether the three RUN models (gpt-5-mini, gpt-4o-mini, deepseek-reasoner) ever produce list-content AIMessages that _prune_for_llm's str() collapse would have altered"
    - "The audit conclusion is written into docs/replay_arm_verdicts.md before any live A/B spend (D-14-05)"
    - "docs/replay_arm_verdicts.md exists as a scaffold mirroring the DEC-05 document structure with empty per-arm run-dir / table / falsifier-output slots ready for the live-run plans to fill"
    - "The verdict doc cross-links docs/decisiveness_arm_verdicts.md rather than appending to it (Phase-13 record stays immutable)"
  artifacts:
    - path: "scripts/audit_list_content_aimessages.py"
      provides: "zero-spend list-content audit over existing run dirs + structural adapter analysis"
      exports: ["main"]
    - path: "docs/replay_arm_verdicts.md"
      provides: "REPLAY verdict doc scaffold + R2 evidence-audit result section"
      contains: "ARCH-FUT-01 Evaluation"
    - path: "tests/unit/test_audit_list_content.py"
      provides: "unit coverage for the audit script's run-JSON parsing + structural conclusion"
      contains: "def test_"
  key_links:
    - from: "docs/replay_arm_verdicts.md"
      to: "docs/decisiveness_arm_verdicts.md"
      via: "cross-link reference (not append)"
      pattern: "decisiveness_arm_verdicts"
---

<objective>
Run the D-14-05 evidence audit BEFORE any live spend: determine from existing run-dir artifacts and adapter structure whether the `_prune_for_llm` `str()` collapse (graph.py:232) was ever causing observable loss for the three RUN models, write that conclusion into a new verdict document, and scaffold that document to mirror the Phase-13 DEC-05 record so the live-run plans (Wave 3/4) only fill in numbers.

Purpose: Roadmap criterion 2 demands "an explanation of whether str() collapse was causing observable loss in run JSONs" — that explanation must be grounded in evidence, written up front, and is the cheapest task in the phase (zero API spend). It also de-risks the R2 live run by telling us in advance whether R2 is expected-null on the tested cells.
Output: A zero-spend audit script with unit tests, the audit conclusion, and a scaffolded docs/replay_arm_verdicts.md ready for the live runs.

IMPORTANT GROUND TRUTH (verified during planning — do not assume otherwise): the persisted EvalRunReport run JSONs do NOT contain full message traces. `queries[i].deterministic` carries `tool_calls` as an integer COUNT, `tool_names`, `step_telemetry`, `arm_flags`, etc. — there is NO serialized AIMessage `.content` and NO `additional_kwargs`. CONTEXT.md's phrasing "full message traces are recorded" is inaccurate for these artifacts. The audit must handle this honestly (see Task 1).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md
@.planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md
@docs/decisiveness_arm_verdicts.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write the zero-spend list-content audit script + unit tests</name>
  <files>scripts/audit_list_content_aimessages.py, tests/unit/test_audit_list_content.py</files>
  <read_first>
    - scripts/eval_agent.py lines 184-260 and 900-940 (EvalRunReport dataclass + what deterministic actually serializes — confirms run JSONs carry tool_calls as a count, not message content)
    - app/agent/adapters/openai_gpt5.py, deepseek.py, gemini.py (the three RUN-model adapters: all round-trip reasoning via additional_kwargs["reasoning_content"] / additional_kwargs keys; their AIMessage.content is the string reply — NOT a block list)
    - app/agent/adapters/anthropic.py (the ONLY adapter that uses AIMessage.content as a block list — and anthropic is deferred, NOT in the run matrix)
    - app/agent/graph.py lines 218-238 (the _prune_for_llm str() collapse site that R2 targets — collapse only changes content when content is a list)
  </read_first>
  <action>
    Create `scripts/audit_list_content_aimessages.py` (REPLAY-02 evidence audit, D-14-05). The script must do BOTH halves of the audit because the run JSONs alone are insufficient.

    Half (a) Run-dir scan: iterate the Phase-12/13 arm run dirs (accept a `--run-dir` arg, default to scanning the recent A1/A2/A3 dirs recorded in docs/decisiveness_arm_verdicts.md), open each `*.json`, and report what is actually persisted regarding AIMessage content shape. The script MUST detect and report honestly that EvalRunReport JSONs persist `queries[i].deterministic.tool_calls` as an integer count with no serialized message `.content` / `additional_kwargs` — i.e. the run JSONs cannot directly answer "did an AIMessage carry list content pre-cutoff". Emit this as an explicit finding, not a crash.

    Half (b) Structural adapter analysis: derive the answer from adapter design. For the three RUN models — openai/gpt-5-mini, openai/gpt-4o-mini (anchor), deepseek/deepseek-reasoner — reasoning state round-trips via `additional_kwargs["reasoning_content"]` and the AIMessage `.content` is a string reply, so the `str(m.content)` collapse at graph.py:232 is a no-op (string in, identical string out). Only AnthropicAdapter uses a `.content` block list, and anthropic is deferred (not run). Conclusion the script prints: for the tested cells, `str()` collapse caused NO observable content loss because none of the three RUN models emit list-content AIMessages; R2 is therefore EXPECTED-NULL on these cells but still RUNS (criterion 2 requires a measured delta).

    The script must exit 0 and print a structured summary (run dirs scanned, persisted-shape finding, per-run-model structural verdict, overall expected-R2-effect verdict). Keep it stdlib + app imports only; no live network, no LLM, no DB. Follow the project no-sys.path rule (app is poetry-editable-installed; import directly).

    Add `tests/unit/test_audit_list_content.py`: unit tests with a synthesized fixture run-JSON dict (mirroring the real EvalRunReport shape: queries[i].deterministic.tool_calls as int) asserting the script (i) detects the no-message-trace persisted shape without crashing, (ii) returns the structural EXPECTED-NULL verdict for the three RUN-model provider/model pairs, (iii) classifies anthropic as the only list-content adapter. No live runs.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_audit_list_content.py -q && poetry run python scripts/audit_list_content_aimessages.py --help</automated>
  </verify>
  <acceptance_criteria>
    - `poetry run pytest tests/unit/test_audit_list_content.py -q` passes
    - Running `poetry run python scripts/audit_list_content_aimessages.py` against at least one existing run dir exits 0 and prints (i) an explicit finding that run JSONs persist tool_calls as a count with no message `.content`, and (ii) a per-RUN-model structural verdict
    - The printed overall verdict states R2 is expected-null on the three RUN-model cells because only Anthropic (deferred, not run) emits list-content AIMessages (behavior assertion on stdout)
    - The script imports `app...` directly with NO `sys.path.insert` / `REPO_ROOT` bootstrap (source assertion)
  </acceptance_criteria>
  <done>A zero-spend audit script exists with unit tests, honestly reports that run JSONs lack message traces, and derives the structurally-grounded EXPECTED-NULL conclusion for R2 on the three tested cells.</done>
</task>

<task type="auto">
  <name>Task 2: Scaffold docs/replay_arm_verdicts.md and write the R2 evidence-audit result</name>
  <files>docs/replay_arm_verdicts.md</files>
  <read_first>
    - docs/decisiveness_arm_verdicts.md (the FULL DEC-05 document — the exact structure to mirror: falsifier definition, run-budget contract, per-arm sections with run dirs + smoke verification + per-model tables + pasted falsifier output + closing verdict, summary table, explicit closing line)
    - .planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md (D-14-01, D-14-02, D-14-07, D-14-08 — arm structure, run budget, three-delta-column requirement, ARCH-FUT-01 handoff)
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (section "docs/replay_arm_verdicts.md (NEW) — verdict document structure": the exact scaffold layout)
    - The audit output from Task 1 (the R2 evidence-audit conclusion to paste into the R2 Evidence Audit section)
  </read_first>
  <action>
    Create `docs/replay_arm_verdicts.md` (D-14-07, REPLAY verdict record). Mirror the DEC-05 document structure exactly with these sections, leaving live-run slots as explicit `[fill in Wave 3/4]` placeholders.

    Header: Title + Role line ("REPLAY record — canonical per-arm verdict document for Phase 14") + a Cross-link line that references `docs/decisiveness_arm_verdicts.md` as the immutable Phase-13 record (CROSS-LINK, do NOT append to or edit that file).

    INST-05 Falsifier Definition: copy the three-clause definition from the DEC-05 doc; comparison points updated for Phase 14 (flag-off plateau floor AND best-DEC-arm A2 = 0.500).

    Run Budget Contract: ≤4 full live matrix runs total; R1 + R2 + conditional R3 + discretionary best-replay+FORCED_COMMIT_STEP=6 valve (D-14-01); smoke n=1 with arm_flags verification before every full spend (D-14-02); no billing top-ups; partials never written as baselines (D-11-14).

    R1 section (REPLAY_MULTI_MESSAGE_ENABLED=1, all DEC flags UNSET): Run Dirs table [fill], Smoke arm_flags verification [fill], per-model results table with THREE delta columns per model — pooled commit rate, Delta vs flag-off floor, Delta vs A2 (0.500) — plus omakase / refinement_cheaper / falsifier verdict columns [fill], falsifier exit code [fill], pasted falsifier per-scenario breakdown [fill], closing verdict [fill].

    R2 section (REPLAY_CONTENT_BLOCKS_ENABLED=1, all DEC flags UNSET): same structure as R1, PLUS an "R2 Evidence Audit (D-14-05)" subsection FILLED NOW with the Task 1 conclusion (run JSONs lack message traces; structural analysis shows the three RUN models emit string-content AIMessages so str() collapse was a no-op for them; R2 expected-null on tested cells but runs anyway per criterion 2).

    R3 Conditional Combo (R1+R2) section: scaffold [fill]; note the D-14-01 condition (run ONLY if neither clears alone but both show positive signal).

    Closing Verdict: Per-Arm Summary Table (with the three-delta columns) [fill]; "ARCH-FUT-01 Evaluation (on plateau)" section scaffold with the three required parts — (a) cumulative evidence chain, (b) ARCH-FUT-01 contingency restatement, (c) written recommendation bounded by Decision 3 — marked as a USER CHECKPOINT (D-14-08) [fill]; Explicit Closing Line [fill]; Phase-15 Consequence [fill: winning flag config + run dir OR documented plateau + user checkpoint].

    The R2 Evidence Audit subsection is the ONLY section filled with conclusions in this plan; all R1/R2/R3 number slots stay as labeled placeholders for the live-run plans.
  </action>
  <verify>
    <automated>test -f docs/replay_arm_verdicts.md && grep -q "decisiveness_arm_verdicts" docs/replay_arm_verdicts.md && grep -q "ARCH-FUT-01 Evaluation" docs/replay_arm_verdicts.md && grep -q "R2 Evidence Audit" docs/replay_arm_verdicts.md</automated>
  </verify>
  <acceptance_criteria>
    - `docs/replay_arm_verdicts.md` exists and contains an INST-05 Falsifier Definition, a Run Budget Contract, R1 / R2 / R3 sections, a Closing Verdict with an "ARCH-FUT-01 Evaluation" section, and an Explicit Closing Line slot
    - `grep -c "decisiveness_arm_verdicts" docs/replay_arm_verdicts.md` >= 1 AND `docs/decisiveness_arm_verdicts.md` is NOT modified by this plan (git diff shows no change to the Phase-13 record)
    - The R1 and R2 per-model tables each have three delta-related columns (pooled rate, Delta vs flag-off floor, Delta vs A2 0.500) — source assertion on the table headers (D-14-07)
    - The "R2 Evidence Audit (D-14-05)" subsection is FILLED with the structural conclusion (expected-null on tested cells; str() collapse was a no-op for the three RUN models; only Anthropic emits list content and is deferred)
    - All R1/R2/R3 numeric result slots remain explicit `[fill ...]` placeholders (no fabricated numbers)
  </acceptance_criteria>
  <done>The verdict doc exists mirroring the DEC-05 structure, cross-links (not appends to) the Phase-13 record, carries the filled R2 evidence-audit conclusion, and leaves all live-run number slots as labeled placeholders.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| run-dir JSON files → audit script | Local read-only filesystem artifacts parsed by the audit script; no external input |
| audit conclusion → verdict doc | Documentation only; no executable surface |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-14-04 | Tampering | audit script JSON parsing of run dirs | mitigate | Script treats run JSONs as untrusted-shape input: uses `.get()` with defaults and tolerates missing keys (it already must handle the no-message-trace shape), so a malformed run JSON yields a reported finding, not a crash |
| T-14-05 | Information disclosure | verdict doc content | accept | The doc records eval metrics + structural analysis only — no secrets, no PII, no credentials; same disclosure profile as the existing docs/decisiveness_arm_verdicts.md |
| T-14-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs (stdlib + existing app deps only); slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_audit_list_content.py -q` passes
- `make lint` passes (ruff) on the new script + test
- `make typecheck` passes (mypy app/) — note the new script lives in scripts/, mypy targets app/; ensure no app/ change
- The verdict doc scaffold has every required section and the filled R2 Evidence Audit subsection; no fabricated R1/R2/R3 numbers
</verification>

<success_criteria>
- The D-14-05 evidence audit is complete BEFORE any live spend and its conclusion is written into the verdict doc
- The audit honestly reports that run JSONs lack message traces and derives the EXPECTED-NULL R2 verdict from adapter structure
- docs/replay_arm_verdicts.md mirrors the DEC-05 structure, cross-links the immutable Phase-13 record, and is ready for the live-run plans to fill
</success_criteria>

<output>
Create `.planning/phases/14-richer-state-replay-conditional/14-02-SUMMARY.md` when done
</output>
