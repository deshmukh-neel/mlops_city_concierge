---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "09"
subsystem: testing
tags: [eval, baselines, category_compliance, abstain, CR-01, gap-closure]

# Dependency graph
requires:
  - phase: 11-cross-model-baseline-regen-matrix-expansion
    provides: CR-01 abstain fix (score_checks None guard), CR-02 fail-closed baselines-mode, write_baselines.py carry-forward semantics
provides:
  - Contamination-free omakase and refinement baselines: all category_compliance cells reflect CR-01 abstain semantics (zero-stop runs no longer produce n=5 all-0.0 blocks)
  - Re-verified eval_gates.yaml rationales against fresh n=5 data (no rationale edits needed — gates key on commit-rate only)
  - make eval-gates-check-baselines exits 0; full suite 1204 passed
affects: [phase 12, PR merge gates, baseline regen runbook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Scoped regen: temp matrix-configs in /tmp exclude clean anchors and deferred cells; never committed"
    - "write_baselines carry-forward: only cells present in summary.json are overwritten; deferred/clean cells stay byte-identical"
    - "CR-01 abstain proof: zero-stop cells show category_compliance absent or n<5, never n=5 all-0.0"

key-files:
  created: []
  modified:
    - configs/eval_baselines/omakase_mission_open_ended.json
    - configs/eval_baselines/refinement_cheaper.json

key-decisions:
  - "No eval_gates.yaml rationale edits needed: gates key on committed_itinerary_rate (CR-01-clean); no rationale prose cited contaminated category_compliance numbers"
  - "deepseek/deepseek-chat omakase: category_compliance.n=1 (only 1 committed run) — correct abstain, not n=5 all-0.0"
  - "deepseek/deepseek-reasoner omakase + refinement: category_compliance block fully absent (0 committed runs) — correct full abstain"
  - "openai/gpt-4o-mini omakase anchor preserved from 2026-06-11T19-37-42Z run (not re-run, write_baselines carry-forward)"
  - "Plan checker advisory confirmed: refinement/deepseek-chat run-0 was CLEAN (CIR=1.0), not infra-errored; only run-1 was infra-errored; per-run data is source of truth"

patterns-established:
  - "Scoped regen pattern: derive exact contaminated-cell set from per-run JSONs before any live run"
  - "Gap-closure regen seam: temp matrix-config → live run → write_baselines → gate check → commit baselines only (eval_reports gitignored)"

requirements-completed: [BASE-01, BASE-03]

# Metrics
duration: ~2h 30min
completed: 2026-06-11
---

# Phase 11 Plan 09: Contaminated-Cell Regen Summary

**Re-measured all CR-01-contaminated category_compliance cells at n=5 under fixed abstain semantics: zero-stop cells now show absent or n<5 category_compliance, never the error-path n=5 all-0.0 block**

## Performance

- **Duration:** ~2h 30min (including sequential matrix runs for 7 provider-scenario pairs)
- **Started:** ~2026-06-11T21:30:00Z
- **Completed:** 2026-06-11T23:00:00Z
- **Tasks:** 4
- **Files modified:** 2 (configs/eval_baselines/*.json)

## Accomplishments

- Programmatically derived the exact contaminated-cell set from per-run JSONs using `committed_itinerary_rate==0.0 AND check_error_count>=1` predicate; confirmed plan table was accurate with one minor imprecision (see Deviations)
- Ran two scoped matrix regen runs sequentially (D-11-14): 3 omakase providers × 5 runs = 15 cells clean (exit 0, no 429s, no error_cells); 4 refinement providers × 5 runs = 20 cells clean (exit 0)
- Updated both baselines via `write_baselines.py --n-requested 5`; CR-01 abstain semantics verified in all regenerated cells
- Gate check passes (`make eval-gates-check-baselines` exits 0, only documented non-blocking gpt-5-mini aspirational miss); parity test 68/68 passed; full suite 1204/1204 passed

## Task Commits

1. **Tasks 1-2: Contamination analysis + scoped regen runs** - (analysis + live runs; eval_reports gitignored; no separate code commit)
2. **Tasks 3-4: Write baselines + closing verification** - `fa468ec` (chore)
3. **Plan metadata doc:** `eabc983` (docs)

## Files Created/Modified

- `configs/eval_baselines/omakase_mission_open_ended.json` - Regenerated cells for gpt-5-mini (cat_n=4, CIR med=1.0), deepseek-chat (cat_n=1, CIR med=0.0), deepseek-reasoner (cat absent, CIR med=0.0); gpt-4o-mini anchor preserved from 2026-06-11T19-37-42Z
- `configs/eval_baselines/refinement_cheaper.json` - Regenerated cells for gpt-4o-mini (cat present, CIR med=1.0), gpt-5-mini (cat absent, CIR med=0.0), deepseek-chat (cat absent, CIR med=0.0), deepseek-reasoner (cat absent, CIR med=0.0); anthropic preserved (n=1 stale, D-11-20 deferral)

## Before/After category_compliance Summary

| Scenario | Provider | Before (contaminated) | After (clean) |
|----------|----------|----------------------|---------------|
| omakase | deepseek/deepseek-chat | n=5, median=0.0 (CR-01 error path) | n=1, median=1.0 (1 committed run) |
| omakase | deepseek/deepseek-reasoner | n=5, median=0.0 (CR-01 error path) | ABSENT (0 committed runs — full abstain) |
| omakase | openai/gpt-5-mini | n=5, median=0.333 (CR-01 mix) | n=4, median=0.833 (4 committed runs) |
| refinement | openai/gpt-4o-mini | n=5, median=0.333 (run-0 contaminated) | n=5, median=0.0 (committed all 5, but no primary_type match — correct) |
| refinement | openai/gpt-5-mini | n=5, median=0.0 (CR-01 error path) | ABSENT (0 committed runs) |
| refinement | deepseek/deepseek-chat | n=5, median=0.0 (stale Phase-9 + contamination) | ABSENT (0 committed runs) |
| refinement | deepseek/deepseek-reasoner | n=5, median=0.0 (CR-01 error path) | ABSENT (0 committed runs) |

## New eval_reports Directories

- `eval_reports/2026-06-11T22-05-06Z/` — omakase regen: 3 providers × 5 runs = 15 files + summary.json (gitignored)
- `eval_reports/2026-06-11T22-19-17Z/` — refinement regen: 4 providers × 5 runs = 20 files + summary.json (gitignored)

## Decisions Made

- No `eval_gates.yaml` rationale edits were needed: every gate keys on `committed_itinerary_rate` which CR-01 does not contaminate; no rationale prose cited contaminated category_compliance numbers
- D-11-09 (gap-closure regen): documented as the date stamp for this baseline regeneration event

## Deviations from Plan

### Plan Checker Advisory — Data Imprecision Confirmed and Recorded

**Plan table imprecision for refinement/deepseek-chat:**
- **What the table said:** "runs 0,1 infra-errored"
- **What per-run data showed:** run-0 was CLEAN (CIR=1.0, check_err=0); only run-1 was infra-errored (HTTP/auth failure); runs 2,3,4 were CR-01 contaminated (zero-stop, check_err=1)
- **Resolution:** Per-run data is source of truth per plan clause. Cell was still correctly included in REFINE_REGEN set (it needed regen regardless). No behavioral impact. Recorded in SUMMARY per plan instruction.

None - plan executed exactly as written in all other respects.

## Issues Encountered

- `asyncio.run(semantic_search(...))` in the embeddings probe failed because `semantic_search` is synchronous; fixed by calling directly (Rule 3 auto-fix)
- `PlaceHit` is a Pydantic model without `.get()` method; fixed by using `.name` attribute instead (Rule 3 auto-fix)
- Both issues were in probe-only code (not committed)

## Gate Rationale Re-Verification (BASE-03)

Checked all 7 gate entries in `configs/eval_gates.yaml` for prose citing category_compliance or zero-stop-derived numbers:
- `openai/gpt-4o-mini` (line 23): cites omakase median 1.0 — correct, anchor not re-run, still valid
- `openai/gpt-5-mini` (line 35): cites omakase CIR median 1.0 + refinement CIR median 0.0 — both CR-01-clean, confirmed against fresh runs; remains `aspirational`
- `deepseek/deepseek-reasoner` (line 49): cites omakase CIR median 0.0 — confirmed by new run (CIR max=0.0, all 5 runs)
- `deepseek/deepseek-chat` (line 55): cites "1/5 converged" — confirmed by new omakase run (max=1.0 means one run had CIR=1.0)
- `anthropic`, `gemini`, `late_night_closure_cascade`: unchanged deferred/quarantined entries

**Result:** No rationale edits required. `eval_gates.yaml` left byte-identical.

## Closing Verification Results

| Check | Result |
|-------|--------|
| `make eval-gates-check-baselines` | Exit 0; only non-blocking gpt-5-mini aspirational miss printed |
| `pytest tests/unit/test_eval_matrix.py` | 68/68 passed |
| `make test` | 1204 passed, 0 failures |
| Temp configs not tracked | `git ls-files \| grep eval_matrix_regen` returns 0 |

## Next Phase Readiness

- Phase 11 plan 09 closes the three VERIFICATION gaps: (1) D-11-03 None-abstain produces clean category_compliance; (2)+(3) honest n=5 numbers in committed baselines
- All baseline cells for non-deferred providers are now CR-01-clean
- Ready for Phase 11 completion (no further plans in this phase)
- Anthropic and Gemini remain documented deferrals (D-11-20, D-11-11) pending credits/quota restoration

---
*Phase: 11-cross-model-baseline-regen-matrix-expansion*
*Completed: 2026-06-11*
