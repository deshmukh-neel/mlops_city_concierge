# Pre-regen baseline snapshots

This directory holds **reference-only** snapshots of `configs/eval_baselines/*.json`
captured immediately before a re-baseline. Snapshots provide a stable
historical floor that downstream phases can diff against when verifying
no-regression vs. an earlier era's scorer medians.

## What this directory IS

- A stash for pre-regeneration baseline copies (e.g.
  `refinement_cheaper.pre-phase6.json`).
- Read by phase-level no-regression checks during PR review (e.g.
  Phase 6 plan 06-07 Task 3 diffs the regenerated `refinement_cheaper.json`
  vs. the pre-Phase-6 snapshot stored here).

## What this directory is NOT

- **NOT consumed by the eval matrix.** `scripts/eval_matrix.py` only reads
  `configs/eval_baselines/<scenario_id>.json` (top-level, canonical
  scenario IDs). Files under `_snapshots/` are invisible to the runner.
- **NOT a baseline.** The stale-baseline freshness lint at
  `scripts/check_baselines_fresh.py` treats *any* changed file under
  `configs/eval_baselines/` (including this subdirectory) as a "baseline
  refresh" for satisfying the gate, which is the *desired* behavior: a
  snapshot commit is itself evidence of an intentional baseline refresh
  in the same PR. The snapshot does not, however, replace the canonical
  baseline.
- **NOT a place to delete files.** Snapshots are append-only. Once a
  phase ships, future phases may reference its snapshot as the floor for
  their own no-regression checks.

## Naming convention

```
<scenario_id>.<phase_or_milestone>.json
```

Examples:

- `refinement_cheaper.pre-phase6.json` — pre-Phase-6 snapshot of the
  `refinement_cheaper` baseline (legacy-threading, fail-open saturated).

## Lifecycle

1. Before regenerating a canonical baseline, copy it here with a
   `pre-<phase>` (or `pre-<milestone>`) suffix.
2. Regenerate the canonical baseline under the new contract.
3. Reference the snapshot from the phase's plan / SUMMARY for the
   no-regression check.
4. Leave the snapshot in place after the phase ships — future phases may
   need it as a historical floor.
