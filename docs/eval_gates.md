# Eval Gates

This document explains the merge-gate semantics for the eval matrix.
Numbers live in `configs/eval_gates.yaml`; this file explains only the semantics.
Do not add numeric gate values here — edit the YAML with a D-ID rationale instead.

## Gate statuses

Each entry in `configs/eval_gates.yaml` carries a `status` that controls how
`scripts/check_eval_gates.py` (invoked via `make eval-gates-check`) treats it:

- `active` — hard gate enforced by `make eval-gates-check`; a cell below the gate
  value causes exit 1 (HARD GATE VIOLATION).
- `aspirational` — reported but not blocking. The gate value represents a v2.2
  decisiveness target that the model currently misses. `check_eval_gates.py` prints
  an ASPIRATIONAL miss line to stdout and returns exit 0, so Phase 10/11 work is not
  blocked on a known gap.
- `provisional-n1` — hard gate derived from a single evaluation run. Phase 11 will
  re-ratify the value at n=5 and may promote or demote it. Enforced the same way as
  `active` until then.
- `logged` — no gate; the empirical median is captured for reference only.
  `check_eval_gates.py` skips these entries entirely.
- `quarantined-legacy-threading` — excluded from baselines and gates. The scenario
  uses a conversation-history shape that does not match production threading
  (see D-10-09). The scenario stays runnable as a diagnostic but does not contribute
  to any baseline JSON or gate check.

## The strict refinement_minimal_edit == 1.0 gate is formally retired

The Phase-6-era strict gate on `refinement_minimal_edit == 1.0` is retired (D-10-06).
It was authored against a fail-open baseline where errored runs reported 1.0 scores.
After the Phase-7 scorer tightening (D-07-05/D-07-07) the honest anchor
(`openai/gpt-4o-mini`) sits at median 0.0 / max 0.5 — making the strict gate
permanently unsatisfiable without relaxing the scorer.

The gate also fossilized because the numeric value lived in a Makefile comment, not in
a single inspectable source of truth. That failure mode is eliminated: all gate values
now live only in `configs/eval_gates.yaml`, and this doc links to the YAML rather than
duplicating numbers.

`refinement_minimal_edit` medians are now advisory everywhere until v2.2 decisiveness
work earns a credible floor (see `configs/eval_gates.yaml` advisory entries).

## Running the gate check

```bash
make eval-gates-check SUMMARY=eval_reports/{ts}/summary.json
```

The checker loads `configs/eval_gates.yaml` and the provided `summary.json`, then
classifies each gate entry:

- `active` / `provisional-n1` with a failing hard gate → exit 1 (HARD GATE VIOLATION)
- `aspirational` with a failing hard gate → exit 0 with ASPIRATIONAL miss printed
- `logged` / `quarantined-legacy-threading` → skipped entirely

If `committed_itinerary_rate` is not yet present in the summary (Phase 10 — the metric
is wired by Phase 11 BASE-01), the gate is reported as not-evaluable rather than
silently passing. This is by design: the gate infrastructure is complete and Phase 11
will wire the metric.

Exit codes match `check_baselines_fresh.py`:

| Code | Meaning |
|------|---------|
| 0    | All hard gates passed (aspirational misses are printed but non-blocking) |
| 1    | One or more hard-gate violations |
| 2    | Infrastructure failure (missing YAML, unreadable summary.json) |

## Adding a new gate

1. Edit `configs/eval_gates.yaml` — add a new entry with all D-10-08 fields.
2. Supply a `rationale` one-liner that cites the D-ID backing the gate value.
3. Do not add the numeric value to this document or to any Makefile comment.

Aspirational gates (status: aspirational) should be used for v2.2 targets that are
known to fail today. They keep the gap visible in CI output without blocking merges.
