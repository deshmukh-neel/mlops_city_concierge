# Phase 5: Rationale-Stop Alignment Fix - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

After the closure-aware swap node auto-swaps a closed stop for a walking-distance
alternative, the user must never see the placeholder rationale
`"Walking-distance alternative for {closed_stop.name}"` in the final reply. The
committed swap candidate must carry a real, stop-specific description (or no
rationale at all) before `summarize_stops` renders the user-facing itinerary.

Concretely, Phase 5 delivers a one-line code change in `app/agent/swap.py`
(stop writing the placeholder rationale on candidate Stops at line 238) plus
whatever downstream change is needed so the rendered itinerary still reads
naturally without that placeholder — most likely either a fresh deterministic
rationale stamped during `enrich_stops_with_booking` for swap-origin stops, or
falling through `summarize_stops`'s existing `f" {s.rationale}" if s.rationale
else ""` handling so a None rationale is omitted gracefully.

Scope anchor: **RAT-02 only**, per REQUIREMENTS.md and ROADMAP.md. RAT-01
(refinement-turn rationale integrity) and RAT-03 (`rationale_stop_alignment`
merge-gate metric) were transferred to Phase 4 per D-04-09 and D-04-13 — they
are out of scope here. The `late_night_closure_cascade` gating decision deferred
by D-04-12 is also out of scope: it depends on fixing the eval-harness
multi-turn threading bug (see `project_eval_multi_turn_threading_bug.md`),
which is its own work item and was deferred during Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Discussion mode

- **D-05-01:** **No discuss-phase deep-dive performed.** ROADMAP.md Phase 5
  entry explicitly says `Discuss-phase: Not needed`. User confirmed during the
  Phase 5 discuss invocation by selecting "Skip discussion, go to plan" — the
  scope is well-bounded by a single requirement (RAT-02) with a single
  identified source site (`app/agent/swap.py:238`), and the requirement text
  itself names the two viable implementation paths.

### Implementation path (deferred to planner)

- **D-05-02:** The implementation path between (a) generating a fresh
  rationale inside the swap node before `enrich_stops_with_booking` runs,
  versus (b) extending `enrich_stops_with_booking` to overwrite the
  swap-origin placeholder, versus (c) setting the candidate's rationale to
  `None` in `swap.py` and letting `summarize_stops`'s existing
  `if s.rationale else ""` branch render the stop without it — is **left to
  the planner**. All three satisfy RAT-02. The planner picks based on which
  has the smallest blast radius on existing tests and which produces the most
  natural user-facing output. The pre-condition every path must satisfy: the
  user-visible final reply for an auto-swapped stop must not contain the
  substring `"Walking-distance alternative for"`.

### Eval / regression guard

- **D-05-03:** No new merge gate. Phase 4 already enforced
  `rationale_stop_alignment ≥ 0.8 absolute floor` on the gated scenarios
  (D-04-14), and `rationale_stop_alignment`'s docstring at
  `app/agent/critique/checks.py:336-341` explicitly documents that the
  placeholder fails the scorer ("'Walking-distance alternative for
  {closed_stop.name}'. That string names the CLOSED stop, not the swap
  candidate, and contains no family keyword — so this scorer returns 0.0 for
  any stop that still carries the placeholder when committed"). Phase 5 must
  ship a regression test that triggers the auto-swap path on a committed
  itinerary and asserts the rendered final_reply does not contain the
  placeholder substring AND that `rationale_stop_alignment(state) == 1.0`
  post-swap. No baseline re-snapshot required — the existing scorers cover
  this once a regression test exists.

### Claude's Discretion (planner-level)

- Whether the regression test lives as a unit test against `swap_closed_stops`
  with a hand-built `ItineraryState`, a functional test against the full
  graph with a scripted-LLM provider, or both — per the test-layering
  preference (`feedback_test_layering.md`), at minimum a unit test + one
  functional test that exercises the swap path end-to-end.
- Exact wording of any replacement rationale (if path (a) or (b) is chosen) —
  must be deterministic (no LLM call inside swap.py — that path is rejected
  for latency reasons even though not formally locked, since the swap node
  already runs after the user's request is otherwise complete and adding a
  per-candidate LLM call regresses tail latency on the closure path).
- Whether the swap node should also re-run `rationale_stop_alignment` on the
  post-swap state defensively as a runtime guard — nice-to-have, not required.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner) MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` § "Phase 5: Rationale-Stop Alignment Fix" — success criteria, branch name (`feature/v2-rationale-alignment`), scope narrowing per D-04-09
- `.planning/REQUIREMENTS.md` § "Rationale-Stop Alignment Fix" — RAT-02 (the sole in-scope requirement); RAT-01 + RAT-03 are explicitly out of scope and owned by Phase 4
- `.planning/PROJECT.md` § "Current Milestone: v2.0 Production Readiness" — milestone-level context; this phase is bug-fix #2 of 3 in the v2.0 behavior-bug slate
- `.planning/phases/04-category-compliance-fix/04-CONTEXT.md` § D-04-09, D-04-12, D-04-13, D-04-14 — locks the Phase 4↔Phase 5 split, the closure-cascade deferral, and the `rationale_stop_alignment` gating decision

### The bug surface (the one file Phase 5 most likely touches)
- `app/agent/swap.py:238` — the literal source of the placeholder string:
  `rationale=f"Walking-distance alternative for {closed_stop.name}"`. This is
  inside `_candidates_to_matches`, which runs for every walking-distance
  candidate before scoring. The winning match's Stop carries this rationale
  through `swap_closed_stops` → `enrich_stops_with_booking` → `summarize_stops`.

### The downstream render path (where the placeholder bleeds through)
- `app/agent/swap.py::swap_closed_stops` (lines 555-679) — the LangGraph node
  that drives the swap; the candidate Stop becomes `working_stops[idx]` at
  line 614, then `enrich_stops_with_booking(retimed, state)` at line 626,
  then `summarize_stops(probe_state)` at line 673 produces the final_reply.
- `app/agent/commit.py::enrich_stops_with_booking` (lines 76-143) — currently
  overwrites `name`, `primary_type`, `address`, `rating`, `price_level`,
  `latitude`, `longitude`, `booking_url`, `booking_provider`. Does NOT touch
  `rationale`. This is the most natural extension point if path (b) is
  chosen — a 2-line addition that conditionally overwrites a placeholder
  rationale from PlaceDetails fields.
- `app/agent/revision.py::summarize_stops` (lines 295-311) — renders
  `f"{i}. {s.name}{timing}.{rationale}"` where rationale is
  `f" {s.rationale}" if s.rationale else ""`. Already gracefully handles
  empty/None rationale; path (c) (set rationale to None in swap.py) works
  here with zero summarize_stops change.

### The eval scorer that already catches this
- `app/agent/critique/checks.py::rationale_stop_alignment` (lines 324-351) —
  the existing Phase 3 scorer. Its docstring explicitly documents the
  closure-swap placeholder as failure mode #2. Returns 0.0 for any stop
  whose rationale still carries the placeholder. Phase 5 regression test
  asserts this returns 1.0 post-fix.
- `app/agent/critique/checks.py::is_rationale_aligned` (lines ~300-321) —
  per-stop helper called by the scorer; same logic, exposed for the
  revision dispatcher in Phase 4's `rationale_misaligned` hint flow.

### Project memory — must-read before planning
- `project_eval_multi_turn_threading_bug.md` — why `late_night_closure_cascade`
  remains ungated and why we are NOT chasing that fix in Phase 5
- `project_critique_commit_conflict.md` — historical context on
  critique↔commit interactions; informs why path (a) doesn't add an LLM call
  inside the swap path
- `feedback_test_layering.md` — at minimum unit + functional test for the
  new behavior

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `app.tools.retrieval.get_details_many` — already called by
  `enrich_stops_with_booking`; the PlaceDetails it returns has `primary_type`,
  `formatted_address`, `name` — enough to synthesize a deterministic
  swap-origin rationale (e.g., `"{primary_type} near {neighborhood-or-address-fragment} (replaces closed {old_name})"`) without an LLM call.
- `app/tools/filters.py::family_of` and `_PRIMARY_TYPE_FAMILIES` — give the
  scorer the keyword vocabulary that any replacement rationale needs to hit
  for `rationale_stop_alignment` to return 1.0. The replacement must either
  include the new stop's `name` (case-insensitive substring) OR a keyword
  from `family_of(new_primary_type)`'s family. Easiest path: include
  `{name}` in the new rationale.
- `app/agent/revision.py::summarize_stops` — already handles
  `if s.rationale else ""`. Path (c) needs no change here.

### Established Patterns
- Swap node already calls `enrich_stops_with_booking(retimed, state)` at
  `swap.py:626` immediately before computing `final_reply` — this is the
  natural single point of mutation for path (b). Extending it (rather than
  adding a new pass) keeps the swap path's DB read count unchanged.
- The closed stop's original `rationale` was already on `state.stops[idx]`
  before swap — it was authored by the planning LLM and was valid for the
  slot. Inheriting it (path (b) with `candidate.rationale = closed.rationale`
  when closed.rationale is non-empty) is the simplest, lowest-blast-radius
  option and almost certainly satisfies `rationale_stop_alignment` for the
  swap candidate (since the rationale was written for the same family/slot).
- Tests follow the layering memory (`feedback_test_layering.md`): unit +
  smoke + functional + integration. The fix is small enough that
  unit + functional is the right balance.

### Integration Points
- The fix lives entirely inside `app/agent/swap.py` (and possibly
  `app/agent/commit.py`). Nothing in `app/main.py`, `app/agent/graph.py`, or
  `app/agent/prompts.py` needs to change — the surface area is contained.
- No schema changes, no migration, no eval config changes (the existing
  Phase 3 scorer + Phase 4 baseline floor already cover the regression).
- No new MLflow params/tags needed.

</code_context>

<specifics>
## Specific Ideas

- **Recommended path (planner's call):** path (b)-inherit — in
  `swap.py::_candidates_to_matches`, change line 238 to inherit the closed
  stop's existing rationale (`rationale=closed_stop.rationale or None`)
  instead of writing the placeholder. The closed stop's rationale was
  written by the planning LLM for that slot, on a candidate of the same
  family in walking distance — it remains substantively correct for the
  swap candidate. Smallest diff, zero new code paths, satisfies
  `rationale_stop_alignment` because the closed stop's rationale already
  passed the scorer (else the original plan wouldn't have committed).

- **Fallback if inheritance proves wrong:** if the closed stop's rationale
  literally names the closed stop (e.g., "Mission Street Oyster Bar's
  raw bar is the standout"), inheritance puts a wrong-name rationale on
  the new stop and fails the scorer the other way. In that case, set
  `rationale = None` in swap.py (path (c)) and let `summarize_stops`
  render the swap candidate as a bare `"{i}. {name} — arrive {when}, ~{N}
  min."` line. Less informative, but never wrong.

- **Regression test shape:** unit test against
  `_candidates_to_matches` asserting the produced Stop's rationale does
  not contain "Walking-distance alternative for"; plus one functional
  test that runs `swap_closed_stops` on an ItineraryState with one
  closed stop + at least one walking-distance candidate, then asserts
  (a) the final_reply contains no "Walking-distance alternative for"
  substring, and (b) `rationale_stop_alignment(post_state) == 1.0`.

</specifics>

<deferred>
## Deferred Ideas

- **Closure-cascade gating decision (D-04-12 follow-up).** Whether to fix
  the eval-harness multi-turn threading bug
  (`project_eval_multi_turn_threading_bug.md`), flag the scenario as
  eval-only shape, or accept the caveat — deferred. Phase 4 punted this to
  Phase 5; Phase 5 punts it forward because RAT-02 is a single-turn
  closure-swap bug, not a multi-turn threading bug, and conflating the two
  inflates Phase 5's scope unnecessarily. Worth its own focused phase or
  backlog item once v2.0 ships.

- **`rationale_stop_alignment` metric redefinition (RAT-03 revisit).**
  D-04-14 logged that the saturated baseline reflects scorer abstention,
  not "rationale-alignment quality on committed itineraries." This is
  Phase 4's flag, not Phase 5's, but mentioned here so the planner doesn't
  re-discover it.

- **LLM-generated swap rationales.** Per-candidate LLM call to generate a
  fresh rationale was considered (path (a) with LLM) and rejected in
  D-05-02's discretion notes — adds tail latency on the closure path and
  costs MLflow tokens for marginal user-facing benefit when the
  inherit-from-closed-stop path likely suffices. Revisit only if the
  inherit path produces awkward rationales in real traffic.

- **REF-01 / REF-02 / REF-03 / REF-04 (Phase 6 — Minimal-Edit Refinement).**
  Out of scope. Phase 6 work.

</deferred>

---

*Phase: 05-Rationale-Stop Alignment Fix*
*Context gathered: 2026-05-27*
