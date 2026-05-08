# W6 — Eval-loop agent (RAGAS retrieval + custom itinerary checks)

**Branch:** `feature/agent-w6-eval-agent`
**Depends on:** W2 (needs the agent graph to evaluate), W3 (canonical home for the deterministic checks W6 reuses). Optionally W0a, W7 — eval becomes the source of truth for whether the v2 embeddings and KG are wins.
**Unblocks:** —

## What W3 already shipped (consume; do not duplicate)

W3 landed before W6 and put the deterministic itinerary checks in their canonical home so request-time critique and offline eval share one implementation. When implementing W6, **import from `app/agent/critique/checks.py` rather than re-deriving**:

- `constraints_satisfied(state)`, `geographic_coherence(state)`, `temporal_coherence(state)`, `walking_budget_respected(state)`, `no_hallucinated_place_ids(state)` — all return floats in [0, 1].
- `itinerary_violations(state)` — returns the list of failing check names.
- `CRITIQUE_THRESHOLDS` — the per-check thresholds used at request time. Eval should reuse the same thresholds so an offline-bad plan is the same as an online-bad plan.

The function signatures in this plan's §`app/eval/itinerary_checker.py` block predate W3; treat them as the conceptual contract and **delete that file from the W6 plan** in favor of importing from `app/agent/critique/checks.py`.

W3 also shipped `app/agent/critique/vibe.py` — the cheap-LLM "vibe coherence" judge that runs at request time when `EVAL_VIBE_CRITIQUE_ENABLED=true`. **W3 now also owns judge construction**: `vibe.make_judge()` reads `EVAL_JUDGE_PROVIDER` (default `gemini`) + `EVAL_JUDGE_MODEL` (default `gemini-3.1-flash-lite-preview`, flipped from `gpt-4o-mini` in the W5 PR), and `build_agent_graph(llm, judge_llm=None)` auto-constructs a judge when the env var is on. W6 should **import `vibe.make_judge` rather than re-implementing**:

1. Use `vibe.make_judge()` to build the judge LLM for the offline taste rubric — same model + provider as request-time vibe checks.
2. Don't re-thread it into `build_agent_graph`; that wiring is done.

That keeps a single judge construction site (`app/agent/critique/vibe.py:make_judge`).

W3 also tracks `state.revision_hints` and `state.revision_counts` — useful per-request signals. **MLflow logging of `revisions_per_query` as a request-time metric is W6's territory** (the eval/metrics pipeline is yours). State already has the data; W6 wires the logging hook in `/chat`.

## CI + local — Cloud SQL pattern to reuse

W3 wired up an automated path for tests that need real Cloud SQL data (IAM-DB-auth via the Cloud SQL Auth Proxy + Workload Identity Federation, no static secrets). When W6's eval pipeline needs Cloud SQL — it will, for grounding hallucination checks and for retrieval evals against real `places_raw` rows — **copy the existing pattern instead of inventing new auth**:

- **CI:** the `integration-cloud` job in `.github/workflows/ci.yml` shows the full WIF + proxy + IAM-token setup. It runs on `pull_request` to main. For W6, either extend that job to also run eval scripts, or add an `eval-cloud` job that mirrors the same auth steps.
- **Local:** `make test-integration-cloud` does the same dance on a laptop — spawns the proxy, generates a login token, runs pytest, kills the proxy on exit. Pattern: `gcloud auth login` once → `make` does the rest. W6 should add an analogous `make eval-cloud` (or similar) target that wraps `python scripts/eval_agent.py --candidate <run-id>` in the same proxy lifecycle.
- **Auth setup is one-time per IAM identity.** Both you and the CI service account already have `cloudsql.client` + `cloudsql.instanceUser` and CLOUD_IAM_USER / CLOUD_IAM_SERVICE_ACCOUNT entries on the instance. Onboarding any new contributor: grant the two IAM roles + create the IAM DB user + GRANT SELECT on `public`.

## Parallel work — likely collision points

W4 (booking stub) and W5 (coverage-gap ingestion agent) are independent of W6 and may land in any order. If you're working on W6 in parallel with either:

- **`app/agent/state.py`** — W4 will likely add a `BookingProposal` field or extend `PlaceCard.booking_url`. W6 unlikely to need state changes (eval operates on already-finalized states), but if you do, coordinate with whoever owns W4. Same-line conflicts are rare; module-level additions usually merge cleanly.
- **`app/agent/tools.py`** — W4 appends `propose_booking` to the `_TOOLS` list. W6 doesn't add tools; it imports them for offline eval. Append-only list = low conflict risk.
- **`app/agent/prompts.py`** — W4 may extend `SYSTEM_PROMPT` to mention `propose_booking`. W6 doesn't touch this file.
- **`pyproject.toml`** — W6 adds `ragas` (and its LangChain extras). W4/W5 unlikely to add deps. Run `poetry lock` once on the rebase before PR.
- **`Makefile`** — W6 adds eval targets, W5 adds `make coverage-agent`, W4 may add nothing. Different targets, no conflict.
- **`.github/workflows/ci.yml`** — the existing `integration-cloud` job is the precedent; new eval jobs go below it.

Standard practice: branch off `main`, rebase before opening the PR, no shared "feat: parallel work" branches. The plan files for W4, W5, and W6 each live in different `wN_*.md` files — the `**Status:**` footer + README index update happens per-PR with no cross-collision.

## Goal

Today, `scripts/log_model_to_mlflow.py` (`scripts/log_model_to_mlflow.py:171-239`) runs hardcoded sample queries against a candidate config and dumps outputs as text artifacts. There are no metrics, no comparison to the current production alias, and no gating on regressions.

This PR adds an eval-loop that splits cleanly along the two evaluation problems we actually have:

1. **Retrieval evals** — does the retriever surface relevant places for a query? Solved with **RAGAS** (`faithfulness`, `context_recall`, `context_precision`, `answer_relevancy`). Bootstrap the test set with a hybrid approach: ~20 hand-written queries that mirror real user intents (date night, vegan brunch, late-night cafe) plus ~50 RAGAS-generated queries from RAGAS's `TestsetGenerator` over the v2 chunks for breadth.
2. **End-to-end itinerary evals** — does the planned multi-stop date satisfy the constraints the agent claims to satisfy? RAGAS does not natively cover this; we add a **custom deterministic itinerary checker** that mechanically verifies geographic + temporal coherence + constraint satisfaction. No LLM judge needed for this part — it's all SQL/math.
3. **LLM judge for taste/creativity** — a fixed rubric, scored by a *different* provider than the candidate (avoid self-judging bias). A **cheap small model** (e.g. `gpt-4o-mini` or `gemini-2.5-flash`) is appropriate here; this is a structured-rubric reading task, not deep reasoning.

All three sets of metrics land in MLflow as run metrics. Alias promotion to `@production` is gated on non-regression on critical metrics from each.

After this PR: `python scripts/eval_agent.py --candidate <run-id>` is the single command that runs the full eval and either passes or blocks promotion.

## MLflow param expansion

The registered model's params dict expands to track everything that materially affects results. Without this, A/B comparisons are not apples-to-apples. Add to `scripts/log_model_to_mlflow.py`:

```python
params = {
    # Existing:
    "llm_provider":           cfg.llm_provider,
    "chat_model":             cfg.chat_model,
    "k":                      cfg.k,
    "embedding_model":        cfg.embedding_model,
    "temperature":            cfg.temperature,
    # NEW (this PR):
    "embedding_table":        cfg.embedding_table,        # 'place_embeddings' | '_v2'
    "retrieval_mode":         cfg.retrieval_mode,         # 'vector_only' | 'hybrid' | 'vector_plus_kg'
    "agent_strategy":         cfg.agent_strategy,         # 'single_pass' | 'multi_stop'
    "kg_enabled":             cfg.kg_enabled,             # bool — set True after W7 lands
    "default_num_stops":      cfg.default_num_stops,      # 3
    "walking_budget_m":       cfg.walking_budget_m,       # 2400
    "min_user_rating_count":  cfg.min_user_rating_count,  # 50 (W1 floor)
}
```

`RunConfig` in `scripts/log_model_to_mlflow.py` gains the same fields with defaults, so existing YAML configs keep working.

## Test set (hybrid bootstrap)

```yaml
# configs/eval_queries.yaml
hand_written:
  # Each entry: query + the constraints the agent SHOULD satisfy.
  # The custom checker (below) reads `expected_constraints` and asserts the
  # produced itinerary meets them.
  - query: "italian dinner around 7pm tonight in north beach, under $$$"
    expected_constraints:
      neighborhood:        "North Beach"
      price_level_max:     3
      open_at_iso:         "2026-05-04T19:00:00-07:00"
      types_any:           ["italian_restaurant", "restaurant"]
      min_user_rating_count: 50
    expected_stops: 1

  - query: "plan a date night: dinner then drinks within walking distance, vibes-y"
    expected_constraints:
      price_level_max:     3
      min_user_rating_count: 50
    expected_stops: 2
    expected_walking_budget_m: 1200      # tighter for "walking distance"

  - query: "best vegan brunch in the mission this saturday"
    expected_constraints:
      neighborhood:        "Mission"
      serves_brunch:       true
      serves_vegetarian:   true
    expected_stops: 1

  - query: "find a quiet cafe for studying near soma open till late"
    expected_constraints:
      neighborhood:        "SOMA"
      serves_coffee:       true
    expected_stops: 1

  - query: "a 5-star restaurant open at 4am"   # known-bad, tests self-correction
    expects_clarification_or_relaxation: true

  # ... ~20 total hand-written; expand as we observe real user queries.

generated:
  source_table: place_embeddings_v2
  count:        50
  seed:         42
  # RAGAS TestsetGenerator builds its OWN knowledge graph internally to seed
  # multi-hop questions. That KG is separate from our W7 retrieval-time KG.
  # Both can coexist; do not conflate.
```

## Files

### New: `scripts/eval_agent.py`

```python
"""Eval-loop agent.

Three eval surfaces, all logged to MLflow:

  RETRIEVAL  — RAGAS metrics over retrieved contexts vs reference.
               Metrics: faithfulness, context_recall, context_precision,
                        answer_relevancy.
  ITINERARY  — deterministic checker on the final ItineraryState.
               Metrics: constraints_satisfied, geographic_coherence,
                        temporal_coherence, walking_budget_respected,
                        no_hallucinated_place_ids, source_diversity.
  TASTE      — LLM judge with a fixed rubric, scored by a CHEAP small model
               (e.g. gpt-4o-mini, gemini-2.5-flash). Different provider
               than the candidate to avoid self-judging bias.

For each query in configs/eval_queries.yaml:
  1. Run the candidate agent (W2 graph) end-to-end. Capture retrieved contexts
     from state.scratch and the final itinerary.
  2. Compute RAGAS retrieval metrics on the captured contexts.
  3. Compute itinerary checker metrics on state.stops.
  4. Ask the cheap judge for a taste/creativity score.
  5. Compare aggregates to the current @production run.
  6. Block promotion if regression on >=1 critical metric.

Reuses:
  - ExperimentConfig from scripts/log_model_to_mlflow.py:60+
  - log_rag_experiments_from_config() flow at scripts/log_model_to_mlflow.py:242-290
  - Agent graph from app.agent.graph
  - W1 helpers (haversine, place_is_open) for the deterministic checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import mlflow
from langchain_core.messages import HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.state import ItineraryState
from app.tools.retrieval import get_details
from scripts.log_model_to_mlflow import (
    ExperimentConfig, RunConfig, load_experiment_config, build_llm_for_run,
)


# ----- metrics --------------------------------------------------------------

@dataclass
class QueryEval:
    query: str
    constraints_satisfied: float    # 0..1
    geographic_coherence: float     # 0..1, 1.0 if every consecutive pair < 1km
    source_diversity: float         # 0..1, 1.0 if mix of editorial+google
    avg_similarity: float
    tool_calls: int
    judge_score: float              # 0..5
    judge_rationale: str
    stops_count: int
    revision_hints: int


@dataclass
class RunReport:
    run_id: str
    run_name: str
    queries: list[QueryEval]
    aggregate: dict


def evaluate_run(cfg: RunConfig, queries: list[str]) -> RunReport:
    llm = build_llm_for_run(cfg)
    graph = build_agent_graph(llm)
    judge = build_judge_llm()
    evals: list[QueryEval] = []
    for q in queries:
        state = graph.invoke(ItineraryState(messages=[HumanMessage(q)]))
        evals.append(_eval_one(q, state, judge))
    aggregate = _aggregate(evals)
    return RunReport(run_id="", run_name=cfg.run_name(), queries=evals, aggregate=aggregate)


def _eval_one(query: str, state: ItineraryState, judge) -> QueryEval:
    constraints_satisfied = _constraint_satisfaction(state)
    coherence = _geographic_coherence(state)
    diversity = _source_diversity(state)
    avg_sim = _avg_similarity(state)
    judge_score, rationale = _judge(query, state, judge)
    return QueryEval(
        query=query,
        constraints_satisfied=constraints_satisfied,
        geographic_coherence=coherence,
        source_diversity=diversity,
        avg_similarity=avg_sim,
        tool_calls=sum(len(v) for v in state.scratch.values()),
        judge_score=judge_score,
        judge_rationale=rationale,
        stops_count=len(state.stops),
        revision_hints=len(state.revision_hints),
    )


def _constraint_satisfaction(state: ItineraryState) -> float:
    """For each user constraint (price, rating, open_at, neighborhood),
    fraction of stops that satisfy it. Pulls fresh details for each stop."""
    c = state.constraints
    if not state.stops:
        return 0.0
    total = 0
    sat = 0
    for s in state.stops:
        d = get_details(s.place_id)
        if d is None:
            continue
        if c.price_level_max is not None:
            total += 1
            sat += int((d.price_level or 0) <= c.price_level_max)
        if c.min_rating is not None:
            total += 1
            sat += int((d.rating or 0) >= c.min_rating)
        if c.neighborhood:
            total += 1
            sat += int(c.neighborhood.lower() in (d.formatted_address or "").lower())
        # open_at via place_is_open SQL function — call directly
        if c.when is not None:
            total += 1
            sat += int(_is_open_at(d.place_id, c.when))
    return (sat / total) if total else 1.0


def _geographic_coherence(state: ItineraryState) -> float:
    if len(state.stops) < 2:
        return 1.0
    # Use haversine on lat/lng from get_details
    coords = []
    for s in state.stops:
        d = get_details(s.place_id)
        if d and d.latitude and d.longitude:
            coords.append((d.latitude, d.longitude))
    if len(coords) < 2:
        return 1.0
    pairs = zip(coords, coords[1:], strict=False)
    near = sum(1 for a, b in pairs if _haversine_m(a, b) <= 1500)
    return near / max(len(coords) - 1, 1)


def _source_diversity(state: ItineraryState) -> float:
    sources = {s.source for s in state.stops}
    return min(len(sources) / 2, 1.0)


def _avg_similarity(state: ItineraryState) -> float:
    sims = []
    for entries in state.scratch.values():
        for e in entries:
            r = e.get("result")
            if isinstance(r, list):
                sims.extend(getattr(h, "similarity", 0.0) for h in r)
    return sum(sims) / len(sims) if sims else 0.0


JUDGE_PROMPT = """Evaluate this restaurant/itinerary recommendation. Rubric (0-5):
0 = useless or hallucinated; 5 = thoughtful, surprising, well-justified.

User query: {query}

Agent's plan:
{plan}

Reasoning trace summary:
{trace}

Return JSON: {{"score": float, "rationale": "one sentence"}}.
"""


def _judge(query: str, state: ItineraryState, judge) -> tuple[float, str]:
    plan = "\n".join(f"- {s.name} ({s.arrival_time}): {s.rationale}" for s in state.stops)
    trace = f"{sum(len(v) for v in state.scratch.values())} tool calls, {len(state.revision_hints)} revisions"
    resp = judge.invoke(JUDGE_PROMPT.format(query=query, plan=plan, trace=trace)).content
    obj = json.loads(resp)
    return float(obj["score"]), obj["rationale"]


def _aggregate(evals: list[QueryEval]) -> dict:
    n = len(evals) or 1
    return {
        "constraints_satisfied_mean": sum(e.constraints_satisfied for e in evals) / n,
        "geographic_coherence_mean": sum(e.geographic_coherence for e in evals) / n,
        "source_diversity_mean": sum(e.source_diversity for e in evals) / n,
        "avg_similarity_mean": sum(e.avg_similarity for e in evals) / n,
        "judge_score_mean": sum(e.judge_score for e in evals) / n,
        "tool_calls_mean": sum(e.tool_calls for e in evals) / n,
        "stops_mean": sum(e.stops_count for e in evals) / n,
    }


# ----- comparison + gating --------------------------------------------------

CRITICAL_METRICS = {
    # RAGAS retrieval quality
    "ragas_context_recall_mean":     0.05,
    "ragas_context_precision_mean":  0.05,
    "ragas_faithfulness_mean":       0.05,
    # Itinerary correctness
    "constraints_satisfied_mean":    0.05,   # candidate must not be >5 pp worse
    "geographic_coherence_mean":     0.05,
    "temporal_coherence_mean":       0.05,
    "no_hallucinated_place_ids":     0.0,    # zero tolerance — must be 1.0
    # LLM-judge taste (cheap judge)
    "judge_score_mean":              0.25,   # within 0.25 of current
}


def compare_and_gate(
    candidate: RunReport, baseline: Optional[RunReport],
) -> tuple[bool, list[str]]:
    if baseline is None:
        return True, ["no baseline; promotion allowed"]
    failures = []
    for metric, tolerance in CRITICAL_METRICS.items():
        c = candidate.aggregate[metric]
        b = baseline.aggregate[metric]
        if c + tolerance < b:
            failures.append(f"{metric}: candidate {c:.3f} < baseline {b:.3f} - {tolerance}")
    return (not failures), failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments.yaml")
    p.add_argument("--candidate-run", required=True)
    p.add_argument("--baseline-alias", default="production")
    p.add_argument("--allow-promote", action="store_true",
                   help="If checks pass, set @production alias to candidate.")
    args = p.parse_args(argv)

    cfg: ExperimentConfig = load_experiment_config(args.config)
    candidate_cfg = _resolve_run_config(args.candidate_run, cfg)
    baseline_cfg = _resolve_alias(args.baseline_alias)

    candidate = evaluate_run(candidate_cfg, cfg.sample_queries)
    baseline = evaluate_run(baseline_cfg, cfg.sample_queries) if baseline_cfg else None

    ok, failures = compare_and_gate(candidate, baseline)
    _log_to_mlflow(candidate, baseline, ok, failures)

    if ok and args.allow_promote:
        _set_production_alias(args.candidate_run)
        print("PROMOTED")
    elif ok:
        print("PASSED (use --allow-promote to set alias)")
    else:
        print("BLOCKED:")
        for f in failures:
            print("  -", f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

(Helpers `_haversine_m`, `_is_open_at`, `build_judge_llm`, `_resolve_run_config`, `_resolve_alias`, `_log_to_mlflow`, `_set_production_alias` are tiny and fully implementable from the surrounding context — included as stubs in the actual implementation.)

### Modify: `scripts/log_model_to_mlflow.py`

Add an `--eval` flag that, after logging a run, invokes the eval agent against the just-created run and prints the report:

```python
parser.add_argument("--eval", action="store_true",
                    help="Run eval_agent against the new run; block promotion on regression.")
# After log_rag_experiments_from_config():
if args.eval:
    from scripts.eval_agent import main as eval_main
    eval_main(["--candidate-run", new_run_id])
```

This is the single hook that makes the eval gate part of the normal experiment flow.

### Modify: `configs/experiments.yaml`

`sample_queries` is moved to the new `configs/eval_queries.yaml` (see "Test set" section above). The old key still works during the transition; the loader checks the new path first.

### New: `app/eval/ragas_runner.py`

Wraps RAGAS so the rest of the codebase doesn't take a hard dependency on its internals.

```python
"""Compute RAGAS retrieval metrics from agent runs.

We capture (query, retrieved_contexts, agent_answer, reference) tuples by
running the agent and pulling retrieved snippets out of state.scratch
(semantic_search and nearby results). RAGAS computes:
  - faithfulness        — does the answer stay grounded in retrieved contexts?
  - context_recall      — did retrieval cover what the reference says?
  - context_precision   — were retrieved contexts on-topic?
  - answer_relevancy    — does the answer address the query?

References for hand-written queries come from configs/eval_queries.yaml.
References for generated queries come from RAGAS's TestsetGenerator output.
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)
# ...
```

The runner returns a dict of metric → mean float. The eval agent merges these into the same metric namespace as the deterministic + judge metrics so MLflow sees one flat dict.

### Itinerary checker: import from `app/agent/critique/checks.py` (W3)

W3 already shipped these functions in their canonical home. **W6 imports from there; it does not create `app/eval/itinerary_checker.py`**. The five W6-shipped functions live as:

```python
from app.agent.critique.checks import (
    constraints_satisfied,
    geographic_coherence,
    temporal_coherence,
    walking_budget_respected,
    no_hallucinated_place_ids,
    itinerary_violations,
    CRITIQUE_THRESHOLDS,
)
```

Each returns a float in [0, 1]. `itinerary_violations(state)` returns the list of failing check names — useful for W6 to derive a "violation rate" per query without re-running each check.

`source_diversity(state)` (mentioned in earlier drafts of this plan) is **not** in W3 because there's only one source today. If W6 wants it, add it directly to `app/agent/critique/checks.py` so the request-time critique can pick it up too.

### Modify: `pyproject.toml`

Add RAGAS to the `dev` dependency group:

```toml
ragas = "^0.2.0"
```

Note: RAGAS depends on a few LangChain extras and an LLM provider for its judges. Configure RAGAS to use the same cheap judge model we use for taste scoring — controlled by `EVAL_JUDGE_MODEL` env var.

## Tests

### New: `tests/unit/test_eval_agent_metrics.py`

Pure-function tests on the metric calculations:

```python
def test_constraint_satisfaction_perfect(monkeypatch):
    state = ItineraryState(
        constraints=UserConstraints(price_level_max=2, min_rating=4.0,
                                    neighborhood="Mission"),
        stops=[Stop(place_id="p1", name="X", source="google_places", rationale="")],
    )
    monkeypatch.setattr("scripts.eval_agent.get_details",
        lambda pid: PlaceDetails(place_id=pid, name="X", source="google_places",
                                 similarity=0.0, primary_type="r",
                                 formatted_address="Mission, San Francisco",
                                 latitude=0, longitude=0, rating=4.5, price_level=2,
                                 business_status="OPERATIONAL", types=[],
                                 user_rating_count=10, website_uri=None,
                                 maps_uri=None, editorial_summary=None,
                                 regular_opening_hours={}, snippet=None))
    assert _constraint_satisfaction(state) == 1.0


def test_geographic_coherence_under_threshold(): ...
def test_source_diversity_zero_when_single_source(): ...
def test_aggregate_correctness(): ...
def test_compare_and_gate_blocks_on_judge_score_regression():
    candidate_agg = {"constraints_satisfied_mean": 0.9, "judge_score_mean": 3.0}
    baseline_agg  = {"constraints_satisfied_mean": 0.9, "judge_score_mean": 4.5}
    candidate = RunReport("c", "c", [], candidate_agg)
    baseline  = RunReport("b", "b", [], baseline_agg)
    ok, failures = compare_and_gate(candidate, baseline)
    assert not ok
    assert any("judge_score_mean" in f for f in failures)
```

### Integration test (gated)

`tests/integration/test_eval_agent_e2e.py` — run a tiny config with 2 sample queries against a seeded DB, confirm an MLflow run is created with metric tags, confirm exit code is 0 when no baseline exists.

## Manual verification

```bash
# Log a candidate run from a YAML config:
python scripts/log_model_to_mlflow.py --config configs/experiments.yaml

# Pull the new run id from the output, then:
python scripts/eval_agent.py --candidate-run <run-id>

# Force a regression:
# Edit experiments.yaml to use a known-worse model, log it, eval — expect BLOCKED.

# Combined flow:
python scripts/log_model_to_mlflow.py --config configs/experiments.yaml --eval --allow-promote
```

In the MLflow UI, the run should have:
- Tags: `constraints_satisfied_mean`, `geographic_coherence_mean`, `judge_score_mean`, etc.
- Artifact: `comparison_report.json` showing candidate vs baseline.
- Run name format inherited from `scripts/log_model_to_mlflow.py:203`.

## Why MLflow + RAGAS, not MLflow's GenAI eval features

MLflow added `mlflow.evaluate(..., model_type="question-answering")` and a managed eval UI on top of the registry. We deliberately keep the split:

- **MLflow** is the experiment store and model registry. It tracks runs, params, metrics, and which config is `@production`. That role does not change.
- **RAGAS** is a metric library — it computes scores. Scores get logged to MLflow as run metrics.
- **The custom itinerary checker** computes scores no general library covers. Also logged to MLflow.

One pipeline, three metric sources, MLflow as the single system of record. We don't introduce a third tool (Langfuse / Braintrust) in this PR — that's optional follow-up if/when MLflow's UI proves inadequate for diffing N agent runs.

## Future direction: DSPy + a managed eval UI

This v1 hand-rolls metric calculations and a JSON judge. Two upgrades worth knowing about for the next iteration of W6, both deliberately out of scope here:

- **DSPy for programmatic prompt optimization.** DSPy is purpose-built for "agent quality validation, regression testing, or research" and lets you optimize prompts against the metrics this PR defines (constraint satisfaction, judge score) using algorithms like MIPRO/BootstrapFewShot. Once we have ≥3 evals worth of run-to-run data, replacing the static system prompt in `app/agent/prompts.py` with a DSPy-compiled version is a natural next step. Keep this PR's metrics; swap the prompt source.
- **Eval UI (Langfuse / LangSmith / Braintrust) instead of MLflow artifacts.** MLflow is great as a model registry but its UI is poor for diffing N agent runs across M queries with full traces. Langfuse (self-hostable, free tier) and Braintrust both offer purpose-built eval UIs that can ingest the metrics this PR computes. For now we keep MLflow as the gating mechanism (single source of truth for `@production`); a follow-up PR can mirror the same metrics to Langfuse/Braintrust for richer review.

## Risks / open questions

- **Judge LLM bias.** Self-judging (using the same model as judge that generated the answer) is known to inflate scores. Use a different provider for the judge than the candidate (e.g. Gemini judge for OpenAI candidate). Configurable in `build_judge_llm()`.
- **Sample query breadth.** Five queries is too few for a real eval. Plan: start at 5 to keep iteration fast, expand to 25-50 once the rubric stabilizes. Add a `sample_size` arg.
- **Cost.** Each eval run = N queries × (candidate calls + judge call). Cap at 10 queries per eval by default, configurable.
- **Promotion semantics.** This PR only sets `@production` alias if `--allow-promote`. Wiring this into a CI/CD gate is a follow-up — out of scope for the agentic-RAG plan.
- **Threshold tuning.** `CRITICAL_METRICS` tolerances (0.05, 0.25) are guesses. After 2-3 real evals we should reset these from observed run-to-run variance.
- **Cheap judge nuance.** A small model (gpt-4o-mini / gemini-2.5-flash) is fine for rubric-scored taste judgments — it's a structured reading task — but can underweight subtle creativity. Track judge agreement with a manual spot-check on 10% of runs; if drift exceeds 0.5 points consistently, swap to a mid-tier judge for the taste pass only (RAGAS judges can stay on the cheap model).
- **RAGAS internal KG vs our retrieval-time KG.** RAGAS's `TestsetGenerator` builds a knowledge graph internally to seed multi-hop questions. That KG is throw-away — used only at testset generation time. Our W7 `place_relations` KG is a runtime retrieval enhancement. They never interact. Documented to avoid future confusion.
- **`embedding_table` flip is gated by this PR.** Promoting `EMBEDDING_TABLE=place_embeddings_v2` to production happens via a candidate run logged with `embedding_table=place_embeddings_v2`, evaluated, and alias-promoted only if RAGAS retrieval metrics + itinerary metrics are non-regressing. No env-var change without an eval.
