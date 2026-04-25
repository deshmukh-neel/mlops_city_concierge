# W6 — Eval-loop agent

**Branch:** `feature/agent-w6-eval-agent`
**Depends on:** W2 (needs the agent graph to evaluate)
**Unblocks:** —

## Goal

Today, `scripts/log_model_to_mlflow.py` (`scripts/log_model_to_mlflow.py:171-239`) runs hardcoded sample queries against a candidate config and dumps outputs as text artifacts. There are no metrics, no comparison to the current production alias, and no gating on regressions.

This PR adds an eval-loop agent that:
- Runs canonical queries from `configs/experiments.yaml` against a candidate agent build.
- Computes structured metrics: constraint-satisfaction rate, geographic coherence, source diversity, retrieval-similarity statistics, tool-call efficiency.
- Calls an **LLM judge** to score creativity / taste on a fixed rubric.
- Logs metrics + a comparison report to MLflow.
- Refuses to alias-promote a candidate to `@production` if regressions exceed configured thresholds.

After this PR: `python scripts/eval_agent.py --candidate <run-id>` is the single command that runs the full eval and either passes or blocks promotion.

## Files

### New: `scripts/eval_agent.py`

```python
"""Eval-loop agent.

For each canonical query in configs/experiments.yaml:
  1. Run the candidate agent (W2 graph) end-to-end.
  2. Compute structured metrics on the resulting ItineraryState.
  3. Ask an LLM judge for taste / creativity scoring.
  4. Compare to the current @production run.
  5. Block promotion if regression on >=1 critical metric.

Reuses:
  - ExperimentConfig from scripts/log_model_to_mlflow.py:60+
  - log_rag_experiments_from_config() flow at scripts/log_model_to_mlflow.py:242-290
  - Agent graph from app.agent.graph
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
    "constraints_satisfied_mean": 0.05,   # candidate must not be >5 pp worse
    "judge_score_mean": 0.25,             # within 0.25 of current
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

Add a `sample_queries` section that's representative — covers each of the failure modes W3 cares about plus a creative one:

```yaml
sample_queries:
  - "italian dinner around 7pm tonight in north beach, under $$$"
  - "plan a date night: dinner then drinks within walking distance, vibes-y"
  - "best vegan brunch in the mission this saturday"
  - "find a quiet cafe for studying near soma open till late"
  - "a 5-star restaurant open at 4am" # known-bad query, tests self-correction
```

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

## Risks / open questions

- **Judge LLM bias.** Self-judging (using the same model as judge that generated the answer) is known to inflate scores. Use a different provider for the judge than the candidate (e.g. Gemini judge for OpenAI candidate). Configurable in `build_judge_llm()`.
- **Sample query breadth.** Five queries is too few for a real eval. Plan: start at 5 to keep iteration fast, expand to 25-50 once the rubric stabilizes. Add a `sample_size` arg.
- **Cost.** Each eval run = N queries × (candidate calls + judge call). Cap at 10 queries per eval by default, configurable.
- **Promotion semantics.** This PR only sets `@production` alias if `--allow-promote`. Wiring this into a CI/CD gate is a follow-up — out of scope for the agentic-RAG plan.
- **Threshold tuning.** `CRITICAL_METRICS` tolerances (0.05, 0.25) are guesses. After 2-3 real evals we should reset these from observed run-to-run variance.
