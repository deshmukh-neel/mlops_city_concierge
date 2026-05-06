# W0 — Infra hardening for the agent product

**Branch:** `feature/agent-w0-infra`
**Depends on:** nothing
**Unblocks:** all other workstreams (recommended to land first or in parallel with W1)

## Goal

Your current infra (Cloud Run + Cloud SQL + GitHub Actions + shared GCP MLflow) is solid bones for a class project, but a few items will hurt the agent product specifically — slow first impressions, no debuggability when the agent does something weird, MLflow security gaps, and unbounded cost. This PR addresses the load-bearing items before we ship the agent in W2.

After this PR:
- Cloud Run cold starts no longer ruin the first request.
- Every agent request has a trace you can replay (which tool calls happened, what they returned, where it went wrong).
- MLflow is no longer a hardcoded public IP with no auth.
- Cloud Run concurrency is tuned for the agent workload (not the legacy thin REST endpoint).
- Secrets live in GCP Secret Manager, not raw env vars.
- Per-request token + cost telemetry lands in logs from day one.

Items deferred (real, but not blocking the demo): streaming responses, embedding cache, Cloud SQL private VPC audit, per-user rate limiting. Each is a one-paragraph follow-up; flagged at the end of this doc.

## Files

### 1. Cloud Run min-instances + concurrency

**File:** the Cloud Run deploy step in your auto-deploy workflow (likely `.github/workflows/<deploy>.yml` — confirm filename when implementing). The recent commit `3d42d72 ci: auto-deploy to Cloud Run on merge to main` added it.

Add to the `gcloud run deploy` invocation:

```yaml
- name: Deploy to Cloud Run
  run: |
    gcloud run deploy city-concierge-app \
      --image=$IMAGE \
      --region=$REGION \
      --min-instances=1 \
      --max-instances=10 \
      --concurrency=15 \
      --cpu=2 --memory=2Gi \
      --timeout=120 \
      --service-account=$RUN_SA \
      --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest,\
GEMINI_API_KEY=GEMINI_API_KEY:latest,\
ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,\
LANGFUSE_SECRET_KEY=LANGFUSE_SECRET_KEY:latest,\
LANGFUSE_PUBLIC_KEY=LANGFUSE_PUBLIC_KEY:latest,\
MLFLOW_TRACKING_TOKEN=MLFLOW_TRACKING_TOKEN:latest \
      --set-env-vars=MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI,\
MLFLOW_MODEL_NAME=$MLFLOW_MODEL_NAME,\
LANGFUSE_HOST=$LANGFUSE_HOST,\
AGENT_MAX_STEPS=8
```

Why each value:
- `min-instances=1` — keeps one warm. Cold starts on Cloud Run with the langchain/langgraph stack are 3–8s. With min=1 you pay ~$15/mo to never see the cold start during a demo.
- `max-instances=10` — enough headroom for a class-project demo, capped to prevent runaway cost.
- `concurrency=15` — each agent request does 5+ blocking LLM calls. Default concurrency=80 stacks 80 in-flight `astream_events` per instance, exhausting CPU. 15 keeps latency consistent. Tune from observed metrics.
- `timeout=120` — agent requests with self-correction can take 30-60s. Default Cloud Run timeout is 5 min, so 120s is comfortable headroom while still bounding stuck requests.
- `cpu=2 memory=2Gi` — LangGraph + multiple in-flight tool calls + the embedding model client are heavier than the legacy chain. Existing config may already be sufficient; verify post-deploy.

### 2. Tracing via Langfuse (open-source self-hosted or cloud)

**Add to `pyproject.toml`:**

```toml
langfuse = ">=2.0.0,<3.0.0"
```

**New: `app/observability/__init__.py`**

```python
"""Single place to wire request-level tracing. We use Langfuse so we can
self-host alongside MLflow if cost matters, or use Langfuse Cloud for free
class-project usage. Falls back to a no-op if env vars aren't set so dev
still works."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
except ImportError:
    Langfuse = None
    CallbackHandler = None


def get_client() -> Optional["Langfuse"]:
    if Langfuse is None:
        return None
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    return Langfuse(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def langgraph_callbacks() -> list:
    if CallbackHandler is None or not os.getenv("LANGFUSE_SECRET_KEY"):
        return []
    return [CallbackHandler()]


@contextmanager
def trace_request(name: str, **metadata):
    """Wrap an agent invocation. Yields a trace id you can attach to logs."""
    client = get_client()
    if client is None:
        yield None
        return
    trace = client.trace(name=name, metadata=metadata)
    try:
        yield trace.id
    finally:
        client.flush()
```

**Wire into W2's `/chat` endpoint** (modify the W2 plan):

```python
# in app/main.py /chat handler
from app.observability import langgraph_callbacks, trace_request

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    graph = request.app.state.agent_graph
    with trace_request("chat", message=req.message[:200]) as trace_id:
        state = ItineraryState(messages=[
            *_history_to_messages(req.history),
            HumanMessage(content=req.message),
        ])
        final_state = await graph.ainvoke(
            state,
            config={"callbacks": langgraph_callbacks(), "metadata": {"trace_id": trace_id}},
        )
    return state_to_response(final_state, request.app.state.rag_label)
```

That single block gives you a per-request trace UI showing every node transition, every tool call, every LLM call with prompts + outputs + token counts. This is the highest-leverage observability item for an agent product.

### 3. MLflow tracking server: TLS + auth (or migrate)

**Current state:** `35.223.147.177:5000` hardcoded in `docker-compose.yml` and CI. Public IP, no TLS, no auth. For class this is acceptable; for the productionization story it's an open registry — anyone scanning that IP can read or modify your model versions.

**Two paths:**

**Path A (minimum viable):** Front the MLflow server with an Nginx reverse proxy or GCP Identity-Aware Proxy (IAP). Add HTTP basic auth or service-account token auth. Update `MLFLOW_TRACKING_URI` to `https://mlflow.<your-domain>` and store the token in Secret Manager.

```yaml
# In .env.example and Cloud Run env
MLFLOW_TRACKING_URI=https://mlflow.cityconcierge.example
MLFLOW_TRACKING_TOKEN=<from secret manager>
```

```python
# app/config.py — add headers when invoking MLflow
import os
os.environ["MLFLOW_TRACKING_TOKEN"] = settings.mlflow_tracking_token  # picked up by mlflow client
```

**Path B (cleaner, more work):** Migrate to a managed MLflow (Databricks, MLflow on a managed K8s, or an MLflow alternative like Weights & Biases free tier). Out of scope for this PR; track as follow-up.

Pick A for now. Document in PR description.

### 4. Secrets → GCP Secret Manager

**Replace** raw env vars in the Cloud Run deploy with `--set-secrets=` references (already shown in §1). Migrate these out of `.env` (where they live for local dev) and into Secret Manager for deployed environments:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY` (new — for Claude / the agent driver)
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`
- `MLFLOW_TRACKING_TOKEN`
- `POSTGRES_PASSWORD`

**Setup script (one-time, document in PR):**

```bash
# Run once per project, with appropriate IAM:
gcloud secrets create openai-api-key --replication-policy=automatic
echo -n "sk-..." | gcloud secrets versions add openai-api-key --data-file=-
# repeat for each secret

# Grant Cloud Run service account access:
gcloud secrets add-iam-policy-binding openai-api-key \
  --member=serviceAccount:$RUN_SA \
  --role=roles/secretmanager.secretAccessor
```

`.env.example` keeps the same variable names — only their *source* changes. Local dev still uses `.env`; deployed runs read Secret Manager.

### 5. Per-request cost & token telemetry

**New: `app/observability/cost.py`**

```python
"""Light-weight token + cost tracking. Wraps each LLM call so we get a single
log line per request with: model, tokens_in, tokens_out, est_cost_usd, latency_ms.

This is in addition to Langfuse's tracing — Langfuse gives you the UI; this
gives you greppable logs for cost dashboards and spot checks."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger("city_concierge.cost")

# USD per 1M tokens. Update when prices change. Source of truth: the provider.
PRICING = {
    "gpt-4o-mini":             {"in": 0.15, "out": 0.60},
    "gpt-4o":                  {"in": 2.50, "out": 10.00},
    "claude-opus-4-7":         {"in": 15.00, "out": 75.00},
    "claude-sonnet-4-6":       {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5":        {"in": 0.80, "out": 4.00},
    "gemini-2.5-flash":        {"in": 0.075, "out": 0.30},
    "text-embedding-3-small":  {"in": 0.02, "out": 0.0},
}


@dataclass
class CallRecord:
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int

    @property
    def est_cost_usd(self) -> float:
        p = PRICING.get(self.model)
        if not p:
            return 0.0
        return (self.tokens_in / 1e6) * p["in"] + (self.tokens_out / 1e6) * p["out"]


@contextmanager
def record_llm_call(model: str, request_id: str | None = None):
    rec = CallRecord(model=model, tokens_in=0, tokens_out=0, latency_ms=0)
    start = time.monotonic()
    try:
        yield rec
    finally:
        rec.latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "llm_call",
            extra={
                "model": rec.model,
                "tokens_in": rec.tokens_in,
                "tokens_out": rec.tokens_out,
                "latency_ms": rec.latency_ms,
                "est_cost_usd": round(rec.est_cost_usd, 6),
                "request_id": request_id,
            },
        )
```

The agent's `act` node (W2 `app/agent/graph.py`) and the embedding helper (W1 `app/retriever.py:build_embedding`) wrap each provider call in `record_llm_call(...)` and update `rec.tokens_in / rec.tokens_out` from the response usage block. Logs go to Cloud Logging via Cloud Run's stdout pipe — no extra infra.

For dashboards, point Cloud Logging's log-based metrics at `jsonPayload.est_cost_usd` to graph cost per minute. Optional but trivial once the log lines are emitted.

## Tests

### New: `tests/unit/test_observability_cost.py`

```python
from app.observability.cost import CallRecord, record_llm_call


def test_cost_calculation_known_model():
    rec = CallRecord(model="gpt-4o-mini", tokens_in=1_000_000, tokens_out=500_000,
                     latency_ms=0)
    # 0.15 + 0.60 * 0.5 = 0.45
    assert abs(rec.est_cost_usd - 0.45) < 1e-9


def test_cost_zero_for_unknown_model():
    rec = CallRecord(model="never-heard-of-it", tokens_in=999, tokens_out=999, latency_ms=0)
    assert rec.est_cost_usd == 0.0


def test_record_llm_call_logs_on_exit(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="city_concierge.cost")
    with record_llm_call("gpt-4o-mini", request_id="r1") as r:
        r.tokens_in = 100
        r.tokens_out = 50
    assert any("llm_call" in rec.message for rec in caplog.records)
```

### New: `tests/unit/test_observability_tracing.py`

```python
def test_trace_request_no_op_without_env(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from app.observability import trace_request, langgraph_callbacks
    with trace_request("test") as trace_id:
        assert trace_id is None
    assert langgraph_callbacks() == []
```

## Manual verification

After deploy:

```bash
# Cold start gone:
curl -w "%{time_total}s\n" -o /dev/null -s https://city-concierge-app-XXXX.run.app/health
# Expected: <1s on first hit (was 3-8s).

# Agent request lands a trace:
curl -X POST https://...run.app/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "italian dinner tonight in north beach", "history": []}'
# Visit Langfuse dashboard, confirm a trace with the user's message and N tool calls.

# Cost log present:
gcloud logging read 'jsonPayload.message="llm_call"' --limit=5 --format=json
# Expected: 5 entries, each with model, tokens_in/out, est_cost_usd.

# MLflow secured:
curl -I https://mlflow.<your-domain>
# Expected: 401 without auth header, 200 with bearer token.

# Secrets correctly injected:
gcloud run services describe city-concierge-app --region=$REGION \
  --format="value(spec.template.spec.containers[0].env)"
# Expected: secrets referenced via secretKeyRef, not literal values.
```

## Risks / open questions

- **Min-instances cost.** ~$15/mo per always-on instance with cpu=2/memory=2Gi. Negligible vs. demo value. Reduce to min=0 after the demo phase if cost matters.
- **Langfuse vs LangSmith.** Both work. Langfuse is open-source self-hostable and has a free Cloud tier (sufficient for class). LangSmith is fully managed by LangChain. Plan picks Langfuse for portability; either is fine.
- **MLflow path A is a stopgap.** Putting Nginx + IAP in front of an unauthenticated MLflow doesn't fix the underlying issue (no per-user auth, no audit trail). It's enough for "we don't have an open server on the internet." Path B (managed MLflow / W&B migration) is the proper long-term move.
- **Concurrency=15 is a guess.** The right value depends on observed CPU + LLM-call latency. Re-tune from `gcloud run services describe ... --format="value(status.conditions)"` and Cloud Monitoring metrics after a week of real traffic.
- **Pricing table drift.** Provider prices change. The `PRICING` dict in `cost.py` will rot; treat the cost numbers as estimates and refresh quarterly. Better: pull from a centralized config or an env-var override per model.

## Deferred (future PRs, real but not blocking)

- **Streaming responses** for `/chat`. LangGraph supports `astream_events()`; Cloud Run supports HTTP/2 streaming. Adds polish to the demo (user sees the agent thinking) but doesn't change correctness.
- **Embedding cache** keyed on `(query, embedding_model)` in Redis or a Postgres table. High value once W6 evals run regularly — reduces eval cost dramatically.
- **Cloud SQL private VPC audit.** Confirm Cloud Run reaches Cloud SQL via private VPC connector + cloud-sql-proxy, not over public IP. 30-second check, real security implication.
- **Per-user rate limiting.** Once you have real users, an unbounded `/chat` endpoint is a wallet attack vector. Add an IP-based limit (Cloud Armor) or per-API-key quotas.

---

**Status:** Merged in [PR #60](https://github.com/deshmukh-neel/mlops_city_concierge/pull/60) (2026-05-06). MLflow auth proxy (§3) is the remaining piece from this plan and will land as a follow-up PR.
