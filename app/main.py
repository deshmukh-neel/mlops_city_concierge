from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import mlflow
import psycopg2
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from psycopg2.extensions import connection
from pydantic import BaseModel, Field, ValidationError

from .agent.commit import enrich_stops_with_booking
from .agent.graph import build_agent_graph
from .agent.input_parsing import (
    SlotExtractionResult,
    explicit_num_stops_from_conversation,
    has_slot_structure,
    parse_closure_decision,
)
from .agent.io import messages_from_history, state_to_cards
from .agent.revision import summarize_stops
from .agent.state import ClosureContext, ItineraryState, Stop, UserConstraints
from .agent.swap import (
    _bounded_retime_after_swap,
    _build_closure_context_entry,
    _formulate_closure_question,
    _per_stop_closure_status,
    _promote_pending,
    _resolve_anchor,
    _resolve_family_for_stop,
    _resolve_insert_position,
    _try_walking_distance_swap,
)
from .chain import build_rag_chain
from .config import get_settings, resolve_llm_api_key
from .db import get_conn, get_db
from .db_pool import close_db_pool, init_db_pool
from .observability import langgraph_callbacks, trace_request
from .tools.filters import family_of
from .tools.retrieval import get_details

logger = logging.getLogger(__name__)

RAG_UNAVAILABLE_DETAIL = (
    "RAG chain unavailable: MLflow registry could not be reached at startup. "
    "Ensure the MLflow IAP tunnel is open and restart the app."
)
AGENT_UNAVAILABLE_DETAIL = (
    "Agent graph unavailable: MLflow registry could not be reached at startup. "
    "Ensure the MLflow IAP tunnel is open and restart the app."
)


# Phase 4 / D-04-01..D-04-03 — slot-extraction intake prompt template.
# The user's message is interpolated via .format(user_message=...).
# The vocabulary uses Title-Case Google primary_type values so the
# structured-output payload can be validated by family_of() (which is
# case-preserving against `_PRIMARY_TYPE_FAMILIES`). The schema enforced
# by Pydantic + the family_of validation layer means even an obeyed
# prompt injection produces an empty extracted list (T-04-06-01).
_SLOT_INTAKE_PROMPT_TEMPLATE: str = (
    "Extract the user's per-slot category structure. The user's message is:\n"
    '"{user_message}"\n\n'
    'If the user named distinct slots (e.g., "dinner, drinks, dessert" or '
    '"omakase then ramen"), return a list of Google primary_type values, '
    "one per slot in order. If the message is free-text or has no clear "
    "slot structure, return [].\n\n"
    'Output shape: {{"requested_primary_types": '
    '["Restaurant", "Cocktail Bar", "Dessert Shop"]}}\n\n'
    "Use this vocabulary (Google primary_type values, Title Case):\n"
    "- Restaurants: Restaurant, Japanese Restaurant, Sushi Restaurant, "
    "Italian Restaurant, Ramen Restaurant, Mexican Restaurant, "
    "Pizza Restaurant, Steak House, Fine Dining Restaurant\n"
    "- Bars: Bar, Cocktail Bar, Wine Bar, Pub, Sports Bar, Night Club\n"
    "- Dessert: Dessert Shop, Bakery, Ice Cream Shop, Cafe, Coffee Shop, "
    "Donut Shop, Tea House\n"
    "- Cafes: Cafe, Coffee Shop, Tea House"
)


def _intake_bind_kwargs(llm: BaseChatModel) -> dict[str, Any]:
    """Return provider-appropriate `.bind(**kwargs)` arguments for the intake
    LLM call.

    Always sets `temperature=1.0` (per memory
    `feedback_temp1_reasoning_off_all_models.md` — always temp=1.0;
    disable thinking for ALL providers including Gemini). Reasoning-off
    kwargs are picked per provider class so the bind step mirrors what
    `app.llm_factory.build_chat_model` set at construction time.

    Branches:
      ChatOpenAI                  → {"temperature": 1.0}
      ChatGoogleGenerativeAI      → {"temperature": 1.0, "thinking_budget": 0}
                                    (or thinking_level="low" for the
                                    `_GEMINI_THINKING_ONLY` set)
      ChatDeepSeek                → {"temperature": 1.0,
                                     "extra_body": {"thinking":
                                     {"type": "disabled"}}}
      ChatMoonshot / _ToolLoopChatMoonshot
                                  → {"temperature": <llm.temperature>,
                                     "thinking": False}
                                    Kimi forces temp=0.6 for kimi-k2.6 in
                                    the factory (per
                                    `_KIMI_FORCED_TEMPERATURE`), so we do
                                    NOT override the temperature here.
      ScriptedChatModel           → {} (no-op; the scripted model ignores
                                    bind kwargs and the test harness asserts
                                    on no-bind separately)
      anything else               → {"temperature": 1.0}
                                    (safe minimal default — better to set
                                    temp than silently leave it at the
                                    construction-time value)
    """
    class_name = type(llm).__name__
    if class_name == "ChatOpenAI":
        return {"temperature": 1.0}
    if class_name == "ChatGoogleGenerativeAI":
        # Avoid an import-cycle / hard dep on llm_factory by inlining the
        # thinking-only set — same source of truth as the factory.
        from .llm_factory import _GEMINI_THINKING_ONLY

        model_name = getattr(llm, "model", "") or ""
        if model_name in _GEMINI_THINKING_ONLY:
            return {"temperature": 1.0, "thinking_level": "low"}
        return {"temperature": 1.0, "thinking_budget": 0}
    if class_name == "ChatDeepSeek":
        return {
            "temperature": 1.0,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    if class_name in {"ChatMoonshot", "_ToolLoopChatMoonshot"}:
        # Kimi forces a hard temperature for some models — keep whatever
        # the factory chose.
        existing_temp = getattr(llm, "temperature", 1.0)
        return {"temperature": existing_temp, "thinking": False}
    if class_name == "ScriptedChatModel":
        return {}
    # Fallback for unknown / test-stand-in classes — set temp=1.0 so the
    # cross-provider invariant still holds.
    return {"temperature": 1.0}


db_connection_dependency = Depends(get_db)


class RecommendationRequest(BaseModel):
    query: str = Field(
        ...,
        description="User's recommendation query (for example, 'Best tacos in the Mission').",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of source places to include in the response.",
    )


class RecommendationSource(BaseModel):
    name: str | None = None
    rating: float | None = None
    address: str | None = None
    similarity: float | None = None
    place_id: str | None = None
    primary_type: str | None = None


class RecommendationResponse(BaseModel):
    response: str
    sources: list[RecommendationSource]


class ActiveModelConfig(BaseModel):
    llm_provider: str
    chat_model: str
    k: int
    temperature: float = 0.0
    run_id: str | None = None
    model_version: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationState(BaseModel):
    """Opaque-to-frontend state round-tripped via /chat.

    The frontend stores this verbatim from each response and sends it back
    on the next request; the backend is the source of truth for the schema.
    `prior_stops` carries the stops the user just saw so an accept/decline
    early-return path can act on them without /chat round-tripping the full
    places list. `committed_stops` carries the post-graph stops list so the
    Phase 6 refinement injection block (in `chat()`, see plan 06-05) has
    structured ground truth for the prior turn's committed plan; legacy
    payloads without the key decode to [] via `default_factory=list`.
    """

    schema_version: int = 1
    closure_context: list[ClosureContext] = Field(default_factory=list)
    prior_stops: list[Stop] = Field(default_factory=list)
    committed_stops: list[Stop] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    # Opaque dict so a malformed nested object doesn't 422 before the
    # handler runs — `/chat` does manual ConversationState.model_validate
    # and degrades to empty state on ValidationError. `dict | None` still
    # 422s on non-object payloads (string/list/number), which is the right
    # answer for those (developer/curl mistakes; the real frontend can't
    # produce them).
    conversation_state: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    # Field name matches the frontend contract (frontend/src/api/chat.js).
    reply: str
    places: list[dict]
    ragLabel: str  # noqa: N815
    # Backend always emits a typed, validated shape on the response side.
    conversation_state: ConversationState | None = None


@dataclass
class LoadedConfig:
    chain: Any
    llm: BaseChatModel
    params: ActiveModelConfig


def parse_active_model_config(
    params: dict[str, str], run_id: str, model_version: str
) -> ActiveModelConfig:
    import os

    from app.llm_factory import SUPPORTED_PROVIDERS

    settings = get_settings()
    llm_provider = (params.get("llm_provider") or "openai").lower()
    if llm_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")
    if llm_provider == "openai":
        default_chat_model = settings.openai_chat_model
    elif llm_provider == "gemini":
        default_chat_model = settings.gemini_chat_model
    else:
        env_var = {"deepseek": "DEEPSEEK_MODEL", "kimi": "MOONSHOT_MODEL"}[llm_provider]
        default_chat_model = os.getenv(env_var, "")

    return ActiveModelConfig(
        llm_provider=llm_provider,
        chat_model=params.get("chat_model") or default_chat_model,
        k=int(params.get("k", settings.retriever_k)),
        temperature=float(params.get("temperature", "0.0")),
        run_id=run_id,
        model_version=model_version,
    )


def _parse_model_override(raw: str) -> tuple[Literal["version", "alias"], str]:
    """Parse RAG_MODEL_OVERRIDE into (kind, value); accepts 'version:N' or 'alias:NAME'.

    Whitespace around the value is stripped (shell exports leak stray spaces).
    Empty input and any other shape raises ValueError. Caller is responsible
    for short-circuiting on None / unset before invoking this helper.
    """
    prefix, sep, value = raw.partition(":")
    value = value.strip()
    if not sep or prefix not in ("version", "alias") or not value:
        raise ValueError(
            f"RAG_MODEL_OVERRIDE must be 'version:N' or 'alias:NAME'; got {raw!r}. "
            "version:N is recommended (alias:NAME re-resolves per request "
            "and can race with alias moves)."
        )
    return prefix, value  # type: ignore[return-value]


def load_registered_rag_chain() -> LoadedConfig:
    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    # RAG_MODEL_OVERRIDE routes /chat through a candidate version/alias without
    # touching the shared "production" alias. version:N is recommended; alias:NAME
    # re-resolves on every load and can race if someone moves the alias mid-eval.
    raw_override = (settings.rag_model_override or "").strip()
    if raw_override:
        kind, value = _parse_model_override(raw_override)
        try:
            model_version = (
                client.get_model_version(settings.mlflow_model_name, value)
                if kind == "version"
                else client.get_model_version_by_alias(settings.mlflow_model_name, value)
            )
        except Exception as exc:  # pragma: no cover - exercised via startup tests
            raise RuntimeError(
                f"Unable to load MLflow model for RAG_MODEL_OVERRIDE={raw_override!r}."
            ) from exc
        logger.info(
            "load_registered_rag_chain: using override kind=%s value=%s version=%s",
            kind,
            value,
            model_version.version,
        )
    else:
        try:
            model_version = client.get_model_version_by_alias(
                settings.mlflow_model_name,
                "production",
            )
        except Exception as exc:  # pragma: no cover - exercised via startup tests
            raise RuntimeError(
                "Unable to load the MLflow production alias for "
                f"registered model '{settings.mlflow_model_name}'."
            ) from exc

    run = client.get_run(model_version.run_id)
    config = parse_active_model_config(
        run.data.params,
        run_id=model_version.run_id,
        model_version=str(model_version.version),
    )
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")
    built = build_rag_chain(
        connection_string=database_url,
        api_key=resolve_llm_api_key(config.llm_provider),
        llm_provider=config.llm_provider,
        chat_model=config.chat_model,
        k=config.k,
        temperature=config.temperature,
    )
    return LoadedConfig(chain=built.chain, llm=built.llm, params=config)


def serialize_sources(source_documents: list[Any], limit: int) -> list[RecommendationSource]:
    sources: list[RecommendationSource] = []
    for document in source_documents[:limit]:
        metadata = getattr(document, "metadata", {}) or {}
        sources.append(
            RecommendationSource(
                name=metadata.get("name"),
                rating=metadata.get("rating"),
                address=metadata.get("address"),
                similarity=metadata.get("similarity"),
                place_id=metadata.get("place_id"),
                primary_type=metadata.get("primary_type"),
            )
        )
    return sources


def _rag_label_for(config: ActiveModelConfig | None) -> str:
    if config is None:
        return "unknown"
    return f"{config.llm_provider}:{config.chat_model}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")

    init_db_pool(
        database_url,
        settings.db_pool_min_connections,
        settings.db_pool_max_connections,
    )
    try:
        try:
            loaded = load_registered_rag_chain()
        except Exception:
            logger.warning(
                "Failed to load RAG chain from MLflow registry — app will boot in "
                "degraded mode and RAG endpoints will return 503.",
                exc_info=True,
            )
            loaded = None

        if loaded is None:
            app.state.rag_chain = None
            app.state.active_model_config = None
            app.state.agent_graph = None
            # Phase 4: intake pipeline reuses the planning LLM (D-04-02);
            # no LLM is available in this degraded branch.
            app.state.agent_llm = None
            app.state.rag_label = _rag_label_for(None)
        else:
            app.state.rag_chain = loaded.chain
            app.state.active_model_config = loaded.params
            try:
                app.state.agent_graph = build_agent_graph(loaded.llm)
                # Phase 4: expose the same BaseChatModel the graph was
                # built on so the /chat intake call can reuse it (D-04-02
                # — same RAG_MODEL_OVERRIDE resolution, single source).
                app.state.agent_llm = loaded.llm
            except Exception:
                logger.warning(
                    "Failed to build agent graph — /chat will return 503; "
                    "/predict falls back to the legacy RAG chain.",
                    exc_info=True,
                )
                app.state.agent_graph = None
                app.state.agent_llm = None
            app.state.rag_label = _rag_label_for(loaded.params)
        yield
    finally:
        close_db_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_origin_regex=r"https://.*\.vercel\.app$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.get("/root")
def root() -> dict[str, str]:
    return {"message": "Welcome!"}


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    model_config = getattr(request.app.state, "active_model_config", None)
    if model_config is None:
        return {"status": "degraded", "rag_chain": "unavailable"}

    return {
        "status": "ok",
        "llm_provider": model_config.llm_provider,
        "chat_model": model_config.chat_model,
    }


@app.get("/health/db")
def health_db(conn: connection = db_connection_dependency) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()
    return {"status": "ok"}


def _place_is_open_now(hours: dict | None, when: datetime) -> bool:
    """One-shot SQL call mirroring closure_swap's check, used on the accept
    path to re-validate a proposed alternative right before applying it.

    Fails OPEN — a DB blip must not block the swap, matching the codebase's
    "fail-open on DB error" precedent (see app/agent/critique/checks.py:200).
    """
    if not hours:
        return True
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT place_is_open(%s::jsonb, %s)",
                [json.dumps(hours), when],
            )
            row = cur.fetchone()
            return bool(row[0]) if row else True
    except psycopg2.Error:
        logger.warning("_place_is_open_now DB error; treating as open", exc_info=True)
        return True


def _build_outbound_state(
    closure_context: list[ClosureContext],
    stops: list[Stop],
) -> ConversationState:
    return ConversationState(
        schema_version=1,
        closure_context=closure_context,
        prior_stops=stops,
        committed_stops=stops,
    )


async def _try_accept_path(
    pending: ClosureContext,
    incoming: ConversationState,
    rag_label: str,
) -> ChatResponse | None:
    """User accepted the proposed drive alternative.

    Returns a built ChatResponse for the early-return path, or None if
    re-validation fails (caller falls through to the graph).
    """
    if pending.proposed_alternative is None:
        return None
    # Defense in depth: re-fetch the proposal in case the index changed
    # between the question and the answer.
    details = get_details(pending.proposed_alternative.place_id)
    if details is None:
        logger.warning(
            "closure_swap.proposed_alternative_invalidated: place_id=%s missing",
            pending.proposed_alternative.place_id,
        )
        return None
    if not _place_is_open_now(details.regular_opening_hours, pending.attempted_arrival):
        logger.warning("closure_swap.proposed_alternative_invalidated: now closed")
        return None

    # Insert at the resolved position and re-chain arrivals on the new plan.
    insert_at = _resolve_insert_position(pending, incoming.prior_stops)
    replacement = pending.proposed_alternative.model_copy()
    new_stops = list(incoming.prior_stops)
    new_stops.insert(insert_at, replacement)

    retimed = await _bounded_retime_after_swap(new_stops)
    re_closed = _per_stop_closure_status(retimed)

    # Mark the pending entry as accepted up front so subsequent escalations
    # see the right outcome shape.
    updated_context: list[ClosureContext] = [
        c.model_copy(update={"outcome": "user_accepted_drive"})
        if c.place_id == pending.place_id and c.outcome == "pending_user_decision"
        else c
        for c in incoming.closure_context
    ]

    probe_state = ItineraryState(stops=retimed, closure_context=updated_context)

    # If the retime exposed a NEW closure on a different stop, try a single
    # walking-distance swap; escalate anything still closed.
    new_pending_entries: list[ClosureContext] = []
    if any(re_closed):
        still_closed_indices = [i for i, flag in enumerate(re_closed) if flag]
        for idx in still_closed_indices:
            closed_stop = retimed[idx]
            family = _resolve_family_for_stop(closed_stop)
            anchor = _resolve_anchor(probe_state, closed_stop)
            match = None
            if family and anchor:
                probe_ctx = ClosureContext(
                    place_id=closed_stop.place_id,
                    place_name=closed_stop.name,
                    family=family,
                    attempted_arrival=closed_stop.arrival_time or datetime.now(),
                    outcome="pending_user_decision",
                    insert_after_place_id=None,
                    insert_before_place_id=None,
                    stop_index_hint=idx,
                )
                match = _try_walking_distance_swap(probe_state, probe_ctx, anchor_place_id=anchor)
            if match is not None:
                retimed[idx] = match.stop
                updated_context.append(
                    _build_closure_context_entry(retimed, idx, match, "auto_swapped")
                )
            else:
                outcome = (
                    "pending_user_decision" if not new_pending_entries else "queued_user_decision"
                )
                new_pending_entries.append(
                    _build_closure_context_entry(retimed, idx, None, outcome)
                )

    if new_pending_entries:
        updated_context.extend(new_pending_entries)

    # Promote a queued entry if pending was cleared.
    updated_context = _promote_pending(updated_context)

    # Surface either the next question or the summary.
    next_pending = next(
        (c for c in updated_context if c.outcome == "pending_user_decision"),
        None,
    )
    surfaced_stops = list(retimed)
    if next_pending is not None:
        final_reply = _formulate_closure_question(next_pending)
        surfaced_stops = [s for s in surfaced_stops if s.place_id != next_pending.place_id]
    else:
        enrich_stops_with_booking(surfaced_stops, probe_state)
        final_reply = summarize_stops(probe_state.model_copy(update={"stops": surfaced_stops}))

    final_state = ItineraryState(stops=surfaced_stops, closure_context=updated_context)
    return ChatResponse(
        reply=final_reply,
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=_build_outbound_state(updated_context, surfaced_stops),
    )


def _decline_path(
    pending: ClosureContext,
    incoming: ConversationState,
    rag_label: str,
) -> ChatResponse:
    """User declined the drive option. The closed stop wasn't on prior_stops
    (it was demoted to pending before the user saw the plan), so we just flip
    the outcome and promote any queued entry.
    """
    updated_context: list[ClosureContext] = [
        c.model_copy(update={"outcome": "user_declined_dropped"})
        if c.place_id == pending.place_id and c.outcome == "pending_user_decision"
        else c
        for c in incoming.closure_context
    ]
    updated_context = _promote_pending(updated_context)
    next_pending = next(
        (c for c in updated_context if c.outcome == "pending_user_decision"),
        None,
    )
    probe_state = ItineraryState(
        stops=incoming.prior_stops,
        closure_context=updated_context,
    )
    if next_pending is not None:
        final_reply = _formulate_closure_question(next_pending)
        surfaced_stops = [s for s in incoming.prior_stops if s.place_id != next_pending.place_id]
    else:
        final_reply = summarize_stops(probe_state)
        surfaced_stops = list(incoming.prior_stops)

    final_state = ItineraryState(stops=surfaced_stops, closure_context=updated_context)
    return ChatResponse(
        reply=final_reply,
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=_build_outbound_state(updated_context, surfaced_stops),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail=AGENT_UNAVAILABLE_DETAIL)

    rag_label = getattr(request.app.state, "rag_label", _rag_label_for(None))

    incoming: ConversationState
    if req.conversation_state is None:
        incoming = ConversationState()
    else:
        try:
            incoming = ConversationState.model_validate(req.conversation_state)
        except ValidationError:
            logger.warning("conversation_state.decode_failed", exc_info=True)
            incoming = ConversationState()

    pending = next(
        (c for c in incoming.closure_context if c.outcome == "pending_user_decision"),
        None,
    )

    decision: str | None = None
    if pending is not None and req.message.strip():
        decision = parse_closure_decision(req.message)
        if decision == "accept":
            early = await _try_accept_path(pending, incoming, rag_label)
            if early is not None:
                return early
            # Re-validation failed — fall through to the graph with the bad
            # pending entry preserved; the model will plan around it.
        elif decision == "decline":
            return _decline_path(pending, incoming, rag_label)
        # "alternative" falls through to the graph (handled below).

    with trace_request("chat", message=req.message[:200]) as trace_id:
        # If we just routed "alternative", give the model a HumanMessage hint
        # so it knows the user declined the drive option and what they asked
        # for instead.
        hint_messages: list[HumanMessage] = []
        if pending is not None and decision == "alternative":
            hint_messages.append(
                HumanMessage(
                    content=(
                        f"User declined the drive option for {pending.place_name}. "
                        f"They want: '{req.message}'. Plan again with this guidance."
                    )
                )
            )

        # Phase 4 hybrid intake pipeline (D-04-01..D-04-03 / T-04-06-01..09).
        # Deterministic pre-check FIRST so free-text /chat requests pay zero
        # added latency. When the pre-check fires AND an LLM is available,
        # explicitly bind temp=1.0 + provider-specific reasoning-off kwargs,
        # then drive a structured-output extraction. Validate every entry
        # against family_of() — unmappable strings (including obeyed prompt
        # injections like 'admin_access') are silently dropped. Any
        # exception in the intake block is logged and degraded to an empty
        # list (fail-open per D-04-03). The intake call lives INSIDE
        # trace_request so MLflow tracking captures intake latency in the
        # same /chat trace (T-04-06-06).
        extracted_types: list[str] = []
        if has_slot_structure(req.message):
            loaded_llm = getattr(request.app.state, "agent_llm", None)
            if loaded_llm is not None:
                try:
                    bind_kwargs = _intake_bind_kwargs(loaded_llm)
                    intake_llm = loaded_llm.bind(**bind_kwargs)
                    structured = intake_llm.with_structured_output(SlotExtractionResult)
                    intake_prompt = _SLOT_INTAKE_PROMPT_TEMPLATE.format(user_message=req.message)
                    result = await structured.ainvoke(intake_prompt)
                    raw_types = list(result.requested_primary_types or [])
                    extracted_types = [t for t in raw_types if family_of(t) is not None]
                except Exception:
                    logger.warning(
                        "Slot intake LLM call failed; falling back to free-text",
                        exc_info=True,
                    )
                    extracted_types = []

        state = ItineraryState(
            messages=[
                *messages_from_history(req.history),
                *hint_messages,
                HumanMessage(content=req.message),
            ],
            constraints=UserConstraints(
                num_stops=explicit_num_stops_from_conversation(req.history, req.message),
                requested_primary_types=extracted_types,
            ),
            closure_context=incoming.closure_context,
        )
        raw = await graph.ainvoke(
            state,
            config={
                "callbacks": langgraph_callbacks(),
                "metadata": {"trace_id": trace_id},
            },
        )
    final_state = raw if isinstance(raw, ItineraryState) else ItineraryState(**raw)

    return ChatResponse(
        reply=final_state.final_reply or "",
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=_build_outbound_state(final_state.closure_context, final_state.stops),
    )


@app.post("/predict", response_model=RecommendationResponse)
async def predict(request_body: RecommendationRequest, request: Request) -> RecommendationResponse:
    rag_chain = getattr(request.app.state, "rag_chain", None)
    if rag_chain is None:
        raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_DETAIL)
    result = rag_chain.invoke({"query": request_body.query})
    response_text = result.get("result") or result.get("response") or ""
    source_documents = result.get("source_documents") or []
    return RecommendationResponse(
        response=response_text,
        sources=serialize_sources(source_documents, request_body.limit),
    )
