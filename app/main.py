from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import mlflow
import psycopg2
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
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
    is_refinement_request,
    parse_closure_decision,
)
from .agent.io import build_refinement_prompt_message, messages_from_history, state_to_cards
from .agent.revision import summarize_stops
from .agent.state import ClosureContext, ItineraryState, Stop, UserConstraints
from .agent.swap import (
    bounded_retime_after_swap,
    build_closure_context_entry,
    formulate_closure_question,
    per_stop_closure_status,
    promote_pending,
    resolve_anchor,
    resolve_family_for_stop,
    resolve_insert_position,
    try_walking_distance_swap,
)
from .chain import build_rag_chain
from .config import env_flag, get_settings, resolve_llm_api_key
from .db import get_conn, get_db
from .db_pool import close_db_pool, init_db_pool
from .observability import langgraph_callbacks, trace_request
from .query_log import log_user_query
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


SLOT_INTAKE_PROMPT_TEMPLATE: str = (
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


def intake_bind_kwargs(llm: BaseChatModel) -> dict[str, Any]:
    """Return provider-specific kwargs for the lightweight intake call."""
    class_name = type(llm).__name__
    if class_name == "ChatOpenAI":
        return {"temperature": 1.0}
    if class_name == "ChatGoogleGenerativeAI":
        # Avoid an import-cycle / hard dep on llm_factory by inlining the
        # thinking-only set — same source of truth as the factory.
        from .llm_factory import GEMINI_THINKING_ONLY

        model_name = getattr(llm, "model", "") or ""
        if model_name in GEMINI_THINKING_ONLY:
            return {"temperature": 1.0, "thinking_level": "low"}
        return {"temperature": 1.0, "thinking_budget": 0}
    if class_name == "ChatDeepSeek":
        return {
            "temperature": 1.0,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    if class_name in {"ChatMoonshot", "ToolLoopChatMoonshot"}:
        # Kimi forces a hard temperature for some models — keep whatever
        # the factory chose.
        existing_temp = getattr(llm, "temperature", 1.0)
        return {"temperature": existing_temp, "thinking": False}
    if class_name == "ScriptedChatModel":
        return {}
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
    """Opaque state the frontend round-trips between /chat requests."""

    schema_version: int = 1
    closure_context: list[ClosureContext] = Field(default_factory=list)
    prior_stops: list[Stop] = Field(default_factory=list)
    committed_stops: list[Stop] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    # Keep this loose so malformed nested state can degrade inside the handler.
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


def parse_model_override(raw: str) -> tuple[Literal["version", "alias"], str]:
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
        kind, value = parse_model_override(raw_override)
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


def rag_label_for(config: ActiveModelConfig | None) -> str:
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
            app.state.agent_llm = None
            app.state.rag_label = rag_label_for(None)
        else:
            app.state.rag_chain = loaded.chain
            app.state.active_model_config = loaded.params
            try:
                app.state.agent_graph = build_agent_graph(
                    loaded.llm,
                    provider=getattr(loaded.params, "llm_provider", "openai"),
                )
                app.state.agent_llm = loaded.llm
            except Exception:
                logger.warning(
                    "Failed to build agent graph — /chat will return 503; "
                    "/predict falls back to the legacy RAG chain.",
                    exc_info=True,
                )
                app.state.agent_graph = None
                app.state.agent_llm = None
            app.state.rag_label = rag_label_for(loaded.params)
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
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_allow_origin_regex or None,
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


def place_is_open_now(hours: dict | None, when: datetime) -> bool:
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
        logger.warning("place_is_open_now DB error; treating as open", exc_info=True)
        return True


def build_outbound_state(
    closure_context: list[ClosureContext],
    stops: list[Stop],
) -> ConversationState:
    return ConversationState(
        schema_version=1,
        closure_context=closure_context,
        prior_stops=stops,
        committed_stops=stops,
    )


def decode_conversation_state(raw: dict[str, Any] | None) -> ConversationState:
    if raw is None:
        return ConversationState()
    try:
        return ConversationState.model_validate(raw)
    except ValidationError:
        logger.warning("conversation_state.decode_failed", exc_info=True)
        return ConversationState()


async def extract_requested_primary_types(
    message: str,
    llm: BaseChatModel | None,
) -> list[str]:
    if llm is None or not has_slot_structure(message):
        return []

    try:
        intake_llm = llm.bind(**intake_bind_kwargs(llm))
        structured = intake_llm.with_structured_output(SlotExtractionResult)
        result = await structured.ainvoke(
            SLOT_INTAKE_PROMPT_TEMPLATE.format(user_message=message)
        )
    except Exception:
        logger.warning("Slot intake LLM call failed; falling back to free-text", exc_info=True)
        return []

    return [t for t in list(result.requested_primary_types or []) if family_of(t) is not None]


def refinement_messages_for(message: str, incoming: ConversationState) -> list[HumanMessage]:
    matched, target_slot = is_refinement_request(message)
    if (
        matched
        and target_slot is not None
        and incoming.committed_stops
        and 1 <= target_slot <= len(incoming.committed_stops)
        and env_flag("REFINEMENT_STRUCTURED_PLAN_ENABLED")
    ):
        logger.info(
            "chat.refinement.structured_plan.prepended target_slot=%s committed_stops=%d",
            target_slot,
            len(incoming.committed_stops),
        )
        return [build_refinement_prompt_message(incoming.committed_stops)]
    return []


def closure_hint_messages(
    pending: ClosureContext | None,
    decision: str | None,
    user_message: str,
) -> list[HumanMessage]:
    if pending is None or decision != "alternative":
        return []
    return [
        HumanMessage(
            content=(
                f"User declined the drive option for {pending.place_name}. "
                f"They want: '{user_message}'. Plan again with this guidance."
            )
        )
    ]


def initial_chat_state(
    req: ChatRequest,
    incoming: ConversationState,
    *,
    hint_messages: list[HumanMessage],
    refinement_messages: list[HumanMessage],
    requested_primary_types: list[str],
    num_stops: int | None,
) -> ItineraryState:
    return ItineraryState(
        messages=[
            *messages_from_history(req.history),
            *hint_messages,
            *refinement_messages,
            HumanMessage(content=req.message),
        ],
        constraints=UserConstraints(
            num_stops=num_stops,
            requested_primary_types=requested_primary_types,
        ),
        closure_context=incoming.closure_context,
    )


async def try_accept_path(
    pending: ClosureContext,
    incoming: ConversationState,
    rag_label: str,
) -> ChatResponse | None:
    """Return a response for an accepted drive option, or None if it needs replanning."""
    if pending.proposed_alternative is None:
        return None
    details = get_details(pending.proposed_alternative.place_id)
    if details is None:
        logger.warning(
            "closure_swap.proposed_alternative_invalidated: place_id=%s missing",
            pending.proposed_alternative.place_id,
        )
        return None
    if not place_is_open_now(details.regular_opening_hours, pending.attempted_arrival):
        logger.warning("closure_swap.proposed_alternative_invalidated: now closed")
        return None

    insert_at = resolve_insert_position(pending, incoming.prior_stops)
    replacement = pending.proposed_alternative.model_copy()
    new_stops = list(incoming.prior_stops)
    new_stops.insert(insert_at, replacement)

    retimed = await bounded_retime_after_swap(new_stops)
    re_closed = per_stop_closure_status(retimed)

    updated_context: list[ClosureContext] = [
        c.model_copy(update={"outcome": "user_accepted_drive"})
        if c.place_id == pending.place_id and c.outcome == "pending_user_decision"
        else c
        for c in incoming.closure_context
    ]

    probe_state = ItineraryState(stops=retimed, closure_context=updated_context)

    new_pending_entries: list[ClosureContext] = []
    if any(re_closed):
        still_closed_indices = [i for i, flag in enumerate(re_closed) if flag]
        for idx in still_closed_indices:
            closed_stop = retimed[idx]
            family = resolve_family_for_stop(closed_stop)
            anchor = resolve_anchor(probe_state, closed_stop)
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
                match = try_walking_distance_swap(probe_state, probe_ctx, anchor_place_id=anchor)
            if match is not None:
                retimed[idx] = match.stop
                updated_context.append(
                    build_closure_context_entry(retimed, idx, match, "auto_swapped")
                )
            else:
                outcome = (
                    "pending_user_decision" if not new_pending_entries else "queued_user_decision"
                )
                new_pending_entries.append(
                    build_closure_context_entry(retimed, idx, None, outcome)
                )

    if new_pending_entries:
        updated_context.extend(new_pending_entries)

    updated_context = promote_pending(updated_context)

    next_pending = next(
        (c for c in updated_context if c.outcome == "pending_user_decision"),
        None,
    )
    surfaced_stops = list(retimed)
    if next_pending is not None:
        final_reply = formulate_closure_question(next_pending)
        surfaced_stops = [s for s in surfaced_stops if s.place_id != next_pending.place_id]
    else:
        enrich_stops_with_booking(surfaced_stops, probe_state)
        final_reply = summarize_stops(probe_state.model_copy(update={"stops": surfaced_stops}))

    final_state = ItineraryState(stops=surfaced_stops, closure_context=updated_context)
    return ChatResponse(
        reply=final_reply,
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=build_outbound_state(updated_context, surfaced_stops),
    )


def decline_path(
    pending: ClosureContext,
    incoming: ConversationState,
    rag_label: str,
) -> ChatResponse:
    """Return a response after the user declines a proposed drive option."""
    updated_context: list[ClosureContext] = [
        c.model_copy(update={"outcome": "user_declined_dropped"})
        if c.place_id == pending.place_id and c.outcome == "pending_user_decision"
        else c
        for c in incoming.closure_context
    ]
    updated_context = promote_pending(updated_context)
    next_pending = next(
        (c for c in updated_context if c.outcome == "pending_user_decision"),
        None,
    )
    probe_state = ItineraryState(
        stops=incoming.prior_stops,
        closure_context=updated_context,
    )
    if next_pending is not None:
        final_reply = formulate_closure_question(next_pending)
        surfaced_stops = [s for s in incoming.prior_stops if s.place_id != next_pending.place_id]
    else:
        final_reply = summarize_stops(probe_state)
        surfaced_stops = list(incoming.prior_stops)

    final_state = ItineraryState(stops=surfaced_stops, closure_context=updated_context)
    return ChatResponse(
        reply=final_reply,
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=build_outbound_state(updated_context, surfaced_stops),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest, request: Request, background_tasks: BackgroundTasks
) -> ChatResponse:
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail=AGENT_UNAVAILABLE_DETAIL)

    rag_label = getattr(request.app.state, "rag_label", rag_label_for(None))

    incoming = decode_conversation_state(req.conversation_state)
    pending = next(
        (c for c in incoming.closure_context if c.outcome == "pending_user_decision"),
        None,
    )

    decision: str | None = None
    if pending is not None and req.message.strip():
        decision = parse_closure_decision(req.message)
        if decision == "accept":
            early = await try_accept_path(pending, incoming, rag_label)
            if early is not None:
                return early
        elif decision == "decline":
            return decline_path(pending, incoming, rag_label)

    with trace_request("chat", message=req.message[:200]) as trace_id:
        hint_messages = closure_hint_messages(pending, decision, req.message)
        extracted_types = await extract_requested_primary_types(
            req.message,
            getattr(request.app.state, "agent_llm", None),
        )
        refinement_messages = refinement_messages_for(req.message, incoming)
        num_stops = explicit_num_stops_from_conversation(req.history, req.message)
        state = initial_chat_state(
            req,
            incoming,
            hint_messages=hint_messages,
            refinement_messages=refinement_messages,
            requested_primary_types=extracted_types,
            num_stops=num_stops,
        )
        raw = await graph.ainvoke(
            state,
            config={
                "callbacks": langgraph_callbacks(),
                "metadata": {"trace_id": trace_id},
            },
        )
    final_state = raw if isinstance(raw, ItineraryState) else ItineraryState(**raw)

    response = ChatResponse(
        reply=final_state.final_reply or "",
        places=state_to_cards(final_state),
        ragLabel=rag_label,
        conversation_state=build_outbound_state(final_state.closure_context, final_state.stops),
    )
    background_tasks.add_task(
        log_user_query,
        message=req.message,
        requested_primary_types=extracted_types,
        num_stops=num_stops,
        rag_label=rag_label,
        session_id=trace_id,
    )
    return response


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
