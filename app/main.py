from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .bootstrap import load_registered_rag_chain
from .config import get_settings
from .db import close_pool
from .routes import router

logging.basicConfig(
    level=get_settings().log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_chain = None
    app.state.active_model_config = None
    app.state.startup_error = None
    try:
        rag_chain, model_config = load_registered_rag_chain()
        app.state.rag_chain = rag_chain
        app.state.active_model_config = model_config
    except Exception as exc:
        logger.exception("Failed to load RAG chain at startup")
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
    try:
        yield
    finally:
        close_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_origin_regex=settings.cors_allowed_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
