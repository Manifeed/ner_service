from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.gliner_client import get_gliner_client
from app.domain.config import CANONICAL_SERVICE_NAME
from app.errors import register_exception_handlers
from app.routers.ner_router import ner_router
from app.schemas.ner_schema import InternalServiceHealthRead
from app.services.gliner_runtime import check_gliner_model_ready


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    client = get_gliner_client()
    client.start_warmup()
    try:
        yield
    finally:
        client.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Manifeed NER Service", lifespan=_app_lifespan)
    app.include_router(ner_router)
    register_exception_handlers(app)

    @app.get("/internal/health", response_model=InternalServiceHealthRead)
    def read_internal_health() -> InternalServiceHealthRead:
        return InternalServiceHealthRead(service=CANONICAL_SERVICE_NAME, status="ok")

    @app.get("/internal/ready", response_model=InternalServiceHealthRead)
    def read_internal_ready() -> InternalServiceHealthRead:
        check_gliner_model_ready()
        return InternalServiceHealthRead(service=CANONICAL_SERVICE_NAME, status="ready")

    return app


app = create_app()
