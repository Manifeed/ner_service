from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.ner_schema import (
    NerBatchRequestSchema,
    NerBatchResponseRead,
    NerRequestSchema,
    NerResponseRead,
)
from app.services.api_key_auth_service import require_ner_service_api_key_auth
from app.services.ner_service import create_entities, create_entities_batch


ner_router = APIRouter(prefix="/v1", tags=["entities"])


@ner_router.post(
    "/entities",
    response_model=NerResponseRead,
    dependencies=[Depends(require_ner_service_api_key_auth)],
)
def read_entities(payload: NerRequestSchema) -> NerResponseRead:
    return create_entities(payload)


@ner_router.post(
    "/entities/batch",
    response_model=NerBatchResponseRead,
    dependencies=[Depends(require_ner_service_api_key_auth)],
)
def read_entities_batch(payload: NerBatchRequestSchema) -> NerBatchResponseRead:
    return create_entities_batch(payload)
