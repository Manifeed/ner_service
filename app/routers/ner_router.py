from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.ner_schema import NerRequestSchema, NerResponseRead
from app.services.api_key_auth_service import require_ner_service_api_key_auth
from app.services.ner_service import extract_entities


ner_router = APIRouter(prefix="/v1", tags=["entities"])


@ner_router.post(
    "/entities",
    response_model=NerResponseRead,
    dependencies=[Depends(require_ner_service_api_key_auth)],
)
def read_entities(payload: NerRequestSchema) -> NerResponseRead:
    return extract_entities(payload)
