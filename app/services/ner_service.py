from __future__ import annotations

from app.clients.gliner_client import NerInferenceRequest, get_gliner_client
from app.domain.config import resolve_ner_threshold
from app.schemas.ner_schema import (
    NerBatchItemRead,
    NerBatchRequestSchema,
    NerBatchResponseRead,
    NerRequestSchema,
    NerResponseRead,
)
from app.services.label_service import resolve_ner_labels


def create_entities(payload: NerRequestSchema) -> NerResponseRead:
    request = _build_inference_request(payload)
    entities = get_gliner_client().extract(request)
    return NerResponseRead(entities=entities)


def create_entities_batch(payload: NerBatchRequestSchema) -> NerBatchResponseRead:
    requests = [_build_inference_request(item) for item in payload.items]
    results = get_gliner_client().extract_batch(requests)
    return NerBatchResponseRead(
        data=[
            NerBatchItemRead(
                index=index,
                article_id=item.article_id,
                entities=entities,
            )
            for index, (item, entities) in enumerate(zip(payload.items, results, strict=True))
        ]
    )


def _build_inference_request(payload: NerRequestSchema) -> NerInferenceRequest:
    return NerInferenceRequest(
        article_id=payload.article_id,
        normalized_text=_build_text(payload.title, payload.summary),
        labels=tuple(resolve_ner_labels(payload.themes)),
        threshold=resolve_ner_threshold(),
    )


def _build_text(title: str, summary: str | None) -> str:
    return "\n\n".join(part.strip() for part in (title, summary or "") if part and part.strip())
