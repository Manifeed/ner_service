from __future__ import annotations

from app.schemas.ner_schema import NerEntityRead, NerRequestSchema, NerResponseRead
from app.services.gliner_runtime import GlinerLikeModel, get_gliner_model
from app.services.label_service import resolve_ner_labels


def extract_entities(payload: NerRequestSchema, model: GlinerLikeModel | None = None) -> NerResponseRead:
    labels = resolve_ner_labels(payload.themes)
    text = _build_text(payload.title, payload.summary)
    active_model = model or get_gliner_model()
    if not text:
        return NerResponseRead(entities=[])
    raw_entities = active_model.predict_entities(text, labels, threshold=0.5)
    entities = [_coerce_entity(raw_entity) for raw_entity in raw_entities if isinstance(raw_entity, dict)]
    return NerResponseRead(entities=[entity for entity in entities if entity is not None])


def _build_text(title: str, summary: str | None) -> str:
    return "\n\n".join(part.strip() for part in (title, summary or "") if part and part.strip())


def _coerce_entity(raw_entity: dict[str, object]) -> NerEntityRead | None:
    label = raw_entity.get("label")
    text = raw_entity.get("text")
    if not label or not text:
        return None
    return NerEntityRead(
        label=str(label),
        text=str(text),
        score=_optional_float(raw_entity.get("score")),
        start_offset=_optional_int(raw_entity.get("start")),
        end_offset=_optional_int(raw_entity.get("end")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
