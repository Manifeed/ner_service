from __future__ import annotations

from app.schemas.ner_schema import NerRequestSchema
from app.services.gliner_runtime import NerModelUnavailable
from app.services.label_service import resolve_ner_labels
from app.services.ner_service import extract_entities


class FakeGlinerModel:
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5):
        assert "PERSON" in labels
        assert "SOFTWARE" in labels
        assert threshold == 0.5
        return [{"label": "PERSON", "text": "Ada", "score": 0.9, "start": 0, "end": 3}]


def test_resolve_ner_labels_adds_common_and_theme_specific_labels() -> None:
    labels = resolve_ner_labels(["technology", "politics"])

    assert "PERSON" in labels
    assert "ORGANIZATION" in labels
    assert "SOFTWARE" in labels
    assert "GOVERNMENT_BODY" in labels


def test_extract_entities_uses_resolved_labels() -> None:
    response = extract_entities(
        NerRequestSchema(
            article_id=1,
            title="Ada ships a software product",
            summary=None,
            language="en",
            themes=["technology"],
        ),
        model=FakeGlinerModel(),
    )

    assert response.entities[0].label == "PERSON"
    assert response.entities[0].text == "Ada"


def test_extract_entities_raises_when_model_is_missing(monkeypatch) -> None:
    import app.services.ner_service as ner_service

    def raise_model_unavailable():
        raise NerModelUnavailable("Python package gliner is not installed")

    monkeypatch.setattr(ner_service, "get_gliner_model", raise_model_unavailable)

    try:
        extract_entities(
            NerRequestSchema(
                article_id=1,
                title="Ada ships a software product",
                summary=None,
                language="en",
                themes=["technology"],
            )
        )
    except NerModelUnavailable as exception:
        assert "gliner" in str(exception)
    else:
        raise AssertionError("Expected NerModelUnavailable")


def test_resolve_ner_device_defaults_to_cuda(monkeypatch) -> None:
    from app.domain.config import resolve_ner_device

    monkeypatch.delenv("NER_DEVICE", raising=False)

    assert resolve_ner_device() == "cuda"
