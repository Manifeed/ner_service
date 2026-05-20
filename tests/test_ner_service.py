from __future__ import annotations

from app.schemas.ner_schema import NerBatchRequestSchema, NerRequestSchema
from app.services.label_service import resolve_ner_labels
from app.services.ner_service import create_entities_batch


def test_resolve_ner_labels_adds_common_and_theme_specific_labels() -> None:
    labels = resolve_ner_labels(["technology", "politics"])

    assert "PERSON" in labels
    assert "ORGANIZATION" in labels
    assert "SOFTWARE" in labels
    assert "GOVERNMENT_BODY" in labels


def test_create_entities_batch_maps_entities_by_item_order(monkeypatch) -> None:
    import app.services.ner_service as module_under_test

    requests_seen = []

    class DummyClient:
        def extract_batch(self, requests):
            requests_seen.extend(requests)
            return [
                [],
                [],
            ]

    monkeypatch.setattr(module_under_test, "get_gliner_client", lambda: DummyClient())

    response = create_entities_batch(
        NerBatchRequestSchema(
            items=[
                NerRequestSchema(
                    article_id=1,
                    title="Ada ships a software product",
                    summary=None,
                    language="en",
                    themes=["technology"],
                ),
                NerRequestSchema(
                    article_id=2,
                    title="Budget vote in parliament",
                    summary=None,
                    language="fr",
                    themes=["politics"],
                ),
            ]
        )
    )

    assert [item.article_id for item in response.data] == [1, 2]
    assert requests_seen[0].labels == tuple(resolve_ner_labels(["technology"]))
    assert requests_seen[1].labels == tuple(resolve_ner_labels(["politics"]))


def test_resolve_ner_device_defaults_to_cuda(monkeypatch) -> None:
    from app.domain.config import resolve_ner_device

    monkeypatch.delenv("NER_DEVICE", raising=False)

    assert resolve_ner_device() == "cuda"
