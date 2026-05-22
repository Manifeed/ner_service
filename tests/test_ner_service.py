from __future__ import annotations

from app.schemas.ner_schema import NerBatchRequestSchema, NerRequestSchema
from app.services.label_service import NER_LABELS
from app.services.ner_service import create_entities_batch


def test_ner_labels_are_fixed_common_labels() -> None:
	assert NER_LABELS == (
		"COMPANY",
		"CURRENCY",
		"DATE",
		"EVENT",
		"LOCATION",
		"ORGANIZATION",
		"PERSON",
		"PRODUCT",
		"TECHNOLOGY",
	)


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
				),
				NerRequestSchema(
					article_id=2,
					title="Budget vote in parliament",
					summary=None,
				),
			]
		)
	)

	assert [item.article_id for item in response.data] == [1, 2]
	assert requests_seen[0].labels == NER_LABELS
	assert requests_seen[1].labels == NER_LABELS


def test_resolve_ner_device_defaults_to_cuda(monkeypatch) -> None:
	from app.domain.config import resolve_ner_device

	monkeypatch.delenv("NER_DEVICE", raising=False)

	assert resolve_ner_device() == "cuda"
