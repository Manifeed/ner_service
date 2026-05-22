from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.ner_schema import NerBatchRequestSchema, NerRequestSchema


def test_ner_request_schema_requires_article_id_title() -> None:
	request = NerRequestSchema(
		article_id=1,
		title="Ada",
		summary=None,
	)

	assert request.article_id == 1
	assert request.title == "Ada"
	assert request.summary is None


def test_ner_batch_request_schema_requires_at_least_one_item() -> None:
	with pytest.raises(ValidationError):
		NerBatchRequestSchema(items=[])
