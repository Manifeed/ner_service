from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.ner_schema import NerBatchRequestSchema, NerRequestSchema


def test_ner_request_schema_strips_sql_padding_from_language() -> None:
    request = NerRequestSchema(
        article_id=1,
        title="Ada",
        summary=None,
        language=" fr ",
        themes=["technology"],
    )

    assert request.language == "fr"


def test_ner_batch_request_schema_accepts_three_letter_iso_language_codes() -> None:
    payload = NerBatchRequestSchema(
        items=[
            NerRequestSchema(
                article_id=1,
                title="Dialect article",
                summary=None,
                language="arz",
                themes=[],
            )
        ]
    )

    assert payload.items[0].language == "arz"


def test_ner_request_schema_rejects_non_alpha_language_codes() -> None:
    with pytest.raises(ValidationError):
        NerRequestSchema(
            article_id=1,
            title="Ada",
            summary=None,
            language="fr-CA",
            themes=[],
        )
