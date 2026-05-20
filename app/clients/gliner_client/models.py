from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass

from app.domain.batching import NerTaskSignature
from app.schemas.ner_schema import NerEntityRead


NerResult = list[NerEntityRead]


@dataclass(frozen=True)
class NerInferenceRequest:
    article_id: int
    normalized_text: str
    labels: tuple[str, ...]
    threshold: float

    @property
    def signature(self) -> NerTaskSignature:
        return NerTaskSignature(labels=self.labels, threshold=self.threshold)


@dataclass
class NerTask:
    request: NerInferenceRequest
    token_estimate: int
    future: Future[NerResult]
    enqueued_at: float
