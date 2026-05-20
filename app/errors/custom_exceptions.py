from __future__ import annotations


class NerServiceError(RuntimeError):
    """Base class for stable NER service errors exposed over HTTP."""

    code = "ner_model_unavailable"
    default_message = "Unable to serve NER entities"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class NerRuntimeNotReadyError(NerServiceError):
    code = "warming_up"
    default_message = "ner_service is still warming up"


class NerOverloadedError(NerServiceError):
    code = "overloaded"
    default_message = "ner_service queue is full"


class NerStoppingError(NerServiceError):
    code = "stopping"
    default_message = "ner_service is stopping"


class NerRequestTimeoutError(NerServiceError):
    code = "request_timeout"
    default_message = "ner_service request timed out"


class NerModelError(NerServiceError):
    code = "ner_model_unavailable"
    default_message = "Unable to serve NER entities"
