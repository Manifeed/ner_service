from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.gliner_runtime import NerModelUnavailable


def ner_model_unavailable_handler(_: Request, exception: NerModelUnavailable) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "code": "ner_model_unavailable",
            "message": str(exception) or "Unable to serve NER entities",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(NerModelUnavailable, ner_model_unavailable_handler)
