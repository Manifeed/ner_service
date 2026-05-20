from __future__ import annotations

from app.clients.gliner_client import get_gliner_client
from app.errors import NerModelError


NerModelUnavailable = NerModelError


def check_gliner_model_ready() -> None:
    get_gliner_client().check_ready()
