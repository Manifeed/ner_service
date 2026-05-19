from __future__ import annotations

import os


CANONICAL_SERVICE_NAME = "ner_service"
DEFAULT_GLINER_MODEL_PATH = "/opt/models/gliner"
DEFAULT_NER_DEVICE = "cuda"


def resolve_ner_service_api_key() -> str:
    return os.getenv("NER_SERVICE_API_KEY", "").strip()


def resolve_gliner_model_path() -> str:
    return os.getenv("GLINER_MODEL_PATH", DEFAULT_GLINER_MODEL_PATH).strip() or DEFAULT_GLINER_MODEL_PATH


def resolve_ner_device() -> str:
    return os.getenv("NER_DEVICE", DEFAULT_NER_DEVICE).strip().casefold() or DEFAULT_NER_DEVICE
