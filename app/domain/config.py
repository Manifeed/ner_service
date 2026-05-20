from __future__ import annotations

import os

from app.domain.gpu_profile import resolve_ner_gpu_batch_profile


CANONICAL_SERVICE_NAME = "ner_service"
DEFAULT_GLINER_MODEL_PATH = "/opt/models/gliner"
DEFAULT_NER_DEVICE = "cuda"
DEFAULT_NER_BATCH_MAX_WAIT_MS = 5
DEFAULT_NER_QUEUE_MAX_ITEMS = 1024
DEFAULT_NER_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_NER_SHUTDOWN_GRACE_SECONDS = 30.0
DEFAULT_NER_GPU_MEMORY_FRACTION = 0.25
DEFAULT_NER_THRESHOLD = 0.5


def resolve_ner_service_api_key() -> str:
    return os.getenv("NER_SERVICE_API_KEY", "").strip()


def resolve_gliner_model_path() -> str:
    return os.getenv("GLINER_MODEL_PATH", DEFAULT_GLINER_MODEL_PATH).strip() or DEFAULT_GLINER_MODEL_PATH


def resolve_gliner_model_id() -> str:
    return os.getenv("GLINER_MODEL_ID", "urchade/gliner_multi-v2.1").strip() or "urchade/gliner_multi-v2.1"


def resolve_ner_device() -> str:
    return os.getenv("NER_DEVICE", DEFAULT_NER_DEVICE).strip().casefold() or DEFAULT_NER_DEVICE


def resolve_ner_threshold() -> float:
    return _bounded_float_env("NER_THRESHOLD", default=DEFAULT_NER_THRESHOLD, minimum=0.0, maximum=1.0)


def resolve_ner_batch_max_items() -> int:
    profile = resolve_ner_gpu_batch_profile()
    return _positive_int_env("NER_BATCH_MAX_ITEMS", default=profile.batch_max_items)


def resolve_ner_batch_max_tokens() -> int:
    profile = resolve_ner_gpu_batch_profile()
    return _positive_int_env("NER_BATCH_MAX_TOKENS", default=profile.batch_max_tokens)


def resolve_ner_batch_max_wait_ms() -> int:
    return _positive_int_env("NER_BATCH_MAX_WAIT_MS", default=DEFAULT_NER_BATCH_MAX_WAIT_MS)


def resolve_ner_queue_max_items() -> int:
    return _positive_int_env("NER_QUEUE_MAX_ITEMS", default=DEFAULT_NER_QUEUE_MAX_ITEMS)


def resolve_ner_request_timeout_seconds() -> float:
    return _positive_float_env(
        "NER_REQUEST_TIMEOUT_SECONDS",
        default=DEFAULT_NER_REQUEST_TIMEOUT_SECONDS,
    )


def resolve_ner_shutdown_grace_seconds() -> float:
    return _positive_float_env(
        "NER_SHUTDOWN_GRACE_SECONDS",
        default=DEFAULT_NER_SHUTDOWN_GRACE_SECONDS,
    )


def resolve_ner_gpu_memory_fraction() -> float:
    return _bounded_float_env(
        "NER_GPU_MEMORY_FRACTION",
        default=DEFAULT_NER_GPU_MEMORY_FRACTION,
        minimum=0.05,
        maximum=0.95,
    )


def _positive_int_env(name: str, *, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def _positive_float_env(name: str, *, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def _bounded_float_env(
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        return default
    if parsed < minimum or parsed > maximum:
        return default
    return parsed
