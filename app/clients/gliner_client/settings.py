from __future__ import annotations

from dataclasses import dataclass

from app.domain.config import (
    resolve_gliner_model_path,
    resolve_ner_batch_max_items,
    resolve_ner_batch_max_tokens,
    resolve_ner_batch_max_wait_ms,
    resolve_ner_device,
    resolve_ner_gpu_memory_fraction,
    resolve_ner_queue_max_items,
    resolve_ner_request_timeout_seconds,
    resolve_ner_shutdown_grace_seconds,
)


@dataclass
class GlinerClientSettings:
    model_source: str
    device: str
    batch_max_items: int
    batch_max_tokens: int
    batch_max_wait_ms: int
    queue_max_items: int
    request_timeout_seconds: float
    shutdown_grace_seconds: float
    gpu_memory_fraction: float


def build_gliner_client_settings() -> GlinerClientSettings:
    return GlinerClientSettings(
        model_source=resolve_gliner_model_path(),
        device=resolve_ner_device(),
        batch_max_items=resolve_ner_batch_max_items(),
        batch_max_tokens=resolve_ner_batch_max_tokens(),
        batch_max_wait_ms=resolve_ner_batch_max_wait_ms(),
        queue_max_items=resolve_ner_queue_max_items(),
        request_timeout_seconds=resolve_ner_request_timeout_seconds(),
        shutdown_grace_seconds=resolve_ner_shutdown_grace_seconds(),
        gpu_memory_fraction=resolve_ner_gpu_memory_fraction(),
    )
