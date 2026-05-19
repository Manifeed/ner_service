from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from app.domain.config import resolve_gliner_model_path, resolve_ner_device


class NerModelUnavailable(RuntimeError):
    """Raised when GLiNER is not installed or cannot load its configured model."""


class GlinerLikeModel(Protocol):
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5): ...


@lru_cache(maxsize=1)
def get_gliner_model() -> GlinerLikeModel | None:
    model_path = resolve_gliner_model_path()
    device = resolve_ner_device()
    try:
        from gliner import GLiNER  # type: ignore[import-not-found]
        import torch
    except Exception as exception:
        raise NerModelUnavailable("Python package gliner or torch is not installed") from exception
    if device == "cuda" and not torch.cuda.is_available():
        raise NerModelUnavailable("CUDA GPU is required for NER but no GPU is available")
    try:
        model = GLiNER.from_pretrained(model_path)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model
    except Exception as exception:
        raise NerModelUnavailable(f"Unable to load GLiNER model from {model_path} on {device}") from exception


def check_gliner_model_ready() -> None:
    get_gliner_model()
