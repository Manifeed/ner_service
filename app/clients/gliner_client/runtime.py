from __future__ import annotations

from typing import Any

from app.errors import NerModelError
from app.schemas.ner_schema import NerEntityRead


class GlinerRuntime:
    def __init__(
        self,
        *,
        model_source: str,
        device: str,
        gpu_memory_fraction: float,
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.gpu_memory_fraction = gpu_memory_fraction
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from gliner import GLiNER  # type: ignore[import-not-found]
            import torch
            from transformers import AutoTokenizer
        except Exception as exception:
            raise NerModelError("Python package gliner, torch, or transformers is not installed") from exception

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise NerModelError("CUDA GPU is required for NER but no GPU is available")
            try:
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, device=torch.cuda.current_device())
            except Exception:
                pass

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_source, local_files_only=True)
            model = GLiNER.from_pretrained(self.model_source)
            if hasattr(model, "to"):
                model = model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            self._model = model
        except Exception as exception:
            raise NerModelError(f"Unable to load GLiNER model from {self.model_source} on {self.device}") from exception

    def predict_batch(
        self,
        *,
        texts: list[str],
        labels: tuple[str, ...],
        threshold: float,
        batch_size: int,
    ) -> list[list[NerEntityRead]]:
        if self._model is None:
            raise NerModelError("NER model is not loaded")
        try:
            raw_outputs = self._predict_entities(texts=texts, labels=list(labels), threshold=threshold, batch_size=batch_size)
        except Exception as exception:
            raise NerModelError(f"Unable to extract entities: {exception}") from exception
        outputs = [self._coerce_entities(item) for item in raw_outputs]
        if len(outputs) != len(texts):
            raise NerModelError(
                f"NER runtime returned {len(outputs)} entity payloads for {len(texts)} inputs"
            )
        return outputs

    def estimate_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return max(1, len(text.split()))
        token_ids = self._tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=1024)
        return max(1, len(token_ids))

    def is_cuda_out_of_memory_error(self, exception: NerModelError) -> bool:
        return "out of memory" in str(exception).lower()

    def _predict_entities(
        self,
        *,
        texts: list[str],
        labels: list[str],
        threshold: float,
        batch_size: int,
    ) -> list[Any]:
        assert self._model is not None
        batch_predict_entities = getattr(self._model, "batch_predict_entities", None)
        if callable(batch_predict_entities):
            try:
                return batch_predict_entities(texts, labels, threshold=threshold, batch_size=batch_size)
            except TypeError:
                return batch_predict_entities(texts, labels, threshold=threshold)
        predict_entities = getattr(self._model, "predict_entities", None)
        if callable(predict_entities):
            return [predict_entities(text, labels, threshold=threshold) for text in texts]
        raise NerModelError("GLiNER runtime does not expose a compatible prediction method")

    def _coerce_entities(self, value: Any) -> list[NerEntityRead]:
        entities = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(entities, list):
            raise NerModelError("NER entity payload must be a list")
        coerced: list[NerEntityRead] = []
        for item in entities:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            text = item.get("text")
            if not label or not text:
                continue
            coerced.append(
                NerEntityRead(
                    label=str(label),
                    text=str(text),
                    score=self._optional_float(item.get("score")),
                    start_offset=self._optional_int(item.get("start")),
                    end_offset=self._optional_int(item.get("end")),
                )
            )
        return coerced

    def _optional_float(self, value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
