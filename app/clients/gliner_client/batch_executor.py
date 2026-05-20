from __future__ import annotations

import logging
import time

from app.domain.batching import NerTaskSignature
from app.errors import NerModelError

from .models import NerTask
from .queue_manager import NerTaskQueue
from .runtime import GlinerRuntime


logger = logging.getLogger(__name__)


class GlinerBatchExecutor:
    def __init__(self, *, runtime: GlinerRuntime, task_queue: NerTaskQueue) -> None:
        self._runtime = runtime
        self._task_queue = task_queue

    def process_batch(self, batch: list[NerTask]) -> None:
        signature = batch[0].request.signature
        texts = [task.request.normalized_text for task in batch]
        estimated_tokens = sum(task.token_estimate for task in batch)
        batch_size = self._task_queue.effective_batch_size(signature=signature, task_count=len(texts))
        start_time = time.perf_counter()

        results = self._predict_with_retry(
            texts=texts,
            signature=signature,
            batch_size=batch_size,
        )

        self._log_completed_batch(
            batch=batch,
            estimated_tokens=estimated_tokens,
            duration_seconds=max(time.perf_counter() - start_time, 1e-9),
            batch_size=batch_size,
            signature=signature,
        )
        self._complete_tasks(batch=batch, results=results)

    def fail_tasks(self, tasks: list[NerTask], error: Exception) -> None:
        for task in tasks:
            if not task.future.done():
                task.future.set_exception(error)

    def _predict_with_retry(
        self,
        *,
        texts: list[str],
        signature: NerTaskSignature,
        batch_size: int,
    ) -> list[list]:
        try:
            return self._runtime.predict_batch(
                texts=texts,
                labels=signature.labels,
                threshold=signature.threshold,
                batch_size=batch_size,
            )
        except NerModelError as exception:
            if not self._runtime.is_cuda_out_of_memory_error(exception) or batch_size <= 1:
                raise
        reduced_batch_size = self._task_queue.reduce_batch_size(
            signature=signature,
            previous_batch_size=batch_size,
        )
        logger.warning(
            "ner_service OOM detected retrying labels=%s previous_batch_size=%s reduced_batch_size=%s",
            len(signature.labels),
            batch_size,
            reduced_batch_size,
        )
        return self._runtime.predict_batch(
            texts=texts,
            labels=signature.labels,
            threshold=signature.threshold,
            batch_size=reduced_batch_size,
        )

    def _log_completed_batch(
        self,
        *,
        batch: list[NerTask],
        estimated_tokens: int,
        duration_seconds: float,
        batch_size: int,
        signature: NerTaskSignature,
    ) -> None:
        tokens_per_second = estimated_tokens / duration_seconds
        logger.info(
            "ner_service batch completed items=%s estimated_tokens=%s duration_s=%.4f tokens_per_second=%.2f effective_batch_size=%s labels=%s",
            len(batch),
            estimated_tokens,
            duration_seconds,
            tokens_per_second,
            batch_size,
            len(signature.labels),
        )

    def _complete_tasks(self, *, batch: list[NerTask], results: list[list]) -> None:
        for index, task in enumerate(batch):
            if not task.future.done():
                task.future.set_result(results[index])
