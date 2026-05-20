from __future__ import annotations

import logging
import threading
import time

from app.domain.batching import NerTaskSignature, effective_max_items, effective_max_tokens

from .models import NerTask
from .settings import GlinerClientSettings


logger = logging.getLogger(__name__)


class NerTaskQueue:
    def __init__(self, *, condition: threading.Condition, settings: GlinerClientSettings) -> None:
        self._condition = condition
        self._settings = settings
        self._pending: list[NerTask] = []
        self._adaptive_batch_items: dict[NerTaskSignature, int] = {}

    def size(self) -> int:
        return len(self._pending)

    def is_full(self) -> bool:
        return len(self._pending) >= self._settings.queue_max_items

    def enqueue(self, task: NerTask) -> None:
        self._pending.append(task)
        self._condition.notify_all()
        logger.info(
            "ner_service task queued queue_size=%s queue_capacity=%s estimated_tokens=%s labels=%s",
            len(self._pending),
            self._settings.queue_max_items,
            task.token_estimate,
            len(task.request.labels),
        )

    def collect_batch(self, *, stop_event: threading.Event) -> list[NerTask]:
        while True:
            self._discard_cancelled_tasks()
            while not self._pending and not stop_event.is_set():
                self._condition.wait(timeout=0.1)
                self._discard_cancelled_tasks()
            if stop_event.is_set():
                return []

            first_task = self._pending[0]
            signature = first_task.request.signature
            deadline = first_task.enqueued_at + (self._settings.batch_max_wait_ms / 1000.0)
            indexes, token_count = self._select_batch(signature)
            max_items = self.effective_max_items(signature)
            max_tokens = self.effective_max_tokens(signature)
            should_flush = len(indexes) >= max_items or token_count >= max_tokens
            if should_flush or time.monotonic() >= deadline:
                return self._pop_selected_batch(indexes=indexes, token_count=token_count, first_task=first_task)
            remaining = max(0.0, deadline - time.monotonic())
            self._condition.wait(timeout=remaining)
            if stop_event.is_set():
                return []

    def drain(self) -> list[NerTask]:
        tasks = list(self._pending)
        self._pending.clear()
        return tasks

    def effective_batch_size(self, *, signature: NerTaskSignature, task_count: int) -> int:
        return min(task_count, self.effective_max_items(signature))

    def effective_max_items(self, signature: NerTaskSignature) -> int:
        return effective_max_items(
            configured_limit=self._settings.batch_max_items,
            adaptive_limit=self._adaptive_batch_items.get(signature),
        )

    def effective_max_tokens(self, signature: NerTaskSignature) -> int:
        return effective_max_tokens(configured_limit=self._settings.batch_max_tokens)

    def reduce_batch_size(self, *, signature: NerTaskSignature, previous_batch_size: int) -> int:
        reduced_batch_size = max(1, previous_batch_size // 2)
        self._adaptive_batch_items[signature] = reduced_batch_size
        return reduced_batch_size

    def _discard_cancelled_tasks(self) -> None:
        if not self._pending:
            return
        self._pending = [task for task in self._pending if not task.future.cancelled()]

    def _select_batch(self, signature: NerTaskSignature) -> tuple[list[int], int]:
        indexes: list[int] = []
        token_count = 0
        max_items = self.effective_max_items(signature)
        max_tokens = self.effective_max_tokens(signature)
        for index, task in enumerate(self._pending):
            if task.request.signature != signature:
                continue
            next_token_count = token_count + task.token_estimate
            if indexes and next_token_count > max_tokens:
                break
            indexes.append(index)
            token_count = next_token_count
            if len(indexes) >= max_items:
                break
        if indexes:
            return indexes, token_count
        if not self._pending:
            return [], 0
        first_task = self._pending[0]
        return [0], first_task.token_estimate

    def _pop_selected_batch(
        self,
        *,
        indexes: list[int],
        token_count: int,
        first_task: NerTask,
    ) -> list[NerTask]:
        batch = [self._pending[index] for index in indexes]
        for index in reversed(indexes):
            del self._pending[index]
        oldest_wait_ms = max(0.0, (time.monotonic() - first_task.enqueued_at) * 1000.0)
        logger.info(
            "ner_service batch dispatched batch_items=%s queue_remaining=%s estimated_tokens=%s wait_ms=%.2f labels=%s",
            len(batch),
            len(self._pending),
            token_count,
            oldest_wait_ms,
            len(first_task.request.labels),
        )
        return batch
