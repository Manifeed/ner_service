from __future__ import annotations

from concurrent.futures import Future, TimeoutError
import logging
import threading
import time

from app.domain.runtime_state import NerRuntimeState
from app.errors import (
    NerModelError,
    NerOverloadedError,
    NerRequestTimeoutError,
    NerRuntimeNotReadyError,
    NerStoppingError,
)
from app.schemas.ner_schema import NerEntityRead

from .batch_executor import GlinerBatchExecutor
from .models import NerInferenceRequest, NerTask
from .queue_manager import NerTaskQueue
from .runtime import GlinerRuntime
from .settings import build_gliner_client_settings


logger = logging.getLogger(__name__)


class GlinerClient:
    def __init__(self) -> None:
        self.settings = build_gliner_client_settings()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._warmup_thread: threading.Thread | None = None
        self._load_error: NerModelError | None = None
        self._warmup_started = False
        self._worker_started = False
        self._state = NerRuntimeState.STARTING

        self._runtime = GlinerRuntime(
            model_source=self.settings.model_source,
            device=self.settings.device,
            gpu_memory_fraction=self.settings.gpu_memory_fraction,
        )
        self._task_queue = NerTaskQueue(condition=self._condition, settings=self.settings)
        self._batch_executor = GlinerBatchExecutor(runtime=self._runtime, task_queue=self._task_queue)

    def start_warmup(self) -> None:
        with self._condition:
            if self._state in {NerRuntimeState.STOPPING, NerRuntimeState.STOPPED}:
                return
            self._start_worker_locked()
            if self._warmup_started:
                return
            self._warmup_started = True
            self._transition_state_locked(NerRuntimeState.STARTING, reason="warmup_started")
            self._warmup_thread = threading.Thread(target=self._warmup_runtime, name="ner-runtime-warmup", daemon=True)
            self._warmup_thread.start()

    def stop(self) -> None:
        with self._condition:
            if self._state == NerRuntimeState.STOPPED:
                return
            self._transition_state_locked(NerRuntimeState.STOPPING, reason="shutdown_requested")
            self._stop_event.set()
            pending_tasks = self._task_queue.drain()
            self._condition.notify_all()
        self._batch_executor.fail_tasks(pending_tasks, NerStoppingError())
        self._join_thread(self._worker_thread, name="worker")
        self._join_thread(self._warmup_thread, name="warmup")
        with self._condition:
            self._transition_state_locked(NerRuntimeState.STOPPED, reason="shutdown_completed")
            self._condition.notify_all()

    def extract(self, request: NerInferenceRequest) -> list[NerEntityRead]:
        return self.extract_batch([request])[0]

    def extract_batch(self, requests: list[NerInferenceRequest]) -> list[list[NerEntityRead]]:
        if not requests:
            return []
        self.start_warmup()
        self._ensure_serving_ready(require_queue_capacity=False)

        futures: list[Future[list[NerEntityRead]]] = []
        for request in requests:
            if not request.normalized_text:
                future: Future[list[NerEntityRead]] = Future()
                future.set_result([])
                futures.append(future)
                continue
            futures.append(self._submit_task(request))
        deadline = time.monotonic() + self.settings.request_timeout_seconds
        return self._collect_results(futures=futures, deadline=deadline)

    def check_ready(self) -> None:
        self.start_warmup()
        self._ensure_serving_ready(require_queue_capacity=True)

    def _collect_results(
        self,
        *,
        futures: list[Future[list[NerEntityRead]]],
        deadline: float,
    ) -> list[list[NerEntityRead]]:
        results: list[list[NerEntityRead]] = []
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._cancel_futures(futures)
                raise NerRequestTimeoutError()
            try:
                results.append(future.result(timeout=remaining))
            except TimeoutError as exception:
                self._cancel_futures(futures)
                raise NerRequestTimeoutError() from exception
        return results

    def _cancel_futures(self, futures: list[Future[list[NerEntityRead]]]) -> None:
        for future in futures:
            future.cancel()

    def _submit_task(self, request: NerInferenceRequest) -> Future[list[NerEntityRead]]:
        token_estimate = self._runtime.estimate_tokens(request.normalized_text)
        future: Future[list[NerEntityRead]] = Future()
        task = NerTask(
            request=request,
            token_estimate=token_estimate,
            future=future,
            enqueued_at=time.monotonic(),
        )
        with self._condition:
            self._ensure_serving_ready_locked(require_queue_capacity=True)
            self._task_queue.enqueue(task)
        return future

    def _start_worker_locked(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        self._worker_thread = threading.Thread(target=self._worker_loop, name="ner-batch-worker", daemon=True)
        self._worker_thread.start()

    def _warmup_runtime(self) -> None:
        try:
            self._runtime.ensure_loaded()
        except NerModelError as exception:
            with self._condition:
                self._load_error = exception
                self._transition_state_locked(NerRuntimeState.DEGRADED, reason="warmup_failed")
        else:
            with self._condition:
                if self._state not in {NerRuntimeState.STOPPING, NerRuntimeState.STOPPED}:
                    self._transition_state_locked(NerRuntimeState.READY, reason="warmup_completed")
        finally:
            self._ready_event.set()
            with self._condition:
                self._condition.notify_all()

    def _worker_loop(self) -> None:
        logger.info("ner_service worker started")
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                self._process_batch(batch)
            except Exception as exception:  # pragma: no cover - defensive path
                logger.exception("ner_service batch failed")
                error = exception if isinstance(exception, NerModelError) else NerModelError(
                    "Unable to extract entities"
                )
                self._batch_executor.fail_tasks(batch, error)
        logger.info("ner_service worker stopped")

    def _collect_batch(self) -> list[NerTask]:
        with self._condition:
            return self._task_queue.collect_batch(stop_event=self._stop_event)

    def _process_batch(self, batch: list[NerTask]) -> None:
        self._ensure_runtime_loaded_for_batch()
        self._batch_executor.process_batch(batch)

    def _ensure_serving_ready(self, *, require_queue_capacity: bool) -> None:
        with self._condition:
            self._ensure_serving_ready_locked(require_queue_capacity=require_queue_capacity)

    def _ensure_serving_ready_locked(self, *, require_queue_capacity: bool) -> None:
        if self._state in {NerRuntimeState.STOPPING, NerRuntimeState.STOPPED} or self._stop_event.is_set():
            raise NerStoppingError()
        if not self._ready_event.is_set():
            raise NerRuntimeNotReadyError()
        if self._load_error is not None:
            self._transition_state_locked(NerRuntimeState.DEGRADED, reason="load_error_present")
            raise self._load_error
        if not self._worker_is_alive():
            self._transition_state_locked(NerRuntimeState.DEGRADED, reason="worker_not_alive")
            raise NerModelError("ner_service batch worker is not running")
        if require_queue_capacity and self._task_queue.is_full():
            raise NerOverloadedError()
        if self._state == NerRuntimeState.STARTING:
            self._transition_state_locked(NerRuntimeState.READY, reason="runtime_ready")

    def _ensure_runtime_loaded_for_batch(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        if not self._ready_event.is_set():
            raise NerRuntimeNotReadyError()

    def _worker_is_alive(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _join_thread(self, thread: threading.Thread | None, *, name: str) -> None:
        if thread is None:
            return
        thread.join(timeout=self.settings.shutdown_grace_seconds)
        if thread.is_alive():
            logger.warning(
                "ner_service %s join timed out shutdown_grace_s=%.2f",
                name,
                self.settings.shutdown_grace_seconds,
            )

    def _transition_state_locked(self, new_state: NerRuntimeState, *, reason: str) -> None:
        if self._state == new_state:
            return
        previous_state = self._state
        self._state = new_state
        logger.info(
            "ner_service state transition previous=%s current=%s reason=%s queue_size=%s",
            previous_state.value,
            new_state.value,
            reason,
            self._task_queue.size(),
        )
