from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
import time

import pytest

from app.clients.gliner_client.models import NerInferenceRequest, NerTask
from app.domain.runtime_state import NerRuntimeState
from app.errors import NerModelError, NerOverloadedError, NerRequestTimeoutError, NerStoppingError


def _mark_client_ready(client) -> None:
    client._ready_event.set()
    client._state = NerRuntimeState.READY
    client._worker_thread = SimpleNamespace(is_alive=lambda: True)


def _build_request(*, article_id: int = 1, text: str = "hello", labels: tuple[str, ...] = ("PERSON",)) -> NerInferenceRequest:
    return NerInferenceRequest(
        article_id=article_id,
        normalized_text=text,
        labels=labels,
        threshold=0.5,
    )


def test_submit_task_rejects_when_queue_is_full() -> None:
    from app.clients.gliner_client.client import GlinerClient

    client = GlinerClient()
    _mark_client_ready(client)
    client.settings.queue_max_items = 1
    request = _build_request()
    client._task_queue._pending = [
        NerTask(request=request, token_estimate=1, future=Future(), enqueued_at=0.0),
    ]

    with pytest.raises(NerOverloadedError):
        client._submit_task(request)


def test_extract_batch_returns_empty_for_blank_payload(monkeypatch) -> None:
    from app.clients.gliner_client.client import GlinerClient

    client = GlinerClient()
    _mark_client_ready(client)
    monkeypatch.setattr(client, "start_warmup", lambda: None)

    results = client.extract_batch([_build_request(text="")])

    assert results == [[]]


def test_collect_results_cancels_outstanding_futures_on_timeout() -> None:
    from app.clients.gliner_client.client import GlinerClient

    client = GlinerClient()
    completed = Future()
    completed.set_result([])
    pending = Future()

    with pytest.raises(NerRequestTimeoutError):
        client._collect_results(futures=[completed, pending], deadline=time.monotonic() - 1.0)

    assert pending.cancelled()


def test_stop_drains_pending_tasks_with_stopping_error() -> None:
    from app.clients.gliner_client.client import GlinerClient

    client = GlinerClient()
    future = Future()
    request = _build_request()
    client._task_queue._pending = [
        NerTask(request=request, token_estimate=1, future=future, enqueued_at=0.0),
    ]

    client.stop()

    assert isinstance(future.exception(), NerStoppingError)
    assert client._state == NerRuntimeState.STOPPED


def test_process_batch_retries_with_smaller_batch_after_cuda_oom(monkeypatch) -> None:
    from app.clients.gliner_client.client import GlinerClient

    client = GlinerClient()
    client._ready_event.set()
    request = _build_request(labels=("PERSON", "SOFTWARE"))
    futures = [Future(), Future(), Future(), Future()]
    batch = [
        NerTask(request=request, token_estimate=4, future=future, enqueued_at=0.0)
        for future in futures
    ]
    calls: list[int] = []

    def fake_predict_batch(*, texts, labels, threshold, batch_size):
        calls.append(batch_size)
        if len(calls) == 1:
            raise NerModelError("CUDA out of memory")
        return [[] for _ in texts]

    monkeypatch.setattr(client._runtime, "predict_batch", fake_predict_batch)

    client._process_batch(batch)

    assert calls == [4, 2]
    assert client._task_queue._adaptive_batch_items[request.signature] == 2

