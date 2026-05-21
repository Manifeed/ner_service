from __future__ import annotations

from fastapi.testclient import TestClient

from app.errors import (
    NerModelError,
    NerOverloadedError,
    NerRequestTimeoutError,
    NerRuntimeNotReadyError,
    NerStoppingError,
)
from app.main import app
from app.schemas.ner_schema import NerBatchResponseRead, NerResponseRead


def _patch_runtime_client(monkeypatch) -> None:
    class DummyRuntimeClient:
        def start_warmup(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr("app.main.get_gliner_client", lambda: DummyRuntimeClient())


def test_internal_ready_returns_503_when_runtime_is_warming_up(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setattr(
        "app.main.check_gliner_model_ready",
        lambda: (_ for _ in ()).throw(NerRuntimeNotReadyError()),
    )

    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "warming_up",
        "message": "ner_service is still warming up",
    }


def test_internal_ready_returns_canonical_service_name(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setattr("app.main.check_gliner_model_ready", lambda: None)

    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 200
    assert response.json() == {
        "service": "ner_service",
        "status": "ready",
    }


def test_entities_route_returns_503_when_runtime_is_unavailable(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setattr(
        "app.routers.ner_router.create_entities",
        lambda payload: (_ for _ in ()).throw(NerModelError("Unable to load GLiNER model")),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/entities",
        json={
            "article_id": 1,
            "title": "Ada",
            "summary": None,
            "language": "en",
            "themes": ["technology"],
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "ner_model_unavailable",
        "message": "Unable to load GLiNER model",
    }


def test_entities_route_returns_401_with_invalid_api_key(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("NER_SERVICE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.ner_router.create_entities",
        lambda payload: NerResponseRead(entities=[]),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/entities",
        json={
            "article_id": 1,
            "title": "Ada",
            "summary": None,
            "language": "en",
            "themes": ["technology"],
        },
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401


def test_entities_batch_route_returns_payload(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setattr(
        "app.routers.ner_router.create_entities_batch",
        lambda payload: NerBatchResponseRead(
            data=[
                {
                    "index": 0,
                    "article_id": 42,
                    "entities": [{"label": "PERSON", "text": "Ada", "score": 0.9}],
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/entities/batch",
        json={
            "items": [
                {
                    "article_id": 42,
                    "title": "Ada",
                    "summary": None,
                    "language": "ARZ",
                    "themes": ["technology"],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["data"][0]["article_id"] == 42


def test_entities_batch_route_normalizes_padded_iso_language_codes(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    seen = {}

    def fake_create_entities_batch(payload):
        seen["language"] = payload.items[0].language
        return NerBatchResponseRead(data=[{"index": 0, "article_id": 42, "entities": []}])

    monkeypatch.setattr("app.routers.ner_router.create_entities_batch", fake_create_entities_batch)

    client = TestClient(app)
    response = client.post(
        "/v1/entities/batch",
        json={
            "items": [
                {
                    "article_id": 42,
                    "title": "Ada",
                    "summary": None,
                    "language": "fr ",
                    "themes": ["technology"],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert seen["language"] == "fr"


def test_entities_route_returns_specific_service_errors(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    client = TestClient(app)

    for exception, expected_code in (
        (NerOverloadedError(), "overloaded"),
        (NerStoppingError(), "stopping"),
        (NerRequestTimeoutError(), "request_timeout"),
    ):
        monkeypatch.setattr(
            "app.routers.ner_router.create_entities",
            lambda payload, exception=exception: (_ for _ in ()).throw(exception),
        )
        response = client.post(
            "/v1/entities",
            json={
                "article_id": 1,
                "title": "Ada",
                "summary": None,
                "language": "en",
                "themes": ["technology"],
            },
        )
        assert response.status_code == 503
        assert response.json()["code"] == expected_code
