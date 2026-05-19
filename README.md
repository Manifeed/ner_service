# Manifeed NER Service

`ner_service` is the internal named-entity recognition worker for Manifeed.

It exposes a FastAPI API that extracts entities from article titles and
summaries using a pre-baked [GLiNER](https://github.com/urchade/GLiNER) model.
Entity labels are derived from article themes plus a shared core label set.

This service is not browser-facing. It is meant to run inside the backend
platform and is consumed by `indexer_service` and other internal workers.

## Responsibilities

- Extract named entities from article title and summary text
- Resolve NER labels from article themes (politics, technology, sports, etc.)
- Expose internal health and readiness endpoints
- Authenticate callers with a shared API key

## Service Endpoints

- `GET /internal/health`: process-level liveness check
- `GET /internal/ready`: verifies the GLiNER model is loaded and available
- `POST /v1/entities`: entity extraction (requires Bearer token)

## Architecture Overview

- [app/main.py](./app/main.py): FastAPI bootstrap and health routes
- [app/routers/ner_router.py](./app/routers/ner_router.py): `/v1/entities` route
- [app/services/ner_service.py](./app/services/ner_service.py): extraction orchestration
- [app/services/gliner_runtime.py](./app/services/gliner_runtime.py): GLiNER model lifecycle
- [app/services/label_service.py](./app/services/label_service.py): theme-to-label mapping
- [app/domain/config.py](./app/domain/config.py): runtime configuration resolution

## Quick Start

### 1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Export a minimal local environment

```bash
export NER_SERVICE_API_KEY='replace-with-local-service-key'
export GLINER_MODEL_PATH='/opt/models/gliner'
export NER_DEVICE='cuda'
```

For CPU-only local experiments (slower, not recommended for production):

```bash
export NER_DEVICE='cpu'
```

### 3. Run the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API

### Request

```json
{
  "article_id": 42,
  "title": "Ada Lovelace launches a software startup",
  "summary": "The company focuses on developer tools.",
  "language": "en",
  "themes": ["technology"]
}
```

### Response

```json
{
  "entities": [
    {
      "label": "PERSON",
      "text": "Ada Lovelace",
      "score": 0.91,
      "start_offset": 0,
      "end_offset": 13
    }
  ]
}
```

### `curl` example

```bash
curl http://127.0.0.1:8000/v1/entities \
  -H "Authorization: Bearer ${NER_SERVICE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "article_id": 1,
    "title": "Ada Lovelace launches a software startup",
    "summary": null,
    "language": "en",
    "themes": ["technology"]
  }'
```

## Configuration

Core settings used by the service:

- `NER_SERVICE_API_KEY`: required Bearer token for `/v1/entities`
- `GLINER_MODEL_PATH`: on-disk model directory (default `/opt/models/gliner`)
- `GLINER_MODEL_ID`: Hugging Face model id used at image build time
- `NER_DEVICE`: inference device (`cuda` or `cpu`, default `cuda`)

## Tests

Run the test suite with:

```bash
pytest -q
```

Tests use a fake GLiNER model and do not require GPU access.

## Docker

Build from the repository root:

```bash
docker build -t manifeed-ner-service .
```

Run (requires NVIDIA GPU in production):

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e NER_SERVICE_API_KEY='replace-with-strong-secret' \
  -e NER_DEVICE='cuda' \
  manifeed-ner-service
```

The image pre-downloads the GLiNER model at build time and runs offline at
runtime (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

## Manifeed Integration

In the Manifeed `infra` stack, this service is referenced as `ner_service` and
is reached by `indexer_service` through `NER_SERVICE_URL`. Clone this
repository next to the main platform checkout and point
`NER_SERVICE_REPO_PATH` to it when building with Docker Compose.
