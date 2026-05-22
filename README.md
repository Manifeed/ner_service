# Manifeed NER Service

GPU-backed standalone NER service built around GLiNER. It exposes:

- `POST /v1/entities/batch`
- `GET /internal/health`
- `GET /internal/ready`

The service batches requests internally, warms the model once per process, and
adapts batch defaults to detected GPU memory. It is intentionally standalone
and can run on a separate GPU host from the rest of Manifeed.

## Quick Start

```bash
cp .env.example .env
make build-base
make up
```

The service listens on `http://localhost:8002` by default.

## Common Commands

```bash
make build-base
make build
make up
make down
make clean
make test
make release-pull
make release-up
```

## Release Images

GitHub Actions publishes two images on tags matching `v*`:

- `ghcr.io/manifeed/ner_service-base:<tag>`
- `ghcr.io/manifeed/ner_service:<tag>`

The final image is always built from the base image digest produced by the same
workflow run.

## Configuration

- `NER_SERVICE_API_KEY`: bearer token for inference endpoints
- `GLINER_MODEL_ID`: model id downloaded at base-image build time
- `GLINER_MODEL_PATH`: in-container model path
- `NER_DEVICE`: `cuda` or `cpu`
- `NER_THRESHOLD`: GLiNER confidence threshold
- `NER_BATCH_MAX_ITEMS`: explicit batch item cap
- `NER_BATCH_MAX_TOKENS`: explicit batch token cap
- `NER_BATCH_MAX_WAIT_MS`: queue flush delay
- `NER_QUEUE_MAX_ITEMS`: bounded queue size before `503 overloaded`
- `NER_REQUEST_TIMEOUT_SECONDS`: max wait for a request result
- `NER_SHUTDOWN_GRACE_SECONDS`: graceful stop budget
- `NER_GPU_MEMORY_FRACTION`: per-process CUDA memory budget, default `0.25`

If `NER_BATCH_MAX_ITEMS` and `NER_BATCH_MAX_TOKENS` are not set, the service
chooses defaults from detected VRAM:

- `<=8GB`: `4 items / 1024 tokens`
- `<=16GB`: `12 items / 3072 tokens`
- `>16GB`: `24 items / 6144 tokens`

## API Examples

Batch request (single-item batches are supported):

```json
{
  "items": [
    {
      "article_id": 42,
      "title": "Ada Lovelace launches a software startup",
      "summary": "The company focuses on developer tools."
    }
  ]
}
```

Entity labels are fixed: `PERSON`, `ORGANIZATION`, `COMPANY`, `LOCATION`, `DATE`,
`EVENT`, `PRODUCT`, `TECHNOLOGY`, `CURRENCY`.

## Failure Modes

- `503 warming_up`
- `503 overloaded`
- `503 stopping`
- `503 request_timeout`
- `503 ner_model_unavailable`

## Integration Notes

- `indexer_service` now uses `POST /v1/entities/batch` by default.
- `infra` should consume this service strictly through `NER_SERVICE_URL` and
  `NER_SERVICE_API_KEY`.
