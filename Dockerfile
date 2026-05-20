# syntax=docker/dockerfile:1.7

ARG SERVICE_BASE_IMAGE=manifeed_ner_service-base:local

FROM ${SERVICE_BASE_IMAGE}

WORKDIR /app

ENV NER_BATCH_MAX_WAIT_MS=5 \
    NER_QUEUE_MAX_ITEMS=1024 \
    NER_REQUEST_TIMEOUT_SECONDS=300 \
    NER_SHUTDOWN_GRACE_SECONDS=30 \
    NER_GPU_MEMORY_FRACTION=0.25

COPY --chown=appuser:appuser app /app/app

RUN find /app -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && find /app -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/internal/ready').read()"

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
