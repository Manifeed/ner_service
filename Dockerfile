# syntax=docker/dockerfile:1.7

ARG PYTORCH_RUNTIME_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_RUNTIME_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GLINER_MODEL_ID=urchade/gliner_multi-v2.1 \
    GLINER_MODEL_PATH=/opt/models/gliner \
    NER_DEVICE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/tmp/huggingface

RUN useradd --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
from gliner import GLiNER

model_id = os.environ["GLINER_MODEL_ID"]
model_path = os.environ["GLINER_MODEL_PATH"]

snapshot_download(repo_id=model_id, local_dir=model_path)
GLiNER.from_pretrained(model_path)
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=appuser:appuser app /app/app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/internal/ready').read()"

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
