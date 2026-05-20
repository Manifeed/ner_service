COMPOSE := $(shell if docker compose version >/dev/null 2>&1; then echo "docker compose"; elif docker-compose version >/dev/null 2>&1; then echo "docker-compose"; else echo "docker compose"; fi)
DC := $(COMPOSE) -f docker-compose.yml
RELEASE_DC := $(COMPOSE) -f docker-compose.release.yml
PYTORCH_TEST_IMAGE ?= pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
PYTEST_ARGS ?= tests -vv --color=yes --tb=short -ra
BASE_IMAGE_LOCAL ?= manifeed_ner_service-base:local
LOAD_DOTENV := set -a; if [ -f .env ]; then . ./.env; fi; set +a;
GPU_PRECHECK := if [ "$${NER_SERVICE_SKIP_GPU_CHECK:-0}" != "1" ]; then \
		if ! command -v nvidia-smi >/dev/null 2>&1; then \
			printf '%s\n' 'nvidia-smi is not available on the host. Install the NVIDIA driver/toolkit or rerun with NER_SERVICE_SKIP_GPU_CHECK=1 if you know the container runtime is already configured.'; \
			exit 2; \
		fi; \
		if ! nvidia-smi >/dev/null 2>&1; then \
			printf '%s\n' 'The NVIDIA driver is not loaded on the host. Start/fix the NVIDIA driver before running make up.'; \
			exit 2; \
		fi; \
	fi
CANONICALIZED_ENV := export NER_SERVICE_API_KEY="$${NER_SERVICE_API_KEY:-}"; \
	export NER_SERVICE_PORT="$${NER_SERVICE_PORT:-8002}"; \
	export NER_SERVICE_CONTAINER_NAME="$${NER_SERVICE_CONTAINER_NAME:-ner_service}"; \
	export NER_SERVICE_BASE_IMAGE="$${NER_SERVICE_BASE_IMAGE:-$(BASE_IMAGE_LOCAL)}"; \
	export NER_SERVICE_IMAGE_TAG="$${NER_SERVICE_IMAGE_TAG:-latest}"; \
	export GLINER_MODEL_ID="$${GLINER_MODEL_ID:-urchade/gliner_multi-v2.1}"; \
	export GLINER_MODEL_PATH="$${GLINER_MODEL_PATH:-/opt/models/gliner}"; \
	export NER_DEVICE="$${NER_DEVICE:-cuda}"; \
	export NER_THRESHOLD="$${NER_THRESHOLD:-0.5}"; \
	export NER_BATCH_MAX_ITEMS="$${NER_BATCH_MAX_ITEMS:-}"; \
	export NER_BATCH_MAX_TOKENS="$${NER_BATCH_MAX_TOKENS:-}"; \
	export NER_BATCH_MAX_WAIT_MS="$${NER_BATCH_MAX_WAIT_MS:-5}"; \
	export NER_QUEUE_MAX_ITEMS="$${NER_QUEUE_MAX_ITEMS:-1024}"; \
	export NER_REQUEST_TIMEOUT_SECONDS="$${NER_REQUEST_TIMEOUT_SECONDS:-300}"; \
	export NER_SHUTDOWN_GRACE_SECONDS="$${NER_SHUTDOWN_GRACE_SECONDS:-30}"; \
	export NER_GPU_MEMORY_FRACTION="$${NER_GPU_MEMORY_FRACTION:-0.25}";

.DEFAULT_GOAL := help

.PHONY: help build-base build up down clean test release-pull release-up

help:
	@printf '%s\n' 'Available targets:'
	@printf '%s\n' '  make build-base'
	@printf '%s\n' '  make build'
	@printf '%s\n' '  make up'
	@printf '%s\n' '  make down'
	@printf '%s\n' '  make clean'
	@printf '%s\n' '  make test'
	@printf '%s\n' '  make release-pull'
	@printf '%s\n' '  make release-up'

build-base:
	@$(LOAD_DOTENV) docker build -f Dockerfile.base -t "$${NER_SERVICE_BASE_IMAGE:-$(BASE_IMAGE_LOCAL)}" .

build:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(MAKE) build-base
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) build

up:
	@$(LOAD_DOTENV) $(GPU_PRECHECK); $(CANONICALIZED_ENV) $(MAKE) build-base
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) up --build -d

down:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) down

clean:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) down --remove-orphans --rmi local -v

test:
	docker run --rm \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		$(PYTORCH_TEST_IMAGE) \
		bash -lc "python3 -m pip install --disable-pip-version-check -r requirements.txt -r requirements-dev.txt && python3 -m pytest $(PYTEST_ARGS)"

release-pull:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(RELEASE_DC) pull

release-up:
	@$(LOAD_DOTENV) $(GPU_PRECHECK); $(CANONICALIZED_ENV) $(RELEASE_DC) up -d
