# Fraud detection MLOps — convenience targets (run `make help`).
PYTHON ?= .venv/bin/python
IMAGE ?= fraud-api
VERSION ?= 1.0.0
MODEL_PATH ?= outputs/model/model.gguf
PORT ?= 8000
# CA bundle for builds behind a TLS-inspecting proxy (optional): make docker-build PIP_CA=/path/ca.crt
PIP_CA ?=
comma := ,
DOCKER_SECRET := $(if $(PIP_CA),--secret id=pip_ca$(comma)src=$(PIP_CA),)

.DEFAULT_GOAL := help
.PHONY: help install data train pipeline-local pipeline-validate pipeline-azure serve smoke-test test lint format docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

.venv/bin/python:
	@python3 -m venv .venv

install: .venv/bin/python ## Create .venv and install all (dev) dependencies
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements/dev.txt
	@$(PYTHON) -m pip install --no-deps -e .

data: ## Hito 1: generate the synthetic fraud dataset (data/transactions.csv)
	@$(PYTHON) src/generate_data.py --output data/transactions.csv

data/transactions.csv:
	@$(MAKE) data

train: data/transactions.csv ## Hito 2: train + quantize, artifacts in outputs/model/
	@$(PYTHON) src/train_and_optimize.py --train_data data/transactions.csv --model_output outputs/model

outputs/model/model.gguf:
	@$(MAKE) train

pipeline-local: ## Hito 3: execute the Azure ML pipeline definition locally
	@$(PYTHON) mlops_pipeline.py --mode local

pipeline-validate: ## Hito 3: offline SDK validation + dry-run submission
	@$(PYTHON) mlops_pipeline.py --mode validate

pipeline-azure: ## Hito 3: submit to Azure ML (requires credentials, fails without them)
	@$(PYTHON) mlops_pipeline.py --mode azure --strict --stream

serve: $(MODEL_PATH) ## Hito 4: run the inference API on 127.0.0.1:$(PORT)
	@FRAUD_API_MODEL_PATH=$(MODEL_PATH) $(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --no-access-log

smoke-test: ## Hito 4: contract + latency (<200 ms) check against a running API
	@$(PYTHON) scripts/smoke_test.py --url http://127.0.0.1:$(PORT)

test: ## Run the test suite
	@$(PYTHON) -m pytest

lint: ## Static checks (ruff lint + format check)
	@$(PYTHON) -m ruff check .
	@$(PYTHON) -m ruff format --check .

format: ## Auto-format the code
	@$(PYTHON) -m ruff check --fix .
	@$(PYTHON) -m ruff format .

docker-build: $(MODEL_PATH) ## Hito 5: build the inference image
	@docker build $(DOCKER_SECRET) --build-arg MODEL_PATH=$(MODEL_PATH) --build-arg VERSION=$(VERSION) -t $(IMAGE):$(VERSION) .

docker-run: ## Hito 5: run the inference image on port $(PORT)
	@docker run --rm -p $(PORT):8000 $(IMAGE):$(VERSION)

clean: ## Remove generated data and artifacts
	@rm -rf data outputs .pytest_cache .ruff_cache
