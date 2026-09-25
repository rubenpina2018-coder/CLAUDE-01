# Orquestación local del proyecto.   Uso típico:  make setup && make all
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

VENV    ?= .venv
PY      := $(VENV)/bin/python
COMPOSE ?= docker compose

.PHONY: help setup up down reset-db init data etl api loadtest test all

help: ## Muestra los objetivos disponibles
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

setup: ## Crea el entorno virtual e instala las dependencias
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

up: ## Levanta PostgreSQL 16 y espera a que esté healthy
	$(COMPOSE) up -d --wait

down: ## Detiene PostgreSQL (conserva los datos)
	$(COMPOSE) down

reset-db: ## Detiene PostgreSQL y borra el volumen de datos
	$(COMPOSE) down -v

init: ## Crea la tabla, sus índices y la vista de resumen
	$(PY) init_db.py

data: ## Genera el dataset sintético (1M filas) en data/
	@mkdir -p results
	$(PY) generate_data.py 2>&1 | tee results/generate_data.log

etl: ## Ejecuta el pipeline ETL (métricas en results/)
	@mkdir -p results
	$(PY) etl_pipeline.py 2>&1 | tee results/etl_run.log

api: ## Levanta la API en primer plano (4 workers)
	$(VENV)/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4 --no-access-log

loadtest: ## Prueba de carga: API en segundo plano + Locust 100 usuarios, 30 s
	BIN=$(VENV)/bin ./scripts/run_load_test.sh

test: ## Tests unitarios (ETL) y de integración (API)
	$(VENV)/bin/pytest -v

all: up init data etl test loadtest ## Flujo completo de punta a punta
