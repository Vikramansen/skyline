PYTHON ?= python3
VENV ?= .venv
RUN_DIR ?= runs/latest

.PHONY: setup tonight test serve train train-venv serve-venv dashboard

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e '.[jax,api,dev]'

tonight:
	PYTHONPATH=src $(PYTHON) -m skyline.pipeline --alerts data/demo_alerts.jsonl --catalog data/demo_catalog.jsonl --confirmed data/demo_confirmed.jsonl --output $(RUN_DIR) --budget 20

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

serve:
	PYTHONPATH=src $(PYTHON) -m uvicorn skyline.service:app --reload

train:
	PYTHONPATH=src $(PYTHON) -m skyline.jax_training --training-data data/demo_historical_features.jsonl --output artifacts/jax_ranker.json

train-venv:
	PYTHONPATH=src $(VENV)/bin/python -m skyline.jax_training --training-data data/demo_historical_features.jsonl --output artifacts/jax_ranker.json

serve-venv:
	PYTHONPATH=src $(VENV)/bin/python -m uvicorn skyline.service:app --reload

dashboard:
	SKYLINE_MODEL=artifacts/jax_ranker.json PYTHONPATH=src $(VENV)/bin/python -m uvicorn skyline.service:app --reload --port 8000
