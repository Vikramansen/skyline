PYTHON ?= python3
VENV ?= .venv
RUN_DIR ?= runs/latest

.PHONY: setup tonight test serve train train-venv serve-venv dashboard real-data real-object real-tonight

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e '.[jax,api,dev,broker]'

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

real-data:
	@echo "Set MJD_START and MJD_END, e.g. make real-data MJD_START=60200 MJD_END=60201"
	PYTHONPATH=src $(VENV)/bin/python -m skyline.alerce_ingest --first-mjd $(MJD_START) --last-mjd $(MJD_END) --output data/real/latest

real-object:
	@echo "Set OID, e.g. make real-object OID=ZTF18abbuksn MIN_DRB=0"
	PYTHONPATH=src $(VENV)/bin/python -m skyline.alerce_ingest --oids $(OID) --min-drb $(or $(MIN_DRB),0.5) --output data/real/object-$(OID)

real-tonight:
	PYTHONPATH=src $(VENV)/bin/python -m skyline.pipeline --alerts data/real/latest/alerts.jsonl --catalog data/real/latest/catalog.jsonl --model artifacts/jax_ranker.json --output runs/real/latest --budget 20
