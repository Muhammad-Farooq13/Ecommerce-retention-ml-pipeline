PYTHON ?= python

.PHONY: setup pipeline train evaluate test api

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

pipeline:
	$(PYTHON) scripts/run_pipeline.py

train:
	$(PYTHON) -m src.models.train

evaluate:
	$(PYTHON) -m src.models.evaluate

test:
	pytest -q

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
