# Convenience wrappers. Everything here is a plain command you can also run by hand.
.PHONY: help install data train evaluate serve dev test lint frontend synthetic clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## install python and node dependencies
	pip install -r requirements-dev.txt
	cd frontend && npm install

synthetic:  ## generate the synthetic stand-in dataset and its manifest
	python scripts/make_synthetic_dataset.py --output data/raw/synthetic
	python scripts/prepare_data.py --dataset configs/datasets/synthetic.yaml --force

data:  ## download the real dataset and build the manifest
	python scripts/download_dataset.py
	python scripts/prepare_data.py

train:  ## train on the configured dataset
	python scripts/train.py

evaluate:  ## evaluate the current checkpoint on the test split
	python scripts/evaluate.py --save-errors

frontend:  ## build the frontend into frontend/dist
	cd frontend && npm run build

serve: frontend  ## build the frontend and serve everything on one port
	uvicorn api.main:app --reload

dev:  ## API only; run `cd frontend && npm run dev` alongside it
	uvicorn api.main:app --reload

test:  ## run the test suite
	pytest -q

lint:  ## ruff and tsc
	ruff check src api scripts tests
	cd frontend && npm run lint

clean:  ## remove caches and build output
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache frontend/dist
