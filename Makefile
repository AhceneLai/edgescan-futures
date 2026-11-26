# Raccourcis de développement. Prérequis : uv (https://docs.astral.sh/uv/)
VENV := .venv
BIN  := $(VENV)/bin

.PHONY: help install test lint format run data quality clean

help:  ## Affiche cette aide
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

install:  ## Crée le .venv (Python 3.12), installe le projet et le hook pre-commit
	test -d $(VENV) || uv venv --python 3.12 $(VENV)
	uv pip install --python $(BIN)/python -e ".[dev]"
	$(BIN)/pre-commit install

test:  ## Lance les tests
	$(BIN)/pytest -q

lint:  ## Vérifie le style sans rien modifier (comme la CI)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Corrige le style automatiquement
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

run:  ## Scan de démonstration sur données synthétiques, sans envoyer d'alerte
	$(BIN)/edgescan scan --source synthetic --dry-run

data:  ## Télécharge / met à jour les barres journalières Yahoo et les enregistre dans data/store/
	$(BIN)/edgescan download

quality:  ## Contrôle qualité des données -> reports/data_quality.md
	$(BIN)/edgescan quality

clean:  ## Supprime les caches
	rm -rf .pytest_cache .ruff_cache build src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
