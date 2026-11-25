# EdgeScan

[![CI](https://github.com/AhceneLai/edgescan-futures/actions/workflows/ci.yml/badge.svg)](https://github.com/AhceneLai/edgescan-futures/actions/workflows/ci.yml)

Scanner quotidien de setups sur futures (ES, NQ, CL, GC) : probabilités calibrées, espérance nette de frais, justification de chaque trade et alertes.

> 🚧 En construction. Outil d'aide à la décision, pas un conseil en investissement.

## Le problème
<!-- TODO -->

## La solution
<!-- TODO -->

## Architecture
<!-- TODO : schéma -->

## Démarrage rapide
Prérequis : [uv](https://docs.astral.sh/uv/) (installe Python 3.12 automatiquement).

```bash
make install   # .venv + dépendances + hook pre-commit
make test      # tests
make lint      # style (ruff)
make data      # télécharge les barres journalières (Yahoo, gratuit)
make run       # scan de démonstration sur données synthétiques
```

## Données
Fournisseurs, coûts et configuration des accès : [docs/DATA_PROVIDERS.md](docs/DATA_PROVIDERS.md).

## Choix méthodologiques
<!-- TODO -->

## Résultats
<!-- TODO -->

## Limites et pistes d'amélioration
<!-- TODO -->
