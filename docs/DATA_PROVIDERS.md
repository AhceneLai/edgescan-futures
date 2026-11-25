# Fournisseurs de données

EdgeScan a besoin de deux choses : un **historique** propre (pour construire et valider le modèle)
et, à partir de la phase 6, un **flux temps réel** (pour surveiller les marchés 24 h/24).
Ce document explique quel fournisseur sert à quoi, quand l'activer et où mettre les accès.

> Prix relevés en novembre 2026, hors taxes. Ils changent souvent : vérifie-les avant de t'abonner.

## Vue d'ensemble

| Fournisseur | Rôle | Phases | Coût mensuel | Accès |
|---|---|---|---|---|
| **yfinance** | Historique journalier (prototype) | 1 à 5 | 0 $ | aucun |
| **Interactive Brokers** | Temps réel + paper trading | 6, 11 | 10 $ (offert dès 30 $ de commissions) | IB Gateway local |
| **Databento** | Historique 1 min / tick + temps réel pro | 1.2, 6 (niveau 3) | 199 $ (CME Standard) | clé API |
| **Financial Modeling Prep** | Calendrier macro (CPI, FOMC, NFP…) | 7.4 | 0 à 29 $ | clé API |

## yfinance — développement (niveau 1, gratuit)

- Contrats continus Yahoo : `ES=F`, `NQ=F`, `CL=F`, `GC=F` (voir `config.yaml`).
- **Limites** : données différées, barres journalières seulement sur un long historique,
  rolls non maîtrisés (sauts de prix possibles à l'échéance), aucune garantie de service.
- Suffisant pour les phases 1 à 5 : construire, tester et backtester sur données journalières.
- Installation : `uv pip install -e ".[live]"`.

## Interactive Brokers — 24 h/24 économique (niveau 2)

- **Compte** : compte IBKR avec au moins 500 $ de dépôt.
- **Abonnement** : bundle *US Securities Snapshot and Futures Value* (10 $/mois) —
  temps réel CME, CBOT, NYMEX, COMEX.
- **Connexion** : pas de clé API. Le programme parle à **IB Gateway** (ou TWS) qui tourne
  sur la même machine, via la bibliothèque `ib_async`.
  - Port 4002 = compte paper, port 4001 = compte réel. **Toujours commencer par le paper.**
  - IB Gateway se déconnecte une fois par jour : redémarrage planifié prévu en phase 8.3.
- **Limites** : pas de profondeur de marché dans ce bundle, historique intraday limité
  (il faut le construire soi-même à partir du flux).

## Databento — version « parfaite » (niveau 3)

- **Offre** : CME Standard, 199 $/mois. Dataset `GLBX.MDP3` (CME Globex).
- Historique propre (tick, 1 min, journalier) et flux temps réel par la même API.
- Symboles continus gérés par Databento (ex. `ES.v.0` = contrat le plus traité en volume),
  ce qui simplifie la tâche 1.3.
- À l'inscription, vérifier les **frais de licence CME** pour un usage personnel (non professionnel).
- À n'activer **que si le paper trading confirme un avantage** (phase 11).

## Financial Modeling Prep — calendrier macro

- Offre gratuite limitée, ou Starter à 29 $/mois.
- Remplacera le fichier manuel `data/macro_calendar.csv` en phase 7.4.

## Où mettre les accès

1. Copier `.env.example` en `.env` à la racine du projet.
2. Remplir uniquement les variables des services utilisés.
3. `.env` est exclu par `.gitignore` : **il ne doit jamais être commité**.
   Le hook pre-commit bloque aussi les clés privées détectées dans les fichiers.
4. Sur le serveur (phase 8.4), les mêmes variables seront fournies par les secrets GitHub /
   l'environnement du conteneur.

## Recommandation

1. **Mois 1 à 3** : yfinance uniquement. Aucun abonnement.
2. **Après la phase 5**, si le backtest montre un avantage : ouvrir le compte IBKR (paper),
   prendre le bundle à 10 $, lancer le moteur temps réel.
3. **Databento** seulement si le paper trading valide la calibration des probabilités.

Sources : voir la section « Sources des prix » de [EdgeScan_Plan_de_realisation.docx](EdgeScan_Plan_de_realisation.docx).
