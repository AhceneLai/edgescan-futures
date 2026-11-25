# Roadmap EdgeScan

Plan détaillé : [EdgeScan_Plan_de_realisation.docx](EdgeScan_Plan_de_realisation.docx).
Une tâche = un commit (convention `type(module): description`). Cocher au fur et à mesure.

## Phase 0 — Mise en place (≈ 4 h)
- [ ] 0.1 Dépôt GitHub, licence MIT, description et topics — `chore: initial project structure`
- [ ] 0.2 Renommer le package `fds` en `edgescan` — `refactor: rename package to edgescan`
- [ ] 0.3 Outillage : venv, ruff, pre-commit, Makefile — `chore: add dev tooling`
- [ ] 0.4 CI : lint + tests à chaque push, badge README — `ci: run lint and tests on push`
- [ ] 0.5 Fournisseurs de données (IBKR / Databento), clés dans `.env` — `docs: document data providers`

## Phase 1 — Données historiques (≈ 20 h)
- [ ] 1.1 Chargeur yfinance (ES, NQ, CL, GC) — `feat(data): yfinance loader`
- [ ] 1.2 Chargeur pro, barres 1 min + journalières — `feat(data): databento/ibkr historical loader`
- [ ] 1.3 Contrats continus ajustés aux rolls — `feat(data): back-adjusted continuous contracts`
- [ ] 1.4 Calendrier des séances CME — `feat(data): CME session calendar`
- [ ] 1.5 Stockage barres et signaux — `feat(storage): bars and signals schema`
- [ ] 1.6 Contrôles qualité — `feat(data): data quality checks`
- [ ] 1.7 Données synthétiques — `test(data): synthetic data generator`

## Phase 2 — Indicateurs et setups (≈ 12 h)
- [ ] 2.1 Indicateurs journaliers normalisés ATR — `feat(features): daily indicators`
- [ ] 2.2 Test anti-biais d'anticipation — `test(features): no look-ahead`
- [ ] 2.3 Indicateurs intraday — `feat(features): intraday indicators`
- [ ] 2.4 Les 4 setups + justification — `feat(setups): daily setups with rationale`
- [ ] 2.5 Niveaux clés du jour — `feat(setups): key daily levels`

## Phase 3 — Labels et modèle de probabilités (≈ 20 h)
- [ ] 3.1 Labellisation 3 issues avec frais — `feat(dataset): trade labelling with costs`
- [ ] 3.2 Features orientées dans le sens du trade — `feat(dataset): direction-adjusted features`
- [ ] 3.3 Modèle de référence — `feat(model): baseline frequencies`
- [ ] 3.4 LightGBM calibré — `feat(model): calibrated gradient boosting`
- [ ] 3.5 Explications SHAP — `feat(model): SHAP explanations`
- [ ] 3.6 Situations comparables — `feat(model): similar historical trades`

## Phase 4 — Backtest et validation (≈ 15 h)
- [ ] 4.1 Walk-forward — `feat(backtest): walk-forward validation`
- [ ] 4.2 Métriques et calibration — `feat(backtest): metrics and calibration`
- [ ] 4.3 Sélection vs tout prendre — `feat(backtest): selection vs baseline`
- [ ] 4.4 Tests de robustesse — `feat(backtest): robustness tests`
- [ ] 4.5 Rapport de backtest — `feat(backtest): backtest report`

## Phase 5 — Gestion du risque (≈ 8 h)
- [ ] 5.1 Espérance nette — `feat(risk): expected value`
- [ ] 5.2 Taille de position + micros — `feat(risk): position sizing with micros`
- [ ] 5.3 Monte Carlo corrélé — `feat(risk): correlated Monte Carlo`
- [ ] 5.4 Limites de risque — `feat(risk): daily risk limits`

## Phase 6 — Moteur temps réel 24 h/24 (≈ 30 h)
- [ ] 6.1 Flux en direct — `feat(live): market data stream`
- [ ] 6.2 Reconnexion auto + flux gelé — `feat(live): auto-reconnect and stale feed detection`
- [ ] 6.3 Barres 1 min en direct — `feat(live): live 1-min bars`
- [ ] 6.4 Planificateur de séance — `feat(live): session scheduler`
- [ ] 6.5 Déclencheurs intraday — `feat(live): intraday triggers`
- [ ] 6.6 Probabilités mises à jour en séance — `feat(live): intraday probability updates`
- [ ] 6.7 Pauses, week-ends, jours fériés — `feat(live): market hours awareness`
- [ ] 6.8 Rolls automatiques — `feat(live): automatic contract rolls`
- [ ] 6.9 État persistant — `feat(live): persistent state`

## Phase 7 — Alertes et rapports (≈ 12 h)
- [ ] 7.1 Telegram + Discord — `feat(alerts): telegram and discord`
- [ ] 7.2 Priorités et anti-spam — `feat(alerts): priorities and dedup`
- [ ] 7.3 Commandes du bot — `feat(alerts): bot commands`
- [ ] 7.4 Calendrier macro automatique — `feat(macro): economic calendar API`
- [ ] 7.5 Rapports avant-séance et fin de journée — `feat(report): pre-session and end-of-day reports`

## Phase 8 — Infrastructure (≈ 15 h)
- [ ] 8.1 Docker compose — `build: docker compose stack`
- [ ] 8.2 Serveur VPS sécurisé — `docs: server setup guide`
- [ ] 8.3 Redémarrage automatique — `build: restart policies`
- [ ] 8.4 Gestion des secrets — `chore: secrets management`
- [ ] 8.5 Logs structurés — `feat(ops): structured logging`
- [ ] 8.6 Supervision / heartbeat — `feat(ops): heartbeat monitoring`
- [ ] 8.7 Sauvegardes — `feat(ops): database backups`
- [ ] 8.8 Déploiement automatique — `ci: automated deployment`

## Phase 9 — MLOps (≈ 12 h)
- [ ] 9.1 Réentraînement hebdomadaire — `feat(mlops): weekly retraining`
- [ ] 9.2 Registre MLflow — `feat(mlops): MLflow model registry`
- [ ] 9.3 Surveillance de dérive — `feat(mlops): drift monitoring`
- [ ] 9.4 Revue automatique — `feat(mlops): automated review`

## Phase 10 — Tableau de bord (≈ 10 h)
- [ ] 10.1 Dashboard web — `feat(ui): dashboard`

## Phase 11 — Paper trading (1 à 3 mois)
- [ ] 11.1 Exécution fictive — `feat(paper): paper trading journal`
- [ ] 11.2 Bilan — `docs: paper trading results`

## Phase 12 — Documentation et portfolio (≈ 8 h)
- [ ] 12.1 README final — `docs: complete README`
- [ ] 12.2 Résultats réels et limites — `docs: real-data results`
- [ ] 12.3 Démo visuelle — `docs: demo assets`
