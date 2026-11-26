"""Point d'entrée : python -m edgescan {download,quality,scan,backtest,review}"""

import argparse
import logging
from pathlib import Path

import yaml

from . import data
from .dataset import build_dataset
from .features import compute_features
from .levels import daily_levels, describe_plan, trade_plan
from .model import BaselineModel, ProbModel, similar_trades, summarize_similar
from .quality import check_bars, to_markdown
from .setups import fr
from .storage import Store


def download(cfg: dict, source: str) -> None:
    """Télécharge (ou met à jour) les barres journalières, les enregistre et affiche un résumé."""
    bars = data.load_all(cfg["contracts"], source, raw_dir=cfg["paths"]["raw_data"])
    store = Store(cfg["paths"]["store"])
    for key, df in bars.items():
        store.write_bars(key, df)
        first, last = df.index[0], df.index[-1]
        print(f"{key:3} {len(df):5} barres  {first:%Y-%m-%d} → {last:%Y-%m-%d}  dernière clôture {df['close'].iloc[-1]:,.2f}")
    print(f"Enregistré dans {store.root}/")


def quality(cfg: dict, source: str) -> None:
    """Contrôle qualité de chaque contrat ; rapport affiché et écrit dans reports/data_quality.md."""
    bars = data.load_all(cfg["contracts"], source, raw_dir=cfg["paths"]["raw_data"])
    reports = [check_bars(df, key) for key, df in bars.items()]
    text = to_markdown(reports)
    out = Path(cfg["paths"]["reports"]) / "data_quality.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Rapport écrit dans {out}")


def scan(cfg: dict, source: str) -> None:
    """Setups détectés à la dernière clôture : justification, plan, probabilités et trades comparables.

    Pas encore d'espérance ni de taille de position (phase 5), ni de validation walk-forward (phase 4) :
    ce sont des situations à étudier, pas des recommandations.
    """
    bars = data.load_all(cfg["contracts"], source, raw_dir=cfg["paths"]["raw_data"])
    trades, tonight = build_dataset(bars, cfg)
    if tonight.empty:
        print("Aucun setup à la dernière clôture.")
        return

    model = ProbModel(cfg["model"]).fit(trades)
    baseline = BaselineModel().fit(trades).predict_proba(tonight)
    probs = model.predict_proba(tonight)
    factors = model.explain(tonight)
    engine = "fréquences historiques (trop peu de trades)" if model.uses_baseline_ else "gradient boosting calibré"
    print(f"Modèle : {engine}, appris sur {model.n_train_} trades passés.")

    for i, sig in tonight.iterrows():
        key, last = sig["symbol"], sig["date"]
        df = bars[key]
        plan = trade_plan(sig, daily_levels(df, compute_features(df)).loc[last])
        p, b = probs.loc[i], baseline.loc[i]
        similar = summarize_similar(similar_trades(trades, sig))

        print(f"\n{key} — {sig['setup']} {sig['direction'].upper()}  (clôture du {last:%Y-%m-%d})")
        print(f"  Pourquoi    : {sig['reason']}")
        print(f"  Logique     : {sig['rationale']}")
        print(f"  Plan        : {describe_plan(plan, sig['direction'])}")
        print(
            f"  Probabilités: objectif {p['p_target']:.0%} · stop {p['p_stop']:.0%} · clôture {p['p_close']:.0%}"
            f"   (fréquence habituelle du setup : objectif {b['p_target']:.0%})"
        )
        if factors[i]:
            print("  Facteurs principaux (valeurs vues du côté du trade : > 0 = dans son sens) :")
        for f in factors[i]:
            arrow = "↑" if f["shap"] > 0 else "↓"
            print(f"    {arrow} {f['label']} = {fr(f['value'], 2)}")
        if similar["n"]:
            print(
                f"  {similar['n']} trades comparables : objectif {similar['freq_target']:.0%}, stop {similar['freq_stop']:.0%}, "
                f"clôture {similar['freq_close']:.0%}, R net moyen {fr(similar['mean_net_r'], 2)}"
            )
    print("\nRappel : probabilités non encore validées en walk-forward (phase 4), pas d'espérance ni de taille (phase 5).")
    print("Outil d'aide à la décision, pas un conseil en investissement.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="edgescan", description="EdgeScan — scanner de futures")
    p.add_argument("command", choices=["download", "quality", "scan", "backtest", "review"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "csv", "synthetic"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if a.command == "download":
        download(cfg, a.source)
    elif a.command == "quality":
        quality(cfg, a.source)
    elif a.command == "scan":
        scan(cfg, a.source)
    else:
        print(f"Commande '{a.command}' pas encore implémentée ({len(cfg['contracts'])} contrats configurés).")
