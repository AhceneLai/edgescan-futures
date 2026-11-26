"""Point d'entrée : python -m edgescan {download,quality,scan,backtest,review}"""

import argparse
import logging
from pathlib import Path

import yaml

from . import data
from .features import compute_features
from .levels import daily_levels, describe_plan, trade_plan
from .quality import check_bars, to_markdown
from .setups import detect_setups
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
    """Setups détectés à la dernière clôture, avec leur justification et leur plan de trade.

    Pas encore de probabilités (phase 3) : ce ne sont que des situations à étudier, pas des recommandations.
    """
    bars = data.load_all(cfg["contracts"], source, raw_dir=cfg["paths"]["raw_data"])
    found = 0
    for key, df in bars.items():
        feat = compute_features(df)
        last = df.index[-1]
        signals = detect_setups(feat.loc[[last]], cfg["setups"], symbol=key)
        levels = daily_levels(df, feat).loc[last]
        for _, sig in signals.iterrows():
            found += 1
            plan = trade_plan(sig, levels)
            print(f"\n{key} — {sig['setup']} {sig['direction'].upper()}  (clôture du {last:%Y-%m-%d})")
            print(f"  Pourquoi : {sig['reason']}")
            print(f"  Logique  : {sig['rationale']}")
            print(f"  Plan     : {describe_plan(plan, sig['direction'])}")
    if not found:
        print("Aucun setup à la dernière clôture.")
    print("\nRappel : probabilités et taille de position pas encore calculées (phases 3 et 5). Pas un conseil.")


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
