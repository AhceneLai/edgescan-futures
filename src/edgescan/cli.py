"""Point d'entrée : python -m edgescan {download,scan,backtest,review}"""

import argparse
import logging

import yaml

from . import data


def download(cfg: dict, source: str) -> None:
    """Télécharge (ou met à jour) les barres journalières et affiche un résumé par contrat."""
    bars = data.load_all(cfg["contracts"], source, raw_dir=cfg["paths"]["raw_data"])
    for key, df in bars.items():
        first, last = df.index[0], df.index[-1]
        print(f"{key:3} {len(df):5} barres  {first:%Y-%m-%d} → {last:%Y-%m-%d}  dernière clôture {df['close'].iloc[-1]:,.2f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="edgescan", description="EdgeScan — scanner de futures")
    p.add_argument("command", choices=["download", "scan", "backtest", "review"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "csv", "synthetic"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if a.command == "download":
        download(cfg, a.source)
    else:
        print(f"Commande '{a.command}' pas encore implémentée ({len(cfg['contracts'])} contrats configurés).")
