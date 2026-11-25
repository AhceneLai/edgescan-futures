"""Point d'entrée : python -m fds {scan,backtest,review}"""
import argparse

import yaml


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="fds", description="Futures Daily Scanner")
    p.add_argument("command", choices=["scan", "backtest", "review"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "csv", "synthetic"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Commande '{a.command}' pas encore implémentée ({len(cfg['contracts'])} contrats configurés).")
