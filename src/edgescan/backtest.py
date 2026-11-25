"""Validation walk-forward : le modèle n'est jamais évalué sur ce qu'il a vu."""

import pandas as pd


def walk_forward(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # TODO: réentraîner tous les N jours sur le passé, prédire le bloc suivant
    raise NotImplementedError
