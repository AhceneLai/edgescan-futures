"""Setups journaliers définis à l'avance. Signal à la clôture de J, trade à J+1."""
import pandas as pd

SETUP_RATIONALE = {
    "trend_pullback": "TODO: expliquer la logique de marché",
    "donchian_breakout": "TODO",
    "nr7_breakout": "TODO",
    "bollinger_reversion": "TODO",
}


def detect_setups(feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    # TODO: retourner une ligne par (date, setup, direction)
    raise NotImplementedError
