"""Stockage des barres et des signaux : fichiers Parquet, interrogés en SQL avec DuckDB.

Pourquoi Parquet + DuckDB plutôt que PostgreSQL : aucun serveur à installer ni à faire tourner,
des fichiers compacts et typés, et du SQL rapide dessus. Plusieurs programmes peuvent lire en
même temps (moteur temps réel, tableau de bord). Si un serveur devient nécessaire (phase 8),
les mêmes schémas se transposent dans PostgreSQL / TimescaleDB.

Organisation sur disque (racine par défaut : data/store, ignorée par git) :

    bars_1d/ES.parquet              barres journalières, une ligne par séance
    bars_1m/ES/2026-11.parquet      barres 1 minute, un fichier par mois (phase 1.2 / 6.3)
    signals.parquet                 un signal par setup détecté, mis à jour jusqu'à sa clôture

Chaque écriture est un « upsert » : les lignes existantes avec la même clé sont remplacées.
Les fichiers sont écrits dans un fichier temporaire puis renommés : jamais de fichier à moitié écrit.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import OHLCV

TIMEFRAMES = ("1d", "1m")

_PRICE_FIELDS = [pa.field(c, pa.float64(), nullable=(c == "volume")) for c in OHLCV]

# Barres journalières : date de la séance (jour où elle se termine).
DAILY_BARS_SCHEMA = pa.schema([pa.field("symbol", pa.string(), False), pa.field("date", pa.date32(), False), *_PRICE_FIELDS])

# Barres 1 minute : heure d'ouverture de la minute, en UTC.
MINUTE_BARS_SCHEMA = pa.schema(
    [pa.field("symbol", pa.string(), False), pa.field("ts", pa.timestamp("us", tz="UTC"), False), *_PRICE_FIELDS]
)

# Un signal = un setup détecté sur un contrat pour une séance. Les colonnes « résultat » restent
# vides jusqu'à la clôture du trade.
SIGNALS_SCHEMA = pa.schema(
    [
        pa.field("signal_id", pa.string(), False),  # ex. "2026-11-25_ES_trend_pullback_long"
        pa.field("created_at", pa.timestamp("us", tz="UTC"), False),
        pa.field("trade_date", pa.date32(), False),  # séance où le trade peut être pris
        pa.field("symbol", pa.string(), False),
        pa.field("setup", pa.string(), False),
        pa.field("direction", pa.string(), False),  # "long" ou "short"
        pa.field("entry", pa.float64(), False),
        pa.field("stop", pa.float64(), False),
        pa.field("target", pa.float64(), False),
        pa.field("p_target", pa.float64()),  # probabilités calibrées des 3 issues
        pa.field("p_stop", pa.float64()),
        pa.field("p_close", pa.float64()),
        pa.field("expected_r", pa.float64()),  # espérance nette de frais, en R
        pa.field("contracts", pa.int32()),
        pa.field("rationale", pa.string()),  # justification en clair
        pa.field("model_version", pa.string()),
        pa.field("status", pa.string(), False),  # "open", "taken", "skipped", "closed"
        pa.field("outcome", pa.string()),  # "target", "stop", "close"
        pa.field("exit_price", pa.float64()),
        pa.field("pnl_r", pa.float64()),
        pa.field("closed_at", pa.timestamp("us", tz="UTC")),
    ]
)
SIGNAL_STATUSES = {"open", "taken", "skipped", "closed"}


class Store:
    def __init__(self, root: str | Path = "data/store"):
        self.root = Path(root)

    # ------------------------------------------------------------------ barres

    def write_bars(self, symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> int:
        """Ajoute ou remplace des barres (format standard de edgescan.data). Renvoie le nombre de lignes écrites."""
        _check_timeframe(timeframe)
        if df.empty:
            return 0
        if timeframe == "1d":
            table = _to_table(_daily_frame(symbol, df), DAILY_BARS_SCHEMA)
            self._upsert(self._daily_path(symbol), table, key=["date"])
        else:
            frame = _minute_frame(symbol, df)
            for month, part in frame.groupby(frame["ts"].dt.strftime("%Y-%m")):
                self._upsert(self._minute_path(symbol, month), _to_table(part, MINUTE_BARS_SCHEMA), key=["ts"])
        return len(df)

    def read_bars(self, symbol: str, timeframe: str = "1d", start=None, end=None) -> pd.DataFrame:
        """Barres d'un contrat au format standard (index date pour 1d, ts UTC pour 1m), bornes incluses."""
        _check_timeframe(timeframe)
        if timeframe == "1d":
            path = self._daily_path(symbol)
            files = [path] if path.exists() else []
        else:
            files = sorted((self.root / "bars_1m" / symbol).glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=OHLCV, dtype=float)

        df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
        index_col = "date" if timeframe == "1d" else "ts"
        df[index_col] = pd.to_datetime(df[index_col])
        if timeframe == "1d":
            df[index_col] = df[index_col].astype("datetime64[us]")  # même résolution que le reste du projet
        df = df.set_index(index_col).sort_index()[OHLCV]
        if start is not None:
            df = df[df.index >= _bound(start, df.index)]
        if end is not None:
            df = df[df.index <= _bound(end, df.index)]
        return df

    def symbols(self, timeframe: str = "1d") -> list[str]:
        _check_timeframe(timeframe)
        folder = self.root / f"bars_{timeframe}"
        if timeframe == "1d":
            return sorted(p.stem for p in folder.glob("*.parquet"))
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else []

    # ------------------------------------------------------------------ signaux

    def write_signals(self, signals: pd.DataFrame) -> int:
        """Ajoute des signaux ou met à jour ceux qui existent déjà (même signal_id)."""
        if signals.empty:
            return 0
        bad = set(signals["status"]) - SIGNAL_STATUSES
        if bad:
            raise ValueError(f"statut inconnu : {sorted(bad)}")
        if signals["signal_id"].duplicated().any():
            raise ValueError("signal_id en double dans le lot")
        self._upsert(self.root / "signals.parquet", _to_table(signals, SIGNALS_SCHEMA), key=["signal_id"])
        return len(signals)

    def read_signals(self) -> pd.DataFrame:
        path = self.root / "signals.parquet"
        if not path.exists():
            return SIGNALS_SCHEMA.empty_table().to_pandas()
        return pq.read_table(path).to_pandas()

    # ------------------------------------------------------------------ SQL

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Requête SQL DuckDB sur les vues bars_1d, bars_1m et signals (seules celles qui existent)."""
        views = {"bars_1d": "bars_1d/*.parquet", "bars_1m": "bars_1m/*/*.parquet", "signals": "signals.parquet"}
        con = duckdb.connect()
        try:
            for name, pattern in views.items():
                if any(self.root.glob(pattern)):
                    path = (self.root / pattern).as_posix().replace("'", "''")
                    con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
            return con.execute(sql, params or []).df()
        finally:
            con.close()

    # ------------------------------------------------------------------ interne

    def _daily_path(self, symbol: str) -> Path:
        return self.root / "bars_1d" / f"{symbol}.parquet"

    def _minute_path(self, symbol: str, month: str) -> Path:
        return self.root / "bars_1m" / symbol / f"{month}.parquet"

    def _upsert(self, path: Path, table: pa.Table, key: list[str]) -> None:
        if path.exists():
            old = pq.read_table(path).to_pandas()
            new = table.to_pandas()
            merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=key, keep="last")
            table = _to_table(merged.sort_values(key), table.schema)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)  # atomique : un lecteur voit l'ancien fichier ou le nouveau, jamais un mélange


def _check_timeframe(timeframe: str) -> None:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe inconnu : {timeframe} (attendu : {TIMEFRAMES})")


def _daily_frame(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df[OHLCV].astype(float).copy()
    out.insert(0, "date", pd.DatetimeIndex(df.index).date)
    out.insert(0, "symbol", symbol)
    return out.reset_index(drop=True)


def _minute_frame(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("les barres 1 minute doivent avoir un fuseau horaire")
    out = df[OHLCV].astype(float).copy()
    out.insert(0, "ts", idx.tz_convert("UTC"))
    out.insert(0, "symbol", symbol)
    return out.reset_index(drop=True)


def _to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    missing = [f.name for f in schema if f.name not in df.columns and not f.nullable]
    if missing:
        raise ValueError(f"colonnes obligatoires manquantes : {missing}")
    # Colonnes facultatives absentes : colonne vide du bon type (sinon pandas les crée en float).
    present = [n for n in schema.names if n in df.columns]
    table = pa.Table.from_pandas(df[present], schema=pa.schema([schema.field(n) for n in present]), preserve_index=False)
    columns = [table.column(f.name) if f.name in present else pa.nulls(len(df), f.type) for f in schema]
    return pa.Table.from_arrays(columns, schema=schema)


def _bound(value, index: pd.DatetimeIndex) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if index.tz is not None and t.tz is None:
        t = t.tz_localize(index.tz)
    return t
