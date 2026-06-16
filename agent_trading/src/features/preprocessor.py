"""
src/features/preprocessor.py

Load data, normalize dataset-provided features, and build RL observations.

Design:
  - Raw OHLCV columns stay in df_raw for execution and PnL.
  - The agent observes only feature columns already present in the CSV.
  - Normalization is fit on train data only via RobustScaler.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEAT_COLS = [
    "price_change_percent",
    "volume_ma_20",
    "rsi",
    "stochastic_k",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema_50",
    "ema_200",
    "sma_20",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_width",
    "atr",
    "obv",
    "vwap",
    "vnindex_close",
    "correlation_20",
    "beta_20",
]

PORTFOLIO_STATE_SIZE = 7


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV, parse date, sort by time, and keep dataset columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    print(f"[Data] {p.name}: {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"[Data] {p.name} Close range: {df['close'].min():.2f} -> {df['close'].max():.2f}")
    return df


def time_split(df: pd.DataFrame, train_r=0.70, val_r=0.15):
    """Time-based split: train / val / test."""
    n = len(df)
    i1 = int(n * train_r)
    i2 = int(n * (train_r + val_r))
    tr = df.iloc[:i1].copy().reset_index(drop=True)
    va = df.iloc[i1:i2].copy().reset_index(drop=True)
    te = df.iloc[i2:].copy().reset_index(drop=True)
    print(f"[Split] Train={len(tr)} | Val={len(va)} | Test={len(te)}")
    return tr, va, te


class RobustScaler:
    """Median/IQR scaler for feature columns available in the dataset."""

    def __init__(self):
        self._med: dict[str, float] = {}
        self._iqr: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "RobustScaler":
        cols = [c for c in FEAT_COLS if c in df.columns]
        for c in cols:
            s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                self._med[c] = 0.0
                self._iqr[c] = 1.0
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            self._med[c] = float(s.median())
            self._iqr[c] = (q3 - q1) if (q3 - q1) > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self._med:
            if c in df.columns:
                s = df[c].replace([np.inf, -np.inf], np.nan)
                df[c] = ((s - self._med[c]) / self._iqr[c]).clip(-4, 4)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def get_obs(df_norm: pd.DataFrame, step: int, window: int) -> np.ndarray:
    """Return flattened feature window at step."""
    cols = [c for c in FEAT_COLS if c in df_norm.columns]
    start = max(0, step - window + 1)
    mat = df_norm.iloc[start: step + 1][cols].fillna(0.0).values
    if mat.shape[0] < window:
        pad = np.zeros((window - mat.shape[0], mat.shape[1]), dtype=np.float32)
        mat = np.vstack([pad, mat])
    return mat.flatten().astype(np.float32)


def obs_size_of(df_norm: pd.DataFrame, window: int, n_tickers: int = 4) -> int:
    """Observation size = window * dataset features + portfolio state + ticker id."""
    cols = [c for c in FEAT_COLS if c in df_norm.columns]
    return window * len(cols) + PORTFOLIO_STATE_SIZE + n_tickers
