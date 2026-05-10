"""
src/features/preprocessor.py
════════════════════════════
Load dữ liệu, normalize features, tạo observation cho RL agent.

Key design:
  - Raw price (open/high/low/close) KHÔNG normalize → dùng cho chart + PnL tính đúng
  - Features kỹ thuật normalize bằng RobustScaler (median/IQR)
  - Observation = cửa sổ 20 bar × 27 features (normalized) + 4 portfolio state
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

FEAT_COLS = [
    # Trend & Momentum
    "sma_20", "ema_20", "macd_histogram", "rsi_14",
    # Volatility
    "atr_14", "rolling_std_20",
    # Price pattern & Range
    "log_return", "body_size", "distance_from_high_20", "distance_from_low_20",
    # Volume & Market
    "volume_ratio", "vnindex_return",
]


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV và parse date, sort theo thời gian."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_csv(p)

    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.reset_index(drop=True)
    print(f"[Data] {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[Data] Close range: {df['close'].min():.2f} → {df['close'].max():.2f} VND×1000")
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
    """RobustScaler: fit trên train, transform val/test.
    Dùng median/IQR để robust với outliers."""

    def __init__(self):
        self._med: dict[str, float] = {}
        self._iqr: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "RobustScaler":
        cols = [c for c in FEAT_COLS if c in df.columns]
        for c in cols:
            s = df[c].dropna()
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            self._med[c] = float(s.median())
            self._iqr[c] = (q3 - q1) if (q3 - q1) > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self._med:
            if c in df.columns:
                df[c] = ((df[c] - self._med[c]) / self._iqr[c]).clip(-4, 4)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def get_obs(df_norm: pd.DataFrame, step: int, window: int) -> np.ndarray:
    """
    Observation vector tại bước step.

    Shape: (window * n_features,) flattened.
    """
    cols  = [c for c in FEAT_COLS if c in df_norm.columns]
    start = max(0, step - window + 1)
    mat   = df_norm.iloc[start: step + 1][cols].fillna(0.0).values
    if mat.shape[0] < window:
        pad = np.zeros((window - mat.shape[0], mat.shape[1]), dtype=np.float32)
        mat = np.vstack([pad, mat])
    return mat.flatten().astype(np.float32)


def obs_size_of(df_norm: pd.DataFrame, window: int) -> int:
    """Tính obs_size = window × n_features + 4 portfolio state."""
    cols = [c for c in FEAT_COLS if c in df_norm.columns]
    return window * len(cols) + 4   # +4 portfolio state