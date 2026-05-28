"""
src/technical/preprocessor.py
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
    "dist_sma_20", "dist_ema_50", "dist_ema_200", 
    "macd_pct", "macd_signal_pct", "macd_diff_pct", "rsi", "stochastic_k",
    # Volatility
    "atr_pct", "bollinger_width_pct", "dist_bb_upper", "dist_bb_lower", "rolling_std_20",
    # Price pattern & Range
    "price_change_percent", "log_return", "body_size", 
    "distance_from_high_20", "distance_from_low_20",
    # Volume & Market
    "volume_ratio", "vnindex_return", "correlation_20", "beta_20"
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
    
    # Tính toán thêm các features dẫn xuất (tương đối)
    c = df["close"]
    
    if "sma_20" in df.columns: df["dist_sma_20"] = (c - df["sma_20"]) / (df["sma_20"] + 1e-9)
    if "ema_50" in df.columns: df["dist_ema_50"] = (c - df["ema_50"]) / (df["ema_50"] + 1e-9)
    if "ema_200" in df.columns: df["dist_ema_200"] = (c - df["ema_200"]) / (df["ema_200"] + 1e-9)
    if "vwap" in df.columns: df["dist_vwap"] = (c - df["vwap"]) / (df["vwap"] + 1e-9)
    
    if "bollinger_upper" in df.columns: 
        df["dist_bb_upper"] = (df["bollinger_upper"] - c) / (c + 1e-9)
        df["dist_bb_lower"] = (c - df["bollinger_lower"]) / (c + 1e-9)
        df["bollinger_width_pct"] = df["bollinger_width"] / (c + 1e-9)
        
    if "macd" in df.columns:
        df["macd_pct"] = df["macd"] / (c + 1e-9)
        df["macd_signal_pct"] = df["macd_signal"] / (c + 1e-9)
        df["macd_diff_pct"] = df["macd_diff"] / (c + 1e-9)
        
    if "atr" in df.columns:
        df["atr_pct"] = df["atr"] / (c + 1e-9)
        
    if "volume_ma_20" in df.columns:
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-9)
        
    if "vnindex_close" in df.columns:
        df["vnindex_return"] = df["vnindex_close"].pct_change()
        
    df["log_return"] = np.log(c / c.shift(1).replace(0, np.nan))
    df["body_size"] = (c - df["open"]) / (df["open"] + 1e-9)
    
    df["rolling_max_20"] = c.rolling(20).max()
    df["rolling_min_20"] = c.rolling(20).min()
    df["distance_from_high_20"] = (df["rolling_max_20"] - c) / (df["rolling_max_20"] + 1e-9)
    df["distance_from_low_20"]  = (c - df["rolling_min_20"]) / (df["rolling_min_20"] + 1e-9)
    
    df["rolling_std_20"] = df["log_return"].rolling(20).std()
    
    df = df.reset_index(drop=True)
    print(f"[Data] {p.name}: {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[Data] {p.name} Close range: {df['close'].min():.2f} → {df['close'].max():.2f}")
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
    """Tính obs_size = window × n_features + 7 portfolio state.
    
    Portfolio state gồm:
      1. in_position    — Có đang giữ cổ phiếu?
      2. unrealized_pnl — PnL tạm tính (clip -0.3 ~ 0.3)
      3. cash_ratio     — Tỷ lệ tiền mặt / tổng tài sản
      4. held_ratio     — % thời gian giữ so với max_hold
      5. t_plus_avail   — Hàng đã về T+2 chưa? (0/1)
      6. atr_pct        — ATR(14) / close — biến động hiện tại
      7. dist_to_sl     — Khoảng cách % từ giá hiện tại tới mức SL
    """
    cols = [c for c in FEAT_COLS if c in df_norm.columns]
    return window * len(cols) + 7   # +7 portfolio state
