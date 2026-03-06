"""
Module xử lý dữ liệu:
  - Đọc & làm sạch CSV
  - Chuẩn hóa features (StandardScaler)
  - Tạo chuỗi thời gian (sliding window)
  - Tạo nhãn xu hướng 3 lớp
  - Chia tập train / val / test theo thứ tự thời gian
  - PyTorch Dataset & DataLoader
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from config import (
    DATA_PATH, MODEL_DIR, FEATURE_COLUMNS, TARGET_COLUMN,
    SEQUENCE_LENGTH, FORECAST_HORIZON, TREND_THRESHOLD,
    TRAIN_RATIO, VAL_RATIO, TRAIN_CONFIG
)


# ─────────────────────────────────────────────────────────────────────────────
class VNMDataProcessor:
    """Xử lý toàn bộ pipeline dữ liệu cho VNM."""

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.scaler    = StandardScaler()
        self.feature_cols: list[str] = []

    # Load vaf clean ───────────────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        df = df.sort_values("date").reset_index(drop=True)
        print(f"[DataLoader] Đã đọc {len(df)} dòng  |  "
              f"{df['date'].min().date()} → {df['date'].max().date()}")
        return df

    # Feature engineering ────────────────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # chọn các cột có trong FEATURE_COLUMNS và tồn tại trong df
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing   = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            print(f"[DataLoader] Thiếu columns: {missing}  → bỏ qua")

        # add log_return nếu chưa có
        if "log_return" not in available and "log_return" in df.columns:
            available.append("log_return")

        # fill none: forward-fill rồi backward-fill
        df[available] = df[available].ffill().bfill()

        # xử lý inf
        df[available] = df[available].replace([np.inf, -np.inf], np.nan)
        df[available] = df[available].ffill().bfill()

        self.feature_cols = available
        print(f"[DataLoader] Sử dụng {len(available)} features: {available}")
        return df

    # 3. Tạo nhãn xu hướng ─────────────────────────────────────────────────
    @staticmethod
    def create_trend_labels(df: pd.DataFrame) -> pd.Series:
        """
        Tính log-return 1 phiên tiếp theo và gán nhãn:
          0 = Giảm  (< -THRESHOLD)
          1 = Đi ngang
          2 = Tăng  (> +THRESHOLD)
        """
        future_return = np.log(
            df[TARGET_COLUMN].shift(-FORECAST_HORIZON) / df[TARGET_COLUMN]
        )
        labels = np.where(
            future_return >  TREND_THRESHOLD, 2,      # tăng
            np.where(
                future_return < -TREND_THRESHOLD, 0,  # giảm
                1                                     # ngang
            )
        )
        return pd.Series(labels, name="trend_label")

    # 4. Normalize ─────────────────────────────────────────────────────────
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def save_scaler(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(self.scaler, path)
        print(f"[DataLoader] Đã lưu scaler → {path}")

    def load_scaler(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "scaler.pkl")
        self.scaler = joblib.load(path)
        print(f"[DataLoader] Đã load scaler ← {path}")

    # 5. Tạo chuỗi thời gian (sliding window) ───────────────────────────────
    @staticmethod
    def create_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = SEQUENCE_LENGTH
    ):
        """
        X: (T, n_features)   y: (T,)
        Returns:
          X_seq: (N, seq_len, n_features)
          y_seq: (N,)
        """
        xs, ys = [], []
        for i in range(len(X) - seq_len):
            xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    # 6. Pipeline chính ────────────────────────────────────────────────────
    def prepare(self):
        """
        Trả về:
          (X_train, y_train), (X_val, y_val), (X_test, y_test)
          feature_names, close_prices_test
        """
        df     = self.load_data()
        df     = self.engineer_features(df)
        labels = self.create_trend_labels(df)

        # Ghép nhãn 
        df["trend_label"] = labels
        df = df.dropna(subset=["trend_label"] + self.feature_cols).reset_index(drop=True)

        # train/val/test theo thời gian ────────────────────
        n       = len(df)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        n_test  = n - n_train - n_val

        df_train = df.iloc[:n_train]
        df_val   = df.iloc[n_train : n_train + n_val]
        df_test  = df.iloc[n_train + n_val :]

        print(f"[DataLoader] Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")

        # Chuẩn hóa: fit trên train, transform trên val/test ───────────────
        X_train_raw = df_train[self.feature_cols].values
        X_val_raw   = df_val[self.feature_cols].values
        X_test_raw  = df_test[self.feature_cols].values

        X_train_sc = self.fit_transform(X_train_raw)
        X_val_sc   = self.transform(X_val_raw)
        X_test_sc  = self.transform(X_test_raw)

        y_train = df_train["trend_label"].values
        y_val   = df_val["trend_label"].values
        y_test  = df_test["trend_label"].values

        # Tạo chuỗi ─────────────────────────────────────────────────────────
        X_tr, y_tr = self.create_sequences(X_train_sc, y_train)
        X_vl, y_vl = self.create_sequences(X_val_sc,   y_val)
        X_te, y_te = self.create_sequences(X_test_sc,  y_test)

        print(f"[DataLoader] Shapes — Train: {X_tr.shape}  Val: {X_vl.shape}  Test: {X_te.shape}")

        # gias close test 
        close_test = df_test[TARGET_COLUMN].values[SEQUENCE_LENGTH:]

        # Thống kê phân phối nhãn
        for split, y_arr in [("Train", y_tr), ("Val", y_vl), ("Test", y_te)]:
            vals, cnts = np.unique(y_arr, return_counts=True)
            dist = {int(v): int(c) for v, c in zip(vals, cnts)}
            print(f"[DataLoader] {split} labels: {dist}")

        return (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te), self.feature_cols, close_test


# ─────────────────────────────────────────────────────────────────────────────
class StockDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    train_data, val_data, test_data,
    batch_size: int = TRAIN_CONFIG["batch_size"]
) -> tuple:
    X_tr, y_tr = train_data
    X_vl, y_vl = val_data
    X_te, y_te = test_data

    # Trọng số class, xử lý imbalance
    class_counts = np.bincount(y_tr)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_tr]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        StockDataset(X_tr, y_tr),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        StockDataset(X_vl, y_vl),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        StockDataset(X_te, y_te),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
