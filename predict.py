"""
predict.py
==========
Script inference (suy luận) cho mô hình xLSTM đã huấn luyện:
  - Load dữ liệu mới (hoặc dữ liệu test)
  - Tiền xử lý giống pipeline training
  - Dự báo xu hướng cho N phiên tiếp theo
  - Xuất kết quả ra CSV và in ra màn hình
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime

from config import MODEL_DIR, RESULT_DIR, SEQUENCE_LENGTH, TREND_THRESHOLD, MODEL_CONFIG
from xlstm_model import xLSTM

LABEL_MAP  = {0: "GIẢM  ↓", 1: "NGANG →", 2: "TĂNG  ↑"}
LABEL_SHORT = {0: "Bear", 1: "Neutral", 2: "Bull"}


# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_scaler(model_dir: str = MODEL_DIR):
    """Load model, scaler, feature list từ thư mục đã lưu."""
    device = get_device()

    # Feature columns
    with open(os.path.join(model_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Model
    cfg = MODEL_CONFIG.copy()
    cfg["input_size"] = len(feature_cols)
    model = xLSTM.from_config(cfg).to(device)

    model_path = os.path.join(model_dir, "xlstm_vnm_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Predict] Model loaded ← {model_path}")

    # Scaler
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    print(f"[Predict] Scaler loaded")

    return model, scaler, feature_cols, device


# ─────────────────────────────────────────────────────────────────────────────
def prepare_inference_data(df: pd.DataFrame, scaler, feature_cols: list) -> np.ndarray:
    """
    Nhận DataFrame mới (đã có đủ columns),
    chuẩn hóa và tạo chuỗi SEQUENCE_LENGTH phiên cuối.
    Returns: (1, SEQUENCE_LENGTH, n_features) tensor
    """
    # Chọn features
    available = [c for c in feature_cols if c in df.columns]
    df_feat = df[available].copy()

    # Điền NaN
    df_feat = df_feat.ffill().bfill()
    df_feat = df_feat.replace([float("inf"), float("-inf")], float("nan"))
    df_feat = df_feat.ffill().bfill()

    # Chuẩn hóa
    X_scaled = scaler.transform(df_feat.values)

    # Lấy SEQUENCE_LENGTH phiên cuối
    if len(X_scaled) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Cần ít nhất {SEQUENCE_LENGTH} phiên, chỉ có {len(X_scaled)}"
        )
    X_seq = X_scaled[-SEQUENCE_LENGTH:]          # (seq_len, n_feat)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    return X_tensor


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_next_session(df: pd.DataFrame, model, scaler, feature_cols, device):
    """Dự báo xu hướng 1 phiên tiếp theo."""
    X_tensor = prepare_inference_data(df, scaler, feature_cols).to(device)
    logits   = model(X_tensor)                        # (1, 3)
    probs    = torch.softmax(logits, dim=-1)[0]        # (3,)
    pred_cls = probs.argmax().item()

    result = {
        "predicted_label":    pred_cls,
        "predicted_trend":    LABEL_MAP[pred_cls],
        "prob_bear":          round(probs[0].item(), 4),
        "prob_neutral":       round(probs[1].item(), 4),
        "prob_bull":          round(probs[2].item(), 4),
        "confidence":         round(probs[pred_cls].item(), 4),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
def predict_rolling(df: pd.DataFrame, model, scaler, feature_cols, device,
                    start_idx: int = None):
    """
    Dự báo cuốn chiếu (rolling) từ phiên start_idx đến cuối.
    Hữu ích để backtest trên dữ liệu lịch sử.

    Returns: DataFrame với cột predicted_trend, actual_trend, và giá
    """
    if start_idx is None:
        start_idx = SEQUENCE_LENGTH

    results = []
    dates = df["date"].values if "date" in df.columns else np.arange(len(df))

    for i in range(start_idx, len(df)):
        window = df.iloc[i - SEQUENCE_LENGTH : i]
        try:
            X_tensor = prepare_inference_data(window, scaler, feature_cols).to(device)
        except ValueError:
            continue

        logits = model(X_tensor)
        probs  = torch.softmax(logits, dim=-1)[0]
        pred   = probs.argmax().item()

        # Thực tế (nếu có)
        if i + 1 < len(df):
            ret = np.log(df["close"].iloc[i+1] / df["close"].iloc[i])
            actual = 2 if ret > TREND_THRESHOLD else (0 if ret < -TREND_THRESHOLD else 1)
        else:
            actual = np.nan

        results.append({
            "date":           str(dates[i])[:10],
            "close":          df["close"].iloc[i],
            "predicted":      pred,
            "predicted_name": LABEL_SHORT[pred],
            "actual":         actual,
            "actual_name":    LABEL_SHORT[actual] if not np.isnan(actual) else "N/A",
            "prob_bear":      round(probs[0].item(), 4),
            "prob_neutral":   round(probs[1].item(), 4),
            "prob_bull":      round(probs[2].item(), 4),
            "correct":        pred == actual if not np.isnan(actual) else None,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(data_path: str = None):
    """
    Chạy backtest trên toàn bộ dữ liệu và tính toán các chỉ số.
    """
    from config import DATA_PATH
    if data_path is None:
        data_path = DATA_PATH

    print(f"\n{'═'*60}")
    print(" BACKTEST  –  xLSTM VNM Stock Predictor")
    print(f"{'═'*60}")

    # Load
    model, scaler, feature_cols, device = load_model_and_scaler()
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)

    # Điền NaN cho các features
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    # Rolling predict
    print("\n[Predict] Đang chạy rolling prediction...")
    results_df = predict_rolling(df, model, scaler, feature_cols, device)

    # Tính accuracy
    valid = results_df.dropna(subset=["actual"])
    acc   = (valid["predicted"] == valid["actual"]).mean()
    n     = len(valid)

    # Accuracy theo từng class
    for cls, name in [(0, "Giảm"), (1, "Đi ngang"), (2, "Tăng")]:
        mask     = valid["actual"] == cls
        cls_acc  = (valid[mask]["predicted"] == cls).mean() if mask.sum() > 0 else 0.0
        print(f"  {name:12s}: {mask.sum():4d} mẫu  |  acc = {cls_acc:.4f}")

    print(f"\n  TỔNG CỘNG : {n:4d} mẫu  |  Overall acc = {acc:.4f}")

    # Lưu kết quả
    out_path = os.path.join(RESULT_DIR, "backtest_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n[Predict] Kết quả backtest → {out_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
def predict_latest(data_path: str = None):
    """Dự báo phiên giao dịch tiếp theo (LATEST)."""
    from config import DATA_PATH
    if data_path is None:
        data_path = DATA_PATH

    model, scaler, feature_cols, device = load_model_and_scaler()
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)

    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    result = predict_next_session(df, model, scaler, feature_cols, device)

    last_date  = df["date"].iloc[-1].strftime("%d/%m/%Y")
    last_close = df["close"].iloc[-1]

    print(f"\n{'═'*60}")
    print(f" DỰ BÁO PHIÊN TIẾP THEO")
    print(f"{'═'*60}")
    print(f"  Phiên cuối đã có: {last_date}  |  Giá đóng cửa: {last_close:.2f}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Xu hướng dự báo : {result['predicted_trend']}")
    print(f"  Độ tin cậy      : {result['confidence']*100:.1f}%")
    print(f"  Xác suất Tăng   : {result['prob_bull']*100:.1f}%")
    print(f"  Xác suất Ngang  : {result['prob_neutral']*100:.1f}%")
    print(f"  Xác suất Giảm   : {result['prob_bear']*100:.1f}%")
    print(f"{'═'*60}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtest()
    else:
        predict_latest()
