"""
evaluate.py
===========
Đánh giá mô hình xLSTM trên tập test:
  - Classification report (precision, recall, F1)
  - Confusion matrix
  - Vẽ đường giá kèm dự báo xu hướng
  - Vẽ learning curves
  - Phân tích lỗi
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score
)

from config import MODEL_DIR, LOG_DIR, RESULT_DIR, LOG_CONFIG, TRAIN_CONFIG, MODEL_CONFIG
from data_loader import VNMDataProcessor, StockDataset
from xlstm_model import xLSTM
from torch.utils.data import DataLoader

LABEL_NAMES = ["Giảm (↓)", "Đi ngang (→)", "Tăng (↑)"]
COLORS      = ["#e6301b", "#95a5a6", "#00f767"]


# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device):
    """Lấy toàn bộ predictions & probabilities từ loader."""
    model.eval()
    all_logits, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()
    labels = labels.numpy()
    return preds, probs, labels


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_dir):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix – VNM xLSTM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves(history_path: str, save_dir: str):
    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss", color="#3498db")
    ax.plot(epochs, hist["val_loss"],   label="Val Loss",   color="#e74c3c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, hist["train_acc"], label="Train Acc", color="#3498db")
    ax.plot(epochs, hist["val_acc"],   label="Val Acc",   color="#e74c3c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("xLSTM – VNM Stock Trend Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] Learning curves → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_predictions_on_price(close_prices, labels, preds, save_dir):
    """Vẽ đường giá kèm dự báo đúng/sai theo màu sắc."""
    T      = min(len(close_prices), len(labels))
    prices = close_prices[:T]
    labs   = labels[:T]
    prs    = preds[:T]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # ── Panel 1: Giá đóng cửa ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(prices, color="#2c3e50", linewidth=0.8, label="Giá đóng cửa VNM")
    ax.set_ylabel("Giá (VND × 1000)")
    ax.set_title("Giá đóng cửa VNM – Tập Test", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Dự báo xu hướng ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(prices, color="#bdc3c7", linewidth=0.6, zorder=1)

    for t in range(T):
        if prs[t] == labs[t]:          # Dự báo đúng
            color = COLORS[prs[t]]
        else:                           # Dự báo sai
            color = "black"
        ax.axvline(x=t, color=color, alpha=0.35, linewidth=1.5, zorder=2)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS[2],  label="Tăng – đúng"),
        mpatches.Patch(color=COLORS[1],  label="Đi ngang – đúng"),
        mpatches.Patch(color=COLORS[0],  label="Giảm – đúng"),
        mpatches.Patch(color="black",    label="Sai"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_xlabel("Phiên giao dịch (tập test)")
    ax.set_ylabel("Giá (VND × 1000)")
    ax.set_title("Dự báo xu hướng vs. Thực tế", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "predictions_on_price.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] Price prediction chart → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_probability_distribution(probs, labels, save_dir):
    """Violin plot phân phối xác suất của từng nhãn."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        ax     = axes[cls_idx]
        mask_t = labels == cls_idx     # thực tế là cls_idx
        mask_f = labels != cls_idx

        p_t = probs[mask_t, cls_idx] if mask_t.sum() > 0 else np.array([0])
        p_f = probs[mask_f, cls_idx] if mask_f.sum() > 0 else np.array([0])

        ax.boxplot([p_t, p_f], labels=["Đúng nhãn", "Sai nhãn"],
                   patch_artist=True,
                   boxprops=dict(facecolor=COLORS[cls_idx], alpha=0.6))
        ax.set_title(f"{cls_name}", fontweight="bold")
        ax.set_ylabel("Xác suất dự báo")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phân phối xác suất dự báo – xLSTM", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "probability_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] Probability distribution → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def full_evaluation(model=None, test_data=None, close_test=None):
    """
    Chạy toàn bộ pipeline đánh giá.
    Có thể gọi độc lập (load model từ disk) hoặc truyền model/data trực tiếp.
    """
    device = get_device()

    # ── Load model nếu chưa có ────────────────────────────────────────────────
    if model is None:
        feature_col_path = os.path.join(MODEL_DIR, "feature_cols.json")
        with open(feature_col_path) as f:
            feature_cols = json.load(f)

        cfg = MODEL_CONFIG.copy()
        cfg["input_size"] = len(feature_cols)
        model = xLSTM.from_config(cfg).to(device)
        model_path = os.path.join(MODEL_DIR, LOG_CONFIG["model_name"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Eval] Model loaded ← {model_path}")

    # ── Load test data nếu chưa có ────────────────────────────────────────────
    if test_data is None:
        processor = VNMDataProcessor()
        processor.load_scaler()
        _, _, (X_te, y_te), _, close_test = processor.prepare()
        test_data = (X_te, y_te)

    X_te, y_te = test_data
    test_loader = DataLoader(
        StockDataset(X_te, y_te),
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
    )

    # ── Lấy predictions ───────────────────────────────────────────────────────
    preds, probs, labels = get_predictions(model, test_loader, device)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(labels, preds, target_names=LABEL_NAMES)
    print("\n" + "═"*60)
    print(" CLASSIFICATION REPORT")
    print("═"*60)
    print(report)

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        print(f" Macro ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f" AUC: N/A ({e})")

    # ── Lưu report ────────────────────────────────────────────────────────────
    report_path = os.path.join(RESULT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Eval] Report → {report_path}")

    # ── Vẽ biểu đồ ───────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, RESULT_DIR)

    history_path = os.path.join(LOG_DIR, LOG_CONFIG["history_name"])
    if os.path.exists(history_path):
        plot_learning_curves(history_path, RESULT_DIR)

    if close_test is not None:
        plot_predictions_on_price(close_test, labels, preds, RESULT_DIR)

    plot_probability_distribution(probs, labels, RESULT_DIR)

    print(f"\n[Eval] Tất cả kết quả đã lưu tại: {RESULT_DIR}")
    return preds, probs, labels


if __name__ == "__main__":
    full_evaluation()
