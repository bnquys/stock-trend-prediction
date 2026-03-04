"""
train.py
========
Pipeline huấn luyện mô hình xLSTM:
  - Setup thiết bị, seed
  - Load & chuẩn bị dữ liệu
  - Khởi tạo model, optimizer, scheduler
  - Vòng lặp train/val với Early Stopping
  - Lưu best model & lịch sử huấn luyện
"""

import os
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from config import MODEL_DIR, LOG_DIR, TRAIN_CONFIG, MODEL_CONFIG, LOG_CONFIG
from data_loader import VNMDataProcessor, build_dataloaders
from xlstm_model import xLSTM


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, model_path: str):
        self.patience   = patience
        self.min_delta  = min_delta
        self.model_path = model_path
        self.best_val   = float("inf")
        self.counter    = 0
        self.stop       = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.counter  = 0
            torch.save(model.state_dict(), self.model_path)
            return True   # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


# ─────────────────────────────────────────────────────────────────────────────
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, y)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, y)
    n = len(loader)
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer_scheduler(model, cfg, n_train_batches):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-5,
    )

    total_steps = cfg["epochs"] * n_train_batches

    if cfg["lr_scheduler"] == "cosine":
        # Warmup + Cosine annealing
        warmup_steps = cfg["warmup_epochs"] * n_train_batches

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
        )

    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Setup ─────────────────────────────────────────────────────────────────
    cfg_train = TRAIN_CONFIG
    cfg_model = MODEL_CONFIG.copy()

    set_seed(cfg_train["seed"])
    device = get_device(cfg_train["device"])
    print(f"[Train] Thiết bị: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    processor = VNMDataProcessor()
    (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te), feature_cols, close_test = \
        processor.prepare()
    processor.save_scaler()

    # Lưu feature names để dùng lại khi inference
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    train_loader, val_loader, test_loader = build_dataloaders(
        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te),
        batch_size=cfg_train["batch_size"]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg_model["input_size"] = len(feature_cols)
    model = xLSTM.from_config(cfg_model).to(device)

    # ── Loss: Label smoothing để tránh overfit ────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_scheduler(
        model, cfg_train, len(train_loader)
    )

    # ── Early Stopping ────────────────────────────────────────────────────────
    model_path   = os.path.join(MODEL_DIR, LOG_CONFIG["model_name"])
    early_stop   = EarlyStopping(
        patience   = cfg_train["patience"],
        min_delta  = cfg_train["min_delta"],
        model_path = model_path,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    print(f"\n{'═'*60}")
    print(f" BẮT ĐẦU HUẤN LUYỆN  —  {cfg_train['epochs']} epochs")
    print(f"{'═'*60}")

    for epoch in range(1, cfg_train["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg_train["grad_clip"]
        )

        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        if isinstance(scheduler, optim.lr_scheduler.LambdaLR):
     
            # step mỗi epoch ở đây
            scheduler.step()
        else:
            scheduler.step(vl_loss)

        lr_now = optimizer.param_groups[0]["lr"]

        # Lưu history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(lr_now)

        # Early stopping
        improved = early_stop.step(vl_loss, model)
        marker   = "[best]" if improved else ""

        if epoch % LOG_CONFIG["log_interval"] == 0 or epoch == 1 or improved:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{cfg_train['epochs']}  |  "
                f"tr_loss: {tr_loss:.4f}  tr_acc: {tr_acc:.4f}  |  "
                f"vl_loss: {vl_loss:.4f}  vl_acc: {vl_acc:.4f}  |  "
                f"lr: {lr_now:.2e}  |  {elapsed:.1f}s{marker}"
            )

        if early_stop.stop:
            print(f"\n[Train] Early stopping tại epoch {epoch}")
            break

    # ── Lưu history ───────────────────────────────────────────────────────────
    hist_path = os.path.join(LOG_DIR, LOG_CONFIG["history_name"])
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] Lịch sử huấn luyện → {hist_path}")

    # ── Đánh giá trên test set ────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'═'*60}")
    print(f" KẾT QUẢ TEST  —  loss: {te_loss:.4f}   acc: {te_acc:.4f}")
    print(f"{'═'*60}")

    # Lưu kết quả test
    results = {
        "test_loss": te_loss,
        "test_acc":  te_acc,
        "best_val_loss": early_stop.best_val,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(LOG_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, history, (X_te, y_te), close_test


if __name__ == "__main__":
    main()
