"""
config.py
=========
Tập trung toàn bộ siêu tham số (hyperparameters) và cấu hình cho dự án
dự báo xu hướng cổ phiếu VNM bằng mô hình xLSTM.
"""

import os

# ─── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data/VNM_2225.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── Cấu hình dữ liệu ─────────────────────────────────────────────────────────
# Các features được sử dụng làm đầu vào mô hình
FEATURE_COLUMNS = [
    # Giá & khối lượng cơ bản
    "open", "high", "low", "close", "volume",
    # Trung bình động
    "sma_10", "sma_20", "ema_20",
    # Chỉ báo momentum
    "macd_histogram", "rsi_14", "cci_14", "momentum_10", "roc_12",
    # Biến động giá
    "atr_14", "body_size", "daily_range", "upper_shadow", "lower_shadow",
    "rolling_std_20", "historical_volatility_20",
    # Lag returns
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    # Vị trí trong dải giá 20 phiên
    "distance_from_high_20", "distance_from_low_20",
    # Khối lượng
    "volume_ratio", "volume_change",
    # Thị trường chung
    "vnindex_return", "correlation_market_20", "beta_20",
    # Thời gian
    "day_of_week",
]

TARGET_COLUMN = "close"   
SEQUENCE_LENGTH = 60       # Số phiên lịch sử mỗi mẫu (window size)
FORECAST_HORIZON = 1       # Dự báo 1 phiên tiếp theo

# ─── Nhãn phân loại xu hướng ──────────────────────────────────────────────────
# 0 = Giảm   (return < -THRESHOLD)
# 1 = Đi ngang (-THRESHOLD <= return <= THRESHOLD)
# 2 = Tăng   (return > THRESHOLD)
TREND_THRESHOLD = 0.005     # 0.5%
NUM_CLASSES = 3

# Tỷ lệ chia train / validation / test
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO 

# ─── Cấu hình mô hình xLSTM ───────────────────────────────────────────────────
MODEL_CONFIG = {
    # Số features đầu vào (sẽ được cập nhật sau preprocessing)
    "input_size":    None,          # được gán tự động

    # Chiều ẩn của mô hình
    "hidden_size":   64,

    # Số heads cho mLSTM / sLSTM
    "num_heads":     4,

    # Số block xLSTM
    "num_layers":    3,

    # Tỷ lệ mLSTM : sLSTM block  (theo paper: xLSTM[7:1])
    # num_mlstm_layers blocks + num_slstm_layers blocks = num_layers
    "num_mlstm_layers": 2,          # mLSTM blocks (parallelizable)
    "num_slstm_layers": 1,          # sLSTM blocks (memory mixing)

    # Up-projection factor (pre / post)
    "proj_factor":   4/3,

    # Dropout
    "dropout":       0.3,

    # Số lớp fully-connected phân loại
    "fc_hidden":     64,

    # Số lớp đầu ra
    "num_classes":   NUM_CLASSES,
}

# ─── Cấu hình huấn luyện ──────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs":           100,
    "batch_size":       32,
    "learning_rate":    1e-3,
    "weight_decay":     0.05,
    "patience":         20,          # Early stopping
    "min_delta":        1e-4,        # Ngưỡng cải thiện tối thiểu
    "lr_scheduler":     "cosine",    # "cosine" | "plateau"
    "warmup_epochs":    5,
    "grad_clip":        1.0,         # Gradient clipping
    "seed":             42,
    "device":           "auto",      # "auto" | "cpu" | "cuda" | "mps"
}

# ─── Cấu hình logging ─────────────────────────────────────────────────────────
LOG_CONFIG = {
    "log_interval":   10,   # In kết quả mỗi N epoch
    "save_best_only": True,
    "model_name":     "xlstm_vnm_best.pt",
    "history_name":   "training_history.json",
}
