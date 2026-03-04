# xLSTM – Dự báo xu hướng cổ phiếu VNM

Triển khai mô hình **xLSTM (Extended Long Short-Term Memory)** theo paper  
*"xLSTM: Extended Long Short-Term Memory"* (Beck et al., arXiv:2405.04517v2)  
để dự báo xu hướng giá cổ phiếu VNM trên thị trường chứng khoán Việt Nam.

---

## 📁 Cấu trúc dự án

```
xlstm_vnm/
├── VNM_2225.csv          # Dữ liệu đầu vào (10/2021 → 6/2025)
├── config.py             # Tất cả hyperparameters & đường dẫn
├── data_loader.py        # Tiền xử lý, tạo chuỗi, DataLoader
├── xlstm_model.py        # Triển khai sLSTM, mLSTM, xLSTM blocks
├── train.py              # Pipeline huấn luyện + early stopping
├── evaluate.py           # Đánh giá + vẽ biểu đồ
├── predict.py            # Inference & backtest
├── main.py               # Entry point
├── requirements.txt
│
├── saved_models/         # Model weights, scaler, feature list
│   ├── xlstm_vnm_best.pt
│   ├── scaler.pkl
│   └── feature_cols.json
│
├── logs/                 # Lịch sử huấn luyện
│   ├── training_history.json
│   └── test_results.json
│
└── results/              # Biểu đồ & kết quả
    ├── confusion_matrix.png
    ├── learning_curves.png
    ├── predictions_on_price.png
    ├── probability_distribution.png
    ├── classification_report.txt
    └── backtest_results.csv
```

---

## 🏗️ Kiến trúc mô hình

```
Input (B, T=30, F=33)
        │
  ┌─────▼──────┐
  │ Input Proj │  Linear → LayerNorm → SiLU
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │ mLSTM Blk  │  Pre-Up-Proj → mLSTM (matrix memory) → GroupNorm → Gate → Down-Proj
  ├─────┬──────┤
  │ mLSTM Blk  │
  ├─────┬──────┤
  │ sLSTM Blk  │  LayerNorm → sLSTM (scalar + exp gates) → GroupNorm → Gated MLP
  ├─────┬──────┤
  │ mLSTM Blk  │
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │ Global Avg │  Mean pooling theo chiều thời gian
  │  Pool      │
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │ Classifier │  Linear → LayerNorm → GELU → Dropout → Linear
  └─────┬──────┘
        │
  Output (B, 3)   0=Giảm  1=Đi ngang  2=Tăng
```

### Thành phần chính

| Module | Mô tả | Theo paper |
|--------|-------|-----------|
| `sLSTMCell` | Scalar memory + Exponential gate + Memory mixing | Eq. 8–17 |
| `mLSTMCell` | Matrix memory C∈R^(d×d) + Covariance update | Eq. 19–27 |
| `sLSTMBlock` | Post up-projection residual block | Figure 10 |
| `mLSTMBlock` | Pre up-projection residual block | Figure 11 |
| `xLSTM` | Stacked blocks + Classifier | Figure 1 |

---

## ⚙️ Cấu hình mặc định (`config.py`)

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `SEQUENCE_LENGTH` | 30 | Số phiên lịch sử mỗi mẫu |
| `TREND_THRESHOLD` | 0.5% | Ngưỡng phân loại Tăng/Giảm |
| `hidden_size` | 128 | Chiều ẩn |
| `num_heads` | 4 | Số attention heads |
| `num_mlstm_layers` | 3 | Số mLSTM blocks |
| `num_slstm_layers` | 1 | Số sLSTM blocks |
| `learning_rate` | 1e-3 | Learning rate ban đầu |
| `epochs` | 100 | Số epoch tối đa |
| `patience` | 15 | Early stopping patience |

---

## 🚀 Cách chạy

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy toàn bộ pipeline (train → eval → predict)

```bash
python main.py --mode all
```

### 3. Chỉ huấn luyện

```bash
python main.py --mode train
```

### 4. Chỉ đánh giá (cần đã có model)

```bash
python main.py --mode eval
```

### 5. Dự báo phiên tiếp theo + Backtest

```bash
python main.py --mode predict

# Hoặc trực tiếp:
python predict.py                # Dự báo phiên tiếp theo
python predict.py backtest       # Chạy backtest toàn bộ
```

---

## 📊 Nhãn xu hướng

| Nhãn | Ký hiệu | Điều kiện |
|------|---------|-----------|
| 0    | GIẢM ↓  | log_return < -0.5% |
| 1    | NGANG → | -0.5% ≤ log_return ≤ 0.5% |
| 2    | TĂNG ↑  | log_return > 0.5% |

---

## 📈 Features sử dụng (33 features)

- **Giá & khối lượng**: open, high, low, close, volume
- **Trung bình động**: sma_10, sma_20, ema_20
- **Momentum**: macd_histogram, rsi_14, cci_14, momentum_10, roc_12
- **Biến động**: atr_14, body_size, daily_range, upper/lower shadow, rolling_std_20, hv_20
- **Lag returns**: return_lag_1 → return_lag_5
- **Vị trí giá**: distance_from_high/low_20
- **Khối lượng**: volume_ratio, volume_change
- **Thị trường**: vnindex_return, correlation_market_20, beta_20
- **Thời gian**: day_of_week

---

## 💡 Điều chỉnh hyperparameters

Mở `config.py` để thay đổi:

```python
# Tăng độ phức tạp model
MODEL_CONFIG = {
    "hidden_size":      256,   # 128 → 256
    "num_heads":        8,     # 4 → 8
    "num_mlstm_layers": 6,     # 3 → 6 (xLSTM[7:1])
    "num_slstm_layers": 1,
    ...
}

# Điều chỉnh ngưỡng phân loại
TREND_THRESHOLD = 0.01   # 1% thay vì 0.5%

# Tăng window size
SEQUENCE_LENGTH = 60     # 30 → 60 phiên
```

---

## 📚 Tham khảo

- Beck et al. (2024). *xLSTM: Extended Long Short-Term Memory*. arXiv:2405.04517v2
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.
- Code gốc: https://github.com/NX-AI/xlstm
