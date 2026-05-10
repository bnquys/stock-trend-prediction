# VNM RL Trading  v5.0
## Double Dueling DQN + Elliott Wave  |  Thị trường Việt Nam (HOSE)

---

## Cài đặt & Chạy

```bash
pip install -r requirements.txt

# Train model (500 episodes)
python train.py

# Tùy chỉnh số episodes
python train.py --episodes 1000

# Chỉ tạo chart (sau khi đã train)
python train.py --mode charts

# Cập nhật dữ liệu sau phiên (sau 15:30 VN)
python src/data/pull_data.py --update

# Kiểm tra trạng thái thị trường
python src/data/pull_data.py --status
```

---

## Cấu trúc Project

```
vnm_rl_project/
├── train.py                        ← Script chính — chạy file này
├── configs/config.yaml             ← Tất cả tham số (chỉnh tại đây)
├── requirements.txt
│
├── data/
│   └── VNM_2225.csv                ← 913 phiên, 43 features
│
├── src/
│   ├── data/
│   │   └── pull_data.py            ← Pull + cập nhật dữ liệu từ VNStock
│   ├── features/
│   │   ├── preprocessor.py         ← Load, split, RobustScaler
│   │   └── elliott.py              ← Elliott Wave detection (ElliottAgents paper)
│   ├── rl/
│   │   ├── env/trading_env.py      ← Môi trường VN (T+2, ±7%, phí 0.15%)
│   │   └── agent/dqn_agent.py      ← Double Dueling DQN + N-step + Soft target
│   └── visualization/
│       └── charts.py               ← 6 biểu đồ nền trắng
│
├── models/
│   ├── best_model.pkl              ← Lưu tự động khi val cải thiện
│   ├── last_model.pkl
│   └── scaler.pkl
│
├── logs/
│   ├── train.log                   ← Log chi tiết
│   ├── training_log.json           ← Metrics từng episode
│   └── test_results.pkl
│
└── outputs/
    ├── 0_dashboard.png             ← ⭐ Xem file này trước
    ├── 1_price_signals.png
    ├── 2_equity_drawdown.png
    ├── 3_training_curves.png
    ├── 4_trade_analysis.png
    └── 5_elliott_waves.png
```

---

## Bug Fixes so với v4

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|---------|
| **T=0 episode** | env bị done ngay lập tức | `reset()` bắt đầu từ `window`, assert `episode_len > 0` |
| **Warmup loss=0** | learn() chạy trước khi buffer đủ | Guard: `if len(buffer) < warmup_steps: return None` |
| **Sharpe nổ** | std≈0 khi 0 trades → div by zero | Clip std: `if std < 1e-7: sharpe = 0.0` |
| **Backward direction** | `reversed(body_grads)` sai thứ tự | Sửa thành `zip(body_grads, reversed(self.body))` |
| **Checkpoint bias** | Lưu model khi chưa học | Chỉ lưu khi `agent.learn_count > 0` |
| **T+2 block cứng** | Agent bị block không bán được | Chuyển thành penalty nhẹ `-0.001` thay vì block |

---

## Tham số quan trọng (`configs/config.yaml`)

### Để model học tốt hơn

```yaml
agent:
  n_episodes: 1000          # Tăng episodes để train lâu hơn
  lr: 0.0001                # Giữ nhỏ, tăng → loss nổ
  eps_decay_steps: 80000    # Explore nhiều hơn trước khi khai thác
  warmup_steps: 2000        # Phải fill buffer trước khi học
  batch_size: 256           # Lớn hơn → stable hơn nhưng chậm hơn
  hidden_layers: [256, 128, 64]  # Tăng capacity

env:
  stop_loss: 0.05           # 5% — tốt cho VNM
  take_profit: 0.10         # 10%
  max_hold_days: 30         # Tối đa 30 phiên
```

### Khi loss tăng liên tục
```yaml
agent:
  lr: 0.00005               # Giảm learning rate
  batch_size: 512           # Tăng batch size
  target_hard_every: 100    # Update target thường xuyên hơn
```

---

## Cập nhật dữ liệu hàng ngày

```bash
# Sau 15:30 VN mỗi ngày giao dịch:
python src/data/pull_data.py --update

# Lập lịch cron (chạy 16:00 VN = 09:00 UTC):
0 9 * * 1-5 cd /path/to/project && python src/data/pull_data.py --update

# Pull cổ phiếu khác
python src/data/pull_data.py --symbol HPG --output data/HPG_features.csv
```

---

## Kiến trúc Model

### Double Dueling DQN

```
Observation (725 dims):
  ├── 20-bar × 27 technical features  (SMA,EMA,MACD,RSI,ATR,BB,CCI,OBV,Volume,...)
  ├── 20-bar × 9 Elliott Wave features (wave_pos, direction, fib_618, fib_382,
  │                                     support, resistance, target, confidence, signal)
  └── 5 portfolio state dims           (in_pos, unrealized_pnl, cash_ratio,
                                        hold_fraction, t_plus_available)

Network (Dueling):
  Input(725) → FC(256,ReLU) → FC(128,ReLU) → FC(64,ReLU)
                                              ↙              ↘
                    Value: FC(64)→V(1)        Advantage: FC(64)→A(3)
                                              ↘              ↙
                              Q(s,a) = V(s) + [A(s,a) - mean(A)]

Training:
  • Double DQN: online net chọn action, target net đánh giá
  • N-step returns (n=3): G = r_t + γr_{t+1} + γ²r_{t+2}
  • Soft target update: θ_tgt ← 0.01·θ + 0.99·θ_tgt (mỗi step)
  • Hard target update: mỗi 300 learn steps
  • Gradient clipping: global norm ≤ 0.5
```

### Reward Function

```python
# SELL: Sharpe-adjusted PnL
reward = pnl_pct + pnl_pct/(volatility+ε) * 0.2
if pnl >= take_profit: reward += 0.04    # TP bonus
if pnl <= -stop_loss:  reward -= 0.04   # SL penalty

# Auto SL/TP khi pnl vượt ngưỡng (tôn trọng T+2)
# HOLD với vị thế: daily_log_return * 0.15
# HOLD không vị thế: -0.0001 (penalty nhỏ)
# Over-trading: -0.003
# T+2 vi phạm: -0.001 (nhẹ, không block cứng)
```

### Elliott Wave Features (từ ElliottAgents, arXiv:2507.03435)

Paper chứng minh: DRL backtesting tăng accuracy nhận diện sóng từ 53% → **89%**

```
wave_position      : Đang ở sóng nào (0→1 normalized)
wave_direction     : +1=impulse bullish, -1=bearish, ±0.5=ABC
fib_dist_618       : % cách mức Fibonacci 61.8%
fib_dist_382       : % cách mức Fibonacci 38.2%  
support_dist       : % cách support từ wave endpoint
resistance_dist    : % cách resistance từ wave endpoint
target_dist        : % cách target price (161.8% extension)
pattern_conf       : Độ tin cậy pattern (0-1)
elliott_signal     : Composite signal [-1, +1]
```

---

## Biểu đồ Output

| File | Nội dung | Mục đích |
|------|----------|---------|
| `0_dashboard.png` | Tổng hợp tất cả | In báo cáo 1 trang |
| `1_price_signals.png` | Nến Nhật + MA + MUA/BÁN | Xem tín hiệu giao dịch |
| `2_equity_drawdown.png` | Vốn RL vs B&H + MDD | Đánh giá rủi ro/lợi nhuận |
| `3_training_curves.png` | Return/Loss/WinRate/Epsilon | Theo dõi quá trình học |
| `4_trade_analysis.png` | PnL/Holding/Exit reasons | Phân tích chi tiết lệnh |
| `5_elliott_waves.png` | Sóng Elliott + Fibonacci | Hiểu pattern thị trường |

---

## Nguồn

- **ElliottAgents** (arXiv:2507.03435) — Elliott Wave + DRL backtesting
- **FinVision** (arXiv:2411.08899) — Multi-indicator technical analysis
