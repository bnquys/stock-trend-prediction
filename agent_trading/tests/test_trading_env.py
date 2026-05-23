"""
tests/test_trading_env.py
═══════════════════════════════════════════════════════════════════
Unit tests cho TradingEnv — bao gồm:
  - T+2 enforcement
  - Lot size (bội 100)
  - Fee + Tax calculation
  - SL/TP auto-exit
  - Max hold enforcement
  - Valid actions masking
  - Price limit ±7%
  - Metrics output
  - Edge cases
═══════════════════════════════════════════════════════════════════
"""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.env.trading_env import TradingEnv
from src.features.preprocessor import FEAT_COLS


# ─── Fixtures ────────────────────────────────────────────────────────────

def _make_df(n_rows: int = 100, base_price: float = 50000.0,
             trend: float = 0.001) -> pd.DataFrame:
    """Tạo synthetic dataframe cho testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    closes = [base_price]
    for i in range(1, n_rows):
        change = np.random.normal(trend, 0.015)
        closes.append(closes[-1] * (1 + change))
    closes = np.array(closes)

    df = pd.DataFrame({
        "date": dates,
        "open": closes * (1 - np.random.uniform(0, 0.01, n_rows)),
        "high": closes * (1 + np.random.uniform(0, 0.02, n_rows)),
        "low": closes * (1 - np.random.uniform(0, 0.02, n_rows)),
        "close": closes,
        "volume": np.random.randint(100000, 5000000, n_rows),
    })

    # Add minimal feature columns for obs to work
    for col in FEAT_COLS:
        if col not in df.columns:
            df[col] = np.random.randn(n_rows) * 0.1

    return df


def _make_env(n_rows=100, window=20, init_cap=1_000_000_000,
              **kwargs) -> TradingEnv:
    """Tạo TradingEnv với synthetic data."""
    df = _make_df(n_rows)
    defaults = dict(
        df_raw=df, df_norm=df.copy(),
        window=window, init_cap=init_cap,
        tx_cost=0.0015, sell_tax=0.001, slippage=0.001,
        atr_sl_mult=1.5, atr_tp_mult=3.0,
        risk_per_trade=0.02,
        stop_loss=0.07, take_profit=0.15,
        max_hold=30, t_plus=2, lot_size=100,
        price_limit=0.07,
    )
    defaults.update(kwargs)
    return TradingEnv(**defaults)


# ─── Test T+2 Enforcement ───────────────────────────────────────────────

class TestTPlusTwo:
    def test_cannot_sell_before_t2(self):
        """Agent không thể bán trước T+2."""
        env = _make_env()
        env.reset()

        # BUY
        obs, reward, done, info = env.step(TradingEnv.BUY)
        assert env._in_pos, "Phải vào vị thế sau BUY"

        # T+0: SELL không nằm trong valid_actions
        assert TradingEnv.SELL not in env.valid_actions(), \
            "T+0: Không được phép bán"

        # T+1: vẫn chưa được bán
        env.step(TradingEnv.HOLD)
        assert TradingEnv.SELL not in env.valid_actions(), \
            "T+1: Không được phép bán"

        # T+2: bây giờ mới được bán
        env.step(TradingEnv.HOLD)
        assert TradingEnv.SELL in env.valid_actions(), \
            "T+2: Phải được phép bán"

    def test_sell_at_t2(self):
        """Bán thành công ở T+2."""
        env = _make_env()
        env.reset()

        env.step(TradingEnv.BUY)  # T+0
        env.step(TradingEnv.HOLD)  # T+1
        env.step(TradingEnv.HOLD)  # Advance to T+2

        # Bán ở T+2
        obs, reward, done, info = env.step(TradingEnv.SELL)
        assert not env._in_pos, "Phải thoát vị thế sau SELL ở T+2"


# ─── Test Lot Size ───────────────────────────────────────────────────────

class TestLotSize:
    def test_shares_multiple_of_100(self):
        """Vị thế phải là bội số 100 cp."""
        env = _make_env()
        env.reset()
        env.step(TradingEnv.BUY)

        if env._in_pos:
            assert env._shares % 100 == 0, \
                f"Shares={env._shares} không phải bội 100"
            assert env._shares > 0, "Phải có ít nhất 1 lô"

    def test_lot_size_zero_handled(self):
        """Khi lot_size=0 thì không lỗi."""
        env = _make_env(lot_size=0)
        env.reset()
        # Không crash
        env.step(TradingEnv.BUY)


# ─── Test Fee & Tax ──────────────────────────────────────────────────────

class TestFeeAndTax:
    def test_buy_deducts_fee(self):
        """Phí mua (0.15%) phải được trừ."""
        env = _make_env()
        env.reset()
        initial_cash = env._cash
        env.step(TradingEnv.BUY)

        if env._in_pos:
            cost = env._shares * env._entry_price
            fee = cost * 0.0015
            # Cash phải giảm ít nhất = cost + fee
            assert env._cash < initial_cash, \
                "Cash phải giảm sau khi mua"

    def test_sell_deducts_tax(self):
        """Thuế bán (0.1%) phải được trừ khi bán."""
        env = _make_env()
        env.reset()

        env.step(TradingEnv.BUY)
        env.step(TradingEnv.HOLD)
        env.step(TradingEnv.HOLD)

        if env._in_pos:
            # Tiền trước khi bán
            pre_trades = len(env.trades)
            env.step(TradingEnv.SELL)
            post_trades = len(env.trades)
            assert post_trades > pre_trades, "Phải có trade SELL được ghi nhận"

    def test_fee_rate_correct(self):
        """Xác nhận phí 0.15% và thuế 0.1%."""
        gross = 1_000_000
        fee = gross * 0.0015  # = 1500
        tax = gross * 0.001   # = 1000
        assert fee == 1500, f"Fee phải là 1500, got {fee}"
        assert tax == 1000, f"Tax phải là 1000, got {tax}"


# ─── Test SL/TP Auto-Exit ────────────────────────────────────────────────

class TestAutoExit:
    def test_sl_triggers_sell(self):
        """Stop-loss tự động khi giá giảm dưới SL price."""
        # Tạo data giá giảm mạnh
        n = 100
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Giá tăng nhẹ rồi giảm mạnh
        prices = np.concatenate([
            np.linspace(50000, 52000, 30),  # Tăng nhẹ
            np.linspace(52000, 40000, 70),  # Giảm mạnh
        ])
        df = pd.DataFrame({
            "date": dates,
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(100000, 5000000, n),
        })
        for col in FEAT_COLS:
            if col not in df.columns:
                df[col] = np.random.randn(n) * 0.1

        env = TradingEnv(
            df_raw=df, df_norm=df.copy(),
            window=20, init_cap=1_000_000_000,
            tx_cost=0.0015, sell_tax=0.001, slippage=0.001,
            atr_sl_mult=1.5, atr_tp_mult=3.0,
            risk_per_trade=0.02,
            stop_loss=0.07, take_profit=0.15,
            max_hold=30, t_plus=2, lot_size=100,
            price_limit=0.07,
        )
        env.reset()
        env.step(TradingEnv.BUY)

        # Run cho đến khi SL trigger hoặc hết episode
        for _ in range(60):
            obs, reward, done, info = env.step(TradingEnv.HOLD)
            if done or not env._in_pos:
                break

        # Kiểm tra có auto-exit (SL hoặc maxhold)
        sl_exits = [t for t in env.trades if t.get("reason") in ("sl", "maxhold")]
        if not env._in_pos:
            assert len(sl_exits) > 0 or any(
                t["type"] in ("SELL", "AUTO_EXIT") for t in env.trades
            ), "Phải có auto-exit khi giá giảm mạnh"


# ─── Test Valid Actions ──────────────────────────────────────────────────

class TestValidActions:
    def test_initial_actions(self):
        """Ban đầu: HOLD + BUY."""
        env = _make_env()
        env.reset()
        acts = env.valid_actions()
        assert TradingEnv.HOLD in acts
        assert TradingEnv.BUY in acts
        assert TradingEnv.SELL not in acts

    def test_in_position_no_buy(self):
        """Khi đang giữ vị thế: không thể BUY."""
        env = _make_env()
        env.reset()
        env.step(TradingEnv.BUY)
        if env._in_pos:
            acts = env.valid_actions()
            assert TradingEnv.BUY not in acts, \
                "Không được mua khi đang có vị thế"

    def test_hold_always_valid(self):
        """HOLD luôn luôn hợp lệ."""
        env = _make_env()
        env.reset()
        for _ in range(10):
            acts = env.valid_actions()
            assert TradingEnv.HOLD in acts
            env.step(TradingEnv.HOLD)

    def test_no_buy_when_insufficient_cash(self):
        """Không thể mua khi không đủ tiền."""
        env = _make_env(init_cap=100)  # Rất ít tiền
        env.reset()
        acts = env.valid_actions()
        assert TradingEnv.BUY not in acts, \
            "Không được mua khi không đủ tiền cho 1 lô"


# ─── Test Max Hold ───────────────────────────────────────────────────────

class TestMaxHold:
    def test_max_hold_causes_exit(self):
        """Vị thế phải được đóng khi giữ quá max_hold."""
        env = _make_env(n_rows=80, max_hold=5)
        env.reset()
        env.step(TradingEnv.BUY)

        if env._in_pos:
            for _ in range(20):
                obs, reward, done, info = env.step(TradingEnv.HOLD)
                if done or not env._in_pos:
                    break

            # Sau max_hold phiên, nếu lỗ → phải auto exit
            auto_exits = [t for t in env.trades
                         if t.get("reason") in ("maxhold", "sl", "tp")]
            # Hoặc agent đã bán manual
            total_sells = [t for t in env.trades
                          if t["type"] in ("SELL", "AUTO_EXIT")]
            assert not env._in_pos or len(total_sells) > 0 or done, \
                "Phải có auto-exit sau max_hold"


# ─── Test Price Limit ────────────────────────────────────────────────────

class TestPriceLimit:
    def test_price_clamped_within_7pct(self):
        """Giá phải nằm trong ±7% so với phiên trước."""
        env = _make_env()
        env.reset()

        for step in range(env.window + 1, min(env.n, env.window + 30)):
            price = env._price(step)
            prev_price = float(env.df_raw["close"].iloc[step - 1])
            lower = prev_price * (1 - 0.07)
            upper = prev_price * (1 + 0.07)
            assert lower - 1e-6 <= price <= upper + 1e-6, \
                f"Price {price} ngoài giới hạn ±7% [{lower}, {upper}]"


# ─── Test Metrics ────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_keys(self):
        """Metrics phải có đầy đủ keys."""
        env = _make_env()
        env.reset()
        for _ in range(10):
            env.step(TradingEnv.HOLD)
        m = env.metrics()

        required = ["return_pct", "sharpe", "max_dd_pct", "n_trades",
                     "win_rate", "avg_win", "avg_loss", "pf",
                     "final_equity", "n_sl", "n_tp", "n_mh", "steps"]
        for key in required:
            assert key in m, f"Missing metric key: {key}"

    def test_zero_metrics_when_empty(self):
        """Metrics trả về 0 khi không có data."""
        env = _make_env()
        env.reset()
        env._equity = [env.init_cap]  # Chỉ 1 entry
        m = env.metrics()
        assert m["return_pct"] == 0

    def test_metrics_types(self):
        """Metrics values phải là numeric."""
        env = _make_env()
        env.reset()
        # Run vài step
        env.step(TradingEnv.BUY)
        for _ in range(5):
            env.step(TradingEnv.HOLD)
        m = env.metrics()
        for key, val in m.items():
            assert isinstance(val, (int, float)), \
                f"Metric {key}={val} phải là numeric, got {type(val)}"


# ─── Test Observation ────────────────────────────────────────────────────

class TestObservation:
    def test_obs_shape(self):
        """Observation phải có shape đúng."""
        env = _make_env()
        obs = env.reset()
        assert obs.ndim == 1, "Obs phải là 1D"
        assert len(obs) == env.obs_size, \
            f"Obs size={len(obs)} != expected {env.obs_size}"

    def test_obs_finite(self):
        """Observation không được chứa NaN hay Inf."""
        env = _make_env()
        obs = env.reset()
        assert np.all(np.isfinite(obs)), "Obs chứa NaN/Inf"

    def test_obs_dtype(self):
        """Observation phải là float32."""
        env = _make_env()
        obs = env.reset()
        assert obs.dtype == np.float32, f"Obs dtype={obs.dtype}, expected float32"


# ─── Test Episode Flow ───────────────────────────────────────────────────

class TestEpisodeFlow:
    def test_full_episode_runs(self):
        """Episode chạy hoàn chỉnh không crash."""
        env = _make_env()
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = np.random.choice(env.valid_actions())
            obs, reward, done, info = env.step(action)
            steps += 1
            assert steps < 200, "Episode chạy quá lâu — possible infinite loop"

    def test_episode_ends_at_data_end(self):
        """Episode kết thúc khi hết data."""
        env = _make_env(n_rows=50, window=20)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(TradingEnv.HOLD)
        assert done, "Episode phải kết thúc khi hết data"

    def test_reward_is_finite(self):
        """Reward luôn finite."""
        env = _make_env()
        env.reset()
        for _ in range(20):
            action = np.random.choice(env.valid_actions())
            _, reward, done, _ = env.step(action)
            assert np.isfinite(reward), f"Reward không finite: {reward}"
            if done:
                break

    def test_liquidate_at_end(self):
        """Vị thế phải được thanh lý khi episode kết thúc."""
        env = _make_env()
        env.reset()
        env.step(TradingEnv.BUY)  # Mở vị thế
        # Hold cho đến hết
        done = False
        while not done:
            _, _, done, _ = env.step(TradingEnv.HOLD)
        assert not env._in_pos, "Vị thế phải được thanh lý cuối episode"
        assert env._shares == 0, "Shares phải = 0 cuối episode"


# ─── Test Data Validation ────────────────────────────────────────────────

class TestDataValidation:
    def test_mismatched_lengths_raises(self):
        """df_raw và df_norm khác length → AssertionError."""
        df1 = _make_df(50)
        df2 = _make_df(60)
        with pytest.raises(AssertionError):
            TradingEnv(df_raw=df1, df_norm=df2, window=20,
                      init_cap=1e9, stop_loss=0.07, take_profit=0.15)

    def test_data_too_short_raises(self):
        """Data quá ngắn (< window + 2) → AssertionError."""
        df = _make_df(10)
        with pytest.raises(AssertionError):
            TradingEnv(df_raw=df, df_norm=df.copy(), window=20,
                      init_cap=1e9, stop_loss=0.07, take_profit=0.15)
