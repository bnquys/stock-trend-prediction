"""
src/rl/env/trading_env.py
══════════════════════════════════════════════════════════════════════════
Trading Environment — Tuân thủ quy tắc HOSE (Việt Nam)

Đặc điểm:
   T+2: mua ngày T, bán sớm nhất ngày T+2
   Lot size 100 cổ phiếu (quy định HOSE)
   Phí giao dịch + thuế bán 0.1%
   Slippage mô phỏng
   Giới hạn giá ±7% (trần/sàn HOSE)
   ATR-based Stop-loss / Take-profit / Max hold tự động
   Reward = Risk-adjusted PnL khi SELL

Prices: dùng df_raw (RAW, không normalize) để tính PnL chính xác.
Obs:    dùng df_norm (NORMALIZED) để cho agent học.

Actions:  0 = HOLD   1 = BUY   2 = SELL
══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.technical.preprocessor import get_obs, FEAT_COLS

log = logging.getLogger(__name__)


class TradingEnv:
    HOLD = 0
    BUY  = 1
    SELL = 2

    def __init__(
        self,
        df_raw:  pd.DataFrame,    # RAW prices — dùng tính PnL
        df_norm: pd.DataFrame,    # Normalized features — dùng làm obs
        window:      int   = 20,
        init_cap:    float = 1_000_000_000,
        tx_cost:     float = 0.0015,     # 0.15% phí mua/bán
        sell_tax:    float = 0.001,      # 0.1% thuế bán (HOSE)
        slippage:    float = 0.001,      # 0.1% slippage
        # ── ATR-based Dynamic SL/TP ────────────────────────────
        atr_sl_mult: float = 1.5,        # SL = entry - 1.5 × ATR
        atr_tp_mult: float = 3.0,        # TP = entry + 3.0 × ATR → R:R = 1:2
        # ── Fallback SL/TP cố định ─────────────────────────────
        stop_loss:   float = 0.07,       # Cắt lỗ tối đa 7%
        take_profit: float = 0.15,       # Chốt lời tối đa 15%
        risk_per_trade: float = 0.02,    # Rủi ro tối đa 2% vốn
        max_hold:    int   = 30,         # Tối đa 30 phiên
        t_plus:      int   = 2,          # T+2 HOSE
        lot_size:    int   = 100,        # 1 lô = 100 cp (HOSE)
        price_limit: float = 0.07,       # ±7% trần/sàn HOSE
        advance_fee_rate: float = 0.0004,# Phí ứng trước tiền bán
        # ── Analysis Embedding (phân tích cơ bản) ──────────────
        stock_id:        str | None = None,   # Mã cổ phiếu (VNM, FPT, ...)
        analysis_model:  str | None = None,   # LLM model cho pipeline
        embedding_cache = None,               # EmbeddingCache instance (pre-loaded, zero I/O)
    ):
        assert len(df_raw) == len(df_norm), "df_raw và df_norm phải cùng length"
        assert len(df_raw) > window + 2, f"Data quá ngắn: {len(df_raw)}"

        self.df_raw      = df_raw.reset_index(drop=True)
        self.df_norm     = df_norm.reset_index(drop=True)
        self.window      = window
        self.init_cap    = init_cap
        self.tx_cost     = tx_cost
        self.sell_tax    = sell_tax
        self.slippage    = slippage
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.risk_per_trade = risk_per_trade
        self.stop_loss   = stop_loss
        self.take_profit = take_profit
        self.max_hold    = max_hold
        self.t_plus      = t_plus
        self.lot_size    = lot_size
        self.price_limit = price_limit
        self.advance_fee_rate = advance_fee_rate
        self.n             = len(df_raw)

        self._feat_cols = [c for c in FEAT_COLS if c in df_norm.columns]
        self.obs_size   = window * len(self._feat_cols) + 7  # 7 portfolio state dims

        # ── Precompute obs matrix (zero DataFrame overhead per step) ──
        self._obs_matrix = df_norm[self._feat_cols].fillna(0.0).values.astype(np.float32)
        # Precompute raw close prices as numpy array
        self._close_arr = df_raw["close"].values.astype(np.float64)

        # ── Analysis Embedding ─────────────────────────────────
        self.stock_id       = stock_id
        self.analysis_model = analysis_model
        self._embedding_cache = embedding_cache
        self._analysis_enabled = (stock_id is not None and analysis_model is not None)

        # ── Precompute ATR(14) cho df_raw ──────────────────────
        self._atr = self._compute_atr(df_raw, period=14)

        # State (sẽ được reset)
        self._cash        = 0.0
        self._shares      = 0.0
        self._entry_price = 0.0
        self._entry_step  = 0
        self._sl_price    = 0.0   # ATR-based stop-loss price
        self._tp_price    = 0.0   # ATR-based take-profit price
        self._step        = window
        self._in_pos      = False
        self._done        = True
        self._cash_queue  = []
        self._last_trade_step = 0  # Theo dõi over-trading
        self.trades: list[dict] = []
        self._equity: list[float] = []
        self._rewards: list[float] = []

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Tính ATR(14) vectorized — không dùng Python loop."""
        c = df["close"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        n = len(c)
        # True Range vectorized
        tr = np.empty(n)
        tr[0] = h[0] - l[0]
        hl = h[1:] - l[1:]
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(l[1:] - c[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))
        # Cumulative sum for rolling mean ATR
        cs = np.cumsum(tr)
        atr = np.empty(n)
        for i in range(min(period, n)):
            atr[i] = cs[i] / (i + 1)
        if n > period:
            atr[period:] = (cs[period:] - cs[:-period]) / period
        return atr

    def _get_atr(self, step: int | None = None) -> float:
        """Lấy ATR(14) tại bước hiện tại."""
        idx = step if step is not None else self._step
        idx = int(np.clip(idx, 0, self.n - 1))
        return float(self._atr[idx])

    def _get_atr_pct(self, step: int | None = None) -> float:
        """ATR(14) / close — biến động tương đối."""
        price = self._price(step)
        atr = self._get_atr(step)
        return atr / (price + 1e-9)

    def reset(self) -> np.ndarray:
        self._cash        = float(self.init_cap)
        self._shares      = 0.0
        self._entry_price = 0.0
        self._entry_step  = 0
        self._sl_price    = 0.0
        self._tp_price    = 0.0
        self._step        = self.window
        self._in_pos      = False
        self._done        = False
        self._cash_queue  = []
        self._last_trade_step = 0
        self.trades       = []
        self._equity      = [self.init_cap]
        self._rewards     = []
        return self._obs()

    def step(self, action: int):
        assert not self._done
        price  = self._price()
        reward = 0.0
        held   = self._step - self._entry_step if self._in_pos else 0

        # ── Cập nhật Tiền chờ về (T+2 Cash Settlement) ────────────
        new_cash_queue = []
        for item in self._cash_queue:
            item["wait"] -= 1
            if item["wait"] <= 0:
                self._cash += item["amount"]
            else:
                new_cash_queue.append(item)
        self._cash_queue = new_cash_queue

        # ── T+2: có thể bán không? ────────────────────────────────
        can_sell = self._in_pos and (held >= self.t_plus)
        total_cash_available = self._cash + sum(item["amount"] for item in self._cash_queue)

        # ── Execute ────────────────────────────────────────────────
        if action == self.BUY:
            if not self._in_pos and total_cash_available > 0:
                exec_p = self._round_tick(price * (1 + self.slippage))
                fee    = total_cash_available * self.tx_cost
                usable = total_cash_available - fee

                # Lot size: làm tròn xuống bội số lot_size
                if self.lot_size > 0 and exec_p > 0:
                    # 1. Tính toán rủi ro & Position Sizing
                    atr = self._get_atr()
                    atr_sl = atr * self.atr_sl_mult
                    atr_tp = atr * self.atr_tp_mult

                    sl_pct = min(atr_sl / (exec_p + 1e-9), self.stop_loss)
                    tp_pct = min(atr_tp / (exec_p + 1e-9), self.take_profit)
                    sl_pct = max(sl_pct, 0.01)
                    tp_pct = max(tp_pct, 0.02)
                    
                    sl_price = exec_p * (1 - sl_pct)
                    tp_price = exec_p * (1 + tp_pct)
                    
                    risk_per_share = exec_p - sl_price
                    max_risk_amount = (self._cash + sum(i["amount"] for i in self._cash_queue)) * self.risk_per_trade
                    
                    max_shares_by_risk = max_risk_amount / risk_per_share if risk_per_share > 0 else float("inf")
                    max_shares_by_cash = usable / exec_p
                    
                    max_shares = int(min(max_shares_by_risk, max_shares_by_cash))
                    shares     = (max_shares // self.lot_size) * self.lot_size
                    
                    if shares <= 0:
                        reward = -0.001  # Không đủ tiền/khả năng rủi ro để mua 1 lô
                    else:
                        cost = shares * exec_p
                        actual_fee = cost * self.tx_cost
                        total_cost = cost + actual_fee
                        
                        # Xử lý ứng trước tiền bán
                        advance_fee = 0.0
                        if self._cash >= total_cost:
                            self._cash -= total_cost
                        else:
                            shortfall = total_cost - self._cash
                            self._cash = 0.0
                            self._cash_queue.sort(key=lambda x: x["wait"])
                            for item in self._cash_queue:
                                if shortfall <= 0: break
                                take = min(shortfall, item["amount"])
                                item["amount"] -= take
                                shortfall -= take
                                advance_fee += take * self.advance_fee_rate * item["wait"]
                                
                        self._cash -= advance_fee

                        self._shares = float(shares)
                        self._entry_price = exec_p
                        self._entry_step  = self._step

                        self._sl_price = sl_price
                        self._tp_price = tp_price
                        self._in_pos   = True
                        self._last_trade_step = self._step

                        self.trades.append({
                            "type":      "BUY",
                            "step":      self._step,
                            "date":      self._date(),
                            "price":     round(exec_p, 2),
                            "shares":    shares,
                            "adv_fee":   round(advance_fee, 2),
                            "atr":       round(atr, 2),
                            "sl_price":  round(self._sl_price, 2),
                            "tp_price":  round(self._tp_price, 2),
                        })
                else:
                    reward = -0.001

        elif action == self.SELL:
            if can_sell:
                reward = self._execute_sell(price, held, reason="manual")
            elif self._in_pos and not can_sell:
                # T+2 chưa thoả: penalty nhẹ
                reward = -0.001
            else:
                # SELL khi không có vị thế
                reward = -0.001

        else:  # HOLD
            if self._in_pos:
                unrealized = (price - self._entry_price) / self._entry_price
                if unrealized > 0:
                    # Thưởng nhẹ khi đang lãi — khuyến khích giữ vị thế tốt
                    reward = 0.02
                elif unrealized < -0.02:
                    # Nhắc nhở nhẹ khi đang lỗ >2%
                    reward = -0.01
                else:
                    reward = 0.0
            # HOLD không vị thế: reward = 0 (không phạt, không thưởng)

        # ── Auto SL / TP / MaxHold (ATR-based) ─────────────────────
        if self._in_pos:
            held = self._step - self._entry_step
            can_auto_sell = (held >= self.t_plus)

            if can_auto_sell:
                hit_sl = (price <= self._sl_price)
                hit_tp = (price >= self._tp_price)
                
                is_max_hold = (self.max_hold > 0 and held >= self.max_hold)
                unrealized = (price - self._entry_price) / self._entry_price
                hit_max_loss = is_max_hold and unrealized <= 0
                hit_max_profit = is_max_hold and unrealized > 0

                if hit_sl or hit_tp or hit_max_loss:
                    reason = ("sl" if hit_sl else "tp" if hit_tp else "maxhold")
                    auto_reward = self._execute_sell(price, held, reason=reason)
                    reward += auto_reward
                elif hit_max_profit:
                    # Đã quá hạn nhưng đang lãi -> không tự bán, phạt nhẹ để giục agent bán
                    reward -= 0.05

        # ── Over-trading penalty ───────────────────────────────────
        if action in (self.BUY, self.SELL):
            steps_since = self._step - self._last_trade_step
            if steps_since < 3 and steps_since > 0:
                reward -= 0.005

        # ── Clip reward để ổn định training ────────────────────────
        reward = float(np.clip(reward, -10.0, 10.0))

        # ── Update ────────────────────────────────────────────────
        total_cash = self._cash + sum(item["amount"] for item in self._cash_queue)
        equity = total_cash + self._shares * price
        self._equity.append(equity)
        self._rewards.append(reward)

        self._step += 1
        done = (self._step >= self.n)
        if done:
            if self._in_pos:  # Liquidate cuối episode
                lp    = self._round_tick(float(self.df_raw["close"].iloc[-1]))
                gross = self._shares * lp
                fee   = gross * self.tx_cost
                tax   = gross * self.sell_tax
                self._cash = self._cash + gross - fee - tax
                self._shares = 0.0; self._in_pos = False
                
            # Thanh toán toàn bộ tiền chờ về
            self._cash += sum(item["amount"] for item in self._cash_queue)
            self._cash_queue = []
            self._equity[-1] = self._cash
            self._done = True

        obs = self._obs() if not done else np.zeros(self.obs_size, np.float32)
        info = {"equity": equity, "step": self._step, "in_pos": self._in_pos}
        return obs, reward, done, info

    # ── Execute SELL logic (dùng chung cho manual + auto) ──────────
    def _execute_sell(self, price: float, held: int, reason: str = "manual") -> float:
        """Thực hiện lệnh bán và trả về reward."""
        exec_p = self._round_tick(price * (1 - self.slippage))
        gross  = self._shares * exec_p
        fee    = gross * self.tx_cost
        tax    = gross * self.sell_tax
        net_proceeds = gross - fee - tax

        # Đưa tiền bán vào hàng đợi T+2
        self._cash_queue.append({"amount": net_proceeds, "wait": 2})

        # PnL thực tế (đã bao gồm slippage mua + bán)
        pnl = (exec_p - self._entry_price) / self._entry_price

        # ── Reward = Risk-adjusted PnL ──────────────────────────
        reward = pnl * 100  # Base: PnL %

        if pnl > 0:
            # Bonus tỷ lệ với mức lãi, cap tại 3.0
            reward += min(pnl * 50, 3.0)
        else:
            # Phạt nhẹ khi thua — để agent không sợ giao dịch
            reward -= 0.5

        # Bonus/penalty theo exit reason
        if reason == "tp":
            reward += 0.5   # Chốt lời đúng kế hoạch
        elif reason == "sl":
            reward -= 0.3   # Cắt lỗ đúng kế hoạch (nhẹ hơn)
        # maxhold: không bonus/penalty thêm

        self.trades.append({
            "type":        "AUTO_EXIT" if reason != "manual" else "SELL",
            "reason":      reason,
            "step":        self._step,
            "date":        self._date(),
            "price":       round(exec_p, 2),
            "entry_price": round(self._entry_price, 2),
            "pnl_pct":     round(pnl * 100, 3),
            "hold_days":   held,
            "sl_price":    round(self._sl_price, 2),
            "tp_price":    round(self._tp_price, 2),
        })

        self._shares      = 0.0
        self._entry_price = 0.0
        self._entry_step  = 0
        self._sl_price    = 0.0
        self._tp_price    = 0.0
        self._in_pos      = False
        self._last_trade_step = self._step

        return reward

    # ── Helpers ────────────────────────────────────────────────────
    def _price(self, step: int | None = None) -> float:
        """Lấy giá close, enforce price limit ±7%. Dùng precomputed numpy array."""
        idx = step if step is not None else self._step
        idx = int(np.clip(idx, 0, self.n - 1))
        raw = float(self._close_arr[idx])
        # Apply price limit ±7% so với phiên trước
        if idx > 0 and self.price_limit > 0:
            prev = float(self._close_arr[idx - 1])
            raw  = float(np.clip(raw, prev * (1 - self.price_limit),
                                      prev * (1 + self.price_limit)))
        return raw

    def _round_tick(self, price: float) -> float:
        """Làm tròn giá theo bước giá chuẩn của HOSE."""
        # Nếu giá gốc lớn (VND thực tế)
        if price > 1000:
            if price < 10000: return round(price / 10) * 10
            elif price < 50000: return round(price / 50) * 50
            else: return round(price / 100) * 100
        # Nếu giá gốc đang ở đơn vị nghìn đồng (vd 55.4)
        else:
            if price < 10.0: return round(price / 0.01) * 0.01
            elif price < 50.0: return round(price / 0.05) * 0.05
            else: return round(price / 0.1) * 0.1

    def valid_actions(self) -> list[int]:
        """Trả về danh sách các action hợp lệ (Action Masking)."""
        acts = [self.HOLD]
        total_cash = self._cash + sum(item["amount"] for item in self._cash_queue)
        
        # Có thể mua nếu chưa có hàng và đủ tiền mua ít nhất 1 lô
        if not self._in_pos:
            price = self._price()
            est_cost = self.lot_size * price * (1 + self.slippage + self.tx_cost)
            if total_cash >= est_cost:
                acts.append(self.BUY)
        
        # Có thể bán nếu có hàng và đã đủ T+2
        if self._in_pos:
            held = self._step - self._entry_step
            if held >= self.t_plus:
                acts.append(self.SELL)
                
        return acts

    def get_analysis_embed(self) -> np.ndarray | None:
        """
        Lấy analysis embedding cho cửa sổ 20 ngày hiện tại.

        Ưu tiên:
        1. EmbeddingCache (pre-loaded in RAM) → O(1) dict lookup, zero I/O
        2. Fallback: gọi pipeline() (disk I/O mỗi step)

        Returns:
            numpy array (2560,) hoặc None nếu analysis không enabled.
        """
        if not self._analysis_enabled:
            return None
        assert self.analysis_model is not None and self.stock_id is not None

        # Xác định date_start và date_end cho window hiện tại
        safe = min(self._step, self.n - 1)
        start_idx = max(0, safe - self.window + 1)

        date_end = pd.Timestamp(self.df_raw["date"].iloc[safe])
        date_start = pd.Timestamp(self.df_raw["date"].iloc[start_idx])

        # ── Fast path: EmbeddingCache (zero I/O) ──────────────────
        if self._embedding_cache is not None:
            vector = self._embedding_cache.get(
                stock_id=self.stock_id,
                date_start=date_start.to_pydatetime(),
                date_end=date_end.to_pydatetime(),
            )
            if vector is not None:
                return vector.astype(np.float32)
            # Cache miss — fall through to pipeline

        # ── Slow path: pipeline() (disk I/O) ─────────────────────
        try:
            from stock_analysis import pipeline

            result = pipeline(
                model=self.analysis_model,
                stock_id=self.stock_id,
                date_start=date_start.to_pydatetime(),
                date_end=date_end.to_pydatetime(),
            )
            return result["vector"].astype(np.float32)

        except Exception as e:
            log.warning(f"[Analysis] Failed to get embedding at step {self._step}: {e}")
            return None

    def _date(self) -> str:
        i = min(self._step, self.n - 1)
        return str(self.df_raw["date"].iloc[i])[:10]

    def _get_obs_fast(self, step: int) -> np.ndarray:
        """Fast observation: numpy slice from precomputed matrix (zero-copy view)."""
        start = max(0, step - self.window + 1)
        mat = self._obs_matrix[start:step + 1]
        if mat.shape[0] < self.window:
            pad = np.zeros((self.window - mat.shape[0], mat.shape[1]), dtype=np.float32)
            mat = np.vstack([pad, mat])
        return mat.ravel()

    def _obs(self) -> np.ndarray:
        safe = min(self._step, self.n - 1)
        feat = self._get_obs_fast(safe)

        price  = self._price()
        unreal = (price - self._entry_price) / self._entry_price if self._in_pos else 0.0
        total_cash = self._cash + sum(item["amount"] for item in self._cash_queue)
        total  = total_cash + self._shares * price
        cash_r = total_cash / (total + 1e-9)
        held   = self._step - self._entry_step if self._in_pos else 0
        held_r = min(held / self.max_hold, 1.0) if self._in_pos and self.max_hold > 0 else 0.0

        # T+2 availability: 1.0 nếu hàng đã về, 0.0 nếu chưa
        t_plus_avail = 1.0 if (self._in_pos and held >= self.t_plus) else 0.0

        # ATR% hiện tại
        atr_pct = self._get_atr_pct()

        # Khoảng cách % tới mức SL (âm = đang gần SL)
        if self._in_pos and self._sl_price > 0:
            dist_to_sl = (price - self._sl_price) / (price + 1e-9)
            dist_to_sl = float(np.clip(dist_to_sl, -0.2, 0.2))
        else:
            dist_to_sl = 0.0

        port = np.array([
            float(self._in_pos),                          # 1. Có vị thế?
            float(np.clip(unreal, -0.3, 0.3)),           # 2. PnL tạm tính
            float(cash_r),                                # 3. Tỷ lệ tiền mặt
            float(held_r),                                # 4. % thời gian giữ
            float(t_plus_avail),                          # 5. T+2 đã về?
            float(np.clip(atr_pct, 0.0, 0.15)),         # 6. ATR% (clip 0~15%)
            float(dist_to_sl),                            # 7. Khoảng cách tới SL
        ], dtype=np.float32)
        return np.concatenate([feat, port])

    # ── Episode metrics ────────────────────────────────────────────
    def metrics(self) -> dict:
        eq  = np.array(self._equity, dtype=np.float64)
        if len(eq) < 2:
            return self._zero_metrics()

        T   = len(eq) - 1
        ret = (eq[-1] - eq[0]) / eq[0]
        arr = ret * (252 / T)

        rets   = np.diff(eq) / (eq[:-1] + 1e-9)
        std_r  = rets.std()
        sharpe = float((rets.mean() - 0.04/252) / (std_r + 1e-9) * np.sqrt(252))
        sharpe = float(np.clip(sharpe, -10, 10))

        peak   = np.maximum.accumulate(eq)
        mdd    = float(((eq - peak) / (peak + 1e-9)).min())

        closed = [t for t in self.trades if "pnl_pct" in t]
        pnls   = [t["pnl_pct"] for t in closed]
        wins   = [p for p in pnls if p > 0]
        loss_l = [p for p in pnls if p <= 0]

        return {
            "return_pct":   round(float(ret * 100), 2),
            "arr_pct":      round(float(arr * 100), 2),
            "sharpe":       round(sharpe, 4),
            "max_dd_pct":   round(float(mdd * 100), 2),
            "n_trades":     len(closed),
            "win_rate":     round(len(wins) / max(len(closed), 1) * 100, 1),
            "avg_win":      round(float(np.mean(wins)),   2) if wins   else 0.0,
            "avg_loss":     round(float(np.mean(loss_l)), 2) if loss_l else 0.0,
            "pf":           round(sum(wins) / (abs(sum(loss_l)) + 1e-9), 2),
            "final_equity": round(float(eq[-1])),
            "n_sl":  sum(1 for t in closed if t.get("reason") == "sl"),
            "n_tp":  sum(1 for t in closed if t.get("reason") == "tp"),
            "n_mh":  sum(1 for t in closed if t.get("reason") == "maxhold"),
            "steps": T,
        }

    def _zero_metrics(self) -> dict:
        return {k: 0 for k in ["return_pct","arr_pct","sharpe","max_dd_pct",
                                 "n_trades","win_rate","avg_win","avg_loss","pf",
                                 "final_equity","n_sl","n_tp","n_mh","steps"]}

    @property
    def equity_series(self) -> np.ndarray:
        return np.array(self._equity)
