"""
src/ui/simulator.py
════════════════════════════════════════════════════════════════════════════
TradingSimulator — Quản lý state cho 1 mã cổ phiếu, step-by-step trên test-set.

Mỗi lần gọi next_day():
  1. Agent chọn action cho ngày hiện tại
  2. Env.step(action) → cập nhật portfolio
  3. Agent predict cho ngày tiếp theo (peek)
  4. Lookup fundamental response (.md) cho window hiện tại
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.rl.env.trading_env import TradingEnv
from src.rl.agent.dqn_agent import DQNAgent

log = logging.getLogger(__name__)

ACTION_NAMES = {0: "HOLD 🔄", 1: "BUY 🟢", 2: "SELL 🔴"}
ACTION_NAMES_SHORT = {0: "HOLD", 1: "BUY", 2: "SELL"}

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class TradingSimulator:
    """Step-by-step trading simulator for a single stock on test-set."""

    def __init__(
        self,
        stock_id: str,
        cfg: Config,
        agent: DQNAgent,
        test_raw: pd.DataFrame,
        test_norm: pd.DataFrame,
        embedding_cache=None,
    ):
        self.stock_id = stock_id
        self.cfg = cfg
        self.agent = agent
        self.embedding_cache = embedding_cache

        self.test_raw = test_raw.reset_index(drop=True)
        self.test_norm = test_norm.reset_index(drop=True)

        # Create environment
        env_cfg = cfg.env
        self.env = TradingEnv(
            df_raw=self.test_raw,
            df_norm=self.test_norm,
            window=env_cfg["window"],
            init_cap=env_cfg["initial_cap"],
            tx_cost=env_cfg.get("tx_cost", 0.0015),
            sell_tax=env_cfg.get("sell_tax", 0.001),
            slippage=env_cfg.get("slippage", 0.001),
            atr_sl_mult=env_cfg.get("atr_sl_mult", 1.5),
            atr_tp_mult=env_cfg.get("atr_tp_mult", 3.0),
            risk_per_trade=env_cfg.get("risk_per_trade", 0.02),
            stop_loss=env_cfg["stop_loss"],
            take_profit=env_cfg["take_profit"],
            max_hold=env_cfg.get("max_hold", 30),
            t_plus=env_cfg.get("t_plus", 2),
            lot_size=env_cfg.get("lot_size", 100),
            price_limit=env_cfg.get("price_limit", 0.07),
            stock_id=stock_id,
            analysis_model=cfg.analysis.get("model") if cfg.analysis.get("enabled") else None,
            embedding_cache=embedding_cache,
        )

        # State
        self.obs = self.env.reset()
        self.current_step = self.env._step  # = window
        self.done = False
        self.history: list[dict] = []  # [{step, date, action, price, equity, ...}]
        self.total_days = len(self.test_raw) - env_cfg["window"]

        # Fundamental cache paths
        self._reports_log = self._load_json(
            _PROJECT_ROOT / "artifacts" / "embeddings" / stock_id / "reports" / "logs.json"
        )
        self._responses_log = self._load_json(
            _PROJECT_ROOT / "artifacts" / "embeddings" / stock_id / "responses" / "logs.json"
        )
        # Build reverse map: report_hash → response_hash
        self._report_to_response: dict[str, str] = {}
        for resp_hash, info in self._responses_log.items():
            rpt_hash = info.get("report_hash_id", "")
            if rpt_hash:
                self._report_to_response[rpt_hash] = resp_hash

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    # ─────────────────────────────────────────────────────────────────
    # Core: advance one day
    # ─────────────────────────────────────────────────────────────────

    def next_day(self) -> dict:
        """
        Advance simulation by one day.

        Returns dict with:
            - action: int (0/1/2)
            - action_name: str
            - date: str
            - price: float
            - equity: float
            - pnl_pct: float
            - in_position: bool
            - trades: list[dict]
            - next_prediction: dict (action, confidence, q_values for tomorrow)
            - fundamental_md: str (LLM response markdown)
            - day_index: int (current day in test-set)
            - total_days: int
            - done: bool
        """
        if self.done:
            return self._make_result(action=0, action_name="DONE", next_pred=None)

        # 1. Agent chooses action for current observation
        valid = self.env.valid_actions()
        action = self.agent.act(self.obs, valid_actions=valid, greedy=True)

        # 2. Step environment
        next_obs, reward, done, info = self.env.step(action)

        # Record history
        step_idx = min(self.env._step - 1, len(self.test_raw) - 1)
        date_str = str(self.test_raw["date"].iloc[step_idx])[:10]
        price = float(self.test_raw["close"].iloc[step_idx])
        equity = info.get("equity", self.cfg.env["initial_cap"])

        self.history.append({
            "step": step_idx,
            "date": date_str,
            "action": action,
            "action_name": ACTION_NAMES_SHORT[action],
            "price": price,
            "equity": equity,
            "in_pos": info.get("in_pos", False),
        })

        self.done = done
        self.obs = next_obs
        self.current_step = self.env._step

        # 3. Predict next day action (peek ahead)
        next_pred = None
        if not done:
            next_pred = self._predict_next(next_obs)

        # 4. Lookup fundamental
        fundamental_md = self._get_fundamental_md()

        return self._make_result(
            action=action,
            action_name=ACTION_NAMES[action],
            next_pred=next_pred,
            fundamental_md=fundamental_md,
        )

    def _make_result(self, action: int, action_name: str,
                     next_pred: dict | None, fundamental_md: str = "") -> dict:
        """Build result dict from current state."""
        equity = self.history[-1]["equity"] if self.history else self.cfg.env["initial_cap"]
        init_cap = self.cfg.env["initial_cap"]
        pnl_pct = (equity - init_cap) / init_cap * 100

        return {
            "action": action,
            "action_name": action_name,
            "date": self.history[-1]["date"] if self.history else "",
            "price": self.history[-1]["price"] if self.history else 0.0,
            "equity": equity,
            "pnl_pct": round(pnl_pct, 2),
            "in_position": self.env._in_pos,
            "trades": self.env.trades,
            "next_prediction": next_pred,
            "fundamental_md": fundamental_md,
            "day_index": len(self.history),
            "total_days": self.total_days,
            "done": self.done,
        }

    def _predict_next(self, obs: np.ndarray) -> dict:
        """Predict agent's action for the next day (without stepping)."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.q(obs_t).cpu().numpy()[0]

        valid = self.env.valid_actions()
        # Mask invalid actions
        masked_q = np.array([
            q_values[a] if a in valid else -np.inf for a in range(3)
        ])
        best_action = int(np.argmax(masked_q))

        # Confidence via softmax over valid Q-values
        q_valid = np.array([q_values[a] for a in valid])
        exp_q = np.exp(q_valid - q_valid.max())
        probs = exp_q / exp_q.sum()
        confidence = float(probs.max())

        return {
            "action": best_action,
            "action_name": ACTION_NAMES[best_action],
            "confidence": round(confidence * 100, 1),
            "q_values": {ACTION_NAMES_SHORT[i]: round(float(q_values[i]), 4) for i in range(3)},
            "valid_actions": [ACTION_NAMES_SHORT[a] for a in valid],
        }

    # ─────────────────────────────────────────────────────────────────
    # Fundamental lookup
    # ─────────────────────────────────────────────────────────────────

    def _get_fundamental_md(self) -> str:
        """Get LLM response markdown for current window's date range."""
        window = self.cfg.env["window"]
        safe_step = min(self.env._step - 1, len(self.test_raw) - 1)
        start_idx = max(0, safe_step - window + 1)

        date_end = pd.Timestamp(self.test_raw["date"].iloc[safe_step])
        date_start = pd.Timestamp(self.test_raw["date"].iloc[start_idx])

        # Compute date_hash (same logic as Report.get_hash_id / EmbeddingCache)
        content = f"{self.stock_id}::{date_start.to_pydatetime().isoformat()}::{date_end.to_pydatetime().isoformat()}"
        date_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # date_hash → content_hash (report_hash)
        report_info = self._reports_log.get(date_hash)
        if report_info is None:
            return "*Không có dữ liệu phân tích cơ bản cho giai đoạn này.*"

        content_hash = report_info.get("content_hash", date_hash)

        # content_hash → response_hash
        response_hash = self._report_to_response.get(content_hash)
        if response_hash is None:
            return "*Chưa có phân tích LLM cho báo cáo này.*"

        # Read .md file
        md_path = _PROJECT_ROOT / "artifacts" / "embeddings" / self.stock_id / "responses" / f"{response_hash}.md"
        if md_path.exists():
            try:
                return md_path.read_text(encoding="utf-8")
            except OSError:
                return "*Lỗi đọc file phân tích.*"

        return "*File phân tích không tồn tại.*"

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    def get_portfolio_summary(self) -> str:
        """Format portfolio summary as text."""
        if not self.history:
            return "Chưa bắt đầu giao dịch."

        equity = self.history[-1]["equity"]
        init_cap = self.cfg.env["initial_cap"]
        pnl_pct = (equity - init_cap) / init_cap * 100

        closed_trades = [t for t in self.env.trades if "pnl_pct" in t]
        n_trades = len(closed_trades)
        wins = [t for t in closed_trades if t["pnl_pct"] > 0]
        win_rate = len(wins) / max(n_trades, 1) * 100

        lines = [
            f"📊 **Portfolio — {self.stock_id}**",
            f"",
            f"| Chỉ số | Giá trị |",
            f"|--------|---------|",
            f"| Vốn ban đầu | {init_cap/1e6:,.0f}M VND |",
            f"| Equity hiện tại | {equity/1e6:,.0f}M VND |",
            f"| PnL | {pnl_pct:+.2f}% |",
            f"| Số lệnh đã đóng | {n_trades} |",
            f"| Win Rate | {win_rate:.1f}% |",
            f"| Đang giữ vị thế | {'✅ Có' if self.env._in_pos else '❌ Không'} |",
            f"| Ngày | {self.history[-1]['date']} |",
            f"| Tiến độ | {len(self.history)}/{self.total_days} ngày |",
        ]
        return "\n".join(lines)

    def get_buy_sell_data(self) -> tuple[list[dict], list[dict]]:
        """Get buy/sell points for chart markers."""
        buys = []
        sells = []
        for t in self.env.trades:
            step = t["step"]
            if step >= len(self.test_raw):
                continue
            entry = {
                "date": self.test_raw["date"].iloc[step],
                "price": t["price"],
                "step": step,
            }
            if t["type"] == "BUY":
                buys.append(entry)
            elif t["type"] in ("SELL", "AUTO_EXIT"):
                sells.append(entry)
        return buys, sells

    def reset(self):
        """Reset simulator to initial state."""
        self.obs = self.env.reset()
        self.current_step = self.env._step
        self.done = False
        self.history = []
