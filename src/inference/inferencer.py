"""
src/inference/inferencer.py
════════════════════════════════════════════════════════════════════════════
Inferencer — Load a trained model and predict trading actions.

Usage:
    from src.config import Config
    from src.inference import Inferencer

    cfg = Config.load("configs/")
    infer = Inferencer(cfg, model_path="models/best_model.pkl")
    result = infer.predict("data/VNM.csv", last_n=20)
    print(result)
    # {"action": "BUY", "confidence": 0.85, "q_values": [0.1, 0.85, 0.3]}
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import Config
from src.features.preprocessor import load_csv, RobustScaler, obs_size_of
from src.rl.env.trading_env import TradingEnv
from src.rl.agent.dqn_agent import DQNAgent

log = logging.getLogger(__name__)

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


class Inferencer:
    """
    Load a trained DQN agent and predict actions for given market data.

    Attributes:
        agent: Trained DQNAgent (greedy mode)
        scaler: RobustScaler fitted during training
        cfg: Config object
    """

    def __init__(self, cfg: Config, model_path: str = "models/best_model.pkl",
                 scaler_path: str = "models/scaler.pkl"):
        self.cfg = cfg
        self.scaler = self._load_scaler(scaler_path)
        self.agent = self._load_agent(model_path)
        self.agent.eps = 0.0  # Greedy inference

    def _load_scaler(self, path: str) -> RobustScaler:
        if not Path(path).exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_agent(self, path: str) -> DQNAgent:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")

        window = self.cfg.env["window"]
        ac = self.cfg.agent
        analysis_cfg = self.cfg.analysis
        analysis_enabled = analysis_cfg.get("enabled", False)

        # We need obs_size — create a dummy to compute it
        # For now, use a placeholder; actual obs_size comes from scaler
        agent = DQNAgent(
            obs_size=1,  # placeholder, will be overwritten by load
            n_actions=3,
            hidden=ac["hidden"],
            lr=ac["lr"],
            gamma=ac["gamma"],
            tau=ac.get("tau", 0.01),
            eps=0.0,
            eps_end=ac["eps_end"],
            eps_decay=ac["eps_decay"],
            buffer_cap=ac["buffer_cap"],
            batch_size=ac["batch_size"],
            warmup=ac["warmup"],
            analysis_embed_dim=analysis_cfg.get("embed_dim") if analysis_enabled else None,
            analysis_proj_layers=analysis_cfg.get("projection") if analysis_enabled else None,
        )
        agent.load(path)
        return agent

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def predict(self, csv_path: str, last_n: int | None = None) -> dict:
        """
        Predict action for the most recent window of data.

        Args:
            csv_path: Path to stock CSV file
            last_n: Use only last N rows (default: use all data)

        Returns:
            dict with keys: action, action_name, confidence, q_values
        """
        df_raw = load_csv(csv_path)
        if last_n:
            df_raw = df_raw.tail(last_n).reset_index(drop=True)

        df_norm = self.scaler.transform(df_raw)
        env_cfg = self.cfg.env
        window = env_cfg["window"]

        if len(df_raw) < window + 1:
            raise ValueError(f"Need at least {window + 1} rows, got {len(df_raw)}")

        # Create env and step to the last position
        env = TradingEnv(
            df_raw=df_raw, df_norm=df_norm,
            window=window,
            init_cap=env_cfg["initial_cap"],
            tx_cost=env_cfg.get("tx_cost", 0.0015),
            sell_tax=env_cfg.get("sell_tax", 0.001),
            slippage=env_cfg.get("slippage", 0.0003),
            atr_sl_mult=env_cfg.get("atr_sl_mult", 1.5),
            atr_tp_mult=env_cfg.get("atr_tp_mult", 3.0),
            risk_per_trade=env_cfg.get("risk_per_trade", 0.02),
            stop_loss=env_cfg["stop_loss"],
            take_profit=env_cfg["take_profit"],
            max_hold=env_cfg.get("max_hold", 60),
            t_plus=env_cfg.get("t_plus", 2),
            lot_size=env_cfg.get("lot_size", 100),
            price_limit=env_cfg.get("price_limit", 0.07),
        )

        obs = env.reset()
        # Step through to the end to get final observation
        done = False
        while not done:
            action = self.agent.act(obs, valid_actions=env.valid_actions(), greedy=True)
            next_obs, _, done, _ = env.step(action)
            if not done:
                obs = next_obs

        # Get Q-values for the last observation
        q_values = self.agent.get_q_values(obs)
        valid = env.valid_actions()
        best_action = int(np.argmax([q_values[a] if a in valid else -np.inf for a in range(3)]))

        # Confidence = softmax of Q-values
        q_valid = np.array([q_values[a] for a in valid])
        exp_q = np.exp(q_valid - q_valid.max())
        probs = exp_q / exp_q.sum()
        confidence = float(probs.max())

        return {
            "action": best_action,
            "action_name": ACTION_NAMES[best_action],
            "confidence": round(confidence, 4),
            "q_values": [round(float(q), 4) for q in q_values],
            "valid_actions": valid,
            "latest_date": str(df_raw["date"].iloc[-1].date()),
            "latest_close": float(df_raw["close"].iloc[-1]),
        }

    def predict_batch(self, stock_paths: dict[str, str]) -> dict[str, dict]:
        """
        Predict actions for multiple stocks.

        Args:
            stock_paths: {"VNM": "data/VNM.csv", "FPT": "data/FPT.csv", ...}

        Returns:
            {"VNM": {action, confidence, ...}, "FPT": {...}, ...}
        """
        results = {}
        for symbol, path in stock_paths.items():
            try:
                results[symbol] = self.predict(path)
            except Exception as e:
                log.error(f"[{symbol}] Inference failed: {e}")
                results[symbol] = {"error": str(e)}
        return results
