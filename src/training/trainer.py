"""
src/training/trainer.py
════════════════════════════════════════════════════════════════════════════
Trainer — OOP wrapper for the training loop.

Usage:
    from src.config import Config
    from src.training import Trainer

    cfg = Config.load("configs/")
    trainer = Trainer(cfg)
    trainer.run()
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json, logging, os, pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from tqdm.auto import tqdm

from src.config import Config
from src.features.preprocessor import load_csv, time_split, RobustScaler, obs_size_of
from src.rl.env.trading_env import TradingEnv
from src.rl.agent.dqn_agent import DQNAgent
from stock_analysis.cache import EmbeddingCache

log = logging.getLogger(__name__)


class Trainer:
    """
    Encapsulates the full training pipeline:
      1. Data loading & splitting
      2. Feature scaling
      3. Agent creation
      4. Training loop with curriculum learning
      5. Checkpointing & early stopping
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._setup_dirs()
        self._setup_data()
        self._setup_agent()
        self._setup_cache()

    # ─────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────

    def _setup_dirs(self):
        self.model_dir = self.cfg.output.get("model_dir", "models")
        self.log_dir = self.cfg.output.get("log_dir", "logs")
        self.chart_dir = self.cfg.output.get("chart_dir", "outputs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chart_dir, exist_ok=True)

    def _setup_data(self):
        """Load CSVs, split, scale."""
        paths = self.cfg.data.get("paths", [])
        if not paths:
            raise ValueError("No data paths defined in config!")

        self.paths = paths
        self.stock_ids = [Path(p).stem for p in paths]
        sp = self.cfg.split

        self.train_raws, self.val_raws, self.test_raws = [], [], []
        for p in paths:
            raw_df = load_csv(p)
            tr, va, te = time_split(raw_df, sp["train_ratio"], sp["val_ratio"])
            self.train_raws.append(tr)
            self.val_raws.append(va)
            self.test_raws.append(te)

        # Scale
        combined_tr = pd.concat(self.train_raws, ignore_index=True)
        self.scaler = RobustScaler().fit(combined_tr)
        self.train_norms = [self.scaler.transform(df) for df in self.train_raws]
        self.val_norms = [self.scaler.transform(df) for df in self.val_raws]
        self.test_norms = [self.scaler.transform(df) for df in self.test_raws]

        with open(f"{self.model_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        log.info(f"[Data] {len(paths)} stocks loaded: {self.stock_ids}")

    def _setup_agent(self):
        """Create DQNAgent from config."""
        window = self.cfg.env["window"]
        obs_sz = obs_size_of(self.train_norms[0], window)
        ac = self.cfg.agent
        analysis_cfg = self.cfg.analysis
        analysis_enabled = analysis_cfg.get("enabled", False)

        self.window = window
        self.obs_sz = obs_sz
        self.analysis_enabled = analysis_enabled
        self.analysis_model = analysis_cfg.get("model") if analysis_enabled else None

        self.agent = DQNAgent(
            obs_size=obs_sz,
            n_actions=3,
            hidden=ac["hidden"],
            lr=ac["lr"],
            lr_decay=ac.get("lr_decay", 0.995),
            lr_min=ac.get("lr_min", 5e-5),
            gamma=ac["gamma"],
            tau=ac.get("tau", 0.01),
            eps=ac.get("eps_start", 1.0),
            eps_end=ac["eps_end"],
            eps_decay=ac["eps_decay"],
            buffer_cap=ac["buffer_cap"],
            batch_size=ac["batch_size"],
            warmup=ac["warmup"],
            analysis_embed_dim=analysis_cfg.get("embed_dim") if analysis_enabled else None,
            analysis_proj_layers=analysis_cfg.get("projection") if analysis_enabled else None,
        )
        log.info(f"[Agent] obs_size={obs_sz} | window={window} | type={ac.get('type', 'dqn')}")

    def _setup_cache(self):
        """Pre-load embedding cache if enabled."""
        self.embedding_cache = None
        training_cfg = self.cfg.training
        if self.analysis_enabled and training_cfg.get("preload_embeddings", True):
            self.embedding_cache = EmbeddingCache(self.stock_ids)
            log.info(f"[EmbeddingCache] {self.embedding_cache.stats}")

    # ─────────────────────────────────────────────────────────────────
    # Episode runners
    # ─────────────────────────────────────────────────────────────────

    def _run_episode(self, df_raw, df_norm, train_mode: bool,
                     episode_num: int = 1, stock_id: str | None = None) -> tuple[dict, list]:
        """Run a single episode."""
        env_cfg = self.cfg.env
        learn_every = self.cfg.training.get("learn_every", 4)

        try:
            env = TradingEnv(
                df_raw=df_raw, df_norm=df_norm,
                window=env_cfg["window"],
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
                stock_id=stock_id,
                analysis_model=self.analysis_model,
                embedding_cache=self.embedding_cache,
            )

            obs = env.reset()
            done = False
            step_count = 0
            total_steps = env.n - env.window
            cum_reward = 0.0

            ep_label = f"Ep{episode_num}"
            if stock_id:
                ep_label += f" ({stock_id})"
            step_bar = tqdm(total=total_steps, desc=f"  {ep_label}",
                            unit="step", leave=False, dynamic_ncols=True)

            while not done:
                try:
                    analysis_embed = env.get_analysis_embed()
                    action = self.agent.act(obs, valid_actions=env.valid_actions(),
                                            greedy=not train_mode, analysis_embed=analysis_embed)
                    next_obs, reward, done, info = env.step(action)
                    cum_reward += reward

                    if train_mode:
                        self.agent.store(obs, action, reward, next_obs, done,
                                         analysis_embed=analysis_embed)
                        if step_count % learn_every == 0:
                            self.agent.learn()

                    obs = next_obs
                    step_count += 1
                    step_bar.update(1)
                    if step_count % 50 == 0:
                        step_bar.set_postfix_str(f"R={cum_reward:+.1f}")

                except Exception as e:
                    log.error(f"Step error at ep {episode_num}, step {step_count}: {e}")
                    done = True
                    break

            step_bar.close()
            m = env.metrics()
            m["episode_steps"] = step_count
            return m, env.trades

        except Exception as e:
            log.error(f"Episode {episode_num} failed: {e}")
            return self._empty_metrics(), []

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
            "n_trades": 0, "win_rate": 0.0, "episode_steps": 0,
            "arr_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "pf": 0.0, "final_equity": 0, "n_sl": 0, "n_tp": 0,
            "n_mh": 0, "steps": 0,
        }

    # ─────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────

    def run(self, n_episodes: int | None = None, resume_from: str | None = None):
        """Execute the full training loop."""
        np.random.seed(self.cfg.project.get("seed", 42))
        training_cfg = self.cfg.training
        n_ep = n_episodes or training_cfg.get("n_episodes", 500)
        patience = training_cfg.get("patience", 80)
        val_every = training_cfg.get("val_every", 5)
        ckpt_every = training_cfg.get("checkpoint_every", 50)

        # Resume
        start_ep = 1
        if resume_from and Path(resume_from).exists():
            self.agent.load(resume_from)
            start_ep = getattr(self.agent, "episode_num", 0) + 1
            log.info(f"[Resume] from episode {start_ep}")

        # State
        history: list[dict] = []
        best_val = -float("inf")
        best_ep = 0
        pat_cnt = 0
        start_t = time.time()
        learned_once = False
        val_history: deque = deque(maxlen=10)
        vl_m = self._empty_metrics()

        log.info(f"Training: {n_ep} episodes | patience={patience} | val_every={val_every}")

        pbar = tqdm(range(start_ep, n_ep + 1), desc="Training", unit="ep",
                    dynamic_ncols=True, initial=start_ep - 1, total=n_ep)

        for ep in pbar:
            # ── Curriculum: select stock ──────────────────────────────
            stock_idx = self._curriculum_select(ep, n_ep)
            stock_id = self.stock_ids[stock_idx] if self.analysis_enabled else None

            # ── Train episode ─────────────────────────────────────────
            tr_m, _ = self._run_episode(
                self.train_raws[stock_idx], self.train_norms[stock_idx],
                train_mode=True, episode_num=ep, stock_id=stock_id
            )

            # ── Validation (every N episodes) ─────────────────────────
            if ep % val_every == 0 or ep == start_ep or ep == n_ep:
                vl_m = self._validate()

            # ── Decay ─────────────────────────────────────────────────
            self.agent.decay_epsilon()
            self.agent.decay_lr()

            avg_loss = 0.0
            if self.agent.losses:
                avg_loss = float(np.mean(list(self.agent.losses)[-200:]))
                learned_once = True

            # ── Record ────────────────────────────────────────────────
            rec = {
                "episode": ep,
                "train_return": round(float(tr_m["return_pct"]), 2),
                "val_return": round(float(vl_m["return_pct"]), 2),
                "val_sharpe": round(float(vl_m["sharpe"]), 4),
                "val_trades": int(vl_m["n_trades"]),
                "val_winrate": round(float(vl_m.get("win_rate", 0)), 1),
                "epsilon": round(float(self.agent.eps), 5),
                "avg_loss": round(float(avg_loss), 6),
            }
            history.append(rec)

            # ── Progress bar ──────────────────────────────────────────
            pbar.set_postfix_str(
                f"TR={rec['train_return']:+.1f}% VR={rec['val_return']:+.1f}% "
                f"Sh={rec['val_sharpe']:.2f} WR={rec['val_winrate']:.0f}% "
                f"ε={self.agent.eps:.3f} loss={avg_loss:.4f}"
            )

            # ── Checkpointing ─────────────────────────────────────────
            if learned_once:
                val_history.append(vl_m)
                score = self._score(vl_m)
                if vl_m["n_trades"] > 0 and score > best_val:
                    best_val = score
                    best_ep = ep
                    pat_cnt = 0
                    self.agent.save(f"{self.model_dir}/best_model.pkl")
                    log.info(f"  → New best (score={score:.4f}) @ ep {ep}")
                else:
                    pat_cnt += 1

            if ep % ckpt_every == 0:
                self.agent.save(f"{self.model_dir}/ckpt_ep{ep}.pkl")

            # ── Early stopping ────────────────────────────────────────
            if learned_once and pat_cnt >= patience and self.agent.eps < 0.3:
                log.info(f"Early stop @ ep {ep}")
                break

        # ── Save final ────────────────────────────────────────────────
        self.agent.save(f"{self.model_dir}/last_model.pkl")
        with open(f"{self.log_dir}/training_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        elapsed = time.time() - start_t
        log.info(f"Training done: {elapsed/60:.1f}min | best @ ep {best_ep} (score={best_val:.4f})")
        return history

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _validate(self) -> dict:
        """Run validation on all stocks, return averaged metrics."""
        metrics_list = []
        for idx in range(len(self.val_raws)):
            stock_id = self.stock_ids[idx] if self.analysis_enabled else None
            m, _ = self._run_episode(
                self.val_raws[idx], self.val_norms[idx],
                train_mode=False, stock_id=stock_id
            )
            metrics_list.append(m)

        return {
            "return_pct": np.mean([m["return_pct"] for m in metrics_list]),
            "sharpe": np.mean([m["sharpe"] for m in metrics_list]),
            "max_dd_pct": np.mean([m["max_dd_pct"] for m in metrics_list]),
            "n_trades": int(np.mean([m["n_trades"] for m in metrics_list])),
            "win_rate": np.mean([m.get("win_rate", 0) for m in metrics_list]),
            "episode_steps": int(np.mean([m.get("episode_steps", 0) for m in metrics_list])),
        }

    def _curriculum_select(self, ep: int, n_ep: int) -> int:
        """Curriculum learning: gradually introduce more stocks."""
        progress = ep / n_ep
        n_stocks = len(self.train_raws)
        if progress < 0.2:
            return 0
        elif progress < 0.4:
            return np.random.randint(min(n_stocks, 2))
        elif progress < 0.6:
            return np.random.randint(min(n_stocks, 3))
        else:
            return np.random.randint(n_stocks)

    @staticmethod
    def _score(metrics: dict) -> float:
        """Normalized score for model selection."""
        ret = np.clip((metrics.get("return_pct", 0) + 10) / 30, 0, 1)
        sharpe = np.clip((metrics.get("sharpe", 0) + 1) / 4, 0, 1)
        trades = np.clip(metrics.get("n_trades", 0) / 50, 0, 1)
        wr = np.clip(metrics.get("win_rate", 0) / 100, 0, 1)
        return float(ret * 0.35 + sharpe * 0.30 + trades * 0.15 + wr * 0.20)
