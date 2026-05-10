"""
train.py  —  VNM RL Trading Pipeline v6.0
════════════════════════════════════════════════════════════════════════════

  Tuân thủ HOSE: T+2, lot 100cp, thuế bán 0.1%, giới hạn ±7%
  Double DQN với per-episode epsilon decay
  Reward = PnL % khi SELL, clip [-10, 10]
  Observation = 20 bar × 27 features (normalized) + 4 portfolio state

Chạy:
    python train.py
    python train.py --episodes 500
    python train.py --mode charts     # tạo chart từ kết quả đã train
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse, json, logging, os, pickle, sys, time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="w", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(__file__))

from src.features.preprocessor import load_csv, time_split, RobustScaler, obs_size_of
from src.rl.env.trading_env import TradingEnv
from src.rl.agent.dqn_agent import DQNAgent


def load_cfg(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────

def run_episode(agent: DQNAgent,
                df_raw: pd.DataFrame,
                df_norm: pd.DataFrame,
                env_cfg: dict,
                train_mode: bool) -> tuple[dict, list]:
    """
    Chạy 1 episode hoàn chỉnh.
    Đảm bảo:
      - env.reset() → obs hợp lệ
      - loop chạy cho đến khi done=True
      - nếu train_mode: store + learn mỗi bước
    """
    env = TradingEnv(
        df_raw   = df_raw,
        df_norm  = df_norm,
        window       = env_cfg["window"],
        init_cap     = env_cfg["initial_cap"],
        tx_cost      = env_cfg.get("tx_cost", 0.0015),
        sell_tax     = env_cfg.get("sell_tax", 0.001),
        slippage     = env_cfg.get("slippage", 0.0003),
        stop_loss    = env_cfg["stop_loss"],
        take_profit  = env_cfg["take_profit"],
        max_hold     = env_cfg.get("max_hold", 60),
        t_plus       = env_cfg.get("t_plus", 2),
        lot_size     = env_cfg.get("lot_size", 100),
        price_limit  = env_cfg.get("price_limit", 0.07),
        cooldown     = env_cfg.get("cooldown", 5),
        trade_penalty = env_cfg.get("trade_penalty", 0.3),
    )

    obs  = env.reset()
    done = False
    step_count = 0

    while not done:
        action = agent.act(obs, greedy=not train_mode)
        next_obs, reward, done, info = env.step(action)

        if train_mode:
            agent.store(obs, action, reward, next_obs, done)
            agent.learn()   # learn() nội bộ đã có warmup guard

        obs = next_obs
        step_count += 1

    m = env.metrics()
    m["episode_steps"] = step_count
    return m, env.trades


# ─────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────

def train(cfg: dict, n_ep_override: int | None = None):
    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["log_dir"],   exist_ok=True)
    os.makedirs(cfg["output"]["chart_dir"], exist_ok=True)

    np.random.seed(cfg["project"].get("seed", 42))

    log.info("=" * 65)
    log.info("  VNM RL Trading v6.0 — Double DQN (HOSE compliant)")
    log.info("  T+2 | Lot 100cp | Thuế 0.1% | ±7% price limit")
    log.info("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────────────
    raw_df = load_csv(cfg["data"]["path"])
    sp = cfg["split"]
    tr_raw, va_raw, te_raw = time_split(raw_df, sp["train_ratio"], sp["val_ratio"])

    # ── 2. Scale features ────────────────────────────────────────────
    log.info("\n[Scale] RobustScaler (fit on train)...")
    scaler = RobustScaler().fit(tr_raw)
    tr_norm = scaler.transform(tr_raw)
    va_norm = scaler.transform(va_raw)
    te_norm = scaler.transform(te_raw)

    model_dir = cfg["output"]["model_dir"]
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── 3. Agent setup ────────────────────────────────────────────────
    window  = cfg["env"]["window"]
    obs_sz  = obs_size_of(tr_norm, window)
    ac      = cfg["agent"]
    env_cfg = cfg["env"]

    log.info(f"[Agent] obs_size={obs_sz} | window={window}")

    # Validate: episode_len phải > 0 cho tất cả splits
    for name, df_ in [("train", tr_raw), ("val", va_raw), ("test", te_raw)]:
        ep_len = len(df_) - window
        assert ep_len > 0, f"{name} df quá ngắn: {len(df_)} rows, window={window}"
        log.info(f"  {name}: {len(df_)} rows, episode_len={ep_len}")

    agent = DQNAgent(
        obs_size   = obs_sz,
        n_actions  = 3,
        hidden     = ac["hidden"],
        lr         = ac["lr"],
        gamma      = ac["gamma"],
        tau        = ac.get("tau", 0.01),
        eps        = ac.get("eps_start", 1.0),
        eps_end    = ac["eps_end"],
        eps_decay  = ac["eps_decay"],
        buffer_cap = ac["buffer_cap"],
        batch_size = ac["batch_size"],
        warmup     = ac["warmup"],
    )
    log.info(f"[Agent] Network: {obs_sz} → {ac['hidden']} → Q(3)")
    log.info(f"[Agent] Warmup={ac['warmup']} steps before learning starts")

    # ── 4. Training ───────────────────────────────────────────────────
    n_ep      = n_ep_override or ac["n_episodes"]
    patience  = ac["patience"]
    log_dir   = cfg["output"]["log_dir"]

    history:  list[dict] = []
    best_val  = -float("inf")
    best_ep   = 0
    pat_cnt   = 0
    start_t   = time.time()
    learned_at_least_once = False

    log.info(f"\n{'─'*65}")
    log.info(f"Training: {n_ep} episodes | patience={patience}")
    log.info(f"HOSE rules: T+{env_cfg.get('t_plus',2)} | Lot={env_cfg.get('lot_size',100)} | Tax={env_cfg.get('sell_tax',0.001)}")
    log.info(f"{'─'*65}")

    for ep in range(1, n_ep + 1):
        # ── Train episode ───────────────────────────────────────────
        tr_m, _  = run_episode(agent, tr_raw, tr_norm, env_cfg, train_mode=True)
        # ── Val episode (no learning) ───────────────────────────────
        vl_m, _  = run_episode(agent, va_raw, va_norm, env_cfg, train_mode=False)

        # ── Decay epsilon sau mỗi episode ───────────────────────────
        agent.decay_epsilon()

        # Chỉ tính loss sau khi đã học ít nhất 1 lần
        avg_loss = 0.0
        if agent.losses:
            avg_loss = float(np.mean(agent.losses[-200:]))
            learned_at_least_once = True

        rec = {
            "episode":        ep,
            "train_return":   round(float(tr_m["return_pct"]), 2),
            "train_sharpe":   round(float(tr_m["sharpe"]),     4),
            "train_mdd":      round(float(tr_m["max_dd_pct"]), 2),
            "train_trades":   int(tr_m["n_trades"]),
            "train_winrate":  round(float(tr_m.get("win_rate", 0)), 1),
            "train_steps":    int(tr_m.get("episode_steps", 0)),
            "val_return":     round(float(vl_m["return_pct"]), 2),
            "val_sharpe":     round(float(vl_m["sharpe"]),     4),
            "val_mdd":        round(float(vl_m["max_dd_pct"]), 2),
            "val_trades":     int(vl_m["n_trades"]),
            "val_winrate":    round(float(vl_m.get("win_rate", 0)), 1),
            "epsilon":        round(float(agent.eps),              5),
            "avg_loss":       round(float(avg_loss),               6),
            "total_steps":    int(agent.steps),
            "learn_count":    int(agent.learn_count),
        }
        history.append(rec)

        # Log mỗi N episodes
        log_n = max(1, n_ep // 60)
        if ep % log_n == 0 or ep <= 5:
            warmup_status = (f"[WARMUP {len(agent.buf)}/{agent.warmup}]"
                             if len(agent.buf) < agent.warmup else "[LEARNING]")
            elapsed = time.time() - start_t
            eta     = (elapsed / ep) * (n_ep - ep)
            log.info(
                f"Ep{ep:>4}/{n_ep} {warmup_status}"
                f" TR={rec['train_return']:>+7.2f}%"
                f" VR={rec['val_return']:>+7.2f}%"
                f" Sh={rec['val_sharpe']:>5.3f}"
                f" WR={rec['val_winrate']:>4.0f}%"
                f" T={rec['val_trades']:>3}"
                f" ε={agent.eps:.4f}"
                f" loss={avg_loss:.5f}"
                f" steps={rec['train_steps']}"
                f" ETA={eta/60:.0f}m"
            )

        # ── Checkpoint: chỉ lưu khi đã học (có loss) ───────────────
        if learned_at_least_once:
            # QUAN TRỌNG: Không lưu model không giao dịch (policy collapse)
            if vl_m["n_trades"] > 0:
                # Score = val_return * 0.4 + val_sharpe * 0.3 + trade_activity * 0.3
                trade_bonus = np.log1p(vl_m["n_trades"]) * 5
                wr_bonus    = max(0, vl_m.get("win_rate", 0) - 40) * 0.1
                score = (vl_m["return_pct"] * 0.4
                         + vl_m["sharpe"] * 30
                         + trade_bonus
                         + wr_bonus)
                if score > best_val:
                    best_val = score
                    best_ep  = ep
                    pat_cnt  = 0
                    agent.save(f"{model_dir}/best_model.pkl")
                    log.info(f"  → New best: val_return={vl_m['return_pct']:+.2f}% "
                             f"sharpe={vl_m['sharpe']:.3f} trades={vl_m['n_trades']} "
                             f"WR={vl_m.get('win_rate',0):.0f}% (ep {ep})")
                else:
                    pat_cnt += 1
            else:
                # Model không giao dịch — không tính vào patience
                log.debug(f"  Ep {ep}: val_trades=0, bỏ qua (không tính patience)")
        else:
            log.debug(f"  Ep {ep}: still in warmup, no checkpoint")

        # Periodic checkpoint
        ckpt_every = ac.get("checkpoint_every", 50)
        if ep % ckpt_every == 0:
            agent.save(f"{model_dir}/ckpt_ep{ep}.pkl")

        # Early stopping (chỉ sau warmup VÀ đã khám phá đủ)
        if learned_at_least_once and pat_cnt >= patience and agent.eps < 0.3:
            log.info(f"\nEarly stop @ ep {ep} — {patience} ep không cải thiện")
            break
        elif learned_at_least_once and pat_cnt >= patience and agent.eps >= 0.3:
            log.debug(f"  Ep {ep}: patience hết nhưng eps={agent.eps:.3f} > 0.3, tiếp tục")
            pat_cnt = patience // 2  # Reset 1 nửa patience

    agent.save(f"{model_dir}/last_model.pkl")
    with open(f"{log_dir}/training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_t
    log.info(f"\n{'='*65}")
    log.info(f"Training xong: {elapsed/60:.1f} phút | {agent.learn_count} gradient updates")
    log.info(f"Best model @ ep {best_ep} (score={best_val:.3f})")
    log.info(f"{'='*65}")

    # ── 5. Evaluate test set ──────────────────────────────────────────
    log.info("\n[Evaluate] Test set ...")
    best_path = f"{model_dir}/best_model.pkl"
    if Path(best_path).exists():
        agent.load(best_path)
    agent.eps = 0.0   # Greedy

    test_env = TradingEnv(
        df_raw   = te_raw,
        df_norm  = te_norm,
        window       = env_cfg["window"],
        init_cap     = env_cfg["initial_cap"],
        tx_cost      = env_cfg.get("tx_cost", 0.0015),
        sell_tax     = env_cfg.get("sell_tax", 0.001),
        slippage     = env_cfg.get("slippage", 0.0003),
        stop_loss    = env_cfg["stop_loss"],
        take_profit  = env_cfg["take_profit"],
        max_hold     = env_cfg.get("max_hold", 60),
        t_plus       = env_cfg.get("t_plus", 2),
        lot_size     = env_cfg.get("lot_size", 100),
        price_limit  = env_cfg.get("price_limit", 0.07),
        cooldown     = env_cfg.get("cooldown", 5),
        trade_penalty = env_cfg.get("trade_penalty", 0.3),
    )
    obs = test_env.reset(); done = False
    acts: list[int] = []; eqs: list[float] = []
    while not done:
        a = agent.act(obs, greedy=True)
        obs, _, done, info = test_env.step(a)
        acts.append(a); eqs.append(info["equity"])

    test_m = test_env.metrics()
    test_m["obs_size"] = obs_sz

    log.info("\n[Test Results]")
    for k, v in test_m.items():
        log.info(f"  {k:<28} {v}")

    # Build result_df
    n_valid   = min(len(acts), len(te_raw) - window)
    result_df = te_raw.iloc[window: window + n_valid].copy().reset_index(drop=True)
    result_df["rl_action"]   = acts[:n_valid]
    result_df["rl_equity"]   = eqs[:n_valid]
    result_df["buy_signal"]  = [a == TradingEnv.BUY  for a in acts[:n_valid]]
    result_df["sell_signal"] = [a == TradingEnv.SELL for a in acts[:n_valid]]

    # Save pkl
    res = {
        "result_df":    result_df,
        "test_metrics": test_m,
        "test_trades":  test_env.trades,
        "history":      history,
    }
    with open(f"{log_dir}/test_results.pkl", "wb") as f: pickle.dump(res, f)
    log.info(f"Results → {log_dir}/test_results.pkl")

    # ── 6. Charts ─────────────────────────────────────────────────────
    log.info("\n[Charts] Generating ...")
    _make_charts(res, cfg)


def _make_charts(res: dict, cfg: dict):
    try:
        from src.visualization.charts import generate_all
        generate_all(
            result_df   = res["result_df"],
            metrics     = res["test_metrics"],
            trades      = res.get("test_trades", []),
            history     = res.get("history", []),
            patterns    = res.get("patterns", []),
            pivots      = res.get("pivots",   []),
            initial_cap = cfg["env"]["initial_cap"],
            out_dir     = cfg["output"]["chart_dir"],
        )
    except Exception as e:
        log.warning(f"[Charts] Lỗi tạo chart: {e}")
        log.warning("[Charts] Bỏ qua chart, kết quả training vẫn OK.")


def charts_only(cfg: dict):
    pkl_path = f"{cfg['output']['log_dir']}/test_results.pkl"
    log_path = f"{cfg['output']['log_dir']}/training_log.json"
    if not Path(pkl_path).exists():
        print(f"Không tìm thấy {pkl_path}. Chạy training trước.")
        return
    with open(pkl_path, "rb") as f:   res = pickle.load(f)
    with open(log_path, encoding="utf-8") as f: res["history"] = json.load(f)
    _make_charts(res, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VNM RL Trading v6.0")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--mode",     default="train", choices=["train", "charts"])
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    cfg = load_cfg(args.config)

    if args.mode == "train":
        train(cfg, n_ep_override=args.episodes)
    else:
        charts_only(cfg)
