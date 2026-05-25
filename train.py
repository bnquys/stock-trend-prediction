"""
train.py 
════════════════════════════════════════════════════════════════════════════

  Tuân thủ HOSE: T+2, lot 100cp, thuế bán 0.1%, giới hạn ±7%
  Double DQN với per-episode epsilon decay
  Reward = PnL % khi Ư, clip [-10, 10]
  Observation = 20 bar × 27 features (normalized) + 4 portfolio state

Chạy:
    python train.py
    python train.py --episodes 500
    python train.py --mode charts     # tạo chart từ kết quả đã train
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse, json, logging, os, pickle, re, sys, time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import deque
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.logging_config import setup_logging
setup_logging(level="INFO", log_file="logs/train.log")
log = logging.getLogger(__name__)

from src.features.preprocessor import load_csv, time_split, RobustScaler, obs_size_of
from src.rl.env.trading_env import TradingEnv
from src.rl.agent.dqn_agent import DQNAgent
from stock_analysis.cache import EmbeddingCache


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
                train_mode: bool,
                episode_num: int = 1,
                stock_id: str | None = None,
                analysis_model: str | None = None,
                embedding_cache=None,
                learn_every: int = 1) -> tuple[dict, list]:
    """
    Chạy 1 episode hoàn chỉnh với error handling.
    Đảm bảo:
      - env.reset() → obs hợp lệ
      - loop chạy cho đến khi done=True
      - nếu train_mode: store + learn mỗi bước
      - Nếu analysis enabled: lấy embedding mỗi step và truyền vào agent
      - try-catch bảo vệ từng step và toàn bộ episode
    """
    try:
        env = TradingEnv(
            df_raw   = df_raw,
            df_norm  = df_norm,
            window       = env_cfg["window"],
            init_cap     = env_cfg["initial_cap"],
            tx_cost      = env_cfg.get("tx_cost", 0.0015),
            sell_tax     = env_cfg.get("sell_tax", 0.001),
            slippage     = env_cfg.get("slippage", 0.0003),
            atr_sl_mult  = env_cfg.get("atr_sl_mult", 1.5),
            atr_tp_mult  = env_cfg.get("atr_tp_mult", 3.0),
            risk_per_trade = env_cfg.get("risk_per_trade", 0.02),
            stop_loss    = env_cfg["stop_loss"],
            take_profit  = env_cfg["take_profit"],
            max_hold     = env_cfg.get("max_hold", 60),
            t_plus       = env_cfg.get("t_plus", 2),
            lot_size     = env_cfg.get("lot_size", 100),
            price_limit  = env_cfg.get("price_limit", 0.07),
            # Analysis embedding
            stock_id     = stock_id,
            analysis_model = analysis_model,
            embedding_cache = embedding_cache,
        )

        obs  = env.reset()
        done = False
        step_count = 0
        total_steps = env.n - env.window  # Số steps tối đa trong episode
        cum_reward = 0.0

        ep_label = f"Ep{episode_num}"
        if stock_id:
            ep_label += f" ({stock_id})"
        step_bar = tqdm(total=total_steps, desc=f"  {ep_label}",
                        unit="step", leave=False, dynamic_ncols=True)

        while not done:
            try:
                # Lấy analysis embedding cho window hiện tại (None nếu disabled)
                analysis_embed = env.get_analysis_embed()

                action = agent.act(obs, valid_actions=env.valid_actions(),
                                   greedy=not train_mode, analysis_embed=analysis_embed)
                next_obs, reward, done, info = env.step(action)
                cum_reward += reward

                if train_mode:
                    agent.store(obs, action, reward, next_obs, done,
                                analysis_embed=analysis_embed)
                    # Learn every N steps (giảm gradient updates, tăng tốc)
                    if step_count % learn_every == 0:
                        agent.learn()

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
        return {
            "return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
            "n_trades": 0, "win_rate": 0.0, "episode_steps": 0,
            "arr_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "pf": 0.0, "final_equity": 0, "n_sl": 0, "n_tp": 0,
            "n_mh": 0, "steps": 0,
        }, []


# ─────────────────────────────────────────────────────────────────────────
# Score normalization — thay thế magic numbers cũ
# ─────────────────────────────────────────────────────────────────────────

def normalize_score(metrics: dict) -> float:
    """
    Chuẩn hoá metrics về [0,1] rồi tính weighted score.
    Thay thế công thức cũ dùng magic numbers.
    """
    # Normalize mỗi metric về khoảng [0, 1]
    ret = metrics.get("return_pct", 0.0)
    return_norm = np.clip((ret + 10) / 30, 0, 1)     # Range: -10% to +20%

    sharpe = metrics.get("sharpe", 0.0)
    sharpe_norm = np.clip((sharpe + 1) / 4, 0, 1)     # Range: -1 to 3

    trades = metrics.get("n_trades", 0)
    trades_norm = np.clip(trades / 50, 0, 1)           # Range: 0-50 trades

    wr = metrics.get("win_rate", 0.0)
    wr_norm = np.clip(wr / 100, 0, 1)                  # Range: 0-100%

    # Weighted combination
    score = (return_norm * 0.35
             + sharpe_norm * 0.30
             + trades_norm * 0.15
             + wr_norm     * 0.20)

    return float(score)


# ─────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────

def train(cfg: dict, n_ep_override: int | None = None, resume_from: str | None = None):
    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["log_dir"],   exist_ok=True)
    os.makedirs(cfg["output"]["chart_dir"], exist_ok=True)

    np.random.seed(cfg["project"].get("seed", 42))

    log.info("=" * 65)
    log.info("  VNM RL Trading v6.0 — Double DQN (HOSE compliant)")
    log.info("  T+2 | Lot 100cp | Thuế 0.1% | ±7% price limit")
    log.info("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────────────
    paths = cfg["data"].get("paths", [cfg["data"].get("path")])
    if not paths or paths[0] is None:
        raise ValueError("No data paths defined in config!")

    train_raws, val_raws, test_raws = [], [], []
    sp = cfg["split"]
    for p in paths:
        raw_df = load_csv(p)
        tr, va, te = time_split(raw_df, sp["train_ratio"], sp["val_ratio"])
        train_raws.append(tr)
        val_raws.append(va)
        test_raws.append(te)

    # ── 2. Scale features ────────────────────────────────────────────
    log.debug("[Scale] RobustScaler (fit on all train sets)...")
    combined_tr = pd.concat(train_raws, ignore_index=True)
    scaler = RobustScaler().fit(combined_tr)
    
    train_norms = [scaler.transform(df) for df in train_raws]
    va_norms = [scaler.transform(df) for df in val_raws]
    te_norms = [scaler.transform(df) for df in test_raws]

    model_dir = cfg["output"]["model_dir"]
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── 3. Agent setup ────────────────────────────────────────────────
    window  = cfg["env"]["window"]
    obs_sz  = obs_size_of(train_norms[0], window)
    ac      = cfg["agent"]
    env_cfg = cfg["env"]

    # ── Analysis Embedding config ─────────────────────────────────
    analysis_cfg = cfg.get("analysis", {})
    analysis_enabled = analysis_cfg.get("enabled", False)
    analysis_model   = analysis_cfg.get("model", "mistral-small-4-119b-2603")
    analysis_embed_dim = analysis_cfg.get("embed_dim", 2560) if analysis_enabled else None
    analysis_proj_layers = analysis_cfg.get("projection", [1024, 512, 128]) if analysis_enabled else None

    # ── Training optimization config ─────────────────────────────
    training_cfg = cfg.get("training", {})
    val_every    = training_cfg.get("val_every", 5)
    learn_every  = training_cfg.get("learn_every", 4)
    preload_embeddings = training_cfg.get("preload_embeddings", True)

    log.debug(f"[Agent] obs_size={obs_sz} | window={window}")
    log.debug(f"[Optim] val_every={val_every} | learn_every={learn_every} | preload={preload_embeddings}")
    if analysis_enabled:
        log.info(f"[Analysis] Enabled: model={analysis_model}, dim={analysis_embed_dim}")
    else:
        log.info("[Analysis] Disabled — agent chỉ dùng technical features")

    # ── Pre-load embeddings vào RAM (zero I/O during training) ────
    embedding_cache = None
    if analysis_enabled and preload_embeddings:
        stock_ids = [Path(p).stem for p in paths]
        embedding_cache = EmbeddingCache(stock_ids)
        log.debug(f"[EmbeddingCache] Stats: {embedding_cache.stats}")

    # Validate: episode_len phải > 0 cho tất cả splits
    for stock_idx, p in enumerate(paths):
        symbol = Path(p).stem
        for name, df_ in [("train", train_raws[stock_idx]), ("val", val_raws[stock_idx]), ("test", test_raws[stock_idx])]:
            ep_len = len(df_) - window
            assert ep_len > 0, f"[{symbol}] {name} df quá ngắn: {len(df_)} rows, window={window}"
            log.debug(f"[{symbol}] {name}: {len(df_)} rows, episode_len={ep_len}")

    agent = DQNAgent(
        obs_size   = obs_sz,
        n_actions  = 3,
        hidden     = ac["hidden"],
        lr         = ac["lr"],
        lr_decay   = ac.get("lr_decay", 0.995),
        lr_min     = ac.get("lr_min", 5e-5),
        gamma      = ac["gamma"],
        tau        = ac.get("tau", 0.01),
        eps        = ac.get("eps_start", 1.0),
        eps_end    = ac["eps_end"],
        eps_decay  = ac["eps_decay"],
        buffer_cap = ac["buffer_cap"],
        batch_size = ac["batch_size"],
        warmup     = ac["warmup"],
        # Analysis embedding
        analysis_embed_dim   = analysis_embed_dim,
        analysis_proj_layers = analysis_proj_layers,
    )
    log.debug(f"[Agent] Network: {obs_sz} → {ac['hidden']} → Q(3)")
    log.debug(f"[Agent] Warmup={ac['warmup']} steps before learning starts")
    log.debug(f"[Agent] LR={ac['lr']}, decay={ac.get('lr_decay', 0.995)}, min={ac.get('lr_min', 5e-5)}")

    # ── Resume from checkpoint ────────────────────────────────────
    start_ep = 1
    if resume_from and Path(resume_from).exists():
        agent.load(resume_from)
        start_ep = agent.episode_num + 1
        log.info(f"[Resume] Continuing from episode {start_ep} (loaded {resume_from})")

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
    val_history: deque = deque(maxlen=10)  # Validation smoothing
    vl_m = {"return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
            "n_trades": 0, "win_rate": 0.0, "episode_steps": 0}

    log.info(f"\n{'─'*65}")
    log.info(f"Training: {n_ep} episodes | patience={patience}")
    log.info(f"HOSE rules: T+{env_cfg.get('t_plus',2)} | Lot={env_cfg.get('lot_size',100)} | Tax={env_cfg.get('sell_tax',0.001)}")
    log.info(f"{'─'*65}")

    pbar = tqdm(range(start_ep, n_ep + 1), desc="Training", unit="ep",
                dynamic_ncols=True, initial=start_ep - 1, total=n_ep)
    for ep in pbar:
        # ── 4.1 Curriculum Learning: Chọn cổ phiếu theo tiến độ ──────
        # Quy tắc: 
        # - 0-20% episodes: Chỉ VNM (Stable)
        # - 20-40% episodes: VNM + FPT
        # - 40-60% episodes: VNM + FPT + HPG
        # - 60-80% episodes: Toàn bộ (VNM, FPT, HPG, VIC)
        # - 80-100% episodes: Random hoàn toàn
        
        progress = ep / n_ep
        if progress < 0.2:
            # Chỉ mã đầu tiên (giả định là VNM)
            stock_idx = 0
        elif progress < 0.4:
            stock_idx = np.random.randint(min(len(train_raws), 2))
        elif progress < 0.6:
            stock_idx = np.random.randint(min(len(train_raws), 3))
        elif progress < 0.8:
            stock_idx = np.random.randint(len(train_raws))
        else:
            # Tập trung vào những mã Agent còn yếu hoặc random đều
            stock_idx = np.random.randint(len(train_raws))
        
        # ── Train episode ───────────────────────────────────────────
        train_symbol = Path(paths[stock_idx]).stem if analysis_enabled else None
        tr_m, _  = run_episode(agent, train_raws[stock_idx], train_norms[stock_idx], env_cfg,
                               train_mode=True, episode_num=ep,
                               stock_id=train_symbol,
                               analysis_model=analysis_model if analysis_enabled else None,
                               embedding_cache=embedding_cache,
                               learn_every=learn_every)
        
        # ── Val episode (mỗi val_every episodes) ────────────────────
        if ep % val_every == 0 or ep == start_ep or ep == n_ep:
            vl_metrics_list = []
            for v_idx, (v_raw, v_norm) in enumerate(zip(val_raws, va_norms)):
                val_symbol = Path(paths[v_idx]).stem if analysis_enabled else None
                vl_m, _ = run_episode(agent, v_raw, v_norm, env_cfg, train_mode=False,
                                      stock_id=val_symbol,
                                      analysis_model=analysis_model if analysis_enabled else None,
                                      embedding_cache=embedding_cache)
                vl_metrics_list.append(vl_m)
                
            vl_m = {
                "return_pct": np.mean([m["return_pct"] for m in vl_metrics_list]),
                "sharpe": np.mean([m["sharpe"] for m in vl_metrics_list]),
                "max_dd_pct": np.mean([m["max_dd_pct"] for m in vl_metrics_list]),
                "n_trades": int(np.mean([m["n_trades"] for m in vl_metrics_list])),
                "win_rate": np.mean([m.get("win_rate", 0) for m in vl_metrics_list]),
                "episode_steps": int(np.mean([m.get("episode_steps", 0) for m in vl_metrics_list])),
            }

        # ── Decay epsilon + LR sau mỗi episode ────────────────────
        agent.decay_epsilon()
        agent.decay_lr()

        # Chỉ tính loss sau khi đã học ít nhất 1 lần
        avg_loss = 0.0
        if agent.losses:
            avg_loss = float(np.mean(list(agent.losses)[-200:]))
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

        # ── Update tqdm postfix ────────────────────────────────────
        pbar.set_postfix_str(
            f"TR={rec['train_return']:+.1f}% VR={rec['val_return']:+.1f}% "
            f"Sh={rec['val_sharpe']:.2f} WR={rec['val_winrate']:.0f}% "
            f"ε={agent.eps:.3f} loss={avg_loss:.4f}"
        )

        # Log mỗi N episodes (chi tiết hơn)
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

        # ── Validation smoothing ───────────────────────────────────
        val_history.append(vl_m)
        smoothed_val = {
            "return_pct": float(np.mean([v["return_pct"] for v in val_history])),
            "sharpe":     float(np.mean([v["sharpe"] for v in val_history])),
            "n_trades":   int(np.mean([v["n_trades"] for v in val_history])),
            "win_rate":   float(np.mean([v.get("win_rate", 0) for v in val_history])),
        }

        # ── Checkpoint: chỉ lưu khi đã học (có loss) ───────────────
        if learned_at_least_once:
            # QUAN TRỌNG: Không lưu model không giao dịch (policy collapse)
            if smoothed_val["n_trades"] > 0:
                # Normalized score (thay thế magic numbers cũ)
                score = normalize_score(smoothed_val)
                if score > best_val:
                    best_val = score
                    best_ep  = ep
                    pat_cnt  = 0
                    agent.save(f"{model_dir}/best_model.pkl")
                    log.info(f"  → New best (score={score:.4f}): val_return={vl_m['return_pct']:+.2f}% "
                             f"sharpe={vl_m['sharpe']:.3f} trades={vl_m['n_trades']} "
                             f"WR={vl_m.get('win_rate',0):.0f}% lr={agent.current_lr:.6f} (ep {ep})")
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

        # Early stopping (dùng smoothed metrics)
        if learned_at_least_once and pat_cnt >= patience and agent.eps < 0.3:
            log.info(f"\nEarly stop @ ep {ep} — {patience} ep không cải thiện")
            break
        elif learned_at_least_once and pat_cnt >= patience and agent.eps >= 0.3:
            log.debug(f"  Ep {ep}: patience hết nhưng eps={agent.eps:.3f} > 0.3, tiếp tục")
            pat_cnt = patience // 2  

    agent.save(f"{model_dir}/last_model.pkl")
    with open(f"{log_dir}/training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_t
    log.info(f"\n{'='*65}")
    log.info(f"Training xong: {elapsed/60:.1f} phút | {agent.learn_count} gradient updates")
    log.info(f"Best model @ ep {best_ep} (score={best_val:.4f})")
    log.info(f"Final LR: {agent.current_lr:.6f} | Final eps: {agent.eps:.4f}")
    log.info(f"{'='*65}")

    # ── 5. Evaluate test set ──────────────────────────────────────────
    log.info("\n[Evaluate] Test set on ALL stocks...")
    best_path = f"{model_dir}/best_model.pkl"
    if Path(best_path).exists():
        agent.load(best_path)
    agent.eps = 0.0   # Greedy

    for stock_idx, (te_raw, te_norm, path) in enumerate(zip(test_raws, te_norms, paths)):
        symbol = Path(path).stem
        log.info(f"\n{'='*40}")
        log.info(f"  Testing {symbol}")
        log.info(f"{'='*40}")
        
        test_env = TradingEnv(
            df_raw   = te_raw,
            df_norm  = te_norm,
            window       = env_cfg["window"],
            init_cap     = env_cfg["initial_cap"],
            tx_cost      = env_cfg.get("tx_cost", 0.0015),
            sell_tax     = env_cfg.get("sell_tax", 0.001),
            slippage     = env_cfg.get("slippage", 0.0003),
            atr_sl_mult  = env_cfg.get("atr_sl_mult", 1.5),
            atr_tp_mult  = env_cfg.get("atr_tp_mult", 3.0),
            risk_per_trade = env_cfg.get("risk_per_trade", 0.02),
            stop_loss    = env_cfg["stop_loss"],
            take_profit  = env_cfg["take_profit"],
            max_hold     = env_cfg.get("max_hold", 60),
            t_plus       = env_cfg.get("t_plus", 2),
            lot_size     = env_cfg.get("lot_size", 100),
            price_limit  = env_cfg.get("price_limit", 0.07),
            # Analysis embedding
            stock_id       = symbol if analysis_enabled else None,
            analysis_model = analysis_model if analysis_enabled else None,
            embedding_cache = embedding_cache,
        )
        obs = test_env.reset(); done = False
        acts: list[int] = []; eqs: list[float] = []
        while not done:
            analysis_embed = test_env.get_analysis_embed()
            a = agent.act(obs, valid_actions=test_env.valid_actions(),
                          greedy=True, analysis_embed=analysis_embed)
            obs, _, done, info = test_env.step(a)
            acts.append(a); eqs.append(info["equity"])

        test_m = test_env.metrics()
        test_m["obs_size"] = obs_sz

        log.info(f"\n[Test Results - {symbol}]")
        for k, v in test_m.items():
            log.info(f"  {k:<28} {v}")

        # Build result_df
        n_valid   = min(len(acts), len(te_raw) - window)
        result_df = te_raw.iloc[window: window + n_valid].copy().reset_index(drop=True)
        result_df["rl_action"]   = acts[:n_valid]
        result_df["rl_equity"]   = eqs[:n_valid]
        # Populate buy_signal and sell_signal from actual trades to include AUTO_EXITs
        buy_flags  = [False] * n_valid
        sell_flags = [False] * n_valid
        for t in test_env.trades:
            # step in trade is the index in env, we subtract window to get index in result_df
            idx = t["step"] - window
            if 0 <= idx < n_valid:
                if t["type"] == "BUY":
                    buy_flags[idx] = True
                elif t["type"] in ["SELL", "AUTO_EXIT"]:
                    sell_flags[idx] = True
                    
        result_df["buy_signal"]  = buy_flags
        result_df["sell_signal"] = sell_flags

        # Save pkl
        res = {
            "symbol":       symbol,
            "result_df":    result_df,
            "test_metrics": test_m,
            "test_trades":  test_env.trades,
            "history":      history,
        }
        with open(f"{log_dir}/test_results_{symbol}.pkl", "wb") as f: pickle.dump(res, f)
        log.debug(f"Results → {log_dir}/test_results_{symbol}.pkl")

        # ── 6. Export ROI Table ───────────────────────────────────────────
        log.debug(f"[ROI Table] Generating for {symbol}...")
        _export_roi_table(res, cfg, symbol=symbol)

        # ── 7. Charts ─────────────────────────────────────────────────────
        log.debug(f"[Charts] Generating for {symbol}...")
        _make_charts(res, cfg, symbol=symbol)

        # ── 8. LLM Integration Report ─────────────────────────────────────
        log.debug(f"[LLM Report] Generating LLM Integration Summary for {symbol}...")
        _export_llm_integration_report(res, cfg, symbol=symbol)


def _make_charts(res: dict, cfg: dict, symbol: str = "VNM"):
    try:
        from src.visualization.charts import generate_all
        generate_all(
            result_df   = res["result_df"],
            metrics     = res["test_metrics"],
            trades      = res.get("test_trades", []),
            history     = res.get("history", []),
            initial_cap = cfg["env"]["initial_cap"],
            out_dir     = f"{cfg['output']['chart_dir']}/{symbol}",
            symbol      = symbol,
        )
    except Exception as e:
        log.warning(f"[Charts] Lỗi tạo chart: {e}")
        log.warning("[Charts] Bỏ qua chart, kết quả training vẫn OK.")


def _export_roi_table(res: dict, cfg: dict, symbol: str = "VNM"):
    trades = res.get("test_trades", [])
    if not trades:
        log.debug("Không có giao dịch nào trong test_trades để tạo bảng ROI.")
        return
    
    # Lọc ra các giao dịch đã đóng (SELL hoặc AUTO_EXIT)
    closed_trades = [t for t in trades if t["type"] in ["SELL", "AUTO_EXIT"]]
    
    if not closed_trades:
        log.debug("Chưa có giao dịch đóng nào để tính ROI.")
        return

    # Lọc ra các lệnh mua
    buy_trades = [t for t in trades if t["type"] == "BUY"]
    
    roi_data = []
    buy_idx = 0
    for ct in closed_trades:
        buy_date = "N/A"
        shares = 0
        # Cố gắng match với lệnh mua gần nhất trước khi đóng
        while buy_idx < len(buy_trades) and buy_trades[buy_idx]["step"] <= ct["step"]:
            b_trade = buy_trades[buy_idx]
            buy_date = b_trade["date"]
            shares = b_trade.get("shares", 0)
            buy_idx += 1
            break
            
        entry_price = ct["entry_price"]
        exit_price = ct["price"]
        pnl_pct = ct["pnl_pct"]
        hold_days = ct.get("hold_days", 0)
        reason = ct.get("reason", "manual")
        
        capital_invested = shares * entry_price
        profit = capital_invested * (pnl_pct / 100)
        
        roi_data.append({
            "Buy Date": buy_date,
            "Sell Date": ct["date"],
            "Shares": shares,
            "Entry (VND)": entry_price,
            "Exit (VND)": exit_price,
            "Hold Days": hold_days,
            "Exit Reason": reason.upper(),
            "Invested (VND)": f"{capital_invested:,.0f}",
            "P/L (VND)": f"{profit:,.0f}",
            "ROI (%)": pnl_pct
        })
        
    df_roi = pd.DataFrame(roi_data)
    
    out_dir = cfg["output"].get("log_dir", "logs")
    out_path = f"{out_dir}/roi_table_{symbol}.csv"
    df_roi.to_csv(out_path, index=False)
    
    log.debug(f"Đã xuất bảng ROI tại: {out_path}")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    log.debug(f"Bảng ROI:\n{df_roi.to_string(index=False)}")


def _export_llm_integration_report(res: dict, cfg: dict, symbol: str = "VNM"):
    df = res.get("result_df")
    if df is None or df.empty:
        return
        
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row
    
    # Xác định khuyến nghị từ RL
    rl_action_code = last_row.get("rl_action", 0)
    if rl_action_code == TradingEnv.BUY:
        rl_rec = "MUA (BUY)"
    elif rl_action_code == TradingEnv.SELL:
        rl_rec = "BÁN (SELL)"
    else:
        rl_rec = "NẮM GIỮ / ĐỨNG NGOÀI (HOLD)"
        
    # Technical Indicators
    rsi = last_row.get("rsi_14", 0)
    macd = last_row.get("macd_histogram", 0)
    sma20 = last_row.get("sma_20", 0)
    trend_sma20 = "TĂNG" if sma20 > prev_row.get("sma_20", 0) else "GIẢM"
    
    # Metrics
    m = res.get("test_metrics", {})
    
    llm_data = {
        "report_date": str(last_row.get("date", "N/A"))[:10],
        "ticker": symbol,
        "latest_close_price": float(last_row.get("close", 0)),
        "rl_recommendation": rl_rec,
        "technical_context": {
            "rsi": round(float(rsi), 2),
            "macd": round(float(macd), 4),
            "sma_20_trend": trend_sma20
        },
        "rl_performance_history": {
            "win_rate_pct": float(m.get("win_rate", 0)),
            "sharpe_ratio": float(m.get("sharpe", 0)),
            "profit_factor": float(m.get("pf", 0))
        }
    }
    
    out_dir = cfg["output"].get("log_dir", "logs")
    out_json = f"{out_dir}/llm_integration_summary_{symbol}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=4)
        
    log.debug(f"Đã xuất file JSON cho LLM tại: {out_json}")
    
    # In ra dạng bảng Markdown cho User copy
    md_table = f"""
### BẢNG TÓM TẮT KỸ THUẬT VÀ TÍN HIỆU RL (DÀNH CHO LLM)
**Ngày cập nhật:** {llm_data['report_date']} | **Mã CP:** {llm_data['ticker']} | **Giá đóng cửa:** {llm_data['latest_close_price']:,.0f} VND

| Phân loại | Chỉ tiêu | Giá trị hiện tại | Ý nghĩa đối với LLM (Gợi ý) |
| :--- | :--- | :--- | :--- |
| **TÍN HIỆU RL** | **Khuyến nghị Agent** | **{llm_data['rl_recommendation']}** | Quyết định giao dịch ngắn hạn dựa trên AI. |
| | Lịch sử Win Rate | {llm_data['rl_performance_history']['win_rate_pct']}% | Độ tin cậy của Agent trong quá khứ. |
| | Sharpe Ratio | {llm_data['rl_performance_history']['sharpe_ratio']} | Mức độ rủi ro trên lợi nhuận của Agent. |
| **KỸ THUẬT** | RSI (14 ngày) | {llm_data['technical_context']['rsi']} | >70: Quá mua, <30: Quá bán. |
| | MACD Histogram | {llm_data['technical_context']['macd']} | >0: Động lượng tăng, <0: Động lượng giảm. |
| | Xu hướng SMA 20 | {llm_data['technical_context']['sma_20_trend']} | Xu hướng giá trung hạn (20 phiên). |

*Lưu ý cho LLM: Hãy kết hợp tín hiệu MUA/BÁN của RL Agent ở trên với sức khỏe tài chính doanh nghiệp từ BCTC để đưa ra kết luận đầu tư cuối cùng.*
"""
    print(md_table)
    with open(f"{out_dir}/llm_summary_{symbol}.md", "w", encoding="utf-8") as f:
        f.write(md_table)


def charts_only(cfg: dict):
    paths = cfg["data"].get("paths", [cfg["data"].get("path")])
    for p in paths:
        symbol = Path(p).stem
        pkl_path = f"{cfg['output']['log_dir']}/test_results_{symbol}.pkl"
        log_path = f"{cfg['output']['log_dir']}/training_log.json"
        if not Path(pkl_path).exists():
            print(f"Không tìm thấy {pkl_path}. Chạy training trước.")
            continue
        with open(pkl_path, "rb") as f:   res = pickle.load(f)
        
        if Path(log_path).exists():
            with open(log_path, encoding="utf-8") as f: res["history"] = json.load(f)

        _export_roi_table(res, cfg, symbol)
        _make_charts(res, cfg, symbol)
        _export_llm_integration_report(res, cfg, symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VNM RL Trading v7.0")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--mode",     default="train", choices=["train", "charts"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint .pkl to resume training")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    cfg = load_cfg(args.config)

    if args.mode == "train":
        try:
            train(cfg, n_ep_override=args.episodes, resume_from=args.resume)
        except KeyboardInterrupt:
            log.info("\n[!] Training interrupted by user. Last checkpoint saved.")
        except Exception as e:
            log.error(f"\n[!] Training crashed: {e}")
            log.error("[!] Check latest checkpoint in weights/ to resume.")
            raise
    else:
        charts_only(cfg)
