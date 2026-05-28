"""
src/ui/app.py
════════════════════════════════════════════════════════════════════════════
Trading Simulation UI — Gradio App

4 tabs riêng biệt cho 4 mã cổ phiếu (VNM, FPT, VIC, HPG).
Mỗi tab mô phỏng trading thời gian thực trên test-set.
Hỗ trợ chọn model từ Dropdown (chỉ hiện các run đã train hoàn chỉnh).

Usage:
    python -m src.ui.app
    # → Mở browser tại http://localhost:7860
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import Config
from src.technical.preprocessor import load_csv, time_split, RobustScaler, obs_size_of
from src.rl.agent.dqn_agent import DQNAgent
from src.fundamental.cache import EmbeddingCache
from src.ui.simulator import TradingSimulator
from src.ui.components import create_candlestick_chart, format_prediction, format_action_badge

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Model Discovery
# ─────────────────────────────────────────────────────────────────────────

def list_available_models() -> list[dict]:
    """
    Scan artifacts/outputs/ for completed training runs.
    A run is considered complete if it has both:
      - weights/best_model.pkl
      - logs.json
    Returns list of dicts with run info, sorted newest first.
    """
    outputs_dir = _PROJECT_ROOT / "artifacts" / "outputs"
    if not outputs_dir.exists():
        return []

    models = []
    for logs_path in sorted(outputs_dir.glob("output_*/logs.json"), reverse=True):
        run_dir = logs_path.parent
        weights_path = run_dir / "weights" / "best_model.pkl"
        if not weights_path.exists():
            continue

        # Read logs.json for display info
        try:
            with open(logs_path, encoding="utf-8") as f:
                run_log = json.load(f)

            # Extract key info for display
            result = run_log.get("result", {})
            test_metrics = result.get("test_metrics", {})
            avg = test_metrics.get("average", {})
            params = run_log.get("parameters", {})
            env_params = params.get("env", {})
            agent_params = params.get("agent", {})
            models_info = run_log.get("models", {})

            models.append({
                "run_id": run_log.get("run_id", run_dir.name),
                "run_dir": str(run_dir),
                "weights_dir": str(run_dir / "weights"),
                "finished_at": run_log.get("finished_at", ""),
                "elapsed_min": run_log.get("elapsed_minutes", 0),
                "best_score": result.get("best_score", 0),
                "sharpe": avg.get("sharpe", 0),
                "return_pct": avg.get("return_pct", 0),
                "win_rate": avg.get("win_rate", 0),
                "n_trades": avg.get("n_trades", 0),
                "window": env_params.get("window", 20),
                "gamma": agent_params.get("gamma", 0.97),
                "lr": agent_params.get("lr", 0.0005),
                "llm_model": models_info.get("llm_model", "N/A"),
                "embedding_model": models_info.get("embedding_model", "N/A"),
            })
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping {run_dir.name}: {e}")
            continue

    return models


def format_model_choice(model: dict) -> str:
    """Format model info for Dropdown display."""
    return (
        f"{model['run_id']} | "
        f"Sharpe={model['sharpe']:.2f} | "
        f"Return={model['return_pct']:+.1f}% | "
        f"WR={model['win_rate']:.0f}% | "
        f"γ={model['gamma']} | "
        f"w={model['window']}"
    )


def format_model_info(model: dict) -> str:
    """Format detailed model info as Markdown."""
    return (
        f"**Model:** `{model['run_id']}`  \n"
        f"**Finished:** {model['finished_at']}  \n"
        f"**Score:** {model['best_score']:.4f} | "
        f"**Sharpe:** {model['sharpe']:.3f} | "
        f"**Return:** {model['return_pct']:+.2f}% | "
        f"**Win Rate:** {model['win_rate']:.0f}% | "
        f"**Trades:** {model['n_trades']}  \n"
        f"**Window:** {model['window']} | "
        f"**Gamma:** {model['gamma']} | "
        f"**LR:** {model['lr']}  \n"
        f"**LLM:** {model['llm_model']} | "
        f"**Embedding:** {model['embedding_model']}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────

def find_latest_weights() -> Path:
    """Find the latest output directory with trained weights."""
    outputs_dir = _PROJECT_ROOT / "artifacts" / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(
            f"No outputs directory found at {outputs_dir}. "
            "Please train a model first with `python -m scripts.train`."
        )

    # Find latest output_* directory with weights/best_model.pkl AND logs.json
    candidates = sorted(
        [p for p in outputs_dir.glob("output_*/weights/best_model.pkl")
         if (p.parent.parent / "logs.json").exists()],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed training run found in {outputs_dir}/output_*/. "
            "Please train a model first."
        )

    return candidates[0].parent  # Return weights/ directory


def init_app(weights_dir: Path | None = None) -> tuple[Config, DQNAgent, dict[str, TradingSimulator]]:
    """Initialize config, agent, and simulators for all stocks."""
    print("[*] Initializing Trading Simulation UI...")

    # Load config
    cfg = Config.load(str(_PROJECT_ROOT / "configs"))
    print(f"   [OK] Config loaded")

    # Find weights
    if weights_dir is None:
        weights_dir = find_latest_weights()
    print(f"   [OK] Weights: {weights_dir}")

    # Load scaler
    scaler_path = weights_dir / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler: RobustScaler = pickle.load(f)
    print(f"   [OK] Scaler loaded")

    # Load data & split
    paths = cfg.data.get("paths", [])
    stock_ids = [Path(p).stem for p in paths]
    sp = cfg.split

    test_raws = []
    test_norms = []
    for p in paths:
        raw_df = load_csv(p)
        _, _, te = time_split(raw_df, sp["train_ratio"], sp["val_ratio"])
        test_raws.append(te)
        test_norms.append(scaler.transform(te))

    print(f"   [OK] Data loaded: {stock_ids}")

    # Create agent
    window = cfg.env["window"]
    obs_sz = obs_size_of(test_norms[0], window)
    ac = cfg.agent
    analysis_cfg = cfg.analysis
    analysis_enabled = analysis_cfg.get("enabled", False)

    agent = DQNAgent(
        obs_size=obs_sz,
        n_actions=3,
        hidden=ac["hidden"],
        lr=ac["lr"],
        gamma=ac["gamma"],
        tau=ac.get("tau", 0.01),
        eps=0.0,  # Greedy
        eps_end=ac["eps_end"],
        eps_decay=ac["eps_decay"],
        buffer_cap=ac["buffer_cap"],
        batch_size=ac["batch_size"],
        warmup=ac["warmup"],
        weight_decay=ac.get("weight_decay", 1e-4),
        grad_clip=ac.get("grad_clip", 1.0),
        loss_fn=ac.get("loss_fn", "huber"),
        analysis_embed_dim=analysis_cfg.get("embed_dim") if analysis_enabled else None,
        analysis_proj_layers=analysis_cfg.get("projection") if analysis_enabled else None,
    )
    model_path = weights_dir / "best_model.pkl"
    agent.load(str(model_path))
    agent.eps = 0.0
    print(f"   [OK] Agent loaded (obs_size={obs_sz})")

    # Load embedding cache
    embedding_cache = None
    if analysis_enabled:
        embedding_cache = EmbeddingCache(stock_ids)
        print(f"   [OK] EmbeddingCache loaded: {embedding_cache.stats}")

    # Create simulators
    simulators = {}
    for i, sid in enumerate(stock_ids):
        simulators[sid] = TradingSimulator(
            stock_id=sid,
            cfg=cfg,
            agent=agent,
            test_raw=test_raws[i],
            test_norm=test_norms[i],
            embedding_cache=embedding_cache,
        )
    print(f"   [OK] Simulators created: {list(simulators.keys())}")
    print("[OK] Ready!")

    return cfg, agent, simulators


# ─────────────────────────────────────────────────────────────────────────
# Gradio UI Builder
# ─────────────────────────────────────────────────────────────────────────

def build_tab(stock_id: str, sim: TradingSimulator, cfg: Config):
    """Build a single stock tab with all components."""

    def on_next_day(state):
        """Handle Next Day button click."""
        simulator: TradingSimulator = state
        result = simulator.next_day()

        # Build chart
        buys, sells = simulator.get_buy_sell_data()
        chart = create_candlestick_chart(
            df_raw=simulator.test_raw,
            current_step=simulator.current_step - 1,
            buys=buys,
            sells=sells,
            stock_id=stock_id,
            window=cfg.env["window"],
        )

        # Format texts
        action_md = format_action_badge(result)
        prediction_md = format_prediction(result["next_prediction"])
        fundamental_md = result["fundamental_md"]
        portfolio_md = simulator.get_portfolio_summary()

        return chart, action_md, prediction_md, fundamental_md, portfolio_md, state

    def on_reset(state):
        """Handle Reset button click."""
        simulator: TradingSimulator = state
        simulator.reset()

        # Empty chart
        chart = create_candlestick_chart(
            df_raw=simulator.test_raw,
            current_step=cfg.env["window"],
            buys=[],
            sells=[],
            stock_id=stock_id,
            window=cfg.env["window"],
        )

        return (
            chart,
            "Bấm **Next Day ▶** để bắt đầu simulation.",
            "Chưa có prediction.",
            "Chưa có dữ liệu phân tích.",
            "Chưa bắt đầu giao dịch.",
            state,
        )

    # ── Layout ──────────────────────────────────────────────────────
    with gr.Column():
        # State
        state = gr.State(sim)

        # Row 1: Chart
        chart_plot = gr.Plot(label=f"📈 {stock_id} — Biểu đồ Nến")

        # Row 2: Action + Controls
        with gr.Row():
            with gr.Column(scale=2):
                action_md = gr.Markdown(
                    value="Bấm **Next Day ▶** để bắt đầu simulation.",
                    label="Hành động hôm nay",
                )
            with gr.Column(scale=1):
                next_btn = gr.Button(
                    "Next Day ▶",
                    variant="primary",
                    size="lg",
                )
                reset_btn = gr.Button("🔄 Reset", variant="secondary", size="sm")

        # Row 3: Prediction + Fundamental
        with gr.Row():
            with gr.Column(scale=1):
                prediction_md = gr.Markdown(
                    value="Chưa có prediction.",
                    label="🤖 Agent Prediction — Ngày tiếp theo",
                )
            with gr.Column(scale=1):
                portfolio_md = gr.Markdown(
                    value="Chưa bắt đầu giao dịch.",
                    label="📊 Portfolio",
                )

        # Row 4: Fundamental (full width, scrollable)
        fundamental_md = gr.Markdown(
            value="Chưa có dữ liệu phân tích.",
            label="📋 Phân tích Cơ bản (LLM Response)",
        )

        # ── Events ─────────────────────────────────────────────────
        next_btn.click(
            fn=on_next_day,
            inputs=[state],
            outputs=[chart_plot, action_md, prediction_md, fundamental_md, portfolio_md, state],
        )
        reset_btn.click(
            fn=on_reset,
            inputs=[state],
            outputs=[chart_plot, action_md, prediction_md, fundamental_md, portfolio_md, state],
        )


def build_app(simulators: dict[str, TradingSimulator], cfg: Config) -> gr.Blocks:
    """Build the full Gradio app with model selector and stock tabs."""

    # Discover available models
    available_models = list_available_models()
    model_choices = [format_model_choice(m) for m in available_models]
    model_map = {format_model_choice(m): m for m in available_models}

    # Default to first (latest) model
    default_choice = model_choices[0] if model_choices else "No models found"
    default_info = format_model_info(available_models[0]) if available_models else "Không tìm thấy model nào."

    with gr.Blocks(
        title="RL Trading Simulator",
    ) as app:
        gr.Markdown(
            """
            # 🏦 RL Trading Simulator — HOSE Vietnam
            **Mô phỏng giao dịch thời gian thực** trên test-set với DQN Agent đã huấn luyện.
            Chọn model bên dưới, rồi bấm **Next Day ▶** để tiến từng ngày giao dịch.
            """
        )

        # ── Model Selector ────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_choice,
                    label="🧠 Chọn Model (chỉ hiện các run đã train hoàn chỉnh)",
                    interactive=True,
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")

        model_info_md = gr.Markdown(value=default_info, label="Model Info")

        # ── Model load callback ───────────────────────────────────────
        def on_load_model(choice):
            if choice not in model_map:
                return "❌ Model không hợp lệ. Vui lòng chọn lại."

            model = model_map[choice]
            weights_dir = Path(model["weights_dir"])

            # Reload agent with new weights
            model_path = weights_dir / "best_model.pkl"
            if not model_path.exists():
                return f"❌ Không tìm thấy weights: {model_path}"

            try:
                # All simulators share the same agent reference
                # Load new weights once, then reset all simulators
                agent_ref = next(iter(simulators.values())).agent
                agent_ref.load(str(model_path))
                agent_ref.eps = 0.0

                for sim in simulators.values():
                    sim.reset()

                info = format_model_info(model)
                return f"✅ Đã load model thành công!\n\n{info}"
            except Exception as e:
                return f"❌ Lỗi khi load model: {e}"

        load_btn.click(
            fn=on_load_model,
            inputs=[model_dropdown],
            outputs=[model_info_md],
        )

        # ── Stock Tabs ────────────────────────────────────────────────
        stock_ids = list(simulators.keys())
        with gr.Tabs():
            for sid in stock_ids:
                with gr.Tab(label=f"📊 {sid}"):
                    build_tab(sid, simulators[sid], cfg)

    return app


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    """Launch the trading simulation UI."""
    import os
    from dotenv import load_dotenv

    # Load .env (includes PYTHONIOENCODING=utf-8 for Windows)
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg, agent, simulators = init_app()
    app = build_app(simulators, cfg)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
