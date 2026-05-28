"""
src/ui/components.py
════════════════════════════════════════════════════════════════════════════
UI Components — Plotly candlestick chart + formatting helpers.
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df_raw: pd.DataFrame,
    current_step: int,
    buys: list[dict],
    sells: list[dict],
    stock_id: str = "",
    window: int = 20,
) -> go.Figure:
    """
    Create an interactive Plotly candlestick chart up to current_step.

    Args:
        df_raw: Full test-set DataFrame with OHLCV + date
        current_step: Show data from index 0 to current_step (inclusive)
        buys: List of buy trade dicts with 'date' and 'price'
        sells: List of sell trade dicts with 'date' and 'price'
        stock_id: Stock symbol for title
        window: Lookback window size

    Returns:
        Plotly Figure
    """
    # Slice data up to current step
    end_idx = min(current_step + 1, len(df_raw))
    df = df_raw.iloc[:end_idx].copy()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Chưa có dữ liệu")
        return fig

    # Create subplots: candlestick + volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[None, None],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#198754",
            decreasing_line_color="#dc3545",
            increasing_fillcolor="#198754",
            decreasing_fillcolor="#dc3545",
        ),
        row=1, col=1,
    )

    # SMA overlays
    if "sma_10" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sma_10"],
                mode="lines", name="SMA10",
                line=dict(color="#fd7e14", width=1.2),
                opacity=0.8,
            ),
            row=1, col=1,
        )
    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sma_20"],
                mode="lines", name="SMA20",
                line=dict(color="#6f42c1", width=1.2),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    # Buy markers
    if buys:
        buy_dates = [b["date"] for b in buys if b["date"] <= df["date"].iloc[-1]]
        buy_prices = [b["price"] for b in buys if b["date"] <= df["date"].iloc[-1]]
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode="markers",
                    name="▲ MUA",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="#198754",
                        line=dict(width=1.5, color="white"),
                    ),
                ),
                row=1, col=1,
            )

    # Sell markers
    if sells:
        sell_dates = [s["date"] for s in sells if s["date"] <= df["date"].iloc[-1]]
        sell_prices = [s["price"] for s in sells if s["date"] <= df["date"].iloc[-1]]
        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode="markers",
                    name="▼ BÁN",
                    marker=dict(
                        symbol="triangle-down",
                        size=14,
                        color="#dc3545",
                        line=dict(width=1.5, color="white"),
                    ),
                ),
                row=1, col=1,
            )

    # Volume bars
    if "volume" in df.columns:
        colors = np.where(
            df["close"].values >= df["open"].values,
            "rgba(25, 135, 84, 0.6)",   # green
            "rgba(220, 53, 69, 0.6)",   # red
        )
        fig.add_trace(
            go.Bar(
                x=df["date"], y=df["volume"],
                name="Volume",
                marker_color=colors.tolist(),
                showlegend=False,
            ),
            row=2, col=1,
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"📈 {stock_id} — Biểu đồ Nến + Điểm Mua/Bán",
            font=dict(size=16),
        ),
        template="plotly_white",
        height=500,
        margin=dict(l=50, r=20, t=50, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def format_prediction(pred: dict | None) -> str:
    """Format agent prediction as markdown text."""
    if pred is None:
        return "⏸️ **Simulation kết thúc** — không còn ngày giao dịch."

    lines = [
        f"## 🤖 Agent Prediction — Ngày tiếp theo",
        f"",
        f"### Hành động dự kiến: **{pred['action_name']}**",
        f"",
        f"**Confidence:** {pred['confidence']}%",
        f"",
        f"### Q-Values:",
        f"| Action | Q-Value |",
        f"|--------|---------|",
    ]

    for action, qval in pred["q_values"].items():
        marker = " ← " if action == list(pred["q_values"].keys())[pred["action"]] else ""
        lines.append(f"| {action} | {qval:.4f}{marker} |")

    lines.extend([
        f"",
        f"**Actions hợp lệ:** {', '.join(pred['valid_actions'])}",
    ])

    return "\n".join(lines)


def format_action_badge(result: dict) -> str:
    """Format current day's action as a prominent badge."""
    if not result.get("date"):
        return ""

    action_colors = {
        0: "🔄 HOLD",
        1: "🟢 BUY",
        2: "🔴 SELL",
    }

    action_text = action_colors.get(result["action"], result["action_name"])

    lines = [
        f"## Ngày: **{result['date']}**",
        f"",
        f"### Action hôm nay: **{action_text}**",
        f"",
        f"| | |",
        f"|---|---|",
        f"| Giá Close | {result['price']:,.0f} VND |",
        f"| Equity | {result['equity']/1e6:,.1f}M VND |",
        f"| PnL | {result['pnl_pct']:+.2f}% |",
        f"| Vị thế | {'Đang giữ' if result['in_position'] else 'Trống'} |",
        f"| Tiến độ | Ngày {result['day_index']}/{result['total_days']} |",
    ]

    return "\n".join(lines)
