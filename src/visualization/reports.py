"""
src/visualization/reports.py
════════════════════════════════════════════════════════════════════════════
Export functions: ROI Table, LLM Integration Summary.
Tách ra từ train.py để tái sử dụng trong Trainer và scripts.
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def export_roi_table(
    trades: list[dict],
    out_dir: str | Path,
    symbol: str = "VNM",
) -> Path | None:
    """
    Xuất bảng ROI từ danh sách trades.
    Returns path to CSV file hoặc None nếu không có trades.
    """
    if not trades:
        log.debug("Không có giao dịch nào để tạo bảng ROI.")
        return None

    closed_trades = [t for t in trades if t["type"] in ["SELL", "AUTO_EXIT"]]
    if not closed_trades:
        log.debug("Chưa có giao dịch đóng nào để tính ROI.")
        return None

    buy_trades = [t for t in trades if t["type"] == "BUY"]

    roi_data = []
    buy_idx = 0
    for ct in closed_trades:
        buy_date = "N/A"
        shares = 0
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
            "ROI (%)": pnl_pct,
        })

    df_roi = pd.DataFrame(roi_data)
    out_path = Path(out_dir) / f"roi_table_{symbol}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_roi.to_csv(out_path, index=False)
    log.debug(f"Đã xuất bảng ROI tại: {out_path}")
    return out_path


def export_llm_integration_report(
    result_df: pd.DataFrame,
    test_metrics: dict,
    out_dir: str | Path,
    symbol: str = "VNM",
) -> Path | None:
    """
    Xuất file JSON + Markdown summary cho LLM integration.
    Returns path to JSON file hoặc None nếu không có data.
    """
    from src.rl.env.trading_env import TradingEnv

    if result_df is None or result_df.empty:
        return None

    last_row = result_df.iloc[-1]
    prev_row = result_df.iloc[-2] if len(result_df) > 1 else last_row

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

    llm_data = {
        "report_date": str(last_row.get("date", "N/A"))[:10],
        "ticker": symbol,
        "latest_close_price": float(last_row.get("close", 0)),
        "rl_recommendation": rl_rec,
        "technical_context": {
            "rsi": round(float(rsi), 2),
            "macd": round(float(macd), 4),
            "sma_20_trend": trend_sma20,
        },
        "rl_performance_history": {
            "win_rate_pct": float(test_metrics.get("win_rate", 0)),
            "sharpe_ratio": float(test_metrics.get("sharpe", 0)),
            "profit_factor": float(test_metrics.get("pf", 0)),
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"llm_integration_summary_{symbol}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=4)

    # Markdown summary
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
    md_path = out_dir / f"llm_summary_{symbol}.md"
    md_path.write_text(md_table, encoding="utf-8")

    log.debug(f"Đã xuất LLM summary tại: {out_json}")
    return out_json
