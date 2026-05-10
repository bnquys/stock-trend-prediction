"""
src/visualization/charts.py
════════════════════════════════════════════════════════════════════════════
6 biểu đồ chuyên nghiệp cho báo cáo — nền trắng, label tiếng Việt.
Tập trung vào clarity: người xem không chuyên cũng hiểu ngay.
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings; warnings.filterwarnings("ignore")

# ── Palette ──────────────────────────────────────────────────────────────
C = {
    "bg": "white", "panel": "#f8f9fa", "grid": "#e9ecef",
    "border": "#dee2e6", "text": "#212529", "muted": "#6c757d",
    "buy": "#198754", "buy_l": "#d1e7dd", "sell": "#dc3545", "sell_l": "#f8d7da",
    "blue": "#0d6efd", "blue_l": "#cfe2ff", "orange": "#fd7e14",
    "purple": "#6f42c1", "teal": "#20c997", "yellow": "#ffc107",
}
FIB_C = ["#e03131", "#e8590c", "#2f9e44", "#1971c2", "#7048e8"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": C["panel"],
    "figure.facecolor": C["bg"], "axes.edgecolor": C["border"],
    "axes.linewidth": 0.8, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "grid.alpha": 0.9,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def _ax(ax, title="", sub="", ylabel=""):
    ax.set_title((title + (f"\n{sub}" if sub else "")),
                 loc="left", color=C["text"], fontsize=10,
                 fontweight="bold", pad=7)
    if ylabel: ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)


def _xt(ax, dates, n=10):
    step = max(1, len(dates) // n)
    pos  = range(0, len(dates), step)
    ax.set_xticks(list(pos))
    ax.set_xticklabels([dates.iloc[i].strftime("%d/%m/%y") for i in pos],
                       rotation=30, ha="right", fontsize=8, color=C["muted"])


def _smooth(a, w=5):
    return pd.Series(a).rolling(w, min_periods=1).mean().values


def _buy_sell(ax, df, x, annotate=True):
    bi = df[df["buy_signal"]].index.tolist()
    si = df[df["sell_signal"]].index.tolist()
    if bi:
        ax.scatter(bi, df.loc[bi,"close"]*0.976, marker="^", s=150,
                   color=C["buy"], zorder=8, edgecolors="white", lw=0.8, label="▲ MUA")
        if annotate:
            for b in bi[:20]:
                ax.annotate(f"MUA\n{df.loc[b,'close']:.0f}",
                            (b, df.loc[b,"close"]*0.974),
                            fontsize=6.5, color=C["buy"], ha="center", va="top",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc=C["buy_l"],
                                      ec=C["buy"], alpha=0.9, lw=0.7))
    if si:
        ax.scatter(si, df.loc[si,"close"]*1.024, marker="v", s=150,
                   color=C["sell"], zorder=8, edgecolors="white", lw=0.8, label="▼ BÁN")
        if annotate:
            for s in si[:20]:
                ax.annotate(f"BÁN\n{df.loc[s,'close']:.0f}",
                            (s, df.loc[s,"close"]*1.026),
                            fontsize=6.5, color=C["sell"], ha="center", va="bottom",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc=C["sell_l"],
                                      ec=C["sell"], alpha=0.9, lw=0.7))


def _shade(ax, df):
    in_pos = False
    for i in range(len(df)):
        if df["buy_signal"].iloc[i]: seg=i; in_pos=True
        if df["sell_signal"].iloc[i] and in_pos:
            ax.axvspan(seg, i, alpha=0.06, color=C["buy"], zorder=1); in_pos=False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Price Signals
# ═══════════════════════════════════════════════════════════════════════════
def plot_price_signals(df, out_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(16,12), facecolor=C["bg"],
                              gridspec_kw={"height_ratios":[4,1.2,1.2]})
    fig.suptitle("VNM — Phân tích Kỹ thuật & Điểm Mua/Bán",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.98)
    x = np.arange(len(df))

    # Price panel
    ax = axes[0]
    if "bb_upper" in df.columns:
        ax.fill_between(x, df["bb_lower"].fillna(df["close"]),
                           df["bb_upper"].fillna(df["close"]),
                        alpha=0.09, color=C["blue_l"])
        ax.plot(x, df["bb_upper"], color=C["blue"], lw=0.8, ls="--", alpha=0.5)
        ax.plot(x, df["bb_lower"], color=C["blue"], lw=0.8, ls="--", alpha=0.5)

    # Candlestick bars (simplified)
    closes = df["close"].values; opens = df["open"].values
    hi = df["high"].values; lo = df["low"].values
    for i in range(len(df)):
        up = closes[i] >= opens[i]
        col = C["buy"] if up else C["sell"]
        h   = abs(closes[i]-opens[i]) or closes[i]*0.001
        ax.bar(i, h, bottom=min(closes[i],opens[i]), color=col, width=0.65, alpha=0.85, zorder=3)
        ax.plot([i,i],[lo[i], min(opens[i],closes[i])], color=col, lw=0.7, alpha=0.6)
        ax.plot([i,i],[max(opens[i],closes[i]), hi[i]], color=col, lw=0.7, alpha=0.6)

    if "sma_10" in df.columns: ax.plot(x, df["sma_10"], color=C["orange"], lw=1.2, label="SMA10")
    if "sma_20" in df.columns: ax.plot(x, df["sma_20"], color=C["purple"], lw=1.2, label="SMA20")
    if "ema_20" in df.columns: ax.plot(x, df["ema_20"], color=C["teal"], lw=1.0, ls="--", alpha=0.75, label="EMA20")

    _shade(ax, df); _buy_sell(ax, df, x)
    _ax(ax, "Giá VNM (VND) — Nến Nhật + Bollinger Bands + SMA/EMA",
        "▲ Xanh = Điểm Mua  |  ▼ Đỏ = Điểm Bán  |  Vùng xanh nhạt = Đang giữ cổ phiếu",
        "Giá (VND)")
    _xt(ax, df["date"]); ax.legend(loc="upper left", fontsize=8, ncol=4)
    n_b = df["buy_signal"].sum(); n_s = df["sell_signal"].sum()
    ax.text(0.99, 0.98, f"▲ {n_b} lần MUA   ▼ {n_s} lần BÁN",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            fontweight="bold", color=C["text"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C["border"], alpha=0.95))

    # MACD
    ax2 = axes[1]
    if "macd_histogram" in df.columns:
        hist = df["macd_histogram"].fillna(0).values
        ax2.bar(x, hist, color=np.where(hist>=0,C["buy"],C["sell"]), alpha=0.75, width=1)
        ax2.fill_between(x, hist, 0, where=(hist>0), alpha=0.1, color=C["buy"])
        ax2.fill_between(x, hist, 0, where=(hist<0), alpha=0.1, color=C["sell"])
        ax2.axhline(0, color=C["muted"], lw=0.8, ls="--")
    _ax(ax2, "MACD Histogram", "Dương (xanh) = xu thế tăng  |  Âm (đỏ) = xu thế giảm")
    _xt(ax2, df["date"])

    # RSI
    ax3 = axes[2]
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"].fillna(50).values
        ax3.plot(x, rsi, color=C["purple"], lw=1.3, label="RSI 14")
        ax3.axhline(70, color=C["sell"], lw=1.0, ls="--", alpha=0.8, label="Quá mua (70)")
        ax3.axhline(30, color=C["buy"],  lw=1.0, ls="--", alpha=0.8, label="Quá bán (30)")
        ax3.fill_between(x, rsi, 70, where=(rsi>70), alpha=0.15, color=C["sell"])
        ax3.fill_between(x, rsi, 30, where=(rsi<30), alpha=0.15, color=C["buy"])
        ax3.set_ylim(0, 100)
    _ax(ax3, "RSI (14 ngày)", ">70 cân nhắc BÁN  |  <30 cân nhắc MUA", "RSI")
    _xt(ax3, df["date"]); ax3.legend(loc="upper left", fontsize=8, ncol=3)

    plt.tight_layout(rect=[0,0,1,0.97])
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches="tight"); print(f"[Chart] 1 → {out_path}")
    plt.close(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. Equity Curve + Drawdown
# ═══════════════════════════════════════════════════════════════════════════
def plot_equity(df, metrics, initial_cap=100_000_000, out_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(14,9), facecolor=C["bg"],
                              gridspec_kw={"height_ratios":[3,1]})
    fig.suptitle("Hiệu suất Danh mục — RL Agent vs Mua & Giữ (Buy & Hold)",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.98)
    x = np.arange(len(df))
    bh = initial_cap * (df["close"] / df["close"].iloc[0])

    ax = axes[0]
    if "rl_equity" in df.columns:
        eq = df["rl_equity"].values
        ax.plot(x, eq/1e6, color=C["blue"], lw=2.2, label="RL Agent", zorder=4)
        ax.fill_between(x, eq/1e6, bh.values/1e6, where=(eq>=bh.values),
                        alpha=0.13, color=C["buy"], label="Agent tốt hơn B&H")
        ax.fill_between(x, eq/1e6, bh.values/1e6, where=(eq<bh.values),
                        alpha=0.13, color=C["sell"], label="Agent tệ hơn B&H")
    ax.plot(x, bh/1e6, color=C["orange"], lw=1.6, ls="--", label="Buy & Hold")
    ax.axhline(initial_cap/1e6, color=C["muted"], lw=0.8, ls=":", label="Vốn ban đầu")

    m = metrics; bh_r = round((df["close"].iloc[-1]-df["close"].iloc[0])/df["close"].iloc[0]*100,2)
    info = (f"RL Return:    {m.get('total_return',0):+.2f}%\n"
            f"B&H Return:  {bh_r:+.2f}%\n"
            f"Sharpe:       {m.get('sharpe',0):.4f}\n"
            f"MDD:          {m.get('max_drawdown',0):.2f}%\n"
            f"Win Rate:     {m.get('win_rate',0):.1f}%\n"
            f"Trades:       {m.get('n_trades',0)}")
    ax.text(0.99, 0.05, info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=C["text"], family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C["border"], alpha=0.95))
    _ax(ax, "Giá trị Danh mục (Triệu VND)", "Xanh=Agent tốt hơn  |  Đỏ=Agent tệ hơn", "M VND")
    _xt(ax, df["date"]); ax.legend(loc="upper left", fontsize=9, ncol=2)

    ax2 = axes[1]
    if "rl_equity" in df.columns:
        eq_arr = df["rl_equity"].values
        peak   = np.maximum.accumulate(eq_arr)
        dd     = (eq_arr - peak) / (peak + 1e-9) * 100
        ax2.fill_between(x, dd, 0, alpha=0.6, color=C["sell"], label="Drawdown")
        ax2.plot(x, dd, color=C["sell"], lw=0.8)
        ax2.axhline(0, color=C["muted"], lw=0.7)
        wi = int(np.argmin(dd))
        ax2.annotate(f"MDD: {dd.min():.1f}%", (wi, dd.min()),
                     xytext=(wi+max(1,len(df)//15), dd.min()-1),
                     fontsize=8.5, color=C["sell"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C["sell"], lw=0.9))
    _ax(ax2, "Drawdown (%)", "0% = đang ở đỉnh  |  Số âm lớn = rủi ro cao", "DD (%)")
    _xt(ax2, df["date"]); ax2.legend(fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.97])
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches="tight"); print(f"[Chart] 2 → {out_path}")
    plt.close(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. Training Curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_training(history, out_path=None):
    if not history:
        print("[Chart] No training history."); return None
    fig, axes = plt.subplots(2, 2, figsize=(15,10), facecolor=C["bg"])
    fig.suptitle("Quá trình Huấn luyện DQN Agent — VNM",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.98)

    eps  = [h["episode"] for h in history]
    tr_r = [h["train_return"] for h in history]
    vl_r = [h["val_return"]   for h in history]
    tr_sh= [h.get("train_sharpe",0) for h in history]
    vl_sh= [h.get("val_sharpe",0)   for h in history]
    loss = [h["avg_loss"] for h in history if h.get("avg_loss",0) > 1e-9]
    epss = [h["epsilon"]  for h in history]
    tr_wr= [h.get("train_winrate",0) for h in history]
    vl_wr= [h.get("val_winrate",0)   for h in history]
    best_idx = int(np.argmax(vl_r))

    # 1. Return
    ax = axes[0,0]
    ax.plot(eps, tr_r, color="#adb5bd", lw=0.7, alpha=0.5)
    ax.plot(eps, _smooth(tr_r), color=C["blue"], lw=1.8, label="Train (smoothed)")
    ax.plot(eps, vl_r, color="#f4a261", lw=0.7, alpha=0.5)
    ax.plot(eps, _smooth(vl_r), color=C["sell"], lw=1.8, label="Val (smoothed)")
    ax.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax.axvline(eps[best_idx], color=C["buy"], lw=1.5, ls="--", alpha=0.9,
               label=f"Best ep={eps[best_idx]}: {vl_r[best_idx]:+.1f}%")
    ax.fill_between(eps, _smooth(vl_r), 0,
                    where=[v>0 for v in _smooth(vl_r)], alpha=0.1, color=C["buy"])
    _ax(ax, "1. Lợi nhuận / Episode (%)",
        "Tốt khi Val (cam) tăng dần & ổn định", "Return (%)")
    ax.set_xlabel("Episode", color=C["muted"]); ax.legend(fontsize=8)

    # 2. Loss
    ax2 = axes[0,1]
    if loss:
        ep_l = eps[:len(loss)]
        ax2.plot(ep_l, loss, color="#dee2e6", lw=0.6, alpha=0.6)
        ax2.plot(ep_l, _smooth(loss, 10), color=C["purple"], lw=1.8, label="TD Loss (smooth)")
    _ax(ax2, "2. TD Loss", "Giảm dần = model đang học", "Avg TD Loss")
    ax2.set_xlabel("Episode", color=C["muted"]); ax2.legend(fontsize=8)

    # 3. Win Rate
    ax3 = axes[1,0]
    if any(v>0 for v in tr_wr+vl_wr):
        ax3.plot(eps, _smooth(tr_wr), color=C["blue"], lw=1.8, label="Train")
        ax3.plot(eps, _smooth(vl_wr), color=C["sell"], lw=1.8, label="Val")
        ax3.axhline(50, color=C["muted"], lw=1.0, ls="--", label="50% baseline")
        ax3.set_ylim(0, 100)
    _ax(ax3, "3. Win Rate (%)", ">50% = thắng nhiều hơn thua ", "Win %")
    ax3.set_xlabel("Episode", color=C["muted"]); ax3.legend(fontsize=8)

    # 4. Epsilon + Val Sharpe
    ax4 = axes[1,1]
    ax4.plot(eps, epss, color=C["orange"], lw=1.8, label="Epsilon ε")
    ax4.set_ylabel("Epsilon ε", color=C["orange"], fontsize=9)
    ax4r = ax4.twinx()
    ax4r.plot(eps, _smooth(vl_sh), color=C["teal"], lw=1.8, ls="--", label="Val Sharpe")
    ax4r.set_ylabel("Sharpe", color=C["teal"], fontsize=9)
    ax4r.tick_params(axis="y", colors=C["teal"], labelsize=8)
    l1,b1 = ax4.get_legend_handles_labels(); l2,b2 = ax4r.get_legend_handles_labels()
    ax4.legend(l1+l2, b1+b2, fontsize=8, loc="upper right")
    _ax(ax4, "4. Epsilon Decay & Sharpe",
        "ε giảm = bớt random  |  Sharpe tăng = hiệu quả hơn")
    ax4.set_xlabel("Episode", color=C["muted"])

    plt.tight_layout(rect=[0,0,1,0.97])
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches="tight"); print(f"[Chart] 3 → {out_path}")
    plt.close(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Trade Analysis
# ═══════════════════════════════════════════════════════════════════════════
def plot_trades(trades, out_path=None):
    closed = [t for t in trades if "pnl_pct" in t]
    if not closed:
        print("[Chart] No closed trades."); return None
    pnls  = [t["pnl_pct"] for t in closed]
    holds = [t.get("hold_days",0) for t in closed]
    wins  = [p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
    cumul = np.cumsum(pnls)

    fig, axes = plt.subplots(2,2, figsize=(14,10), facecolor=C["bg"])
    fig.suptitle(f"Phân tích {len(closed)} Giao dịch Đã Đóng",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.98)

    # 1. PnL distribution
    ax=axes[0,0]
    if wins:  ax.hist(wins,  bins=max(5,len(wins)//2),  color=C["buy"],  alpha=0.8, ec="white", lw=0.5, label=f"Thắng ({len(wins)})")
    if losses:ax.hist(losses,bins=max(5,len(losses)//2),color=C["sell"], alpha=0.8, ec="white", lw=0.5, label=f"Thua ({len(losses)})")
    ax.axvline(0, color=C["muted"], lw=1.5, ls="--")
    ax.axvline(np.mean(pnls), color=C["orange"], lw=1.5, label=f"TB: {np.mean(pnls):+.2f}%")
    wr=len(wins)/max(len(closed),1)*100; pf=sum(wins)/(abs(sum(losses))+1e-9) if losses else 999
    ax.text(0.98,0.98,f"Win Rate: {wr:.1f}%\nTB thắng: +{np.mean(wins):.2f}%\nTB thua: {np.mean(losses):.2f}%\nProfit Factor: {pf:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",fc="white",ec=C["border"],alpha=0.95))
    _ax(ax,"1. Phân phối Lãi/Lỗ mỗi Lệnh","Xanh=thắng  |  Đỏ=thua","Số lệnh")
    ax.set_xlabel("PnL (%)", color=C["muted"]); ax.legend(fontsize=9)

    # 2. Waterfall
    ax2=axes[0,1]
    bc=[C["buy"] if p>0 else C["sell"] for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=bc, alpha=0.85, ec="white", lw=0.4)
    ax2.plot(range(len(pnls)), cumul, color=C["blue"], lw=2.0, marker="o", ms=3.5, zorder=5, label="Tích lũy")
    ax2.axhline(0, color=C["muted"], lw=0.9, ls="--")
    if len(pnls)>0:
        fc=C["buy"] if cumul[-1]>0 else C["sell"]
        ax2.text(len(pnls)-1,cumul[-1],f" {cumul[-1]:+.1f}%",ha="left",va="center",fontsize=9,color=fc,fontweight="bold")
    _ax(ax2,"2. Lãi/Lỗ mỗi Lệnh + Tích lũy","Cột=từng lệnh  |  Đường xanh=tổng tích lũy","PnL (%)")
    ax2.set_xlabel("Lệnh #", color=C["muted"]); ax2.legend(fontsize=8)

    # 3. Holding period
    ax3=axes[1,0]
    hold_arr=np.array(holds)
    if hold_arr.max()>0:
        ax3.hist(hold_arr,bins=min(20,len(holds)),color=C["blue"],alpha=0.8,ec="white",lw=0.5,label=f"Tất cả (TB: {hold_arr.mean():.1f}N)")
        wh=[holds[i] for i,p in enumerate(pnls) if p>0]
        lh=[holds[i] for i,p in enumerate(pnls) if p<=0]
        if wh: ax3.axvline(np.mean(wh),color=C["buy"],lw=1.5,ls="--",label=f"TB thắng: {np.mean(wh):.1f}N")
        if lh: ax3.axvline(np.mean(lh),color=C["sell"],lw=1.5,ls="--",label=f"TB thua: {np.mean(lh):.1f}N")
    _ax(ax3,"3. Số ngày Nắm giữ","Thời gian giữ cổ phiếu mỗi lệnh","Số lệnh")
    ax3.set_xlabel("Số ngày", color=C["muted"]); ax3.legend(fontsize=8)

    # 4. Exit reasons
    ax4=axes[1,1]
    rm={"SELL":"Tín hiệu\nAgent","sl":"Cắt lỗ\n(SL)","tp":"Chốt lời\n(TP)","maxhold":"Hết hạn\nMax Hold","AUTO_EXIT":"Auto\nExit"}
    cnt:dict={}; apnl:dict={}
    for t in closed:
        r=t.get("exit_reason",t.get("type","SELL"))
        cnt[r]=cnt.get(r,0)+1; apnl.setdefault(r,[]).append(t["pnl_pct"])
    if cnt:
        lbs=[rm.get(r,r) for r in cnt]; vals=[cnt[r] for r in cnt]
        avgs=[np.mean(apnl[r]) for r in cnt]
        bc4=[C["buy"] if a>0 else C["sell"] for a in avgs]
        bars=ax4.bar(lbs,vals,color=bc4,alpha=0.85,ec="white",lw=0.5)
        for bar,a in zip(bars,avgs):
            ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                     f"{a:+.2f}%",ha="center",va="bottom",fontsize=9,fontweight="bold",color=C["text"])
    _ax(ax4,"4. Lý do Thoát Lệnh","Số trên cột=PnL trung bình |  Xanh>0, Đỏ<0","Số lệnh")

    plt.tight_layout(rect=[0,0,1,0.97])
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches="tight"); print(f"[Chart] 4 → {out_path}")
    plt.close(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. Elliott Waves
# ═══════════════════════════════════════════════════════════════════════════
def plot_elliott(df, patterns, pivots, out_path=None):
    fig,axes=plt.subplots(2,1,figsize=(16,11),facecolor=C["bg"],
                           gridspec_kw={"height_ratios":[3,1]})
    fig.suptitle("Elliott Wave Analysis — VNM ",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.98)
    x=np.arange(len(df))

    ax=axes[0]
    ax.plot(x, df["close"], color="#374151", lw=1.3, alpha=0.85, label="Giá đóng cửa")
    if "sma_20" in df.columns:
        ax.plot(x, df["sma_20"], color=C["purple"], lw=1.0, alpha=0.7, ls="--", label="SMA20")

    if pivots:
        ph=[p for p in pivots if p.kind=='H' and p.idx<len(df)]
        pl=[p for p in pivots if p.kind=='L' and p.idx<len(df)]
        ax.scatter([p.idx for p in ph],[p.price for p in ph],s=35,color=C["sell"],alpha=0.6,zorder=6,label=f"Pivot High ({len(ph)})")
        ax.scatter([p.idx for p in pl],[p.price for p in pl],s=35,color=C["buy"], alpha=0.6,zorder=6,label=f"Pivot Low ({len(pl)})")

    sorted_pats=sorted(patterns,key=lambda p:p.confidence,reverse=True) if patterns else []
    for pi,pat in enumerate(sorted_pats[:3]):
        pvs=pat.pivots
        col=C["blue"] if "bull" in pat.direction else C["sell"]
        wx=[p.idx for p in pvs]; wy=[p.price for p in pvs]
        ax.plot(wx,wy,color=col,lw=2.0,alpha=0.75,zorder=7,marker="D",ms=6,mec="white",mew=0.8)
        labels=(["1","2","3","4","5"] if len(pvs)-1==5 else ["A","B","C","D"][:len(pvs)])
        for j,(wx_j,wy_j,wl) in enumerate(zip(wx,wy,labels)):
            off=-0.018 if pvs[j].kind=='L' else 0.018
            ax.text(wx_j,wy_j*(1+off),wl,ha="center",va="bottom" if pvs[j].kind=='H' else "top",
                    fontsize=9,color=col,fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.15",fc="white",ec=col,lw=0.8,alpha=0.9))
        if pi==0:
            pmin,pmax=df["close"].min()*0.97,df["close"].max()*1.03
            for (ratio,lvl),fc in zip(pat.fib_levels.items(),FIB_C):
                if pmin<lvl<pmax:
                    ax.axhline(lvl,color=fc,lw=0.9,ls=":",alpha=0.8)
                    ax.text(len(df)*0.01,lvl,f" Fib {ratio}",color=fc,fontsize=6.5,va="bottom",fontweight="bold")
            if pat.support>0:
                ax.axhline(pat.support,color=C["buy"],lw=1.0,ls="--",alpha=0.7)
                ax.text(len(df)*0.99,pat.support,f"Support {pat.support:.1f} ",color=C["buy"],fontsize=8,ha="right",va="top",fontweight="bold")
            if pat.resistance>0:
                ax.axhline(pat.resistance,color=C["sell"],lw=1.0,ls="--",alpha=0.7)
                ax.text(len(df)*0.99,pat.resistance,f"Resistance {pat.resistance:.1f} ",color=C["sell"],fontsize=8,ha="right",va="bottom",fontweight="bold")
            if pat.target>0 and pmin<pat.target<df["close"].max()*1.4:
                ax.axhline(pat.target,color="#7048e8",lw=1.2,ls="-.",alpha=0.9)
                ax.text(len(df)*0.5,pat.target,f" ← Target 161.8% = {pat.target:.1f}",
                        color="#7048e8",fontsize=8,fontweight="bold",va="bottom")

    if "buy_signal" in df.columns: _buy_sell(ax,df,x,annotate=False)

    n_imp=sum(1 for p in patterns if p.pattern=='impulse')
    n_abc=sum(1 for p in patterns if p.pattern=='abc')
    ax.text(0.99,0.98,
            f"Pivots: {len(pivots)}\nImpulse 5-sóng: {n_imp}\nABC Correction: {n_abc}",
            transform=ax.transAxes,ha="right",va="top",fontsize=9,family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",fc="white",ec=C["border"],alpha=0.95))
    _ax(ax,"Elliott Wave + Fibonacci + Support/Resistance",
        "Kim cương=pivot  |  1-5=impulse  |  A-B-C=corrective  |  Fib=mức thoái lui","Giá (VND)")
    _xt(ax,df["date"]); ax.legend(loc="upper left",fontsize=8,ncol=3)

    ax2=axes[1]
    if "pattern_conf" in df.columns:
        conf=df["pattern_conf"].fillna(0).values
        ax2.fill_between(x,conf,0,alpha=0.5,color=C["blue"],label="Độ tin cậy pattern")
        ax2.plot(x,conf,color=C["blue"],lw=1.0)
        ax2.axhline(0.5,color=C["muted"],lw=0.7,ls="--",label="Ngưỡng 0.5")
        ax2.set_ylim(0,1.1)
    if "elliott_signal" in df.columns:
        ax2r=ax2.twinx()
        ax2r.plot(x,df["elliott_signal"].fillna(0).values,color=C["orange"],lw=1.3,ls="--",label="Elliott Signal [-1,+1]")
        ax2r.axhline(0,color=C["muted"],lw=0.5); ax2r.set_ylabel("Signal",color=C["orange"],fontsize=9)
        ax2r.tick_params(axis="y",colors=C["orange"],labelsize=8)
        ax2r.legend(fontsize=8,loc="upper right")
    _ax(ax2,"Độ tin cậy & Tín hiệu Elliott","Xanh cao=pattern rõ  |  Cam=tín hiệu (+tăng/-giảm)","Conf")
    _xt(ax2,df["date"]); ax2.legend(fontsize=8,loc="upper left")

    plt.tight_layout(rect=[0,0,1,0.97])
    if out_path: plt.savefig(out_path,dpi=150,bbox_inches="tight"); print(f"[Chart] 5 → {out_path}")
    plt.close(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dashboard (1 trang báo cáo)
# ═══════════════════════════════════════════════════════════════════════════
def plot_dashboard(df, metrics, history, trades, patterns, pivots,
                   initial_cap=100_000_000, out_path=None):
    fig=plt.figure(figsize=(22,30),facecolor=C["bg"])
    gs=gridspec.GridSpec(5,2,figure=fig,height_ratios=[2.8,1.4,1.2,1.2,1.3],hspace=0.52,wspace=0.30)
    x=np.arange(len(df)); dates=df["date"]

    # Row 0: Price (full width)
    ax0=fig.add_subplot(gs[0,:])
    closes=df["close"].values; opens=df["open"].values
    for i in range(0,len(df),max(1,len(df)//250)):
        up=closes[i]>=opens[i]; col=C["buy"] if up else C["sell"]
        h=abs(closes[i]-opens[i]) or closes[i]*0.001
        ax0.bar(i,h,bottom=min(closes[i],opens[i]),color=col,width=0.65,alpha=0.85,zorder=3)
        ax0.plot([i,i],[df["low"].iloc[i],min(opens[i],closes[i])],color=col,lw=0.5,alpha=0.5)
        ax0.plot([i,i],[max(opens[i],closes[i]),df["high"].iloc[i]],color=col,lw=0.5,alpha=0.5)
    if "sma_10" in df.columns: ax0.plot(x,df["sma_10"],color=C["orange"],lw=1.1,alpha=0.85,label="SMA10")
    if "sma_20" in df.columns: ax0.plot(x,df["sma_20"],color=C["purple"],lw=1.1,alpha=0.85,label="SMA20")
    if pivots:
        ph=[p.idx for p in pivots if p.kind=='H' and p.idx<len(df)]
        pl=[p.idx for p in pivots if p.kind=='L' and p.idx<len(df)]
        ax0.scatter(ph,[df["close"].iloc[i] for i in ph],s=22,color=C["sell"],alpha=0.5,zorder=5)
        ax0.scatter(pl,[df["close"].iloc[i] for i in pl],s=22,color=C["buy"],alpha=0.5,zorder=5)
    if patterns:
        best=sorted(patterns,key=lambda p:p.confidence,reverse=True)[0]
        pmin,pmax=df["close"].min()*0.97,df["close"].max()*1.03
        for (r,lvl),fc in zip(best.fib_levels.items(),FIB_C):
            if pmin<lvl<pmax:
                ax0.axhline(lvl,color=fc,lw=0.8,ls=":",alpha=0.8)
                ax0.text(len(df)*0.005,lvl,f"Fib {r}",color=fc,fontsize=6,va="bottom")
    _shade(ax0,df); _buy_sell(ax0,df,x,annotate=True)
    _ax(ax0,"VNM — Giá + SMA/BB + Elliott Pivots + Điểm MUA/BÁN",
        "▲ Xanh=Mua  |  ▼ Đỏ=Bán  |  Vùng xanh nhạt=Đang giữ CP","Giá (VND)")
    ax0.legend(loc="upper left",fontsize=8,ncol=4); _xt(ax0,dates)

    # Row 1L: Equity
    ax1=fig.add_subplot(gs[1,0])
    bh=initial_cap*(df["close"]/df["close"].iloc[0])
    if "rl_equity" in df.columns:
        eq=df["rl_equity"].values
        ax1.plot(x,eq/1e6,color=C["blue"],lw=1.8,label="RL Agent",zorder=4)
        ax1.fill_between(x,eq/1e6,bh.values/1e6,where=(eq>=bh.values),alpha=0.13,color=C["buy"])
        ax1.fill_between(x,eq/1e6,bh.values/1e6,where=(eq<bh.values),alpha=0.13,color=C["sell"])
    ax1.plot(x,bh/1e6,color=C["orange"],lw=1.3,ls="--",label="Buy & Hold")
    ax1.axhline(initial_cap/1e6,color=C["muted"],lw=0.7,ls=":")
    _ax(ax1,"Equity Curve (M VND)","RL vs Buy & Hold","M VND")
    ax1.legend(fontsize=8); _xt(ax1,dates)

    # Row 1R: Training
    ax2=fig.add_subplot(gs[1,1])
    if history:
        ep=[h["episode"] for h in history]
        tr=[h["train_return"] for h in history]; vl=[h["val_return"] for h in history]
        ax2.plot(ep,_smooth(tr),color=C["blue"],lw=1.6,label="Train")
        ax2.plot(ep,_smooth(vl),color=C["sell"],lw=1.6,label="Val")
        ax2.axhline(0,color=C["muted"],lw=0.7,ls="--")
        bi=int(np.argmax(vl))
        ax2.axvline(ep[bi],color=C["buy"],lw=1.2,ls="--",alpha=0.8,label=f"Best ep={ep[bi]}")
    _ax(ax2,"Training Curves","Return theo episode")
    ax2.set_xlabel("Episode",color=C["muted"]); ax2.set_ylabel("Return (%)",color=C["muted"])
    ax2.legend(fontsize=8)

    # Row 2L: MACD
    ax3=fig.add_subplot(gs[2,0])
    if "macd_histogram" in df.columns:
        hist=df["macd_histogram"].fillna(0).values
        ax3.bar(x,hist,color=np.where(hist>=0,C["buy"],C["sell"]),alpha=0.75,width=1)
        ax3.axhline(0,color=C["muted"],lw=0.7,ls="--")
    _ax(ax3,"MACD Histogram","Dương=tăng  |  Âm=giảm"); _xt(ax3,dates)

    # Row 2R: RSI
    ax4=fig.add_subplot(gs[2,1])
    if "rsi_14" in df.columns:
        rsi=df["rsi_14"].fillna(50).values
        ax4.plot(x,rsi,color=C["purple"],lw=1.3)
        ax4.axhline(70,color=C["sell"],lw=0.9,ls="--",label="Quá mua (70)")
        ax4.axhline(30,color=C["buy"],lw=0.9,ls="--",label="Quá bán (30)")
        ax4.fill_between(x,rsi,70,where=(rsi>70),alpha=0.15,color=C["sell"])
        ax4.fill_between(x,rsi,30,where=(rsi<30),alpha=0.15,color=C["buy"])
        ax4.set_ylim(0,100)
    _ax(ax4,"RSI (14)","RSI"); ax4.legend(fontsize=8); _xt(ax4,dates)

    # Row 3L: Volume
    ax5=fig.add_subplot(gs[3,0])
    chg=df["close"].pct_change().fillna(0)
    ax5.bar(x,df["volume"]/1e6,color=np.where(chg>=0,C["buy"],C["sell"]),alpha=0.75,width=1)
    if "volume_sma_20" in df.columns:
        ax5.plot(x,df["volume_sma_20"]/1e6,color=C["orange"],lw=1.2,label="SMA20")
    _ax(ax5,"Khối lượng (M CP)","Xanh=tăng  |  Đỏ=giảm","M CP")
    ax5.legend(fontsize=8); _xt(ax5,dates)

    # Row 3R: Trade waterfall
    ax6=fig.add_subplot(gs[3,1])
    closed=[t for t in trades if "pnl_pct" in t]
    if closed:
        pnls=[t["pnl_pct"] for t in closed]; cumul=np.cumsum(pnls)
        bc=[C["buy"] if p>0 else C["sell"] for p in pnls]
        ax6.bar(range(len(pnls)),pnls,color=bc,alpha=0.85,ec="white",lw=0.4)
        ax6.plot(range(len(pnls)),cumul,color=C["blue"],lw=1.8,marker="o",ms=3)
        ax6.axhline(0,color=C["muted"],lw=0.8,ls="--")
    _ax(ax6,"PnL mỗi Lệnh + Tích lũy","Cột=từng lệnh  |  Đường=tích lũy","PnL (%)")
    ax6.set_xlabel("Lệnh #",color=C["muted"])

    # Row 4: Metrics (full width)
    ax7=fig.add_subplot(gs[4,:])
    ax7.set_facecolor("#eef2ff"); ax7.axis("off")
    for sp in ax7.spines.values(): sp.set_color(C["border"])
    m=metrics; bh_r=round((df["close"].iloc[-1]-df["close"].iloc[0])/df["close"].iloc[0]*100,2)
    col1=[("Tổng lợi nhuận",f"{m.get('total_return',0):+.2f}%"),
          ("ARR (lợi nhuận/năm)",f"{m.get('arr',0):+.2f}%"),
          ("B&H Return",f"{bh_r:+.2f}%"),
          ("Alpha vs B&H",f"{(m.get('total_return',0)-bh_r):+.2f}%")]
    col2=[("Sharpe Ratio",f"{m.get('sharpe',0):.4f}"),
          ("Max Drawdown",f"{m.get('max_drawdown',0):.2f}%"),
          ("Win Rate",f"{m.get('win_rate',0):.1f}%"),
          ("Profit Factor",f"{m.get('profit_factor',0):.2f}")]
    col3=[("Tổng lệnh",str(m.get('n_trades',0))),
          ("Avg Win",f"{m.get('avg_win',0):+.2f}%"),
          ("Avg Loss",f"{m.get('avg_loss',0):+.2f}%"),
          ("Stop-Loss exits",str(m.get('n_sl',0)))]
    for ci,col in enumerate([col1,col2,col3]):
        xo=0.02+ci*0.33
        for ri,(lbl,val) in enumerate(col):
            y=0.85-ri*0.22
            ax7.text(xo,y,lbl,transform=ax7.transAxes,color=C["muted"],fontsize=9)
            try:
                vf=float(val.replace("+","").replace("%",""))
                vc=C["buy"] if vf>0 else C["sell"] if vf<0 else C["text"]
            except: vc=C["text"]
            ax7.text(xo+0.22,y,val,transform=ax7.transAxes,color=vc,fontsize=11,fontweight="bold")
    ax7.text(0.68,0.92,"Model Architecture",transform=ax7.transAxes,color=C["blue"],fontsize=9,fontweight="bold")
    cfg=[ "Model: Double Dueling DQN + N-step(3)",
         f"Obs: 20-bar × {m.get('n_feat',36)} features + 5 portfolio state",
         "Optim: Adam lr=1e-4 | Soft target τ=0.01",
         "VN: T+2 | ±7% HOSE | Phí 0.15% | Slippage 0.03%"]
    for ci,line in enumerate(cfg):
        ax7.text(0.68,0.78-ci*0.155,line,transform=ax7.transAxes,
                 color=C["text"] if ci==0 else C["muted"],fontsize=8)

    fig.suptitle("VNM TRADING DASHBOARD — DQN + Elliott Wave | HOSE Vietnam",
                 color=C["text"],fontsize=14,fontweight="bold",y=0.999)
    if out_path: plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor=C["bg"]); print(f"[Chart] 6 → {out_path}")
    plt.close(); return fig


def generate_all(result_df, metrics, trades, history,
                 patterns, pivots, initial_cap, out_dir):
    """Sinh tất cả 6 chart một lần."""
    import os; os.makedirs(out_dir, exist_ok=True)
    plot_price_signals(result_df,                       f"{out_dir}/1_price_signals.png")
    plot_equity(result_df, metrics, initial_cap,        f"{out_dir}/2_equity_drawdown.png")
    plot_training(history,                              f"{out_dir}/3_training_curves.png")
    plot_trades(trades,                                 f"{out_dir}/4_trade_analysis.png")
    plot_elliott(result_df, patterns, pivots,           f"{out_dir}/5_elliott_waves.png")
    plot_dashboard(result_df, metrics, history, trades, patterns, pivots,
                   initial_cap,                         f"{out_dir}/0_dashboard.png")
    print(f"[Charts] 6 biểu đồ → {out_dir}/")
