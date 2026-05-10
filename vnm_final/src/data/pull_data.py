"""
src/data/pull_data.py
════════════════════════════════════════════════════════════════════════════
Pull dữ liệu từ VNStock API, tính đầy đủ 43 features.
Hỗ trợ cập nhật tăng dần sau phiên giao dịch (15:30 VN = UTC+7).

DÙNG:
    # Pull lần đầu
    python src/data/pull_data.py --symbol VNM --start 2021-10-01

    # Cập nhật hàng ngày (chạy sau 15:30 VN)
    python src/data/pull_data.py --update

    # Lập lịch cron chạy lúc 16:00 VN (09:00 UTC):
    0 9 * * 1-5 cd /path/to/project && python src/data/pull_data.py --update

Schema output (43 cột) — khớp VNM_2225.csv:
    date, open, high, low, close, volume,
    sma_10, sma_20, ema_20, macd_histogram, rsi_14, atr_14, cci_14,
    momentum_10, roc_12, obv, log_return, body_size, daily_range,
    upper_shadow, lower_shadow, rolling_std_20, historical_volatility_20,
    return_lag_1..5, rolling_max_20, rolling_min_20, rolling_mean_20,
    distance_from_high_20, distance_from_low_20, volume_sma_20,
    volume_ratio, volume_change, price_volume,
    vnindex, vnindex_volume, vnindex_return,
    correlation_market_20, beta_20, day_of_week
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Market timing ─────────────────────────────────────────────────────────
MARKET_CLOSE_VN = datetime.strptime("15:30", "%H:%M").time()
VN_UTC_OFFSET   = 7

SCHEMA_COLS = [
    "date", "open", "high", "low", "close", "volume",
    "sma_10", "sma_20", "ema_20", "macd_histogram",
    "rsi_14", "atr_14", "cci_14", "momentum_10", "roc_12",
    "obv", "log_return", "body_size", "daily_range",
    "upper_shadow", "lower_shadow",
    "rolling_std_20", "historical_volatility_20",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    "rolling_max_20", "rolling_min_20", "rolling_mean_20",
    "distance_from_high_20", "distance_from_low_20",
    "volume_sma_20", "volume_ratio", "volume_change", "price_volume",
    "vnindex", "vnindex_volume", "vnindex_return",
    "correlation_market_20", "beta_20", "day_of_week",
]


# ═══════════════════════════════════════════════════════════════════════════
# Timing helpers
# ═══════════════════════════════════════════════════════════════════════════

def vn_now() -> datetime:
    """Thời gian hiện tại theo múi giờ Việt Nam."""
    return datetime.now(timezone.utc) + timedelta(hours=VN_UTC_OFFSET)


def market_is_closed() -> bool:
    t = vn_now()
    return t.weekday() < 5 and t.time() >= MARKET_CLOSE_VN


def last_trading_day() -> str:
    t = vn_now()
    if market_is_closed():
        return t.strftime("%Y-%m-%d")
    d = t.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def market_status_str() -> str:
    t = vn_now()
    if t.weekday() >= 5:
        return f"Cuối tuần ({t.strftime('%A')}). Thị trường đóng."
    open_time = datetime.strptime("09:00", "%H:%M").time()
    if t.time() < open_time:
        return f"Chưa mở phiên ({t.strftime('%H:%M')} VN). Mở lúc 09:00."
    if t.time() < MARKET_CLOSE_VN:
        return f"Đang trong phiên ({t.strftime('%H:%M')} VN). Đóng lúc 15:30."
    return f"Đã đóng phiên ({t.strftime('%H:%M')} VN). Có thể cập nhật."


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching - FIX: xử lý mọi định dạng trả về
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_ohlcv(symbol: str, start: str, end: str, source: str) -> pd.DataFrame | None:
    """Fetch OHLCV, tự động thích ứng với cấu trúc response của vnstock."""
    try:
        from vnstock import Vnstock
        st = Vnstock().stock(symbol=symbol, source=source)
        raw = st.quote.history(start=start, end=end, interval="1D")

        # ---- Chuyển đổi raw -> DataFrame ----
        if isinstance(raw, pd.DataFrame):
            df = raw
        elif isinstance(raw, dict):
            # Thử lấy 'data' trước
            if 'data' in raw:
                data = raw['data']
                df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
            else:
                # Thử các key phổ biến khác
                found = False
                for key in ['list', 'records', 'result', 'items', 'values']:
                    if key in raw and isinstance(raw[key], list):
                        df = pd.DataFrame(raw[key])
                        found = True
                        break
                if not found:
                    # Nếu dict có cấu trúc {cột: [giá trị], ...}
                    df = pd.DataFrame(raw)
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
        else:
            log.error(f"Không thể xử lý kiểu dữ liệu: {type(raw)}")
            return None

        # Kiểm tra rỗng
        if df.empty:
            log.warning(f"Không có dữ liệu cho {symbol} từ {start} đến {end}")
            return None

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.lower()

        # Xác định cột ngày
        time_col = None
        for col in ['date', 'time', 'trading_date', 'ngay', 'datetime']:
            if col in df.columns:
                time_col = col
                break
        if time_col is None:
            log.error(f"Không tìm thấy cột thời gian. Các cột: {list(df.columns)}")
            return None

        df = df.rename(columns={time_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Kiểm tra các cột bắt buộc
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            log.error(f"Thiếu cột: {missing}. Các cột có: {list(df.columns)}")
            return None

        log.info(f"  {symbol}: {len(df)} phiên ({df['date'].min().date()} → {df['date'].max().date()})")
        return df

    except ImportError:
        log.error("Thiếu thư viện vnstock. Chạy: pip install vnstock")
        return None
    except Exception as e:
        log.error(f"Lỗi khi fetch {symbol}: {e}")
        # In thông tin raw để debug (nếu có)
        if 'raw' in locals():
            log.error(f"Raw response type: {type(raw)}")
            if isinstance(raw, dict):
                log.error(f"Dict keys: {raw.keys()}")
            elif isinstance(raw, list) and len(raw) > 0:
                log.error(f"First element: {raw[0]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Feature computation (giữ nguyên)
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_features(stock_df: pd.DataFrame,
                         vnindex_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Tính toàn bộ 43 features từ OHLCV raw.
    Input: stock_df (OHLCV), vnindex_df (close + volume của VNINDEX)
    Output: DataFrame với đầy đủ SCHEMA_COLS
    """
    df = stock_df.copy()
    c  = df["close"]; v = df["volume"]

    # ── Moving Averages ──────────────────────────────────────────
    df["sma_10"] = c.rolling(10).mean()
    df["sma_20"] = c.rolling(20).mean()
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_histogram"] = macd - macd.ewm(span=9, adjust=False).mean()

    # ── RSI (14) ─────────────────────────────────────────────────
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + g / (l + 1e-9))

    # ── ATR (14) ─────────────────────────────────────────────────
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - c.shift()).abs(),
        (df["low"]  - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ── CCI (14) ─────────────────────────────────────────────────
    tp     = (df["high"] + df["low"] + c) / 3
    sma_tp = tp.rolling(14).mean()
    mad    = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_14"] = (tp - sma_tp) / (0.015 * mad + 1e-9)

    # ── Momentum & ROC ────────────────────────────────────────────
    df["momentum_10"] = c.diff(10)
    df["roc_12"]      = ((c - c.shift(12)) / (c.shift(12) + 1e-9)) * 100

    # ── OBV ──────────────────────────────────────────────────────
    df["obv"] = (np.sign(c.diff()) * v).fillna(0).cumsum()

    # ── Price features ────────────────────────────────────────────
    daily_ret = c.pct_change()
    df["log_return"]    = np.log(c / c.shift(1))
    df["body_size"]     = c - df["open"]
    df["daily_range"]   = df["high"] - df["low"]
    df["upper_shadow"]  = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"]  = df[["open", "close"]].min(axis=1) - df["low"]

    # ── Volatility ────────────────────────────────────────────────
    df["rolling_std_20"]           = c.rolling(20).std()
    df["historical_volatility_20"] = daily_ret.rolling(20).std() * np.sqrt(252)

    # ── Lag returns ───────────────────────────────────────────────
    for i in range(1, 6):
        df[f"return_lag_{i}"] = daily_ret.shift(i)

    # ── Rolling stats ─────────────────────────────────────────────
    df["rolling_max_20"]  = c.rolling(20).max()
    df["rolling_min_20"]  = c.rolling(20).min()
    df["rolling_mean_20"] = c.rolling(20).mean()
    df["distance_from_high_20"] = (df["rolling_max_20"] - c) / (df["rolling_max_20"] + 1e-9)
    df["distance_from_low_20"]  = (c - df["rolling_min_20"]) / (df["rolling_min_20"] + 1e-9)

    # ── Volume indicators ─────────────────────────────────────────
    df["volume_sma_20"] = v.rolling(20).mean()
    df["volume_ratio"]  = v / (df["volume_sma_20"] + 1e-9)
    df["volume_change"] = v.pct_change()
    df["price_volume"]  = c * v

    # ── Market correlation (VNINDEX) ──────────────────────────────
    if vnindex_df is not None:
        df = pd.merge(df, vnindex_df, on="date", how="left")
        df["vnindex_return"]        = df["vnindex"].pct_change()
        df["correlation_market_20"] = daily_ret.rolling(20).corr(df["vnindex_return"])
        df["beta_20"]               = (
            daily_ret.rolling(20).cov(df["vnindex_return"])
            / (df["vnindex_return"].rolling(20).var() + 1e-9)
        )
    else:
        log.warning("Không có dữ liệu VNINDEX — các cột market correlation sẽ là NaN")
        for col in ["vnindex", "vnindex_volume", "vnindex_return",
                    "correlation_market_20", "beta_20"]:
            df[col] = np.nan

    # ── Time feature ─────────────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek

    # ── Reorder theo SCHEMA ───────────────────────────────────────
    final_cols = [c for c in SCHEMA_COLS if c in df.columns]
    df = df[final_cols]

    n_missing = df.isnull().sum().sum()
    log.info(f"Features computed: {len(df)} rows × {len(df.columns)} cols "
             f"| NaN: {n_missing:,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

class DataPipeline:
    def __init__(self, symbol: str = "VNM", source: str = "VCI"):
        self.symbol = symbol
        self.source = source

    def fetch_full(self, start: str, end: str | None = None) -> pd.DataFrame | None:
        end = end or last_trading_day()
        log.info(f"=== Pull {self.symbol}: {start} → {end} ===")
        log.info(f"Trạng thái thị trường: {market_status_str()}")

        stock_df = _fetch_ohlcv(self.symbol, start, end, self.source)
        if stock_df is None:
            return None

        vnidx_df = _fetch_ohlcv("VNINDEX", start, end, self.source)
        if vnidx_df is not None:
            vnidx_df = vnidx_df[["date", "close", "volume"]].copy()
            vnidx_df.columns = ["date", "vnindex", "vnindex_volume"]

        return compute_all_features(stock_df, vnidx_df)

    def save(self, df: pd.DataFrame, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"Saved: {len(df)} rows → {path}")


def incremental_update(output_path: str,
                       symbol: str = "VNM",
                       source: str = "VCI") -> pd.DataFrame | None:
    log.info(f"=== INCREMENTAL UPDATE: {symbol} ===")
    log.info(f"Trạng thái thị trường: {market_status_str()}")

    if not market_is_closed():
        log.warning("⚠️  Phiên chưa kết thúc (< 15:30 VN). Dữ liệu ngày hôm nay có thể chưa đầy đủ.")

    end_date = last_trading_day()
    path = Path(output_path)

    if path.exists():
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"])
        last_date = existing["date"].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log.info(f"Existing: {len(existing)} rows | Last: {last_date.date()}")
        log.info(f"Cần pull: {start_date} → {end_date}")
    else:
        log.info(f"File {path} chưa tồn tại — pull từ đầu (2021-10-01)")
        existing = None
        start_date = "2021-10-01"

    if start_date > end_date:
        log.info("✅ Dữ liệu đã cập nhật mới nhất. Không cần pull thêm.")
        return pd.read_csv(path) if path.exists() else None

    pipeline = DataPipeline(symbol, source)
    new_df = pipeline.fetch_full(start_date, end_date)
    if new_df is None or len(new_df) == 0:
        log.warning("Không có dữ liệu mới.")
        return existing

    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = (combined
                    .drop_duplicates(subset="date")
                    .sort_values("date")
                    .reset_index(drop=True))
    else:
        combined = new_df

    pipeline.save(combined, output_path)
    log.info(f"✅ Updated: {len(combined)} rows total")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull dữ liệu VNStock + tính features")
    parser.add_argument("--symbol", default="VNM", help="Mã chứng khoán")
    parser.add_argument("--start", default="2021-10-01", help="Ngày bắt đầu YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Ngày kết thúc")
    parser.add_argument("--source", default="VCI", choices=["VCI", "TCBS"], help="Nguồn dữ liệu")
    parser.add_argument("--output", default="data/VNM_2225.csv", help="Đường dẫn file output")
    parser.add_argument("--update", action="store_true", help="Cập nhật tăng dần")
    parser.add_argument("--status", action="store_true", help="Kiểm tra trạng thái thị trường")
    args = parser.parse_args()

    if args.status:
        print(f"\nTrạng thái: {market_status_str()}")
        print(f"Ngày giao dịch gần nhất: {last_trading_day()}")
        print(f"Thời gian VN hiện tại:   {vn_now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.update:
        incremental_update(args.output, args.symbol, args.source)

    else:
        pipeline = DataPipeline(args.symbol, args.source)
        df = pipeline.fetch_full(args.start, args.end)
        if df is not None:
            pipeline.save(df, args.output)
            print(f"\nMẫu 5 dòng cuối:")
            print(df[["date", "close", "rsi_14", "macd_histogram", "volume_ratio"]].tail(5).to_string(index=False))