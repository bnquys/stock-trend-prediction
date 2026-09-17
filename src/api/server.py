"""
src/api/server.py — TenacoreX Market API
═══════════════════════════════════════════════════════════════════════════
Phục vụ dữ liệu thị trường cho thi-truong.html và tin tức/summary cho tin-tuc.html.

Endpoints:
    GET /healthz                  → kiểm tra server
    GET /api/market/status        → {is_open, status_text, server_time_vn}
    GET /api/market/indexes       → 3 chỉ số (chỉ VN-INDEX có dữ liệu)
    GET /api/stocks               → 4 mã từ artifacts/data/*.csv
    GET /api/news                 → danh sách tin CafeF
    POST /api/news/summary        → summary không stream từ tin hôm nay

Chạy:
    uv run python thi-truong-api.py
    # hoặc: uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ai.news_summary import simulate_ai_summary

from src.news_scraper import (
    CAFEF_NEWS_URL,
    CafeFError,
    CafeFArticle,
    crawl_cafef_article,
    crawl_cafef_news_list,
    news_published_today,
    VIETNAM_TZ,
)

log = logging.getLogger("uvicorn.error")

# ── Paths ────────────────────────────────────────────────────────────
_ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT_DIR / "artifacts" / "data"
SYMBOLS = ["FPT", "HPG", "VIC", "VNM"]

# ── Cache DataFrames sau khi load 1 lần ở startup ────────────────────
_frames: dict[str, pd.DataFrame] = {}


def _load_all_csv() -> None:
    for sym in SYMBOLS:
        p = DATA_DIR / f"{sym}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        _frames[sym] = df
    log.info(
        "[TenacoreX] loaded %d stocks from %s (rows: %s)",
        len(_frames),
        DATA_DIR,
        {s: len(d) for s, d in _frames.items()},
    )


# ── Pydantic schemas ──────────────────────────────────────────────────
class MarketStatus(BaseModel):
    is_open: bool
    status_text: str
    server_time_vn: str  # format "YYYY-MM-DD HH:MM:SS"


class IndexInfo(BaseModel):
    code: str
    available: bool
    value: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    series_sparkline: List[float] = []


class StockRow(BaseModel):
    symbol: str
    last: float
    change: float
    change_pct: float
    open: float
    high: float
    low: float
    avg: float
    volume_thousand: float


class NewsItem(BaseModel):
    id: Optional[str] = None
    title: str
    url: str
    image_url: Optional[str] = None
    published_at: Optional[str] = None
    description: Optional[str] = None
    source: str = "CafeF"


class NewsResponse(BaseModel):
    source: str
    source_url: str
    fetched_at: str
    count: int
    today_count: int
    items: List[NewsItem]


class NewsSummaryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    refresh: bool = False


class NewsSummaryResponse(BaseModel):
    source: str
    date: str
    article_count: int
    summary: str
    prompt: str
    generated_at: str
    simulated_delay_seconds: float


# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="TenacoreX Market API", version="1.0-csv-only")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    _load_all_csv()


@app.get("/healthz")
def healthz():
    return {"ok": True, "stocks": list(_frames.keys())}


# ── Trạng thái thị trường (giờ VN, không cần dữ liệu) ────────────────
@app.get("/api/market/status", response_model=MarketStatus)
def market_status():
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    except Exception:
        now = datetime.now()
    minutes = now.hour * 60 + now.minute
    weekday = now.weekday()  # 0=Mon..6=Sun
    in_session = weekday < 5 and (
        (9 * 60 <= minutes <= 11 * 60 + 30) or (13 * 60 <= minutes <= 15 * 60)
    )
    if weekday >= 5:
        text = "Cuối tuần — Thị trường nghỉ"
    elif in_session:
        text = "Phiên HOSE đang mở cửa"
    else:
        text = "Phiên HOSE đã đóng cửa"
    return MarketStatus(
        is_open=in_session,
        status_text=text,
        server_time_vn=now.strftime("%Y-%m-%d %H:%M:%S"),
    )


# ── 3 chỉ số (chỉ VN-INDEX có dữ liệu từ CSV) ──────────────────────
@app.get("/api/market/indexes", response_model=List[IndexInfo])
def market_indexes():
    df = _frames["VNM"]  # vnindex_close có trong artifacts/data/VNM.csv
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    v_close = float(last["vnindex_close"])
    v_prev = float(prev["vnindex_close"])
    chg = v_close - v_prev
    pct = (chg / v_prev * 100) if v_prev else 0.0
    spark = df["vnindex_close"].tail(20).astype(float).tolist()
    return [
        IndexInfo(
            code="VN-INDEX",
            available=True,
            value=round(v_close, 2),
            change=round(chg, 2),
            change_pct=round(pct, 2),
            open=round(float(last["open"]), 2),
            high=round(float(last["high"]), 2),
            low=round(float(last["low"]), 2),
            series_sparkline=spark,
        ),
        IndexInfo(code="HNX-INDEX", available=False),  # không có trong CSV
        IndexInfo(code="UPCOM-INDEX", available=False),  # không có trong CSV
    ]


# ── Bảng cổ phiếu ────────────────────────────────────────────────────
@app.get("/api/stocks", response_model=List[StockRow])
def stocks():
    rows: List[StockRow] = []
    for sym, df in _frames.items():
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last
        chg = float(last["close"] - prev["close"])
        pct = (chg / float(prev["close"]) * 100) if float(prev["close"]) else 0.0
        rows.append(
            StockRow(
                symbol=sym,
                last=round(float(last["close"]), 2),
                change=round(chg, 2),
                change_pct=round(pct, 2),
                open=round(float(last["open"]), 2),
                high=round(float(last["high"]), 2),
                low=round(float(last["low"]), 2),
                avg=round(float(last["vwap"]), 2),
                volume_thousand=round(float(last["volume"]) / 1000.0, 1),
            )
        )
    return rows


@app.get("/api/news", response_model=NewsResponse)
def news(refresh: bool = False):
    """Return the latest market-news list; article bodies stay on CafeF."""
    try:
        result = crawl_cafef_news_list(refresh=refresh)
    except CafeFError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    items = [NewsItem(**item.to_dict()) for item in result.items]
    today = datetime.now(VIETNAM_TZ).date()
    today_count = sum(news_published_today(item, today=today) for item in result.items)
    return NewsResponse(
        source=result.source,
        source_url=result.source_url or CAFEF_NEWS_URL,
        fetched_at=result.fetched_at,
        count=len(items),
        today_count=today_count,
        items=items,
    )


@app.post("/api/news/summary", response_model=NewsSummaryResponse)
def news_summary(request: NewsSummaryRequest):
    """Summarize the readable CafeF articles published today.

    The endpoint intentionally returns one complete response after the mock
    model delay. It is a drop-in boundary for a real non-streaming LLM later.
    """
    try:
        result = crawl_cafef_news_list(refresh=request.refresh)
    except CafeFError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    today = datetime.now(VIETNAM_TZ).date()
    today_items = [item for item in result.items if news_published_today(item, today=today)]
    if not today_items:
        raise HTTPException(status_code=404, detail="Chưa có bài viết hôm nay để tóm tắt.")

    articles: list[CafeFArticle] = []
    failed_articles = 0
    for item in today_items:
        try:
            articles.append(crawl_cafef_article(item, refresh=request.refresh))
        except (CafeFError, ValueError) as exc:
            failed_articles += 1
            log.warning("Bỏ qua bài CafeF không thể đọc để tóm tắt (%s): %s", item.url, exc)

    if not articles:
        raise HTTPException(
            status_code=502,
            detail="Không thể đọc nội dung các bài viết hôm nay để tóm tắt.",
        )

    try:
        summary = simulate_ai_summary(articles, request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if failed_articles:
        log.info("Summary hoàn tất với %d bài bị bỏ qua", failed_articles)
    return NewsSummaryResponse(
        source="CafeF",
        date=today.isoformat(),
        article_count=summary.article_count,
        summary=summary.summary,
        prompt=summary.prompt,
        generated_at=summary.generated_at,
        simulated_delay_seconds=summary.simulated_delay_seconds,
    )