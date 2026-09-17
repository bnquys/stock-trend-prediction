"""Deterministic, non-streaming mock for the daily news summary."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from src.news_scraper import CafeFArticle


DEFAULT_SIMULATED_DELAY_SECONDS = 2.0
MAX_SIMULATED_DELAY_SECONDS = 10.0
MAX_ARTICLE_CHARACTERS = 8_000


@dataclass(frozen=True)
class SummaryResult:
    """Complete response returned after the simulated model finishes."""

    summary: str
    article_count: int
    prompt: str
    generated_at: str
    simulated_delay_seconds: float


def _article_excerpt(article: CafeFArticle) -> str:
    text = " ".join((article.description or article.content).split())
    if len(text) > 180:
        return f"{text[:177].rstrip()}..."
    return text


def _find_topics(text: str) -> list[str]:
    topic_keywords = (
        ("VN-Index và chỉ số thị trường", ("vn-index", "vnindex", "vn index", "chỉ số")),
        ("dòng tiền và thanh khoản", ("dòng tiền", "thanh khoản", "giá trị giao dịch")),
        ("lãi suất và chính sách tiền tệ", ("lãi suất", "tiền tệ", "ngân hàng nhà nước")),
        ("cổ phiếu ngân hàng", ("ngân hàng", "bank", "tín dụng")),
        ("cổ phiếu bất động sản", ("bất động sản", "địa ốc", "real estate")),
        ("doanh nghiệp và kết quả kinh doanh", ("lợi nhuận", "doanh thu", "kết quả kinh doanh")),
    )
    lowered = text.lower()
    return [label for label, keywords in topic_keywords if any(keyword in lowered for keyword in keywords)]


def simulate_ai_summary(
    articles: Sequence[CafeFArticle],
    prompt: str,
    *,
    delay_seconds: float = DEFAULT_SIMULATED_DELAY_SECONDS,
) -> SummaryResult:
    """Return a complete mock-AI summary after a realistic response delay.

    This deliberately does not call an external model and does not stream. The
    function keeps the same input boundary that a real model adapter can use
    later: normalized article bodies plus a user prompt.
    """
    normalized_prompt = " ".join(prompt.split())
    if not normalized_prompt:
        raise ValueError("Prompt không được để trống.")
    if not articles:
        raise ValueError("Không có bài viết để tóm tắt.")

    safe_delay = min(max(float(delay_seconds), 0.0), MAX_SIMULATED_DELAY_SECONDS)
    time.sleep(safe_delay)

    combined_text = "\n".join(
        f"{article.title}\n{article.description or ''}\n{article.content[:MAX_ARTICLE_CHARACTERS]}"
        for article in articles
    )
    topics = _find_topics(combined_text)
    topic_text = ", ".join(topics[:4]) if topics else "các diễn biến được nêu trong tiêu đề và nội dung bài báo"
    article_lines = "\n".join(
        f"• {article.title}: {_article_excerpt(article)}" for article in articles[:6]
    )
    remaining = len(articles) - min(len(articles), 6)
    if remaining:
        article_lines += f"\n• Và {remaining} bài viết khác trong cùng ngày."

    summary = (
        "TÓM TẮT THỊ TRƯỜNG HÔM NAY\n\n"
        f"Đã tổng hợp {len(articles)} bài báo CafeF. Các chủ đề nổi bật gồm {topic_text}.\n\n"
        f"Định hướng yêu cầu: {normalized_prompt}\n\n"
        "Các điểm chính:\n"
        f"{article_lines}\n\n"
        "Lưu ý: Đây là bản tóm tắt mô phỏng từ nội dung bài báo, không phải khuyến nghị mua hoặc bán. "
        "Nhà đầu tư nên kiểm tra bài viết gốc và dữ liệu thị trường trước khi ra quyết định."
    )
    return SummaryResult(
        summary=summary,
        article_count=len(articles),
        prompt=normalized_prompt,
        generated_at=datetime.now(timezone.utc).isoformat(),
        simulated_delay_seconds=safe_delay,
    )