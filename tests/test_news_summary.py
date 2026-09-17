from __future__ import annotations

from src.ai.news_summary import simulate_ai_summary
from src.news_scraper import CafeFArticle


def _article(title: str = "VN-Index phục hồi") -> CafeFArticle:
    return CafeFArticle(
        title=title,
        url="https://cafef.vn/example.chn",
        published_at="2026-09-17T09:00:00",
        description="Dòng tiền quay lại thị trường.",
        content="Thanh khoản thị trường cải thiện trong phiên hôm nay.",
    )


def test_simulate_ai_summary_returns_complete_non_streaming_result():
    result = simulate_ai_summary([_article()], "Tập trung vào dòng tiền.", delay_seconds=0)

    assert result.article_count == 1
    assert "TÓM TẮT THỊ TRƯỜNG HÔM NAY" in result.summary
    assert "Tập trung vào dòng tiền." in result.summary
    assert result.simulated_delay_seconds == 0


def test_simulate_ai_summary_includes_multiple_articles():
    result = simulate_ai_summary(
        [_article(), _article("Lãi suất mới nhất")],
        "Tóm tắt ngắn gọn.",
        delay_seconds=0,
    )

    assert result.article_count == 2
    assert "Lãi suất mới nhất" in result.summary


def test_simulate_ai_summary_rejects_empty_inputs():
    try:
        simulate_ai_summary([], "Tóm tắt.", delay_seconds=0)
    except ValueError as exc:
        assert "Không có bài viết" in str(exc)
    else:
        raise AssertionError("Expected empty article input to be rejected")

    try:
        simulate_ai_summary([_article()], "   ", delay_seconds=0)
    except ValueError as exc:
        assert "Prompt" in str(exc)
    else:
        raise AssertionError("Expected empty prompt to be rejected")