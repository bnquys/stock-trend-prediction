from __future__ import annotations

from pathlib import Path

import pytest

from src.news_scraper import (
    CAFEF_NEWS_URL,
    CafeFError,
    parse_news_list,
)


FIXTURE = Path(__file__).with_name("fixtures") / "cafef_news.html"


def test_parse_news_list_prefers_full_selector_and_normalizes_fields():
    items = parse_news_list(FIXTURE.read_text(encoding="utf-8"))

    assert len(items) == 2
    assert items[0].id == "news-001"
    assert items[0].title == "VN-Index phục hồi mạnh trong phiên hôm nay"
    assert items[0].url == "https://cafef.vn/vn-index-phuc-hoi.chn"
    assert items[0].image_url == "https://cafef.vn/images/vn-index.jpg"
    assert items[0].published_at == "2026-09-17T19:33:56"
    assert items[0].description == "Dòng tiền quay lại nhóm cổ phiếu chứng khoán."


def test_parse_news_list_uses_fallback_selector_and_lazy_image():
    html = """
    <div id="admWrapsite">
      <div class="list-news-main top5_news">
        <div role="article" class="tlitem box-category-item" data-id="fallback-001">
          <h3><a href="/fallback.chn">Tin dự phòng</a></h3>
          <img data-src="https://cdn.cafef.vn/fallback.jpg">
        </div>
      </div>
    </div>
    """

    items = parse_news_list(html)

    assert len(items) == 1
    assert items[0].url == "https://cafef.vn/fallback.chn"
    assert items[0].image_url == "https://cdn.cafef.vn/fallback.jpg"


def test_parse_news_list_rejects_missing_container():
    with pytest.raises(CafeFError, match="Không tìm thấy"):
        parse_news_list("<html><body>Không có danh sách</body></html>")


def test_default_source_url_is_cafef():
    assert CAFEF_NEWS_URL == "https://cafef.vn/thi-truong-chung-khoan.chn"