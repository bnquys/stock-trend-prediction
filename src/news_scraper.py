"""CafeF market-news and article-content scraper."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

log = logging.getLogger(__name__)

CAFEF_NEWS_URL = "https://cafef.vn/thi-truong-chung-khoan.chn"
CAFEF_ORIGIN = "https://cafef.vn"
CAFEF_ALLOWED_HOSTS = {"cafef.vn", "www.cafef.vn"}
NEWS_CONTAINER_SELECTOR = (
    "#admWrapsite > div.main.loadedStock > div.list-section.loadedStock > div > "
    "div.list-main.loadedStock > div.list-news.loadedStock > "
    "div.list-news-main.top5_news.loadedStock"
)
FALLBACK_CONTAINER_SELECTOR = "#admWrapsite .list-news-main.top5_news"
NEWS_ITEM_SELECTOR = '[role="article"].tlitem.box-category-item'
ARTICLE_CONTENT_SELECTORS = (
    "#ContentDetail",
    ".detail-content",
    ".detail-content-main",
)
REQUEST_TIMEOUT_SECONDS = 10
CACHE_TTL = timedelta(minutes=5)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


class CafeFError(RuntimeError):
    """Raised when CafeF cannot provide parseable content."""


@dataclass(frozen=True)
class News:
    id: str | None
    title: str
    url: str
    image_url: str | None
    published_at: str | None
    description: str | None
    source: str = "CafeF"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsListResult:
    items: list[News]
    fetched_at: str
    source: str = "CafeF"
    source_url: str = CAFEF_NEWS_URL


@dataclass(frozen=True)
class CafeFArticle:
    """Normalized text extracted from one CafeF article page."""

    title: str
    url: str
    published_at: str | None
    description: str | None
    content: str


_cached_result: NewsListResult | None = None
_cached_at: datetime | None = None
_article_cache: dict[str, tuple[datetime, CafeFArticle]] = {}


def _fetch_html(url: str) -> str:
    """Download HTML using the same requests/BeautifulSoup flow as the sample."""
    log.info("Đang tải dữ liệu CafeF từ %s", url)
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        log.warning("Không thể kết nối CafeF: %s", exc)
        raise CafeFError("Không thể kết nối tới CafeF.") from exc

    if response.status_code != 200:
        log.warning("CafeF trả về status code %s", response.status_code)
        raise CafeFError(f"CafeF trả về mã trạng thái {response.status_code}.")

    response.encoding = "utf-8"
    return response.text


def _text(node: Tag | None) -> str | None:
    if node is None:
        return None
    value = " ".join(node.get_text(" ", strip=True).split())
    return value or None


def _absolute_url(value: str | None, base_url: str) -> str | None:
    if not value:
        return None
    return urljoin(base_url, value.strip())


def _image_url(item: Tag, base_url: str) -> str | None:
    image = item.select_one("img")
    if image is None:
        return None
    for attribute in ("src", "data-src", "data-original", "data-lazy-src"):
        value = image.get(attribute)
        if value:
            return _absolute_url(value, base_url)
    return None


def _find_container(soup: BeautifulSoup) -> Tag:
    container = soup.select_one(NEWS_CONTAINER_SELECTOR)
    if container is None:
        container = soup.select_one(FALLBACK_CONTAINER_SELECTOR)
    if container is None:
        raise CafeFError("Không tìm thấy vùng danh sách tin tức CafeF.")
    return container


def _find_article_content(soup: BeautifulSoup) -> Tag:
    for selector in ARTICLE_CONTENT_SELECTORS:
        content = soup.select_one(selector)
        if content is not None:
            return content
    raise CafeFError("Không tìm thấy vùng nội dung bài viết CafeF.")


def _clean_article_content(container: Tag) -> str:
    """Return readable article paragraphs while dropping page chrome/ads."""
    unwanted_tokens = (
        "advert",
        "banner",
        "comment",
        "facebook",
        "related",
        "share",
        "social",
        "tag-list",
    )
    for node in container.select("script, style, iframe, noscript, figure, aside"):
        node.decompose()
    for node in container.find_all(True):
        marker = " ".join(
            filter(None, [node.get("id"), " ".join(node.get("class", []))])
        ).lower()
        if any(token in marker for token in unwanted_tokens):
            node.decompose()

    paragraphs: list[str] = []
    seen: set[str] = set()
    for node in container.select("p"):
        value = _text(node)
        if value and value not in seen:
            seen.add(value)
            paragraphs.append(value)
    if not paragraphs:
        fallback = _text(container)
        if fallback:
            paragraphs.append(fallback)
    return "\n\n".join(paragraphs)


def _parse_item(item: Tag, base_url: str) -> News | None:
    title_link = item.select_one("h3 > a[href]")
    title = _text(title_link)
    article_url = _absolute_url(title_link.get("href") if title_link else None, base_url)
    if not title or not article_url:
        log.debug("Bỏ qua một item CafeF thiếu tiêu đề hoặc URL")
        return None

    time_node = item.select_one(".time.time-ago")
    published_at = None
    if time_node is not None:
        published_at = time_node.get("title") or _text(time_node)

    description = _text(item.select_one("p.sapo.box-category-sapo"))
    return News(
        id=item.get("data-id"),
        title=title,
        url=article_url,
        image_url=_image_url(item, base_url),
        published_at=published_at,
        description=description,
    )


def parse_news_list(html: str, source_url: str = CAFEF_NEWS_URL) -> list[News]:
    """Parse category-page HTML into normalized news-list items."""
    soup = BeautifulSoup(html, "html.parser")
    container = _find_container(soup)
    items: list[News] = []
    for item in container.select(NEWS_ITEM_SELECTOR):
        parsed = _parse_item(item, source_url)
        if parsed is not None:
            items.append(parsed)
    return items


def _parse_published_date(value: str | None) -> date | None:
    """Parse the date formats exposed by CafeF without adding dependencies."""
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    return None


def news_published_today(
    item: News,
    *,
    today: date | None = None,
) -> bool:
    """Return whether a list item is dated today in the Vietnam timezone."""
    vietnam_today = today or datetime.now(VIETNAM_TZ).date()
    return _parse_published_date(item.published_at) == vietnam_today


def parse_cafef_article(
    html: str,
    url: str,
    *,
    fallback: News | None = None,
) -> CafeFArticle:
    """Extract the title, metadata and readable body from a CafeF article."""
    soup = BeautifulSoup(html, "html.parser")
    content = _find_article_content(soup)
    title = _text(soup.select_one("h1")) or (fallback.title if fallback else None)
    if not title:
        raise CafeFError("Không tìm thấy tiêu đề bài viết CafeF.")
    time_node = soup.select_one("time[datetime], .time.time-ago")
    published_at = None
    if time_node is not None:
        published_at = time_node.get("datetime") or time_node.get("title") or _text(time_node)
    description = _text(
        soup.select_one(".sapo, .detail-sapo, meta[name='description']")
    )
    if description is None and fallback is not None:
        description = fallback.description
    article_text = _clean_article_content(content)
    if not article_text:
        raise CafeFError("Bài viết CafeF không có nội dung để tóm tắt.")
    return CafeFArticle(
        title=title,
        url=url,
        published_at=published_at or (fallback.published_at if fallback else None),
        description=description,
        content=article_text,
    )


def _validate_cafef_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in CAFEF_ALLOWED_HOSTS:
        raise ValueError("URL bài viết phải thuộc https://cafef.vn.")


def crawl_cafef_article(news: News, *, refresh: bool = False) -> CafeFArticle:
    """Fetch one trusted CafeF article, with a short in-memory body cache."""
    _validate_cafef_url(news.url)
    cached = _article_cache.get(news.url)
    now = datetime.now(timezone.utc)
    if not refresh and cached is not None and now - cached[0] < CACHE_TTL:
        return cached[1]
    article = parse_cafef_article(_fetch_html(news.url), news.url, fallback=news)
    _article_cache[news.url] = (now, article)
    return article


def crawl_cafef_news_list(
    url: str = CAFEF_NEWS_URL,
    *,
    refresh: bool = False,
) -> NewsListResult:
    """Fetch and parse the CafeF category list, using a short in-memory cache."""
    global _cached_at, _cached_result

    if not refresh and _cached_result is not None and _cached_at is not None:
        if datetime.now(timezone.utc) - _cached_at < CACHE_TTL:
            return _cached_result

    html = _fetch_html(url)
    items = parse_news_list(html, url)
    if not items:
        raise CafeFError("CafeF không có bài viết trong vùng danh sách tin.")

    result = NewsListResult(
        items=items,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        source_url=url,
    )
    _cached_result = result
    _cached_at = datetime.now(timezone.utc)
    return result