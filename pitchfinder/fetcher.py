"""Unified RSS / Atom / podcast feed fetcher."""

from __future__ import annotations

import email.utils as eut
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("pitchfinder.fetcher")

USER_AGENT = "pitchfinder/0.1 (+https://github.com/shanda-internal/pitchfinder)"
HTTP_TIMEOUT = 15.0
SUMMARY_CAP = 2000


@dataclass
class FeedItem:
    title: str
    url: str
    published_at: Optional[datetime]
    summary: str


def fetch_feed(feed_url: str, content_type: str, lookback_days: int) -> list[FeedItem]:
    """Fetch a feed URL and return items newer than lookback_days.

    Returns [] on any failure. Never raises.
    """
    if not feed_url:
        return []

    try:
        raw = _http_get_bytes(feed_url)
    except Exception as exc:
        logger.warning("fetch failed %s: %s", feed_url, exc)
        return []

    try:
        parsed = feedparser.parse(raw)
    except Exception as exc:
        logger.warning("parse failed %s: %s", feed_url, exc)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: list[FeedItem] = []
    for entry in parsed.entries:
        url = entry.get("link") or entry.get("id") or ""
        if not url:
            continue

        published = _entry_published(entry)
        if published and published < cutoff:
            continue

        title = (entry.get("title") or "").strip() or "(untitled)"
        summary = _entry_summary(entry, content_type)
        items.append(
            FeedItem(
                title=title,
                url=url,
                published_at=published,
                summary=summary,
            )
        )

    return items


def _http_get_bytes(url: str) -> bytes:
    with httpx.Client(
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _entry_published(entry) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            try:
                return datetime(*struct[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    for key in ("published", "updated"):
        s = entry.get(key)
        if s:
            try:
                dt = eut.parsedate_to_datetime(s)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    return None


def _entry_summary(entry, content_type: str) -> str:
    candidates: list[str] = []
    content = entry.get("content")
    if content and isinstance(content, list):
        candidates.append(content[0].get("value", "") if hasattr(content[0], "get") else "")
    candidates.append(entry.get("summary", "") or "")
    candidates.append(entry.get("description", "") or "")
    if content_type == "episode":
        candidates.append(entry.get("itunes_summary", "") or "")
        candidates.append(entry.get("subtitle", "") or "")

    raw = max(candidates, key=len) if candidates else ""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    return text[:SUMMARY_CAP]
