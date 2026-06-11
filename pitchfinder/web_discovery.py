"""Cheap, broad creator discovery via web search (Brave primary, Querit supplement).

Why: the rest of discovery only finds creators that already have a known RSS feed
(seed library + MiroThinker). This sweeps the open web for newsletters / blogs /
YouTube channels on the launch themes, resolves their feeds, and emits seed-loadable
candidates — the wide top of the funnel before scoring + tiering + (pricey) MiroThinker.

API shapes mirror shanda-pulse's verified clients:
  Brave : GET  {BRAVE_SEARCH_URL}?q=..&count=N   header X-Subscription-Token
          -> {"web": {"results": [{"title","url","description"}]}}
  Querit: POST {QUERIT_SEARCH_URL} {"query","count"}  header Authorization: Bearer
          -> {"results": {"result": [{"title","url","snippet"}]}}

Env-gated: a provider with no key is silently skipped.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

import feedparser
import httpx

from pitchfinder.discovery import HTTP_TIMEOUT, _slugify, resolve_feed_url, search_podcasts

logger = logging.getLogger("pitchfinder.web_discovery")


def feed_title(feed_url: str) -> Optional[str]:
    """The feed's own <title> = the publication/channel name (not a post title).
    Web search results are often individual posts, so prefer this for naming."""
    if not feed_url:
        return None
    try:
        parsed = feedparser.parse(feed_url)
        t = (parsed.feed.get("title") or "").strip() if parsed and parsed.feed else ""
        # strip trailing " | Substack" / " - YouTube" noise
        t = t.split(" | ")[0].split(" - YouTube")[0].strip()
        return t or None
    except Exception:
        return None


# ---------- provider clients ----------

def _brave_search(query: str, count: int = 10) -> list[dict[str, Any]]:
    key = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
    if not key:
        return []
    url = os.getenv("BRAVE_SEARCH_URL", "https://api.search.brave.com/res/v1/web/search")
    try:
        r = httpx.get(
            url,
            params={"q": query, "count": min(max(count, 1), 20)},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        results = (r.json().get("web") or {}).get("results") or []
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": x.get("description", "")} for x in results]
    except Exception as exc:
        logger.warning("Brave search failed (%r): %s", query, exc)
        return []


def _querit_search(query: str, count: int = 10) -> list[dict[str, Any]]:
    token = os.getenv("QUERIT_API_TOKEN")
    if not token:
        return []
    url = os.getenv("QUERIT_SEARCH_URL", "https://api.querit.ai/v1/search")
    try:
        r = httpx.post(
            url,
            json={"query": query, "count": count},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                     "Accept": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        results_obj = data.get("results")
        inner = results_obj.get("result") if isinstance(results_obj, dict) else results_obj
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": x.get("snippet", "")} for x in (inner or [])]
    except Exception as exc:
        logger.warning("Querit search failed (%r): %s", query, exc)
        return []


_PROVIDERS = {"brave": _brave_search, "querit": _querit_search}


def providers_available() -> list[str]:
    avail = []
    if os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY"):
        avail.append("brave")
    if os.getenv("QUERIT_API_TOKEN"):
        avail.append("querit")
    return avail


# ---------- URL → creator normalization ----------

def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _normalize(url: str, platform_hint: str) -> Optional[tuple[str, str, str]]:
    """Return (platform, creator_root_url, handle) or None if not a usable creator URL."""
    host = _host(url)
    if not host:
        return None

    # Substack (and substack-custom domains we can't detect → only *.substack.com here)
    if host.endswith("substack.com"):
        sub = host.split(".substack.com")[0]
        if sub in ("", "www", "open", "api"):
            return None
        return ("substack", f"https://{host}", sub)

    # YouTube — only channel-rooted URLs (skip /watch, /shorts, /results)
    if "youtube.com" in host:
        path = urlparse(url).path
        for marker in ("/@", "/channel/", "/c/", "/user/"):
            if marker in path:
                seg = path.split(marker, 1)[1].split("/")[0]
                root = f"https://www.youtube.com{path[:path.find(marker)]}{marker}{seg}"
                return ("youtube", root, _slugify(seg))
        return None

    # Skip pure social / aggregator hosts (creators there have no RSS we can pull)
    skip = ("twitter.com", "x.com", "reddit.com", "linkedin.com", "facebook.com",
            "instagram.com", "tiktok.com", "medium.com", "apple.com", "spotify.com")
    if any(s in host for s in skip):
        return None

    # Everything else → treat as a blog; normalize to the domain root.
    if platform_hint == "youtube":
        return None
    return ("blog", f"https://{host}", _slugify(host.replace("www.", "")))


def _queries(themes: list[str], platforms: list[str], per_platform: int) -> list[tuple[str, str]]:
    """(platform_hint, query) pairs. per_platform caps themes used per platform."""
    out: list[tuple[str, str]] = []
    th = themes[:per_platform] if per_platform else themes
    for t in th:
        if "substack" in platforms:
            out.append(("substack", f"site:substack.com {t} newsletter"))
        if "blog" in platforms:
            out.append(("blog", f"{t} newsletter OR blog"))
        if "youtube" in platforms:
            out.append(("youtube", f"{t} youtube channel deep dive OR explained"))
    return out


def discover_web_creators(
    themes: list[str],
    platforms: list[str],
    existing_names: Optional[set[str]] = None,
    providers: Optional[list[str]] = None,
    per_platform_queries: int = 4,
    results_per_query: int = 10,
) -> list[dict[str, Any]]:
    """Sweep Brave/Querit for creators on the given themes; resolve feeds; dedupe.

    Returns candidate dicts in the same shape as discovery.write_candidates_yaml
    expects (name/platform/handle/url/feed_url/topics/influence_score/contact/notes).
    """
    existing = {n.lower() for n in (existing_names or set())}
    use = [p for p in (providers or list(_PROVIDERS)) if p in providers_available()]
    if not use and "podcast" not in platforms:
        logger.info("No web-search provider key set (BRAVE_API_KEY / QUERIT_API_TOKEN); skipping.")

    seen_handles: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []

    # web search → substack / blog / youtube
    for hint, query in _queries(themes, platforms, per_platform_queries):
        rows: list[dict] = []
        for p in use:
            rows += _PROVIDERS[p](query, results_per_query)
        for res in rows:
            norm = _normalize(res.get("url", ""), hint)
            if not norm:
                continue
            platform, root, handle = norm
            if (platform, handle) in seen_handles:
                continue
            feed = resolve_feed_url(platform, root)
            # Prefer the feed's own title (publication/channel name) over the
            # search-result title, which is usually a single post's headline.
            name = feed_title(feed) if feed else None
            if not name:
                name = res.get("title", "").strip().split(" | ")[0].split(" - ")[0].strip()[:120]
            name = name or handle
            if name.lower() in existing:
                continue
            seen_handles.add((platform, handle))
            candidates.append({
                "name": name, "platform": platform, "handle": handle,
                "url": root, "feed_url": feed,
                "topics": [], "influence_score": 50,
                "contact": {"email": None, "other": None},
                "notes": f"DISCOVERED via web search ({'+'.join(use) or 'n/a'}). "
                         f"{'feed resolved' if feed else 'feed UNRESOLVED — verify'}.",
            })

    # podcasts via iTunes (web search returns un-RSS-able directory pages)
    if "podcast" in platforms:
        for t in themes[:per_platform_queries or len(themes)]:
            for c in search_podcasts(t, limit=results_per_query):
                fu = c.get("feed_url")
                name = (c.get("name") or "").strip()
                if not fu or not name or name.lower() in existing:
                    continue
                handle = _slugify(name)
                if ("podcast", handle) in seen_handles:
                    continue
                seen_handles.add(("podcast", handle))
                candidates.append({
                    "name": name, "platform": "podcast", "handle": handle,
                    "url": c.get("view_url", ""), "feed_url": fu,
                    "topics": c.get("genres", [])[:4], "influence_score": 50,
                    "contact": {"email": None, "other": None},
                    "notes": f"DISCOVERED via iTunes (term: {t}).",
                })

    logger.info("web discovery: %d candidates (providers=%s)", len(candidates), use)
    return candidates
