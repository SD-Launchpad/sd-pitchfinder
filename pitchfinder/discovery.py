"""Discovery helpers — bootstrap the seed library from external sources."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("pitchfinder.discovery")

USER_AGENT = "pitchfinder/0.1"
# Some hosts (notably YouTube) serve different / blocked markup to non-browser
# agents, so feed resolution uses a browser UA.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 15.0

console = Console()

ITUNES_SEARCH = "https://itunes.apple.com/search"


_YT_CANONICAL_RE = re.compile(
    r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{20,})"'
)
_YT_OGURL_RE = re.compile(
    r'<meta property="og:url" content="https://www\.youtube\.com/channel/(UC[\w-]{20,})"'
)
_FEEDISH = ("<rss", "<feed", "<?xml")


def _looks_like_feed(text: str) -> bool:
    head = text.lstrip()[:200].lower()
    return any(tok in head for tok in _FEEDISH)


def resolve_feed_url(platform: str, url: str, name: Optional[str] = None) -> Optional[str]:
    """Best-effort RSS/Atom feed URL for a creator's homepage.

    Discovered creators often arrive with feed_url=None, so `refresh` can't pick
    them up. This fills the gap per platform:
      - substack: <url>/feed
      - youtube:  parse the channel page's canonical /channel/UC… link, then
                  feeds/videos.xml?channel_id=UC…  (NOT the first channelId in
                  the HTML — that is often a *recommended* channel)
      - blog:     try /feed /rss /atom.xml /index.xml /feed.xml and keep the
                  first that returns feed-like XML
      - podcast:  resolve by name via the iTunes API (url alone isn't enough)
    Returns None if nothing resolves.
    """
    platform = (platform or "").lower()
    url = (url or "").rstrip("/")

    if platform == "substack" and url:
        return f"{url}/feed"

    if platform == "youtube" and url:
        try:
            r = httpx.get(url, headers={"User-Agent": BROWSER_UA},
                          follow_redirects=True, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                m = _YT_CANONICAL_RE.search(r.text) or _YT_OGURL_RE.search(r.text)
                if m:
                    return f"https://www.youtube.com/feeds/videos.xml?channel_id={m.group(1)}"
        except Exception as exc:
            logger.warning("youtube feed resolve failed for %s: %s", url, exc)
        return None

    if platform == "podcast":
        if name:
            for c in search_podcasts(name, limit=5):
                if c.get("name", "").lower() == name.lower() and c.get("feed_url"):
                    return c["feed_url"]
            for c in search_podcasts(name, limit=5):
                if c.get("feed_url"):
                    return c["feed_url"]
        return None

    # blog (or unknown): probe common feed paths
    if url:
        for path in ("/feed", "/rss", "/rss.xml", "/atom.xml", "/index.xml", "/feed.xml"):
            cand = url + path
            try:
                r = httpx.get(cand, headers={"User-Agent": BROWSER_UA},
                              follow_redirects=True, timeout=HTTP_TIMEOUT)
                if r.status_code == 200 and _looks_like_feed(r.text):
                    return cand
            except Exception:
                continue
    return None


def search_podcasts(keyword: str, limit: int = 10) -> list[dict[str, Any]]:
    """Apple Podcasts iTunes search. No API key required."""
    params = {"term": keyword, "entity": "podcast", "limit": str(limit)}
    try:
        with httpx.Client(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(ITUNES_SEARCH, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("iTunes search failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for r in data.get("results", []):
        results.append(
            {
                "name": r.get("collectionName") or r.get("trackName") or "",
                "artist": r.get("artistName") or "",
                "view_url": r.get("collectionViewUrl", ""),
                "feed_url": r.get("feedUrl"),
                "genres": r.get("genres", []),
            }
        )
    return results


def print_podcast_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        console.print("[yellow]No podcast candidates returned.[/yellow]")
        return
    table = Table(title="Podcast candidates (iTunes)")
    table.add_column("#")
    table.add_column("Name")
    table.add_column("Artist")
    table.add_column("Feed URL")
    for i, c in enumerate(candidates, 1):
        table.add_row(str(i), c["name"], c["artist"], c["feed_url"] or "—")
    console.print(table)

    console.print("\n[bold]Ready-to-paste YAML:[/bold]\n")
    for c in candidates:
        if not c.get("feed_url"):
            continue
        handle = _slugify(c["name"]) or "unknown"
        console.print(
            f"  - name: {c['name']}\n"
            f"    platform: podcast\n"
            f"    handle: {handle}\n"
            f"    url: {c['view_url'] or ''}\n"
            f"    feed_url: {c['feed_url']}\n"
            f"    topics: []\n"
            f"    influence_score: 50\n"
            f"    contact:\n"
            f"      email: null\n"
            f"      other: null\n"
            f"    notes: \"Discovered via iTunes search. Verify before adding.\"\n"
        )


def discover_substack_recommendations(substack_url: str) -> list[dict[str, Any]]:
    """Best-effort scrape of a Substack's /recommendations page."""
    base = substack_url.rstrip("/")
    rec_url = base + "/recommendations"
    try:
        with httpx.Client(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(rec_url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.warning("recommendations fetch failed %s: %s", rec_url, exc)
        return []

    soup = BeautifulSoup(html, "html.parser")
    candidates: dict[str, dict[str, Any]] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(rec_url, href)
        parsed = urlparse(full)
        host = parsed.netloc.lower()
        if not host:
            continue
        # Look for substack.com hosts or links to root paths of likely substack-style domains
        is_substack_host = host.endswith(".substack.com")
        is_root_path = parsed.path in ("", "/")
        if not (is_substack_host or is_root_path):
            continue
        # Skip the source substack itself
        source_host = urlparse(base).netloc.lower()
        if host == source_host:
            continue
        # Skip obvious non-publication links
        if host in {"substack.com", "www.substack.com"}:
            continue

        key = host
        if key not in candidates:
            name = (a.get_text(" ", strip=True) or host)[:120]
            candidates[key] = {
                "host": host,
                "url": f"https://{host}",
                "guess_feed_url": f"https://{host}/feed",
                "label": name,
                "pending": True,
            }
    return list(candidates.values())


def print_substack_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        console.print("[yellow]No Substack recommendation candidates found.[/yellow]")
        return
    table = Table(title="Substack recommendation candidates")
    table.add_column("#")
    table.add_column("Host")
    table.add_column("Label")
    table.add_column("Guessed feed")
    for i, c in enumerate(candidates, 1):
        table.add_row(str(i), c["host"], c["label"], c["guess_feed_url"])
    console.print(table)

    console.print("\n[bold]Ready-to-paste YAML (verify each first!):[/bold]\n")
    for c in candidates:
        handle = c["host"].split(".")[0]
        console.print(
            f"  - name: {c['label']}\n"
            f"    platform: substack\n"
            f"    handle: {handle}\n"
            f"    url: {c['url']}\n"
            f"    feed_url: {c['guess_feed_url']}\n"
            f"    topics: []\n"
            f"    influence_score: 50\n"
            f"    contact:\n"
            f"      email: null\n"
            f"      other: null\n"
            f"    notes: \"Discovered via Substack recommendations; pending verification.\"\n"
        )


def discover_relevant_creators(
    topic_description: str,
    existing_names: list[str],
    limit: int = 30,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Use MiroThinker (web search) to find AI/tech creators relevant to a launch topic.

    Returns a list of candidate dicts:
      { name, platform, handle, url, feed_url, topics, influence_score, why_relevant }

    Caller is expected to review the output before adding to seed_creators.yaml.
    """
    from pitchfinder.llm import _call_json
    from pitchfinder.research import deep_research_model

    chosen = model or deep_research_model()
    skip_block = "\n".join(f"- {n}" for n in existing_names[:60])

    prompt = f"""You are helping expand a creator-pitch database. Find {limit} ACTIVE AI/tech creators
(newsletter writers, podcast hosts, YouTubers, bloggers) who would be the most
relevant audience for the launch described below. Use live web search to confirm
each one is real, active in 2025-2026, and has a working public channel.

LAUNCH WE'RE PITCHING:
{topic_description}

Already in our database — DO NOT return any of these (find different people):
{skip_block}

For each candidate:
- Confirm the channel exists and they posted/episode'd/uploaded in the last 6 months.
- Identify their platform (substack / blog / podcast / youtube).
- Find their channel URL (the human-facing page, not the RSS).
- Find their RSS / Atom / podcast feed URL where applicable (search the page source for <link rel="alternate" type="application/rss+xml"> or "/feed" or megaphone/transistor/anchor URLs).
- Suggest 3-5 topic tags.
- Estimate their influence (50=niche, 70=mid, 85=well-known in AI, 95=top-tier reach).
- Write one sentence on why they fit this launch.

Skip creators who are mainstream news outlets (Bloomberg, NYT, etc.) — we want
individual voices. Skip mainland Chinese media (机器之心, 量子位, 36Kr, etc.).
Skip X/Twitter-only personalities (we ingest RSS, not Twitter).
Prefer creators in software engineering, AI research, AI applications, agents,
verifiable reasoning, deep research, open-source models, frontier AI.

Return JSON only — an array of objects with this exact shape:
[
  {{
    "name": "Full Name",
    "platform": "substack|blog|podcast|youtube",
    "handle": "short-lowercase-handle-for-unique-key",
    "url": "https://...",
    "feed_url": "https://... or empty string if you couldn't find one",
    "topics": ["topic1", "topic2", "..."],
    "influence_score": 70,
    "why_relevant": "one-sentence rationale"
  }}
]"""
    try:
        result = _call_json(chosen, prompt, max_tokens=8192)
    except Exception as exc:
        logger.warning("discover_relevant_creators LLM call failed: %s", exc)
        return []

    if not isinstance(result, list):
        logger.warning("expected JSON array, got %s", type(result).__name__)
        return []

    cleaned: list[dict[str, Any]] = []
    existing_lower = {n.strip().lower() for n in existing_names}
    for entry in result:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name or name.lower() in existing_lower:
            continue
        platform = str(entry.get("platform", "")).strip().lower()
        if platform not in {"substack", "blog", "podcast", "youtube", "beehiiv"}:
            continue
        handle = str(entry.get("handle", "")).strip().lower() or _slugify(name)
        url = str(entry.get("url", "")).strip()
        feed_url = str(entry.get("feed_url", "")).strip() or None
        topics = entry.get("topics", []) or []
        topics = [str(t)[:60] for t in topics if t][:8]
        try:
            inf = int(entry.get("influence_score", 50))
        except (TypeError, ValueError):
            inf = 50
        inf = max(0, min(100, inf))
        why = str(entry.get("why_relevant", ""))[:400]
        cleaned.append(
            {
                "name": name,
                "platform": platform,
                "handle": handle,
                "url": url,
                "feed_url": feed_url,
                "topics": topics,
                "influence_score": inf,
                "why_relevant": why,
            }
        )
    return cleaned


def write_candidates_yaml(candidates: list[dict[str, Any]], output_path: Path) -> None:
    """Write candidates as YAML stanzas matching seed_creators.yaml format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "creators": [
            {
                "name": c["name"],
                "platform": c["platform"],
                "handle": c["handle"],
                "url": c["url"],
                "feed_url": c["feed_url"],
                "topics": c["topics"],
                "influence_score": c["influence_score"],
                "contact": {"email": None, "other": None},
                "notes": f"DISCOVERED via MiroThinker. Why: {c['why_relevant']}. Verify feed_url before loading.",
            }
            for c in candidates
        ]
    }
    output_path.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True))


def print_discovered_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        console.print("[yellow]No candidates returned.[/yellow]")
        return
    table = Table(title=f"Discovered creator candidates ({len(candidates)})")
    table.add_column("#")
    table.add_column("Name")
    table.add_column("Platform")
    table.add_column("Influence", justify="right")
    table.add_column("Why")
    for i, c in enumerate(candidates, 1):
        table.add_row(
            str(i),
            c["name"],
            c["platform"],
            str(c["influence_score"]),
            c["why_relevant"][:90],
        )
    console.print(table)


def _slugify(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug
