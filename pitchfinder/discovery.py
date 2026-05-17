"""Discovery helpers — bootstrap the seed library from external sources."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("pitchfinder.discovery")

USER_AGENT = "pitchfinder/0.1"
HTTP_TIMEOUT = 15.0

console = Console()

ITUNES_SEARCH = "https://itunes.apple.com/search"


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
