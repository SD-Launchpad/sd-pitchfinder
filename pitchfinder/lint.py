"""LLM content-aware URL validator for creator records.

Catches the failure mode where a URL returns HTTP 200 but the page is
actually a domain-for-sale parking page, a squatter, dead site, or
unrelated content. Spotted in production when seed_creators.yaml had
`https://www.nopriors.com` (a GoDaddy parking page) for the No Priors
podcast — HTTP 200 hid the rot.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from pitchfinder.llm import _call_json, _relevance_model

logger = logging.getLogger("pitchfinder.lint")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Intentionally don't set Accept-Encoding — httpx negotiates gzip/deflate
    # and can decompress them. Including 'br' (brotli) without the brotli
    # package would return compressed bytes interpreted as text → gibberish.
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
}
HTTP_TIMEOUT = 12.0
CONTENT_CAP = 3000

VALID_STATUSES = {"ok", "squatter", "wrong_content", "uncertain"}


@dataclass
class LintResult:
    creator_id: int
    name: str
    url: str
    http_status: int | None
    content_status: str  # ok | squatter | wrong_content | uncertain | unreachable | redirected
    reason: str
    final_url: str | None = None


def _fetch_text(url: str) -> tuple[int | None, str, str | None]:
    """Fetch URL, return (status_code, plain_text_excerpt, final_url).

    Status code None = network/DNS failure (not even an HTTP response).
    """
    try:
        with httpx.Client(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        ) as client:
            resp = client.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Drop boilerplate that swamps signal on bot-protected sites like YouTube
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Surface <title> + <meta description> first — they're harder to bot-block
        head_bits: list[str] = []
        if soup.title and soup.title.string:
            head_bits.append(f"TITLE: {soup.title.string.strip()}")
        for m in soup.find_all("meta"):
            name = (m.get("name") or m.get("property") or "").lower()
            if name in {"description", "og:title", "og:description", "og:site_name", "twitter:title", "twitter:description"}:
                content = (m.get("content") or "").strip()
                if content:
                    head_bits.append(f"{name.upper()}: {content}")
        body_text = soup.get_text(" ", strip=True)
        excerpt = ("\n".join(head_bits) + "\n" + body_text).strip()[:CONTENT_CAP]
        return resp.status_code, excerpt, str(resp.url)
    except Exception as exc:
        logger.warning("fetch failed %s: %s", url, exc)
        return None, str(exc)[:200], None


def _llm_validate(creator_name: str, url: str, text: str) -> dict:
    prompt = f"""You are verifying that a URL belongs to a specific creator's actual channel.

CREATOR NAME: {creator_name}
URL: {url}
PAGE TEXT (first {CONTENT_CAP} chars, HTML stripped):
{text}

Decide what this page actually is:
- "ok": the page is the creator's real channel (newsletter, blog, podcast site, YouTube channel, Linktree, etc) — it mentions their name, brand, episodes, or matches the expected platform layout.
- "squatter": domain-for-sale / GoDaddy / Sedo / Afternic parking page / "This domain may be for sale" / "buy this domain" / generic registrar landing.
- "wrong_content": the page loads but is about a different person/company than the creator.
- "uncertain": page loads but content is ambiguous (e.g. CAPTCHA, login wall, JS-only site with no visible text, anti-bot block).

Return JSON only: {{"status": "ok|squatter|wrong_content|uncertain", "reason": "<one sentence citing specific evidence from the page>"}}"""
    try:
        result = _call_json(_relevance_model(), prompt, max_tokens=256)
        if not isinstance(result, dict):
            return {"status": "uncertain", "reason": "non-dict LLM response"}
        status = str(result.get("status", "uncertain"))
        if status not in VALID_STATUSES:
            status = "uncertain"
        return {"status": status, "reason": str(result.get("reason", ""))[:400]}
    except Exception as exc:
        return {"status": "uncertain", "reason": f"LLM error: {type(exc).__name__}"}


def lint_creators(rows: list[dict], concurrency: int = 6) -> list[LintResult]:
    """Lint every (creator_id, name, url) row. Concurrent fetches + LLM."""

    def _one(row: dict) -> LintResult:
        url = row["url"]
        code, text, final = _fetch_text(url)
        if code is None:
            return LintResult(
                creator_id=row["id"],
                name=row["name"],
                url=url,
                http_status=None,
                content_status="unreachable",
                reason=text,
                final_url=None,
            )
        if code >= 400:
            return LintResult(
                creator_id=row["id"],
                name=row["name"],
                url=url,
                http_status=code,
                content_status="unreachable",
                reason=f"HTTP {code}",
                final_url=final,
            )
        verdict = _llm_validate(row["name"], url, text)
        return LintResult(
            creator_id=row["id"],
            name=row["name"],
            url=url,
            http_status=code,
            content_status=verdict["status"],
            reason=verdict["reason"],
            final_url=final if final and final != url else None,
        )

    results: list[LintResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, row) for row in rows]
        for fut in as_completed(futures):
            results.append(fut.result())
    # Preserve original ordering by creator_id for stable output
    results.sort(key=lambda r: r.creator_id)
    return results
