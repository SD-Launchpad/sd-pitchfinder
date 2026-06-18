"""OpenRouter (OpenAI-compatible) LLM client + 3 prompts: topic extraction, relevance scoring, pitch angles."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger("pitchfinder.llm")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RELEVANCE_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_PITCH_MODEL = "anthropic/claude-sonnet-4.6"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _client_for_model(model: str) -> OpenAI:
    """Route to Apodex for apodex-* model ids, OpenRouter otherwise."""
    if model.startswith("apodex"):
        api_key = os.getenv("APODEX_API_KEY")
        if not api_key:
            raise RuntimeError("APODEX_API_KEY is not set")
        base_url = os.getenv("APODEX_BASE_URL", "https://api.apodex.ai/v1")
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    default_headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if referer:
        default_headers["HTTP-Referer"] = referer
    app_name = os.getenv("OPENROUTER_APP_NAME")
    if app_name:
        default_headers["X-Title"] = app_name
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)


def _relevance_model() -> str:
    return os.getenv("PITCHFINDER_RELEVANCE_MODEL", DEFAULT_RELEVANCE_MODEL)


def _pitch_model() -> str:
    return os.getenv("PITCHFINDER_PITCH_MODEL", DEFAULT_PITCH_MODEL)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


# Hard per-call timeout (s). Apodex SSE stream can otherwise hang indefinitely
# (a stuck campaign once ran 16h). Normal calls finish far under this.
LLM_TIMEOUT_SECONDS = 240


def _call_once(model: str, prompt: str, max_tokens: int) -> str:
    """One LLM round-trip. Returns raw assistant text (may be empty)."""
    client = _client_for_model(model)
    use_stream = model.startswith("apodex")

    if use_stream:
        stream = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        buf: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                buf.append(delta.content)
        return "".join(buf)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return resp.choices[0].message.content or ""


# A backslash NOT starting a valid JSON escape (" \ / b f n r t u) is illegal.
# Apodex occasionally emits these (e.g. raw LaTeX / Windows paths / "\$"),
# which makes json.loads raise "Invalid \escape" and the whole deep-dive return
# empty. Doubling such lone backslashes recovers the payload.
_BAD_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _sanitize_escapes(s: str) -> str:
    return _BAD_ESCAPE_RE.sub(r"\\\\", s)


def _loads_lenient(s: str) -> Any:
    """json.loads, retrying once with invalid backslash-escapes sanitized."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_sanitize_escapes(s))


def _parse_json(text: str) -> Any:
    """Try strict, then fence-strip + first {...}/[...] salvage, tolerating
    invalid backslash escapes that some models emit."""
    cleaned = _strip_fences(text)
    if not cleaned:
        raise ValueError("empty response")
    try:
        return _loads_lenient(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            return _loads_lenient(m.group(1))
        raise


def _is_timeout(exc: Exception) -> bool:
    """Detect read/connect timeouts (httpx/openai timeout classes or 'timeout' msg)."""
    s = f"{type(exc).__name__} {exc}".lower()
    return "timeout" in s or "timed out" in s


def _call_json(model: str, prompt: str, max_tokens: int = 1024, max_retries: int = 2) -> Any:
    """Make an LLM call expecting JSON output with bounded retries.

    Retries on:
    - empty / whitespace-only responses (common on flaky OpenRouter routes
      like v4-pro / glm-5.1 → ~40-86% empty in our testing)
    - transient SSE/network errors (Apodex occasionally drops the
      streamed body, e.g. 'peer closed connection')
    - JSON parse failures (model output a near-miss we couldn't salvage)

    Does NOT retry on timeout: a deepresearch timeout usually means the server
    stalled, and retrying just burns another full LLM_TIMEOUT_SECONDS window
    (once made a single deep-dive eat 3×240s). Apodex uses streaming (SSE).
    Final failure raises so the caller can degrade gracefully.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            text = _call_once(model, prompt, max_tokens)
            if not text.strip():
                raise ValueError("empty response from model")
            return _parse_json(text)
        except Exception as exc:
            last_exc = exc
            if _is_timeout(exc):
                logger.warning("LLM call timed out, not retrying (model=%s): %s", model, exc)
                break
            if attempt < max_retries:
                # Exponential backoff with jitter so we don't sync retry storms.
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "LLM call attempt %d/%d failed (model=%s): %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    model,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "LLM call exhausted retries (model=%s, attempts=%d): %s",
                    model,
                    attempt + 1,
                    exc,
                )
    assert last_exc is not None
    raise last_exc


# ---------- 1. Topic extraction ----------


def extract_topics(description: str) -> dict:
    prompt = f"""From this product launch description, extract searchable topics.

Launch description:
{description}

Return JSON only:
{{
  "topics": ["..."],
  "keywords": ["..."],
  "summary_for_match": "..."
}}

- topics: 5-10 topic tags
- keywords: 5-10 matching keywords
- summary_for_match: one sentence for matching against article topics"""
    try:
        result = _call_json(_relevance_model(), prompt, max_tokens=512)
        if not isinstance(result, dict):
            return {"topics": [], "keywords": [], "summary_for_match": description[:200]}
        return result
    except Exception as exc:
        logger.warning("extract_topics failed: %s", exc)
        return {"topics": [], "keywords": [], "summary_for_match": description[:200]}


# ---------- 2. Relevance scoring ----------


def score_relevance(description: str, item: dict, content_type_label: str) -> dict:
    title = item.get("title", "")
    summary = (item.get("summary", "") or "")[:1500]
    ct_cap = content_type_label.capitalize()
    prompt = f"""Score how relevant this {content_type_label} is to a product launch.

Launch description:
{description}

{ct_cap} title: {title}
{ct_cap} description/excerpt: {summary}

Score 0-100 on TOPICAL relevance AND source quality together:
- 90+: an individual expert / established outlet engaging deeply and specifically
  with the same problem space, techniques, or category
- 70-89: a credible voice on an adjacent topic; the author would have an informed view
- 50-69: tangentially related, or relevant but shallow
- <50: not relevant

Penalise hard (cap at 40) if the source looks like a CONTENT FARM rather than a
real expert worth pitching: generic AI news aggregators / rewrites, SEO listicles
("Top 10 AI tools…"), keyword-stuffed roundups, churned daily briefs with no
individual point of view, or vague hype with no concrete argument. A piece that
merely name-drops the keywords without substance is NOT a 90.

Return JSON only: {{"score": <int>, "reason": "<one sentence; note if content-farm>"}}"""
    result = _call_json(_relevance_model(), prompt, max_tokens=256)
    if not isinstance(result, dict):
        return {"score": 0, "reason": "non-dict response"}
    try:
        score = int(result.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return {"score": max(0, min(100, score)), "reason": str(result.get("reason", ""))[:500]}


# ---------- 3. Pitch angles ----------


def generate_pitch_angles(description: str, creator_name: str, top_items: list[dict]) -> list[dict]:
    lines: list[str] = []
    for i, it in enumerate(top_items[:3], 1):
        date_s = (it.get("published_at") or "")[:10]
        ct = it.get("content_type", "")
        summary = (it.get("summary") or "")[:300]
        title = it.get("title", "")
        lines.append(f'{i}. "{title}" ({date_s}, {ct}) — {summary}')
    items_block = "\n".join(lines) if lines else "(no recent relevant content)"

    prompt = f"""We're pitching this product launch to a creator. Based on their recent work,
suggest 2-3 specific pitch angles.

Launch description:
{description}

Creator: {creator_name}
Recent relevant content:
{items_block}

For each angle:
- Reference a specific argument or theme from the creator's recent work
- Show how the launch extends, challenges, or supports that argument
- 1-2 sentences, concrete and specific (no generic "this aligns with your interests")

Return JSON only:
[
  {{"angle": "...", "references_item": "<title>"}},
  ...
]"""
    try:
        result = _call_json(_pitch_model(), prompt, max_tokens=1024)
    except Exception as exc:
        logger.warning("generate_pitch_angles failed: %s", exc)
        return []

    if not isinstance(result, list):
        return []
    cleaned: list[dict] = []
    for entry in result:
        if isinstance(entry, dict) and entry.get("angle"):
            cleaned.append(
                {
                    "angle": str(entry["angle"]),
                    "references_item": str(entry.get("references_item", "")),
                }
            )
    return cleaned


# ---------- 4. Tier classification (A / B / drop) ----------


def _build_tier_prompt(brand_summary: str, competitors: list[str] | None, batch: list[dict]) -> str:
    """Construct the A/B/drop classification prompt for one batch. Pure string
    building (no network) so it's unit-testable."""
    rows = "\n".join(
        f'{c["creator_id"]}\t{c["name"]} ({c["platform"]}, influence {c.get("influence_score", "?")}) '
        f'[{c.get("url") or "no-url"}] — recent: {c.get("signal", "")[:240]}'
        for c in batch
    )
    comp_line = (
        "Brand competitors (DROP these and any of their owned blogs/media): "
        + ", ".join(competitors) + "\n\n"
        if competitors else ""
    )
    return f"""You are triaging creators for a founder's press/outreach list.

GOAL: find NEUTRAL, INDEPENDENT third-party thought leaders — newsletters,
podcasts, YouTubers, editorial media, individual experts — that we could
partner with to amplify this launch.

Brand / launch context:
{brand_summary}

{comp_line}For EACH creator below, assign an outreach tier:
- "A": high relevance AND high confidence — an independent expert or established
  editorial outlet whose audience maps directly to this launch. Strongly recommend.
- "B": moderate or narrower relevance, or lower confidence, but still a genuine,
  on-topic, independent voice worth a pitch. Recall-first: unsure between B and
  drop on an INDEPENDENT creator → pick B.
- "drop": see the hard rules below.

ALWAYS "drop" (on-topic wording is NOT enough to save these):
1. NOT NEUTRAL — the source is a company/vendor and its blog/channel exists to
   market its OWN product or business: a SaaS vendor's blog, OR a commercial
   product-comparison / lead-gen / affiliate-marketplace site that sells or
   monetises this exact category (e.g. a "best AI receptionist" directory that
   sells leads, like GetVoIP). We cannot partner with commercially self-interested
   sources to promote us.
2. COMPETITOR — one of the listed competitors, or any direct competitor, or their
   owned media.
3. content farm / SEO aggregator / generic keyword-matching rewrite / off-topic.

KEEP (A/B) independent newsletters, podcasts, YouTubers, editorial media and
individual experts EVEN IF they run ads or affiliate links — an independent
reviewer who covers many products is still a neutral third party we want. The
test is "neutral independent voice" vs "company marketing its own product/category".
Use the [url] as a signal: a product/company/comparison domain → likely drop;
substack/youtube/personal-blog/editorial domain → likely keep.

Creators (id<TAB>name (platform, influence) [url] — recent):
{rows}

Return JSON only, one object per creator:
[{{"creator_id": <int>, "tier": "A"|"B"|"drop", "rationale": "<one short sentence; if drop, say why: vendor/competitor/farm>"}}]"""


def classify_tiers(
    brand_summary: str,
    creators: list[dict],
    model: str | None = None,
    batch_size: int = 10,
    competitors: list[str] | None = None,
) -> dict[int, dict]:
    """Classify each creator into A / B / drop for an outreach campaign.

    Recall-first: only `drop` content farms / clearly off-topic sources; keep
    every genuine, on-topic creator (A = high relevance + high confidence,
    strongly recommend; B = moderate relevance, worth reaching out to).

    `creators` items need: creator_id, name, platform, influence_score, and
    `signal` (a short string, e.g. top item titles). Returns
    {creator_id: {"tier": "A"|"B"|"drop", "rationale": str}}. Batched to keep
    cost low. Creators missing from the model's reply default to tier B.
    """
    out: dict[int, dict] = {}
    chosen = model or _relevance_model()
    for start in range(0, len(creators), batch_size):
        batch = creators[start:start + batch_size]
        prompt = _build_tier_prompt(brand_summary, competitors, batch)
        try:
            result = _call_json(chosen, prompt, max_tokens=1200)
        except Exception as exc:
            logger.warning("classify_tiers batch failed (%s): %s", chosen, exc)
            result = []
        if isinstance(result, list):
            for e in result:
                if not isinstance(e, dict):
                    continue
                try:
                    cid = int(e.get("creator_id"))
                except (TypeError, ValueError):
                    continue
                tier = str(e.get("tier", "B")).strip().upper()
                tier = {"A": "A", "B": "B", "DROP": "drop"}.get(tier, "B")
                out[cid] = {"tier": tier, "rationale": str(e.get("rationale", ""))[:300]}
        # default any unreturned creators in this batch to B
        for c in batch:
            out.setdefault(c["creator_id"], {"tier": "B", "rationale": "(defaulted)"})
    return out
