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
    """Route to MiroMind for mirothinker-* model ids, OpenRouter otherwise."""
    if model.startswith("mirothinker"):
        api_key = os.getenv("MIROMIND_API_KEY")
        if not api_key:
            raise RuntimeError("MIROMIND_API_KEY is not set")
        base_url = os.getenv("MIROMIND_BASE_URL", "https://api.miromind.ai/v1")
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


def _call_once(model: str, prompt: str, max_tokens: int) -> str:
    """One LLM round-trip. Returns raw assistant text (may be empty)."""
    client = _client_for_model(model)
    use_stream = model.startswith("mirothinker")

    if use_stream:
        stream = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
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
    )
    return resp.choices[0].message.content or ""


def _parse_json(text: str) -> Any:
    """Try strict, then fence-strip + first {...}/[...] salvage."""
    cleaned = _strip_fences(text)
    if not cleaned:
        raise ValueError("empty response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def _call_json(model: str, prompt: str, max_tokens: int = 1024, max_retries: int = 2) -> Any:
    """Make an LLM call expecting JSON output with bounded retries.

    Retries on:
    - empty / whitespace-only responses (common on flaky OpenRouter routes
      like v4-pro / glm-5.1 → ~40-86% empty in our testing)
    - transient SSE/network errors (MiroThinker occasionally drops the
      streamed body, e.g. 'peer closed connection')
    - JSON parse failures (model output a near-miss we couldn't salvage)

    MiroThinker uses streaming (SSE). Everything else uses non-stream.
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

Score 0-100:
- 90+: directly discusses the same problem space, techniques, or category
- 70-89: discusses an adjacent topic; author would have an informed view
- 50-69: tangentially related
- <50: not relevant

Return JSON only: {{"score": <int>, "reason": "<one sentence>"}}"""
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
