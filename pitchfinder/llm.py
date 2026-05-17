"""OpenRouter (OpenAI-compatible) LLM client + 3 prompts: topic extraction, relevance scoring, pitch angles."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger("pitchfinder.llm")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RELEVANCE_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_PITCH_MODEL = "deepseek/deepseek-chat-v3.1"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    # OpenRouter recommends sending HTTP-Referer and X-Title headers
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


def _call_json(model: str, prompt: str, max_tokens: int = 1024) -> Any:
    """Make a single LLM call expecting JSON output."""
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Salvage: pull out first {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        logger.warning("JSON parse failed (model=%s): %s; raw=%r", model, exc, cleaned[:300])
        raise


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
