"""MiroThinker deep-research enrichment for ranked creators.

Uses MiroMind's mirothinker-1-7-deepresearch model via OpenAI-compatible
chat completions. Each call does web search + verification under the hood.
Expensive (~$0.30 per creator) but produces high-quality "verified context"
that is otherwise impossible to get from in-DB items alone (we only keep
the last 90 days).
"""

from __future__ import annotations

import logging
import os
import time

from pitchfinder.llm import _call_json

logger = logging.getLogger("pitchfinder.research")

DEFAULT_MIROTHINKER_MODEL = "mirothinker-1-7-deepresearch"

# Empty/default payload returned on any failure so the renderer never crashes.
EMPTY_PAYLOAD = {
    "verified_active": None,
    "recent_themes": [],
    "sharp_quotes": [],
    "current_stance": "",
    "pitch_hook": "",
    "contact": {
        "email": "",
        "twitter": "",
        "linkedin": "",
        "contact_form": "",
        "preferred_channel": "",
        "notes": "",
    },
    "error": None,
}


def deep_research_model() -> str:
    return os.getenv("MIROMIND_DEEPRESEARCH_MODEL", DEFAULT_MIROTHINKER_MODEL)


def deep_dive_creator(
    creator_name: str,
    creator_url: str,
    launch_description: str,
    max_tokens: int = 4096,
    model: str | None = None,
) -> dict:
    """Run a deep-research pass on a single creator.

    If model starts with 'mirothinker' (default), uses MiroThinker's
    web-search + verification prompt. Otherwise (e.g. Sonnet 4.6 used as
    cheaper fallback for lower-ranked creators) uses a best-effort-from-
    training-data prompt that is honest about what cannot be verified.

    Returns a dict matching EMPTY_PAYLOAD's shape. Never raises — failures
    return EMPTY_PAYLOAD with `error` populated.
    """
    chosen_model = model or deep_research_model()
    is_research_agent = chosen_model.startswith("mirothinker")

    if is_research_agent:
        prompt = _research_prompt_web_search(creator_name, creator_url, launch_description)
    else:
        prompt = _research_prompt_best_effort(creator_name, creator_url, launch_description)

    try:
        result = _call_json(chosen_model, prompt, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning("deep_dive_creator failed for %s (model=%s): %s", creator_name, chosen_model, exc)
        payload = dict(EMPTY_PAYLOAD)
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["_source_model"] = chosen_model
        return payload

    if not isinstance(result, dict):
        payload = dict(EMPTY_PAYLOAD)
        payload["error"] = "non-dict response"
        payload["_source_model"] = chosen_model
        return payload

    # Normalize fields with safe defaults
    out = dict(EMPTY_PAYLOAD)
    out["verified_active"] = bool(result.get("verified_active", True)) if result.get("verified_active") is not None else None
    out["recent_themes"] = [str(t) for t in result.get("recent_themes", []) if t][:8]
    quotes_raw = result.get("sharp_quotes", []) or []
    cleaned_quotes: list[dict] = []
    for q in quotes_raw:
        if isinstance(q, dict) and q.get("quote"):
            cleaned_quotes.append(
                {
                    "quote": str(q["quote"])[:600],
                    "source": str(q.get("source", "")),
                    "date": str(q.get("date", ""))[:7],
                }
            )
    out["sharp_quotes"] = cleaned_quotes[:5]
    out["current_stance"] = str(result.get("current_stance", ""))[:1200]
    out["pitch_hook"] = str(result.get("pitch_hook", ""))[:400]

    contact_in = result.get("contact") or {}
    if isinstance(contact_in, dict):
        out["contact"] = {
            "email": str(contact_in.get("email", ""))[:200],
            "twitter": str(contact_in.get("twitter", ""))[:80],
            "linkedin": str(contact_in.get("linkedin", ""))[:300],
            "contact_form": str(contact_in.get("contact_form", ""))[:500],
            "preferred_channel": str(contact_in.get("preferred_channel", ""))[:300],
            "notes": str(contact_in.get("notes", ""))[:400],
        }
    out["_source_model"] = chosen_model
    return out


def _research_prompt_web_search(creator_name: str, creator_url: str, launch_description: str) -> str:
    return f"""You are doing deep research on an AI/tech creator to inform a personalized pitch.

CREATOR: {creator_name}
CHANNEL: {creator_url}

LAUNCH WE'RE PITCHING:
{launch_description}

Do the following — use web search and verification:
1. Confirm this creator is still active (posts/episodes/videos in the last 6 months). If not, set verified_active to false and stop early.
2. Identify 3-5 dominant themes they have been covering in the last 6-12 months.
3. Find 2-3 sharp, specific quotes or arguments they have made that connect to the launch's problem space. Each quote MUST be verifiable — include the source URL and approximate date (YYYY-MM).
4. Summarize their current stance on the launch's topic area in one paragraph (2-4 sentences).
5. Suggest the single best opening line for a cold pitch email to them, referencing one specific recent piece of their work.
6. Find their PUBLIC contact info for pitching. Search their channel, About page, Substack profile, podcast site, LinkedIn, X/Twitter bio. List ONLY publicly listed channels — never guess emails. If you can find an explicit "pitch me at X" or "for inquiries email Y", note that in preferred_channel. Leave fields empty rather than fabricate.

Return JSON only, no prose, no markdown fences:
{{
  "verified_active": true,
  "recent_themes": ["theme 1", "theme 2", "..."],
  "sharp_quotes": [
    {{"quote": "exact words in quotes", "source": "https://...", "date": "2026-04"}}
  ],
  "current_stance": "one paragraph summary",
  "pitch_hook": "one-line cold-email opener",
  "contact": {{
    "email": "name@domain.com or empty",
    "twitter": "@handle or empty",
    "linkedin": "https://linkedin.com/in/... or empty",
    "contact_form": "https://... or empty",
    "preferred_channel": "e.g. 'Twitter DM, replies fastest there' or empty",
    "notes": "e.g. 'No public email; pitch via podcast inquiry form' or empty"
  }}
}}"""


def _research_prompt_best_effort(creator_name: str, creator_url: str, launch_description: str) -> str:
    return f"""You are helping prepare a pitch to an AI/tech creator. You do NOT have live web search — answer from your training data and be honest about uncertainty. Leave any field empty rather than fabricate.

CREATOR: {creator_name}
CHANNEL: {creator_url}

LAUNCH WE'RE PITCHING:
{launch_description}

Based ONLY on what you reliably know about this creator from your training:
1. Do you remember whether they were still active recently? (Set verified_active to true if you have strong evidence, null/missing if uncertain.)
2. What 3-5 dominant themes do they typically cover?
3. If you can recall any specific argument or position they have made that connects to the launch's problem space, list it as a quote. Do NOT fabricate quotes. If unsure, leave sharp_quotes as an empty array.
4. Summarize their general stance on the launch's topic area in one paragraph (based on training data).
5. Suggest the single best opening line for a cold pitch email to them, referencing the kind of work they do.
6. Public contact info you reliably know (e.g. well-known Twitter handle, public email if widely known). NEVER guess emails. Leave fields empty if uncertain.

Return JSON only, no prose, no markdown fences:
{{
  "verified_active": true,
  "recent_themes": ["theme 1", "theme 2", "..."],
  "sharp_quotes": [
    {{"quote": "exact words in quotes", "source": "https://...", "date": "2026-04"}}
  ],
  "current_stance": "one paragraph summary",
  "pitch_hook": "one-line cold-email opener",
  "contact": {{
    "email": "name@domain.com or empty",
    "twitter": "@handle or empty",
    "linkedin": "https://linkedin.com/in/... or empty",
    "contact_form": "https://... or empty",
    "preferred_channel": "e.g. 'Twitter DM, replies fastest there' or empty",
    "notes": "Mark anything uncertain. E.g. 'no public email known'"
  }}
}}"""


def time_deep_dive(creator_name: str, creator_url: str, launch_description: str) -> tuple[dict, float]:
    """Self-test helper: returns (payload, seconds)."""
    t0 = time.time()
    payload = deep_dive_creator(creator_name, creator_url, launch_description)
    return payload, time.time() - t0
