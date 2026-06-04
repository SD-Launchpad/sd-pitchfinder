"""Unit tests for the pure-logic pieces of PitchFinder v2 (no network / no LLM)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from pitchfinder.config import load_brand_config
from pitchfinder.llm import _build_tier_prompt
from pitchfinder.db import get_conn, init_schema
from pitchfinder.discovery import resolve_feed_url
from pitchfinder.pipeline import _render_csv
from pitchfinder.web_discovery import _normalize

REPO = Path(__file__).resolve().parents[1]


# ---------- config ----------

def test_load_brand_config_apodex():
    cfg = load_brand_config(REPO / "brands" / "apodex.yaml")
    assert cfg.brand == "apodex"
    assert "verifiable AI" in cfg.themes
    assert cfg.tiering.model.startswith("anthropic/")   # judgment → strong model
    assert cfg.deepdive.top_n == 5
    desc = cfg.launch_description()
    assert "verifiable AI" in desc and "axiommath" in desc.lower() or "axiommath.ai" in desc


# ---------- feed resolution (offline branches) ----------

def test_resolve_feed_url_substack():
    assert resolve_feed_url("substack", "https://www.interconnects.ai/") == "https://www.interconnects.ai/feed"


def test_resolve_feed_url_podcast_no_name_is_none():
    assert resolve_feed_url("podcast", "https://example.com") is None


# ---------- web discovery URL normalization ----------

def test_normalize_substack():
    assert _normalize("https://natolambert.substack.com/p/abc", "substack") == (
        "substack", "https://natolambert.substack.com", "natolambert",
    )


def test_normalize_youtube_channel_handle():
    plat, root, handle = _normalize("https://www.youtube.com/@AndrejKarpathy/videos", "youtube")
    assert plat == "youtube" and root.endswith("/@AndrejKarpathy") and handle == "andrejkarpathy"


def test_normalize_youtube_watch_is_skipped():
    assert _normalize("https://www.youtube.com/watch?v=xyz", "youtube") is None


def test_normalize_social_skipped():
    assert _normalize("https://twitter.com/someone", "blog") is None


def test_normalize_blog():
    plat, root, handle = _normalize("https://simonwillison.net/2026/abc", "blog")
    assert plat == "blog" and root == "https://simonwillison.net"


# ---------- CSV rendering ----------

def test_render_csv_columns_and_values():
    creators = [{
        "creator_id": 1, "name": "Jane Doe", "platform": "substack",
        "url": "https://jane.substack.com", "top_score": 88, "influence_score": 70,
        "tier": "A", "tier_rationale": "core expert",
        "top_items": [{"title": "Verifiable AI", "url": "https://jane.substack.com/p/x"}],
        "angles": [{"angle": "a1"}, {"angle": "a2"}],
        "deep_dive": {"contact": {"email": "jane@x.com", "twitter": "@jane"},
                      "verified_active": True, "recent_themes": ["evals", "rl"]},
    }]
    out = _render_csv("desc", creators, 1)
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 1
    r = rows[0]
    assert r["tier"] == "A"
    assert r["email"] == "jane@x.com"
    assert r["twitter"] == "@jane"
    assert r["verified_active"] == "yes"
    assert r["angle_1"] == "a1" and r["angle_2"] == "a2"
    assert r["top_match_title"] == "Verifiable AI"
    assert r["recent_themes"] == "evals; rl"


def test_render_csv_handles_missing_fields():
    # minimal creator (no deep_dive, no tier, no angles) must not raise
    creators = [{
        "creator_id": 2, "name": "Min", "platform": "blog",
        "top_score": 60, "influence_score": 50, "top_items": [],
    }]
    out = _render_csv("d", creators, 1)
    rows = list(csv.DictReader(io.StringIO(out)))
    assert rows[0]["name"] == "Min" and rows[0]["tier"] == "" and rows[0]["email"] == ""


# ---------- tier prompt construction (neutral-third-party gate) ----------

def test_build_tier_prompt_includes_url_competitors_and_neutral_rule():
    batch = [{
        "creator_id": 7, "name": "GetVoIP", "platform": "blog",
        "url": "https://getvoip.com", "influence_score": 80, "signal": "Best AI receptionists 2026",
    }]
    p = _build_tier_prompt("Solvea — AI receptionist.", ["Retell AI", "Vapi"], batch)
    # creator url surfaced for the model to judge vendor-ness
    assert "https://getvoip.com" in p
    # competitors injected as a drop list
    assert "Retell AI" in p and "Vapi" in p
    # the neutral-third-party gate + vendor drop rule present
    assert "NEUTRAL" in p and "drop" in p.lower()
    assert "competitor" in p.lower() and "affiliate" in p.lower()


def test_build_tier_prompt_no_competitors_ok():
    batch = [{"creator_id": 1, "name": "X", "platform": "substack",
              "url": None, "influence_score": 50, "signal": "y"}]
    p = _build_tier_prompt("desc", None, batch)
    assert "no-url" in p and "1\tX" in p


# ---------- db migration idempotency ----------

def test_schema_has_tiers_and_brand(tmp_path):
    db = str(tmp_path / "t.db")
    init_schema(db)
    init_schema(db)  # idempotent — must not raise on re-run
    conn = get_conn(db)
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "creator_tiers" in tables
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(searches)")}
        assert "brand" in cols
    finally:
        conn.close()
