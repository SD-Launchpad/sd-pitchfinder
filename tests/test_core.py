"""Unit tests for the pure-logic pieces of PitchFinder v2 (no network / no LLM)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from pitchfinder.config import load_brand_config
from pitchfinder.llm import _build_tier_prompt, _parse_json
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


# ---------- social search (LinkedIn/X) name-validated ----------

def test_pick_social_name_match_and_reject():
    from pitchfinder.contacts import _pick_social
    name = "Sharon Goldman"
    # slug/handle 含名字 → 采信（连写 handle 也命中）
    li, tw = _pick_social([
        {"url": "https://www.linkedin.com/in/sharon-goldman-12a"},
        {"url": "https://x.com/sharongoldman"},
    ], name)
    assert li == "sharon-goldman-12a" and tw == "sharongoldman"
    # 与名字无关 → 丢（防抓错人）
    assert _pick_social([
        {"url": "https://www.linkedin.com/in/john-smith"},
        {"url": "https://x.com/randomdude"},
    ], name) == ("", "")
    # 非 handle 路径词（intent/share）排除
    assert _pick_social([{"url": "https://twitter.com/intent/tweet"}], name)[1] == ""
    # 大小写无关
    assert _pick_social([{"url": "https://x.com/SharonGoldman"}], name)[1] == "SharonGoldman"
    # 名字太短(连写 <4) → 不校验、不采信
    assert _pick_social([{"url": "https://x.com/whoever"}], "AI") == ("", "")
    # 泛名 creator 防误匹配：连写互不包含 → 砍（@MusicStarAI ≠ "AI Music Unmuted"）
    assert _pick_social([{"url": "https://x.com/MusicStarAI"}], "AI Music Unmuted")[1] == ""
    # handle ≈ 名字连写 → 留
    assert _pick_social([{"url": "https://x.com/ainewsletter"}], "AI Newsletter")[1] == "ainewsletter"


def test_run_search_default_concurrency_is_16():
    import inspect
    from pitchfinder.pipeline import run_search
    assert inspect.signature(run_search).parameters["concurrency"].default == 16


# ---------- scoring prefilter (quality-preserving throughput optimization) ----------

def test_prefilter_keeps_keyword_overlap_drops_zero_overlap():
    from pitchfinder.pipeline import _prefilter_candidates
    items = [
        {"title": "Suno vs Udio: the AI music showdown", "summary": ""},   # competitor term
        {"title": "How to bake sourdough bread", "summary": "yeast tips"}, # zero overlap -> drop
        {"title": "New video-to-music tool launches", "summary": "score"}, # theme term
        {"title": "", "summary": "generative music for creators"},         # theme term in summary
    ]
    terms = ["video to music", "AI music", "Suno", "generative music"]
    kept = _prefilter_candidates(items, terms)
    titles = [k["title"] for k in kept]
    assert "How to bake sourdough bread" not in titles      # the only off-topic one is removed
    assert len(kept) == 3


def test_prefilter_case_insensitive_and_ignores_short_terms():
    from pitchfinder.pipeline import _prefilter_candidates
    items = [{"title": "MUSIC AI", "summary": ""}, {"title": "xy", "summary": "ab"}]
    kept = _prefilter_candidates(items, ["music", "ai"])  # "ai" (<3) ignored; "music" matches
    assert len(kept) == 1 and kept[0]["title"] == "MUSIC AI"


def test_prefilter_phrase_word_partial_match_no_miss():
    from pitchfinder.pipeline import _prefilter_candidates
    # term is a multi-word phrase; an item containing only one of its long words
    # MUST be kept (else we漏召 a likely-relevant item). recall-first.
    items = [{"title": "Best AI music tools 2026", "summary": ""}]
    kept = _prefilter_candidates(items, ["AI Music Generation"])  # no full phrase, but "music"
    assert len(kept) == 1


def test_prefilter_no_usable_terms_returns_all():
    from pitchfinder.pipeline import _prefilter_candidates
    items = [{"title": "x", "summary": "y"}]
    assert _prefilter_candidates(items, []) == items          # no terms -> no filtering
    assert _prefilter_candidates(items, ["ab"]) == items      # only short terms -> no filtering


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


# ---------- JSON parse robustness (Apodex bad escapes) ----------

def test_parse_json_tolerates_invalid_backslash_escape():
    # Apodex-style payload with an illegal \$ and a fenced wrapper
    raw = '```json\n{"pitch_hook": "save \\$10k", "ok": true}\n```'
    out = _parse_json(raw)
    assert out["ok"] is True and "10k" in out["pitch_hook"]


def test_parse_json_still_parses_clean():
    assert _parse_json('{"a": 1}') == {"a": 1}
    assert _parse_json('[{"x": "y\\n"}]') == [{"x": "y\n"}]  # valid escapes untouched


# ---------- contact enrichment ----------

def test_contacts_same_org_and_denylist():
    from pitchfinder.contacts import _same_org, _clean_emails
    assert _same_org("jane@acme.com", "https://acme.com/blog") is True
    assert _same_org("jane@acme.com", "https://other.com") is False
    assert _same_org("x@gmail.com", "https://gmail.com") is False        # generic host
    # denylist drops transactional locals; keeps real ones
    assert _clean_emails(["billing@x.com", "noreply@x.com", "hello@x.com"]) == ["hello@x.com"]


def test_contacts_parse_substack_preloads():
    from pitchfinder.contacts import parse_substack
    blob = '{\\"pub\\":{\\"author_bio\\":\\"hi\\",\\"twitter_screen_name\\":\\"jackclarkSF\\",\\"support_email\\":\\"team@importai.org\\"}}'
    html = f'<script>window._preloads = JSON.parse("{blob}")</script>'
    info = parse_substack(html)
    assert info["twitter"] == ["jackclarkSF"]
    assert "team@importai.org" in info["emails"]


def test_resolve_contact_priority():
    from pitchfinder.pipeline import _resolve_contact
    # new priority: LinkedIn > X(twitter) > Email > about
    # LinkedIn wins even when twitter + verified email present
    c = {"deep_dive": {"contact": {"email": "v@a.com", "linkedin": "https://linkedin.com/in/h"}},
         "twitter": "https://x.com/h", "url": "https://x.sub.com", "platform": "substack"}
    assert _resolve_contact(c)[4] == "linkedin"
    assert _resolve_contact(c)[3] == "https://linkedin.com/in/h"
    # no linkedin → twitter beats email
    c2 = {"contact_email": "e@b.com", "twitter": "https://x.com/h",
          "platform": "substack", "url": "https://p.substack.com"}
    assert _resolve_contact(c2)[4] == "twitter"
    # only email
    c3 = {"contact_email": "e@b.com", "platform": "blog", "url": "https://b.com"}
    assert _resolve_contact(c3)[4] == "email"
    # nothing but url → about
    c4 = {"platform": "substack", "url": "https://p.substack.com"}
    assert _resolve_contact(c4) == ("", "", "", "https://p.substack.com/about", "about")


# ---------- HTML report rendering ----------

def test_render_html_branded_header_and_brief():
    from pitchfinder.pipeline import _render_html
    creators = [{
        "creator_id": 1, "name": "Jane Doe", "platform": "substack", "url": "https://j.substack.com",
        "top_score": 90, "influence_score": 70, "tier": "A", "top_items": [], "angles": [],
        "deep_dive": None, "contact_email": "", "twitter": "https://x.com/jane", "linkedin": "",
    }]
    meta = {"title": "BlockRun · ClawRouter", "one_liner": "One endpoint, pay-per-call.",
            "positioning": "Agent-native infra.", "themes": ["x402 payments"], "competitors": ["OpenRouter"]}
    out = _render_html("desc", creators, 12, meta)
    assert "BlockRun · ClawRouter" in out
    assert "search 12" not in out and "PitchFinder report" not in out   # no dead id strings
    assert "x402 payments" in out and "OpenRouter" in out               # chips
    assert 'class="stat-num' in out and "Fraunces" in out               # stats + editorial font


def test_render_html_no_meta_fallback():
    from pitchfinder.pipeline import _render_html
    creators = [{"creator_id": 1, "name": "X", "platform": "blog", "url": "", "top_score": 60,
                 "influence_score": 50, "tier": "B", "top_items": [], "angles": [], "deep_dive": None}]
    out = _render_html("Some launch description here.", creators, 7, None)  # meta=None
    assert "Some launch description here." in out      # falls back to raw description
    assert "Creator &amp; Press Outreach Brief" in out


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
