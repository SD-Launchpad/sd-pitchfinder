"""Contact enrichment — give every creator a reachable channel, cheaply.

Priority per creator: email (MX-valid) > Twitter/X > LinkedIn > about page.

PRECISION OVER RECALL — a wrong contact is worse than none. Contacts come ONLY
from trusted / structured places, never a blind scan of the page body (that pulls
in embedded third-party / ad / footer links):
  - podcast : RSS itunes:owner/email
  - substack: window._preloads JSON → pub.twitter_screen_name / support_email /
              author_bio (author-authored)
  - blog    : /about /contact → mailto, kept only if the email domain matches the
              site (or a Brave-found personal site)
  - fallback: Brave "<name> contact" → personal site → same-domain mailto

Email check: format + MX (dnspython, socket fallback) + role/denylist; no SMTP probe.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import time
from urllib.parse import urlparse

import feedparser
import httpx

from pitchfinder.discovery import BROWSER_UA
from pitchfinder.web_discovery import _brave_search, _querit_search

try:
    import dns.resolver
    _HAVE_DNS = True
except ImportError:
    _HAVE_DNS = False

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
MAILTO_RE = re.compile(r"mailto:([^\"'>?\s]+)", re.I)
TW_RE = re.compile(r"(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})", re.I)
LK_RE = re.compile(r"linkedin\.com/(?:in|company)/([A-Za-z0-9\-_%.]+)", re.I)
PRELOADS_RE = re.compile(r'window\._preloads\s*=\s*JSON\.parse\((".*?")\)\s*</script>', re.S)

_JUNK = ("example.", "sentry.", "wixpress.", "schema.org", "w3.org", "godaddy",
         "domain.com", "email.com", "yourdomain", ".png", ".jpg", ".gif", ".svg",
         ".webp", "@2x", "yoursite", "company.com", "cloudflare", "substackcdn",
         "reachinbox")
_BAD_LOCAL = {"billing", "noreply", "no-reply", "donotreply", "do-not-reply",
              "abuse", "postmaster", "mailer-daemon", "root", "wordpress",
              "sentry", "notifications", "notification", "unsubscribe", "bounce"}
_ROLE = {"info", "hello", "contact", "team", "press", "hi", "podcast", "production",
         "support", "admin", "editor", "media", "hey", "newsletter", "inquiries",
         "partnerships", "podcasts", "show", "pr", "sponsors", "mail"}
_TW_SKIP = {"intent", "share", "home", "substack", "hashtag", "search", "i",
            "privacy", "tos", "settings", "compose", "messages", "explore", "login"}
_SOCIAL_HOSTS = ("twitter.com", "x.com", "linkedin.com", "facebook.com", "instagram.com",
                 "youtube.com", "youtu.be", "tiktok.com", "apple.com", "spotify.com",
                 "substack.com", "podcasts.apple.com", "amazon.", "google.", "medium.com",
                 "github.com", "reddit.com", "t.co", "bit.ly", "patreon.com")

_mx_cache: dict[str, bool] = {}
_brave_lock = threading.Lock()
_brave_last = [0.0]


def _brave_throttled(q: str, n: int = 5) -> list[dict]:
    with _brave_lock:                       # Brave free tier ~1 req/s
        dt = time.monotonic() - _brave_last[0]
        if dt < 1.2:
            time.sleep(1.2 - dt)
        _brave_last[0] = time.monotonic()
    return _brave_search(q, n) or _querit_search(q, n)


def _reg(host: str) -> str:
    parts = (host or "").lower().lstrip("www.").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else (host or "")


def _same_org(email: str, *urls: str) -> bool:
    edom = _reg(email.split("@")[-1])
    if edom in ("substack.com", "gmail.com", "googlemail.com"):
        return False
    return any(u and _reg(urlparse(u).hostname or "") == edom for u in urls)


def _domain_ok(domain: str) -> bool:
    if domain in _mx_cache:
        return _mx_cache[domain]
    ok = False
    if _HAVE_DNS:
        try:
            ok = len(dns.resolver.resolve(domain, "MX", lifetime=5)) > 0
        except Exception:
            ok = False
    if not ok:
        try:
            socket.gethostbyname(domain)
            ok = True
        except Exception:
            ok = False
    _mx_cache[domain] = ok
    return ok


def _clean_emails(cands) -> list[str]:
    out = []
    for raw in cands:
        e = str(raw).lower().strip().strip(".")
        if e.count("@") != 1 or any(j in e for j in _JUNK) or len(e) > 70:
            continue
        if e.split("@")[0] in _BAD_LOCAL:
            continue
        out.append(e)
    return list(dict.fromkeys(out))


def _tw(text: str) -> list[str]:
    return list(dict.fromkeys(h for h in TW_RE.findall(text or "") if h.lower() not in _TW_SKIP))


def _lk(text: str) -> list[str]:
    return list(dict.fromkeys(h for h in LK_RE.findall(text or "")
                              if h.lower() not in ("sharing", "sharearticle", "company")))


def _fetch(url: str) -> str:
    try:
        r = httpx.get(url, headers={"User-Agent": BROWSER_UA}, follow_redirects=True, timeout=12)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ""


def parse_substack(html: str) -> dict:
    """Author socials from the trusted _preloads.pub object (not page body)."""
    m = PRELOADS_RE.search(html)
    if not m:
        return {}
    try:
        pub = (json.loads(json.loads(m.group(1))) or {}).get("pub") or {}
    except Exception:
        return {}
    bio = pub.get("author_bio") or ""
    emails = [e for e in (pub.get("support_email"), pub.get("email_from")) if e]
    emails += EMAIL_RE.findall(bio)
    tw = ([pub.get("twitter_screen_name")] if pub.get("twitter_screen_name") else []) + _tw(bio)
    return {
        "emails": _clean_emails(emails),
        "twitter": [h for h in dict.fromkeys(tw) if h and h.lower() not in _TW_SKIP],
        "linkedin": _lk(bio),
    }


def _name_matches_host(name: str, host: str) -> bool:
    """A Brave-found 'personal site' is only trusted if its domain shares a word
    with the creator name — guards against grabbing an unrelated site's email."""
    htoks = set(re.findall(r"[a-z]{3,}", _reg(host)))
    ntoks = set(re.findall(r"[a-z]{3,}", (name or "").lower()))
    return bool(htoks & ntoks)


def _personal_site(results: list[dict], name: str) -> str:
    for r in results:
        host = (urlparse(r.get("url", "")).hostname or "").lower()
        if host and not any(s in host for s in _SOCIAL_HOSTS) and _name_matches_host(name, host):
            return f"https://{host}"
    return ""


def _pick_social(results: list[dict], name: str) -> tuple[str, str]:
    """从 Brave 结果里抽 LinkedIn /in/<slug> 与 X handle，名字校验后返回 (li_slug, tw_handle)。

    纯函数(不触网，可单测)。名字校验防抓错人：slug/handle 必须与 creator 名字有词重叠
    (token ≥3)。creator 名常是节目/机构名，词不重叠就丢 —— 宁缺勿编。
    """
    # 名字校验防抓错人：handle/slug 连写 与 creator 名连写**一方完整包含另一方**(≥4 字符)。
    # 比单词 substring 严得多：泛名 creator("AI Music Unmuted")不会误匹配 @MusicStarAI
    # (连写 "aimusicunmuted" 与 "musicstarai" 互不包含)。宁缺勿编 > 覆盖率。
    ncomp = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    if len(ncomp) < 4:
        return "", ""

    def _belongs(cand: str) -> bool:
        hc = re.sub(r"[^a-z0-9]", "", cand.lower())
        return len(hc) >= 4 and (hc in ncomp or ncomp in hc)

    li, tw = "", ""
    for r in results:
        url = r.get("url", "") or ""
        if not li:
            m = re.search(r"linkedin\.com/in/([A-Za-z0-9\-_%.]+)", url, re.I)
            if m:
                slug = m.group(1).rstrip("/")
                if _belongs(slug):
                    li = slug
        if not tw:
            m = re.search(r"(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})", url, re.I)
            if m:
                h = m.group(1)
                if h.lower() not in _TW_SKIP and _belongs(h):
                    tw = h
        if li and tw:
            break
    return li, tw


def _social_from_brave(name: str) -> tuple[str, str]:
    """Brave 搜 creator 的 LinkedIn / X，名字校验后返回 (li_slug, tw_handle)。缺 key 返回空。"""
    results: list[dict] = []
    for q in (f"{name} linkedin", f"{name} twitter"):
        results += _brave_throttled(q, 5)
    return _pick_social(results, name)


def enrich_creator(creator: dict, use_brave: bool = True) -> dict:
    """Resolve a reachable contact for one creator.

    Input keys: name, platform, url, feed_url. Returns:
    {email, email_status, is_general, twitter, linkedin, best_contact, contact_type, source}.
    """
    name = " ".join(str(creator.get("name", "")).split())
    plat = creator.get("platform", "")
    base = (creator.get("url") or "").rstrip("/")
    feed = creator.get("feed_url")
    emails: list[str] = []
    tw: list[str] = []
    lk: list[str] = []
    src: list[str] = []

    if plat == "podcast" and feed:
        try:
            f = feedparser.parse(feed).feed
            for key in ("itunes_owner", "publisher_detail", "author_detail"):
                v = f.get(key)
                if isinstance(v, dict) and v.get("email"):
                    emails.append(v["email"])
                    src.append("rss")
        except Exception:
            pass

    if plat == "substack" and base:
        info = parse_substack(_fetch(base + "/about") or _fetch(base))
        if info:
            emails += info["emails"]
            tw += info["twitter"]
            lk += info["linkedin"]
            src.append("substack:_preloads")

    if plat == "blog" and base:
        for p in (base + "/about", base, base + "/contact"):
            html = _fetch(p)
            if not html:
                continue
            for e in _clean_emails(MAILTO_RE.findall(html)):
                if _same_org(e, base) or e.split("@")[0] in _ROLE:
                    emails.append(e)
            if emails:
                src.append(p.replace(base, "") or "/")
                break

    if use_brave and not _clean_emails(emails):
        site = _personal_site(_brave_throttled(f"{name} contact", 5), name)
        if site:
            html = _fetch(site + "/about") or _fetch(site) or _fetch(site + "/contact")
            if html:
                for e in _clean_emails(MAILTO_RE.findall(html)):
                    if _same_org(e, site):
                        emails.append(e)
                if emails:
                    src.append("brave:" + (urlparse(site).hostname or ""))

    # Brave 搜 LinkedIn/X 补 social（名字校验防抓错人；creator 比记者难匹配但仍提升覆盖）
    if use_brave and (not lk or not tw):
        s_lk, s_tw = _social_from_brave(name)
        if s_lk and not lk:
            lk.append(s_lk)
            src.append("brave:linkedin")
        if s_tw and not tw:
            tw.append(s_tw)
            src.append("brave:twitter")

    emails = _clean_emails(emails)
    scored = []
    for e in emails:
        mx = _domain_ok(e.split("@")[-1])
        role = e.split("@")[0] in _ROLE
        scored.append((0 if (mx and not role) else 1 if mx else 2, e, mx, role))
    scored.sort()
    best_e = scored[0] if scored else None
    tw = [h for h in dict.fromkeys(tw) if h]
    lk = list(dict.fromkeys(lk))
    twitter = f"https://x.com/{tw[0]}" if tw else ""
    linkedin = f"https://linkedin.com/in/{lk[0]}" if lk else ""

    # 优先级：LinkedIn > X(twitter) > Email(mx_ok) > Email?(no mx) > about
    if linkedin:
        best, btype = linkedin, "linkedin"
    elif twitter:
        best, btype = twitter, "twitter"
    elif best_e and best_e[2]:
        best, btype = best_e[1], "email"
    elif best_e:
        best, btype = best_e[1], "email?"
    elif base:
        best, btype = (base + "/about" if plat == "substack" else base), "about"
    else:
        best, btype = "", "none"

    return {
        "email": best_e[1] if best_e else "",
        "email_status": ("mx_ok" if best_e and best_e[2] else "no_mx" if best_e else "none"),
        "is_general": ("yes" if best_e and best_e[3] else ("no" if best_e else "")),
        "twitter": twitter, "linkedin": linkedin,
        "best_contact": best, "contact_type": btype,
        "source": ";".join(dict.fromkeys(src)),
    }
