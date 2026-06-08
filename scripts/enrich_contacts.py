"""One-shot contact enrichment for a search's A/B creators.

Gives EVERY kept creator at least one reachable channel, cheaply, from the
source itself. Priority: email (MX-valid) > Twitter/X > LinkedIn > about page.

PRECISION OVER RECALL — a wrong contact is worse than none. So we only take
contacts from TRUSTED / STRUCTURED places, never a blind scan of the whole page
(that pulls in embedded third-party / ad / footer links):
  - podcast : RSS itunes:owner/email
  - substack: window._preloads JSON → pub.twitter_screen_name / support_email /
              author_bio (author-authored), NOT the rendered page body
  - blog    : /about /contact → mailto only, kept only if the email's domain
              matches the site (or a Brave-found personal site)
  - fallback: Brave "<name> contact" → personal site → same-domain mailto

Email check: format + MX record (dnspython, socket fallback) + role/denylist.
No per-address SMTP probe. Non-destructive (does not touch the DB).
Usage: .venv/bin/python scripts/enrich_contacts.py <search_id> [out.csv]
"""
from __future__ import annotations

import csv
import json
import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import feedparser
import httpx

from pitchfinder.db import get_conn
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
    with _brave_lock:
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
        return False  # generic hosts can't prove ownership
    for u in urls:
        if u and _reg(urlparse(u).hostname or "") == edom:
            return True
    return False


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


def _clean_emails(cands: list[str]) -> list[str]:
    out = []
    for raw in cands:
        e = raw.lower().strip().strip(".")
        if e.count("@") != 1 or any(j in e for j in _JUNK) or len(e) > 70:
            continue
        if e.split("@")[0] in _BAD_LOCAL:
            continue
        out.append(e)
    return list(dict.fromkeys(out))


def _tw_handles(text: str) -> list[str]:
    return list(dict.fromkeys(h for h in TW_RE.findall(text or "") if h.lower() not in _TW_SKIP))


def _lk_handles(text: str) -> list[str]:
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


def _parse_substack(html: str) -> dict:
    """Pull author socials from the trusted _preloads.pub object (NOT page body)."""
    m = PRELOADS_RE.search(html)
    if not m:
        return {}
    try:
        pub = (json.loads(json.loads(m.group(1))) or {}).get("pub") or {}
    except Exception:
        return {}
    bio = pub.get("author_bio") or ""
    emails = [e for e in (pub.get("support_email"), pub.get("email_from")) if e]
    emails += EMAIL_RE.findall(bio)            # author-written bio is trustworthy
    tw = ([pub.get("twitter_screen_name")] if pub.get("twitter_screen_name") else []) + _tw_handles(bio)
    return {
        "author": pub.get("author_name") or "",
        "emails": _clean_emails(emails),
        "twitter": [h for h in dict.fromkeys(tw) if h and h.lower() not in _TW_SKIP],
        "linkedin": _lk_handles(bio),
    }


def _personal_site(results: list[dict]) -> str:
    for r in results:
        host = (urlparse(r.get("url", "")).hostname or "").lower()
        if host and not any(s in host for s in _SOCIAL_HOSTS):
            return f"https://{host}"
    return ""


def enrich(row: tuple) -> dict:
    cid, name, plat, tier, url, feed = row
    name = " ".join(str(name).split())
    emails: list[str] = []
    tw: list[str] = []
    lk: list[str] = []
    src: list[str] = []
    base = (url or "").rstrip("/")

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
        html = _fetch(base + "/about") or _fetch(base)
        info = _parse_substack(html)
        if info:
            emails += info["emails"]            # bio/official only — trusted
            tw += info["twitter"]
            lk += info["linkedin"]
            src.append("substack:_preloads")

    if plat == "blog" and base:
        for p in (base + "/about", base, base + "/contact"):
            html = _fetch(p)
            if not html:
                continue
            # mailto only (more reliable than body text), kept only if same-domain
            for e in _clean_emails(MAILTO_RE.findall(html)):
                if _same_org(e, base) or e.split("@")[0] in _ROLE:
                    emails.append(e)
            if emails:
                src.append(p.replace(base, "") or "/")
                break

    # Brave fallback when no email yet: find the author's personal site, take a
    # same-domain mailto from it (ownership-proven).
    if not _clean_emails(emails):
        site = _personal_site(_brave_throttled(f"{name} contact", 5))
        if site:
            html = _fetch(site + "/about") or _fetch(site) or _fetch(site + "/contact")
            if html:
                for e in _clean_emails(MAILTO_RE.findall(html)):
                    if _same_org(e, site):
                        emails.append(e)
                tw += _tw_handles(html)[:0]      # don't trust blind page socials
                if emails:
                    src.append("brave:" + (urlparse(site).hostname or ""))

    emails = _clean_emails(emails)
    scored = []
    for e in emails:
        dom = e.split("@")[-1]
        mx = _domain_ok(dom)
        role = e.split("@")[0] in _ROLE
        scored.append((0 if (mx and not role) else 1 if mx else 2, e, mx, role))
    scored.sort()
    best_e = scored[0] if scored else None
    tw = [h for h in dict.fromkeys(tw) if h]
    lk = list(dict.fromkeys(lk))
    twitter = f"https://x.com/{tw[0]}" if tw else ""
    linkedin = f"https://linkedin.com/in/{lk[0]}" if lk else ""

    if best_e and best_e[2]:
        best, btype = best_e[1], "email"
    elif twitter:
        best, btype = twitter, "twitter"
    elif linkedin:
        best, btype = linkedin, "linkedin"
    elif best_e:
        best, btype = best_e[1], "email?"
    elif base:
        best, btype = (base + "/about" if plat == "substack" else base), "about"
    else:
        best, btype = "", "none"

    return {
        "creator_id": cid, "name": name, "platform": plat, "tier": tier,
        "best_contact": best, "contact_type": btype,
        "email": best_e[1] if best_e else "",
        "email_status": ("mx_ok" if best_e and best_e[2] else "no_mx" if best_e else "none"),
        "is_general": ("yes" if best_e and best_e[3] else ("no" if best_e else "")),
        "twitter": twitter, "linkedin": linkedin,
        "all_emails": "; ".join(e for _, e, _, _ in scored[:4]),
        "source": ";".join(dict.fromkeys(src)),
    }


def main() -> None:
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"reports/search{sid}-contacts.csv"
    conn = get_conn("pitchfinder.db")
    rows = [
        (r["id"], r["name"], r["platform"], r["tier"], r["url"], r["feed_url"])
        for r in conn.execute(
            """SELECT c.id, c.name, c.platform, ct.tier, c.url, c.feed_url
               FROM creator_tiers ct JOIN creators c ON c.id=ct.creator_id
               WHERE ct.search_id=? AND ct.tier IN ('A','B')
               ORDER BY ct.tier, c.name""", (sid,),
        ).fetchall()
    ]
    conn.close()
    print(f"enriching {len(rows)} A/B creators (search {sid}); dnspython={_HAVE_DNS}")

    results = list(ThreadPoolExecutor(max_workers=6).map(enrich, rows))

    cols = ["tier", "name", "platform", "best_contact", "contact_type", "email",
            "email_status", "is_general", "twitter", "linkedin", "all_emails",
            "source", "creator_id"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["tier"], x["name"])):
            w.writerow(r)

    n = len(results)
    real = sum(1 for r in results if r["contact_type"] in ("email", "twitter", "linkedin"))
    print(f"  REAL contact (email/twitter/linkedin): {real}/{n}  | about-only: {n-real}")
    print(f"  email {sum(1 for r in results if r['email'])} | "
          f"twitter {sum(1 for r in results if r['twitter'])} | "
          f"linkedin {sum(1 for r in results if r['linkedin'])}")
    for plat in ("podcast", "substack", "blog", "youtube"):
        sub = [r for r in results if r["platform"] == plat]
        if sub:
            rc = sum(1 for r in sub if r["contact_type"] in ("email", "twitter", "linkedin"))
            print(f"    {plat:9} real {rc}/{len(sub)}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
