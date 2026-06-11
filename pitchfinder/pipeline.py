"""Orchestration: load seeds, refresh feeds, run searches, show/status."""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.table import Table

from pitchfinder.db import get_conn, init_schema

logger = logging.getLogger("pitchfinder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

console = Console()


# ---------- load ----------


def load_seeds(yaml_path: Path, db: str) -> int:
    init_schema(db)
    data = yaml.safe_load(Path(yaml_path).read_text())
    creators = data.get("creators", [])
    conn = get_conn(db)
    try:
        n = 0
        for c in creators:
            contact = c.get("contact") or {}
            conn.execute(
                """
                INSERT INTO creators
                  (name, platform, handle, url, feed_url, contact_email, contact_other,
                   topics, influence_score, notes, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual')
                ON CONFLICT(platform, handle) DO UPDATE SET
                  name=excluded.name,
                  url=excluded.url,
                  feed_url=excluded.feed_url,
                  contact_email=excluded.contact_email,
                  contact_other=excluded.contact_other,
                  topics=excluded.topics,
                  influence_score=excluded.influence_score,
                  notes=excluded.notes
                """,
                (
                    c["name"],
                    c["platform"],
                    c["handle"],
                    c.get("url"),
                    c.get("feed_url"),
                    contact.get("email"),
                    contact.get("other"),
                    json.dumps(c.get("topics", [])),
                    int(c.get("influence_score", 50)),
                    c.get("notes"),
                ),
            )
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()


# ---------- refresh ----------


def refresh_feeds(
    db: str,
    lookback_days: int = 90,
    platforms: Optional[list[str]] = None,
) -> None:
    from pitchfinder.fetcher import fetch_feed

    conn = get_conn(db)
    try:
        q = "SELECT id, name, platform, feed_url FROM creators WHERE feed_url IS NOT NULL"
        params: list = []
        if platforms:
            q += " AND platform IN (" + ",".join(["?"] * len(platforms)) + ")"
            params.extend(platforms)
        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]No creators with feed_url found.[/yellow]")
        return

    content_type_map = {
        "substack": "article",
        "beehiiv": "article",
        "blog": "article",
        "podcast": "episode",
        "youtube": "video",
    }

    table = Table(title=f"Refresh ({len(rows)} creators, lookback {lookback_days}d)")
    table.add_column("Creator")
    table.add_column("Platform")
    table.add_column("New", justify="right")
    table.add_column("Status")

    total_new = 0
    for row in rows:
        ct = content_type_map.get(row["platform"], "article")
        items = fetch_feed(row["feed_url"], ct, lookback_days)
        new_n = _insert_items(db, row["id"], ct, items)
        total_new += new_n
        status_str = "[green]ok[/green]" if items else "[red]empty/failed[/red]"
        table.add_row(row["name"], row["platform"], str(new_n), status_str)

    console.print(table)
    console.print(f"[green]Total new items:[/green] {total_new}")


def _insert_items(db: str, creator_id: int, content_type: str, items: list) -> int:
    if not items:
        return 0
    conn = get_conn(db)
    try:
        n = 0
        for it in items:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO items
                  (creator_id, content_type, title, url, published_at, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    creator_id,
                    content_type,
                    it.title,
                    it.url,
                    it.published_at.isoformat() if it.published_at else None,
                    it.summary,
                ),
            )
            if cur.rowcount > 0:
                n += 1
        conn.commit()
        return n
    finally:
        conn.close()


# ---------- search ----------


def run_search(
    db: str,
    description: str,
    min_score: int = 70,
    max_creators: int = 30,
    lookback_days: int = 90,
    concurrency: int = 8,
    output: Optional[Path] = None,
) -> int:
    from pitchfinder.llm import (
        extract_topics,
        score_relevance,
        generate_pitch_angles,
    )

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set. Put it in .env or export it.")

    # 1. Topic extraction
    console.print("[bold]1/4[/bold] Extracting topics from description...")
    topics_payload = extract_topics(description)

    # 2. Create search row
    conn = get_conn(db)
    try:
        cur = conn.execute(
            "INSERT INTO searches (description, extracted_topics) VALUES (?, ?)",
            (description, json.dumps(topics_payload)),
        )
        search_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    console.print(f"   search_id={search_id}")

    # 3. Pull candidate items within lookback window
    candidates = _candidate_items(db, lookback_days)
    console.print(f"[bold]2/4[/bold] Scoring {len(candidates)} items (concurrency={concurrency})...")
    if not candidates:
        console.print("[yellow]No items in DB. Did you run `pitchfinder refresh`?[/yellow]")
        return search_id

    # 4. Score in parallel; collect results to list, write serially
    scored: list[tuple[int, int, str]] = []  # (item_id, score, reason)

    def _score_one(item_row):
        ct_label = {"article": "article", "episode": "podcast episode", "video": "YouTube video"}.get(
            item_row["content_type"], "article"
        )
        try:
            result = score_relevance(
                description,
                {"title": item_row["title"], "summary": item_row["summary"] or ""},
                ct_label,
            )
            return (item_row["id"], int(result.get("score", 0)), result.get("reason", ""))
        except Exception as exc:
            logger.warning("score failed item=%s: %s", item_row["id"], exc)
            return (item_row["id"], 0, f"error: {exc}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_score_one, row) for row in candidates]
        for fut in as_completed(futures):
            scored.append(fut.result())

    conn = get_conn(db)
    try:
        for item_id, score, reason in scored:
            conn.execute(
                """
                INSERT OR REPLACE INTO relevance_scores (search_id, item_id, score, reason)
                VALUES (?, ?, ?, ?)
                """,
                (search_id, item_id, score, reason),
            )
        conn.commit()
    finally:
        conn.close()

    # 5. Rank creators
    creators_ranked = _rank_creators(db, search_id, min_score, max_creators)
    console.print(
        f"[bold]3/4[/bold] {len(creators_ranked)} creators meet min_score={min_score}"
    )

    # 6. Generate pitch angles for top N
    console.print(f"[bold]4/4[/bold] Generating pitch angles for {len(creators_ranked)} creators...")
    for entry in creators_ranked:
        try:
            angles = generate_pitch_angles(description, entry["name"], entry["top_items"])
        except Exception as exc:
            logger.warning("pitch angles failed creator=%s: %s", entry["creator_id"], exc)
            angles = []
        entry["angles"] = angles
        conn = get_conn(db)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO pitch_angles (search_id, creator_id, angles_json)
                VALUES (?, ?, ?)
                """,
                (search_id, entry["creator_id"], json.dumps(angles)),
            )
            conn.commit()
        finally:
            conn.close()

    # 7. Render
    _render_results(description, creators_ranked, search_id, output)
    return search_id


def _candidate_items(db: str, lookback_days: int) -> list:
    conn = get_conn(db)
    try:
        rows = conn.execute(
            f"""
            SELECT i.id, i.creator_id, i.content_type, i.title, i.summary, i.url, i.published_at
            FROM items i
            WHERE i.published_at IS NOT NULL
              AND i.published_at >= datetime('now', '-{int(lookback_days)} days')
            """
        ).fetchall()
        return list(rows)
    finally:
        conn.close()


def _rank_creators(db: str, search_id: int, min_score: int, max_creators: int) -> list[dict]:
    conn = get_conn(db)
    try:
        # Top per-creator score + their best items
        rows = conn.execute(
            """
            SELECT
              c.id AS creator_id,
              c.name,
              c.platform,
              c.url,
              c.contact_email,
              c.contact_other,
              c.twitter,
              c.linkedin,
              c.influence_score,
              MAX(rs.score) AS top_score
            FROM relevance_scores rs
            JOIN items i ON i.id = rs.item_id
            JOIN creators c ON c.id = i.creator_id
            WHERE rs.search_id = ?
              AND rs.score >= ?
            GROUP BY c.id
            ORDER BY top_score DESC, c.influence_score DESC
            LIMIT ?
            """,
            (search_id, min_score, max_creators),
        ).fetchall()

        out: list[dict] = []
        for r in rows:
            items = conn.execute(
                """
                SELECT i.id, i.title, i.url, i.published_at, i.summary, i.content_type,
                       rs.score, rs.reason
                FROM relevance_scores rs
                JOIN items i ON i.id = rs.item_id
                WHERE rs.search_id = ? AND i.creator_id = ? AND rs.score >= ?
                ORDER BY rs.score DESC, i.published_at DESC
                LIMIT 3
                """,
                (search_id, r["creator_id"], min_score),
            ).fetchall()
            out.append(
                {
                    "creator_id": r["creator_id"],
                    "name": r["name"],
                    "platform": r["platform"],
                    "url": r["url"],
                    "contact_email": r["contact_email"],
                    "contact_other": r["contact_other"],
                    "twitter": r["twitter"],
                    "linkedin": r["linkedin"],
                    "influence_score": r["influence_score"],
                    "top_score": r["top_score"],
                    "top_items": [dict(it) for it in items],
                }
            )
        return out
    finally:
        conn.close()


def _resolve_contact(c: dict) -> tuple[str, str, str, str, str]:
    """Best reachable contact for a creator across all sources.
    Priority: LinkedIn > X(twitter) > email (MiroThinker-verified > enriched) >
    about page. Returns (email, twitter, linkedin, best_contact, contact_type)."""
    dd = c.get("deep_dive") or {}
    ddc = dd.get("contact") or {}
    email = (ddc.get("email") or c.get("contact_email") or "").strip()
    tw = (c.get("twitter") or ddc.get("twitter") or "").strip()
    lk = (c.get("linkedin") or ddc.get("linkedin") or "").strip()
    url = (c.get("url") or "").rstrip("/")
    if lk:
        best, bt = lk, "linkedin"
    elif tw:
        best, bt = tw, "twitter"
    elif email:
        best, bt = email, "email"
    elif url:
        best, bt = (url + "/about" if c.get("platform") == "substack" else url), "about"
    else:
        best, bt = "", "none"
    return email, tw, lk, best, bt


def _render_results(
    description: str,
    creators: list[dict],
    search_id: int,
    output: Optional[Path],
    meta: Optional[dict] = None,
) -> None:
    if not creators:
        console.print("[yellow]No creators matched. Try lowering --min-score.[/yellow]")
        return

    table = Table(title=f"Top creators (search_id={search_id})")
    table.add_column("#")
    table.add_column("Creator")
    table.add_column("Platform")
    table.add_column("Score", justify="right")
    table.add_column("Influence", justify="right")
    table.add_column("Contact")
    for i, c in enumerate(creators, 1):
        contact = c.get("contact_email") or c.get("contact_other") or "—"
        table.add_row(
            str(i),
            c["name"],
            c["platform"],
            str(c["top_score"]),
            str(c["influence_score"]),
            contact,
        )
    console.print(table)

    for c in creators:
        console.print(f"\n[bold cyan]{c['name']}[/bold cyan] ({c['platform']}, top_score={c['top_score']})")
        for it in c["top_items"]:
            console.print(f"  • [{it['score']}] {it['title']}")
            console.print(f"      [dim]{it['url']}[/dim]")
            console.print(f"      [italic]{it['reason']}[/italic]")
        if c.get("angles"):
            console.print("  [bold]Pitch angles:[/bold]")
            for a in c["angles"]:
                console.print(f"    → {a.get('angle', '')}")
                ref = a.get("references_item")
                if ref:
                    console.print(f"      [dim](refs: {ref})[/dim]")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        if suffix in (".html", ".htm"):
            output.write_text(_render_html(description, creators, search_id, meta))
            label = "HTML report"
        elif suffix == ".csv":
            output.write_text(_render_csv(description, creators, search_id))
            label = "CSV export"
        else:
            output.write_text(_render_markdown(description, creators, search_id, meta))
            label = "Markdown report"
        console.print(f"\n[green]{label} written:[/green] {output.resolve()}")


def _render_markdown(description: str, creators: list[dict], search_id: int,
                     meta: Optional[dict] = None) -> str:
    title = (meta or {}).get("title")
    heading = f"{title} — Creator & Press Outreach Brief" if title else "Creator & Press Outreach Brief"
    lines: list[str] = []
    lines.append(f"# {heading}")
    lines.append("")
    lines.append(f"_Prepared by PitchFinder · {datetime.utcnow().isoformat(timespec='seconds')}Z_")
    lines.append("")
    lines.append("## Launch description")
    lines.append("")
    lines.append(f"> {description}")
    lines.append("")
    lines.append(f"## Ranked creators ({len(creators)})")
    lines.append("")
    for i, c in enumerate(creators, 1):
        contact = c.get("contact_email") or c.get("contact_other") or "_(no public contact)_"
        tier_prefix = f"[{c['tier']}] " if c.get("tier") else ""
        lines.append(f"### {i}. {tier_prefix}{c['name']} — score {c['top_score']}")
        lines.append("")
        lines.append(f"- **Platform**: {c['platform']}")
        lines.append(f"- **URL**: {c.get('url') or '—'}")
        lines.append(f"- **Influence**: {c['influence_score']}")
        lines.append(f"- **Contact**: {contact}")
        lines.append("")
        lines.append("**Recent relevant content:**")
        lines.append("")
        for it in c["top_items"]:
            lines.append(f"- **[{it['score']}]** [{it['title']}]({it['url']}) ({it['content_type']})")
            lines.append(f"  - _{it['reason']}_")
        lines.append("")
        if c.get("angles"):
            lines.append("**Pitch angles:**")
            lines.append("")
            for a in c["angles"]:
                lines.append(f"- {a.get('angle', '')}")
                ref = a.get("references_item")
                if ref:
                    lines.append(f"  - _refs: {ref}_")
            lines.append("")
    return "\n".join(lines)


# Flat, Google-Sheet-friendly columns (one row per creator). No nested JSON so
# the file pastes straight into a spreadsheet.
_CSV_COLUMNS = [
    "rank", "tier", "name", "platform", "url", "score", "influence",
    "best_contact", "contact_type", "email", "twitter", "linkedin",
    "verified_active", "outreach_status",
    "top_match_title", "top_match_url",
    "angle_1", "angle_2", "angle_3",
    "recent_themes", "tier_rationale",
]


def _render_csv(description: str, creators: list[dict], search_id: int) -> str:
    """Render creators as a flat CSV. Pulls contact/quotes from deep_dive when
    present, else falls back to the creator's stored contact fields."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for i, c in enumerate(creators, 1):
        dd = c.get("deep_dive") or {}
        angles = [a.get("angle", "") for a in (c.get("angles") or [])]
        top = (c.get("top_items") or [{}])[0]
        va = dd.get("verified_active")
        email, tw, lk, best, btype = _resolve_contact(c)
        writer.writerow({
            "rank": i,
            "tier": c.get("tier", ""),
            "name": c["name"],
            "platform": c["platform"],
            "url": c.get("url") or "",
            "score": c.get("top_score", ""),
            "influence": c.get("influence_score", ""),
            "best_contact": best,
            "contact_type": btype,
            "email": email,
            "twitter": tw,
            "linkedin": lk,
            "verified_active": "" if va is None else ("yes" if va else "no"),
            "outreach_status": c.get("outreach_status", ""),
            "top_match_title": top.get("title", ""),
            "top_match_url": top.get("url", ""),
            "angle_1": angles[0] if len(angles) > 0 else "",
            "angle_2": angles[1] if len(angles) > 1 else "",
            "angle_3": angles[2] if len(angles) > 2 else "",
            "recent_themes": "; ".join(dd.get("recent_themes") or []),
            "tier_rationale": c.get("tier_rationale", ""),
        })
    return buf.getvalue()


def _render_html(description: str, creators: list[dict], search_id: int,
                 meta: Optional[dict] = None) -> str:
    meta = meta or {}
    def esc(s: object) -> str:
        return html.escape(str(s if s is not None else ""), quote=True)

    def link(url: str, text: str | None = None) -> str:
        if not url or not (url.startswith("http://") or url.startswith("https://") or url.startswith("mailto:")):
            return esc(text or url or "")
        return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text or url)}</a>'

    generated = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    rtitle = meta.get("title") or "Creator & Press Outreach Brief"

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{esc(rtitle)} — Outreach Brief · PitchFinder</title>")
    parts.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
    parts.append('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    parts.append('<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,900&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">')
    parts.append("""<style>
:root {
  --paper:#faf8f3; --panel:#fffdf8; --ink:#23262b; --muted:#73706a;
  --rule:#e4ddcf; --accent:#1e3a5f; --accent-soft:#eef2f7;
  --tier-a:#2f6b46; --tier-a-bg:#eaf3ed; --tier-b:#9a6a1a; --tier-b-bg:#f6efe1;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --display:"Fraunces", Georgia, serif;
  --mono:"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
* { box-sizing: border-box; }
body {
  font-family: var(--serif);
  max-width: 780px; margin: 0 auto; padding: 0 1.4em 5em;
  line-height: 1.65; color: var(--ink); background: var(--paper);
  -webkit-font-smoothing: antialiased; font-size: 17px;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; text-underline-offset: 2px; }
.num, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* Letterhead */
.letterhead { padding: 3.2em 0 2em; border-bottom: 1px solid var(--rule);
  animation: rise .6s ease both; }
.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 1.1em;
  display: flex; align-items: center; gap: 0.7em; }
.eyebrow::before { content:""; width: 26px; height: 2px; background: var(--accent); display:inline-block; }
.report-title { font-family: var(--display); font-weight: 600; font-size: 3rem;
  line-height: 1.05; letter-spacing: -0.015em; margin: 0 0 0.35em; }
.report-sub { font-family: var(--display); font-weight: 400; font-style: italic;
  font-size: 1.35rem; color: var(--muted); margin: 0 0 1.1em; }
.report-meta { font-family: var(--mono); font-size: 0.8rem; color: var(--muted);
  letter-spacing: 0.04em; }

/* Stats */
.statbar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
  background: var(--rule); border: 1px solid var(--rule); margin: 2.4em 0 1em;
  animation: rise .6s .1s ease both; }
.stat { background: var(--panel); padding: 1.1em 1.2em; }
.stat-num { font-family: var(--display); font-weight: 600; font-size: 2.1rem;
  line-height: 1; color: var(--accent); }
.stat-num.a { color: var(--tier-a); } .stat-num.b { color: var(--tier-b); }
.stat-label { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); margin-top: 0.5em; }
.platdist { font-family: var(--mono); font-size: 0.76rem; color: var(--muted);
  margin: 0 0 2.6em; letter-spacing: 0.03em; }
.platdist b { color: var(--ink); font-weight: 600; }

/* Section headings */
h2.sec { font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--accent); margin: 3em 0 1.1em;
  padding-bottom: 0.6em; border-bottom: 1px solid var(--rule); }

/* The Brief */
.brief-lead { font-family: var(--display); font-size: 1.5rem; font-weight: 400;
  line-height: 1.4; margin: 0 0 0.9em; letter-spacing: -0.01em; }
.brief-body { font-size: 1.02rem; color: #41454c; margin: 0 0 1.3em; }
.chips { display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.5em 0 0.4em; }
.chip { font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.02em;
  padding: 0.3em 0.7em; background: var(--accent-soft); color: var(--accent);
  border-radius: 2px; }
.chip.comp { background: #f3eee6; color: #7a5a2a; }
.chips-label { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); margin: 1.1em 0 0.2em; }

/* Index table */
.index-wrap { overflow-x: auto; margin: 0 0 1em; border: 1px solid var(--rule); }
table.index { width: 100%; border-collapse: collapse; font-size: 0.9rem; min-width: 640px; }
table.index thead th { position: sticky; top: 0; background: var(--panel);
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); text-align: left;
  padding: 0.85em 0.9em; border-bottom: 1px solid var(--rule); font-weight: 600; }
table.index td { padding: 0.7em 0.9em; border-bottom: 1px solid #efe9dc; vertical-align: middle; }
table.index tr:last-child td { border-bottom: none; }
table.index td.r, table.index th.r { text-align: right; }
table.index .tier-dot { display:inline-block; width: 8px; height: 8px; border-radius: 50%; }
table.index .tier-dot.A { background: var(--tier-a); }
table.index .tier-dot.B { background: var(--tier-b); }
table.index .cname { font-weight: 600; }
table.index .plat { font-family: var(--mono); font-size: 0.74rem; color: var(--muted); }
table.index .sc { font-family: var(--mono); }
table.index .ct { font-size: 0.84rem; }

/* Creator cards */
.creator { background: var(--panel); border: 1px solid var(--rule);
  border-left: 3px solid var(--rule); padding: 1.5em 1.7em; margin: 1.1em 0;
  break-inside: avoid; }
.creator.A { border-left-color: var(--tier-a); }
.creator.B { border-left-color: var(--tier-b); }
.creator h3 { font-family: var(--display); font-weight: 600; margin: 0 0 0.5em;
  font-size: 1.45rem; letter-spacing: -0.01em; }
.creator h3 .rk { color: var(--muted); font-weight: 400; }
.badge { display: inline-block; padding: 0.12em 0.6em; margin-left: 0.5em;
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.06em;
  text-transform: uppercase; border-radius: 2px; background: var(--accent-soft);
  color: var(--accent); vertical-align: middle; }
.badge.platform { background: #f0ece3; color: #6b6357; }
.badge.tier-a { background: var(--tier-a-bg); color: var(--tier-a); }
.badge.tier-b { background: var(--tier-b-bg); color: var(--tier-b); }
.creator .meta-row { color: var(--muted); font-size: 0.86rem; margin-bottom: 0.9em;
  font-family: var(--mono); letter-spacing: 0.02em; }
.item { margin: 0.45em 0; padding: 0.6em 0.8em; background: var(--paper); border-radius: 2px; }
.item .title-line { font-weight: 600; }
.item .score-pill { display:inline-block; min-width: 2.4em; text-align:center;
  padding: 0.05em 0.45em; margin-right: 0.5em; background: var(--accent);
  color: #fff; border-radius: 2px; font-family: var(--mono); font-size: 0.78rem; }
.item .reason { color: var(--muted); font-size: 0.86rem; margin: 0.3em 0 0 2.9em; font-style: italic; }
.angles { margin-top: 1em; }
.angles-label, .why-label, .dd-label, .contact-block .label {
  font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 0.5em; }
.angles-label { color: var(--accent); }
.angle { padding: 0.6em 0.85em; background: #fbf6ec; border-left: 2px solid var(--tier-b);
  margin: 0.4em 0; }
.angle .ref { display:block; color: var(--muted); font-size: 0.8rem; margin-top: 0.35em; font-family: var(--mono); }
.why { background: var(--tier-a-bg); border-left: 2px solid var(--tier-a);
  padding: 0.7em 0.9em; margin: 0.7em 0 0.9em; }
.why-label { color: var(--tier-a); }
.dd { background: #f4f1ea; border: 1px solid var(--rule); border-left: 2px solid var(--accent);
  padding: 0.8em 1em; margin: 1em 0; }
.dd.dd-sonnet { border-left-color: #94a3b8; }
.dd-label { color: var(--accent); }
.dd.dd-sonnet .dd-label { color: #5a6573; }
.dd-row { margin: 0.45em 0; font-size: 0.95rem; }
.dd-row strong { color: var(--accent); }
.dd .quote { margin: 0.4em 0; padding: 0.5em 0.7em; background: var(--panel);
  border-left: 2px solid var(--accent); font-style: italic; color: #41454c; }
.dd .quote .src { display:block; font-style: normal; font-size: 0.8rem; color: var(--muted); margin-top: 0.3em; font-family: var(--mono); }
.contact-block { background: var(--accent-soft); border: 1px solid #dbe3ec;
  padding: 0.75em 1em; margin: 0.9em 0; }
.contact-block .label { color: var(--accent); }
.contact-block .row { margin: 0.25em 0; font-size: 0.92rem; }

footer { margin-top: 4em; padding-top: 1.5em; border-top: 1px solid var(--rule);
  font-family: var(--mono); font-size: 0.74rem; color: var(--muted);
  letter-spacing: 0.03em; line-height: 1.8; }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

@media (max-width: 640px) {
  .statbar { grid-template-columns: repeat(2, 1fr); }
  .report-title { font-size: 2.2rem; }
}
@media print {
  body { background: #fff; font-size: 11pt; max-width: none; }
  .letterhead, .statbar, .creator { animation: none; }
  .creator, .dd, .why, .angle, .contact-block, .item { box-shadow: none; }
  .creator { break-inside: avoid; border-left-width: 3px; }
  a { color: var(--ink); }
  .index-wrap { overflow: visible; }
}
</style>""")
    parts.append("</head><body>")

    # --- Letterhead ---
    parts.append('<div class="letterhead">')
    parts.append('<div class="eyebrow">PitchFinder · Creator &amp; Press Outreach</div>')
    parts.append(f'<h1 class="report-title">{esc(rtitle)}</h1>')
    parts.append('<div class="report-sub">Creator &amp; Press Outreach Brief</div>')
    parts.append(f'<div class="report-meta">Prepared {esc(generated)} · {len(creators)} ranked targets</div>')
    parts.append("</div>")

    # --- Stats ---
    tier_a = sum(1 for c in creators if c.get("tier") == "A")
    tier_b = sum(1 for c in creators if c.get("tier") == "B")
    with_contact = sum(1 for c in creators if _resolve_contact(c)[4] in ("email", "twitter", "linkedin"))
    parts.append('<div class="statbar">')
    for n, lbl, cls in [(len(creators), "Ranked targets", ""), (tier_a, "Tier A · recommend", "a"),
                        (tier_b, "Tier B · worth it", "b"), (with_contact, "With contact", "")]:
        parts.append(f'<div class="stat"><div class="stat-num {cls}">{n}</div><div class="stat-label">{esc(lbl)}</div></div>')
    parts.append("</div>")
    pcounts: dict = {}
    for c in creators:
        pcounts[c.get("platform", "")] = pcounts.get(c.get("platform", ""), 0) + 1
    dist = " · ".join(f"<b>{pcounts[p]}</b> {p}" for p in ("substack", "podcast", "blog", "youtube") if pcounts.get(p))
    if dist:
        parts.append(f'<div class="platdist">By platform — {dist}</div>')

    # --- The Brief ---
    parts.append('<h2 class="sec">The Brief</h2>')
    if meta.get("one_liner"):
        parts.append(f'<div class="brief-lead">{esc(meta["one_liner"])}</div>')
    if meta.get("positioning"):
        parts.append(f'<div class="brief-body">{esc(meta["positioning"])}</div>')
    if not meta.get("one_liner") and not meta.get("positioning"):
        parts.append(f'<div class="brief-body">{esc(description)}</div>')
    if meta.get("themes"):
        parts.append('<div class="chips-label">Themes</div><div class="chips">')
        parts.append("".join(f'<span class="chip">{esc(t)}</span>' for t in meta["themes"]))
        parts.append("</div>")
    if meta.get("competitors"):
        parts.append('<div class="chips-label">Competitors</div><div class="chips">')
        parts.append("".join(f'<span class="chip comp">{esc(t)}</span>' for t in meta["competitors"]))
        parts.append("</div>")

    # --- Index table ---
    parts.append(f'<h2 class="sec">Ranked Creators · {len(creators)}</h2>')
    parts.append('<div class="index-wrap"><table class="index"><thead><tr>')
    parts.append('<th class="r">#</th><th>Tier</th><th>Creator</th><th>Platform</th><th class="r">Score</th><th>Contact</th>')
    parts.append("</tr></thead><tbody>")
    for i, c in enumerate(creators, 1):
        em, tw, lk, best, bt = _resolve_contact(c)
        tier = c.get("tier", "")
        cdot = f'<span class="tier-dot {esc(tier)}"></span> {esc(tier)}' if tier else ""
        cname = link(c.get("url"), c["name"]) if c.get("url") else esc(c["name"])
        if bt in ("email", "email?"):
            ct = link("mailto:" + em, "✉ email")
        elif bt == "twitter":
            ct = link(tw, "𝕏 X")
        elif bt == "linkedin":
            ct = link(lk, "in LinkedIn")
        elif bt == "about":
            ct = link(best, "🔗 about")
        else:
            ct = "—"
        parts.append(
            f'<tr><td class="r sc">{i}</td>'
            f'<td>{cdot}</td>'
            f'<td class="cname">{cname}</td>'
            f'<td class="plat">{esc(c["platform"])}</td>'
            f'<td class="r sc">{c["top_score"]}</td>'
            f'<td class="ct">{ct}</td></tr>'
        )
    parts.append("</tbody></table></div>")

    parts.append('<h2 class="sec">Target Profiles</h2>')

    # Per-creator detail
    for i, c in enumerate(creators, 1):
        tier = c.get("tier", "")
        parts.append(f'<div class="creator {esc(tier)}">')
        tier_badge = (
            f'<span class="badge tier-{esc(tier).lower()}">Tier {esc(tier)}</span> '
            if tier else ""
        )
        parts.append(
            f'<h3><span class="rk">{i}.</span> {esc(c["name"])} '
            f'{tier_badge}'
            f'<span class="badge platform">{esc(c["platform"])}</span> '
            f'<span class="badge">score {c["top_score"]}</span>'
            f'</h3>'
        )

        meta_bits: list[str] = []
        if c.get("url"):
            meta_bits.append(f"Channel: {link(c['url'])}")
        meta_bits.append(f"Influence: {c['influence_score']}")
        parts.append(f'<div class="meta-row">{" · ".join(meta_bits)}</div>')

        # WHY WE PICKED THEM — most important block, surfaced first
        parts.append('<div class="why">')
        parts.append('<div class="why-label">Why we picked them</div>')
        if c.get("top_items"):
            top = c["top_items"][0]
            parts.append(
                f'<div>Highest-relevance match (score {top["score"]}): '
                f'{link(top.get("url"), top.get("title") or "(untitled)")}</div>'
            )
            if top.get("reason"):
                parts.append(f'<div style="margin-top:0.4em; color:#065f46;">→ {esc(top["reason"])}</div>')
        else:
            parts.append("<div>(no matching items)</div>")
        parts.append("</div>")

        # All matching items list
        if c.get("top_items"):
            parts.append("<div><strong>Their recent work that matches the launch:</strong></div>")
            for it in c["top_items"]:
                title = it.get("title") or "(untitled)"
                ct = it.get("content_type", "")
                pub = (it.get("published_at") or "")[:10]
                parts.append('<div class="item">')
                parts.append(
                    f'<div class="title-line">'
                    f'<span class="score-pill">{it["score"]}</span>'
                    f'{link(it.get("url"), title)} '
                    f'<span style="color:#6b7280; font-size:0.85em;">({esc(ct)}{", " + esc(pub) if pub else ""})</span>'
                    f"</div>"
                )
                if it.get("reason"):
                    parts.append(f'<div class="reason">{esc(it["reason"])}</div>')
                parts.append("</div>")

        # DEEP-DIVE enrichment (only if present)
        dd = c.get("deep_dive")
        if dd and not dd.get("error"):
            source_model = dd.get("_source_model") or c.get("deep_dive_model") or ""
            if source_model.startswith("mirothinker"):
                dd_class = "dd"
                dd_label = "✓ Verified context (MiroThinker • live web search)"
            else:
                dd_class = "dd dd-sonnet"
                dd_label = f"○ Context ({esc(source_model or 'LLM')} • from training data, no live search)"
            parts.append(f'<div class="{dd_class}">')
            parts.append(f'<div class="dd-label">{dd_label}</div>')
            if dd.get("verified_active") is False:
                parts.append('<div class="dd-row"><strong>⚠ Status:</strong> No posts found in last 6 months — may be inactive.</div>')
            if dd.get("recent_themes"):
                themes = ", ".join(esc(t) for t in dd["recent_themes"])
                parts.append(f'<div class="dd-row"><strong>Recent themes:</strong> {themes}</div>')
            if dd.get("current_stance"):
                parts.append(f'<div class="dd-row"><strong>Current stance:</strong> {esc(dd["current_stance"])}</div>')
            if dd.get("sharp_quotes"):
                parts.append('<div class="dd-row"><strong>Sharp quotes (verified):</strong></div>')
                for q in dd["sharp_quotes"]:
                    src_link = link(q.get("source"), q.get("source"))
                    date_str = f" — {esc(q['date'])}" if q.get("date") else ""
                    parts.append(
                        f'<div class="quote">"{esc(q.get("quote", ""))}"'
                        f'<span class="src">{src_link}{date_str}</span></div>'
                    )
            if dd.get("pitch_hook"):
                parts.append(f'<div class="dd-row"><strong>Pitch hook:</strong> <em>"{esc(dd["pitch_hook"])}"</em></div>')
            parts.append("</div>")
        elif dd and dd.get("error"):
            parts.append(f'<div class="dd-row" style="color:#b91c1c;">⚠ Deep dive failed: {esc(dd["error"])}</div>')

        # CONTACT block — unified via _resolve_contact (enriched + deep-dive)
        c_email, c_tw, c_lk, c_best, c_btype = _resolve_contact(c)
        dd_contact = (dd or {}).get("contact", {}) if dd else {}
        if c_best:
            parts.append('<div class="contact-block">')
            parts.append('<div class="label">How to reach them</div>')
            # 按优先级展示：LinkedIn > X(twitter) > Email
            if c_lk:
                parts.append(f'<div class="row">💼 LinkedIn: {link(c_lk)}</div>')
            if c_tw:
                h = c_tw.rstrip("/").split("/")[-1].lstrip("@")
                parts.append(f'<div class="row">🐦 Twitter/X: {link(c_tw, "@" + h)}</div>')
            if c_email:
                parts.append(f'<div class="row">📧 Email: {link("mailto:" + c_email, c_email)}</div>')
            if dd_contact.get("contact_form"):
                parts.append(f'<div class="row">📨 Contact form: {link(dd_contact["contact_form"])}</div>')
            if dd_contact.get("preferred_channel"):
                parts.append(f'<div class="row"><strong>Best channel:</strong> {esc(dd_contact["preferred_channel"])}</div>')
            if c_btype == "about" and not (c_email or c_tw or c_lk):
                parts.append(f'<div class="row">🔗 About / subscribe page: {link(c_best)}</div>')
            if dd_contact.get("notes"):
                parts.append(f'<div class="row" style="color:#92400e;">{esc(dd_contact["notes"])}</div>')
            parts.append("</div>")
        else:
            parts.append('<div class="contact-block" style="background:#fef2f2; border-color:#fecaca; color:#991b1b;">No public contact found.</div>')

        if c.get("angles"):
            parts.append('<div class="angles">')
            parts.append('<div class="angles-label">Pitch angles</div>')
            for a in c["angles"]:
                parts.append('<div class="angle">')
                parts.append(esc(a.get("angle", "")))
                ref = a.get("references_item")
                if ref:
                    parts.append(f'<span class="ref">refs: {esc(ref)}</span>')
                parts.append("</div>")
            parts.append("</div>")

        parts.append("</div>")

    parts.append("<footer>")
    parts.append("Prepared by PitchFinder · Creator &amp; Press Outreach Brief<br>")
    parts.append("Method — open-web discovery (Brave · Querit) → relevance scoring → A/B tiering "
                 "(neutral third parties only) → contact resolution → deep verification of Tier-A leads.")
    parts.append("</footer>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------- show ----------


def _build_report(db: str, search_id: int, min_score: int) -> Optional[tuple[str, list[dict], dict]]:
    """Load ranked creators for a search with angles + deep-dive + tier attached
    and tier-ordered (drops removed). Returns (description, creators, meta) or None.
    meta = {title, one_liner, positioning, themes, competitors} when a brand ran it."""
    conn = get_conn(db)
    try:
        row = conn.execute(
            "SELECT description, brand, brand_meta FROM searches WHERE id = ?", (search_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    description = row["description"]
    meta: dict = {}
    if row["brand_meta"]:
        try:
            meta = json.loads(row["brand_meta"]) or {}
        except Exception:
            meta = {}
    if not meta.get("title") and row["brand"]:
        meta["title"] = str(row["brand"]).replace("-", " ").replace("_", " ").title()

    creators = _rank_creators(db, search_id, min_score, max_creators=999)
    conn = get_conn(db)
    try:
        for c in creators:
            ar = conn.execute(
                "SELECT angles_json FROM pitch_angles WHERE search_id = ? AND creator_id = ?",
                (search_id, c["creator_id"]),
            ).fetchone()
            c["angles"] = json.loads(ar["angles_json"]) if ar and ar["angles_json"] else []

            dr = conn.execute(
                "SELECT payload_json, model FROM deep_dives WHERE search_id = ? AND creator_id = ?",
                (search_id, c["creator_id"]),
            ).fetchone()
            if dr and dr["payload_json"]:
                payload = json.loads(dr["payload_json"])
                if not payload.get("_source_model"):
                    payload["_source_model"] = dr["model"] or ""
                c["deep_dive"] = payload
            else:
                c["deep_dive"] = None

            # Latest outreach status (any campaign) so the CSV is trackable.
            orow = conn.execute(
                "SELECT status FROM outreach WHERE creator_id = ? ORDER BY id DESC LIMIT 1",
                (c["creator_id"],),
            ).fetchone()
            c["outreach_status"] = orow["status"] if orow else ""
    finally:
        conn.close()

    creators = _attach_and_order_tiers(db, search_id, creators)
    return description, creators, meta


def show_search(db: str, search_id: int, min_score: int, output: Optional[Path]) -> None:
    built = _build_report(db, search_id, min_score)
    if built is None:
        console.print(f"[red]Search {search_id} not found.[/red]")
        return
    description, creators, meta = built
    _render_results(description, creators, search_id, output, meta)


# ---------- tiering (A / B / drop) ----------

_TIER_RANK = {"A": 0, "B": 1, "": 2}


def _attach_and_order_tiers(db: str, search_id: int, creators: list[dict]) -> list[dict]:
    """Attach tier + rationale from creator_tiers. If any tiers exist for this
    search, drop 'drop' creators and order A before B (stable within tier,
    preserving score order). If none exist yet, return creators unchanged."""
    conn = get_conn(db)
    try:
        rows = {
            r["creator_id"]: (r["tier"], r["rationale"])
            for r in conn.execute(
                "SELECT creator_id, tier, rationale FROM creator_tiers WHERE search_id = ?",
                (search_id,),
            ).fetchall()
        }
    finally:
        conn.close()
    if not rows:
        return creators
    kept: list[dict] = []
    for c in creators:
        tier, rationale = rows.get(c["creator_id"], ("B", ""))
        if tier == "drop":
            continue
        c["tier"] = tier
        c["tier_rationale"] = rationale or ""
        kept.append(c)
    kept.sort(key=lambda c: _TIER_RANK.get(c.get("tier", ""), 2))
    return kept


def run_classify_tiers(
    db: str,
    search_id: int,
    brand_summary: str,
    min_score: int = 40,
    model: str | None = None,
    competitors: list[str] | None = None,
) -> dict[str, int]:
    """Auto-classify all ranked creators for a search into A/B/drop and persist
    to creator_tiers (source='auto', won't overwrite a 'manual' override).
    Returns a {tier: count} summary."""
    from pitchfinder.llm import classify_tiers

    creators = _rank_creators(db, search_id, min_score, max_creators=999)
    payload = [
        {
            "creator_id": c["creator_id"],
            "name": c["name"],
            "platform": c["platform"],
            "url": c.get("url"),
            "influence_score": c["influence_score"],
            "signal": " | ".join(it.get("title", "") for it in c.get("top_items", [])[:3]),
        }
        for c in creators
    ]
    verdicts = classify_tiers(brand_summary, payload, model=model, competitors=competitors)

    conn = get_conn(db)
    counts = {"A": 0, "B": 0, "drop": 0}
    try:
        manual = {
            r["creator_id"]
            for r in conn.execute(
                "SELECT creator_id FROM creator_tiers WHERE search_id = ? AND source = 'manual'",
                (search_id,),
            ).fetchall()
        }
        for cid, v in verdicts.items():
            if cid in manual:  # never clobber a human override
                continue
            conn.execute(
                """INSERT INTO creator_tiers (search_id, creator_id, tier, rationale, source)
                   VALUES (?, ?, ?, ?, 'auto')
                   ON CONFLICT(search_id, creator_id)
                   DO UPDATE SET tier=excluded.tier, rationale=excluded.rationale,
                                 source='auto', set_at=CURRENT_TIMESTAMP""",
                (search_id, cid, v["tier"], v["rationale"]),
            )
            counts[v["tier"]] = counts.get(v["tier"], 0) + 1
        conn.commit()
    finally:
        conn.close()
    return counts


def set_tier(db: str, search_id: int, creator_id: int, tier: str, note: str | None = None) -> None:
    """Manual tier override (source='manual'); survives future auto-classify."""
    conn = get_conn(db)
    try:
        conn.execute(
            """INSERT INTO creator_tiers (search_id, creator_id, tier, rationale, source)
               VALUES (?, ?, ?, ?, 'manual')
               ON CONFLICT(search_id, creator_id)
               DO UPDATE SET tier=excluded.tier, rationale=excluded.rationale,
                             source='manual', set_at=CURRENT_TIMESTAMP""",
            (search_id, creator_id, tier, note or ""),
        )
        conn.commit()
    finally:
        conn.close()


def run_enrich_contacts(db: str, search_id: int, use_brave: bool = True) -> dict:
    """Resolve a reachable contact for EVERY A/B creator (not just deep-dived
    top-N) and persist to the creators table (contact_email / twitter / linkedin).
    Cheap: RSS / about-page / _preloads + throttled Brave fallback."""
    from concurrent.futures import ThreadPoolExecutor
    from pitchfinder.contacts import enrich_creator

    conn = get_conn(db)
    try:
        rows = [dict(r) for r in conn.execute(
            """SELECT c.id, c.name, c.platform, c.url, c.feed_url
               FROM creator_tiers ct JOIN creators c ON c.id=ct.creator_id
               WHERE ct.search_id=? AND ct.tier IN ('A','B')""", (search_id,)).fetchall()]
    finally:
        conn.close()
    if not rows:
        return {"total": 0, "real": 0}

    def work(r):
        return r["id"], enrich_creator(r, use_brave=use_brave)

    results = list(ThreadPoolExecutor(max_workers=6).map(work, rows))
    conn = get_conn(db)
    try:
        for cid, res in results:
            # enrich is the single source of truth for these columns → overwrite
            # (deep-dive verified contact is separate and wins via _resolve_contact)
            conn.execute(
                "UPDATE creators SET contact_email = ?, twitter = ?, linkedin = ? WHERE id = ?",
                (res["email"], res["twitter"], res["linkedin"], cid),
            )
        conn.commit()
    finally:
        conn.close()
    real = sum(1 for _, res in results if res["contact_type"] in ("email", "twitter", "linkedin"))
    return {"total": len(results), "real": real}


# ---------- deep-dive (MiroThinker enrichment) ----------


def run_deep_dive(
    db: str,
    search_id: int,
    top_n: int = 10,
    min_score: int = 55,
    output: Optional[Path] = None,
    only_creator_ids: Optional[list[int]] = None,
    skip_creator_ids: Optional[list[int]] = None,
    model: Optional[str] = None,
) -> None:
    """Run a deep-research pass on top-N creators of a given search.

    Uses MiroThinker (web search + verification) by default; pass `model`
    e.g. 'anthropic/claude-sonnet-4.6' to do a cheaper best-effort run
    without live search.
    """
    is_mirothinker = (model or os.getenv("MIROMIND_DEEPRESEARCH_MODEL", "mirothinker-1-7-deepresearch")).startswith("mirothinker")
    if is_mirothinker and not os.getenv("MIROMIND_API_KEY"):
        raise RuntimeError("MIROMIND_API_KEY is not set. Put it in .env or export it.")
    if not is_mirothinker and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (needed for non-MiroThinker models).")

    from pitchfinder.research import deep_dive_creator, deep_research_model

    conn = get_conn(db)
    try:
        srow = conn.execute(
            "SELECT description FROM searches WHERE id = ?", (search_id,)
        ).fetchone()
    finally:
        conn.close()
    if not srow:
        console.print(f"[red]Search {search_id} not found.[/red]")
        return
    description = srow["description"]

    # When --only is given, the user is naming creators explicitly; don't
    # also clip them to the top_n window (would silently drop lower-ranked
    # IDs even though the user asked for them by id).
    effective_top = 9999 if only_creator_ids else top_n
    candidates = _rank_creators(db, search_id, min_score, max_creators=effective_top)
    if only_creator_ids:
        candidates = [c for c in candidates if c["creator_id"] in only_creator_ids]
    if skip_creator_ids:
        candidates = [c for c in candidates if c["creator_id"] not in skip_creator_ids]
    if not candidates:
        console.print("[yellow]No creators to deep-dive.[/yellow]")
        return

    used_model = model or deep_research_model()
    console.print(f"  using model: [cyan]{used_model}[/cyan]")

    console.print(
        f"[bold]Deep-dive[/bold] on {len(candidates)} creators (search_id={search_id})..."
    )
    for i, c in enumerate(candidates, 1):
        t0 = datetime.utcnow()
        console.print(f"  [{i}/{len(candidates)}] {c['name']} — researching...")
        payload = deep_dive_creator(c["name"], c.get("url") or "", description, model=model)
        elapsed = (datetime.utcnow() - t0).total_seconds()
        ok = not payload.get("error") and payload.get("recent_themes")
        status = "[green]ok[/green]" if ok else "[red]error/empty[/red]"
        if payload.get("error"):
            console.print(f"      {status} {elapsed:.1f}s — {payload['error']}")
        else:
            n_quotes = len(payload.get("sharp_quotes") or [])
            n_themes = len(payload.get("recent_themes") or [])
            console.print(f"      {status} {elapsed:.1f}s — {n_themes} themes, {n_quotes} quotes")

        conn = get_conn(db)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO deep_dives
                  (search_id, creator_id, model, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    search_id,
                    c["creator_id"],
                    used_model,
                    json.dumps(payload),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Optional: backfill creators.contact_email / contact_other when found
        c_contact = (payload.get("contact") or {})
        if c_contact.get("email") and "@" in c_contact["email"]:
            conn = get_conn(db)
            try:
                conn.execute(
                    "UPDATE creators SET contact_email = COALESCE(contact_email, ?) WHERE id = ?",
                    (c_contact["email"], c["creator_id"]),
                )
                conn.commit()
            finally:
                conn.close()
        elif c_contact.get("twitter") or c_contact.get("linkedin") or c_contact.get("contact_form"):
            other_bits = [
                c_contact.get("twitter"),
                c_contact.get("linkedin"),
                c_contact.get("contact_form"),
            ]
            other = " | ".join([x for x in other_bits if x])
            conn = get_conn(db)
            try:
                conn.execute(
                    "UPDATE creators SET contact_other = COALESCE(contact_other, ?) WHERE id = ?",
                    (other, c["creator_id"]),
                )
                conn.commit()
            finally:
                conn.close()

    console.print(f"\n[green]Deep-dive complete.[/green] Run `pitchfinder show {search_id} --output reports/...html` to see the enriched report.")


# ---------- outreach ----------


def set_outreach_status(
    db: str, creator_id: int, campaign: str, new_status: str, notes: Optional[str]
) -> None:
    allowed = {"not_contacted", "pitched", "replied", "confirmed", "declined", "published"}
    if new_status not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}, got {new_status!r}")

    timestamp_col = None
    if new_status == "pitched":
        timestamp_col = "pitched_at"
    elif new_status == "replied":
        timestamp_col = "replied_at"

    conn = get_conn(db)
    try:
        # Ensure row exists, then update
        conn.execute(
            """
            INSERT INTO outreach (creator_id, campaign, status, notes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(creator_id, campaign) DO UPDATE SET
              status=excluded.status,
              notes=COALESCE(excluded.notes, outreach.notes)
            """,
            (creator_id, campaign, new_status, notes),
        )
        if timestamp_col:
            conn.execute(
                f"UPDATE outreach SET {timestamp_col} = CURRENT_TIMESTAMP "
                "WHERE creator_id = ? AND campaign = ?",
                (creator_id, campaign),
            )
        conn.commit()
    finally:
        conn.close()


# ---------- campaign orchestrator (the general-agent funnel) ----------


def _ensure_angles(db: str, search_id: int, description: str, creators: list[dict]) -> int:
    """Generate + persist pitch angles for any creator that lacks them. Returns
    the number generated. Keeps angle generation scoped to the kept (A+B) set."""
    from pitchfinder.llm import generate_pitch_angles

    n = 0
    for c in creators:
        if c.get("angles"):
            continue
        try:
            angles = generate_pitch_angles(description, c["name"], c.get("top_items", []))
        except Exception as exc:
            logger.warning("ensure_angles failed creator=%s: %s", c["creator_id"], exc)
            angles = []
        c["angles"] = angles
        conn = get_conn(db)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO pitch_angles (search_id, creator_id, angles_json) VALUES (?, ?, ?)",
                (search_id, c["creator_id"], json.dumps(angles)),
            )
            conn.commit()
        finally:
            conn.close()
        n += 1
    return n


def run_campaign(
    db: str,
    config_path: str,
    budget: Optional[int] = None,
    skip_discovery: bool = False,
    lookback_days: int = 90,
    search_min_score: int = 50,
    tier_min_score: int = 40,
) -> int:
    """End-to-end funnel for one brand. Returns the search_id.

    discover-web (+ MiroThinker fallback) → refresh → search → auto-tier →
    backfill angles for A+B → deep-dive Tier-A top-N (budget-capped) →
    render html + csv + md to reports/<brand>-<UTCdate>.{html,csv,md}.
    """
    from pitchfinder.config import load_brand_config
    from pitchfinder.discovery import discover_relevant_creators, write_candidates_yaml

    cfg = load_brand_config(config_path)
    init_schema(db)
    desc = cfg.launch_description()
    cap = budget if budget is not None else cfg.budget.max_deepdive
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discovery — cheap web sweep, MiroThinker only as fallback.
    if not skip_discovery:
        from pitchfinder.web_discovery import discover_web_creators

        conn = get_conn(db)
        try:
            existing = {r["name"] for r in conn.execute("SELECT name FROM creators").fetchall()}
        finally:
            conn.close()
        console.print("[bold]Discovery[/bold] — web sweep (Brave/Querit) ...")
        cands = discover_web_creators(
            cfg.themes, cfg.platforms, existing,
            cfg.discovery.providers, cfg.discovery.per_platform_queries,
        )
        if cands:
            p = reports_dir / f"_campaign_{cfg.brand}_web.yaml"
            write_candidates_yaml(cands, p)
            load_seeds(p, db)
        console.print(f"   web candidates: {len(cands)}")
        if len(cands) < cfg.discovery.mirothinker_fallback_min and os.getenv("MIROMIND_API_KEY"):
            console.print("   web recall low → MiroThinker fallback discovery ...")
            try:
                mt = discover_relevant_creators(
                    desc, list(existing), limit=30, model=cfg.discovery.mirothinker_model
                )
                if mt:
                    p2 = reports_dir / f"_campaign_{cfg.brand}_mirothinker.yaml"
                    write_candidates_yaml(mt, p2)
                    load_seeds(p2, db)
                console.print(f"   MiroThinker candidates: {len(mt)}")
            except Exception as exc:
                console.print(f"   [yellow]MiroThinker fallback skipped: {exc}[/yellow]")

    # 2. Refresh feeds.
    console.print("[bold]Refresh[/bold] — pulling feeds ...")
    refresh_feeds(db, lookback_days=lookback_days)

    # 3. Search (scores items, writes search row, angles for top-by-score).
    search_id = run_search(
        db, description=desc, min_score=search_min_score,
        max_creators=40, lookback_days=lookback_days, output=None,
    )
    conn = get_conn(db)
    try:
        conn.execute(
            "UPDATE searches SET brand = ?, brand_meta = ? WHERE id = ?",
            (cfg.brand, json.dumps(cfg.report_meta()), search_id),
        )
        conn.commit()
    finally:
        conn.close()

    # 4. Auto-tier A/B/drop (strong model from config).
    summary = desc + ((" Do not fabricate: " + "; ".join(cfg.do_not) + ".") if cfg.do_not else "")
    console.print("[bold]Tiering[/bold] — A/B/drop ...")
    counts = run_classify_tiers(db, search_id, summary, min_score=tier_min_score,
                                model=cfg.tiering.model, competitors=cfg.competitors)
    console.print(f"   A={counts.get('A',0)} B={counts.get('B',0)} drop={counts.get('drop',0)}")

    # 4b. Enrich contacts for every A/B creator (cheap; covers all, not just top-N).
    console.print("[bold]Contacts[/bold] — resolving a reachable channel per A/B creator ...")
    cc = run_enrich_contacts(db, search_id, use_brave=True)
    console.print(f"   real contact (email/twitter/linkedin): {cc['real']}/{cc['total']}")

    # 5. Backfill pitch angles for the kept (A+B) set.
    built = _build_report(db, search_id, tier_min_score)
    description, creators, _meta = built if built else (desc, [], {})
    made = _ensure_angles(db, search_id, description, creators)
    if made:
        console.print(f"   backfilled {made} pitch-angle set(s)")

    # 6. Deep-dive Tier-A top-N (budget-capped).
    tier_a = [c for c in creators if c.get("tier") == cfg.deepdive.tier]
    tier_a.sort(key=lambda c: c.get("top_score", 0), reverse=True)
    dd_ids = [c["creator_id"] for c in tier_a[: min(cfg.deepdive.top_n, cap)]]
    if dd_ids and os.getenv("MIROMIND_API_KEY"):
        console.print(f"[bold]Deep-dive[/bold] — MiroThinker on {len(dd_ids)} Tier-{cfg.deepdive.tier} creators ...")
        run_deep_dive(
            db, search_id, min_score=tier_min_score,
            only_creator_ids=dd_ids, model=cfg.deepdive.model,
        )
    elif dd_ids:
        console.print("[yellow]MIROMIND_API_KEY not set — skipping deep-dive.[/yellow]")

    # 7. Render all three formats (rebuild to pick up deep-dive + angles).
    built = _build_report(db, search_id, tier_min_score)
    if built:
        description, creators, meta = built
        stem = reports_dir / f"{cfg.brand}-{datetime.utcnow().strftime('%Y%m%d')}"
        for ext in (".html", ".csv", ".md"):
            _render_results(description, creators, search_id, stem.with_suffix(ext), meta)
    console.print(f"[green]Campaign done[/green] (search_id={search_id})")
    return search_id
