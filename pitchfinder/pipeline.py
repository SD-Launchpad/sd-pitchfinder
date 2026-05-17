"""Orchestration: load seeds, refresh feeds, run searches, show/status."""

from __future__ import annotations

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
                    "influence_score": r["influence_score"],
                    "top_score": r["top_score"],
                    "top_items": [dict(it) for it in items],
                }
            )
        return out
    finally:
        conn.close()


def _render_results(
    description: str,
    creators: list[dict],
    search_id: int,
    output: Optional[Path],
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
        output.write_text(_render_markdown(description, creators, search_id))
        console.print(f"\n[green]Markdown report written:[/green] {output}")


def _render_markdown(description: str, creators: list[dict], search_id: int) -> str:
    lines: list[str] = []
    lines.append(f"# PitchFinder report — search {search_id}")
    lines.append("")
    lines.append(f"_Generated {datetime.utcnow().isoformat(timespec='seconds')}Z_")
    lines.append("")
    lines.append("## Launch description")
    lines.append("")
    lines.append(f"> {description}")
    lines.append("")
    lines.append(f"## Ranked creators ({len(creators)})")
    lines.append("")
    for i, c in enumerate(creators, 1):
        contact = c.get("contact_email") or c.get("contact_other") or "_(no public contact)_"
        lines.append(f"### {i}. {c['name']} — score {c['top_score']}")
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


# ---------- show ----------


def show_search(db: str, search_id: int, min_score: int, output: Optional[Path]) -> None:
    conn = get_conn(db)
    try:
        row = conn.execute("SELECT description FROM searches WHERE id = ?", (search_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        console.print(f"[red]Search {search_id} not found.[/red]")
        return
    description = row["description"]

    creators = _rank_creators(db, search_id, min_score, max_creators=999)
    # Attach previously generated angles
    conn = get_conn(db)
    try:
        for c in creators:
            ar = conn.execute(
                "SELECT angles_json FROM pitch_angles WHERE search_id = ? AND creator_id = ?",
                (search_id, c["creator_id"]),
            ).fetchone()
            c["angles"] = json.loads(ar["angles_json"]) if ar and ar["angles_json"] else []
    finally:
        conn.close()

    _render_results(description, creators, search_id, output)


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
