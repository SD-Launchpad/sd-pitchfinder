from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(
    add_completion=False,
    help="PitchFinder — find AI/tech creators worth pitching for product launches.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_DB = "pitchfinder.db"


@app.command()
def init(db: str = typer.Option(DEFAULT_DB, help="SQLite DB path")) -> None:
    """Create the SQLite schema."""
    from pitchfinder.db import init_schema

    init_schema(db)
    console.print(f"[green]Initialized[/green] {db}")


@app.command()
def load(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Upsert seed creators from a YAML file."""
    from pitchfinder.pipeline import load_seeds

    n = load_seeds(yaml_path, db)
    console.print(f"[green]Loaded[/green] {n} creators from {yaml_path}")


@app.command()
def refresh(
    lookback_days: int = typer.Option(90, "--lookback-days"),
    platforms: Optional[str] = typer.Option(
        None, "--platforms", help="Comma-separated subset, e.g. substack,podcast,youtube"
    ),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Fetch RSS/Atom feeds for all creators and insert new items."""
    from pitchfinder.pipeline import refresh_feeds

    filt = [p.strip() for p in platforms.split(",")] if platforms else None
    refresh_feeds(db, lookback_days=lookback_days, platforms=filt)


@app.command()
def search(
    description: str = typer.Argument(..., help="Product launch description"),
    min_score: int = typer.Option(70, "--min-score"),
    max_creators: int = typer.Option(30, "--max-creators"),
    lookback_days: int = typer.Option(90, "--lookback-days"),
    concurrency: int = typer.Option(8, "--concurrency"),
    output: Optional[Path] = typer.Option(None, "--output", help="Markdown report path"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Run the full creator-discovery pipeline against a launch description."""
    from pitchfinder.pipeline import run_search

    run_search(
        db=db,
        description=description,
        min_score=min_score,
        max_creators=max_creators,
        lookback_days=lookback_days,
        concurrency=concurrency,
        output=output,
    )


@app.command()
def show(
    search_id: int = typer.Argument(...),
    min_score: int = typer.Option(70, "--min-score"),
    output: Optional[Path] = typer.Option(None, "--output"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Re-display a previous search."""
    from pitchfinder.pipeline import show_search

    show_search(db=db, search_id=search_id, min_score=min_score, output=output)


@app.command()
def campaign(
    config: Path = typer.Argument(..., exists=True, readable=True, help="brand.yaml"),
    budget: Optional[int] = typer.Option(None, "--budget", help="Hard cap on MiroThinker deep-dives"),
    skip_discovery: bool = typer.Option(False, "--skip-discovery", help="Reuse existing library"),
    lookback_days: int = typer.Option(90, "--lookback-days"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Run the whole funnel for one brand: discover → refresh → search → tier →
    deep-dive Tier-A top-N → HTML + CSV + Markdown. Parameterized by brand.yaml."""
    from pitchfinder.pipeline import run_campaign

    run_campaign(db, str(config), budget=budget, skip_discovery=skip_discovery,
                 lookback_days=lookback_days)


@app.command()
def classify(
    search_id: int = typer.Argument(..., help="A previous search id"),
    brand: Optional[Path] = typer.Option(
        None, "--brand", help="brand.yaml for richer context (themes/competitors/do-not)"
    ),
    min_score: int = typer.Option(40, "--min-score"),
    model: Optional[str] = typer.Option(None, "--model", help="Override classifier model"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Auto-classify ranked creators into A / B / drop (persisted to creator_tiers).

    A = strongly recommend, B = worth reaching out, drop = content farm / off-topic.
    Manual overrides (`pitchfinder tier`) are never clobbered. `show` then renders
    only A+B, A first.
    """
    from pitchfinder.db import get_conn
    from pitchfinder.pipeline import run_classify_tiers

    if brand:
        from pitchfinder.config import load_brand_config

        cfg = load_brand_config(brand)
        summary = cfg.launch_description()
        if cfg.do_not:
            summary += " Do not fabricate: " + "; ".join(cfg.do_not) + "."
        if model is None:
            model = cfg.tiering.model  # judgment task → strong model from config
    else:
        conn = get_conn(db)
        try:
            row = conn.execute("SELECT description FROM searches WHERE id = ?", (search_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            console.print(f"[red]Search {search_id} not found.[/red]")
            raise typer.Exit(1)
        summary = row["description"]

    counts = run_classify_tiers(db, search_id, summary, min_score=min_score, model=model)
    console.print(f"[green]Tiered:[/green] A={counts.get('A', 0)} B={counts.get('B', 0)} drop={counts.get('drop', 0)}")


@app.command()
def tier(
    search_id: int = typer.Argument(...),
    creator_id: int = typer.Argument(...),
    new_tier: str = typer.Argument(..., help="A | B | drop"),
    note: Optional[str] = typer.Option(None, "--note"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Manually override a creator's tier (survives future auto-classify)."""
    from pitchfinder.pipeline import set_tier

    nt = new_tier.strip()
    if nt not in ("A", "B", "drop"):
        console.print("[red]tier must be A, B, or drop[/red]")
        raise typer.Exit(1)
    set_tier(db, search_id, creator_id, nt, note)
    console.print(f"[green]Set[/green] creator={creator_id} -> tier {nt}")


@app.command()
def status(
    creator_id: int = typer.Argument(...),
    campaign: str = typer.Argument(...),
    new_status: str = typer.Argument(...),
    notes: Optional[str] = typer.Option(None, "--notes"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Update outreach status for a creator on a given campaign."""
    from pitchfinder.pipeline import set_outreach_status

    set_outreach_status(db, creator_id, campaign, new_status, notes)
    console.print(f"[green]Updated[/green] creator={creator_id} campaign={campaign} -> {new_status}")


@app.command("deep-dive")
def deep_dive(
    search_id: int = typer.Argument(..., help="A previous search id"),
    top: int = typer.Option(10, "--top", help="Number of top-ranked creators to enrich"),
    min_score: int = typer.Option(55, "--min-score"),
    only: Optional[str] = typer.Option(
        None, "--only", help="Comma-separated creator_ids to restrict to (overrides --top)"
    ),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated creator_ids to exclude (e.g. already deep-dived)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override deep-research model. Default = MIROMIND_DEEPRESEARCH_MODEL "
        "(mirothinker-1-7-deepresearch). Use e.g. 'anthropic/claude-sonnet-4.6' for cheaper best-effort.",
    ),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Enrich top-N ranked creators with web-search + verification.

    Default: MiroThinker (live web search). Pass --model anthropic/claude-sonnet-4.6
    to do a cheaper best-effort pass without live search.
    """
    from pitchfinder.pipeline import run_deep_dive

    only_ids = [int(x) for x in only.split(",")] if only else None
    skip_ids = [int(x) for x in skip.split(",")] if skip else None
    run_deep_dive(
        db=db,
        search_id=search_id,
        top_n=top,
        min_score=min_score,
        only_creator_ids=only_ids,
        skip_creator_ids=skip_ids,
        model=model,
    )


@app.command()
def lint(
    concurrency: int = typer.Option(6, "--concurrency"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Verify every creator url actually points to their channel.

    Catches HTTP-200 squatters (e.g. GoDaddy parking pages), dead sites,
    and pages that load but contain unrelated content. Uses the relevance
    LLM (Haiku-class) — ~$0.01 for the whole seed.
    """
    from rich.table import Table
    from pitchfinder.db import get_conn
    from pitchfinder.lint import lint_creators

    conn = get_conn(db)
    try:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, name, url FROM creators WHERE url IS NOT NULL"
            ).fetchall()
        ]
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]No creators with a url.[/yellow]")
        return

    console.print(f"Linting {len(rows)} URLs (concurrency={concurrency})...")
    results = lint_creators(rows, concurrency=concurrency)

    table = Table(title="URL lint results")
    table.add_column("id", justify="right")
    table.add_column("creator")
    table.add_column("status")
    table.add_column("http", justify="right")
    table.add_column("reason")

    counts = {"ok": 0, "squatter": 0, "wrong_content": 0, "uncertain": 0, "unreachable": 0}
    for r in results:
        counts[r.content_status] = counts.get(r.content_status, 0) + 1
        color = {
            "ok": "green",
            "squatter": "red",
            "wrong_content": "red",
            "uncertain": "yellow",
            "unreachable": "red",
        }.get(r.content_status, "white")
        table.add_row(
            str(r.creator_id),
            r.name,
            f"[{color}]{r.content_status}[/{color}]",
            str(r.http_status or "—"),
            r.reason[:100],
        )
    console.print(table)

    summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    console.print(f"\n[bold]Summary:[/bold] {summary}")
    bad = counts.get("squatter", 0) + counts.get("wrong_content", 0) + counts.get("unreachable", 0)
    if bad:
        console.print(
            f"[red]{bad} URL(s) need attention.[/red] Edit seed_creators.yaml "
            "and run `pitchfinder load seed_creators.yaml` to fix."
        )


@app.command("discover-podcasts")
def discover_podcasts(
    keyword: str = typer.Argument(...),
    limit: int = typer.Option(10),
) -> None:
    """Search Apple Podcasts iTunes API for podcasts matching keyword."""
    from pitchfinder.discovery import search_podcasts, print_podcast_candidates

    candidates = search_podcasts(keyword, limit=limit)
    print_podcast_candidates(candidates)


@app.command("discover-creators")
def discover_creators(
    description: str = typer.Argument(..., help="Launch description / topic to find creators for"),
    limit: int = typer.Option(30, "--limit"),
    output: Optional[Path] = typer.Option(
        Path("reports/discovered_candidates.yaml"),
        "--output",
        help="YAML file to write candidate stanzas to",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override deep-research model (default: MIROMIND_DEEPRESEARCH_MODEL)"
    ),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Use MiroThinker (web search) to expand the seed library.

    Finds active AI/tech creators relevant to a launch topic, skipping
    those already in the DB. Outputs YAML stanzas you can review and
    paste into seed_creators.yaml before running `pitchfinder load`.
    """
    from pitchfinder.db import get_conn
    from pitchfinder.discovery import (
        discover_relevant_creators,
        print_discovered_candidates,
        write_candidates_yaml,
    )

    conn = get_conn(db)
    try:
        rows = conn.execute("SELECT name FROM creators").fetchall()
        existing = [r["name"] for r in rows]
    finally:
        conn.close()
    console.print(f"Found {len(existing)} existing creators; asking MiroThinker for {limit} new ones (this takes 5-15 min)...")

    candidates = discover_relevant_creators(
        topic_description=description,
        existing_names=existing,
        limit=limit,
        model=model,
    )

    print_discovered_candidates(candidates)

    if candidates and output:
        write_candidates_yaml(candidates, output)
        console.print(f"\n[green]Wrote candidate stanzas to[/green] {output.resolve()}")
        console.print(
            "Review them, then `cat reports/discovered_candidates.yaml >> seed_creators.yaml` "
            "(or merge selectively) and run `pitchfinder lint` + `pitchfinder load`."
        )


@app.command("discover-web")
def discover_web(
    themes: Optional[str] = typer.Argument(
        None, help="Comma-separated themes (or use --brand to pull them from a config)"
    ),
    brand: Optional[Path] = typer.Option(None, "--brand", help="brand.yaml (themes + platforms)"),
    platforms: str = typer.Option("substack,blog,youtube,podcast", "--platforms"),
    providers: str = typer.Option("brave,querit", "--providers"),
    per_platform: int = typer.Option(4, "--per-platform", help="themes used per platform"),
    output: Path = typer.Option(Path("reports/discovered_web.yaml"), "--output"),
    db: str = typer.Option(DEFAULT_DB),
) -> None:
    """Broadly discover creators across the open web (Brave primary, Querit supplement).

    Resolves feeds and writes seed-loadable candidates. Env-gated: a provider with
    no key is skipped. Review the YAML, then `load` + `lint` + `refresh`.
    """
    from pitchfinder.db import get_conn
    from pitchfinder.discovery import write_candidates_yaml
    from pitchfinder.web_discovery import discover_web_creators, providers_available

    if brand:
        from pitchfinder.config import load_brand_config

        cfg = load_brand_config(brand)
        theme_list = cfg.themes
        plat_list = cfg.platforms
        prov_list = cfg.discovery.providers
        per_platform = cfg.discovery.per_platform_queries
    else:
        if not themes:
            console.print("[red]Provide themes (positional) or --brand.[/red]")
            raise typer.Exit(1)
        theme_list = [t.strip() for t in themes.split(",") if t.strip()]
        plat_list = [p.strip() for p in platforms.split(",") if p.strip()]
        prov_list = [p.strip() for p in providers.split(",") if p.strip()]

    avail = providers_available()
    console.print(f"Providers with keys: {avail or '(none — only iTunes podcasts will run)'}")

    conn = get_conn(db)
    try:
        existing = {r["name"] for r in conn.execute("SELECT name FROM creators").fetchall()}
    finally:
        conn.close()

    cands = discover_web_creators(
        themes=theme_list, platforms=plat_list, existing_names=existing,
        providers=prov_list, per_platform_queries=per_platform,
    )
    if not cands:
        console.print("[yellow]No new candidates found.[/yellow]")
        return
    write_candidates_yaml(cands, output)
    resolved = sum(1 for c in cands if c.get("feed_url"))
    console.print(
        f"[green]Wrote {len(cands)} candidates[/green] ({resolved} with resolved feed) to {output}\n"
        f"Next: review, then `pitchfinder load {output}` + `lint` + `refresh`."
    )


@app.command("discover-substack")
def discover_substack(
    substack_url: str = typer.Argument(..., help="A Substack URL whose /recommendations to scrape"),
) -> None:
    """Best-effort scrape of a Substack's recommendations page."""
    from pitchfinder.discovery import (
        discover_substack_recommendations,
        print_substack_candidates,
    )

    candidates = discover_substack_recommendations(substack_url)
    print_substack_candidates(candidates)


if __name__ == "__main__":
    app()
