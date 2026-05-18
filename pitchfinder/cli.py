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


@app.command("discover-podcasts")
def discover_podcasts(
    keyword: str = typer.Argument(...),
    limit: int = typer.Option(10),
) -> None:
    """Search Apple Podcasts iTunes API for podcasts matching keyword."""
    from pitchfinder.discovery import search_podcasts, print_podcast_candidates

    candidates = search_podcasts(keyword, limit=limit)
    print_podcast_candidates(candidates)


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
