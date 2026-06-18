"""Brand / campaign config — makes PitchFinder reusable across products.

A campaign is described by a small YAML file (see brands/apodex.yaml). The
`campaign` command reads it and drives the whole funnel. Everything has a
sensible default so a minimal config (just brand + positioning + themes) works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DiscoveryCfg:
    providers: list[str] = field(default_factory=lambda: ["brave", "querit"])
    per_platform_queries: int = 4
    # Trigger the (pricier) Apodex discovery only when cheap web discovery
    # returns fewer than this many candidates.
    apodex_fallback_min: int = 60
    apodex_model: str = "apodex-1-0-deepresearch-mini"


@dataclass
class TieringCfg:
    # A/B/drop is a JUDGMENT task — use a strong model (cheap models over-rate
    # content farms). Scoring (the high-volume filter) stays cheap separately.
    model: str = "anthropic/claude-sonnet-4.6"
    drop_content_farms: bool = True


@dataclass
class DeepDiveCfg:
    tier: str = "A"               # only deep-dive this tier
    top_n: int = 5                # …and only its top N
    model: str = "apodex-1-0-deepresearch"


@dataclass
class BudgetCfg:
    max_deepdive: int = 8         # hard cap on Apodex deep-dive calls / run
    pitch_b_top_n: int = 60       # pitch angles: all Tier-A + top-N Tier-B by score (cost cap)


@dataclass
class BrandConfig:
    brand: str
    display_name: str = ""        # optional pretty title for the report header
    one_liner: str = ""
    positioning: str = ""
    themes: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    do_not: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=lambda: ["substack", "blog", "podcast", "youtube"])
    discovery: DiscoveryCfg = field(default_factory=DiscoveryCfg)
    tiering: TieringCfg = field(default_factory=TieringCfg)
    deepdive: DeepDiveCfg = field(default_factory=DeepDiveCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)

    def report_meta(self) -> dict:
        """Structured metadata for the HTML report header / brief."""
        if self.display_name:
            title = self.display_name
        else:
            # brand (title-cased) + product line if the one-liner names one.
            title = self.brand.replace("-", " ").replace("_", " ").title()
            if "(" in self.one_liner and ")" in self.one_liner:
                inner = self.one_liner.split("(", 1)[1].split(")", 1)[0].strip()
                if inner and inner.lower() not in title.lower():
                    title = f"{title} · {inner}"
        return {
            "title": title,
            "one_liner": self.one_liner,
            "positioning": self.positioning,
            "themes": self.themes,
            "competitors": self.competitors,
        }

    def launch_description(self) -> str:
        """Compose the free-form launch description fed to search/scoring."""
        parts = [self.one_liner, self.positioning]
        if self.themes:
            parts.append("Themes: " + ", ".join(self.themes) + ".")
        if self.competitors:
            parts.append("Category peers: " + ", ".join(self.competitors) + ".")
        return " ".join(p for p in parts if p).strip()


def _sub(cls, data: Any):
    """Build a nested dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return cls()
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_brand_config(path: str | Path) -> BrandConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "brand" not in raw:
        raise ValueError(f"{path}: brand config must have a 'brand' field")
    return BrandConfig(
        brand=raw["brand"],
        display_name=raw.get("display_name", ""),
        one_liner=raw.get("one_liner", ""),
        positioning=raw.get("positioning", ""),
        themes=list(raw.get("themes", []) or []),
        competitors=list(raw.get("competitors", []) or []),
        do_not=list(raw.get("do_not", []) or []),
        platforms=list(raw.get("platforms", []) or []) or ["substack", "blog", "podcast", "youtube"],
        discovery=_sub(DiscoveryCfg, raw.get("discovery")),
        tiering=_sub(TieringCfg, raw.get("tiering")),
        deepdive=_sub(DeepDiveCfg, raw.get("deepdive")),
        budget=_sub(BudgetCfg, raw.get("budget")),
    )
