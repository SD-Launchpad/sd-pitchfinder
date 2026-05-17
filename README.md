# PitchFinder

Internal CLI that finds AI/tech creators worth pitching when you launch a
product. Pulls newsletter / podcast / YouTube RSS feeds, scores recent content
against a launch description, and produces a ranked list of creators with
individualized pitch angles.

Single-user tool. No auth, no SaaS, no multi-tenant.

## Install

Requires Python 3.11+. The build was tested with 3.12.

```bash
cd shanda/pitchfinder
uv venv --python 3.12 .venv          # or: python3 -m venv .venv
source .venv/bin/activate
pip install -e .                      # or: uv pip install -e .
```

Add your OpenRouter key:

```bash
cp .env.example .env
# edit .env to set OPENROUTER_API_KEY=sk-or-...
```

LLM calls go through [OpenRouter](https://openrouter.ai), so any model
OpenRouter exposes is fair game (DeepSeek, GLM, Claude, Gemini, GPT, Kimi, …).
Defaults are cheap (DeepSeek V3.1 for both relevance and pitch); override per
`.env` as you like.

## Usage

```bash
pitchfinder init                          # create pitchfinder.db
pitchfinder load seed_creators.yaml       # upsert seed creators (~30)
pitchfinder refresh                       # fetch all RSS feeds (last 90d)
pitchfinder refresh --platforms substack  # platform-filtered refresh
pitchfinder search "Launch description here" --output reports/launch.md
pitchfinder show 1                        # re-render search #1
pitchfinder status 7 launch-name pitched --notes "sent 2026-05-17"
```

### Pipeline

`search` runs 4 steps:

1. **Topic extraction** — Haiku extracts topics/keywords from the launch description.
2. **Relevance scoring** — every item in the lookback window (default 90d) is
   scored 0–100 by Haiku, in parallel (default 8 workers).
3. **Creator ranking** — items meeting `--min-score` (default 70) roll up to
   their creator; creator score = max of their item scores. Ties break by
   `influence_score`. Top `--max-creators` (default 30) move on.
4. **Pitch angles** — Sonnet generates 2–3 individualized pitch angles per
   creator, referencing their top items.

Outputs go to the terminal (rich tables + per-creator detail) and an optional
Markdown file via `--output`.

### Discovery (seed library expansion)

```bash
pitchfinder discover-podcasts "AI engineering"
pitchfinder discover-substack https://www.interconnects.ai
```

These print YAML stanzas you can review and paste into `seed_creators.yaml`
before re-running `load`. Neither one auto-inserts into the DB.

YouTube channel ID discovery is intentionally manual: open the channel page,
view source, search for `"channelId"`, and assemble
`https://www.youtube.com/feeds/videos.xml?channel_id=<UC...>`.

## Cost per search

Order of magnitude:
- Relevance scoring: one LLM call per item in lookback window. With ~30 seed
  creators and roughly 50 items each / 90d, that's up to ~1500 calls.
- Pitch angles: one LLM call per top-ranked creator (≤30).

| Model combo | Approx cost/search |
|---|---|
| DeepSeek V3.1 for both (default) | ~$0.05–0.15 |
| DeepSeek V3.1 scoring + Claude Sonnet 4.6 pitch | ~$0.15–0.30 |
| Claude Haiku 4.5 + Claude Sonnet 4.6 | ~$0.50–1.00 |

Override via `.env` — any [OpenRouter model id](https://openrouter.ai/models)
works:

```
PITCHFINDER_RELEVANCE_MODEL=deepseek/deepseek-chat-v3.1   # cheap, fast
PITCHFINDER_PITCH_MODEL=anthropic/claude-sonnet-4.5       # pitch quality
```

Other good candidates: `z-ai/glm-4.6`, `google/gemini-2.5-flash`,
`moonshotai/kimi-k2-0905`, `anthropic/claude-haiku-4.5`.

## Data layout

SQLite at `./pitchfinder.db` (gitignored). Schema:

| table | what it holds |
|---|---|
| `creators` | seed/discovered creators, one row per (platform, handle) |
| `items` | recent content (article / episode / video), URL unique |
| `searches` | one row per `search` invocation, with extracted topics |
| `relevance_scores` | per-search per-item score + reason |
| `pitch_angles` | per-search per-creator JSON list of angle objects |
| `outreach` | manual status tracking, one row per (creator, campaign) |

## Notes

- `pitchfinder.db` lives in the cwd. Run all commands from this project root.
- Re-running `load` is idempotent (`ON CONFLICT(platform, handle) DO UPDATE`).
- Re-running `refresh` is idempotent (`INSERT OR IGNORE` on the URL-unique `items`).
- Feeds that return zero new items inside the lookback window are flagged
  `empty/failed` in the refresh table — same UI cell, two different causes.

## Out of scope (by design)

No email sending, no auth / multi-tenant, no Streamlit UI, no X / LinkedIn /
TikTok ingestion, no YouTube transcript extraction, no mainland Chinese media.
