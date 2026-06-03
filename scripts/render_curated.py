"""Render a CURATED, A/B-tiered subset of a PitchFinder search to HTML.

Why: the relevance scorer (DeepSeek) has good recall but poor precision —
content-farm substacks score as high as genuine experts. So we hand-curate
the pitch list, split it into two outreach tiers, and render only the vetted
creators, reusing the tool's own HTML renderer (_render_results).

Tiers (by relevance + confidence, NOT by whether we deep-dived):
  A — high relevance, high confidence — strongly recommend outreach
  B — moderate relevance — worth reaching out to
Anything not in A or B is dropped (content-farm / off-topic).

Usage: .venv/bin/python scripts/render_curated.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pitchfinder.db import get_conn
from pitchfinder.pipeline import _rank_creators, _render_results

DB = "pitchfinder.db"
SEARCH_ID = 8
MIN_SCORE = 40  # explicit allowlist, so a low floor only affects our vetted ids
OUTPUT = Path("reports/apodex-launch-final.html")

# ===== Tier A — high relevance, high confidence — strongly recommend outreach =====
TIER_A = [
    # frontier reasoning / post-training / RL
    1,    # Nathan Lambert — Interconnects (post-training / RLVR)
    184,  # Cameron R. Wolfe — Deep Learning Focus (reasoning)
    189,  # TheSequence — Jesús Rodriguez
    2,    # Jack Clark — Import AI (Anthropic cofounder)
    6,    # swyx — Latent Space (newsletter)
    4,    # Sebastian Raschka — Ahead of AI
    11,   # Gary Marcus — verify-vs-guess, huge reach
    146,  # Nathan Benaich — State of AI
    # podcasts (directly on-topic)
    19,   # Dwarkesh Patel
    16,   # Latent Space Podcast
    17,   # The Cognitive Revolution
    18,   # No Priors
    20,   # Machine Learning Street Talk (podcast)
    217,  # Zero Knowledge Podcast (proof / verifiable)
    # YouTube (deep content / evals)
    27,   # Two Minute Papers
    28,   # AI Explained — frontier-model evals
    29,   # bycloud — LLM research surveys
    26,   # Machine Learning Street Talk (YouTube)
    # AI for science (Apodex differentiator)
    172,  # Decoding Bio — Cantos & Hummingbird
    208,  # Kit Yates — math & science
    162,  # Pablo Lubroth — Decoding Science
    # verifiable AI / formal methods / evals (core to "Verify")
    213,  # Hillel Wayne — formal methods
    222,  # Atlas Computing — FMxAI
    158,  # Software Analyst — agentic remediation / verification
    244,  # AI Evaluation Digest
    149,  # Johannes Gasteiger — AI Safety Frontier (evals)
    # agents + dev
    229,  # Berkeley RDI — Agentic AI Weekly
    160,  # Gergely Orosz — The Pragmatic Engineer
    3,    # Simon Willison — practical LLM tooling
    8,    # Eugene Yan — applied ML / evals
]

# ===== Tier B — moderate relevance — worth reaching out to =====
TIER_B = [
    13,   # Rohit Krishnan — Strange Loop Canon
    12,   # Ethan Mollick — applied AI, huge reach
    10,   # Zvi Mowshowitz — exhaustive roundups
    22,   # TWIML AI Podcast
    23,   # Practical AI (podcast)
    220,  # Boston Computation Club (formal / computation)
    273,  # Welch Labs (deep ML/math explainers)
    169,  # Jassi Pannu — AI for Biology
    179,  # AI-for-Science: Manhattan Project — SCSP
    163,  # Manas Mahale — Cheminformatics
    170,  # Matt Lubin — Bio-Security Stack
    255,  # AI × MedEd (Andrew O'Malley) — medical AI
    210,  # Quinn Dougherty — secure / verifiable AI
    237,  # Ken Huang — Agentic AI
    231,  # Cobus Greyling — agents / LLM
    148,  # Janelle Teng Wade — AI infra / agentic
    150,  # Center for AI Safety — AI Safety Newsletter
    193,  # AI with Aish (Aishwarya Srinivasan)
    178,  # AIhub — academic AI news digest
    254,  # Legal Quants — legal AI (Apodex legal vertical)
    235,  # Oliver Patel — Enterprise AI Governance
    161,  # Sarthak Rastogi — AI Agent Engineering
    157,  # DD Kang — agents / RL (academic)
    156,  # Aaron Tay — agent-based deep research
    267,  # Aakash Gupta — AI PM / evals
    226,  # Sundas Khalid — data science / applied AI
    147,  # Christopher S. Penn — applied AI / deep research
    206,  # Michael Harris — Silicon Reckoner (math critique)
    187,  # Hung Le — Neurocoder Tales (post-training / RL)
    243,  # The Slow AI — research analysis
    249,  # Future AGI — in-depth AI evals writing
]

LEGEND = (
    "Apodex launch outreach list. Tiers: "
    "[A] high relevance + high confidence — strongly recommend outreach;  "
    "[B] moderate relevance — worth reaching out to. "
    "Ranks 1-30 are Tier A (most deep-verified); 31+ are Tier B. — "
)


def main() -> None:
    conn = get_conn(DB)
    try:
        description = conn.execute(
            "SELECT description FROM searches WHERE id = ?", (SEARCH_ID,)
        ).fetchone()["description"]
    finally:
        conn.close()

    ranked = _rank_creators(DB, SEARCH_ID, MIN_SCORE, max_creators=999)
    by_id = {c["creator_id"]: c for c in ranked}

    curated: list[dict] = []
    missing: list[int] = []
    for tier, ids in (("A", TIER_A), ("B", TIER_B)):
        for cid in ids:
            c = by_id.get(cid)
            if not c:
                missing.append(cid)
                continue
            c["name"] = f"[{tier}] {c['name']}"  # tier badge in the card title
            curated.append(c)

    conn = get_conn(DB)
    try:
        for c in curated:
            ar = conn.execute(
                "SELECT angles_json FROM pitch_angles WHERE search_id = ? AND creator_id = ?",
                (SEARCH_ID, c["creator_id"]),
            ).fetchone()
            c["angles"] = json.loads(ar["angles_json"]) if ar and ar["angles_json"] else []
            dr = conn.execute(
                "SELECT payload_json, model FROM deep_dives WHERE search_id = ? AND creator_id = ?",
                (SEARCH_ID, c["creator_id"]),
            ).fetchone()
            if dr and dr["payload_json"]:
                payload = json.loads(dr["payload_json"])
                if not payload.get("_source_model"):
                    payload["_source_model"] = dr["model"] or ""
                c["deep_dive"] = payload
            else:
                c["deep_dive"] = None
    finally:
        conn.close()

    n_a, n_b = len(TIER_A), len(TIER_B)
    n_dd = sum(1 for c in curated if c.get("deep_dive"))
    print(f"curated={len(curated)} (A={n_a}, B={n_b})  with_deep_dive={n_dd}")
    if missing:
        print(f"NOTE: ids not in >= {MIN_SCORE} pool (skipped): {missing}")

    _render_results(LEGEND + description, curated, SEARCH_ID, OUTPUT)


if __name__ == "__main__":
    main()
