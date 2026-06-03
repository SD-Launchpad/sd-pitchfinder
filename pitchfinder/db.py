from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS creators (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  platform TEXT NOT NULL,
  handle TEXT NOT NULL,
  url TEXT,
  feed_url TEXT,
  contact_email TEXT,
  contact_other TEXT,
  topics TEXT,
  influence_score INTEGER DEFAULT 50,
  notes TEXT,
  source TEXT DEFAULT 'manual',
  added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(platform, handle)
);

CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  creator_id INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  title TEXT,
  url TEXT UNIQUE,
  published_at DATETIME,
  summary TEXT,
  fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (creator_id) REFERENCES creators(id)
);

CREATE TABLE IF NOT EXISTS searches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  description TEXT NOT NULL,
  extracted_topics TEXT,
  ran_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relevance_scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  search_id INTEGER NOT NULL,
  item_id INTEGER NOT NULL,
  score INTEGER NOT NULL,
  reason TEXT,
  UNIQUE(search_id, item_id),
  FOREIGN KEY (search_id) REFERENCES searches(id),
  FOREIGN KEY (item_id) REFERENCES items(id)
);

CREATE TABLE IF NOT EXISTS pitch_angles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  search_id INTEGER NOT NULL,
  creator_id INTEGER NOT NULL,
  angles_json TEXT,
  UNIQUE(search_id, creator_id)
);

CREATE TABLE IF NOT EXISTS outreach (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  creator_id INTEGER NOT NULL,
  campaign TEXT NOT NULL,
  status TEXT DEFAULT 'not_contacted',
  pitched_at DATETIME,
  replied_at DATETIME,
  notes TEXT,
  UNIQUE(creator_id, campaign)
);

CREATE TABLE IF NOT EXISTS deep_dives (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  search_id INTEGER NOT NULL,
  creator_id INTEGER NOT NULL,
  model TEXT,
  payload_json TEXT NOT NULL,
  ran_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(search_id, creator_id),
  FOREIGN KEY (search_id) REFERENCES searches(id),
  FOREIGN KEY (creator_id) REFERENCES creators(id)
);

CREATE TABLE IF NOT EXISTS creator_tiers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  search_id INTEGER NOT NULL,
  creator_id INTEGER NOT NULL,
  tier TEXT NOT NULL,                      -- 'A' | 'B' | 'drop'
  rationale TEXT,
  source TEXT DEFAULT 'auto',              -- 'auto' (LLM) | 'manual' (override)
  set_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(search_id, creator_id),
  FOREIGN KEY (search_id) REFERENCES searches(id),
  FOREIGN KEY (creator_id) REFERENCES creators(id)
);

CREATE INDEX IF NOT EXISTS idx_items_creator ON items(creator_id);
CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_at);
CREATE INDEX IF NOT EXISTS idx_relevance_search ON relevance_scores(search_id);
"""


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Idempotent column additions for DBs created before a column existed.
# SQLite has no "ADD COLUMN IF NOT EXISTS", so we guard on PRAGMA table_info.
_MIGRATIONS: list[tuple[str, str, str]] = [
    ("searches", "brand", "ALTER TABLE searches ADD COLUMN brand TEXT"),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, column, ddl in _MIGRATIONS:
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(ddl)
    conn.commit()


def init_schema(db_path: str | Path) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        conn.commit()
    finally:
        conn.close()
