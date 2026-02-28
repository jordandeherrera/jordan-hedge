"""
Database connection and custom SQLite function registration.
Mirrors MEL's db/index.ts custom function pattern.
"""

import sqlite3
import math
import os
from pathlib import Path

DB_PATH = os.environ.get("JORDAN_HEDGE_DB", str(Path(__file__).parent.parent.parent / "jordan_hedge.db"))


def logit_to_prob(log_odds: float) -> float:
    """Sigmoid: converts log-odds to probability."""
    if log_odds is None:
        return 0.5
    try:
        return 1.0 / (1.0 + math.exp(-log_odds))
    except OverflowError:
        return 0.0 if log_odds < 0 else 1.0


def prob_to_logit(p: float) -> float:
    """Logit: converts probability to log-odds."""
    if p is None or p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    return math.log(p / (1.0 - p))


def decay(dt_hours: float, half_life_hours: float) -> float:
    """Exponential decay: exp(-0.693 * dt / half_life). 0 half_life = no decay."""
    if half_life_hours is None or half_life_hours <= 0:
        return 1.0
    if dt_hours is None or dt_hours < 0:
        return 1.0
    return math.exp(-0.693 * dt_hours / half_life_hours)


def belief_entropy(log_odds: float) -> float:
    """Shannon entropy of a Bernoulli belief. Max at P=0.5, 0 at P=0 or P=1."""
    p = logit_to_prob(log_odds)
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def uncertainty(log_odds: float) -> float:
    """Distance from certainty: 0=certain, 1=maximally uncertain."""
    p = logit_to_prob(log_odds)
    return 1.0 - abs(2.0 * p - 1.0)


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Return a connection with custom math functions registered."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Register custom functions (mirrors MEL's db/index.ts)
    conn.create_function("logit_to_prob",  1, logit_to_prob)
    conn.create_function("prob_to_logit",  1, prob_to_logit)
    conn.create_function("decay",          2, decay)
    conn.create_function("belief_entropy", 1, belief_entropy)
    conn.create_function("uncertainty",    1, uncertainty)
    conn.create_function("ln",             1, math.log)
    conn.create_function("exp_fn",         1, math.exp)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    return conn


def initialize_db(db_path: str = None) -> sqlite3.Connection:
    """Initialize database schema if not already present."""
    conn = get_connection(db_path)
    # Run incremental migrations first — adds columns to existing tables
    # before schema.sql's CREATE TABLE IF NOT EXISTS runs (which is a no-op
    # for existing tables and won't pick up new column definitions).
    run_migrations(conn)
    schema_path = Path(__file__).parent.parent.parent / "db" / "schema.sql"
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def run_migrations(conn: sqlite3.Connection) -> None:
    """
    Apply incremental migrations from db/migrations/*.sql in filename order.
    Each migration is applied once; errors on already-applied DDL (e.g.
    duplicate column) are silently swallowed.
    """
    migrations_dir = Path(__file__).parent.parent.parent / "db" / "migrations"
    if not migrations_dir.exists():
        return

    for migration_file in sorted(migrations_dir.glob("*.sql")):
        with open(migration_file) as f:
            raw = f.read()
        import re as _re
        raw = _re.sub(r'--[^\n]*', '', raw)   # strip inline comments before splitting
        statements = [s.strip() for s in raw.split(";") if s.strip()]
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                # Ignore "duplicate column" and "already exists" errors —
                # these mean the migration was already applied.
                msg = str(e).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    pass
                else:
                    raise
    conn.commit()
