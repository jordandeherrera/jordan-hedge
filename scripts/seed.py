"""
Seed the Jordan HEDGE database from ontology/seed.json.
Creates concepts, relations, threads, and initial hypotheses.
"""

import json
import sys
import math
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engine.db import initialize_db, prob_to_logit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED_PATH = Path(__file__).parent.parent / "ontology" / "seed.json"


def seed():
    conn = initialize_db()
    seed_data = json.loads(SEED_PATH.read_text())

    # ── Concepts ──────────────────────────────────
    logger.info("Seeding concepts...")
    for c in seed_data["concepts"]:
        conn.execute("""
            INSERT OR IGNORE INTO concepts (name, type, description, base_rate, risk_weight)
            VALUES (?, ?, ?, ?, ?)
        """, (
            c["name"],
            c["type"],
            c.get("description", ""),
            c.get("base_rate", 0.1),
            c.get("risk_weight", 1.0),
        ))
    conn.commit()
    logger.info(f"  {len(seed_data['concepts'])} concepts seeded")

    # ── Relations ─────────────────────────────────
    logger.info("Seeding relations...")
    for r in seed_data["relations"]:
        signal = conn.execute(
            "SELECT id FROM concepts WHERE name = ?", (r["signal"],)
        ).fetchone()
        hypothesis = conn.execute(
            "SELECT id FROM concepts WHERE name = ?", (r["hypothesis"],)
        ).fetchone()

        if not signal or not hypothesis:
            logger.warning(f"  Skipping relation: {r['signal']} → {r['hypothesis']} (concept not found)")
            continue

        # INSERT OR REPLACE so re-seeding updates LLRs and half-life values
        conn.execute("""
            INSERT OR REPLACE INTO relations
                (signal_concept_id, hypothesis_concept_id, llr_pos, llr_neg,
                 expected_info_gain, evidence_half_life_hours)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            signal["id"],
            hypothesis["id"],
            r["llr_pos"],
            r["llr_neg"],
            r.get("info_gain", 0.0),
            r.get("half_life_hours", 0),
        ))
    conn.commit()
    logger.info(f"  {len(seed_data['relations'])} relations seeded")

    # ── Threads + Hypotheses ──────────────────────
    logger.info("Seeding threads and hypotheses...")
    hypothesis_types = conn.execute(
        "SELECT id, name, base_rate FROM concepts WHERE type = 'hypothesis_type'"
    ).fetchall()

    for t in seed_data["threads"]:
        # Insert thread
        conn.execute("""
            INSERT OR IGNORE INTO threads (name, domain, description)
            VALUES (?, ?, ?)
        """, (t["name"], t["domain"], t.get("description", "")))
        conn.commit()

        thread = conn.execute(
            "SELECT id FROM threads WHERE name = ?", (t["name"],)
        ).fetchone()
        thread_id = thread["id"]

        # Create one hypothesis per hypothesis_type for this thread
        # INSERT OR IGNORE so existing hypotheses (with accumulated evidence) are preserved
        for ht in hypothesis_types:
            log_odds_prior = prob_to_logit(ht["base_rate"])
            conn.execute("""
                INSERT OR IGNORE INTO hypotheses
                    (thread_id, hypothesis_type_id, log_odds_prior, log_odds_posterior)
                VALUES (?, ?, ?, ?)
            """, (thread_id, ht["id"], log_odds_prior, log_odds_prior))

        conn.commit()

        # Ingest initial signals to set starting belief state
        for signal_name in t.get("initial_signals", []):
            concept = conn.execute(
                "SELECT id FROM concepts WHERE LOWER(name) = LOWER(?)", (signal_name,)
            ).fetchone()
            if not concept:
                # Try partial match
                concept = conn.execute(
                    "SELECT id FROM concepts WHERE LOWER(name) LIKE LOWER(?) LIMIT 1",
                    (f"%{signal_name.replace('_', ' ')}%",)
                ).fetchone()
            if not concept:
                logger.warning(f"  Signal not found: {signal_name} for thread '{t['name']}'")
                continue

            conn.execute("""
                INSERT OR REPLACE INTO facts
                    (thread_id, concept_id, value, confidence, source)
                VALUES (?, ?, 'present', 0.9, 'seed')
            """, (thread_id, concept["id"]))

        conn.commit()
        logger.info(f"  Thread '{t['name']}' ({t['domain']}) initialized")

    # ── Verify ────────────────────────────────────
    stats = {
        "concepts":   conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
        "relations":  conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
        "threads":    conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
        "hypotheses": conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0],
        "facts":      conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
    }

    logger.info("\n✅ Seed complete:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    conn.close()


if __name__ == "__main__":
    seed()
