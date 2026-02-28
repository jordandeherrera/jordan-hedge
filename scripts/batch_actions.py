#!/usr/bin/env python3
"""
batch_actions.py — Surface batched action recommendations from HEDGE threads.

Usage:
    python scripts/batch_actions.py               # show all recommendations
    python scripts/batch_actions.py --only batch  # show only batch opportunities
    python scripts/batch_actions.py --only urgent_solo
    python scripts/batch_actions.py --json        # machine-readable output
    python scripts/batch_actions.py --limit 30    # max actions to consider
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Load .env from repo root before anything else
_repo_root = Path(__file__).parent.parent
_env_file = _repo_root / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Manual fallback
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(_repo_root))

from src.engine.db import get_connection
from src.engine.action_batcher import ActionBatcher


ICONS = {
    "batch":        "🔗",
    "individual":   "▶️ ",
    "urgent_solo":  "🚨",
}

COLORS = {
    "batch":       "\033[32m",   # green
    "individual":  "\033[37m",   # white/gray
    "urgent_solo": "\033[31m",   # red
    "reset":       "\033[0m",
}


def fmt_batch(b, use_color: bool = True) -> str:
    icon = ICONS.get(b.recommendation, "•")
    c = COLORS[b.recommendation] if use_color else ""
    r = COLORS["reset"] if use_color else ""

    lines = [
        f"{c}{icon} [{b.recommendation.upper()}]{r}  {b.substrate_label}",
    ]

    for a in sorted(b.actions, key=lambda x: -x.expected_utility):
        entropy_tag = f"  entropy={a.thread_entropy:.2f}" if a.thread_entropy < 0.35 else ""
        urgency_tag = f"  urgency={a.urgency_score:.2f}" if a.urgency_score > 0.5 else ""
        lines.append(
            f"   • [{a.domain}] {a.title}"
            f"  (utility={a.expected_utility:.2f}{urgency_tag}{entropy_tag})"
        )

    lines.append(f"   → {b.rationale}")

    if b.recommendation == "batch":
        domains = sorted(b.domains)
        cross = f"  ⚡ cross-domain [{' + '.join(domains)}]" if len(domains) > 1 else ""
        lines.append(
            f"   💰 Effort: solo={b.estimated_effort_solo:.1f} → "
            f"batch={b.estimated_effort_batch:.1f}  "
            f"(net save: {b.net_savings:.1f} units, "
            f"overlap: {b.intra_similarity:.0%})"
            f"{cross}"
        )

    return "\n".join(lines)


def to_dict(b) -> dict:
    return {
        "cluster_id": b.cluster_id,
        "recommendation": b.recommendation,
        "substrate_label": b.substrate_label,
        "n_actions": b.n,
        "threads": b.threads,
        "domains": list(b.domains),
        "total_utility": round(b.total_utility, 3),
        "max_urgency": round(b.max_urgency, 3),
        "intra_similarity": round(b.intra_similarity, 3),
        "estimated_effort_solo": b.estimated_effort_solo,
        "estimated_effort_batch": b.estimated_effort_batch,
        "net_savings": b.net_savings,
        "rationale": b.rationale,
        "actions": [
            {
                "action_id": a.action_id,
                "thread": a.thread_name,
                "domain": a.domain,
                "action_type": a.action_type,
                "title": a.title,
                "expected_utility": round(a.expected_utility, 3),
                "urgency_score": round(a.urgency_score, 3),
                "thread_entropy": round(a.thread_entropy, 3),
            }
            for a in sorted(b.actions, key=lambda x: -x.expected_utility)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="HEDGE Action Batcher")
    parser.add_argument("--only", choices=["batch", "individual", "urgent_solo"],
                        help="Filter to specific recommendation type")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--limit", type=int, default=50, help="Max actions to consider")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    conn = get_connection()
    try:
        batcher = ActionBatcher(conn)
        batches = batcher.compute_batches(limit=args.limit)
    finally:
        conn.close()

    if args.only:
        batches = [b for b in batches if b.recommendation == args.only]

    if args.json:
        print(json.dumps([to_dict(b) for b in batches], indent=2))
        return

    if not batches:
        print("No pending actions found.")
        return

    use_color = not args.no_color and sys.stdout.isatty()

    # Summary header
    by_rec = {}
    for b in batches:
        by_rec.setdefault(b.recommendation, []).append(b)

    print("═" * 70)
    print("  HEDGE Action Batcher")
    print(f"  {len(batches)} clusters from {sum(b.n for b in batches)} pending actions")

    batch_clusters = by_rec.get("batch", [])
    if batch_clusters:
        total_save = sum(b.net_savings for b in batch_clusters)
        print(f"  🔗 {len(batch_clusters)} batch opportunities  "
              f"(~{total_save:.1f} effort units saveable)")

    urgent = by_rec.get("urgent_solo", [])
    if urgent:
        print(f"  🚨 {len(urgent)} urgent actions needing immediate execution")

    print("═" * 70)
    print()

    # Urgent first
    for b in by_rec.get("urgent_solo", []):
        print(fmt_batch(b, use_color))
        print()

    # Batch opportunities
    for b in by_rec.get("batch", []):
        print(fmt_batch(b, use_color))
        print()

    # Individual
    for b in by_rec.get("individual", []):
        print(fmt_batch(b, use_color))
        print()


if __name__ == "__main__":
    main()
