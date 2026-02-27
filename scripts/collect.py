"""
Run a full collection pass: Gmail → HEDGE signals → updated belief state.
Designed to run on a schedule (heartbeat or cron).
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run():
    from src.engine.hedge_engine import HedgeEngine
    from src.collectors.gmail_collector import GmailCollector

    engine = HedgeEngine()
    collector = GmailCollector(engine=engine)

    logger.info("Starting collection pass...")
    summary = collector.collect()

    if not summary:
        logger.info("No new signals found.")
        return

    print("\n📬 Gmail Collection Results")
    print("─" * 50)
    for thread_name, result in summary.items():
        signals = result["signals"]
        print(f"\n  📌 {thread_name}")
        for sig in signals:
            print(f"     + {sig}")

    # Print updated belief state for affected threads
    print("\n📊 Updated Belief States")
    print("─" * 50)
    for thread_name in summary:
        state = engine.get_thread_state(thread_name)
        if not state:
            continue
        top = state.top_hypothesis
        if top:
            print(f"\n  {thread_name}")
            print(f"  Top: {top.hypothesis_type} — {top.probability:.0%} (uncertainty: {top.uncertainty:.0%})")
        if state.next_actions:
            print(f"  → {state.next_actions[0]['title']}")


if __name__ == "__main__":
    run()
