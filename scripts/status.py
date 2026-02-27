"""
Jordan HEDGE Status — surface current belief state across all threads.
This is the primary read interface: what does the engine think right now?
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engine.hedge_engine import HedgeEngine

logging.basicConfig(level=logging.WARNING)

HYPOTHESIS_EMOJI = {
    "URGENT_ACTION_NEEDED": "⚠️ ",
    "BINDING_CONSTRAINT":   "🔒",
    "AT_RISK":              "🚨",
    "EXTERNAL_BLOCKED":     "📬",
    "STALE_THREAD":         "💤",
    "DECISION_PENDING":     "🤔",
    "OPPORTUNITY":          "✨",
}

DOMAIN_EMOJI = {
    "estate":    "⚖️ ",
    "malpractice": "🏥",
    "MEL":       "🤖",
    "ridgeline": "📊",
    "health":    "💊",
    "family":    "👨‍👩‍👧‍👦",
    "finance":   "💰",
}


def bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def run():
    engine = HedgeEngine()
    health = engine.health()

    print("\n" + "═" * 60)
    print("  JORDAN HEDGE  —  Belief State Dashboard")
    print("═" * 60)
    print(f"  Threads: {health['active_threads']}  |  "
          f"Facts: {health['facts']}  |  "
          f"Hypotheses: {health['active_hypotheses']}  |  "
          f"Pending Actions: {health['pending_actions']}")
    print("═" * 60)

    threads = engine.get_all_threads()

    if not threads:
        print("  No active threads.")
        return

    # Group by domain
    by_domain: dict = {}
    for t in threads:
        by_domain.setdefault(t.domain, []).append(t)

    for domain, domain_threads in sorted(by_domain.items()):
        emoji = DOMAIN_EMOJI.get(domain, "📁")
        print(f"\n  {emoji}  {domain.upper()}")
        print("  " + "─" * 56)

        for thread in domain_threads:
            print(f"\n  📌 {thread.name}")
            print(f"     Priority: {bar(thread.priority_score)} {thread.priority_score:.0%}")

            if thread.hypotheses:
                # Show top 3 hypotheses above 20% probability
                top = [h for h in thread.hypotheses if h.probability >= 0.2][:3]
                for h in top:
                    emoji = HYPOTHESIS_EMOJI.get(h.hypothesis_type, "  ")
                    unc = "?" * round(h.uncertainty * 3)
                    print(f"     {emoji} {h.hypothesis_type:<24} {h.probability:>5.0%}  {bar(h.probability, 12)}  {unc}")

            if thread.next_actions:
                action = thread.next_actions[0]
                print(f"     → {action['title']}")

    # Priority actions
    print("\n" + "═" * 60)
    print("  TOP ACTIONS  (by expected utility)")
    print("═" * 60)
    actions = engine.get_priority_actions(limit=7)
    if actions:
        for i, a in enumerate(actions, 1):
            print(f"  {i}. [{a.thread_name}]  {a.title}")
            print(f"     Utility: {a.expected_utility:.2f}  |  Type: {a.action_type}")
    else:
        print("  No pending actions.")

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    run()
