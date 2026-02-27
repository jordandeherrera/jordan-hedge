"""
Full collection pass: Gmail + Calendar + GitHub → HEDGE signals → updated belief state.
Designed to run on a schedule (heartbeat or cron).
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIGNAL_EMOJI = {
    "deployment_issue":       "🔴",
    "pr_open_stale":          "🟡",
    "progress_stalled":       "🟠",
    "deadline_imminent":      "🚨",
    "deadline_approaching":   "⏰",
    "calendar_event_soon":    "📅",
    "appointment_upcoming":   "🏥",
    "email_received_urgent":  "📨",
    "email_unanswered":       "📭",
    "external_party_silent":  "🔕",
    "counterparty_active":    "⚠️ ",
    "new_evidence_available": "📬",
    "response_received":      "✅",
    "task_stale":             "💤",
}


def print_summary(source: str, summary: dict):
    if not summary:
        print(f"  {source}: no new signals")
        return
    for thread_name, result in summary.items():
        signals = result.get("signals", [])
        emojis = " ".join(SIGNAL_EMOJI.get(s, "•") for s in signals)
        print(f"  {source} → [{thread_name}]  {emojis}")
        for s in signals:
            print(f"           {SIGNAL_EMOJI.get(s,'•')} {s}")


def run():
    from src.engine.hedge_engine import HedgeEngine
    from src.collectors.gmail_collector import GmailCollector
    from src.collectors.calendar_collector import CalendarCollector
    from src.collectors.github_collector import GitHubCollector

    engine = HedgeEngine()

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  JORDAN HEDGE — Collection Pass")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Gmail ──────────────────────────────────
    print("\n📬 Gmail")
    gmail = GmailCollector(engine=engine)
    gmail_summary = gmail.collect()
    print_summary("gmail", gmail_summary)

    # ── Calendar ───────────────────────────────
    print("\n📅 Calendar")
    cal = CalendarCollector(engine=engine)
    cal_summary = cal.collect()
    print_summary("calendar", cal_summary)

    # ── GitHub ─────────────────────────────────
    print("\n🐙 GitHub")
    gh = GitHubCollector(engine=engine)
    gh_summary = gh.collect()
    print_summary("github", gh_summary)

    # ── Updated Belief States ──────────────────
    all_touched = set(gmail_summary) | set(cal_summary) | set(gh_summary)
    if all_touched:
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  Updated Belief States")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for thread_name in sorted(all_touched):
            state = engine.get_thread_state(thread_name)
            if not state:
                continue
            top = state.top_hypothesis
            if top:
                unc_bars = "?" * round(top.uncertainty * 3)
                print(f"\n  📌 {thread_name}")
                print(f"     {top.hypothesis_type}: {top.probability:.0%}  uncertainty: {unc_bars or '✓'}")
            if state.next_actions:
                print(f"     → {state.next_actions[0]['title']}")

    # ── Top Priority Actions ───────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  TOP PRIORITY ACTIONS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    actions = engine.get_priority_actions(limit=5)
    if actions:
        for i, a in enumerate(actions, 1):
            print(f"  {i}. {a.title}")
            print(f"     [{a.thread_name}]  utility: {a.expected_utility:.2f}")
    else:
        print("  No pending actions.")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


if __name__ == "__main__":
    run()
