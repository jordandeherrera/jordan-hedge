"""
Jordan HEDGE Push Notifier

Checks for high-priority threads/hypotheses and fires an OpenClaw system event
when something crosses the threshold. Designed to run after collect.py.

Thresholds:
- priority_score >= 0.75 → HIGH — push immediately
- priority_score >= 0.50 → MEDIUM — push only if not pushed in last 6h
- below 0.50 → silent

State tracked in ~/jordan-hedge/data/push_state.json to avoid duplicate pushes.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engine.hedge_engine import HedgeEngine

STATE_FILE = Path(__file__).parent.parent / "data" / "push_state.json"
HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.50
MEDIUM_COOLDOWN_HOURS = 6


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def should_push(thread_id: str, score: float, state: dict) -> bool:
    if score >= HIGH_THRESHOLD:
        return True
    if score >= MEDIUM_THRESHOLD:
        last_push = state.get(thread_id)
        if not last_push:
            return True
        last_dt = datetime.fromisoformat(last_push)
        return datetime.utcnow() - last_dt > timedelta(hours=MEDIUM_COOLDOWN_HOURS)
    return False


def fire_system_event(text: str):
    result = subprocess.run(
        ["openclaw", "system", "event", "--text", text, "--mode", "now"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def main():
    engine = HedgeEngine()
    threads = engine.get_all_threads(min_priority=MEDIUM_THRESHOLD)

    if not threads:
        print("No threads above threshold.")
        return

    state = load_state()
    now = datetime.utcnow().isoformat()
    pushed = []

    for thread in threads:
        if not should_push(thread.thread_id, thread.priority_score, state):
            continue

        top = thread.top_hypothesis
        top_label = f"{top.hypothesis_type} ({top.probability:.0%})" if top else "unknown"

        actions = thread.next_actions[:1]
        action_text = f" → {actions[0]['description']}" if actions else ""

        msg = (
            f"[Jordan HEDGE] 🔔 {thread.name} | "
            f"priority: {thread.priority_score:.2f} | "
            f"top hypothesis: {top_label}{action_text}"
        )

        if fire_system_event(msg):
            state[thread.thread_id] = now
            pushed.append(thread.name)
            print(f"Pushed: {thread.name}")
        else:
            print(f"Failed to push: {thread.name}", file=sys.stderr)

    save_state(state)

    if not pushed:
        print("No new pushes needed.")


if __name__ == "__main__":
    main()
