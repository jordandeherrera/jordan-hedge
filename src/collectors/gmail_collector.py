"""
Gmail Collector for Jordan HEDGE.

Scans Gmail inbox and sent mail to extract signals for each thread:
- Email received from relevant sender  → email_received_urgent
- Outbound email unanswered > N days   → email_unanswered + external_party_silent
- Counterparty/adversarial email       → counterparty_active
- Response received (after waiting)    → response_received (negative signal on EXTERNAL_BLOCKED)

Uses `gog gmail` CLI — already authenticated.
"""

import json
import subprocess
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from ..engine.hedge_engine import HedgeEngine, SignalItem
from .thread_config import THREAD_CONFIGS, ThreadEmailConfig

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent.parent / ".gmail_state.json"
GMAIL_ACCOUNT = "jordan.deherrera@gmail.com"


class GmailCollector:
    """
    Scans Gmail and ingests signals into Jordan HEDGE.
    
    State is persisted to .gmail_state.json to avoid re-processing emails.
    """

    def __init__(self, engine: HedgeEngine = None, account: str = GMAIL_ACCOUNT):
        self.engine = engine or HedgeEngine()
        self.account = account
        self.state = self._load_state()

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def collect(self) -> dict:
        """
        Run a full collection pass.
        Returns summary of signals ingested per thread.
        """
        summary = {}
        total_signals = 0

        for config in THREAD_CONFIGS:
            signals = self._collect_for_thread(config)
            if signals:
                count = self.engine.ingest_signals(config.thread_name, signals)
                summary[config.thread_name] = {
                    "signals": [s.signal_name for s in signals],
                    "ingested": count,
                }
                total_signals += count
                # Regenerate actions for updated thread
                self.engine.generate_actions(config.thread_name)

        self._save_state()
        logger.info(f"[GmailCollector] Collection complete: {total_signals} signals across {len(summary)} threads")
        return summary

    # ─────────────────────────────────────────────
    # PER-THREAD COLLECTION
    # ─────────────────────────────────────────────

    def _collect_for_thread(self, config: ThreadEmailConfig) -> list[SignalItem]:
        signals = []

        # 1. Check for incoming emails from relevant parties
        inbound = self._search_inbound(config)
        for msg in inbound:
            if self._is_new(msg["id"]):
                sig = self._classify_inbound(msg, config)
                if sig:
                    signals.append(sig)
                    self._mark_seen(msg["id"])

        # 2. Check for unanswered outbound emails
        unanswered = self._check_unanswered_outbound(config)
        if unanswered:
            signals.append(SignalItem(
                signal_name="email_unanswered",
                value="present",
                confidence=0.9,
                source="gmail",
                raw_data={"unanswered_count": len(unanswered),
                          "oldest_sent": unanswered[0].get("date", "") if unanswered else ""},
            ))
            signals.append(SignalItem(
                signal_name="external_party_silent",
                value="present",
                confidence=0.85,
                source="gmail",
                raw_data={"thread": config.thread_name},
            ))
        else:
            # If we previously had unanswered signals and now have a response, ingest resolution
            if self.state.get(f"unanswered_{config.thread_name}"):
                signals.append(SignalItem(
                    signal_name="response_received",
                    value="present",
                    confidence=0.9,
                    source="gmail",
                    raw_data={"thread": config.thread_name, "previously_unanswered": True},
                ))
                self.state.pop(f"unanswered_{config.thread_name}", None)

        # Track unanswered state
        if unanswered:
            self.state[f"unanswered_{config.thread_name}"] = True

        return signals

    # ─────────────────────────────────────────────
    # GMAIL QUERIES
    # ─────────────────────────────────────────────

    def _search_inbound(self, config: ThreadEmailConfig) -> list[dict]:
        """Search inbox for emails relevant to this thread."""
        all_msgs = []
        seen_ids = set()

        for query in config.search_queries:
            # Only search recent mail (last 30 days) to keep it fast
            full_query = f"({query}) newer_than:30d -in:sent"
            msgs = self._gog_search(full_query, limit=10)
            for m in msgs:
                if m["id"] not in seen_ids:
                    all_msgs.append(m)
                    seen_ids.add(m["id"])

        return all_msgs

    def _check_unanswered_outbound(self, config: ThreadEmailConfig) -> list[dict]:
        """
        Find sent emails to expected-reply addresses that haven't gotten a response.
        Strategy: search sent mail to relevant domains > N days old, 
        then check if we've received any reply from those domains since.
        """
        if not config.expected_reply_from:
            return []

        cutoff_days = config.reply_expected_within_days
        sender_query = " OR ".join([f"to:{s}" for s in config.expected_reply_from])
        sent_query = f"in:sent ({sender_query}) older_than:{cutoff_days}d newer_than:60d"

        sent_msgs = self._gog_search(sent_query, limit=5)
        if not sent_msgs:
            return []

        # Check if we've received any reply from these senders recently
        reply_query = f"from:({' OR '.join(config.expected_reply_from)}) newer_than:{cutoff_days}d"
        replies = self._gog_search(reply_query, limit=5)

        if replies:
            # Got a reply — not unanswered
            return []

        return sent_msgs

    def _classify_inbound(self, msg: dict, config: ThreadEmailConfig) -> Optional[SignalItem]:
        """Classify an inbound email into a signal type."""
        sender = (msg.get("from") or "").lower()
        subject = (msg.get("subject") or "").lower()
        labels = msg.get("labels", [])

        # Check if it's from an adversarial/counterparty sender
        is_adversarial = any(
            adv.lower() in sender
            for adv in config.adversarial_senders
        )
        if is_adversarial:
            return SignalItem(
                signal_name="counterparty_active",
                value="present",
                confidence=0.95,
                source="gmail",
                raw_data={"id": msg["id"], "from": msg.get("from"), "subject": msg.get("subject"), "date": msg.get("date")},
            )

        # Check if it's urgent/important
        urgent_keywords = ["urgent", "important", "action required", "response needed",
                           "deadline", "court", "legal", "attorney", "settlement",
                           "denial", "denied", "appeal", "claim", "judgment"]
        is_urgent = any(kw in subject for kw in urgent_keywords)
        is_relevant_sender = any(
            domain.lower() in sender
            for domain in config.relevant_senders
        ) if config.relevant_senders else False

        is_subject_match = any(
            kw.lower() in subject
            for kw in config.subject_keywords
        )

        if is_urgent or (is_relevant_sender and is_subject_match):
            return SignalItem(
                signal_name="email_received_urgent",
                value="present",
                confidence=0.9 if is_urgent else 0.75,
                source="gmail",
                raw_data={"id": msg["id"], "from": msg.get("from"), "subject": msg.get("subject"), "date": msg.get("date")},
            )

        # Generic relevant email — new evidence
        if is_relevant_sender or is_subject_match:
            return SignalItem(
                signal_name="new_evidence_available",
                value="present",
                confidence=0.7,
                source="gmail",
                raw_data={"id": msg["id"], "from": msg.get("from"), "subject": msg.get("subject"), "date": msg.get("date")},
            )

        return None

    # ─────────────────────────────────────────────
    # GOG CLI WRAPPER
    # ─────────────────────────────────────────────

    def _gog_search(self, query: str, limit: int = 20) -> list[dict]:
        """Run a gog gmail search query and return message list."""
        try:
            result = subprocess.run(
                ["gog", "gmail", "messages", "search", query,
                 "-a", self.account, "-j", "--limit", str(limit)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                logger.warning(f"[GmailCollector] gog search failed: {result.stderr[:200]}")
                return []

            data = json.loads(result.stdout)
            return data.get("messages", [])

        except subprocess.TimeoutExpired:
            logger.warning(f"[GmailCollector] gog search timed out for query: {query[:80]}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"[GmailCollector] JSON parse error: {e}")
            return []
        except Exception as e:
            logger.warning(f"[GmailCollector] Unexpected error: {e}")
            return []

    # ─────────────────────────────────────────────
    # STATE MANAGEMENT
    # ─────────────────────────────────────────────

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _is_new(self, msg_id: str) -> bool:
        return msg_id not in self.state.get("seen_ids", [])

    def _mark_seen(self, msg_id: str):
        seen = self.state.setdefault("seen_ids", [])
        if msg_id not in seen:
            seen.append(msg_id)
        # Keep only last 500 to prevent unbounded growth
        self.state["seen_ids"] = seen[-500:]
