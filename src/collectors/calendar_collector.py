"""
Calendar Collector for Jordan HEDGE.

Scans Google Calendar for events that map to thread signals:
- Event within 48h related to thread → calendar_event_soon + appointment_upcoming
- Deadline-like event within 7 days   → deadline_imminent
- Deadline-like event within 30 days  → deadline_approaching
- Event keyword matching by thread    → thread-specific signals
"""

import json
import subprocess
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from ..engine.hedge_engine import HedgeEngine, SignalItem

logger = logging.getLogger(__name__)

GMAIL_ACCOUNT = "jordan.deherrera@gmail.com"

# How keywords in event titles/descriptions map to threads
THREAD_CALENDAR_KEYWORDS: dict[str, list[str]] = {
    "MetLife Policy Investigation":     ["metlife", "insurance", "estate meeting", "troy anderson"],
    "Estate Inventory Deadline":        ["estate", "inventory", "appraisal", "probate", "pueblo", "court", "troy", "24pr316"],
    "Estate Fraud Investigation":       ["estate", "doreen", "sarah", "fraud", "attorney"],
    "Medical Malpractice — Renown":     ["mowbray", "nomura", "malpractice", "deposition", "mediation", "renown", "settlement"],
    "MEL Cloud Deployment":             ["mel", "patient-talk", "deployment", "sprint", "standup", "crystal demo"],
    "Meniscus Tear Recovery":           ["physical therapy", "pt ", "ryan maves", "ortho", "knee", "bosque", "ways2well", "peptide"],
    "UnitedHealthcare Benefits Optimization": ["uhc", "optum", "insurance", "benefits", "ways2well"],
    "Ridgeline L6 Promotion":           ["ridgeline", "1:1", "one on one", "performance", "review", "l6", "principal", "lattice"],
}

# Event title patterns that suggest a hard deadline (higher urgency signals)
DEADLINE_KEYWORDS = [
    "deadline", "due", "filing", "court", "hearing", "deposition",
    "mediation", "response due", "inventory due", "appeal", "statute"
]


class CalendarCollector:

    def __init__(self, engine: HedgeEngine = None, account: str = GMAIL_ACCOUNT):
        self.engine = engine or HedgeEngine()
        self.account = account

    def collect(self) -> dict:
        """Scan calendar and ingest signals. Returns summary per thread."""
        now = datetime.now(timezone.utc)
        look_ahead_days = 30

        events = self._fetch_events(
            from_dt=now,
            to_dt=now + timedelta(days=look_ahead_days),
        )

        if not events:
            logger.info("[CalendarCollector] No upcoming events found")
            return {}

        logger.info(f"[CalendarCollector] Found {len(events)} upcoming events")

        # Map events to threads
        thread_signals: dict[str, list[SignalItem]] = {}

        for event in events:
            title = (event.get("summary") or "").lower()
            desc = (event.get("description") or "").lower()
            text = f"{title} {desc}"
            event_dt = self._parse_event_dt(event)
            if not event_dt:
                continue

            hours_away = (event_dt - now).total_seconds() / 3600
            days_away = hours_away / 24

            for thread_name, keywords in THREAD_CALENDAR_KEYWORDS.items():
                if not any(kw in text for kw in keywords):
                    continue

                signals = thread_signals.setdefault(thread_name, [])
                is_deadline = any(kw in text for kw in DEADLINE_KEYWORDS)
                raw = {"title": event.get("summary"), "start": event.get("start"), "days_away": round(days_away, 1)}

                if days_away <= 2:
                    signals.append(SignalItem(
                        signal_name="calendar_event_soon",
                        value="present",
                        confidence=0.95,
                        source="calendar",
                        raw_data=raw,
                    ))
                    if "appointment" in text or any(k in text for k in ["therapy", "ortho", "doctor", "clinic", "ways2well", "bosque"]):
                        signals.append(SignalItem(
                            signal_name="appointment_upcoming",
                            value="present",
                            confidence=0.95,
                            source="calendar",
                            raw_data=raw,
                        ))

                if is_deadline:
                    if days_away <= 7:
                        signals.append(SignalItem(
                            signal_name="deadline_imminent",
                            value="present",
                            confidence=0.95,
                            source="calendar",
                            raw_data={**raw, "deadline": True},
                        ))
                    elif days_away <= 30:
                        signals.append(SignalItem(
                            signal_name="deadline_approaching",
                            value="present",
                            confidence=0.9,
                            source="calendar",
                            raw_data={**raw, "deadline": True},
                        ))

        # Ingest into engine
        summary = {}
        for thread_name, signals in thread_signals.items():
            if signals:
                # Deduplicate by signal_name
                seen = set()
                unique = []
                for s in signals:
                    if s.signal_name not in seen:
                        unique.append(s)
                        seen.add(s.signal_name)

                count = (self.engine.ingest_signals(thread_name, unique) or {}).get("processed", 0)
                self.engine.generate_actions(thread_name)
                summary[thread_name] = {
                    "signals": [s.signal_name for s in unique],
                    "ingested": count,
                    "event_count": len([e for e in events if any(
                        kw in (e.get("summary") or "").lower()
                        for kw in THREAD_CALENDAR_KEYWORDS[thread_name]
                    )]),
                }

        logger.info(f"[CalendarCollector] Ingested signals for {len(summary)} threads")
        return summary

    def _fetch_events(self, from_dt: datetime, to_dt: datetime) -> list[dict]:
        """Fetch calendar events via gog CLI."""
        try:
            result = subprocess.run(
                ["gog", "calendar", "events", "primary",
                 "--from", from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "--to",   to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "-a", self.account, "-j"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                logger.warning(f"[CalendarCollector] gog failed: {result.stderr[:200]}")
                return []
            data = json.loads(result.stdout)
            return data.get("events", [])
        except Exception as e:
            logger.warning(f"[CalendarCollector] Error fetching events: {e}")
            return []

    def _parse_event_dt(self, event: dict) -> Optional[datetime]:
        """Parse event start datetime."""
        start = event.get("start", {})
        dt_str = start.get("dateTime") or start.get("date")
        if not dt_str:
            return None
        try:
            # Handle both datetime and all-day date strings
            if "T" in dt_str:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            else:
                # All-day event: treat as midnight UTC
                return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
        except Exception:
            return None
