"""
Jordan HEDGE Engine
Core reasoning engine: ingests signals, updates hypotheses, surfaces next actions.
Mirrors MEL's HedgeEngine but for personal/project reasoning.
"""

import json
import math
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from .db import get_connection, logit_to_prob, prob_to_logit, belief_entropy, uncertainty
from .topic_inference import infer_topic
from .hypothesis_validator import (
    HypothesisValidator,
    HypothesisValidationError,
    EvidenceAnomalySignal,
    check_signal_impact,
    DEFAULT_LLR_SUPPORT_THRESHOLD,
    DEFAULT_LLR_CONTRADICT_THRESHOLD,
    DEFAULT_MIN_EVIDENCE_COUNT,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalItem:
    """A piece of evidence about a thread."""
    signal_name: str
    value: str              # 'present' | 'absent' | numeric string
    confidence: float = 1.0
    source: str = "manual"
    raw_data: dict = field(default_factory=dict)
    observed_at: Optional[str] = None


@dataclass
class HypothesisResult:
    """Current belief state for one hypothesis on one thread."""
    hypothesis_type: str
    probability: float
    log_odds_posterior: float
    entropy: float
    uncertainty: float
    status: str
    last_evidence_at: Optional[str]
    risk_weight: float
    # Phase 5: threshold fields (None = defaults applied)
    llr_support_threshold: Optional[float] = None
    llr_contradict_threshold: Optional[float] = None
    min_evidence_count: int = DEFAULT_MIN_EVIDENCE_COUNT
    validation_warnings: Optional[list] = None

    @property
    def resolution_state(self) -> str:
        """
        Derived resolution state based on thresholds:
          'supported'    — posterior >= llr_support_threshold
          'contradicted' — posterior <= llr_contradict_threshold
          'open'         — neither threshold reached
        """
        support    = self.llr_support_threshold    or DEFAULT_LLR_SUPPORT_THRESHOLD
        contradict = self.llr_contradict_threshold or DEFAULT_LLR_CONTRADICT_THRESHOLD
        if self.log_odds_posterior >= support:
            return 'supported'
        if self.log_odds_posterior <= contradict:
            return 'contradicted'
        return 'open'


@dataclass
class ThreadState:
    """Full belief state for a thread."""
    thread_id: int
    name: str
    domain: str
    status: str
    priority_score: float
    hypotheses: list[HypothesisResult]
    top_hypothesis: Optional[HypothesisResult]
    next_actions: list[dict]


@dataclass
class NextAction:
    """A recommended action for Jordan."""
    thread_name: str
    action_type: str
    title: str
    description: str
    expected_utility: float
    urgency_score: float
    due_by: Optional[str]


class HedgeEngine:
    """
    Personal reasoning engine for Jordan.
    
    Maintains probabilistic belief states across life/work threads.
    Updates beliefs from signals (emails, calendar, deadlines, tasks).
    Surfaces next actions ordered by expected_utility.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def _conn(self):
        return get_connection(self.db_path)

    # ─────────────────────────────────────────────
    # SIGNAL INGESTION
    # ─────────────────────────────────────────────

    def ingest_signals(
        self,
        thread_name: str,
        signals: list[SignalItem],
    ) -> dict:
        """
        Ingest evidence signals for a thread.
        SQLite trigger automatically updates hypotheses on fact insert.

        Phase 5: before each insert, checks whether the signal has any
        meaningful LLR impact on the active hypothesis set. Signals with
        near-zero impact are surfaced as EvidenceAnomalySignals — the
        curiosity loop signal indicating a hypothesis set gap.

        Returns:
            {
                "processed": int,
                "anomalies": list[EvidenceAnomalySignal],
            }
        """
        conn = self._conn()
        try:
            thread = self._get_thread(conn, thread_name)
            if not thread:
                logger.warning(f"[HedgeEngine] Thread not found: {thread_name}")
                return {"processed": 0, "anomalies": []}

            thread_id = thread["id"]
            processed = 0
            anomalies: list[EvidenceAnomalySignal] = []

            for signal in signals:
                concept = self._resolve_concept(conn, signal.signal_name)
                if not concept:
                    logger.warning(f"[HedgeEngine] Signal concept not found: {signal.signal_name}")
                    continue

                observed_at = signal.observed_at or datetime.now(timezone.utc).isoformat()

                # ── Phase 5: curiosity loop anomaly check ────────────────────
                anomaly = check_signal_impact(
                    conn=conn,
                    thread_id=thread_id,
                    thread_name=thread_name,
                    concept_id=concept["id"],
                    signal_name=signal.signal_name,
                    value=signal.value,
                )
                if anomaly:
                    anomalies.append(anomaly)
                    logger.warning(f"[HedgeEngine] Anomaly: {anomaly.message}")
                # ─────────────────────────────────────────────────────────────

                conn.execute("""
                    INSERT OR REPLACE INTO facts
                        (thread_id, concept_id, value, confidence, source, raw_data, observed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    thread_id,
                    concept["id"],
                    signal.value,
                    signal.confidence,
                    signal.source,
                    json.dumps(signal.raw_data),
                    observed_at,
                ))
                processed += 1

            conn.execute(
                "UPDATE threads SET last_activity_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), thread_id)
            )
            conn.commit()

            msg = f"[HedgeEngine] Ingested {processed} signals for thread '{thread_name}'"
            if anomalies:
                msg += f", {len(anomalies)} anomaly signal(s) detected"
            logger.info(msg)

            self._last_anomalies = anomalies  # available via last_anomalies property
            return {"processed": processed, "anomalies": anomalies}

        finally:
            conn.close()

    @property
    def last_anomalies(self) -> list[EvidenceAnomalySignal]:
        """Anomalies from the most recent ingest_signals call."""
        return getattr(self, '_last_anomalies', [])

    # ─────────────────────────────────────────────
    # HYPOTHESIS RETRIEVAL
    # ─────────────────────────────────────────────

    def get_thread_state(self, thread_name: str) -> Optional[ThreadState]:
        """Get full belief state for a thread."""
        conn = self._conn()
        try:
            thread = self._get_thread(conn, thread_name)
            if not thread:
                return None

            hypotheses = self._get_hypotheses(conn, thread["id"])
            next_actions = self._get_next_actions(conn, thread["id"])

            top = max(hypotheses, key=lambda h: h.probability * h.risk_weight) if hypotheses else None

            return ThreadState(
                thread_id=thread["id"],
                name=thread["name"],
                domain=thread["domain"],
                status=thread["status"],
                priority_score=thread["priority_score"] or 0.0,
                hypotheses=hypotheses,
                top_hypothesis=top,
                next_actions=next_actions,
            )
        finally:
            conn.close()

    def get_all_threads(self, domain: str = None, min_priority: float = 0.0) -> list[ThreadState]:
        """Get all active threads, optionally filtered."""
        conn = self._conn()
        try:
            query = "SELECT * FROM threads WHERE status = 'active'"
            params = []
            if domain:
                query += " AND domain = ?"
                params.append(domain)
            query += " AND priority_score >= ? ORDER BY priority_score DESC"
            params.append(min_priority)

            rows = conn.execute(query, params).fetchall()
            states = []
            for row in rows:
                hypotheses = self._get_hypotheses(conn, row["id"])
                next_actions = self._get_next_actions(conn, row["id"])
                top = max(hypotheses, key=lambda h: h.probability * h.risk_weight) if hypotheses else None
                states.append(ThreadState(
                    thread_id=row["id"],
                    name=row["name"],
                    domain=row["domain"],
                    status=row["status"],
                    priority_score=row["priority_score"] or 0.0,
                    hypotheses=hypotheses,
                    top_hypothesis=top,
                    next_actions=next_actions,
                ))
            return states
        finally:
            conn.close()

    def get_priority_actions(self, limit: int = 10) -> list[NextAction]:
        """
        Get highest-utility next actions across all threads.
        This is the main output surface: what should Jordan do right now?
        """
        conn = self._conn()
        try:
            rows = conn.execute("""
                SELECT
                    na.*,
                    t.name as thread_name,
                    t.domain
                FROM next_actions na
                JOIN threads t ON t.id = na.thread_id
                WHERE na.status = 'pending'
                ORDER BY na.expected_utility DESC, na.urgency_score DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [NextAction(
                thread_name=row["thread_name"],
                action_type=row["action_type"],
                title=row["title"],
                description=row["description"] or "",
                expected_utility=row["expected_utility"],
                urgency_score=row["urgency_score"],
                due_by=row["due_by"],
            ) for row in rows]
        finally:
            conn.close()

    # ─────────────────────────────────────────────
    # ACTION GENERATION
    # ─────────────────────────────────────────────

    def generate_actions(self, thread_name: str) -> list[dict]:
        """
        Generate next actions for a thread based on current belief state.
        Stores them in next_actions table.
        """
        conn = self._conn()
        try:
            thread = self._get_thread(conn, thread_name)
            if not thread:
                return []

            hypotheses = self._get_hypotheses(conn, thread["id"])
            if not hypotheses:
                return []

            actions = []
            for hyp in hypotheses:
                if hyp.probability < 0.3 or hyp.status != 'active':
                    continue

                action = self._hypothesis_to_action(thread, hyp)
                if not action:
                    continue

                # expected_utility = probability × risk_weight × (1 - uncertainty) / cost
                # cost is normalized to 1.0 for now
                expected_utility = hyp.probability * hyp.risk_weight * (1.0 - hyp.uncertainty * 0.5)
                urgency = hyp.probability * hyp.risk_weight

                # Infer research topic from domain × hypothesis type × action type
                import json as _json
                _meta = {}
                try:
                    _meta = _json.loads(thread["metadata"] or "{}")
                except Exception:
                    pass
                research_topic = infer_topic(
                    domain=thread["domain"],
                    hypothesis_type=hyp.hypothesis_type,
                    action_type=action["type"],
                    thread_name=thread["name"],
                    thread_metadata=_meta,
                )

                conn.execute("""
                    INSERT OR IGNORE INTO next_actions
                        (thread_id, action_type, title, description,
                         expected_utility, urgency_score, status, research_topic)
                    SELECT ?, ?, ?, ?, ?, ?, 'pending', ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM next_actions
                        WHERE thread_id = ? AND action_type = ? AND status = 'pending'
                    )
                """, (
                    thread["id"],
                    action["type"],
                    action["title"],
                    action["description"],
                    expected_utility,
                    urgency,
                    research_topic,
                    thread["id"],
                    action["type"],
                ))
                actions.append(action)

            conn.commit()
            return actions
        finally:
            conn.close()

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _get_thread(self, conn, name: str):
        return conn.execute(
            "SELECT * FROM threads WHERE name = ?", (name,)
        ).fetchone()

    def _resolve_concept(self, conn, name: str):
        """Resolve signal name to concept, case-insensitive with partial fallback."""
        row = conn.execute(
            "SELECT * FROM concepts WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM concepts WHERE LOWER(name) LIKE LOWER(?) LIMIT 1",
                (f"%{name}%",)
            ).fetchone()
        return row

    def _get_hypotheses(self, conn, thread_id: int, status: str = 'active') -> list[HypothesisResult]:
        rows = conn.execute("""
            SELECT
                h.*,
                c.name as hypothesis_type,
                c.risk_weight,
                logit_to_prob(h.log_odds_posterior) as probability,
                belief_entropy(h.log_odds_posterior) as entropy,
                uncertainty(h.log_odds_posterior) as uncertainty_score
            FROM hypotheses h
            JOIN concepts c ON c.id = h.hypothesis_type_id
            WHERE h.thread_id = ? AND h.status = ?
            ORDER BY logit_to_prob(h.log_odds_posterior) * c.risk_weight DESC
        """, (thread_id, status)).fetchall()

        return [HypothesisResult(
            hypothesis_type=row["hypothesis_type"],
            probability=row["probability"],
            log_odds_posterior=row["log_odds_posterior"],
            entropy=row["entropy"],
            uncertainty=row["uncertainty_score"],
            status=row["status"],
            last_evidence_at=row["last_evidence_at"],
            risk_weight=row["risk_weight"],
            llr_support_threshold=row["llr_support_threshold"],
            llr_contradict_threshold=row["llr_contradict_threshold"],
            min_evidence_count=row["min_evidence_count"] or DEFAULT_MIN_EVIDENCE_COUNT,
            validation_warnings=json.loads(row["validation_warnings"]) if row["validation_warnings"] else None,
        ) for row in rows]

    def get_resolved_hypotheses(self, thread_name: str) -> list[HypothesisResult]:
        """
        Phase 5: Return hypotheses that have crossed a resolution threshold
        (supported or contradicted). Used to surface converged belief states.
        """
        conn = self._conn()
        try:
            thread = self._get_thread(conn, thread_name)
            if not thread:
                return []

            rows = conn.execute(f"""
                SELECT
                    h.*,
                    c.name as hypothesis_type,
                    c.risk_weight,
                    logit_to_prob(h.log_odds_posterior) as probability,
                    belief_entropy(h.log_odds_posterior) as entropy,
                    uncertainty(h.log_odds_posterior) as uncertainty_score
                FROM hypotheses h
                JOIN concepts c ON c.id = h.hypothesis_type_id
                WHERE h.thread_id = ?
                  AND h.status = 'active'
                  AND (
                    h.log_odds_posterior >= COALESCE(h.llr_support_threshold, {DEFAULT_LLR_SUPPORT_THRESHOLD})
                    OR
                    h.log_odds_posterior <= COALESCE(h.llr_contradict_threshold, {DEFAULT_LLR_CONTRADICT_THRESHOLD})
                  )
                ORDER BY ABS(h.log_odds_posterior) DESC
            """, (thread["id"],)).fetchall()

            return [HypothesisResult(
                hypothesis_type=row["hypothesis_type"],
                probability=row["probability"],
                log_odds_posterior=row["log_odds_posterior"],
                entropy=row["entropy"],
                uncertainty=row["uncertainty_score"],
                status=row["status"],
                last_evidence_at=row["last_evidence_at"],
                risk_weight=row["risk_weight"],
                llr_support_threshold=row["llr_support_threshold"],
                llr_contradict_threshold=row["llr_contradict_threshold"],
                min_evidence_count=row["min_evidence_count"] or DEFAULT_MIN_EVIDENCE_COUNT,
                validation_warnings=json.loads(row["validation_warnings"]) if row["validation_warnings"] else None,
            ) for row in rows]
        finally:
            conn.close()

    def _get_next_actions(self, conn, thread_id: int) -> list[dict]:
        rows = conn.execute("""
            SELECT * FROM next_actions
            WHERE thread_id = ? AND status = 'pending'
            ORDER BY expected_utility DESC
            LIMIT 5
        """, (thread_id,)).fetchall()
        return [dict(row) for row in rows]

    def _hypothesis_to_action(self, thread, hyp: HypothesisResult) -> Optional[dict]:
        """Map a high-probability hypothesis to a concrete action."""
        h = hyp.hypothesis_type
        t = dict(thread)

        templates = {
            "URGENT_ACTION_NEEDED": {
                "type": "decide",
                "title": f"⚠️ Urgent: {t['name']}",
                "description": f"Belief state indicates urgent action required on '{t['name']}' (P={hyp.probability:.0%}). Review and act.",
            },
            "BINDING_CONSTRAINT": {
                "type": "escalate",
                "title": f"🔒 Unblock: {t['name']}",
                "description": f"'{t['name']}' is likely a binding constraint (P={hyp.probability:.0%}). Identify what's blocking and remove it.",
            },
            "AT_RISK": {
                "type": "follow_up",
                "title": f"🚨 At Risk: {t['name']}",
                "description": f"'{t['name']}' shows elevated risk signals (P={hyp.probability:.0%}). Review exposure and take protective action.",
            },
            "EXTERNAL_BLOCKED": {
                "type": "follow_up",
                "title": f"📬 Follow Up: {t['name']}",
                "description": f"'{t['name']}' appears blocked on external party (P={hyp.probability:.0%}). Send follow-up or escalate.",
            },
            "STALE_THREAD": {
                "type": "research",
                "title": f"💤 Stale: {t['name']}",
                "description": f"'{t['name']}' has gone quiet (P={hyp.probability:.0%}). Review status and either advance or close.",
            },
            "DECISION_PENDING": {
                "type": "decide",
                "title": f"🤔 Decision Needed: {t['name']}",
                "description": f"'{t['name']}' has a pending decision (P={hyp.probability:.0%}). Gather remaining evidence and decide.",
            },
            "OPPORTUNITY_ACTIVE": {
                "type": "decide",
                "title": f"✨ Act Now: {t['name']}",
                "description": f"'{t['name']}' has an open opportunity window (P={hyp.probability:.0%}). Act before it closes.",
            },
            "OPPORTUNITY_EMERGING": {
                "type": "research",
                "title": f"🌱 Emerging: {t['name']}",
                "description": f"'{t['name']}' shows accumulating positive signals (P={hyp.probability:.0%}). Monitor and prepare to act.",
            },
            "OPPORTUNITY_CLOSED": {
                "type": "follow_up",
                "title": f"🔒 Close or Archive: {t['name']}",
                "description": f"'{t['name']}' opportunity appears resolved or passed (P={hyp.probability:.0%}). Review for closure.",
            },
        }

        return templates.get(h)

    # ─────────────────────────────────────────────
    # RESEARCH TOPIC BACKFILL
    # ─────────────────────────────────────────────

    def backfill_research_topics(self) -> int:
        """
        Backfill research_topic on existing pending actions that have NULL or
        default 'product_strategy' values. Safe to call repeatedly.
        Returns the number of rows updated.
        """
        import json as _json
        conn = self._conn()
        try:
            rows = conn.execute("""
                SELECT
                    na.id,
                    na.action_type,
                    na.research_topic,
                    t.domain,
                    t.name   AS thread_name,
                    t.metadata,
                    h.name   AS hypothesis_type
                FROM next_actions na
                JOIN threads t ON t.id = na.thread_id
                LEFT JOIN (
                    -- Best matching hypothesis: highest probability for this thread
                    SELECT h2.thread_id, c.name
                    FROM hypotheses h2
                    JOIN concepts c ON c.id = h2.hypothesis_type_id
                    WHERE h2.status = 'active'
                    GROUP BY h2.thread_id
                    HAVING MAX(logit_to_prob(h2.log_odds_posterior) * c.risk_weight)
                ) h ON h.thread_id = na.thread_id
                WHERE na.status = 'pending'
            """).fetchall()

            updated = 0
            for r in rows:
                meta = {}
                try:
                    meta = _json.loads(r["metadata"] or "{}")
                except Exception:
                    pass
                topic = infer_topic(
                    domain=r["domain"],
                    hypothesis_type=r["hypothesis_type"] or "",
                    action_type=r["action_type"],
                    thread_name=r["thread_name"],
                    thread_metadata=meta,
                )
                if topic != (r["research_topic"] or "product_strategy"):
                    conn.execute(
                        "UPDATE next_actions SET research_topic = ? WHERE id = ?",
                        (topic, r["id"])
                    )
                    updated += 1
                elif r["research_topic"] is None:
                    conn.execute(
                        "UPDATE next_actions SET research_topic = ? WHERE id = ?",
                        (topic, r["id"])
                    )
                    updated += 1

            conn.commit()
            return updated
        finally:
            conn.close()

    # ─────────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────────

    def health(self) -> dict:
        conn = self._conn()
        try:
            threads = conn.execute("SELECT COUNT(*) FROM threads WHERE status='active'").fetchone()[0]
            facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            hypotheses = conn.execute("SELECT COUNT(*) FROM hypotheses WHERE status='active'").fetchone()[0]
            actions = conn.execute("SELECT COUNT(*) FROM next_actions WHERE status='pending'").fetchone()[0]
            return {
                "active_threads": threads,
                "facts": facts,
                "active_hypotheses": hypotheses,
                "pending_actions": actions,
                "status": "healthy",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            conn.close()
