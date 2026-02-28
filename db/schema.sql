-- Jordan HEDGE: Personal Reasoning Engine
-- Mirrors MEL's HEDGE architecture but for personal/project reasoning

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ─────────────────────────────────────────────
-- ONTOLOGY LAYER
-- ─────────────────────────────────────────────

-- Concepts: the things we reason about and with
CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    type        TEXT NOT NULL,  -- 'domain' | 'signal' | 'hypothesis_type' | 'action_type'
    description TEXT,
    base_rate   REAL DEFAULT 0.1,   -- prior probability
    risk_weight REAL DEFAULT 1.0,   -- stakes multiplier
    data        TEXT DEFAULT '{}'   -- JSON: aliases, metadata
);

-- Relations: how signals update hypothesis beliefs
CREATE TABLE IF NOT EXISTS relations (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_concept_id        INTEGER NOT NULL,
    hypothesis_concept_id    INTEGER NOT NULL,
    llr_pos                  REAL NOT NULL DEFAULT 0.0,   -- log-likelihood if signal present
    llr_neg                  REAL NOT NULL DEFAULT 0.0,   -- log-likelihood if signal absent
    expected_info_gain       REAL DEFAULT 0.0,
    evidence_half_life_hours REAL DEFAULT 0,              -- 0 = no decay
    notes                    TEXT,
    UNIQUE(signal_concept_id, hypothesis_concept_id),
    FOREIGN KEY(signal_concept_id)     REFERENCES concepts(id),
    FOREIGN KEY(hypothesis_concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_relations_signal     ON relations(signal_concept_id);
CREATE INDEX IF NOT EXISTS idx_relations_hypothesis ON relations(hypothesis_concept_id);

-- ─────────────────────────────────────────────
-- EVIDENCE LAYER
-- ─────────────────────────────────────────────

-- Facts: observed signals about the world
CREATE TABLE IF NOT EXISTS facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id      INTEGER NOT NULL,
    concept_id     INTEGER NOT NULL,
    value          TEXT NOT NULL,       -- 'present' | 'absent' | numeric string
    confidence     REAL DEFAULT 1.0,
    source         TEXT NOT NULL,       -- 'gmail' | 'calendar' | 'kb' | 'github' | 'manual' | 'heartbeat'
    raw_data       TEXT DEFAULT '{}',   -- JSON: original signal data
    observed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at     TIMESTAMP,           -- NULL = never expires
    UNIQUE(thread_id, concept_id, value) ON CONFLICT REPLACE,
    FOREIGN KEY(thread_id)  REFERENCES threads(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_facts_thread     ON facts(thread_id);
CREATE INDEX IF NOT EXISTS idx_facts_concept    ON facts(concept_id);
CREATE INDEX IF NOT EXISTS idx_facts_observed   ON facts(observed_at);

-- ─────────────────────────────────────────────
-- HYPOTHESIS LAYER
-- ─────────────────────────────────────────────

-- Threads: persistent belief state per domain/project
CREATE TABLE IF NOT EXISTS threads (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT UNIQUE NOT NULL,
    domain              TEXT NOT NULL,   -- 'estate' | 'malpractice' | 'MEL' | 'ridgeline' | 'health' | 'family' | 'finance'
    description         TEXT,
    status              TEXT DEFAULT 'active',   -- 'active' | 'resolved' | 'paused' | 'watching'
    priority_score      REAL DEFAULT 0.5,
    last_activity_at    TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata            TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_threads_domain   ON threads(domain, status);
CREATE INDEX IF NOT EXISTS idx_threads_priority ON threads(priority_score DESC);

-- Hypotheses: belief state per (thread × hypothesis_type)
CREATE TABLE IF NOT EXISTS hypotheses (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id                INTEGER NOT NULL,
    hypothesis_type_id       INTEGER NOT NULL,   -- FK to concepts where type='hypothesis_type'
    log_odds_prior           REAL NOT NULL,
    log_odds_posterior       REAL NOT NULL,
    alpha                    REAL DEFAULT 1.0,   -- supporting evidence count (Beta distribution)
    beta                     REAL DEFAULT 1.0,   -- contradicting evidence count
    status                   TEXT DEFAULT 'active',   -- 'active' | 'confirmed' | 'dismissed'
    last_evidence_at         TIMESTAMP,
    updated_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Phase 5: validation thresholds (NULL = use defaults: support=+3.0, contradict=-3.0)
    llr_support_threshold    REAL    DEFAULT NULL,
    llr_contradict_threshold REAL    DEFAULT NULL,
    min_evidence_count       INTEGER DEFAULT 1,
    validation_warnings      TEXT    DEFAULT NULL,  -- JSON array of non-blocking warnings
    UNIQUE(thread_id, hypothesis_type_id),
    FOREIGN KEY(thread_id)          REFERENCES threads(id),
    FOREIGN KEY(hypothesis_type_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_thread ON hypotheses(thread_id, status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_updated ON hypotheses(updated_at DESC);
-- Phase 5: resolution and gap-detection indexes
CREATE INDEX IF NOT EXISTS idx_hypotheses_thresholds
    ON hypotheses(thread_id, llr_support_threshold, llr_contradict_threshold);
CREATE INDEX IF NOT EXISTS idx_hypotheses_missing_thresholds
    ON hypotheses(thread_id)
    WHERE llr_support_threshold IS NULL OR llr_contradict_threshold IS NULL;

-- ─────────────────────────────────────────────
-- ACTION LAYER
-- ─────────────────────────────────────────────

-- Next actions: what the engine recommends Jordan do
CREATE TABLE IF NOT EXISTS next_actions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id        INTEGER NOT NULL,
    hypothesis_id    INTEGER,
    action_type      TEXT NOT NULL,    -- 'notify' | 'follow_up' | 'escalate' | 'research' | 'decide'
    title            TEXT NOT NULL,
    description      TEXT,
    expected_utility REAL DEFAULT 0.0,
    urgency_score    REAL DEFAULT 0.0,
    status           TEXT DEFAULT 'pending',   -- 'pending' | 'actioned' | 'dismissed' | 'snoozed'
    due_by           TIMESTAMP,
    actioned_at      TIMESTAMP,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata         TEXT DEFAULT '{}',
    FOREIGN KEY(thread_id)     REFERENCES threads(id),
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id)
);

CREATE INDEX IF NOT EXISTS idx_actions_thread   ON next_actions(thread_id, status);
CREATE INDEX IF NOT EXISTS idx_actions_utility  ON next_actions(expected_utility DESC, status);
CREATE INDEX IF NOT EXISTS idx_actions_due      ON next_actions(due_by, status);

-- ─────────────────────────────────────────────
-- CUSTOM MATH FUNCTIONS (registered in Python)
-- ─────────────────────────────────────────────
-- logit_to_prob(log_odds) → 1 / (1 + exp(-log_odds))
-- prob_to_logit(p)        → log(p / (1 - p))
-- decay(dt_hours, half_life_hours) → exp(-0.693 * dt / half_life)  [0 half_life = no decay]

-- ─────────────────────────────────────────────
-- TRIGGERS: auto-update hypotheses on fact insert
-- ─────────────────────────────────────────────

CREATE TRIGGER IF NOT EXISTS facts_update_hypotheses
AFTER INSERT ON facts
BEGIN
    -- Update all hypotheses for this thread where a relation exists
    UPDATE hypotheses
    SET
        log_odds_posterior = log_odds_posterior + COALESCE((
            SELECT
                CASE
                    WHEN LOWER(NEW.value) IN ('present','yes','true','high','elevated','overdue','unanswered','imminent')
                        THEN r.llr_pos * NEW.confidence
                    WHEN LOWER(NEW.value) IN ('absent','no','false','low','normal','resolved','answered')
                        THEN r.llr_neg * NEW.confidence
                    ELSE 0.0
                END
            FROM relations r
            WHERE r.signal_concept_id     = NEW.concept_id
              AND r.hypothesis_concept_id = hypotheses.hypothesis_type_id
        ), 0.0),
        alpha = alpha + CASE
            WHEN LOWER(NEW.value) IN ('present','yes','true','high','elevated','overdue','unanswered','imminent') THEN 1
            ELSE 0
        END,
        beta = beta + CASE
            WHEN LOWER(NEW.value) IN ('absent','no','false','low','normal','resolved','answered') THEN 1
            ELSE 0
        END,
        last_evidence_at = NEW.observed_at,
        updated_at = CURRENT_TIMESTAMP
    WHERE hypotheses.thread_id = NEW.thread_id
      AND hypotheses.status = 'active';
END;

-- Update thread priority score when hypothesis changes
CREATE TRIGGER IF NOT EXISTS hypotheses_update_thread_priority
AFTER UPDATE ON hypotheses
BEGIN
    UPDATE threads
    SET
        priority_score = (
            SELECT MAX(logit_to_prob(log_odds_posterior) * c.risk_weight)
            FROM hypotheses h
            JOIN concepts c ON c.id = h.hypothesis_type_id
            WHERE h.thread_id = NEW.thread_id
              AND h.status = 'active'
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.thread_id;
END;
