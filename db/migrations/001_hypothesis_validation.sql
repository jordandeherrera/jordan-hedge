-- HEDGE Phase 5: Hypothesis Validation Layer
-- Adds per-hypothesis LLR thresholds and minimum evidence requirements.
--
-- These columns enforce modeling discipline at the schema level:
--   llr_support_threshold:    cumulative LLR to consider hypothesis "supported"
--   llr_contradict_threshold: cumulative LLR to effectively dismiss hypothesis
--   min_evidence_count:       minimum independent signals before resolving
--   validation_warnings:      JSON array of non-blocking warnings from creation
--
-- NULL = threshold not declared; engine applies defaults
-- (support=+3.0, contradict=-3.0, min_evidence_count=1)

ALTER TABLE hypotheses ADD COLUMN llr_support_threshold    REAL    DEFAULT NULL;
ALTER TABLE hypotheses ADD COLUMN llr_contradict_threshold REAL    DEFAULT NULL;
ALTER TABLE hypotheses ADD COLUMN min_evidence_count       INTEGER DEFAULT 1;
ALTER TABLE hypotheses ADD COLUMN validation_warnings      TEXT    DEFAULT NULL;  -- JSON array

-- Index for resolution queries (hypotheses that have crossed a threshold)
CREATE INDEX IF NOT EXISTS idx_hypotheses_thresholds
    ON hypotheses(thread_id, llr_support_threshold, llr_contradict_threshold);

-- Partial index: hypotheses with undeclared thresholds (modeling gap detection)
CREATE INDEX IF NOT EXISTS idx_hypotheses_missing_thresholds
    ON hypotheses(thread_id)
    WHERE llr_support_threshold IS NULL OR llr_contradict_threshold IS NULL;
