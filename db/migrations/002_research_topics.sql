-- Migration 002: add research_topic to next_actions
-- Safe to run multiple times (checks column existence via try/catch in Python runner).

ALTER TABLE next_actions ADD COLUMN research_topic TEXT DEFAULT 'product_strategy';

CREATE INDEX IF NOT EXISTS idx_actions_topic ON next_actions(research_topic, status);
