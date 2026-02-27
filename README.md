# Jordan HEDGE

**Personal reasoning engine for Jordan DeHerrera.**  
A HEDGE-inspired (Hypothesis-Evidence-Driven-Graph-Evaluation) system for maintaining calibrated belief states across life, work, and project threads.

---

## What This Is

Most productivity tools track tasks. This tracks *beliefs about what matters* — with explicit uncertainty.

Inspired by MEL's clinical HEDGE engine, adapted for personal reasoning:
- **Threads** replace patients (estate, malpractice, MEL, health, work)
- **Signals** replace symptoms (deadline proximity, unanswered emails, task staleness)
- **Hypothesis types** replace diagnoses (URGENT_ACTION_NEEDED, AT_RISK, BINDING_CONSTRAINT)
- **Relations** encode how signals update beliefs (LLR_pos, LLR_neg)

Multiple weak signals compound. A thread with a nearby deadline + unanswered email + financial exposure gets a dramatically higher AT_RISK posterior than any single signal alone. That's the Bayesian advantage over a task list.

---

## Architecture

```
jordan-hedge/
  db/schema.sql          — SQLite schema (concepts, relations, facts, hypotheses, threads, actions)
  ontology/seed.json     — Concepts, relations, initial threads
  src/engine/
    db.py                — Connection + custom SQLite functions (logit_to_prob, belief_entropy, etc.)
    hedge_engine.py      — Core engine: ingest signals, update hypotheses, generate actions
  scripts/
    seed.py              — Initialize database from ontology/seed.json
    status.py            — Print current belief state dashboard
```

### Core Loop

1. **Signal ingestion** → facts inserted into SQLite
2. **SQLite trigger fires** → updates log_odds_posterior for all hypotheses on that thread
3. **Probability = sigmoid(log_odds_posterior)** via custom SQLite function
4. **Actions generated** → ordered by `expected_utility = P × risk_weight × (1 - uncertainty)`
5. **Status dashboard** → surfaces what to do right now

---

## Quick Start

```bash
# Initialize and seed
python3 scripts/seed.py

# Generate actions + view belief state
python3 -c "
from src.engine.hedge_engine import HedgeEngine
engine = HedgeEngine()
for t in engine.get_all_threads():
    engine.generate_actions(t.name)
"
python3 scripts/status.py

# Ingest a new signal
python3 -c "
from src.engine.hedge_engine import HedgeEngine, SignalItem
engine = HedgeEngine()
engine.ingest_signals('MetLife Policy Investigation', [
    SignalItem('response_received', 'present', confidence=1.0, source='gmail',
               raw_data={'subject': 'MetLife response re: policy search'})
])
"
```

---

## Ontology

### Hypothesis Types

| Type | Description | Risk Weight |
|------|-------------|-------------|
| URGENT_ACTION_NEEDED | Requires immediate action | 3.0 |
| AT_RISK | Risk of bad outcome without intervention | 2.0 |
| BINDING_CONSTRAINT | Bottleneck blocking other things | 2.5 |
| EXTERNAL_BLOCKED | Waiting on external party | 1.0 |
| DECISION_PENDING | Significant decision needs to be made | 1.5 |
| STALE_THREAD | No activity, needs attention or closure | 0.5 |
| OPPORTUNITY | Actionable opportunity present | 1.5 |

### Signal Types

Signals with strong positive LLRs for `URGENT_ACTION_NEEDED`:
- `deadline_imminent` (llr_pos: 3.0)
- `deadline_missed` (llr_pos: 3.5)
- `counterparty_active` (llr_pos: 2.5)
- `email_received_urgent` (llr_pos: 2.0)

Signals with strong positive LLRs for `AT_RISK`:
- `legal_exposure` (llr_pos: 2.5)
- `financial_exposure_high` (llr_pos: 2.5)
- `counterparty_active` (llr_pos: 2.0)
- `email_unanswered` (llr_pos: 1.5)

---

## Roadmap

- [ ] Gmail collector: ingest signals from email patterns
- [ ] Calendar collector: deadline proximity, appointment signals
- [ ] GitHub collector: stale PRs, deployment status
- [ ] KB collector: pull signals from knowledge base MCP
- [ ] OpenClaw heartbeat integration: run on schedule, surface alerts
- [ ] Dynamic LLR calibration: update weights from outcome feedback
- [ ] Self-expanding ontology: discover new concept types from signal patterns

---

## Relationship to MEL

This engine shares MEL's HEDGE architecture but diverges in domain:
- MEL: clinical ontology, patient threads, diagnostic hypotheses
- Jordan HEDGE: personal ontology, life/work threads, decision/risk hypotheses

The calibration methodology is the same. The moat in both cases is the validated ontology built from real outcomes over time.
