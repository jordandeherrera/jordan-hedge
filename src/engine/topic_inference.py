"""
Topic Inference — maps (domain, hypothesis_type, action_type) → research_topic.

Research topics are the underlying knowledge work required to act on a thread.
They are orthogonal to domain: Advocate (advocate domain) and MEL Prior Auth
Voice Agent (MEL domain) both need 'competitive_landscape' research. That
cross-domain signal is what drives batch opportunities the embedding backend
alone cannot see.

Topic taxonomy:
    competitive_landscape   — market research, competitor analysis, positioning
    regulatory_environment  — insurance regs, legal compliance, filings
    clinical_literature     — medical evidence, treatment protocols, ontology
    technical_architecture  — system design, infra, implementation decisions
    product_strategy        — GTM, pricing, roadmap, entity structure decisions
    stakeholder_outreach    — follow-ups, escalations to external parties
    legal_proceedings       — court filings, attorney coordination, evidence
    financial_modeling      — pricing, unit economics, benefits analysis
    personal_health         — medical decisions, recovery, treatment tracking
    infrastructure_ops      — deployment, cloud, DevOps, system reliability
    knowledge_capture       — documentation, open-source prep, write-up

Thread-level overrides take precedence over inferred defaults.
Set in thread.metadata as {"research_topics": ["competitive_landscape", ...]}.
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────
# DOMAIN DEFAULTS
# Base research topic when a domain doesn't have a more specific rule.
# ─────────────────────────────────────────────
DOMAIN_DEFAULTS: dict[str, str] = {
    "MEL":          "product_strategy",
    "advocate":     "competitive_landscape",
    "estate":       "legal_proceedings",
    "finance":      "financial_modeling",
    "health":       "personal_health",
    "hedge":        "knowledge_capture",
    "malpractice":  "legal_proceedings",
    "ridgeline":    "product_strategy",
}

# ─────────────────────────────────────────────
# HYPOTHESIS × ACTION_TYPE OVERRIDES
# Specific (domain, hypothesis_type, action_type) triples that override
# the domain default. Add more as threads evolve.
# ─────────────────────────────────────────────
SPECIFIC_RULES: dict[tuple[str, str, str], str] = {
    # MEL
    ("MEL", "BINDING_CONSTRAINT",    "escalate"):   "stakeholder_outreach",
    ("MEL", "EXTERNAL_BLOCKED",      "follow_up"):  "stakeholder_outreach",
    ("MEL", "OPPORTUNITY_ACTIVE",    "decide"):     "product_strategy",
    ("MEL", "OPPORTUNITY_EMERGING",  "research"):   "competitive_landscape",
    ("MEL", "STALE_THREAD",          "research"):   "infrastructure_ops",

    # Advocate
    ("advocate", "OPPORTUNITY_ACTIVE",   "decide"):    "product_strategy",
    ("advocate", "OPPORTUNITY_EMERGING", "research"):  "competitive_landscape",
    ("advocate", "DECISION_PENDING",     "decide"):    "product_strategy",

    # HEDGE open source
    ("hedge", "OPPORTUNITY_ACTIVE",   "decide"):    "knowledge_capture",
    ("hedge", "OPPORTUNITY_EMERGING", "research"):  "knowledge_capture",
    ("hedge", "BINDING_CONSTRAINT",   "escalate"):  "technical_architecture",

    # Ridgeline
    ("ridgeline", "OPPORTUNITY_ACTIVE",   "decide"):   "product_strategy",
    ("ridgeline", "OPPORTUNITY_EMERGING", "research"): "competitive_landscape",
    ("ridgeline", "BINDING_CONSTRAINT",   "escalate"): "stakeholder_outreach",
    ("ridgeline", "STALE_THREAD",         "research"): "product_strategy",

    # Estate
    ("estate", "BINDING_CONSTRAINT",  "escalate"):  "stakeholder_outreach",
    ("estate", "EXTERNAL_BLOCKED",    "follow_up"): "stakeholder_outreach",
    ("estate", "AT_RISK",             "follow_up"): "legal_proceedings",
    ("estate", "URGENT_ACTION_NEEDED","decide"):    "legal_proceedings",
    ("estate", "DECISION_PENDING",    "decide"):    "legal_proceedings",

    # Malpractice
    ("malpractice", "BINDING_CONSTRAINT",   "escalate"):  "stakeholder_outreach",
    ("malpractice", "EXTERNAL_BLOCKED",     "follow_up"): "stakeholder_outreach",
    ("malpractice", "AT_RISK",              "follow_up"): "legal_proceedings",
    ("malpractice", "URGENT_ACTION_NEEDED", "decide"):    "legal_proceedings",

    # Health
    ("health", "DECISION_PENDING",    "decide"):    "clinical_literature",
    ("health", "OPPORTUNITY_ACTIVE",  "decide"):    "clinical_literature",
    ("health", "OPPORTUNITY_CLOSED",  "follow_up"): "personal_health",

    # Finance
    ("finance", "URGENT_ACTION_NEEDED", "decide"):  "financial_modeling",
    ("finance", "DECISION_PENDING",     "decide"):  "financial_modeling",
    ("finance", "BINDING_CONSTRAINT",   "escalate"):"stakeholder_outreach",
}

# ─────────────────────────────────────────────
# THREAD-NAME LEVEL OVERRIDES
# For threads where the name alone tells us the right topic regardless
# of hypothesis type. Checked before domain defaults.
# ─────────────────────────────────────────────
THREAD_NAME_OVERRIDES: dict[str, str] = {
    "MEL Cloud Deployment":             "infrastructure_ops",
    "MEL Ontology Quality":             "clinical_literature",
    "MEL Prior Auth Voice Agent":       "competitive_landscape",
    "Advocate — Patient Navigation AI": "competitive_landscape",
    "HEDGE Open Source":                "knowledge_capture",
    "Ridgeline Knowledge Graph":        "technical_architecture",
    "Ridgeline L6 Promotion":           "product_strategy",
    "Medical Malpractice — Renown":     "legal_proceedings",
    "Estate Inventory Deadline":        "legal_proceedings",
    "Estate Fraud Investigation":       "legal_proceedings",
    "MetLife Policy Investigation":     "legal_proceedings",
    "Meniscus Tear Recovery":           "personal_health",
    "UnitedHealthcare Benefits Optimization": "financial_modeling",
}

# ─────────────────────────────────────────────
# TOPICS THAT BENEFIT FROM CROSS-DOMAIN BATCHING
# When multiple threads share one of these topics, the research
# substrate is truly shared (not just within-thread).
# ─────────────────────────────────────────────
CROSS_DOMAIN_BATCHABLE: set[str] = {
    "competitive_landscape",
    "regulatory_environment",
    "clinical_literature",
    "technical_architecture",
    "knowledge_capture",
    "product_strategy",
}

# Topics that are highly specific to a single thread/context — batch savings
# are mostly within-thread, not across threads.
THREAD_SPECIFIC_TOPICS: set[str] = {
    "legal_proceedings",
    "personal_health",
    "financial_modeling",
    "infrastructure_ops",
    "stakeholder_outreach",
}


def infer_topic(
    domain: str,
    hypothesis_type: str,
    action_type: str,
    thread_name: str = "",
    thread_metadata: Optional[dict] = None,
) -> str:
    """
    Return the research topic for an action.
    Priority order:
    1. Thread metadata override (most specific)
    2. Thread name override
    3. Specific (domain, hypothesis_type, action_type) rule
    4. Domain default
    5. Fallback: 'product_strategy'
    """
    # 1. Thread metadata
    if thread_metadata:
        topics = thread_metadata.get("research_topics", [])
        if topics:
            return topics[0]  # primary topic

    # 2. Thread name override
    if thread_name and thread_name in THREAD_NAME_OVERRIDES:
        return THREAD_NAME_OVERRIDES[thread_name]

    # 3. Specific rule
    key = (domain, hypothesis_type, action_type)
    if key in SPECIFIC_RULES:
        return SPECIFIC_RULES[key]

    # 4. Domain default
    if domain in DOMAIN_DEFAULTS:
        return DOMAIN_DEFAULTS[domain]

    # 5. Fallback
    return "product_strategy"


def is_cross_domain_batchable(topic: str) -> bool:
    return topic in CROSS_DOMAIN_BATCHABLE


ALL_TOPICS: list[str] = [
    "competitive_landscape",
    "regulatory_environment",
    "clinical_literature",
    "technical_architecture",
    "product_strategy",
    "stakeholder_outreach",
    "legal_proceedings",
    "financial_modeling",
    "personal_health",
    "infrastructure_ops",
    "knowledge_capture",
]

TOPIC_LABELS: dict[str, str] = {
    "competitive_landscape":  "Competitive Landscape",
    "regulatory_environment": "Regulatory / Legal Compliance",
    "clinical_literature":    "Clinical Literature & Ontology",
    "technical_architecture": "Technical Architecture",
    "product_strategy":       "Product Strategy & GTM",
    "stakeholder_outreach":   "Stakeholder Outreach",
    "legal_proceedings":      "Legal Proceedings",
    "financial_modeling":     "Financial Modeling",
    "personal_health":        "Personal Health",
    "infrastructure_ops":     "Infrastructure & Ops",
    "knowledge_capture":      "Knowledge Capture & Documentation",
}
