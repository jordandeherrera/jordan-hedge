"""
Convergence detector: determines which HEDGE threads are ready to dispatch to RALPH.

A thread is "executable" when:
1. It has a non-empty next_actions list
2. The top action (or thread research_topic) is implementable by a coding agent
3. Priority score >= threshold
4. Not blocked by signals that require human decision/external response

The "implementable" check runs on:
  (a) The action description text (heuristic keyword match)
  (b) The thread's research_topic field (maps topic domains → dispatchable yes/no)
  (c) The thread name itself (known coding-oriented thread names)

(b) and (c) catch the common case where DB actions are strategic summaries ("Review and act")
but the thread is clearly a coding/engineering context.

Thread → working_dir mapping is also resolved here so the CLI doesn't need to guess.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# research_topic → dispatchable mapping
# Topics that mean "a coding agent can work on this"
# ---------------------------------------------------------------------------
CODING_RESEARCH_TOPICS = {
    'infrastructure_ops',
    'technical_architecture',
    'software_development',
    'api_development',
    'data_engineering',
    'devops',
    'backend',
    'frontend',
    'fullstack',
    'database',
    'ml_ops',
    'deployment',
    'testing',
    'security_engineering',
}

# Topics that mean "human judgment required"
NON_CODING_RESEARCH_TOPICS = {
    'legal_proceedings',
    'personal_health',
    'financial_modeling',
    'clinical_literature',
    'competitive_landscape',
    'knowledge_capture',
    'product_strategy',
}

# ---------------------------------------------------------------------------
# Thread name → working directory mapping
# Extend this as repos are added.
# ---------------------------------------------------------------------------
HOME = os.path.expanduser('~')

THREAD_TO_WORKDIR: dict[str, str] = {
    'MEL Cloud Deployment':         f'{HOME}/patient-talk-summary',
    'MEL Ontology Quality':         f'{HOME}/patient-talk-summary',
    'MEL Prior Auth Voice Agent':   f'{HOME}/patient-talk-summary',
    'Advocate':                     f'{HOME}/patient-talk-summary',
    'HEDGE Open Source':            f'{HOME}/jordan-hedge',
    'Ridgeline Knowledge Graph':    f'{HOME}/jordan-hedge',
    'Ridgeline L6 Promotion':       f'{HOME}/jordan-hedge',
}

# ---------------------------------------------------------------------------
# Action text keyword patterns
# ---------------------------------------------------------------------------
NON_IMPLEMENTABLE_PATTERNS = [
    r'\bcall\b',
    r'\bphone\b',
    r'\bcontact\b',
    r'\bscheduled?\s+appointment\b',
    r'\bwait\s+for\b',
    r'\bfollow[\s-]up\s+with\b',
    r'\bsend\s+(an?\s+)?email\b',
    r'\btext\b',
    r'\bresponse\s+from\b',
    r'\battorney\b',
    r'\bphysician\b',
    r'\binsurance\s+company\b',
    r'\bmeeting\b',
    r'\bask\b',
    r'\bconfirm\s+with\b',
    r'\bapproval\b',
]

IMPLEMENTABLE_PATTERNS = [
    r'\bfix\b',
    r'\bbuild\b',
    r'\bdeploy\b',
    r'\bimplement\b',
    r'\bcreate\s+(a\s+)?(file|script|config|service|endpoint|component|function|module)\b',
    r'\bupdate\s+(the\s+)?(code|config|env|file|script|dockerfile|yaml|json)\b',
    r'\bdebug\b',
    r'\brefactor\b',
    r'\badd\s+(a\s+)?(route|endpoint|feature|test|migration)\b',
    r'\bwrite\s+(a\s+)?(test|script|migration|query)\b',
    r'\bconfigure\b',
    r'\bset\s+up\b',
    r'\binitialize\b',
    r'\brun\s+(the\s+)?(test|migration|script|build)\b',
    r'\bpatch\b',
]

# Signals that block dispatch regardless of other signals
BLOCKING_SIGNAL_CONCEPTS = {
    'EXTERNAL_BLOCKED',
    'deadline_missed',
}

# Signals that are soft blockers (ok if other factors are strong)
SOFT_BLOCKING_CONCEPTS = {
    'DECISION_PENDING',
}


@dataclass
class DispatchCandidate:
    thread_name: str
    thread_id: str
    priority_score: float
    next_action: str
    context: str
    confidence: float          # 0-1: how confident we are it's implementable
    working_dir: str = ''      # resolved working directory for RALPH
    research_topic: str = ''   # from thread metadata


def resolve_working_dir(thread_name: str, fallback: str = '') -> str:
    """
    Return the best working directory for a given thread.
    Checks THREAD_TO_WORKDIR mapping, then falls back to partial-name match,
    then to the provided fallback (usually cwd).
    """
    # Exact match
    if thread_name in THREAD_TO_WORKDIR:
        return THREAD_TO_WORKDIR[thread_name]

    # Partial match (case-insensitive, first keyword)
    name_lower = thread_name.lower()
    for key, path in THREAD_TO_WORKDIR.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return path

    return fallback or os.getcwd()


def is_implementable_by_topic(research_topic: str) -> Optional[tuple[bool, float]]:
    """
    If the research_topic clearly categorizes the thread, return (implementable, confidence).
    Returns None if topic is unknown/ambiguous.
    """
    if research_topic in CODING_RESEARCH_TOPICS:
        return True, 0.85
    if research_topic in NON_CODING_RESEARCH_TOPICS:
        return False, 0.85
    return None


def is_implementable_by_name(thread_name: str) -> Optional[tuple[bool, float]]:
    """
    Check if the thread name itself suggests a coding context.
    Returns None if indeterminate.
    """
    name_lower = thread_name.lower()
    coding_keywords = ['deploy', 'build', 'api', 'cloud', 'service', 'code', 'graph',
                       'architecture', 'open source', 'schema', 'ontology', 'infra']
    human_keywords = ['estate', 'lawsuit', 'malpractice', 'recovery', 'meniscus',
                      'health', 'insurance claim', 'attorney', 'promotion', 'l6']

    if any(k in name_lower for k in coding_keywords):
        return True, 0.75
    if any(k in name_lower for k in human_keywords):
        return False, 0.80
    return None


def is_implementable_heuristic(action: str) -> tuple[bool, float]:
    """
    Fast-path heuristic on action text.
    Returns (implementable, confidence).
    """
    action_lower = action.lower()

    for pat in NON_IMPLEMENTABLE_PATTERNS:
        if re.search(pat, action_lower):
            return False, 0.9

    impl_hits = sum(1 for pat in IMPLEMENTABLE_PATTERNS if re.search(pat, action_lower))
    if impl_hits >= 2:
        return True, 0.9
    if impl_hits == 1:
        return True, 0.7

    return False, 0.3   # ambiguous — not implementable by text alone


def classify_implementable(thread_name: str, research_topic: str, action: str) -> tuple[bool, float]:
    """
    Multi-signal classification. Priority order:
      1. research_topic (most reliable — explicitly categorized)
      2. thread name keywords
      3. action text heuristic
    """
    # 1. Topic-based
    topic_result = is_implementable_by_topic(research_topic)
    if topic_result is not None:
        return topic_result

    # 2. Name-based
    name_result = is_implementable_by_name(thread_name)
    if name_result is not None:
        return name_result

    # 3. Text heuristic on action
    return is_implementable_heuristic(action)


def has_hard_blocking_signals(thread) -> bool:
    """Check for hard-block signals (must not dispatch regardless of priority)."""
    if not hasattr(thread, 'signals') or not thread.signals:
        return False
    recent = thread.signals[-10:]
    return any(
        getattr(s, 'concept', '') in BLOCKING_SIGNAL_CONCEPTS
        for s in recent
    )


def get_dispatch_candidates(
    threads,
    priority_threshold: float = 1.5,
    max_candidates: int = 5,
) -> list[DispatchCandidate]:
    """
    Given a list of HEDGE threads, return those ready for RALPH dispatch.
    Sorted by priority descending.
    """
    candidates = []

    for thread in threads:
        score = getattr(thread, 'priority_score', 0)
        if score < priority_threshold:
            continue

        actions = getattr(thread, 'next_actions', [])
        if not actions:
            continue

        # Extract action description and metadata
        raw = actions[0]
        if isinstance(raw, dict):
            top_action = raw.get('description') or raw.get('title') or str(raw)
            research_topic = raw.get('research_topic', '')
            action_type = raw.get('action_type', '')
        else:
            top_action = str(raw)
            research_topic = ''
            action_type = ''

        # Also pull research_topic from thread metadata if not on action
        if not research_topic:
            research_topic = getattr(thread, 'research_topic', '') or ''
            if isinstance(research_topic, dict):
                research_topic = ''

        # Hard-block check
        if has_hard_blocking_signals(thread):
            continue

        # Implementability classification (multi-signal)
        implementable, confidence = classify_implementable(
            thread.name, research_topic, top_action
        )
        if not implementable:
            continue

        # Build context from thread
        context_parts = []
        if hasattr(thread, 'description') and thread.description:
            context_parts.append(thread.description)

        # Include top hypotheses with their probabilities
        if hasattr(thread, 'hypotheses') and thread.hypotheses:
            active_hyps = [
                h for h in thread.hypotheses
                if hasattr(h, 'probability') and h.probability > 0.5
            ]
            active_hyps.sort(key=lambda h: h.probability, reverse=True)
            for h in active_hyps[:4]:
                htype = getattr(h, 'hypothesis_type', '') or getattr(h, 'text', '')
                prob = getattr(h, 'probability', 0)
                context_parts.append(f"Signal: {htype} (P={prob:.0%})")

        working_dir = resolve_working_dir(thread.name)

        candidates.append(DispatchCandidate(
            thread_name=thread.name,
            thread_id=getattr(thread, 'id', thread.name),
            priority_score=score,
            next_action=top_action,
            context='\n'.join(context_parts),
            confidence=confidence,
            working_dir=working_dir,
            research_topic=research_topic,
        ))

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return candidates[:max_candidates]
