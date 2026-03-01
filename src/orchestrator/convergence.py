"""
Convergence detector: determines which HEDGE threads are ready to dispatch to RALPH.

A thread is "executable" when:
1. It has a non-empty next_actions list
2. The top action is implementable (can be done by running code/commands)
3. Priority score >= threshold
4. Not blocked by external signals (EXTERNAL_BLOCKED, DECISION_PENDING dominant)

The "implementable" check uses LLM classification by default, with heuristic
keyword detection as a fast-path filter.
"""

import re
import subprocess
import os
from dataclasses import dataclass
from typing import Optional

# Keywords that indicate external/human-required actions (NOT dispatchable to RALPH)
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
    r'\bdoctor\b',
    r'\bphysician\b',
    r'\binsurance\b',
    r'\bmeeting\b',
    r'\bdecide\b',
    r'\breview\s+with\b',
    r'\bask\b',
    r'\bconfirm\s+with\b',
    r'\bapproval\b',
]

# Keywords that suggest a coding/file/config action (dispatchable)
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

# Signals that block dispatch (LLR-weighted — if these are dominant, don't dispatch)
BLOCKING_SIGNAL_CONCEPTS = {
    'EXTERNAL_BLOCKED',
    'DECISION_PENDING',
    'deadline_missed',   # too late, no code will help
}


@dataclass
class DispatchCandidate:
    thread_name: str
    thread_id: str
    priority_score: float
    next_action: str
    context: str
    confidence: float  # 0-1: how confident we are it's implementable


def is_implementable_heuristic(action: str) -> tuple[bool, float]:
    """
    Fast-path heuristic: check action text for implementable vs human-required keywords.
    Returns (implementable, confidence).
    """
    action_lower = action.lower()

    # Hard block: non-implementable patterns
    for pat in NON_IMPLEMENTABLE_PATTERNS:
        if re.search(pat, action_lower):
            return False, 0.9

    # Positive signal: implementable patterns
    impl_hits = sum(1 for pat in IMPLEMENTABLE_PATTERNS if re.search(pat, action_lower))
    if impl_hits >= 2:
        return True, 0.9
    if impl_hits == 1:
        return True, 0.7

    # Ambiguous
    return False, 0.4


def has_blocking_signals(thread) -> bool:
    """
    Check if dominant signals block dispatch.
    """
    if not hasattr(thread, 'signals') or not thread.signals:
        return False
    recent = thread.signals[-10:]  # last 10 signals
    blocking_count = sum(
        1 for s in recent
        if getattr(s, 'concept', '') in BLOCKING_SIGNAL_CONCEPTS
    )
    return blocking_count >= 2


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

        # next_actions can be dicts (from DB) or strings
        raw = actions[0]
        if isinstance(raw, dict):
            top_action = raw.get('description') or raw.get('title') or str(raw)
        else:
            top_action = str(raw)

        if has_blocking_signals(thread):
            continue

        implementable, confidence = is_implementable_heuristic(top_action)
        if not implementable:
            continue

        context_parts = []
        if hasattr(thread, 'description') and thread.description:
            context_parts.append(thread.description)
        if hasattr(thread, 'hypotheses'):
            for h in (thread.hypotheses or [])[:3]:
                if hasattr(h, 'text'):
                    context_parts.append(f"Hypothesis: {h.text}")

        candidates.append(DispatchCandidate(
            thread_name=thread.name,
            thread_id=getattr(thread, 'id', thread.name),
            priority_score=score,
            next_action=top_action,
            context='\n'.join(context_parts),
            confidence=confidence,
        ))

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return candidates[:max_candidates]
