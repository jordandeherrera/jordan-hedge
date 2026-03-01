"""
Story generator: converts a HEDGE DispatchCandidate into a RALPH story spec.

RALPH needs:
1. A clear task description (what to build/fix)
2. A working directory (where to run)
3. Optional acceptance criteria
4. Optional context files to include

We use claude-cli to synthesize the thread context into a focused RALPH prompt.
"""

import os
import subprocess
import json
from pathlib import Path
from .convergence import DispatchCandidate

CLAUDE_CLI = os.environ.get(
    'CLAUDE_BIN',
    '/home/ubuntu/prd-orchestrator/bin/claude-cli'
)


def generate_story_prompt(candidate: DispatchCandidate, working_dir: str) -> str:
    """
    Build the story description that gets passed to ralph-loop.
    Uses LLM synthesis if claude-cli is available; falls back to structured template.
    """
    if os.path.exists(CLAUDE_CLI):
        return _llm_synthesize_story(candidate, working_dir)
    return _template_story(candidate, working_dir)


def _llm_synthesize_story(candidate: DispatchCandidate, working_dir: str) -> str:
    """Use claude-cli to write a focused, implementation-ready RALPH story."""
    meta_prompt = f"""You are converting a HEDGE reasoning thread into a RALPH implementation story.

HEDGE Thread: {candidate.thread_name}
Priority Score: {candidate.priority_score:.0f}
Next Action: {candidate.next_action}
Context:
{candidate.context}

Working directory: {working_dir}

Write a RALPH story: a concise, implementation-ready task description.
Requirements:
- Start with the specific action (Fix / Build / Deploy / Configure)
- Include concrete acceptance criteria as a numbered list
- Reference specific files, endpoints, or commands where relevant
- Scope to what can be verified programmatically (tests, curl, file checks)
- Max 200 words
- No preamble, no "here is the story:" — just the story itself

Output only the story text."""

    result = subprocess.run(
        [CLAUDE_CLI, '-p', '--model', 'sonnet'],
        input=meta_prompt,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return _template_story(candidate, working_dir)


def _template_story(candidate: DispatchCandidate, working_dir: str) -> str:
    """Fallback: structured template story."""
    return f"""{candidate.next_action}

Thread: {candidate.thread_name}
Working directory: {working_dir}

Context:
{candidate.context}

Acceptance criteria:
1. The action described above is complete
2. No regressions in existing functionality
3. Changes are committed if in a git repo
"""


def build_story_spec(
    candidate: DispatchCandidate,
    working_dir: str,
    validators: list[str] | None = None,
) -> dict:
    """
    Returns a complete story spec dict ready for dispatch.
    """
    story_text = generate_story_prompt(candidate, working_dir)

    spec = {
        'thread_name': candidate.thread_name,
        'thread_id': candidate.thread_id,
        'priority_score': candidate.priority_score,
        'working_dir': working_dir,
        'story': story_text,
        'validators': validators or [],
        'source_action': candidate.next_action,
    }
    return spec
