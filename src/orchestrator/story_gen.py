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


def _get_repo_context(working_dir: str) -> str:
    """Pull lightweight repo context to ground the story in actual files."""
    import os
    lines = []

    # Recent git log
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '-8'],
            cwd=working_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append(f"Recent commits:\n{result.stdout.strip()}")
    except Exception:
        pass

    # Top-level file structure
    try:
        entries = []
        for name in sorted(os.listdir(working_dir)):
            if name.startswith('.'):
                continue
            full = os.path.join(working_dir, name)
            entries.append(name + ('/' if os.path.isdir(full) else ''))
        if entries:
            lines.append(f"Repo layout: {', '.join(entries[:20])}")
    except Exception:
        pass

    # Package.json or pyproject for stack info
    for meta_file in ['package.json', 'pyproject.toml', 'Makefile']:
        meta_path = os.path.join(working_dir, meta_file)
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    content = f.read(600)
                lines.append(f"{meta_file} (excerpt):\n{content}")
                break
            except Exception:
                pass

    return '\n\n'.join(lines)


def _build_bcvt_section(candidate: DispatchCandidate, working_dir: str) -> str:
    """
    Build the BCVT acceptance criterion section for the story.
    This is the probe definition — Ralph's oracle.
    """
    bcvt = getattr(candidate, 'bcvt_vector', None)
    if not bcvt:
        return ""

    ff = bcvt.first_failure
    if ff < 0:
        return "\nBCVT: All hypotheses already converged — verify no regressions were introduced.\n"

    target = bcvt.target_probe
    if not target:
        return ""

    try:
        from src.orchestrator.bcvt import generate_probe_definition
        probe_def = generate_probe_definition(target, candidate.thread_name, working_dir)
    except Exception:
        return ""

    lines = [
        f"\nBCVT ACCEPTANCE CRITERION (node_{ff} — {target.hypothesis_type}):",
        f"  Current state:   P={target.probability:.2f}, log_odds={target.log_odds_posterior:.2f}, resolution='{target.resolution_state}'",
        f"  Pass condition:  {probe_def.pass_condition}",
    ]
    if probe_def.shell_check:
        lines.append(f"  Shell check:     {probe_def.shell_check}")
    lines.append(f"  HEDGE verify:    {probe_def.hedge_check}")
    lines.append(
        f"\n  Success = firstFailure moves from {ff} to >{ff} (or -1 if all resolved)."
        f"\n  If firstFailure is unchanged after your fix, the hypothesis was wrong — add sub-probes."
    )

    # Also show the full vector state for context
    v_str = str(bcvt.vector).replace("True", "1").replace("False", "0")
    lines.append(f"\n  Current vector:  {v_str}")
    lines.append(f"  Run ID:          {bcvt.run_id}")

    return "\n".join(lines)


def _llm_synthesize_story(candidate: DispatchCandidate, working_dir: str) -> str:
    """Use claude-cli to write a focused, implementation-ready RALPH story."""
    repo_context = _get_repo_context(working_dir)
    bcvt_section = _build_bcvt_section(candidate, working_dir)

    meta_prompt = f"""You are converting a HEDGE reasoning thread into a RALPH implementation story.

HEDGE Thread: {candidate.thread_name}
Priority Score: {candidate.priority_score:.0f}
Research Topic: {getattr(candidate, 'research_topic', '') or 'unknown'}
Next Action: {candidate.next_action}
Target Hypothesis (BCVT firstFailure): {getattr(candidate, 'target_hypothesis', '') or 'see context'}
Context:
{candidate.context}

Working directory: {working_dir}

Repo context (use this to write grounded, accurate acceptance criteria):
{repo_context}

{bcvt_section}

Write a RALPH story: a concise, implementation-ready task description.
Requirements:
- Start with the specific action (Fix / Build / Deploy / Configure)
- Target the BCVT firstFailure hypothesis above — that is the minimum sufficient intervention
- Use the actual repo structure above — reference real files/scripts/commands that exist
- Include concrete acceptance criteria as a numbered list (verifiable by CLI/test)
- The LAST acceptance criterion must always be the BCVT HEDGE verify command above
- Scope to what a coding agent can do autonomously in this directory
- NEVER use "npm run test --workspaces" or "npm test --workspaces" — these hit unrelated pre-existing failures in docs/legacy packages; instead scope tests to specific packages e.g. "npm test --workspace=packages/backend"
- NEVER require full eslint on the whole repo — scope to changed files or skip if not relevant to the task
- Each acceptance criterion must be verifiable with a specific shell command (not "check X" but "run Y and confirm Z")
- Max 300 words
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
    """Fallback: structured template story with BCVT acceptance criterion."""
    bcvt_section = _build_bcvt_section(candidate, working_dir)
    target = getattr(candidate, 'target_hypothesis', '') or 'see context'
    return f"""{candidate.next_action}

Thread: {candidate.thread_name}
Target hypothesis (BCVT firstFailure): {target}
Working directory: {working_dir}

Context:
{candidate.context}

Acceptance criteria:
1. The action described above is complete
2. No regressions in existing functionality
3. Changes are committed if in a git repo
{bcvt_section}
"""


def build_story_spec(
    candidate: DispatchCandidate,
    working_dir: str,
    validators: list[str] | None = None,
) -> dict:
    """
    Returns a complete story spec dict ready for dispatch.
    Includes BCVT state so feedback.py can compare vectors post-execution.
    """
    story_text = generate_story_prompt(candidate, working_dir)

    # Snapshot the BCVT vector before dispatch — needed for post-dispatch comparison
    bcvt_snapshot = None
    if getattr(candidate, 'bcvt_vector', None) is not None:
        try:
            bcvt_snapshot = candidate.bcvt_vector.emit()
        except Exception:
            pass

    spec = {
        'thread_name': candidate.thread_name,
        'thread_id': candidate.thread_id,
        'priority_score': candidate.priority_score,
        'working_dir': working_dir,
        'story': story_text,
        'validators': validators or [],
        'source_action': candidate.next_action,
        'bcvt_prior_snapshot': bcvt_snapshot,   # serialized BCVTVector before dispatch
        'bcvt_target_node': getattr(candidate, 'target_node_index', -1),
        'bcvt_target_hypothesis': getattr(candidate, 'target_hypothesis', ''),
    }
    return spec
