"""
Dispatcher: executes a story spec via ralph-loop subprocess.

Returns a DispatchResult with outcome, stdout/stderr, and exit code.
The result feeds back into HEDGE via feedback.py.
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

RALPH_LOOP = os.environ.get(
    'RALPH_LOOP_BIN',
    '/home/ubuntu/prd-orchestrator/bin/ralph-loop'
)
CLAUDE_CLI = os.environ.get(
    'CLAUDE_BIN',
    '/home/ubuntu/prd-orchestrator/bin/claude-cli'
)


@dataclass
class DispatchResult:
    thread_id: str
    thread_name: str
    story: str
    working_dir: str
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    success: bool
    error_summary: str = ''
    iterations: int = 0

    @property
    def outcome_signal(self) -> str:
        """Signal concept to ingest back into HEDGE."""
        return 'task_complete' if self.success else 'task_failed'

    @property
    def outcome_note(self) -> str:
        if self.success:
            return f"RALPH completed in {self.iterations} iteration(s) ({self.elapsed_seconds:.0f}s)"
        return f"RALPH failed after {self.elapsed_seconds:.0f}s: {self.error_summary}"


def dispatch(spec: dict, dry_run: bool = False, max_iterations: int = 5) -> DispatchResult:
    """
    Run ralph-loop with the given story spec.

    Args:
        spec: story spec from story_gen.build_story_spec()
        dry_run: if True, print the story but don't execute
        max_iterations: RALPH max review cycles
    """
    story = spec['story']
    working_dir = spec['working_dir']
    thread_id = spec['thread_id']
    thread_name = spec['thread_name']

    if not os.path.exists(RALPH_LOOP):
        return DispatchResult(
            thread_id=thread_id,
            thread_name=thread_name,
            story=story,
            working_dir=working_dir,
            exit_code=1,
            stdout='',
            stderr=f'ralph-loop not found at {RALPH_LOOP}',
            elapsed_seconds=0,
            success=False,
            error_summary=f'ralph-loop binary not found: {RALPH_LOOP}',
        )

    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] Would dispatch to RALPH:")
        print(f"  Thread: {thread_name}")
        print(f"  Working dir: {working_dir}")
        print(f"\n--- STORY ---")
        print(story)
        print(f"{'='*60}\n")
        return DispatchResult(
            thread_id=thread_id,
            thread_name=thread_name,
            story=story,
            working_dir=working_dir,
            exit_code=0,
            stdout='[DRY RUN]',
            stderr='',
            elapsed_seconds=0,
            success=True,
            error_summary='',
        )

    env = {
        **os.environ,
        'AI_COMMAND': CLAUDE_CLI,
        'RALPH_MAX_ITERATIONS': str(max_iterations),
        'RALPH_SKIP_VALIDATION': 'false',
        'RALPH_AI_TIMEOUT': '180',
    }

    # Pass validators if specified
    validators = spec.get('validators', [])
    if validators:
        env['RALPH_VALIDATORS'] = ','.join(validators)

    print(f"\n{'='*60}")
    print(f"[DISPATCH] Thread: {thread_name}")
    print(f"[DISPATCH] Working dir: {working_dir}")
    print(f"[DISPATCH] Story preview: {story[:120]}...")
    print(f"{'='*60}\n")

    start = time.time()

    try:
        result = subprocess.run(
            [RALPH_LOOP, '-y', story],
            cwd=working_dir,
            env=env,
            capture_output=False,   # stream to terminal
            timeout=600,            # 10 min max per story
        )
        elapsed = time.time() - start
        success = result.returncode == 0

        # Parse iterations from .ralph/plan.json if available
        iterations = _parse_iterations(working_dir)

        # Extract error summary from stderr if failed
        error_summary = ''
        if not success:
            error_summary = f"Exit code {result.returncode}"

        return DispatchResult(
            thread_id=thread_id,
            thread_name=thread_name,
            story=story,
            working_dir=working_dir,
            exit_code=result.returncode,
            stdout='',
            stderr='',
            elapsed_seconds=elapsed,
            success=success,
            error_summary=error_summary,
            iterations=iterations,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return DispatchResult(
            thread_id=thread_id,
            thread_name=thread_name,
            story=story,
            working_dir=working_dir,
            exit_code=124,
            stdout='',
            stderr='Timed out after 600s',
            elapsed_seconds=elapsed,
            success=False,
            error_summary='RALPH timed out (>10 min)',
        )

    except Exception as e:
        elapsed = time.time() - start
        return DispatchResult(
            thread_id=thread_id,
            thread_name=thread_name,
            story=story,
            working_dir=working_dir,
            exit_code=1,
            stdout='',
            stderr=str(e),
            elapsed_seconds=elapsed,
            success=False,
            error_summary=str(e),
        )


def _parse_iterations(working_dir: str) -> int:
    """Parse iteration count from RALPH's plan.json if it exists."""
    import json
    plan_path = Path(working_dir) / '.ralph' / 'plan.json'
    if plan_path.exists():
        try:
            with open(plan_path) as f:
                plan = json.load(f)
            return plan.get('iterations', 0)
        except Exception:
            pass
    return 0
