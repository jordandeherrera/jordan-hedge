#!/usr/bin/env python3
"""
HEDGE → RALPH Convergence Orchestrator CLI

Usage:
    python3 -m src.cli.orchestrate [options]

Options:
    --list              Show executable candidates ranked by priority (default action)
    --dispatch N        Dispatch candidate #N to RALPH (1-indexed)
    --dispatch-top      Auto-dispatch the top-ranked candidate
    --dry-run           Show what would be dispatched without running
    --dir PATH          Working directory for RALPH (default: cwd or auto-detected)
    --threshold FLOAT   Minimum priority score to consider (default: 60)
    --thread NAME       Filter to a specific thread by name
    --inject            Manually inject a story (enter interactively)
    --story TEXT        Dispatch with a custom story text (bypasses story_gen)

Examples:
    # See what's ready to dispatch
    python3 -m src.cli.orchestrate --list

    # Dispatch the top candidate
    python3 -m src.cli.orchestrate --dispatch-top --dir /home/ubuntu/patient-talk-summary

    # Dispatch candidate #2 with dry run first
    python3 -m src.cli.orchestrate --dispatch 2 --dry-run --dir /path/to/repo

    # Dispatch with a manual story
    python3 -m src.cli.orchestrate --story "Fix the URL parsing bug in patientMedicationsService.ts" \\
        --thread "MEL Cloud Deployment" --dir /home/ubuntu/patient-talk-summary
"""

import argparse
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.hedge_engine import HedgeEngine
from src.orchestrator.convergence import get_dispatch_candidates
from src.orchestrator.story_gen import build_story_spec
from src.orchestrator.dispatcher import dispatch
from src.orchestrator.feedback import ingest_result


def print_candidates(candidates):
    if not candidates:
        print("\n  No executable candidates found above threshold.")
        print("  (Either all threads are blocked on external actions, or priority scores are too low)\n")
        return

    print(f"\n{'='*70}")
    print(f"  EXECUTABLE CANDIDATES — ranked by priority")
    print(f"{'='*70}")
    for i, c in enumerate(candidates, 1):
        print(f"\n  [{i}] {c.thread_name}")
        print(f"      Priority: {c.priority_score:.0f}  |  Confidence: {c.confidence:.0%}")
        print(f"      Action:   {c.next_action}")
        if c.context:
            ctx_preview = c.context[:120].replace('\n', ' ')
            print(f"      Context:  {ctx_preview}...")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='HEDGE → RALPH Convergence Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--list', action='store_true', default=True,
                        help='List executable candidates (default)')
    parser.add_argument('--dispatch', type=int, metavar='N',
                        help='Dispatch candidate #N (1-indexed)')
    parser.add_argument('--dispatch-top', action='store_true',
                        help='Auto-dispatch top-ranked candidate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show story without executing')
    parser.add_argument('--dir', metavar='PATH', default=os.getcwd(),
                        help='Working directory for RALPH')
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Minimum priority score (default: 60)')
    parser.add_argument('--thread', metavar='NAME',
                        help='Filter to specific thread name')
    parser.add_argument('--story', metavar='TEXT',
                        help='Custom story text (bypasses story_gen)')
    parser.add_argument('--max-iter', type=int, default=5,
                        help='RALPH max iterations (default: 5)')
    parser.add_argument('--no-feedback', action='store_true',
                        help='Skip ingesting result back into HEDGE')

    args = parser.parse_args()

    # Load engine
    print("\n[ORCHESTRATOR] Loading HEDGE engine...")
    engine = HedgeEngine()
    threads = engine.get_all_threads()
    print(f"[ORCHESTRATOR] {len(threads)} threads loaded")

    # Get candidates
    candidates = get_dispatch_candidates(
        threads,
        priority_threshold=args.threshold,
    )

    # Filter by thread name if specified
    if args.thread:
        candidates = [c for c in candidates if args.thread.lower() in c.thread_name.lower()]

    # Determine action
    dispatch_idx = None
    if args.dispatch:
        dispatch_idx = args.dispatch - 1  # 0-indexed
    elif args.dispatch_top:
        dispatch_idx = 0

    # List mode
    print_candidates(candidates)

    if dispatch_idx is None and not args.story:
        # Just listing — done
        return 0

    # Custom story mode
    if args.story:
        if not args.thread and not candidates:
            print("[ERROR] --story requires --thread or at least one candidate thread")
            return 1

        thread_name = args.thread or (candidates[0].thread_name if candidates else 'Manual')
        thread_id = thread_name
        priority = candidates[0].priority_score if candidates else 100.0

        from src.orchestrator.convergence import DispatchCandidate
        candidate = DispatchCandidate(
            thread_name=thread_name,
            thread_id=thread_id,
            priority_score=priority,
            next_action=args.story,
            context='',
            confidence=1.0,
        )
        spec = build_story_spec(candidate, args.dir)
        spec['story'] = args.story  # use story verbatim, don't LLM-synthesize
    else:
        # Dispatch from candidates
        if dispatch_idx >= len(candidates):
            print(f"[ERROR] Candidate #{dispatch_idx+1} not found (only {len(candidates)} candidates)")
            return 1

        candidate = candidates[dispatch_idx]
        print(f"[ORCHESTRATOR] Generating story for: {candidate.thread_name}")
        spec = build_story_spec(candidate, args.dir)

    # Dispatch
    result = dispatch(spec, dry_run=args.dry_run, max_iterations=args.max_iter)

    # Feedback
    if not args.dry_run and not args.no_feedback:
        ingest_result(engine, result)

    # Summary
    print(f"\n{'='*60}")
    if result.success:
        print(f"  ✓ SHIPPED — {result.outcome_note}")
    else:
        print(f"  ✗ FAILED  — {result.outcome_note}")
    print(f"{'='*60}\n")

    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
