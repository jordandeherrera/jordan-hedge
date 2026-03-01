"""
Feedback loop: converts a RALPH DispatchResult back into HEDGE signals.

This closes the loop:
  HEDGE thread → convergence trigger → RALPH execution → signal ingestion → updated posterior

The ingested signals update the thread's entropy and next_actions,
so the next convergence cycle operates on current ground truth.
"""

from .dispatcher import DispatchResult


def ingest_result(engine, result: DispatchResult) -> None:
    """
    Ingest a RALPH outcome as a HEDGE signal on the originating thread.

    Args:
        engine: HedgeEngine instance
        result: DispatchResult from dispatcher.dispatch()
    """
    from src.engine.hedge_engine import SignalItem

    concept = result.outcome_signal  # 'task_complete' or 'task_failed'
    confidence = 1.0 if result.success else 0.9

    note = result.outcome_note

    signal = SignalItem(
        signal_name=concept,
        value='present',
        confidence=confidence,
        source='ralph_loop',
        raw_data={
            'story_preview': result.story[:200],
            'working_dir': result.working_dir,
            'elapsed_seconds': result.elapsed_seconds,
            'iterations': result.iterations,
            'exit_code': result.exit_code,
            'note': note,
            'error_summary': result.error_summary,
        }
    )

    print(f"\n[FEEDBACK] Ingesting '{concept}' → thread '{result.thread_name}'")
    print(f"[FEEDBACK] {note}")

    engine.ingest_signals(result.thread_name, [signal])

    # If the task failed, also ingest a stale signal to keep entropy high
    # so it surfaces again in the next convergence cycle
    if not result.success:
        stale_signal = SignalItem(
            signal_name='task_stale',
            value='present',
            confidence=0.7,
            source='ralph_loop',
            raw_data={'reason': f"RALPH failed: {result.error_summary}"}
        )
        engine.ingest_signals(result.thread_name, [stale_signal])
