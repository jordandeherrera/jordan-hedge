"""
Feedback loop: converts a RALPH DispatchResult back into HEDGE signals.

This closes the loop:
  HEDGE thread → convergence trigger → RALPH execution → signal ingestion → updated posterior

With BCVT integration, feedback now compares the prior vector (captured before dispatch)
to the posterior vector (recomputed from HEDGE after Ralph returns).
The comparison determines whether firstFailure moved — confirming progress, regression, or stall.

  prior_vector[firstFailure] = 0  →  dispatch Ralph  →  posterior_vector[firstFailure] = ?
    = 1  →  progress (fix worked, firstFailure advanced)
    = 0  →  stalled  (fix didn't converge, hypothesis may be wrong)
  posterior firstFailure < prior  →  regression (revert immediately)
  posterior firstFailure = -1     →  complete (all bits = 1)
"""

import json
from .dispatcher import DispatchResult


def ingest_result(engine, result: DispatchResult, spec: dict = None) -> dict:
    """
    Ingest a RALPH outcome as a HEDGE signal on the originating thread.
    If a spec with bcvt_prior_snapshot is provided, also performs vector comparison.

    Args:
        engine: HedgeEngine instance
        result: DispatchResult from dispatcher.dispatch()
        spec:   story spec dict (from build_story_spec) — used for BCVT vector comparison

    Returns:
        dict with 'signal', 'bcvt_comparison' (if available), 'posterior_vector' (if available)
    """
    from src.engine.hedge_engine import SignalItem

    # ------------------------------------------------------------------
    # Step 1: BCVT vector comparison (if prior snapshot available)
    # ------------------------------------------------------------------
    bcvt_comparison = None
    posterior_vector = None

    if spec and spec.get('bcvt_prior_snapshot'):
        try:
            from src.orchestrator.bcvt import build_vector_from_thread, compare_vectors, BCVTVector, BCVTProbeResult

            # Reconstruct prior vector from snapshot
            snap = spec['bcvt_prior_snapshot']
            prior_probes = [
                BCVTProbeResult(
                    node_index=p['probe'].replace('node_', '') and int(p['probe'].split('_')[1]),
                    hypothesis_type=p['name'],
                    pass_=p['pass'],
                    resolution_state=p['detail'].get('resolution_state', 'open'),
                    probability=p['detail'].get('probability', 0.5),
                    log_odds_posterior=p['detail'].get('log_odds_posterior', 0.0),
                    detail=p['detail'],
                )
                for p in snap.get('probes', [])
            ]
            prior_vec = BCVTVector(
                thread_name=snap['thread'],
                run_id=snap['runId'],
                probes=prior_probes,
            )

            # Recompute posterior vector from current HEDGE state
            post_vec = engine.get_bcvt_vector(result.thread_name)
            if post_vec:
                posterior_vector = post_vec
                bcvt_comparison = compare_vectors(prior_vec, post_vec)

                print(f"\n[FEEDBACK] BCVT vector comparison for '{result.thread_name}':")
                print(f"[FEEDBACK]   Prior:    {prior_vec.vector} (firstFailure={prior_vec.first_failure})")
                print(f"[FEEDBACK]   Post:     {post_vec.vector} (firstFailure={post_vec.first_failure})")
                print(f"[FEEDBACK]   Outcome:  {bcvt_comparison['outcome'].upper()}")
                print(f"[FEEDBACK]   {bcvt_comparison['explanation']}")
                print(f"[FEEDBACK]   Action:   {bcvt_comparison['action']}")
        except Exception as e:
            print(f"[FEEDBACK] BCVT comparison failed (non-critical): {e}")

    # ------------------------------------------------------------------
    # Step 2: Determine the primary HEDGE signal from BCVT outcome (if available)
    #         Otherwise fall back to task_complete / task_failed from exit code
    # ------------------------------------------------------------------
    if bcvt_comparison:
        # BCVT-derived signal overrides simple pass/fail
        concept = bcvt_comparison['hedge_signal']
        confidence = 1.0 if bcvt_comparison['outcome'] in ('complete', 'progress') else 0.9
        note = bcvt_comparison['explanation']
    else:
        concept = result.outcome_signal  # 'task_complete' or 'task_failed'
        confidence = 1.0 if result.success else 0.9
        note = result.outcome_note

    # ------------------------------------------------------------------
    # Step 3: Ingest primary signal
    # ------------------------------------------------------------------
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
            'bcvt_outcome': bcvt_comparison['outcome'] if bcvt_comparison else None,
            'prior_first_failure': bcvt_comparison['prior_first_failure'] if bcvt_comparison else None,
            'posterior_first_failure': bcvt_comparison['posterior_first_failure'] if bcvt_comparison else None,
        }
    )

    print(f"\n[FEEDBACK] Ingesting '{concept}' → thread '{result.thread_name}'")
    print(f"[FEEDBACK] {note}")

    engine.ingest_signals(result.thread_name, [signal])

    # ------------------------------------------------------------------
    # Step 4: Secondary signals based on BCVT outcome
    # ------------------------------------------------------------------
    if bcvt_comparison:
        outcome = bcvt_comparison['outcome']
        if outcome == 'stalled':
            # Hypothesis was wrong — keep entropy high so thread resurfaces
            stale = SignalItem(
                signal_name='task_stale',
                value='present',
                confidence=0.7,
                source='ralph_loop',
                raw_data={'reason': 'BCVT firstFailure unchanged — hypothesis incorrect or fix incomplete'}
            )
            engine.ingest_signals(result.thread_name, [stale])
        elif outcome == 'regression':
            # Regression introduced — flag as new risk
            risk = SignalItem(
                signal_name='AT_RISK',
                value='present',
                confidence=0.85,
                source='ralph_loop',
                raw_data={'reason': f"BCVT regression: firstFailure moved upstream from "
                                    f"{bcvt_comparison['prior_first_failure']} to "
                                    f"{bcvt_comparison['posterior_first_failure']}. Revert."}
            )
            engine.ingest_signals(result.thread_name, [risk])
    elif not result.success:
        # No BCVT — fall back to stale signal on failure
        stale = SignalItem(
            signal_name='task_stale',
            value='present',
            confidence=0.7,
            source='ralph_loop',
            raw_data={'reason': f"RALPH failed: {result.error_summary}"}
        )
        engine.ingest_signals(result.thread_name, [stale])

    return {
        'signal': concept,
        'bcvt_comparison': bcvt_comparison,
        'posterior_vector': posterior_vector.emit() if posterior_vector else None,
    }
