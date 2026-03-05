"""
BCVT — Binary Causal Vector Traversal
======================================
The quantization layer between HEDGE (continuous reasoning) and Ralph (execution).

  HEDGE  → continuous probability space (sigmoid: log-odds → posterior)
  BCVT   → threshold crossing → bit (sigmoid converged = 1, mid-curve = 0)
  Ralph  → execution loop, iterates until bit flips

Each bit in the vector encodes whether a HEDGE hypothesis has converged:
  bit = 1  →  resolution_state != 'open'  (supported or contradicted)
             The sigmoid has crossed the action threshold.
             HEDGE has accumulated sufficient evidence to call it.

  bit = 0  →  resolution_state == 'open'
             The sigmoid is mid-curve. Still accumulating evidence.
             This is the minimum sufficient intervention point.

firstFailure = min{ i : vector[i] == 0 }

Fixing any hypothesis j > firstFailure is wasted effort — the causal
ordering means downstream hypotheses cannot resolve while upstream ones remain open.

KB reference: 8673504f — BCVT + HEDGE + Ralph: The Three-Layer Reasoning Stack
KB reference: 9ef78742 — Binary Causal Vector Traversal (BCVT) Algorithm
"""

import json
import random
import string
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Probe definitions — the acceptance criteria Ralph uses as oracle
# ---------------------------------------------------------------------------

@dataclass
class ProbeDefinition:
    """
    Defines what observable state confirms a hypothesis has resolved.
    This is Ralph's acceptance criterion — the oracle it evaluates after each fix.

    For code threads: a shell command that exits 0 on pass.
    For non-code threads: a verifiable state description.

    The probe is derived from the HEDGE hypothesis type + thread context.
    It is NOT written by hand per task — it is generated from the model.
    """
    node_index: int
    hypothesis_type: str
    pass_condition: str           # human-readable: "confirm that X is true"
    shell_check: Optional[str]    # shell command: exits 0 = pass, non-zero = fail
    hedge_check: str              # always present: re-run BCVT and confirm firstFailure > node_index


# ---------------------------------------------------------------------------
# Probe result — one bit with structured detail
# ---------------------------------------------------------------------------

@dataclass
class BCVTProbeResult:
    """Single probe result — one bit in the causal vector."""
    node_index: int
    hypothesis_type: str
    pass_: bool                   # True = bit 1 (converged), False = bit 0 (open)
    resolution_state: str         # 'supported' | 'contradicted' | 'open'
    probability: float
    log_odds_posterior: float
    detail: dict = field(default_factory=dict)

    @property
    def bit(self) -> int:
        return 1 if self.pass_ else 0


# ---------------------------------------------------------------------------
# BCVT Vector — the full binary state of a HEDGE thread
# ---------------------------------------------------------------------------

@dataclass
class BCVTVector:
    """
    Binary state vector for a HEDGE thread.

    Ordering: resolved hypotheses first (they've converged), then open
    hypotheses ordered by probability × risk_weight (highest probability =
    most likely root cause = fix first).

    Note on ordering: true causal ordering would require a causal_order field
    in the schema. Current ordering approximates causality via probability —
    the hypothesis with highest posterior is most likely the binding constraint.
    Adding causal_order to the DB schema is a planned improvement.
    """
    thread_name: str
    run_id: str
    probes: list[BCVTProbeResult]

    @property
    def vector(self) -> list[int]:
        return [p.bit for p in self.probes]

    @property
    def first_failure(self) -> int:
        """Index of first unconverged probe. -1 means all converged."""
        for i, p in enumerate(self.probes):
            if not p.pass_:
                return i
        return -1

    @property
    def target_probe(self) -> Optional[BCVTProbeResult]:
        """The probe at firstFailure — the current intervention target."""
        idx = self.first_failure
        return None if idx < 0 else self.probes[idx]

    @property
    def all_pass(self) -> bool:
        return self.first_failure < 0

    def emit(self) -> dict:
        """Emit structured diagnostic JSON — mirrors the MEL BCVT probe format."""
        ff = self.first_failure
        target = self.target_probe
        if ff < 0:
            summary = "ALL HYPOTHESES CONVERGED"
        else:
            summary = (
                f"FIRST UNCONVERGED: node_{ff} ({target.hypothesis_type}) "
                f"P={target.probability:.2f} resolution={target.resolution_state}"
            )
        return {
            "level": "diagnostic",
            "source": "vector",
            "runId": self.run_id,
            "thread": self.thread_name,
            "vector": self.vector,
            "firstFailure": ff,
            "summary": summary,
            "probes": [
                {
                    "probe": f"node_{p.node_index}",
                    "name": p.hypothesis_type,
                    "pass": p.pass_,
                    "detail": {
                        "resolution_state": p.resolution_state,
                        "probability": round(p.probability, 3),
                        "log_odds_posterior": round(p.log_odds_posterior, 3),
                        **p.detail,
                    },
                }
                for p in self.probes
            ],
        }

    def print_summary(self) -> None:
        """Print a compact human-readable summary."""
        v = self.vector
        bits = "".join(str(b) for b in v)
        ff = self.first_failure
        print(f"\n  BCVT [{self.thread_name}] runId={self.run_id}")
        print(f"  Vector: [{', '.join(str(b) for b in v)}]  ({bits})")
        if ff < 0:
            print(f"  Status: ALL CONVERGED ✓")
        else:
            t = self.target_probe
            print(f"  firstFailure: node_{ff} — {t.hypothesis_type}")
            print(f"  Target P={t.probability:.2f}, log_odds={t.log_odds_posterior:.2f}, "
                  f"resolution={t.resolution_state}")
        print()


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

def build_vector_from_thread(thread_state, run_id: str = None) -> BCVTVector:
    """
    Build a BCVT vector from a HEDGE ThreadState.

    Ordering heuristic (until causal_order field is added to schema):
      1. Resolved hypotheses first (supported or contradicted), ordered by |log_odds|
      2. Open hypotheses ordered by probability × risk_weight DESC
         (highest posterior = most likely binding constraint = fix first)

    Bit = 1 iff resolution_state != 'open' (hypothesis has converged,
    either its presence is confirmed or it has been ruled out).
    """
    if run_id is None:
        run_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    hypotheses = thread_state.hypotheses or []

    def sort_key(h):
        if h.resolution_state != "open":
            # Resolved: sort by |log_odds| descending (most confident first)
            return (0, -abs(h.log_odds_posterior))
        # Open: sort by probability × risk_weight descending
        return (1, -(h.probability * h.risk_weight))

    sorted_hyps = sorted(hypotheses, key=sort_key)

    probes = []
    for i, h in enumerate(sorted_hyps):
        is_converged = h.resolution_state != "open"
        probes.append(
            BCVTProbeResult(
                node_index=i,
                hypothesis_type=h.hypothesis_type,
                pass_=is_converged,
                resolution_state=h.resolution_state,
                probability=h.probability,
                log_odds_posterior=h.log_odds_posterior,
                detail={
                    "risk_weight": h.risk_weight,
                    "entropy": round(h.entropy, 3),
                    "uncertainty": round(h.uncertainty, 3),
                    "llr_support_threshold": h.llr_support_threshold,
                    "llr_contradict_threshold": h.llr_contradict_threshold,
                },
            )
        )

    return BCVTVector(thread_name=thread_state.name, run_id=run_id, probes=probes)


def generate_probe_definition(
    probe: BCVTProbeResult,
    thread_name: str,
    working_dir: str = "",
) -> ProbeDefinition:
    """
    Generate an acceptance criterion for a probe based on hypothesis type and context.
    The shell_check is only set for code-oriented threads.
    hedge_check is always set — it re-runs the BCVT vector and checks firstFailure moved.
    """
    h = probe.hypothesis_type
    node = probe.node_index

    hedge_check = (
        f"python3 -c \""
        f"from src.engine.hedge_engine import HedgeEngine; "
        f"e = HedgeEngine(); "
        f"v = e.get_bcvt_vector('{thread_name}'); "
        f"ff = v.first_failure; "
        f"print(v.emit()); "
        f"assert ff > {node} or ff < 0, f'firstFailure still at {{ff}} — fix did not converge'"
        f"\""
    )

    conditions = {
        "BINDING_CONSTRAINT": (
            f"The blocking constraint on '{thread_name}' has been removed. "
            f"The thread can now progress without this impediment.",
            None,
        ),
        "URGENT_ACTION_NEEDED": (
            f"The urgent action for '{thread_name}' has been taken. "
            f"Evidence of completion is present (commit, email sent, document filed, etc.).",
            None,
        ),
        "AT_RISK": (
            f"The risk signal for '{thread_name}' has been mitigated. "
            f"Protective action taken or exposure removed.",
            None,
        ),
        "EXTERNAL_BLOCKED": (
            f"The external party for '{thread_name}' has responded or been re-engaged. "
            f"Follow-up sent with confirmation.",
            None,
        ),
        "STALE_THREAD": (
            f"The stale thread '{thread_name}' has been advanced or closed. "
            f"Status is updated and next step is clear.",
            None,
        ),
        "DECISION_PENDING": (
            f"The pending decision for '{thread_name}' has been made. "
            f"Decision is documented and next steps are defined.",
            None,
        ),
        "OPPORTUNITY_ACTIVE": (
            f"The opportunity window for '{thread_name}' has been acted on. "
            f"Action taken before window closes.",
            None,
        ),
    }

    # Code-oriented hypotheses get shell checks
    code_conditions = {
        "BINDING_CONSTRAINT": (
            f"The blocking constraint on '{thread_name}' has been removed. "
            f"Tests pass and the blocked pipeline stage now succeeds.",
            f"cd {working_dir} && git status && git diff --stat HEAD",
        ),
        "AT_RISK": (
            f"The risk in '{thread_name}' has been mitigated. "
            f"Relevant tests pass.",
            f"cd {working_dir} && git log --oneline -3",
        ),
    }

    if working_dir and h in code_conditions:
        pass_condition, shell_check = code_conditions[h]
    elif h in conditions:
        pass_condition, shell_check = conditions[h]
    else:
        pass_condition = (
            f"Hypothesis '{h}' on thread '{thread_name}' has converged: "
            f"resolution_state moves from 'open' to 'supported' or 'contradicted'."
        )
        shell_check = None

    return ProbeDefinition(
        node_index=node,
        hypothesis_type=h,
        pass_condition=pass_condition,
        shell_check=shell_check,
        hedge_check=hedge_check,
    )


# ---------------------------------------------------------------------------
# Vector comparison — determines if Ralph's fix made progress
# ---------------------------------------------------------------------------

def compare_vectors(prior: BCVTVector, posterior: BCVTVector) -> dict:
    """
    Compare two BCVT vectors from consecutive Ralph iterations.

    Returns a dict with:
      outcome:                 'complete' | 'progress' | 'regression' | 'stalled'
      prior_first_failure:     int (-1 = all pass)
      posterior_first_failure: int
      bits_flipped:            list of node indices that changed
      hedge_signal:            concept name to ingest into HEDGE
      explanation:             human-readable reasoning
      action:                  what to do next
    """
    pf_prior = prior.first_failure
    pf_post = posterior.first_failure
    prior_v = prior.vector
    post_v = posterior.vector

    bits_flipped = [
        i for i in range(min(len(prior_v), len(post_v)))
        if prior_v[i] != post_v[i]
    ]

    if pf_post < 0:
        outcome = "complete"
        signal = "task_complete"
        explanation = "All hypotheses converged. Thread fully resolved."
        action = "Mark thread as resolved. Archive or close."
    elif pf_prior < 0 and pf_post >= 0:
        # Regression from fully-converged to partially-open (shouldn't happen in normal flow)
        outcome = "regression"
        signal = "task_failed"
        explanation = (
            f"REGRESSION: thread was fully converged but firstFailure appeared at node_{pf_post}. "
            f"A fix introduced a new open hypothesis."
        )
        action = "Revert last change. Investigate what re-opened the hypothesis."
    elif pf_post > pf_prior:
        outcome = "progress"
        signal = "task_complete"
        explanation = (
            f"firstFailure moved downstream: node_{pf_prior} → node_{pf_post}. "
            f"Fix confirmed. {len(bits_flipped)} bit(s) flipped."
        )
        action = f"Continue. Target node_{pf_post} in next iteration."
    elif pf_post < pf_prior:
        outcome = "regression"
        signal = "task_failed"
        explanation = (
            f"REGRESSION: firstFailure moved UPSTREAM: node_{pf_prior} → node_{pf_post}. "
            f"Fix introduced a new failure at an earlier causal stage."
        )
        action = "REVERT immediately. Do not continue downstream."
    else:
        outcome = "stalled"
        signal = "task_stale"
        explanation = (
            f"firstFailure unchanged at node_{pf_prior}. "
            f"Fix did not converge HEDGE — hypothesis still open. "
            f"Either the fix was incomplete or the hypothesis was wrong."
        )
        action = (
            f"Add sub-probes to node_{pf_prior} for finer resolution. "
            f"Re-read the hypothesis detail payload for a stronger evidence signal."
        )

    return {
        "outcome": outcome,
        "prior_first_failure": pf_prior,
        "posterior_first_failure": pf_post,
        "bits_flipped": bits_flipped,
        "hedge_signal": signal,
        "explanation": explanation,
        "action": action,
        "prior_vector": prior.vector,
        "posterior_vector": posterior.vector,
    }
