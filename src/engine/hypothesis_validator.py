"""
HEDGE Phase 5 — Hypothesis Validation Layer
Jordan HEDGE personal reasoning engine.

Pre-flight checks before any hypothesis enters the reasoning loop.
The engine enforces modeling discipline here rather than silently
producing degraded posteriors downstream.

Design principles:
  - Every hypothesis must be binary (true/false in context of a thread).
    LLRs are additive in log space only for binary hypotheses.
    Gradient states must be decomposed into mutually exclusive binary
    hypothesis types (one-hot style) before submission.
  - Each hypothesis type must be falsifiable: at least one signal relation
    must exist in the ontology that can shift its LLR.
  - LLR thresholds (support / contradict) and min_evidence_count must be
    declared, forcing the modeler to specify how much evidence is "enough."
  - Priors must be non-degenerate: strictly within (0, 1).
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Error codes ──────────────────────────────────────────────────────────────

class ValidationErrorCode(str, Enum):
    GRADIENT_HYPOTHESIS    = "GRADIENT_HYPOTHESIS"    # hypothesis describes a gradient state
    UNFALSIFIABLE          = "UNFALSIFIABLE"          # no signal relation exists in ontology
    MISSING_LLR_THRESHOLDS = "MISSING_LLR_THRESHOLDS" # support/contradict thresholds not declared
    DEGENERATE_PRIOR       = "DEGENERATE_PRIOR"       # prior is 0, 1, or outside (0,1)
    INVALID_LLR_THRESHOLDS = "INVALID_LLR_THRESHOLDS" # support must be > 0, contradict < 0
    INVALID_EVIDENCE_COUNT = "INVALID_EVIDENCE_COUNT" # min_evidence_count must be >= 1


@dataclass
class ValidationError:
    code: ValidationErrorCode
    message: str
    suggestion: str  # actionable fix shown to caller


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self, name: str = ""):
        if not self.valid:
            label = f'"{name}" ' if name else ""
            lines = "\n  - ".join(
                f"[{e.code}] {e.message}. {e.suggestion}"
                for e in self.errors
            )
            raise HypothesisValidationError(
                f"Hypothesis {label}failed validation:\n  - {lines}",
                self.errors
            )


class HypothesisValidationError(Exception):
    def __init__(self, message: str, errors: list[ValidationError]):
        super().__init__(message)
        self.validation_errors = errors


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_LLR_SUPPORT_THRESHOLD    =  3.0   # ~95% posterior from 50% prior
DEFAULT_LLR_CONTRADICT_THRESHOLD = -3.0   # ~5% posterior from 50% prior
DEFAULT_MIN_EVIDENCE_COUNT       = 1

# LLR impact below this is "near-zero" for anomaly detection
ANOMALY_LLR_THRESHOLD = 0.1


# ─── Gradient patterns ────────────────────────────────────────────────────────
# Heuristics for detecting hypothesis names that compress gradient states.
# Intentionally conservative — only match clearly ambiguous names.

# Use (?<![a-zA-Z]) / (?![a-zA-Z]) instead of \b so underscores don't prevent matching.
# e.g. "severe_risk" should match "severe", "high_urgency" should match "high".
_W = r'(?<![a-zA-Z])'   # not preceded by a letter
_WE = r'(?![a-zA-Z])'  # not followed by a letter

_GRADIENT_PATTERNS = [
    re.compile(_W + r'(mild|moderate|moderately|severely|severe|high|low|elevated|reduced|increased|decreased)' + _WE, re.I),
    re.compile(_W + r'(stage[\s_]*\d|grade[\s_]*\d|class[\s_]*[ivxIVX\d]+)' + _WE, re.I),
    re.compile(_W + r'(partial|incomplete|full|complete).{0,20}(response|recovery|remission)' + _WE, re.I),
    re.compile(_W + r'(well|poorly)[\s_]*(controlled)' + _WE, re.I),
    re.compile(_W + r'(early|late|advanced|chronic|acute).{0,20}(phase|stage|presentation)' + _WE, re.I),
    re.compile(_W + r'(somewhat|slightly|very|extremely|highly)' + _WE, re.I),
]


# ─── Validator ────────────────────────────────────────────────────────────────

class HypothesisValidator:
    """
    Pre-flight validation gate for HEDGE hypotheses.

    Usage:
        validator = HypothesisValidator(conn)
        result = validator.validate(name="AT_RISK", prior=0.15, concept_id=3)
        result.raise_if_invalid("AT_RISK")
    """

    def __init__(self, conn=None):
        """
        Pass a DB connection for falsifiability checks.
        If None, falsifiability check is skipped (with a warning).
        """
        self.conn = conn

    def validate(
        self,
        name: str,
        prior: float,
        concept_id: Optional[int] = None,
        llr_support_threshold: Optional[float] = None,
        llr_contradict_threshold: Optional[float] = None,
        min_evidence_count: Optional[int] = None,
        skip_falsifiability: bool = False,
    ) -> ValidationResult:
        """
        Run all validation gates. Returns a result with all errors found
        (non-short-circuiting — collect all problems before returning).
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        self._check_prior_bounds(name, prior, errors)
        self._check_binary_constraint(name, errors, warnings)
        self._check_llr_thresholds(name, llr_support_threshold, llr_contradict_threshold, errors, warnings)
        self._check_min_evidence_count(name, min_evidence_count, errors)

        if not skip_falsifiability:
            if self.conn is not None and concept_id is not None:
                self._check_falsifiability(name, concept_id, errors, warnings)
            else:
                warnings.append(
                    f'No DB connection or concept_id — falsifiability check skipped for "{name}". '
                    'Ensure at least one signal relation is wired to this hypothesis type.'
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ── Gate 1: Prior bounds ───────────────────────────────────────────────────

    def _check_prior_bounds(self, name: str, prior: float, errors: list):
        if prior is None or not isinstance(prior, (int, float)) or prior <= 0 or prior >= 1:
            errors.append(ValidationError(
                code=ValidationErrorCode.DEGENERATE_PRIOR,
                message=f'Prior {prior!r} is outside the open interval (0, 1) for "{name}"',
                suggestion=(
                    'Set prior to a base rate derived from real-world frequency. '
                    'Use 0.1 for uncommon states, 0.5 for unknown prior, 0.3 for moderate base rates. '
                    'Priors of exactly 0 or 1 assert certainty before any evidence — invalid.'
                ),
            ))

    # ── Gate 2: Binary constraint ──────────────────────────────────────────────

    def _check_binary_constraint(self, name: str, errors: list, warnings: list):
        matched = next((p for p in _GRADIENT_PATTERNS if p.search(name)), None)
        if matched:
            errors.append(ValidationError(
                code=ValidationErrorCode.GRADIENT_HYPOTHESIS,
                message=f'Hypothesis name "{name}" appears to describe a gradient state',
                suggestion=(
                    'Decompose into mutually exclusive binary hypotheses (one-hot style). '
                    'Example: instead of "high_urgency", define '
                    '"URGENT_ACTION_NEEDED" (binary) and "CRITICAL_DEADLINE_BREACH" (binary). '
                    'LLRs are additive in log space only for binary hypotheses — '
                    'gradient names corrupt the math.'
                ),
            ))
        elif re.search(r'\bor\b', name, re.I):
            warnings.append(
                f'Hypothesis name "{name}" contains "or" — verify it represents a single binary '
                'condition, not a disjunction of two separate hypotheses.'
            )

    # ── Gate 3: LLR thresholds ─────────────────────────────────────────────────

    def _check_llr_thresholds(
        self,
        name: str,
        support: Optional[float],
        contradict: Optional[float],
        errors: list,
        warnings: list,
    ):
        has_support    = support is not None
        has_contradict = contradict is not None

        if not has_support or not has_contradict:
            # Warn, don't reject — defaults will be applied.
            # Callers that explicitly declare thresholds opt into stronger modeling discipline.
            warnings.append(
                f'LLR thresholds not declared for "{name}". '
                f'Defaults applied: support={DEFAULT_LLR_SUPPORT_THRESHOLD}, '
                f'contradict={DEFAULT_LLR_CONTRADICT_THRESHOLD}. '
                'Declare explicit thresholds to encode domain knowledge about evidence requirements.'
            )
            return

        if support <= 0:
            errors.append(ValidationError(
                code=ValidationErrorCode.INVALID_LLR_THRESHOLDS,
                message=f'llr_support_threshold must be > 0 (got {support}) for "{name}"',
                suggestion=(
                    'llr_support_threshold is the cumulative LLR needed to confirm this hypothesis. '
                    'Use +3.0 for ~95% posterior from a 50% prior. '
                    'Use higher values (e.g. +4.6) for high-stakes hypotheses where false positives are costly.'
                ),
            ))

        if contradict >= 0:
            errors.append(ValidationError(
                code=ValidationErrorCode.INVALID_LLR_THRESHOLDS,
                message=f'llr_contradict_threshold must be < 0 (got {contradict}) for "{name}"',
                suggestion=(
                    'llr_contradict_threshold is the cumulative LLR needed to rule out this hypothesis. '
                    'Use -3.0 for ~5% posterior from a 50% prior. '
                    'Use lower values (e.g. -4.6 ≈ 1%) for hypotheses where false negatives are costly.'
                ),
            ))

    # ── Gate 4: Minimum evidence count ────────────────────────────────────────

    def _check_min_evidence_count(self, name: str, count: Optional[int], errors: list):
        if count is not None and (not isinstance(count, int) or count < 1):
            errors.append(ValidationError(
                code=ValidationErrorCode.INVALID_EVIDENCE_COUNT,
                message=f'min_evidence_count must be a positive integer (got {count!r}) for "{name}"',
                suggestion=(
                    'Set min_evidence_count to the minimum number of independent observations '
                    'required before the hypothesis resolves. '
                    'Use at least 2 for high-stakes hypotheses to prevent premature convergence '
                    'from a single high-LLR signal.'
                ),
            ))

    # ── Gate 5: Falsifiability ─────────────────────────────────────────────────

    def _check_falsifiability(self, name: str, concept_id: int, errors: list, warnings: list):
        try:
            row = self.conn.execute("""
                SELECT COUNT(*) as cnt
                FROM relations
                WHERE hypothesis_concept_id = ?
                  AND (ABS(llr_pos) > 0.001 OR ABS(llr_neg) > 0.001)
            """, (concept_id,)).fetchone()

            count = row["cnt"] if row else 0

            if count == 0:
                errors.append(ValidationError(
                    code=ValidationErrorCode.UNFALSIFIABLE,
                    message=(
                        f'No signal relations found in ontology for hypothesis type '
                        f'"{name}" (concept_id={concept_id})'
                    ),
                    suggestion=(
                        'Add at least one row to the relations table where '
                        f'hypothesis_concept_id = {concept_id} with a non-zero llr_pos or llr_neg. '
                        'Without signal paths, no observation can shift this hypothesis\'s posterior — '
                        'it will never update from its prior.'
                    ),
                ))
            elif count < 2:
                warnings.append(
                    f'Hypothesis type "{name}" has only {count} signal relation(s). '
                    'Consider adding more relations for richer evidence coverage.'
                )
        except Exception as e:
            logger.warning(f'[HypothesisValidator] Falsifiability check failed for "{name}": {e}')
            warnings.append(f'Falsifiability check could not complete for "{name}" (DB error).')


# ─── Anomaly detector ─────────────────────────────────────────────────────────

@dataclass
class EvidenceAnomalySignal:
    """
    Emitted when an incoming signal has near-zero LLR impact across ALL active
    hypotheses on a thread.

    This is the curiosity loop signal: the hypothesis set is incomplete, not
    the evidence weak. The engine surfaces it so the caller can spawn the
    missing binary hypothesis states.
    """
    signal_name: str
    concept_id: int
    value: str
    max_llr_impact: float
    thread_name: str
    message: str


def check_signal_impact(
    conn,
    thread_id: int,
    thread_name: str,
    concept_id: int,
    signal_name: str,
    value: str,
) -> Optional[EvidenceAnomalySignal]:
    """
    Check whether a signal has any meaningful LLR path to the active
    hypotheses on a thread.

    Returns an EvidenceAnomalySignal if max absolute LLR < ANOMALY_LLR_THRESHOLD,
    otherwise None.

    Call this BEFORE inserting the fact so the anomaly is surfaced regardless
    of whether the DB trigger fires.
    """
    try:
        positive_values = ('present', 'yes', 'true', 'high', 'elevated', 'overdue', 'unanswered', 'imminent')
        negative_values = ('absent', 'no', 'false', 'low', 'normal', 'resolved', 'answered')
        v = value.lower()

        if v in positive_values:
            llr_col = "llr_pos"
        elif v in negative_values:
            llr_col = "llr_neg"
        else:
            llr_col = "MAX(ABS(llr_pos), ABS(llr_neg))"

        row = conn.execute(f"""
            SELECT MAX(ABS(r.{llr_col})) as max_impact
            FROM relations r
            JOIN hypotheses h ON h.hypothesis_type_id = r.hypothesis_concept_id
            WHERE r.signal_concept_id = ?
              AND h.thread_id = ?
              AND h.status = 'active'
        """, (concept_id, thread_id)).fetchone()

        max_impact = row["max_impact"] if row and row["max_impact"] is not None else 0.0

        if max_impact < ANOMALY_LLR_THRESHOLD:
            return EvidenceAnomalySignal(
                signal_name=signal_name,
                concept_id=concept_id,
                value=value,
                max_llr_impact=max_impact,
                thread_name=thread_name,
                message=(
                    f'Signal "{signal_name}" (value="{value}") has no high-LLR path to any '
                    f'active hypothesis on thread "{thread_name}" '
                    f'(max_impact={max_impact:.4f} < {ANOMALY_LLR_THRESHOLD}). '
                    'Possible missing hypothesis type — consider whether this signal '
                    'requires a new binary state in the ontology.'
                ),
            )

    except Exception as e:
        logger.debug(f'[AnomalyCheck] Failed for signal "{signal_name}": {e}')

    return None
