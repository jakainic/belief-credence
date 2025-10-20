"""Validation utilities for credence estimation methods.

This module provides tools to assess the reliability and validity of
credence estimation methods, including consistency checks, correlation
analysis, and comparison against expected patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from belief_credence.core import CredenceEstimate, Claim
from belief_credence.datasets import BeliefType


@dataclass
class ValidationMetrics:
    """Metrics for validating credence estimation methods.

    Attributes:
        method_name: Name of the method being validated
        mean_by_type: Mean P(True) for each belief type
        std_by_type: Standard deviation for each belief type
        negation_consistency: Average |P(A) + P(¬A) - 1| across claims
        phrasing_coherence: Average std dev across paraphrases of same claim
        expected_pattern_score: How well the method matches expected patterns
    """

    method_name: str
    mean_by_type: Dict[BeliefType, float]
    std_by_type: Dict[BeliefType, float]
    negation_consistency: float | None = None
    phrasing_coherence: float | None = None
    expected_pattern_score: float | None = None


@dataclass
class MethodComparison:
    """Comparison between two credence estimation methods.

    Attributes:
        method1: Name of first method
        method2: Name of second method
        correlation: Pearson correlation between methods
        mean_absolute_error: Average |P1 - P2|
        systematic_bias: Mean(P1 - P2), positive if method1 > method2
        agreement_rate: Fraction of claims where |P1 - P2| < threshold
    """

    method1: str
    method2: str
    correlation: float
    mean_absolute_error: float
    systematic_bias: float
    agreement_rate: float


def validate_method(
    estimates: List[CredenceEstimate],
    agreement_threshold: float = 0.15,
) -> ValidationMetrics:
    """Compute validation metrics for a credence estimation method.

    Args:
        estimates: List of credence estimates from the method
        agreement_threshold: Threshold for counting claims as "in agreement"

    Returns:
        ValidationMetrics with computed scores
    """
    # Group by belief type
    by_type: Dict[BeliefType, List[float]] = {}
    for est in estimates:
        if est.claim.metadata and "belief_type" in est.claim.metadata:
            belief_type = est.claim.metadata["belief_type"]
            if belief_type not in by_type:
                by_type[belief_type] = []
            by_type[belief_type].append(est.p_true)

    mean_by_type = {bt: np.mean(vals) for bt, vals in by_type.items()}
    std_by_type = {bt: np.std(vals) for bt, vals in by_type.items()}

    # Compute expected pattern score
    # Expected patterns:
    # - WELL_ESTABLISHED_FACT: should be high (>0.7)
    # - CONTESTED_FACT: should be middling (0.3-0.7)
    # - CERTAIN_PREDICTION: should be very high (>0.8)
    # - UNCERTAIN_PREDICTION: should vary widely
    pattern_score = 0.0
    count = 0

    if BeliefType.WELL_ESTABLISHED_FACT in mean_by_type:
        # Higher is better, target >0.7
        score = min(mean_by_type[BeliefType.WELL_ESTABLISHED_FACT] / 0.7, 1.0)
        pattern_score += score
        count += 1

    if BeliefType.CERTAIN_PREDICTION in mean_by_type:
        # Higher is better, target >0.8
        score = min(mean_by_type[BeliefType.CERTAIN_PREDICTION] / 0.8, 1.0)
        pattern_score += score
        count += 1

    if BeliefType.CONTESTED_FACT in mean_by_type:
        # Should be middling (0.4-0.6 is ideal)
        mean_val = mean_by_type[BeliefType.CONTESTED_FACT]
        if 0.4 <= mean_val <= 0.6:
            score = 1.0
        else:
            # Penalize deviation from middle
            score = 1.0 - abs(mean_val - 0.5) / 0.5
        pattern_score += score
        count += 1

    expected_pattern_score = pattern_score / count if count > 0 else None

    return ValidationMetrics(
        method_name=estimates[0].method if estimates else "unknown",
        mean_by_type=mean_by_type,
        std_by_type=std_by_type,
        expected_pattern_score=expected_pattern_score,
    )


def compare_methods(
    estimates1: List[CredenceEstimate],
    estimates2: List[CredenceEstimate],
    agreement_threshold: float = 0.15,
) -> MethodComparison:
    """Compare two credence estimation methods.

    Args:
        estimates1: Estimates from first method
        estimates2: Estimates from second method
        agreement_threshold: Threshold for counting as "in agreement"

    Returns:
        MethodComparison with correlation, MAE, bias, and agreement rate

    Raises:
        ValueError: If estimates don't match (different claims or lengths)
    """
    if len(estimates1) != len(estimates2):
        raise ValueError(
            f"Estimate lists have different lengths: {len(estimates1)} vs {len(estimates2)}"
        )

    # Verify same claims
    for e1, e2 in zip(estimates1, estimates2):
        if e1.claim.statement != e2.claim.statement:
            raise ValueError(
                f"Claims don't match: '{e1.claim.statement}' vs '{e2.claim.statement}'"
            )

    p1 = np.array([e.p_true for e in estimates1])
    p2 = np.array([e.p_true for e in estimates2])

    correlation = np.corrcoef(p1, p2)[0, 1]
    mae = np.abs(p1 - p2).mean()
    bias = (p1 - p2).mean()
    agreement_rate = (np.abs(p1 - p2) < agreement_threshold).mean()

    return MethodComparison(
        method1=estimates1[0].method,
        method2=estimates2[0].method,
        correlation=correlation,
        mean_absolute_error=mae,
        systematic_bias=bias,
        agreement_rate=agreement_rate,
    )


def print_validation_report(
    estimates_by_method: Dict[str, List[CredenceEstimate]],
    agreement_threshold: float = 0.15,
) -> None:
    """Print comprehensive validation report for multiple methods.

    Args:
        estimates_by_method: Dict mapping method names to their estimates
        agreement_threshold: Threshold for agreement rate calculation
    """
    print("=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    # Validate each method individually
    print("\n1. INDIVIDUAL METHOD VALIDATION")
    print("-" * 80)

    for method_name, estimates in estimates_by_method.items():
        metrics = validate_method(estimates, agreement_threshold)

        print(f"\n{method_name}:")
        print(f"  Expected Pattern Score: {metrics.expected_pattern_score:.3f}" if metrics.expected_pattern_score else "  Expected Pattern Score: N/A")
        print("\n  Mean P(True) by Belief Type:")
        for belief_type, mean_val in metrics.mean_by_type.items():
            std_val = metrics.std_by_type[belief_type]
            print(f"    {belief_type.value:30s} {mean_val:.3f} ± {std_val:.3f}")

    # Compare methods pairwise
    print("\n\n2. PAIRWISE METHOD COMPARISONS")
    print("-" * 80)

    method_names = list(estimates_by_method.keys())
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i + 1 :]:
            comparison = compare_methods(
                estimates_by_method[method1],
                estimates_by_method[method2],
                agreement_threshold,
            )

            print(f"\n{method1} vs {method2}:")
            print(f"  Correlation:          {comparison.correlation:.3f}")
            print(f"  Mean Absolute Error:  {comparison.mean_absolute_error:.3f}")
            print(f"  Systematic Bias:      {comparison.systematic_bias:+.3f}")
            print(f"  Agreement Rate:       {comparison.agreement_rate:.1%} (within ±{agreement_threshold})")

    # Interpretation guidance
    print("\n\n3. INTERPRETATION GUIDE")
    print("-" * 80)
    print("""
Expected Pattern Score:
  >0.8: Excellent match to expected patterns
  0.6-0.8: Good match, reasonable behavior
  0.4-0.6: Moderate match, some unexpected patterns
  <0.4: Poor match, unexpected behavior

Correlation between methods:
  >0.7: Strong agreement, likely measuring similar signal
  0.4-0.7: Moderate agreement, some shared signal
  <0.4: Weak agreement, measuring different things or unreliable

Mean Absolute Error (MAE):
  <0.1: Very close agreement
  0.1-0.2: Moderate disagreement but still useful
  >0.2: Substantial disagreement, investigate further

Agreement Rate (within ±0.15):
  >70%: High agreement on most claims
  50-70%: Moderate agreement
  <50%: Low agreement, methods diverge significantly
    """)

    print("=" * 80)


def compute_calibration_bins(
    estimates: List[CredenceEstimate],
    ground_truth: List[bool] | None = None,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """Compute calibration statistics (requires ground truth labels).

    Args:
        estimates: List of credence estimates
        ground_truth: List of ground truth labels (True/False)
        n_bins: Number of bins for calibration curve

    Returns:
        Dict with 'bin_centers', 'empirical_freq', and 'counts'

    Note:
        This is only applicable when ground truth is available (e.g., for
        well-established facts or resolved predictions).
    """
    if ground_truth is None:
        raise ValueError("Ground truth labels required for calibration analysis")

    if len(estimates) != len(ground_truth):
        raise ValueError("Estimates and ground truth must have same length")

    p_values = np.array([e.p_true for e in estimates])
    labels = np.array(ground_truth, dtype=float)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    empirical_freq = []
    counts = []

    for i in range(n_bins):
        in_bin = (p_values >= bin_edges[i]) & (p_values < bin_edges[i + 1])
        if i == n_bins - 1:  # Include 1.0 in last bin
            in_bin |= (p_values == 1.0)

        count = in_bin.sum()
        if count > 0:
            freq = labels[in_bin].mean()
        else:
            freq = np.nan

        empirical_freq.append(freq)
        counts.append(count)

    return {
        "bin_centers": bin_centers.tolist(),
        "empirical_freq": empirical_freq,
        "counts": counts,
    }
