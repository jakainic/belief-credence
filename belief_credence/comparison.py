"""Utilities for comparing credence estimates across methods."""

from __future__ import annotations

from dataclasses import dataclass

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod


@dataclass
class MethodComparison:
    """Results from running multiple methods on the same claim.

    Attributes:
        claim: The claim that was evaluated
        estimates: Dict mapping method name to CredenceEstimate
    """
    claim: Claim
    estimates: dict[str, CredenceEstimate]

    def mean_p_true(self) -> float:
        """Calculate mean P(True) across all methods."""
        if not self.estimates:
            raise ValueError("No estimates to average")
        return sum(est.p_true for est in self.estimates.values()) / len(self.estimates)

    def std_p_true(self) -> float:
        """Calculate standard deviation of P(True) across methods."""
        if len(self.estimates) < 2:
            return 0.0
        mean = self.mean_p_true()
        variance = sum((est.p_true - mean) ** 2 for est in self.estimates.values()) / len(self.estimates)
        return variance ** 0.5

    def range_p_true(self) -> tuple[float, float]:
        """Get min and max P(True) across methods."""
        if not self.estimates:
            raise ValueError("No estimates")
        p_trues = [est.p_true for est in self.estimates.values()]
        return min(p_trues), max(p_trues)


def compare_methods(claim: Claim, methods: list[CredenceMethod]) -> MethodComparison:
    """Run multiple credence methods on the same claim.

    Args:
        claim: The claim to evaluate
        methods: List of credence methods to apply

    Returns:
        MethodComparison with all estimates
    """
    estimates = {}
    for method in methods:
        est = method.estimate(claim)
        estimates[method.name] = est

    return MethodComparison(claim=claim, estimates=estimates)


def compare_on_dataset(claims: list[Claim], methods: list[CredenceMethod]) -> list[MethodComparison]:
    """Run multiple methods on a dataset of claims.

    Args:
        claims: List of claims to evaluate
        methods: List of credence methods to apply

    Returns:
        List of MethodComparison results, one per claim
    """
    return [compare_methods(claim, methods) for claim in claims]
