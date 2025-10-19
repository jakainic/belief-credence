"""Tests for comparison utilities."""

import pytest

from belief_credence.comparison import MethodComparison
from belief_credence.core import Claim, CredenceEstimate


def test_method_comparison_mean() -> None:
    claim = Claim(statement="Test")
    estimates = {
        "method1": CredenceEstimate(p_true=0.6, method="method1", claim=claim),
        "method2": CredenceEstimate(p_true=0.8, method="method2", claim=claim),
        "method3": CredenceEstimate(p_true=0.7, method="method3", claim=claim),
    }
    comparison = MethodComparison(claim=claim, estimates=estimates)

    assert abs(comparison.mean_p_true() - 0.7) < 1e-9


def test_method_comparison_std() -> None:
    claim = Claim(statement="Test")
    estimates = {
        "method1": CredenceEstimate(p_true=0.5, method="method1", claim=claim),
        "method2": CredenceEstimate(p_true=0.5, method="method2", claim=claim),
    }
    comparison = MethodComparison(claim=claim, estimates=estimates)

    assert comparison.std_p_true() == 0.0


def test_method_comparison_range() -> None:
    claim = Claim(statement="Test")
    estimates = {
        "method1": CredenceEstimate(p_true=0.3, method="method1", claim=claim),
        "method2": CredenceEstimate(p_true=0.9, method="method2", claim=claim),
        "method3": CredenceEstimate(p_true=0.5, method="method3", claim=claim),
    }
    comparison = MethodComparison(claim=claim, estimates=estimates)

    p_min, p_max = comparison.range_p_true()
    assert p_min == 0.3
    assert p_max == 0.9


def test_method_comparison_empty() -> None:
    claim = Claim(statement="Test")
    comparison = MethodComparison(claim=claim, estimates={})

    with pytest.raises(ValueError):
        comparison.mean_p_true()

    with pytest.raises(ValueError):
        comparison.range_p_true()
