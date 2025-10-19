"""Tests for core data structures."""

import pytest

from belief_credence.core import Claim, CredenceEstimate


def test_claim_creation() -> None:
    claim = Claim(statement="The sky is blue.")
    assert claim.statement == "The sky is blue."
    assert claim.negation is None
    assert claim.metadata == {}


def test_claim_with_negation() -> None:
    claim = Claim(
        statement="Paris is the capital of France.",
        negation="Paris is not the capital of France.",
    )
    assert claim.statement == "Paris is the capital of France."
    assert claim.negation == "Paris is not the capital of France."


def test_credence_estimate_valid() -> None:
    claim = Claim(statement="Test claim")
    estimate = CredenceEstimate(p_true=0.75, method="test_method", claim=claim)
    assert estimate.p_true == 0.75
    assert estimate.method == "test_method"
    assert estimate.claim == claim


def test_credence_estimate_invalid_probability() -> None:
    claim = Claim(statement="Test claim")
    with pytest.raises(ValueError, match="p_true must be in"):
        CredenceEstimate(p_true=1.5, method="test_method", claim=claim)

    with pytest.raises(ValueError, match="p_true must be in"):
        CredenceEstimate(p_true=-0.1, method="test_method", claim=claim)


def test_credence_estimate_boundary_values() -> None:
    claim = Claim(statement="Test claim")
    est_zero = CredenceEstimate(p_true=0.0, method="test", claim=claim)
    assert est_zero.p_true == 0.0

    est_one = CredenceEstimate(p_true=1.0, method="test", claim=claim)
    assert est_one.p_true == 1.0
