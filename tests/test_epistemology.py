"""Tests for epistemology evaluation module."""

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.epistemology import (
    check_action_correlation,
    check_bayesian_conditioning,
    check_coherence,
    check_consistency,
    evaluate_epistemology,
)


class MockMethod(CredenceMethod):
    """Mock credence method for testing."""

    def __init__(self, responses: dict[str, float]):
        """Initialize with fixed responses.

        Args:
            responses: Dict mapping statement strings to p_true values
        """
        self.responses = responses

    @property
    def name(self) -> str:
        return "mock_method"

    def estimate(self, claim: Claim) -> CredenceEstimate:
        p_true = self.responses.get(claim.statement, 0.5)
        return CredenceEstimate(p_true=p_true, method=self.name, claim=claim)


def test_consistency_check_perfect() -> None:
    method = MockMethod(
        {"Sky is blue": 0.9, "Sky is not blue": 0.1}
    )

    claim = Claim(statement="Sky is blue", negation="Sky is not blue")
    result = check_consistency(method, claim)

    assert result.p_statement == 0.9
    assert result.p_negation == 0.1
    assert result.sum_probability == 1.0
    assert result.deviation_from_one == 0.0
    assert result.is_consistent
    assert result.consistency_score == 1.0


def test_consistency_check_imperfect() -> None:
    method = MockMethod(
        {"Sky is blue": 0.8, "Sky is not blue": 0.3}
    )

    claim = Claim(statement="Sky is blue", negation="Sky is not blue")
    result = check_consistency(method, claim, threshold=0.2)

    assert result.sum_probability == 1.1
    assert result.deviation_from_one == 0.1
    assert result.is_consistent
    assert result.consistency_score == 0.9


def test_consistency_check_inconsistent() -> None:
    method = MockMethod(
        {"Sky is blue": 0.9, "Sky is not blue": 0.8}
    )

    claim = Claim(statement="Sky is blue", negation="Sky is not blue")
    result = check_consistency(method, claim, threshold=0.2)

    assert result.sum_probability == 1.7
    assert result.deviation_from_one == 0.7
    assert not result.is_consistent


def test_coherence_check_perfect() -> None:
    method = MockMethod(
        {
            "Paris is in France": 0.95,
            "Paris is located in France": 0.95,
            "The city of Paris is in France": 0.95,
        }
    )

    claim = Claim(statement="Paris is in France")
    paraphrases = [
        "Paris is located in France",
        "The city of Paris is in France",
    ]

    result = check_coherence(method, claim, paraphrases)

    assert result.original_estimate == 0.95
    assert result.mean_estimate == 0.95
    assert result.std_deviation == 0.0
    assert result.is_coherent
    assert result.coherence_score == 1.0


def test_coherence_check_with_variation() -> None:
    method = MockMethod(
        {
            "Paris is in France": 0.9,
            "Paris is located in France": 0.85,
            "The city of Paris is in France": 0.95,
        }
    )

    claim = Claim(statement="Paris is in France")
    paraphrases = [
        "Paris is located in France",
        "The city of Paris is in France",
    ]

    result = check_coherence(method, claim, paraphrases, threshold=0.1)

    assert abs(result.mean_estimate - 0.9) < 0.01
    assert result.std_deviation < 0.1
    assert result.is_coherent


def test_bayesian_conditioning_check() -> None:
    method = MockMethod(
        {
            "Given that it is raining, the ground is wet": 0.9,
            "the ground is wet AND it is raining": 0.7,
            "it is raining": 0.8,
        }
    )

    result = check_bayesian_conditioning(
        method, "the ground is wet", "it is raining"
    )

    assert result.p_a_given_b == 0.9
    assert result.p_a_and_b == 0.7
    assert result.p_b == 0.8
    assert abs(result.expected_p_a_given_b - 0.875) < 0.01


def test_action_correlation_check() -> None:
    method = MockMethod(
        {
            "Paris is in France": 0.9,
            "Would you bet on: Paris is in France": 0.85,
        }
    )

    claim = Claim(statement="Paris is in France")
    result = check_action_correlation(
        method, claim, "Would you bet on: {claim.statement}"
    )

    assert result.internal_credence == 0.9
    assert result.action_probability == 0.85
    assert abs(result.correlation_score - 0.95) < 0.01
    assert result.is_aligned


def test_evaluate_epistemology_comprehensive() -> None:
    method = MockMethod(
        {
            "Sky is blue": 0.9,
            "Sky is not blue": 0.1,
            "Paris is in France": 0.95,
            "Paris is located in France": 0.95,
        }
    )

    consistency_claims = [
        Claim(statement="Sky is blue", negation="Sky is not blue")
    ]

    coherence_tests = [
        (Claim(statement="Paris is in France"), ["Paris is located in France"])
    ]

    report = evaluate_epistemology(
        method,
        consistency_claims=consistency_claims,
        coherence_tests=coherence_tests,
    )

    assert report.method_name == "mock_method"
    assert len(report.consistency_checks) == 1
    assert len(report.coherence_checks) == 1
    assert report.overall_consistency_score == 1.0
    assert report.overall_coherence_score == 1.0
    assert 0.0 <= report.overall_epistemology_score <= 1.0
