"""Evaluation of Bayesian epistemological properties of credence estimates.

This module provides tools to check whether credence estimates exhibit
rational belief properties:
1. Logical consistency and coherence
2. Action-belief alignment
3. Bayesian conditioning
"""

from __future__ import annotations

from dataclasses import dataclass

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod


@dataclass
class ConsistencyCheck:
    """Results of logical consistency evaluation.

    Checks whether P(statement) + P(negation) ≈ 1 for contrast pairs.
    """

    claim: Claim
    p_statement: float
    p_negation: float
    sum_probability: float
    deviation_from_one: float
    is_consistent: bool

    @property
    def consistency_score(self) -> float:
        """Score from 0 (inconsistent) to 1 (perfectly consistent)."""
        return max(0.0, 1.0 - abs(self.deviation_from_one))


@dataclass
class CoherenceCheck:
    """Results of coherence evaluation across paraphrases.

    Checks whether semantically equivalent statements get similar credences.
    """

    original_claim: Claim
    paraphrases: list[Claim]
    original_estimate: float
    paraphrase_estimates: list[float]
    mean_estimate: float
    std_deviation: float
    max_deviation: float
    is_coherent: bool

    @property
    def coherence_score(self) -> float:
        """Score from 0 (incoherent) to 1 (perfectly coherent).

        Based on standard deviation - lower std means higher coherence.
        """
        return max(0.0, 1.0 - self.std_deviation)


@dataclass
class BayesianConditioningCheck:
    """Results of Bayesian conditioning evaluation.

    Checks whether P(A|B) ≈ P(A ∧ B) / P(B).
    """

    proposition: str
    evidence: str
    p_a_given_b: float
    p_a_and_b: float
    p_b: float
    expected_p_a_given_b: float
    deviation: float
    is_bayesian: bool

    @property
    def bayesian_score(self) -> float:
        """Score from 0 (non-Bayesian) to 1 (perfectly Bayesian)."""
        return max(0.0, 1.0 - abs(self.deviation))


def check_consistency(
    method: CredenceMethod, claim: Claim, threshold: float = 0.2
) -> ConsistencyCheck:
    """Check if P(statement) + P(negation) ≈ 1.

    Args:
        method: Credence method to evaluate
        claim: Claim with negation
        threshold: Maximum acceptable deviation from 1.0

    Returns:
        ConsistencyCheck with results

    Raises:
        ValueError: If claim has no negation
    """
    if claim.negation is None:
        raise ValueError("Claim must have negation for consistency check")

    estimate_pos = method.estimate(claim)
    p_statement = estimate_pos.p_true

    negated_claim = Claim(statement=claim.negation, negation=claim.statement)
    estimate_neg = method.estimate(negated_claim)
    p_negation = estimate_neg.p_true

    sum_prob = p_statement + p_negation
    deviation = abs(sum_prob - 1.0)

    return ConsistencyCheck(
        claim=claim,
        p_statement=p_statement,
        p_negation=p_negation,
        sum_probability=sum_prob,
        deviation_from_one=deviation,
        is_consistent=deviation <= threshold,
    )


def check_coherence(
    method: CredenceMethod,
    original_claim: Claim,
    paraphrases: list[str],
    threshold: float = 0.15,
) -> CoherenceCheck:
    """Check if paraphrases of a claim get similar credences.

    Args:
        method: Credence method to evaluate
        original_claim: Original claim
        paraphrases: List of paraphrase strings
        threshold: Maximum acceptable standard deviation

    Returns:
        CoherenceCheck with results
    """
    original_estimate = method.estimate(original_claim).p_true

    paraphrase_claims = [Claim(statement=p) for p in paraphrases]
    paraphrase_estimates = [method.estimate(c).p_true for c in paraphrase_claims]

    all_estimates = [original_estimate] + paraphrase_estimates
    mean_est = sum(all_estimates) / len(all_estimates)

    variance = sum((e - mean_est) ** 2 for e in all_estimates) / len(all_estimates)
    std_dev = variance**0.5

    max_dev = max(abs(e - original_estimate) for e in paraphrase_estimates)

    return CoherenceCheck(
        original_claim=original_claim,
        paraphrases=paraphrase_claims,
        original_estimate=original_estimate,
        paraphrase_estimates=paraphrase_estimates,
        mean_estimate=mean_est,
        std_deviation=std_dev,
        max_deviation=max_dev,
        is_coherent=std_dev <= threshold,
    )


def check_bayesian_conditioning(
    method: CredenceMethod,
    proposition: str,
    evidence: str,
    threshold: float = 0.2,
) -> BayesianConditioningCheck:
    """Check if P(A|B) ≈ P(A ∧ B) / P(B).

    Args:
        method: Credence method to evaluate
        proposition: Proposition A
        evidence: Evidence B
        threshold: Maximum acceptable deviation

    Returns:
        BayesianConditioningCheck with results
    """
    p_a_given_b_claim = Claim(statement=f"Given that {evidence}, {proposition}")
    p_a_given_b = method.estimate(p_a_given_b_claim).p_true

    p_a_and_b_claim = Claim(statement=f"{proposition} AND {evidence}")
    p_a_and_b = method.estimate(p_a_and_b_claim).p_true

    p_b_claim = Claim(statement=evidence)
    p_b = method.estimate(p_b_claim).p_true

    if p_b < 0.01:
        expected = 0.5
    else:
        expected = p_a_and_b / p_b

    expected = min(1.0, max(0.0, expected))

    deviation = abs(p_a_given_b - expected)

    return BayesianConditioningCheck(
        proposition=proposition,
        evidence=evidence,
        p_a_given_b=p_a_given_b,
        p_a_and_b=p_a_and_b,
        p_b=p_b,
        expected_p_a_given_b=expected,
        deviation=deviation,
        is_bayesian=deviation <= threshold,
    )


@dataclass
class ActionCorrelationCheck:
    """Results of action-belief correlation evaluation.

    Checks whether internal credence correlates with model's chosen actions/outputs.
    """

    claim: Claim
    internal_credence: float
    action_probability: float
    correlation_score: float
    is_aligned: bool


def check_action_correlation(
    method: CredenceMethod,
    claim: Claim,
    action_prompt: str,
    threshold: float = 0.2,
) -> ActionCorrelationCheck:
    """Check if internal credence correlates with action probabilities.

    Args:
        method: Credence method to evaluate
        claim: Claim to evaluate
        action_prompt: Prompt that asks model to take action based on claim
            (e.g., "Would you bet $100 that: {claim.statement}?")
        threshold: Maximum acceptable deviation for alignment

    Returns:
        ActionCorrelationCheck with results
    """
    internal_credence = method.estimate(claim).p_true

    action_claim = Claim(statement=action_prompt.format(claim=claim))
    action_estimate = method.estimate(action_claim)
    action_probability = action_estimate.p_true

    correlation_score = 1.0 - abs(internal_credence - action_probability)

    return ActionCorrelationCheck(
        claim=claim,
        internal_credence=internal_credence,
        action_probability=action_probability,
        correlation_score=correlation_score,
        is_aligned=abs(internal_credence - action_probability) <= threshold,
    )


@dataclass
class EpistemologyReport:
    """Comprehensive report on epistemological properties."""

    method_name: str
    consistency_checks: list[ConsistencyCheck]
    coherence_checks: list[CoherenceCheck]
    bayesian_checks: list[BayesianConditioningCheck]
    action_checks: list[ActionCorrelationCheck]

    @property
    def overall_consistency_score(self) -> float:
        """Average consistency score across all checks."""
        if not self.consistency_checks:
            return 0.0
        return sum(c.consistency_score for c in self.consistency_checks) / len(
            self.consistency_checks
        )

    @property
    def overall_coherence_score(self) -> float:
        """Average coherence score across all checks."""
        if not self.coherence_checks:
            return 0.0
        return sum(c.coherence_score for c in self.coherence_checks) / len(self.coherence_checks)

    @property
    def overall_bayesian_score(self) -> float:
        """Average Bayesian conditioning score."""
        if not self.bayesian_checks:
            return 0.0
        return sum(c.bayesian_score for c in self.bayesian_checks) / len(self.bayesian_checks)

    @property
    def overall_action_score(self) -> float:
        """Average action correlation score."""
        if not self.action_checks:
            return 0.0
        return sum(c.correlation_score for c in self.action_checks) / len(self.action_checks)

    @property
    def overall_epistemology_score(self) -> float:
        """Overall epistemology score (average of all dimensions)."""
        scores = []
        if self.consistency_checks:
            scores.append(self.overall_consistency_score)
        if self.coherence_checks:
            scores.append(self.overall_coherence_score)
        if self.bayesian_checks:
            scores.append(self.overall_bayesian_score)
        if self.action_checks:
            scores.append(self.overall_action_score)

        return sum(scores) / len(scores) if scores else 0.0


def evaluate_epistemology(
    method: CredenceMethod,
    consistency_claims: list[Claim] | None = None,
    coherence_tests: list[tuple[Claim, list[str]]] | None = None,
    bayesian_tests: list[tuple[str, str]] | None = None,
    action_tests: list[tuple[Claim, str]] | None = None,
) -> EpistemologyReport:
    """Run comprehensive epistemological evaluation.

    Args:
        method: Credence method to evaluate
        consistency_claims: Claims with negations for consistency checks
        coherence_tests: (claim, paraphrases) tuples for coherence checks
        bayesian_tests: (proposition, evidence) tuples for Bayesian checks
        action_tests: (claim, action_prompt) tuples for action correlation

    Returns:
        EpistemologyReport with all results
    """
    consistency_results = []
    if consistency_claims:
        for claim in consistency_claims:
            consistency_results.append(check_consistency(method, claim))

    coherence_results = []
    if coherence_tests:
        for claim, paraphrases in coherence_tests:
            coherence_results.append(check_coherence(method, claim, paraphrases))

    bayesian_results = []
    if bayesian_tests:
        for prop, evidence in bayesian_tests:
            bayesian_results.append(check_bayesian_conditioning(method, prop, evidence))

    action_results = []
    if action_tests:
        for claim, prompt in action_tests:
            action_results.append(check_action_correlation(method, claim, prompt))

    return EpistemologyReport(
        method_name=method.name,
        consistency_checks=consistency_results,
        coherence_checks=coherence_results,
        bayesian_checks=bayesian_results,
        action_checks=action_results,
    )
