from __future__ import annotations

from typing import Tuple


def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """Compute posterior P(H|E) given prior P(H), likelihood P(E|H), and evidence P(E).

    Uses Bayes' rule: P(H|E) = P(E|H) P(H) / P(E)
    """
    if not (0.0 <= prior <= 1.0):
        raise ValueError("prior must be in [0, 1]")
    if not (0.0 <= likelihood <= 1.0):
        raise ValueError("likelihood must be in [0, 1]")
    if not (0.0 < evidence <= 1.0):
        raise ValueError("evidence must be in (0, 1]")
    return (likelihood * prior) / evidence


def odds(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("probability must be in (0, 1) for odds")
    return p / (1.0 - p)


def prob_from_odds(o: float) -> float:
    if o <= 0.0:
        raise ValueError("odds must be > 0")
    return o / (1.0 + o)


def bayes_update_via_likelihood_ratio(prior: float, lr: float) -> float:
    """Update prior using likelihood ratio LR = P(E|H)/P(E|~H).
    Returns posterior probability P(H|E).
    """
    if not (0.0 < prior < 1.0):
        raise ValueError("prior must be in (0, 1)")
    if lr <= 0.0:
        raise ValueError("likelihood ratio must be > 0")
    prior_odds = odds(prior)
    post_odds = prior_odds * lr
    return prob_from_odds(post_odds)
