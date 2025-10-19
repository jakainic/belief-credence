import math
import pytest

from belief_credence.core import bayes_posterior, bayes_update_via_likelihood_ratio, odds, prob_from_odds


def test_bayes_posterior_basic():
    # P(H)=0.3, P(E|H)=0.8, P(E)=0.5 -> 0.48
    assert math.isclose(bayes_posterior(0.3, 0.8, 0.5), 0.48, rel_tol=1e-9)


@pytest.mark.parametrize("prior,lr,expected", [
    (0.5, 1.0, 0.5),
    (0.5, 3.0, 0.75),
    (0.2, 4.0, 0.5),
])
def test_bayes_update_via_likelihood_ratio(prior, lr, expected):
    assert math.isclose(bayes_update_via_likelihood_ratio(prior, lr), expected, rel_tol=1e-9)


def test_odds_roundtrip():
    p = 0.37
    o = odds(p)
    assert math.isclose(prob_from_odds(o), p, rel_tol=1e-12)


@pytest.mark.parametrize("prior", [-0.1, 0.0, 1.0, 1.1])
def test_invalid_prior_for_lr_update(prior):
    with pytest.raises(ValueError):
        bayes_update_via_likelihood_ratio(prior, 2.0)
