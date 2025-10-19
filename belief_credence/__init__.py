"""Utilities for extracting and comparing credence signals from LLMs."""

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.ccs import CCS
from belief_credence.logit_gap import LogitGap
from belief_credence.hallucination_probe import HallucinationProbe
from belief_credence.prompting import DirectPrompting
from belief_credence.comparison import MethodComparison, compare_methods, compare_on_dataset
from belief_credence.model_utils import ModelWrapper
from belief_credence.epistemology import (
    ConsistencyCheck,
    CoherenceCheck,
    BayesianConditioningCheck,
    ActionCorrelationCheck,
    EpistemologyReport,
    check_consistency,
    check_coherence,
    check_bayesian_conditioning,
    check_action_correlation,
    evaluate_epistemology,
)
from belief_credence.datasets import (
    BeliefType,
    ClaimSet,
    get_dataset,
    get_all_claims,
    get_all_claim_sets,
    WELL_ESTABLISHED_FACTS,
    CONTESTED_FACTS,
    CERTAIN_PREDICTIONS,
    UNCERTAIN_PREDICTIONS,
    NORMATIVE_JUDGMENTS,
    METAPHYSICAL_BELIEFS,
)

__all__ = [
    "Claim",
    "CredenceEstimate",
    "CredenceMethod",
    "CCS",
    "LogitGap",
    "HallucinationProbe",
    "DirectPrompting",
    "MethodComparison",
    "ModelWrapper",
    "compare_methods",
    "compare_on_dataset",
    "ConsistencyCheck",
    "CoherenceCheck",
    "BayesianConditioningCheck",
    "ActionCorrelationCheck",
    "EpistemologyReport",
    "check_consistency",
    "check_coherence",
    "check_bayesian_conditioning",
    "check_action_correlation",
    "evaluate_epistemology",
    "BeliefType",
    "ClaimSet",
    "get_dataset",
    "get_all_claims",
    "get_all_claim_sets",
    "WELL_ESTABLISHED_FACTS",
    "CONTESTED_FACTS",
    "CERTAIN_PREDICTIONS",
    "UNCERTAIN_PREDICTIONS",
    "NORMATIVE_JUDGMENTS",
    "METAPHYSICAL_BELIEFS",
]
