"""Utilities for extracting and comparing credence signals from LLMs."""

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.ccs import CCS, LayerSearchResult, search_best_layer
from belief_credence.logit_gap import LogitGap
from belief_credence.prompting import DirectPrompting
from belief_credence.comparison import MethodComparison, compare_methods, compare_on_dataset
from belief_credence.model_utils import ModelWrapper
from belief_credence.uncertainty import (
    HallucinationProbe,
    UncertaintyEstimate,
    check_credence_uncertainty_alignment,
)
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
from belief_credence.output_utils import (
    save_estimates,
    load_estimates,
    batch_evaluate,
    batch_evaluate_methods,
    compare_saved_estimates,
)

__all__ = [
    # Core
    "Claim",
    "CredenceEstimate",
    "CredenceMethod",
    # Credence methods (3 direct methods)
    "CCS",
    "LayerSearchResult",
    "search_best_layer",
    "LogitGap",
    "DirectPrompting",
    # Uncertainty estimation (not a direct credence method)
    "HallucinationProbe",
    "UncertaintyEstimate",
    "check_credence_uncertainty_alignment",
    # Comparison utilities
    "MethodComparison",
    "ModelWrapper",
    "compare_methods",
    "compare_on_dataset",
    # Epistemology checks
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
    # Datasets
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
    # Output utilities
    "save_estimates",
    "load_estimates",
    "batch_evaluate",
    "batch_evaluate_methods",
    "compare_saved_estimates",
]
