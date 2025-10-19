"""Uncertainty and confidence estimation for model outputs.

This module provides tools for estimating model uncertainty/confidence
as a complement to credence extraction. Uncertainty scores can be used
to validate or modulate credence estimates from other methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from belief_credence.core import Claim
from belief_credence.model_utils import ModelWrapper


class UncertaintyProbe(nn.Module):
    """Linear probe for uncertainty estimation.

    Maps hidden states to uncertainty/hallucination probability.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Hidden states of shape (..., input_dim)

        Returns:
            Uncertainty probabilities of shape (..., 1)
        """
        logits = self.linear(x)
        return torch.sigmoid(logits)


@dataclass
class UncertaintyEstimate:
    """Result of uncertainty estimation.

    Attributes:
        claim: The claim being evaluated
        uncertainty_score: Uncertainty/hallucination probability (0-1)
            Higher values indicate more uncertainty
        confidence_score: Confidence score (1 - uncertainty_score)
            Higher values indicate more confidence
        method: Name of uncertainty estimation method
        raw_output: Method-specific raw output
        metadata: Additional information
    """

    claim: Claim
    uncertainty_score: float
    confidence_score: float
    method: str
    raw_output: dict[str, float] | None = None
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty_score <= 1.0:
            raise ValueError(f"uncertainty_score must be in [0,1], got {self.uncertainty_score}")
        if self.metadata is None:
            self.metadata = {}
        if self.raw_output is None:
            self.raw_output = {}


class HallucinationProbe:
    """Estimate model uncertainty using hallucination detection probes.

    Based on https://www.hallucination-probes.com/ and
    https://github.com/obalcells/hallucination_probes

    This estimates model uncertainty/confidence rather than direct credence.
    High uncertainty suggests the model is not confident in its answer,
    which can be used to validate or modulate credence estimates from
    other methods.

    Note: This is a simplified implementation. For production use, consider
    loading pretrained probes from the hallucination-probes repository.
    """

    def __init__(
        self,
        model: ModelWrapper | None = None,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        layer: int = -1,
    ):
        """Initialize the hallucination probe.

        Args:
            model: Pre-loaded ModelWrapper (if None, will load model_name)
            model_name: Name of the model to use
            layer: Which layer to extract activations from (-1 = last layer)
        """
        self.model = model if model is not None else ModelWrapper(model_name)
        self.model_name = model_name
        self.layer = layer
        self._probe: UncertaintyProbe | None = None

    @property
    def name(self) -> str:
        return f"hallucination_probe_{self.model_name.split('/')[-1]}_layer{self.layer}"

    def estimate_uncertainty(self, claim: Claim) -> UncertaintyEstimate:
        """Estimate uncertainty for a claim.

        Args:
            claim: The claim to evaluate

        Returns:
            UncertaintyEstimate with uncertainty and confidence scores
        """
        hidden_states = self.model.get_hidden_states(claim.statement, self.layer)
        hidden_mean = hidden_states.mean(dim=0).unsqueeze(0)

        if self._probe is None:
            input_dim = hidden_mean.shape[1]
            self._probe = UncertaintyProbe(input_dim).to(self.model.device)
            nn.init.normal_(self._probe.linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self._probe.linear.bias)

        with torch.no_grad():
            uncertainty = self._probe(hidden_mean).item()

        confidence = 1.0 - uncertainty

        return UncertaintyEstimate(
            claim=claim,
            uncertainty_score=uncertainty,
            confidence_score=confidence,
            method=self.name,
            raw_output={"uncertainty": uncertainty, "confidence": confidence},
            metadata={
                "layer": str(self.layer),
                "note": "Using untrained probe - for demo purposes only. "
                "Load pretrained probe for actual use.",
            },
        )

    def load_pretrained_probe(self, probe_path: str) -> None:
        """Load a pretrained uncertainty/hallucination probe.

        Args:
            probe_path: Path to saved probe weights
        """
        state_dict = torch.load(probe_path, map_location=self.model.device)
        if self._probe is None:
            input_dim = state_dict["linear.weight"].shape[1]
            self._probe = UncertaintyProbe(input_dim).to(self.model.device)
        self._probe.load_state_dict(state_dict)
        self._probe.eval()


def check_credence_uncertainty_alignment(
    p_true: float, uncertainty_estimate: UncertaintyEstimate, threshold: float = 0.3
) -> bool:
    """Check if credence estimate aligns with uncertainty.

    High confidence (low uncertainty) should correlate with credences
    far from 0.5. High uncertainty should correlate with credences
    near 0.5.

    Args:
        p_true: Credence estimate from another method
        uncertainty_estimate: Uncertainty estimate
        threshold: Threshold for considering alignment

    Returns:
        True if credence and uncertainty are aligned
    """
    deviation_from_half = abs(p_true - 0.5)
    confidence = uncertainty_estimate.confidence_score

    expected_confidence = deviation_from_half * 2

    return abs(confidence - expected_confidence) <= threshold
