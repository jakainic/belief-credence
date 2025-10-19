"""Hallucination probe method for credence estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.model_utils import ModelWrapper


class SimpleHallucinationProbe(nn.Module):
    """Simple linear probe for hallucination detection.

    Maps hidden states to hallucination probability.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Hidden states of shape (..., input_dim)

        Returns:
            Hallucination probabilities of shape (..., 1)
        """
        logits = self.linear(x)
        return torch.sigmoid(logits)


class HallucinationProbe(CredenceMethod):
    """Estimate credence using hallucination detection probes.

    Based on https://www.hallucination-probes.com/ and
    https://github.com/obalcells/hallucination_probes

    This method uses linear probes trained on model hidden states to detect
    hallucinated content. We interpret hallucination probability as uncertainty,
    so P(True) ≈ 1 - P(Hallucination).

    Note: This is a simplified implementation. For production use, consider
    loading pretrained probes from the hallucination-probes repository.
    """

    def __init__(
        self,
        model: ModelWrapper | None = None,
        model_name: str = "meta-llama/Llama-2-8b-hf",
        layer: int = -1,
    ):
        """Initialize the hallucination probe method.

        Args:
            model: Pre-loaded ModelWrapper (if None, will load model_name)
            model_name: Name of the model to use
            layer: Which layer to extract activations from (-1 = last layer)
        """
        self.model = model if model is not None else ModelWrapper(model_name)
        self.model_name = model_name
        self.layer = layer
        self._probe: SimpleHallucinationProbe | None = None

    @property
    def name(self) -> str:
        return f"hallucination_probe_{self.model_name.split('/')[-1]}_layer{self.layer}"

    def estimate(self, claim: Claim) -> CredenceEstimate:
        """Estimate P(True) using hallucination probe.

        Args:
            claim: The claim to evaluate

        Returns:
            CredenceEstimate where p_true = 1 - p_hallucination
        """
        hidden_states = self.model.get_hidden_states(claim.statement, self.layer)

        hidden_mean = hidden_states.mean(dim=0).unsqueeze(0)

        if self._probe is None:
            input_dim = hidden_mean.shape[1]
            self._probe = SimpleHallucinationProbe(input_dim).to(self.model.device)
            nn.init.normal_(self._probe.linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self._probe.linear.bias)

        with torch.no_grad():
            p_hallucination = self._probe(hidden_mean).item()

        p_true = 1.0 - p_hallucination

        return CredenceEstimate(
            p_true=p_true,
            method=self.name,
            claim=claim,
            raw_output={"p_hallucination": p_hallucination},
            metadata={
                "layer": self.layer,
                "probe_initialized": self._probe is not None,
                "note": "Using untrained probe - for demo purposes only. "
                "Load pretrained probe for actual use.",
            },
        )

    def load_pretrained_probe(self, probe_path: str) -> None:
        """Load a pretrained hallucination probe.

        Args:
            probe_path: Path to saved probe weights
        """
        state_dict = torch.load(probe_path, map_location=self.model.device)
        if self._probe is None:
            input_dim = state_dict["linear.weight"].shape[1]
            self._probe = SimpleHallucinationProbe(input_dim).to(self.model.device)
        self._probe.load_state_dict(state_dict)
        self._probe.eval()
