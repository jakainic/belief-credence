"""CCS (Contrast Consistent Search) method for credence estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.model_utils import ModelWrapper


class CCSProbe(nn.Module):
    """Linear probe for CCS.

    Maps hidden states to a probability via a single linear layer.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through probe.

        Args:
            x: Hidden states of shape (..., input_dim)

        Returns:
            Probabilities of shape (..., 1)
        """
        logits = self.linear(x)
        return torch.sigmoid(logits)


class CCS(CredenceMethod):
    """Estimate credence using Contrast Consistent Search.

    Based on Burns et al. (2022) "Discovering Latent Knowledge in Language
    Models Without Supervision"
    https://arxiv.org/abs/2212.03827

    This method finds a direction in the model's activation space that is
    consistent across contrast pairs (statement and its negation). The direction
    is learned to satisfy logical consistency properties without supervision.
    """

    def __init__(
        self,
        model: ModelWrapper | None = None,
        model_name: str = "meta-llama/Llama-2-8b-hf",
        layer: int = -1,
    ):
        """Initialize the CCS method.

        Args:
            model: Pre-loaded ModelWrapper (if None, will load model_name)
            model_name: Name of the model to use
            layer: Which layer to extract activations from (-1 = last layer)
        """
        self.model = model if model is not None else ModelWrapper(model_name)
        self.model_name = model_name
        self.layer = layer
        self._probe: CCSProbe | None = None

    @property
    def name(self) -> str:
        return f"ccs_{self.model_name.split('/')[-1]}_layer{self.layer}"

    def estimate(self, claim: Claim) -> CredenceEstimate:
        """Estimate P(True) using CCS probe.

        Args:
            claim: The claim to evaluate (must have negation)

        Returns:
            CredenceEstimate with CCS-based p_true
        """
        if claim.negation is None:
            raise ValueError("CCS requires claim negation to be provided")

        if self._probe is None:
            self.train_probe([claim])

        pos_hidden, neg_hidden = self.model.get_contrast_activations(
            claim.statement, claim.negation, self.layer
        )

        pos_hidden_mean = pos_hidden.mean(dim=0).unsqueeze(0)
        neg_hidden_mean = neg_hidden.mean(dim=0).unsqueeze(0)

        with torch.no_grad():
            p_pos = self._probe(pos_hidden_mean).item()
            p_neg = self._probe(neg_hidden_mean).item()

        p_true = p_pos

        return CredenceEstimate(
            p_true=p_true,
            method=self.name,
            claim=claim,
            raw_output={"p_pos": p_pos, "p_neg": p_neg},
            metadata={
                "layer": self.layer,
                "consistency_score": abs(p_pos + p_neg - 1.0),
            },
        )

    def train_probe(
        self, claims: list[Claim], epochs: int = 100, lr: float = 1e-3
    ) -> None:
        """Train the CCS probe on a set of contrast pairs.

        Args:
            claims: List of claims with negations for training
            epochs: Number of training epochs
            lr: Learning rate
        """
        activations_pos = []
        activations_neg = []

        for claim in claims:
            if claim.negation is None:
                continue

            pos_hidden, neg_hidden = self.model.get_contrast_activations(
                claim.statement, claim.negation, self.layer
            )

            activations_pos.append(pos_hidden.mean(dim=0))
            activations_neg.append(neg_hidden.mean(dim=0))

        if not activations_pos:
            raise ValueError("No valid contrast pairs found for training")

        X_pos = torch.stack(activations_pos)
        X_neg = torch.stack(activations_neg)

        input_dim = X_pos.shape[1]
        self._probe = CCSProbe(input_dim).to(self.model.device)

        optimizer = torch.optim.Adam(self._probe.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            p_pos = self._probe(X_pos)
            p_neg = self._probe(X_neg)

            consistency_loss = torch.mean((p_pos + p_neg - 1.0) ** 2)

            confidence_loss = -torch.mean(
                torch.log(p_pos + 1e-8) + torch.log(1 - p_pos + 1e-8)
            )
            confidence_loss += -torch.mean(
                torch.log(p_neg + 1e-8) + torch.log(1 - p_neg + 1e-8)
            )

            loss = consistency_loss + 0.1 * confidence_loss

            loss.backward()
            optimizer.step()
