"""CCS (Contrast Consistent Search) method for credence estimation."""

from __future__ import annotations

from dataclasses import dataclass

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

    def evaluate_layer_performance(
        self, claims: list[Claim], epochs: int = 100, lr: float = 1e-3
    ) -> float:
        """Evaluate probe performance on a specific layer.

        Args:
            claims: List of claims with negations for evaluation
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Average consistency score (lower is better)
        """
        self.train_probe(claims, epochs=epochs, lr=lr)

        consistency_scores = []
        for claim in claims:
            if claim.negation is None:
                continue

            estimate = self.estimate(claim)
            consistency_score = estimate.metadata.get("consistency_score", 1.0)
            consistency_scores.append(consistency_score)

        if not consistency_scores:
            return 1.0

        return sum(consistency_scores) / len(consistency_scores)


@dataclass
class LayerSearchResult:
    """Result of layer search.

    Attributes:
        layer: Layer index
        consistency_score: Average consistency score on validation set
        probe: Trained probe for this layer
    """

    layer: int
    consistency_score: float
    probe: CCSProbe


def search_best_layer(
    model: ModelWrapper,
    training_claims: list[Claim],
    validation_claims: list[Claim] | None = None,
    layers: list[int] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
) -> LayerSearchResult:
    """Search for the best layer for CCS probe.

    Args:
        model: ModelWrapper to use
        training_claims: Claims for training probe
        validation_claims: Claims for validation (defaults to training_claims)
        layers: List of layer indices to search (defaults to [-1, -2, -3, -4, -5])
        epochs: Number of training epochs per layer
        lr: Learning rate

    Returns:
        LayerSearchResult with best layer, score, and trained probe
    """
    if validation_claims is None:
        validation_claims = training_claims

    if layers is None:
        layers = [-1, -2, -3, -4, -5]

    best_layer = layers[0]
    best_score = float("inf")
    best_probe = None

    for layer in layers:
        ccs = CCS(model=model, layer=layer)
        ccs.train_probe(training_claims, epochs=epochs, lr=lr)

        # Evaluate on validation set
        consistency_scores = []
        for claim in validation_claims:
            if claim.negation is None:
                continue

            estimate = ccs.estimate(claim)
            consistency_score = estimate.metadata.get("consistency_score", 1.0)
            consistency_scores.append(consistency_score)

        if not consistency_scores:
            continue

        avg_score = sum(consistency_scores) / len(consistency_scores)

        if avg_score < best_score:
            best_score = avg_score
            best_layer = layer
            best_probe = ccs._probe

    return LayerSearchResult(
        layer=best_layer, consistency_score=best_score, probe=best_probe
    )
