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

    To ensure the probe learns "what the model believes is true" rather than
    an arbitrary direction, we use a reference method (logit gap or self-labeling)
    to orient the training data: "positive" examples are statements the model
    believes, "negative" examples are statements it doesn't believe.
    """

    def __init__(
        self,
        model: ModelWrapper | None = None,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        layer: int = -1,
        direction_method: str = "logit_gap",
    ):
        """Initialize the CCS method.

        Args:
            model: Pre-loaded ModelWrapper (if None, will load model_name)
            model_name: Name of the model to use
            layer: Which layer to extract activations from (-1 = last layer)
            direction_method: How to determine probe direction ("logit_gap" or "self_label")
        """
        self.model = model if model is not None else ModelWrapper(model_name)
        self.model_name = model_name
        self.layer = layer
        self.direction_method = direction_method
        self._probe: CCSProbe | None = None

    @property
    def name(self) -> str:
        return f"ccs_{self.model_name.split('/')[-1]}_layer{self.layer}"

    def _get_model_belief_logit_gap(self, claim: Claim) -> bool:
        """Determine if model believes positive statement using logit gap.

        Args:
            claim: Claim with statement and negation

        Returns:
            True if model believes positive statement, False otherwise
        """
        from belief_credence.logit_gap import LogitGap

        logit_method = LogitGap(model=self.model)
        estimate = logit_method.estimate(claim)
        return estimate.p_true > 0.5

    def _get_model_belief_self_label(self, claim: Claim) -> bool:
        """Determine if model believes positive statement by asking it directly.

        Args:
            claim: Claim with statement and negation

        Returns:
            True if model believes positive statement, False otherwise
        """
        prompt = f"""Which statement is true?
A: {claim.statement}
B: {claim.negation}

Answer:"""

        token_probs = self.model.get_token_probabilities(prompt, ["A", "B"])
        p_a = token_probs.get("A", 0.0)
        p_b = token_probs.get("B", 0.0)
        return p_a > p_b

    def _get_model_belief(self, claim: Claim) -> bool:
        """Determine if model believes positive statement.

        Args:
            claim: Claim with statement and negation

        Returns:
            True if model believes positive statement, False otherwise
        """
        if self.direction_method == "logit_gap":
            return self._get_model_belief_logit_gap(claim)
        elif self.direction_method == "self_label":
            return self._get_model_belief_self_label(claim)
        else:
            raise ValueError(
                f"Unknown direction_method: {self.direction_method}. "
                f"Must be 'logit_gap' or 'self_label'"
            )

    def estimate(self, claim: Claim) -> CredenceEstimate:
        """Estimate P(True) using CCS probe.

        Once trained, the probe can evaluate single statements independently.
        No negation required - the probe already learned the correct direction
        during training.

        Args:
            claim: The claim to evaluate

        Returns:
            CredenceEstimate with CCS-based p_true
        """
        if self._probe is None:
            if claim.negation is None:
                raise ValueError(
                    "CCS probe not trained. Either train_probe() first or "
                    "provide claim with negation for automatic training."
                )
            self.train_probe([claim])

        # Get activations for just the statement
        hidden = self.model.get_hidden_states(claim.statement, self.layer)
        hidden_mean = hidden.mean(dim=0).unsqueeze(0)

        # Convert to float32 and move to probe device
        hidden_mean = hidden_mean.to(device=self.model.device, dtype=torch.float32)

        # Apply trained probe
        with torch.no_grad():
            p_true = self._probe(hidden_mean).item()

        return CredenceEstimate(
            p_true=p_true,
            method=self.name,
            claim=claim,
            raw_output={"p_true": p_true},
            metadata={
                "layer": self.layer,
                "direction_method": self.direction_method,
            },
        )

    def train_probe(
        self, claims: list[Claim], epochs: int = 100, lr: float = 1e-3
    ) -> None:
        """Train the CCS probe on a set of contrast pairs.

        The probe is trained to output high values for statements the model
        believes are true and low values for statements it believes are false.
        Training data is reordered based on the model's beliefs (determined by
        the direction_method) to ensure the probe learns the correct direction.

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

            # Determine which statement the model believes is true
            model_believes_positive = self._get_model_belief(claim)

            pos_hidden, neg_hidden = self.model.get_contrast_activations(
                claim.statement, claim.negation, self.layer
            )

            # Reorder: "positive" = what model believes, "negative" = what it doesn't
            if model_believes_positive:
                activations_pos.append(pos_hidden.mean(dim=0))
                activations_neg.append(neg_hidden.mean(dim=0))
            else:
                # Swap them - model believes the negation
                activations_pos.append(neg_hidden.mean(dim=0))
                activations_neg.append(pos_hidden.mean(dim=0))

        if not activations_pos:
            raise ValueError("No valid contrast pairs found for training")

        X_pos = torch.stack(activations_pos)
        X_neg = torch.stack(activations_neg)

        input_dim = X_pos.shape[1]
        # Create probe in float32 on model device
        self._probe = CCSProbe(input_dim).to(self.model.device)

        # Convert activations to float32 and move to same device as probe
        # Do this AFTER probe creation to ensure compatibility
        X_pos = X_pos.to(device=self.model.device, dtype=torch.float32)
        X_neg = X_neg.to(device=self.model.device, dtype=torch.float32)

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
    direction_method: str = "logit_gap",
) -> LayerSearchResult:
    """Search for the best layer for CCS probe.

    Args:
        model: ModelWrapper to use
        training_claims: Claims for training probe
        validation_claims: Claims for validation (defaults to training_claims)
        layers: List of layer indices to search (defaults to [-1, -2, -3, -4, -5])
        epochs: Number of training epochs per layer
        lr: Learning rate
        direction_method: How to determine probe direction ("logit_gap" or "self_label")

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
        ccs = CCS(model=model, layer=layer, direction_method=direction_method)
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
