"""Logit gap method for credence estimation (Kadavath et al. 2022)."""

from __future__ import annotations

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.model_utils import ModelWrapper


class LogitGap(CredenceMethod):
    """Estimate credence using P(True) and P(False) token probabilities.

    Based on Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
    https://arxiv.org/abs/2207.05221

    This method prompts the model with a claim in a True/False format and extracts
    the logit probabilities for the "True" and "False" tokens to estimate credence.
    """

    def __init__(self, model: ModelWrapper | None = None, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """Initialize the logit gap method.

        Args:
            model: Pre-loaded ModelWrapper (if None, will load model_name)
            model_name: Name of the model to use
        """
        self.model = model if model is not None else ModelWrapper(model_name)
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"logit_gap_{self.model_name.split('/')[-1]}"

    def estimate(self, claim: Claim) -> CredenceEstimate:
        """Estimate P(True) using logit probabilities.

        Args:
            claim: The claim to evaluate

        Returns:
            CredenceEstimate with normalized P(True)
        """
        prompt = self._format_prompt(claim.statement)

        token_probs = self.model.get_token_probabilities(prompt, ["True", "False"])

        p_true_raw = token_probs.get("True", 0.0)
        p_false_raw = token_probs.get("False", 0.0)

        total = p_true_raw + p_false_raw
        if total > 0:
            p_true = p_true_raw / total
        else:
            p_true = 0.5

        return CredenceEstimate(
            p_true=p_true,
            method=self.name,
            claim=claim,
            raw_output=token_probs,
            metadata={
                "prompt": prompt,
                "p_true_raw": p_true_raw,
                "p_false_raw": p_false_raw,
                "normalization_total": total,
            },
        )

    def _format_prompt(self, statement: str) -> str:
        """Format the claim as a True/False question.

        Args:
            statement: The claim statement

        Returns:
            Formatted prompt
        """
        return f"""True or False: {statement}

Answer:"""
