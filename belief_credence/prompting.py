"""Direct prompting method for credence estimation."""

from __future__ import annotations

import re

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.model_utils import ModelWrapper


class DirectPrompting(CredenceMethod):
    """Estimate credence by directly prompting the model for a probability.

    This is the simplest baseline method. We ask the model to output
    a credence level between 0 and 1 for a given claim.

    WARNING: This method works poorly with base models (like Llama-2-7b-hf)
    and requires instruction-tuned models (like Llama-2-7b-chat-hf) to produce
    reliable numerical outputs. Base models often default to repeating prompt
    examples or generating conversational text instead of probabilities.

    For base models, prefer LogitGap or CCS methods instead.
    """

    def __init__(
        self,
        model: ModelWrapper | None = None,
        model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        temperature: float = 0.0,
    ):
        """Initialize the direct prompting method.

        Args:
            model: Pre-loaded ModelWrapper (if None, will load model_name)
            model_name: Name of the model to use if model not provided
            temperature: Sampling temperature (0 for deterministic)
        """
        self.model = model if model is not None else ModelWrapper(model_name)
        self.model_name = model_name
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"direct_prompting_{self.model_name.split('/')[-1]}"

    def estimate(self, claim: Claim) -> CredenceEstimate:
        """Estimate P(True) by directly prompting the model.

        Args:
            claim: The claim to evaluate

        Returns:
            CredenceEstimate with p_true from model response
        """
        prompt = self._build_prompt(claim.statement)
        response = self.model.generate(prompt, max_new_tokens=10, temperature=self.temperature)

        p_true = self._parse_credence(response)

        return CredenceEstimate(
            p_true=p_true,
            method=self.name,
            claim=claim,
            raw_output=response,
            metadata={"prompt": prompt, "temperature": self.temperature},
        )

    def _build_prompt(self, statement: str) -> str:
        """Build the prompt asking for credence estimation.

        Args:
            statement: The claim statement

        Returns:
            Formatted prompt string
        """
        return f"""Q: What is the probability that the following claim is true? Answer with only a number between 0 and 1.

Claim: "{statement}"

A:"""

    def _parse_credence(self, response: str) -> float:
        """Parse credence value from model response.

        Args:
            response: Model's text response

        Returns:
            Parsed probability value

        Raises:
            ValueError: If no valid credence found in response
        """
        match = re.search(r"0?\.\d+|[01]\.?\d*", response)
        if match:
            value = float(match.group())
            if 0.0 <= value <= 1.0:
                return value

        raise ValueError(f"Could not parse valid credence from response: {response}")
