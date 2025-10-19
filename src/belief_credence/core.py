"""Core utilities for beliefs, credences, and Bayesian updates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Claim:
    """Represents a claim to be evaluated for credence.

    Attributes:
        statement: The original claim statement
        negation: The negation of the claim (auto-generated or manual)
        metadata: Optional dict for method-specific data
    """
    statement: str
    negation: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CredenceEstimate:
    """Result of a credence measurement.

    Attributes:
        p_true: Probability estimate that the claim is true (0 to 1)
        method: Name of the method used
        claim: The original claim
        raw_output: Method-specific raw output for debugging
        metadata: Additional method-specific information
    """
    p_true: float
    method: str
    claim: Claim
    raw_output: Any | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_true <= 1.0:
            raise ValueError(f"p_true must be in [0, 1], got {self.p_true}")
        if self.metadata is None:
            self.metadata = {}


class CredenceMethod(ABC):
    """Base class for all credence measurement methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this credence measurement method."""
        pass

    @abstractmethod
    def estimate(self, claim: Claim) -> CredenceEstimate:
        """Estimate P(True) for a given claim.

        Args:
            claim: The claim to evaluate

        Returns:
            CredenceEstimate with p_true and method-specific metadata
        """
        pass
