"""Utilities for creating train/validation/test splits from claim datasets.

This module provides functions to split ClaimSets across different belief types
into stratified train/validation/test sets for proper evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from belief_credence.core import Claim
from belief_credence.datasets import BeliefType, ClaimSet, ALL_DATASETS


@dataclass
class DataSplit:
    """Train/validation/test split of claims.

    Attributes:
        train_claims: Claims for training (with negations)
        val_claims: Claims for validation (with negations)
        test_claims: Claims for testing (with negations)
    """

    train_claims: list[Claim]
    val_claims: list[Claim]
    test_claims: list[Claim]

    def __post_init__(self):
        """Print split statistics."""
        print(f"Data split created:")
        print(f"  Train: {len(self.train_claims)} claims")
        print(f"  Val:   {len(self.val_claims)} claims")
        print(f"  Test:  {len(self.test_claims)} claims")


def create_mixed_split(
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int | None = 42,
    include_types: list[BeliefType] | None = None,
) -> DataSplit:
    """Create stratified train/val/test split mixing all belief types.

    This ensures each split contains a representative mix of different belief types
    (well-established facts, contested facts, predictions, normative judgments, etc.)
    rather than segregating by type.

    Args:
        train_ratio: Fraction of data for training (default 0.6 = 60%)
        val_ratio: Fraction of data for validation (default 0.2 = 20%)
        test_ratio: Fraction of data for testing (default 0.2 = 20%)
        seed: Random seed for reproducibility (None = no seeding)
        include_types: List of belief types to include (None = all types)

    Returns:
        DataSplit with stratified train/val/test claims

    Example:
        >>> split = create_mixed_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        >>> # Train CCS on mixed training data
        >>> ccs.train_probe(split.train_claims)
        >>> # Validate hyperparameters on validation set
        >>> val_performance = evaluate_on_dataset(ccs, split.val_claims)
        >>> # Final test on test set
        >>> test_performance = evaluate_on_dataset(ccs, split.test_claims)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if seed is not None:
        random.seed(seed)

    # Collect all claim sets
    if include_types is None:
        include_types = list(BeliefType)

    all_claim_sets: list[ClaimSet] = []
    for belief_type in include_types:
        all_claim_sets.extend(ALL_DATASETS[belief_type])

    # Shuffle claim sets
    shuffled_sets = all_claim_sets.copy()
    random.shuffle(shuffled_sets)

    # Calculate split indices
    n_total = len(shuffled_sets)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split claim sets
    train_sets = shuffled_sets[:n_train]
    val_sets = shuffled_sets[n_train:n_train + n_val]
    test_sets = shuffled_sets[n_train + n_val:]

    # Convert to claims (using first phrasing with canonical negation)
    def sets_to_claims(claim_sets: list[ClaimSet]) -> list[Claim]:
        claims = []
        for cs in claim_sets:
            # Use first phrasing as canonical
            claims.append(Claim(
                statement=cs.positive_phrasings[0],
                negation=cs.negative_phrasings[0],
                metadata={"belief_type": cs.belief_type, "description": cs.description},
            ))
        return claims

    train_claims = sets_to_claims(train_sets)
    val_claims = sets_to_claims(val_sets)
    test_claims = sets_to_claims(test_sets)

    return DataSplit(
        train_claims=train_claims,
        val_claims=val_claims,
        test_claims=test_claims,
    )


def create_type_specific_split(
    belief_type: BeliefType,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int | None = 42,
) -> DataSplit:
    """Create train/val/test split from a single belief type.

    Use this when you want to evaluate on a specific type of belief
    (e.g., only contested facts, only metaphysical beliefs).

    Args:
        belief_type: Which type of beliefs to split
        train_ratio: Fraction of data for training (default 0.6 = 60%)
        val_ratio: Fraction of data for validation (default 0.2 = 20%)
        test_ratio: Fraction of data for testing (default 0.2 = 20%)
        seed: Random seed for reproducibility (None = no seeding)

    Returns:
        DataSplit with train/val/test claims of the specified type

    Example:
        >>> # Train and evaluate only on contested facts
        >>> split = create_type_specific_split(BeliefType.CONTESTED_FACT)
        >>> ccs.train_probe(split.train_claims)
        >>> test_performance = evaluate_on_dataset(ccs, split.test_claims)
    """
    return create_mixed_split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        include_types=[belief_type],
    )


def get_split_statistics(split: DataSplit) -> dict[str, dict[str, int]]:
    """Get statistics about belief type distribution in each split.

    Args:
        split: DataSplit to analyze

    Returns:
        Dictionary mapping split names to belief type counts

    Example:
        >>> split = create_mixed_split()
        >>> stats = get_split_statistics(split)
        >>> print(stats['train'][BeliefType.CONTESTED_FACT])
        9  # number of contested facts in training set
    """
    def count_types(claims: list[Claim]) -> dict[BeliefType, int]:
        """Count claims by belief type."""
        counts: dict[BeliefType, int] = {bt: 0 for bt in BeliefType}

        # Find which type each claim belongs to
        for claim in claims:
            for belief_type, claim_sets in ALL_DATASETS.items():
                for cs in claim_sets:
                    if claim.statement in cs.positive_phrasings:
                        counts[belief_type] += 1
                        break

        return counts

    return {
        "train": count_types(split.train_claims),
        "val": count_types(split.val_claims),
        "test": count_types(split.test_claims),
    }
