"""Utilities for saving and loading credence estimates for comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from belief_credence.core import Claim, CredenceEstimate, CredenceMethod
from belief_credence.datasets import BeliefType


def estimate_to_dict(estimate: CredenceEstimate) -> dict[str, Any]:
    """Convert CredenceEstimate to JSON-serializable dict.

    Args:
        estimate: CredenceEstimate to convert

    Returns:
        Dictionary representation
    """
    # Convert metadata to JSON-serializable format
    claim_metadata = {}
    if estimate.claim.metadata:
        for key, value in estimate.claim.metadata.items():
            if isinstance(value, BeliefType):
                claim_metadata[key] = value.value  # Convert enum to string
            else:
                claim_metadata[key] = value

    return {
        "p_true": estimate.p_true,
        "method": estimate.method,
        "claim": {
            "statement": estimate.claim.statement,
            "negation": estimate.claim.negation,
            "metadata": claim_metadata,
        },
        "raw_output": estimate.raw_output,
        "metadata": estimate.metadata,
    }


def dict_to_estimate(data: dict[str, Any]) -> CredenceEstimate:
    """Convert dict back to CredenceEstimate.

    Args:
        data: Dictionary representation

    Returns:
        CredenceEstimate object
    """
    # Convert metadata back from JSON format
    claim_metadata = {}
    raw_metadata = data["claim"].get("metadata", {})
    for key, value in raw_metadata.items():
        if key == "belief_type" and isinstance(value, str):
            # Convert string back to BeliefType enum
            claim_metadata[key] = BeliefType(value)
        else:
            claim_metadata[key] = value

    claim = Claim(
        statement=data["claim"]["statement"],
        negation=data["claim"].get("negation"),
        metadata=claim_metadata,
    )

    return CredenceEstimate(
        p_true=data["p_true"],
        method=data["method"],
        claim=claim,
        raw_output=data.get("raw_output"),
        metadata=data.get("metadata", {}),
    )


def save_estimates(
    estimates: list[CredenceEstimate], output_path: str | Path
) -> None:
    """Save credence estimates to JSON file.

    Args:
        estimates: List of estimates to save
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "estimates": [estimate_to_dict(est) for est in estimates],
        "count": len(estimates),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_estimates(input_path: str | Path) -> list[CredenceEstimate]:
    """Load credence estimates from JSON file.

    Args:
        input_path: Path to input file

    Returns:
        List of CredenceEstimate objects
    """
    with open(input_path) as f:
        data = json.load(f)

    return [dict_to_estimate(est_data) for est_data in data["estimates"]]


def batch_evaluate(
    method: CredenceMethod,
    claims: list[Claim],
    output_path: str | Path | None = None,
) -> list[CredenceEstimate]:
    """Evaluate a method on multiple claims and optionally save results.

    Args:
        method: CredenceMethod to use
        claims: List of claims to evaluate
        output_path: Optional path to save results

    Returns:
        List of CredenceEstimate objects
    """
    estimates = []

    for claim in claims:
        estimate = method.estimate(claim)
        estimates.append(estimate)

    if output_path is not None:
        save_estimates(estimates, output_path)

    return estimates


def batch_evaluate_methods(
    methods: list[CredenceMethod],
    claims: list[Claim],
    output_dir: str | Path,
) -> dict[str, list[CredenceEstimate]]:
    """Evaluate multiple methods on multiple claims and save all results.

    Creates one file per method in output_dir.

    Args:
        methods: List of CredenceMethod objects
        claims: List of claims to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary mapping method name to list of estimates
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for method in methods:
        method_name = method.name
        estimates = batch_evaluate(
            method,
            claims,
            output_path=output_dir / f"{method_name}.json",
        )
        results[method_name] = estimates

    # Also save a summary
    summary = {
        "methods": [method.name for method in methods],
        "num_claims": len(claims),
        "claims": [
            {
                "statement": claim.statement,
                "negation": claim.negation,
            }
            for claim in claims
        ],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def compare_saved_estimates(
    estimate_files: list[str | Path],
) -> dict[str, Any]:
    """Compare estimates from multiple saved files.

    Args:
        estimate_files: List of paths to estimate JSON files

    Returns:
        Dictionary with comparison statistics
    """
    all_estimates = {}

    for file_path in estimate_files:
        estimates = load_estimates(file_path)
        if estimates:
            method_name = estimates[0].method
            all_estimates[method_name] = estimates

    if not all_estimates:
        return {}

    num_claims = len(next(iter(all_estimates.values())))

    comparisons = []
    for i in range(num_claims):
        claim_comparison = {
            "statement": None,
            "estimates": {},
        }

        for method_name, estimates in all_estimates.items():
            if i < len(estimates):
                if claim_comparison["statement"] is None:
                    claim_comparison["statement"] = estimates[i].claim.statement

                claim_comparison["estimates"][method_name] = estimates[i].p_true

        comparisons.append(claim_comparison)

    return {
        "methods": list(all_estimates.keys()),
        "num_claims": num_claims,
        "comparisons": comparisons,
    }
