"""Example script for batch evaluating methods and saving outputs for later comparison.

This script shows how to:
1. Evaluate multiple methods on a dataset
2. Save outputs to JSON files
3. Load and compare saved outputs
"""

from belief_credence import (
    DirectPrompting,
    LogitGap,
    CCS,
    batch_evaluate_methods,
    compare_saved_estimates,
    get_dataset,
    BeliefType,
)
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("=" * 80)
    print("BATCH EVALUATION WITH OUTPUT SAVING")
    print("=" * 80)

    print("\nLoading model...")
    model = ModelWrapper("meta-llama/Llama-2-8b-hf", load_in_8bit=True)

    print("\nInitializing methods...")
    methods = [
        DirectPrompting(model=model),
        LogitGap(model=model),
        CCS(model=model, direction_method="logit_gap"),
    ]

    print("\nLoading dataset (well-established facts)...")
    claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
    claims = [cs.to_claims()[0] for cs in claim_sets]

    print(f"Evaluating on {len(claims)} claims:")
    for claim in claims:
        print(f"  - {claim.statement}")

    print("\n" + "=" * 80)
    print("RUNNING EVALUATIONS")
    print("=" * 80)

    output_dir = "outputs/well_established_facts"
    results = batch_evaluate_methods(methods, claims, output_dir=output_dir)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Files created:")
    for method in methods:
        print(f"  - {method.name}.json")
    print(f"  - summary.json")

    print("\n" + "=" * 80)
    print("LOADING AND COMPARING SAVED OUTPUTS")
    print("=" * 80)

    estimate_files = [f"{output_dir}/{method.name}.json" for method in methods]
    comparison = compare_saved_estimates(estimate_files)

    print(f"\nMethods compared: {', '.join(comparison['methods'])}")
    print(f"Number of claims: {comparison['num_claims']}")

    print("\n" + "=" * 80)
    print("PER-CLAIM COMPARISON")
    print("=" * 80)

    for i, comp in enumerate(comparison["comparisons"], 1):
        print(f"\n{i}. {comp['statement'][:70]}...")
        for method_name, p_true in comp["estimates"].items():
            print(f"   {method_name:40s}: P(True) = {p_true:.3f}")

        # Calculate agreement stats
        p_values = list(comp["estimates"].values())
        mean_p = sum(p_values) / len(p_values)
        std_p = (sum((p - mean_p) ** 2 for p in p_values) / len(p_values)) ** 0.5

        print(f"   {'Agreement':40s}: mean={mean_p:.3f}, std={std_p:.3f}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for method_name in comparison["methods"]:
        p_values = [
            comp["estimates"][method_name]
            for comp in comparison["comparisons"]
            if method_name in comp["estimates"]
        ]

        mean = sum(p_values) / len(p_values)
        std = (sum((p - mean) ** 2 for p in p_values) / len(p_values)) ** 0.5
        min_p = min(p_values)
        max_p = max(p_values)

        print(f"\n{method_name}:")
        print(f"  Mean P(True):   {mean:.3f}")
        print(f"  Std Dev:        {std:.3f}")
        print(f"  Range:          [{min_p:.3f}, {max_p:.3f}]")


if __name__ == "__main__":
    main()
