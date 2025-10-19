"""Example script demonstrating visualization of method comparisons.

This script shows how to:
1. Run multiple methods on a dataset
2. Create various comparison plots
3. Generate a complete comparison report
"""

from belief_credence import (
    DirectPrompting,
    LogitGap,
    CCS,
    get_dataset,
    BeliefType,
    create_comparison_report,
    plot_method_comparison,
    plot_claim_by_claim_comparison,
)
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("=" * 80)
    print("METHOD COMPARISON VISUALIZATION")
    print("=" * 80)

    print("\nLoading model...")
    model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)

    # Step 1: Train CCS
    print("\nTraining CCS probe...")
    claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
    training_claims = [cs.to_claims()[0] for cs in claim_sets]

    ccs = CCS(model=model, direction_method="logit_gap")
    ccs.train_probe(training_claims)

    # Step 2: Get evaluation claims (can be different from training)
    print("\nPreparing evaluation claims...")
    eval_claim_sets = get_dataset(BeliefType.CONTESTED_FACT)
    eval_claims = [cs.to_claims()[0] for cs in eval_claim_sets]

    print(f"Evaluating on {len(eval_claims)} contested fact claims")

    # Step 3: Initialize all methods
    methods = [
        DirectPrompting(model=model),
        LogitGap(model=model),
        ccs,
    ]

    # Step 4: Evaluate all methods
    print("\n" + "=" * 80)
    print("EVALUATING METHODS")
    print("=" * 80)

    estimates_by_method = {}

    for method in methods:
        print(f"\nEvaluating {method.name}...")
        estimates = []
        for claim in eval_claims:
            estimate = method.estimate(claim)
            estimates.append(estimate)
        estimates_by_method[method.name] = estimates
        print(f"  Complete: {len(estimates)} estimates")

    # Step 5: Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Individual plots
    print("\n1. Comprehensive comparison plot...")
    plot_method_comparison(
        estimates_by_method,
        title="Method Comparison: Contested Facts",
    )

    print("\n2. Claim-by-claim comparison...")
    plot_claim_by_claim_comparison(
        estimates_by_method,
        max_claims=15,
        title="P(True) by Claim: Contested Facts",
    )

    # Complete report
    print("\n3. Generating complete comparison report...")
    create_comparison_report(
        estimates_by_method,
        output_dir="outputs/visualization_report",
        report_name="contested_facts_comparison",
    )

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for method_name, estimates in estimates_by_method.items():
        p_values = [est.p_true for est in estimates]
        mean_p = sum(p_values) / len(p_values)
        std_p = (sum((p - mean_p) ** 2 for p in p_values) / len(p_values)) ** 0.5

        print(f"\n{method_name}:")
        print(f"  Mean P(True): {mean_p:.3f}")
        print(f"  Std Dev:      {std_p:.3f}")
        print(f"  Range:        [{min(p_values):.3f}, {max(p_values):.3f}]")


if __name__ == "__main__":
    main()
