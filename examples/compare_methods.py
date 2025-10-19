"""Example script demonstrating credence estimation across all methods.

This script shows how to:
1. Create claims with negations
2. Run all three credence methods
3. Compare results across methods
4. Use uncertainty estimation for validation
"""

from belief_credence import (
    CCS,
    Claim,
    DirectPrompting,
    HallucinationProbe,
    LogitGap,
    check_credence_uncertainty_alignment,
    compare_methods,
)
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("Loading model...")
    model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)

    claims = [
        Claim(
            statement="The Eiffel Tower is located in Paris, France.",
            negation="The Eiffel Tower is not located in Paris, France.",
        ),
        Claim(
            statement="The Earth is flat.",
            negation="The Earth is not flat.",
        ),
        Claim(
            statement="Python was created by Guido van Rossum.",
            negation="Python was not created by Guido van Rossum.",
        ),
    ]

    print("\nInitializing credence methods...")
    methods = [
        DirectPrompting(model=model),
        LogitGap(model=model),
        CCS(model=model, direction_method="logit_gap"),
    ]

    print("Initializing uncertainty probe...")
    uncertainty_probe = HallucinationProbe(model=model)

    for claim in claims:
        print(f"\n{'=' * 80}")
        print(f"Claim: {claim.statement}")
        print(f"{'=' * 80}")

        # Compare credence methods
        comparison = compare_methods(claim, methods)

        for method_name, estimate in comparison.estimates.items():
            print(f"\n{method_name}:")
            print(f"  P(True) = {estimate.p_true:.3f}")
            if estimate.metadata:
                print(f"  Metadata: {estimate.metadata}")

        print(f"\nSummary Statistics:")
        print(f"  Mean P(True): {comparison.mean_p_true():.3f}")
        print(f"  Std Dev: {comparison.std_p_true():.3f}")
        p_min, p_max = comparison.range_p_true()
        print(f"  Range: [{p_min:.3f}, {p_max:.3f}]")

        # Check uncertainty alignment
        print(f"\nUncertainty Analysis:")
        uncertainty = uncertainty_probe.estimate_uncertainty(claim)
        print(f"  Uncertainty: {uncertainty.uncertainty_score:.3f}")
        print(f"  Confidence: {uncertainty.confidence_score:.3f}")

        mean_p_true = comparison.mean_p_true()
        is_aligned = check_credence_uncertainty_alignment(mean_p_true, uncertainty)
        print(
            f"  Aligned with mean credence? {is_aligned} "
            f"(mean P(True)={mean_p_true:.3f})"
        )


if __name__ == "__main__":
    main()
