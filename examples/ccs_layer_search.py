"""Example script demonstrating CCS layer search for hyperparameter tuning.

This script shows how to:
1. Search across multiple layers to find optimal CCS performance
2. Use curated datasets for training and validation
3. Compare consistency scores across layers
"""

from belief_credence import CCS, search_best_layer, get_dataset, BeliefType
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("=" * 80)
    print("CCS LAYER SEARCH DEMONSTRATION")
    print("=" * 80)

    print("\nLoading model...")
    model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)

    print("\nLoading training data from well-established facts...")
    claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
    training_claims = [cs.to_claims()[0] for cs in claim_sets]

    print(f"Training claims: {len(training_claims)}")
    for claim in training_claims:
        print(f"  - {claim.statement}")

    print("\n" + "=" * 80)
    print("SEARCHING LAYERS -1 through -5")
    print("=" * 80)

    layers_to_search = [-1, -2, -3, -4, -5]

    print("\nSearching for best layer...")
    result = search_best_layer(
        model=model,
        training_claims=training_claims,
        layers=layers_to_search,
        epochs=100,
        lr=1e-3,
        direction_method="logit_gap",
    )

    print(f"\n{'=' * 80}")
    print("LAYER SEARCH RESULTS")
    print(f"{'=' * 80}")
    print(f"Best layer: {result.layer}")
    print(f"Consistency score: {result.consistency_score:.4f}")
    print(f"  (Lower is better - measures deviation from P(+) + P(-) = 1)")

    print(f"\n{'=' * 80}")
    print("USING BEST LAYER FOR EVALUATION")
    print(f"{'=' * 80}")

    # Create CCS method with best layer and trained probe
    ccs = CCS(model=model, layer=result.layer)
    ccs._probe = result.probe

    # Test on contested facts (different category)
    print("\nTesting on contested facts...")
    test_claim_sets = get_dataset(BeliefType.CONTESTED_FACT)
    test_claims = [cs.to_claims()[0] for cs in test_claim_sets]

    print("\nTest results:")
    for claim in test_claims:
        estimate = ccs.estimate(claim)
        print(f"\nClaim: {claim.statement}")
        print(f"  P(True) = {estimate.p_true:.3f}")
        print(f"  P(+) + P(-) = {1.0 - estimate.metadata['consistency_score']:.3f}")
        print(f"  Consistency score = {estimate.metadata['consistency_score']:.4f}")

    print(f"\n{'=' * 80}")
    print("COMPARISON: Manual layer selection vs. layer search")
    print(f"{'=' * 80}")

    # Compare with default layer (-1)
    default_ccs = CCS(model=model, layer=-1, direction_method="logit_gap")
    default_ccs.train_probe(training_claims, epochs=100)

    print("\nUsing default layer (-1):")
    default_score = default_ccs.evaluate_layer_performance(training_claims)
    print(f"  Consistency score: {default_score:.4f}")

    print(f"\nUsing searched best layer ({result.layer}):")
    print(f"  Consistency score: {result.consistency_score:.4f}")

    improvement = ((default_score - result.consistency_score) / default_score) * 100
    print(f"\nImprovement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
