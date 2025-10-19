"""Example script using curated datasets for comprehensive evaluation.

Demonstrates:
1. Loading datasets by belief type
2. Evaluating consistency and coherence across phrasings
3. Comparing methods across different belief categories
"""

from belief_credence import DirectPrompting, LogitGap, evaluate_epistemology
from belief_credence.datasets import (
    ALL_DATASETS,
    BeliefType,
    get_all_claim_sets,
    get_dataset,
)
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION USING CURATED DATASETS")
    print("=" * 80)

    print("\nDataset Summary:")
    print("-" * 80)
    for belief_type, dataset in ALL_DATASETS.items():
        print(f"{belief_type.value}: {len(dataset)} claim sets")
        total_phrasings = sum(
            len(cs.positive_phrasings) + len(cs.negative_phrasings) for cs in dataset
        )
        print(f"  Total phrasings: {total_phrasings}")

    print("\n" + "=" * 80)
    print("Loading model...")
    model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)

    methods = [
        DirectPrompting(model=model),
        LogitGap(model=model),
    ]

    for belief_type in BeliefType:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {belief_type.value.upper()}")
        print(f"{'=' * 80}")

        claim_sets = get_dataset(belief_type)

        for method in methods:
            print(f"\nMethod: {method.name}")
            print("-" * 80)

            consistency_claims = []
            coherence_tests = []

            for claim_set in claim_sets:
                print(f"\n{claim_set.description}:")

                claims = claim_set.to_claims()
                if claims:
                    consistency_claims.append(claims[0])

                    coherence_tests.append(
                        (claims[0], claim_set.positive_phrasings[1:])
                    )

            report = evaluate_epistemology(
                method,
                consistency_claims=consistency_claims,
                coherence_tests=coherence_tests,
            )

            print(f"\nConsistency Checks ({len(report.consistency_checks)} claims):")
            for check in report.consistency_checks:
                status = "✓" if check.is_consistent else "✗"
                print(
                    f"  {status} {check.claim.statement[:60]}... "
                    f"[P(+)={check.p_statement:.2f}, P(-)={check.p_negation:.2f}, "
                    f"sum={check.sum_probability:.2f}]"
                )

            print(f"\nCoherence Checks ({len(report.coherence_checks)} claim sets):")
            for check in report.coherence_checks:
                status = "✓" if check.is_coherent else "✗"
                print(
                    f"  {status} {check.original_claim.statement[:60]}... "
                    f"[std={check.std_deviation:.3f}, max_dev={check.max_deviation:.3f}]"
                )

            print(f"\n{belief_type.value} Results for {method.name}:")
            print(f"  Overall Consistency Score: {report.overall_consistency_score:.3f}")
            print(f"  Overall Coherence Score:   {report.overall_coherence_score:.3f}")
            print(
                f"  Overall Epistemology:      {report.overall_epistemology_score:.3f}"
            )

    print("\n" + "=" * 80)
    print("CROSS-CATEGORY COMPARISON")
    print("=" * 80)

    for method in methods:
        print(f"\n{method.name}:")
        print("-" * 80)

        for belief_type in BeliefType:
            claim_sets = get_dataset(belief_type)
            consistency_claims = [cs.to_claims()[0] for cs in claim_sets]

            report = evaluate_epistemology(
                method, consistency_claims=consistency_claims
            )

            print(
                f"  {belief_type.value:30s}: "
                f"consistency={report.overall_consistency_score:.3f}"
            )


def demo_individual_dataset() -> None:
    """Demonstrate working with individual dataset types."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL DATASET DEMO")
    print("=" * 80)

    print("\nWell-Established Facts:")
    for claim_set in get_dataset(BeliefType.WELL_ESTABLISHED_FACT):
        print(f"\n{claim_set.description}:")
        print(f"  Positive phrasings: {len(claim_set.positive_phrasings)}")
        print(f"  Example: {claim_set.positive_phrasings[0]}")
        print(f"  Negation: {claim_set.negative_phrasings[0]}")

    print("\nMetaphysical Beliefs:")
    for claim_set in get_dataset(BeliefType.METAPHYSICAL_BELIEF):
        print(f"\n{claim_set.description}:")
        print(f"  Statement: {claim_set.positive_phrasings[0]}")
        print(f"  Negation: {claim_set.negative_phrasings[0]}")

    all_claim_sets = get_all_claim_sets()
    print(f"\nTotal claim sets across all categories: {len(all_claim_sets)}")


if __name__ == "__main__":
    demo_individual_dataset()
    main()
