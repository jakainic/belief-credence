"""Example script demonstrating epistemological property evaluation.

This script shows how to evaluate whether credence estimates follow
rational Bayesian principles:
1. Logical consistency (P(T) + P(F) ≈ 1)
2. Coherence across paraphrases
3. Bayesian conditioning
4. Action-belief alignment
"""

from belief_credence import Claim, DirectPrompting, LogitGap
from belief_credence.epistemology import evaluate_epistemology
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("Loading model...")
    model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)

    print("\nInitializing methods...")
    methods = [
        DirectPrompting(model=model),
        LogitGap(model=model),
    ]

    print("\n" + "=" * 80)
    print("EPISTEMOLOGICAL PROPERTY EVALUATION")
    print("=" * 80)

    consistency_claims = [
        Claim(
            statement="The Eiffel Tower is in Paris.",
            negation="The Eiffel Tower is not in Paris.",
        ),
        Claim(
            statement="Water freezes at 0°C at standard pressure.",
            negation="Water does not freeze at 0°C at standard pressure.",
        ),
        Claim(
            statement="The Earth is flat.", negation="The Earth is not flat."
        ),
    ]

    coherence_tests = [
        (
            Claim(statement="Paris is the capital of France."),
            [
                "France's capital is Paris.",
                "Paris serves as the capital city of France.",
                "The capital of France is Paris.",
            ],
        ),
        (
            Claim(statement="Python is a programming language."),
            [
                "Python is used for programming.",
                "The Python language is for programming.",
                "Programming can be done in Python.",
            ],
        ),
    ]

    bayesian_tests = [
        ("the ground is wet", "it is raining"),
        ("you will get wet", "you go outside and it is raining"),
        ("the match will light", "the match is dry"),
    ]

    action_tests = [
        (
            Claim(statement="The Earth orbits the Sun."),
            "If asked to bet $100 on the truth of this claim, you would accept the bet: {claim.statement}",
        ),
        (
            Claim(statement="Humans can breathe underwater without equipment."),
            "You would recommend someone act as if this is true: {claim.statement}",
        ),
    ]

    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {method.name}")
        print(f"{'=' * 80}")

        report = evaluate_epistemology(
            method,
            consistency_claims=consistency_claims,
            coherence_tests=coherence_tests,
            bayesian_tests=bayesian_tests,
            action_tests=action_tests,
        )

        print("\n1. CONSISTENCY CHECKS (P(T) + P(F) ≈ 1)")
        print("-" * 80)
        for check in report.consistency_checks:
            print(f"\nClaim: {check.claim.statement}")
            print(f"  P(statement) = {check.p_statement:.3f}")
            print(f"  P(negation)  = {check.p_negation:.3f}")
            print(f"  Sum = {check.sum_probability:.3f}")
            print(f"  Deviation from 1.0: {check.deviation_from_one:.3f}")
            print(f"  Consistent? {check.is_consistent}")
            print(f"  Score: {check.consistency_score:.3f}")

        print(f"\nOverall Consistency Score: {report.overall_consistency_score:.3f}")

        print("\n2. COHERENCE CHECKS (Paraphrase Robustness)")
        print("-" * 80)
        for check in report.coherence_checks:
            print(f"\nOriginal: {check.original_claim.statement}")
            print(f"  Original P(True) = {check.original_estimate:.3f}")
            print(f"  Paraphrase estimates: {[f'{p:.3f}' for p in check.paraphrase_estimates]}")
            print(f"  Mean: {check.mean_estimate:.3f}")
            print(f"  Std Dev: {check.std_deviation:.3f}")
            print(f"  Max Deviation: {check.max_deviation:.3f}")
            print(f"  Coherent? {check.is_coherent}")
            print(f"  Score: {check.coherence_score:.3f}")

        print(f"\nOverall Coherence Score: {report.overall_coherence_score:.3f}")

        print("\n3. BAYESIAN CONDITIONING CHECKS (P(A|B) ≈ P(A∧B)/P(B))")
        print("-" * 80)
        for check in report.bayesian_checks:
            print(f"\nProposition: {check.proposition}")
            print(f"Evidence: {check.evidence}")
            print(f"  P(A|B) [measured]  = {check.p_a_given_b:.3f}")
            print(f"  P(A∧B)             = {check.p_a_and_b:.3f}")
            print(f"  P(B)               = {check.p_b:.3f}")
            print(f"  P(A|B) [expected]  = {check.expected_p_a_given_b:.3f}")
            print(f"  Deviation: {check.deviation:.3f}")
            print(f"  Bayesian? {check.is_bayesian}")
            print(f"  Score: {check.bayesian_score:.3f}")

        print(f"\nOverall Bayesian Score: {report.overall_bayesian_score:.3f}")

        print("\n4. ACTION-BELIEF CORRELATION")
        print("-" * 80)
        for check in report.action_checks:
            print(f"\nClaim: {check.claim.statement}")
            print(f"  Internal credence:     {check.internal_credence:.3f}")
            print(f"  Action probability:    {check.action_probability:.3f}")
            print(f"  Correlation score:     {check.correlation_score:.3f}")
            print(f"  Aligned? {check.is_aligned}")

        print(f"\nOverall Action Score: {report.overall_action_score:.3f}")

        print("\n" + "=" * 80)
        print(f"OVERALL EPISTEMOLOGY SCORE: {report.overall_epistemology_score:.3f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
