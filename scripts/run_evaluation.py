"""Test script for RunPod evaluation with timing for each method.

This script:
1. Loads the model once
2. Trains CCS probe (with timing)
3. Evaluates each method separately (with timing)
4. Saves outputs for each method
5. Prints summary statistics

Run on RunPod with:
    python scripts/run_evaluation.py
"""

import time
from pathlib import Path

from belief_credence import (
    DirectPrompting,
    LogitGap,
    CCS,
    get_dataset,
    BeliefType,
    save_estimates,
)
from belief_credence.model_utils import ModelWrapper


def format_time(seconds: float) -> str:
    """Format seconds into readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def main() -> None:
    print("=" * 80)
    print("RUNPOD EVALUATION TEST")
    print("=" * 80)

    # Configuration
    output_dir = Path("outputs/runpod_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load model
    print("\n[1/6] Loading model...")
    start = time.time()
    model = ModelWrapper("meta-llama/Llama-2-8b-hf", load_in_8bit=True)
    load_time = time.time() - start
    print(f"✓ Model loaded in {format_time(load_time)}")

    # Step 2: Get training data for CCS
    print("\n[2/6] Loading training data...")
    start = time.time()
    training_claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
    training_claims = [cs.to_claims()[0] for cs in training_claim_sets]
    data_time = time.time() - start
    print(f"✓ Loaded {len(training_claims)} training claims in {format_time(data_time)}")

    # Step 3: Train CCS probe
    print("\n[3/6] Training CCS probe...")
    start = time.time()
    ccs = CCS(model=model, direction_method="logit_gap", layer=-1)
    ccs.train_probe(training_claims, epochs=100, lr=1e-3)
    train_time = time.time() - start
    print(f"✓ CCS probe trained in {format_time(train_time)}")

    # Step 4: Get evaluation data
    print("\n[4/6] Loading evaluation data...")
    start = time.time()
    eval_claim_sets = get_dataset(BeliefType.CONTESTED_FACT)
    eval_claims = [cs.to_claims()[0] for cs in eval_claim_sets]
    eval_data_time = time.time() - start
    print(f"✓ Loaded {len(eval_claims)} evaluation claims in {format_time(eval_data_time)}")

    # Step 5: Evaluate each method separately
    print("\n[5/6] Evaluating methods...")
    print("-" * 80)

    methods = [
        ("DirectPrompting", DirectPrompting(model=model)),
        ("LogitGap", LogitGap(model=model)),
        ("CCS", ccs),
    ]

    results = {}
    timings = {}

    for method_name, method in methods:
        print(f"\nEvaluating {method_name}...")
        start = time.time()

        estimates = []
        for i, claim in enumerate(eval_claims, 1):
            estimate = method.estimate(claim)
            estimates.append(estimate)

            # Progress indicator
            if i % 5 == 0 or i == len(eval_claims):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(eval_claims) - i) / rate if rate > 0 else 0
                print(
                    f"  Progress: {i}/{len(eval_claims)} claims "
                    f"({elapsed:.1f}s elapsed, {rate:.2f} claims/s, "
                    f"ETA: {format_time(eta)})"
                )

        eval_time = time.time() - start
        timings[method_name] = eval_time
        results[method_name] = estimates

        # Save estimates
        output_file = output_dir / f"{method.name}.json"
        save_estimates(estimates, output_file)
        print(f"✓ {method_name} complete in {format_time(eval_time)}")
        print(f"  Saved to: {output_file}")

        # Quick stats
        p_values = [est.p_true for est in estimates]
        mean_p = sum(p_values) / len(p_values)
        std_p = (sum((p - mean_p) ** 2 for p in p_values) / len(p_values)) ** 0.5
        print(f"  Mean P(True): {mean_p:.3f} ± {std_p:.3f}")

    # Step 6: Summary
    print("\n[6/6] Summary")
    print("=" * 80)
    print(f"\nModel: meta-llama/Llama-2-8b-hf (8-bit)")
    print(f"Training claims: {len(training_claims)} (well-established facts)")
    print(f"Evaluation claims: {len(eval_claims)} (contested facts)")
    print(f"\nTiming Breakdown:")
    print(f"  Model loading:      {format_time(load_time)}")
    print(f"  CCS training:       {format_time(train_time)}")
    print()
    for method_name, eval_time in timings.items():
        per_claim = eval_time / len(eval_claims)
        print(f"  {method_name:20s} {format_time(eval_time):>10s} ({per_claim:.2f}s/claim)")

    total_time = load_time + train_time + sum(timings.values())
    print(f"\n  Total runtime:      {format_time(total_time)}")

    print(f"\n✓ All outputs saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Run: python scripts/generate_plots.py")
    print("  2. Download outputs/ folder to view results")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
