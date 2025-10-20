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
    create_mixed_split,
    get_split_statistics,
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
    model = ModelWrapper("meta-llama/Llama-2-7b-chat-hf", load_in_8bit=True)
    load_time = time.time() - start
    print(f"✓ Model loaded in {format_time(load_time)}")

    # Step 2: Create train/val/test split (mixed belief types)
    print("\n[2/6] Creating data splits...")
    start = time.time()
    split = create_mixed_split(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )
    data_time = time.time() - start

    # Print split statistics
    stats = get_split_statistics(split)
    print(f"\nSplit statistics by belief type:")
    for split_name in ["train", "val", "test"]:
        print(f"\n{split_name.upper()}:")
        for belief_type, count in stats[split_name].items():
            if count > 0:
                print(f"  {belief_type.value}: {count}")

    print(f"\n✓ Data splits created in {format_time(data_time)}")

    # Step 3: Train CCS probe on training set
    print("\n[3/6] Training CCS probe on training set...")
    start = time.time()
    ccs = CCS(model=model, direction_method="logit_gap", layer=-1)
    ccs.train_probe(split.train_claims, epochs=100, lr=1e-3)
    train_time = time.time() - start
    print(f"✓ CCS probe trained in {format_time(train_time)}")

    # Step 4: Validate on validation set (optional hyperparameter tuning)
    print("\n[4/6] Validating on validation set...")
    print(f"Using validation set with {len(split.val_claims)} claims for sanity check")

    # Step 5: Evaluate ALL methods on the SAME test set
    print("\n[5/6] Evaluating all methods on the same test set...")
    print("-" * 80)
    print(f"Test set: {len(split.test_claims)} claims")
    print("Note: All methods evaluated on identical test claims for fair comparison")
    print()

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
        for i, claim in enumerate(split.test_claims, 1):
            estimate = method.estimate(claim)
            estimates.append(estimate)

            # Progress indicator
            if i % 5 == 0 or i == len(split.test_claims):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(split.test_claims) - i) / rate if rate > 0 else 0
                print(
                    f"  Progress: {i}/{len(split.test_claims)} claims "
                    f"({elapsed:.1f}s elapsed, {rate:.2f} claims/s, "
                    f"ETA: {format_time(eta)})"
                )

        eval_time = time.time() - start
        timings[method_name] = eval_time
        results[method_name] = estimates

        # Save estimates with metadata about the split
        output_file = output_dir / f"{method.name}.json"
        save_estimates(estimates, output_file)
        print(f"✓ {method_name} complete in {format_time(eval_time)}")
        print(f"  Saved to: {output_file}")
        print(f"  Evaluated on: {len(estimates)} test claims (seed=42)")

        # Quick stats
        p_values = [est.p_true for est in estimates]
        mean_p = sum(p_values) / len(p_values)
        std_p = (sum((p - mean_p) ** 2 for p in p_values) / len(p_values)) ** 0.5
        print(f"  Mean P(True): {mean_p:.3f} ± {std_p:.3f}")

    # Step 6: Summary
    print("\n[6/6] Summary")
    print("=" * 80)
    print(f"\nModel: meta-llama/Llama-2-7b-chat-hf (8-bit)")
    print(f"\nData Split:")
    print(f"  Training:   {len(split.train_claims)} claims (mixed belief types)")
    print(f"  Validation: {len(split.val_claims)} claims (mixed belief types)")
    print(f"  Test:       {len(split.test_claims)} claims (mixed belief types)")
    print(f"\nTiming Breakdown:")
    print(f"  Model loading:      {format_time(load_time)}")
    print(f"  CCS training:       {format_time(train_time)}")
    print()
    for method_name, eval_time in timings.items():
        per_claim = eval_time / len(split.test_claims)
        print(f"  {method_name:20s} {format_time(eval_time):>10s} ({per_claim:.2f}s/claim)")

    total_time = load_time + train_time + sum(timings.values())
    print(f"\n  Total runtime:      {format_time(total_time)}")

    # Save split information for reproducibility
    import json
    split_info = {
        "seed": 42,
        "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "split_sizes": {
            "train": len(split.train_claims),
            "val": len(split.val_claims),
            "test": len(split.test_claims),
        },
        "test_claims": [claim.statement for claim in split.test_claims],
        "train_claims": [claim.statement for claim in split.train_claims],
        "val_claims": [claim.statement for claim in split.val_claims],
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✓ All outputs saved to: {output_dir}/")
    print(f"✓ Split information saved to: {output_dir / 'split_info.json'}")
    print("\nNext steps:")
    print("  1. Run: python scripts/generate_plots.py")
    print("  2. Download outputs/ folder to view results")
    print("\nNote: All methods evaluated on SAME test set (seed=42).")
    print("      CCS trained on separate training set.")
    print("      Validation set available for hyperparameter tuning.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
