"""Quick test script to verify RunPod setup.

This script runs a minimal test to ensure:
1. HF token is configured
2. Model loads successfully
3. All three methods work
4. Outputs can be saved

Run this FIRST on RunPod to verify setup:
    python scripts/test_setup.py

Expected runtime: ~2-3 minutes
"""

import os
import time
from pathlib import Path

from belief_credence import (
    DirectPrompting,
    LogitGap,
    CCS,
    Claim,
    save_estimates,
)
from belief_credence.model_utils import ModelWrapper


def main() -> None:
    print("=" * 80)
    print("RUNPOD SETUP TEST")
    print("=" * 80)

    # Test 1: Check HF token
    print("\n[1/5] Checking HuggingFace token...")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN is set (length: {len(hf_token)})")
    else:
        print("✗ ERROR: HF_TOKEN not found in environment")
        print("  Set it in RunPod pod configuration")
        return

    # Test 2: Load model
    print("\n[2/5] Loading model (this may take ~30s)...")
    start = time.time()
    try:
        model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)
        load_time = time.time() - start
        print(f"✓ Model loaded successfully in {load_time:.1f}s")
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        return

    # Test 3: Create test claims
    print("\n[3/5] Creating test claims...")
    test_claims = [
        Claim(
            statement="The Earth orbits around the Sun.",
            negation="The Earth does not orbit around the Sun.",
        ),
        Claim(
            statement="Water freezes at 0 degrees Celsius.",
            negation="Water does not freeze at 0 degrees Celsius.",
        ),
    ]
    print(f"✓ Created {len(test_claims)} test claims")

    # Test 4: Train CCS and test all methods
    print("\n[4/5] Testing all methods...")

    # Train CCS
    print("  Training CCS probe...")
    start = time.time()
    ccs = CCS(model=model, direction_method="logit_gap")
    ccs.train_probe(test_claims, epochs=10)  # Just 10 epochs for quick test
    train_time = time.time() - start
    print(f"  ✓ CCS trained in {train_time:.1f}s")

    # Test each method
    methods = [
        ("DirectPrompting", DirectPrompting(model=model)),
        ("LogitGap", LogitGap(model=model)),
        ("CCS", ccs),
    ]

    results = {}
    for method_name, method in methods:
        print(f"  Testing {method_name}...")
        start = time.time()

        # Test on single claim (no negation needed for evaluation!)
        test_claim = Claim(statement="Python is a programming language.")
        estimate = method.estimate(test_claim)

        elapsed = time.time() - start
        print(f"    ✓ P(True) = {estimate.p_true:.3f} (took {elapsed:.2f}s)")
        results[method_name] = estimate

    # Test 5: Save outputs
    print("\n[5/5] Testing output saving...")
    output_dir = Path("outputs/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_estimates = [results["DirectPrompting"]]
        output_file = output_dir / "test_output.json"
        save_estimates(test_estimates, output_file)
        print(f"✓ Saved test output to: {output_file}")
    except Exception as e:
        print(f"✗ ERROR saving output: {e}")
        return

    # Summary
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nYour RunPod setup is working correctly!")
    print("\nNext steps:")
    print("  1. Run full evaluation: python scripts/run_evaluation.py")
    print("  2. Generate plots: python scripts/generate_plots.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
