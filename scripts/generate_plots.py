"""Generate visualization plots from saved evaluation outputs.

This script loads saved estimates and creates all comparison plots.

Run after run_evaluation.py:
    python scripts/generate_plots.py

Optional arguments:
    --input-dir PATH    Directory with saved estimates (default: outputs/runpod_evaluation)
    --output-dir PATH   Directory to save plots (default: outputs/visualizations)
"""

import argparse
from pathlib import Path

from belief_credence import (
    load_estimates,
    create_comparison_report,
    plot_method_comparison,
    plot_claim_by_claim_comparison,
    plot_agreement_heatmap,
    plot_calibration_comparison,
    print_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots from saved estimates")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/runpod_evaluation",
        help="Directory containing saved estimate JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="method_comparison",
        help="Base name for saved plot files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)

    # Find all JSON files in input directory (excluding split_info.json)
    print(f"\nLooking for estimates in: {input_dir}/")
    json_files = [f for f in input_dir.glob("*.json") if f.name != "split_info.json"]

    if not json_files:
        print(f"ERROR: No estimate JSON files found in {input_dir}/")
        print("Make sure to run: python scripts/run_evaluation.py first")
        return

    print(f"Found {len(json_files)} estimate files:")
    for f in json_files:
        print(f"  - {f.name}")

    # Load all estimates
    print("\nLoading estimates...")
    estimates_by_method = {}

    for json_file in json_files:
        try:
            estimates = load_estimates(json_file)
            if estimates:
                method_name = estimates[0].method
                estimates_by_method[method_name] = estimates
                print(f"  ✓ Loaded {len(estimates)} estimates from {method_name}")
        except (KeyError, ValueError) as e:
            print(f"  ⚠ Skipping {json_file.name}: not an estimates file")

    if not estimates_by_method:
        print("ERROR: No valid estimates loaded")
        return

    num_claims = len(next(iter(estimates_by_method.values())))
    print(f"\nTotal: {len(estimates_by_method)} methods, {num_claims} claims each")

    # Generate all plots
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Option 1: Generate complete report (all plots)
    print("\nGenerating complete comparison report...")
    create_comparison_report(
        estimates_by_method,
        output_dir=output_dir,
        report_name=args.report_name,
    )

    print(f"\n✓ All plots saved to: {output_dir}/")
    print("\nGenerated files:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")

    # Generate validation report
    print("\n" + "=" * 80)
    print_validation_report(estimates_by_method, agreement_threshold=0.15)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for method_name, estimates in estimates_by_method.items():
        p_values = [est.p_true for est in estimates]
        mean_p = sum(p_values) / len(p_values)
        std_p = (sum((p - mean_p) ** 2 for p in p_values) / len(p_values)) ** 0.5
        min_p = min(p_values)
        max_p = max(p_values)

        print(f"\n{method_name}:")
        print(f"  Mean P(True):   {mean_p:.3f}")
        print(f"  Std Dev:        {std_p:.3f}")
        print(f"  Range:          [{min_p:.3f}, {max_p:.3f}]")

    # Pairwise correlations
    if len(estimates_by_method) >= 2:
        import numpy as np

        print("\nPairwise Correlations:")
        method_names = list(estimates_by_method.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1 :]:
                p1 = np.array([est.p_true for est in estimates_by_method[method1]])
                p2 = np.array([est.p_true for est in estimates_by_method[method2]])
                corr = np.corrcoef(p1, p2)[0, 1]
                mae = np.abs(p1 - p2).mean()
                print(f"  {method1} vs {method2}:")
                print(f"    Correlation: {corr:.3f}")
                print(f"    Mean Absolute Error: {mae:.3f}")

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    print(f"\nDownload {output_dir}/ to view all plots!")


if __name__ == "__main__":
    main()
