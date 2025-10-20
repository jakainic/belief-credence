"""Visualization utilities for comparing credence estimates across methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from belief_credence.core import CredenceEstimate


def plot_method_comparison(
    estimates_by_method: dict[str, list[CredenceEstimate]],
    title: str = "Method Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> None:
    """Create a comprehensive comparison plot of multiple methods.

    Creates a figure with multiple subplots:
    - All pairwise scatter plots comparing methods
    - Distribution plot showing P(True) distributions
    - Correlation matrix
    - Summary statistics

    Args:
        estimates_by_method: Dict mapping method name to list of estimates
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    method_names = list(estimates_by_method.keys())
    num_methods = len(method_names)

    # Extract P(True) values
    p_true_by_method = {
        name: np.array([est.p_true for est in ests])
        for name, ests in estimates_by_method.items()
    }

    # Calculate number of pairwise comparisons
    num_pairs = num_methods * (num_methods - 1) // 2

    # Create layout: pairwise scatters on top, distributions and stats on bottom
    # For 2 methods: 1 scatter, then distributions, correlation matrix, stats
    # For 3 methods: 3 scatters, then distributions, correlation matrix, stats
    if num_methods == 2:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Single scatter plot (spans 2 columns)
        ax_scatter = fig.add_subplot(gs[0, :2])
        method1, method2 = method_names
        ax_scatter.scatter(
            p_true_by_method[method1],
            p_true_by_method[method2],
            alpha=0.6,
            s=50,
        )
        ax_scatter.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax_scatter.set_xlabel(method1, fontsize=10)
        ax_scatter.set_ylabel(method2, fontsize=10)
        ax_scatter.set_title(f"{method1} vs {method2}", fontsize=11)
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_xlim(-0.05, 1.05)
        ax_scatter.set_ylim(-0.05, 1.05)

        # Distribution plot
        ax_dist = fig.add_subplot(gs[0, 2])

        # Correlation matrix
        ax_corr = fig.add_subplot(gs[1, 0])

        # Stats (spans 2 columns)
        ax_stats = fig.add_subplot(gs[1, 1:])

    elif num_methods == 3:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.4)

        # Three scatter plots on top row
        pairs = [
            (method_names[0], method_names[1]),
            (method_names[0], method_names[2]),
            (method_names[1], method_names[2]),
        ]

        for idx, (method1, method2) in enumerate(pairs):
            ax = fig.add_subplot(gs[0, idx])
            ax.scatter(
                p_true_by_method[method1],
                p_true_by_method[method2],
                alpha=0.6,
                s=40,
            )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

            # Shorten method names for labels if too long
            m1_short = method1.split('_')[0] if '_' in method1 else method1
            m2_short = method2.split('_')[0] if '_' in method2 else method2

            ax.set_xlabel(m1_short, fontsize=9)
            ax.set_ylabel(m2_short, fontsize=9)
            ax.set_title(f"{m1_short} vs {m2_short}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        # Distribution plot (top right)
        ax_dist = fig.add_subplot(gs[0, 3])

        # Correlation matrix (bottom left)
        ax_corr = fig.add_subplot(gs[1, 0])

        # Stats (spans remaining columns)
        ax_stats = fig.add_subplot(gs[1, 1:])

    else:
        # For 4+ methods, use a more flexible grid
        fig = plt.figure(figsize=(18, 12))

        # Calculate grid dimensions for scatter plots
        scatter_cols = min(4, num_pairs)
        scatter_rows = (num_pairs + scatter_cols - 1) // scatter_cols

        gs = fig.add_gridspec(scatter_rows + 1, 4, hspace=0.3, wspace=0.4)

        # Create all pairwise scatter plots
        pair_idx = 0
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1:]:
                row = pair_idx // scatter_cols
                col = pair_idx % scatter_cols

                ax = fig.add_subplot(gs[row, col])
                ax.scatter(
                    p_true_by_method[method1],
                    p_true_by_method[method2],
                    alpha=0.6,
                    s=30,
                )
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

                m1_short = method1.split('_')[0] if '_' in method1 else method1
                m2_short = method2.split('_')[0] if '_' in method2 else method2

                ax.set_xlabel(m1_short, fontsize=8)
                ax.set_ylabel(m2_short, fontsize=8)
                ax.set_title(f"{m1_short} vs {m2_short}", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)

                pair_idx += 1

        # Distribution, correlation, and stats on bottom row
        ax_dist = fig.add_subplot(gs[scatter_rows, 0])
        ax_corr = fig.add_subplot(gs[scatter_rows, 1])
        ax_stats = fig.add_subplot(gs[scatter_rows, 2:])

    # Distribution plots
    for method_name in method_names:
        ax_dist.hist(
            p_true_by_method[method_name],
            bins=20,
            alpha=0.5,
            label=method_name.split('_')[0] if '_' in method_name else method_name,
            range=(0, 1),
        )
    ax_dist.set_xlabel("P(True)", fontsize=10)
    ax_dist.set_ylabel("Count", fontsize=10)
    ax_dist.set_title("P(True) Distributions", fontsize=11)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3, axis="y")

    # Correlation matrix
    corr_matrix = np.corrcoef([p_true_by_method[name] for name in method_names])
    im = ax_corr.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_corr.set_xticks(range(num_methods))
    ax_corr.set_yticks(range(num_methods))

    # Shorten method names for axis labels
    short_names = [name.split('_')[0] if '_' in name else name for name in method_names]
    ax_corr.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax_corr.set_yticklabels(short_names, fontsize=9)
    ax_corr.set_title("Correlation Matrix", fontsize=11)

    # Add correlation values
    for i in range(num_methods):
        for j in range(num_methods):
            text = ax_corr.text(
                j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                fontsize=8
            )

    plt.colorbar(im, ax=ax_corr, label="Correlation")

    # Statistics
    ax_stats.axis("off")

    stats_text = f"Statistics (n={len(next(iter(p_true_by_method.values())))} claims)\n\n"

    for method_name in method_names:
        p_values = p_true_by_method[method_name]
        stats_text += f"{method_name}:\n"
        stats_text += f"  Mean:   {np.mean(p_values):.3f}\n"
        stats_text += f"  Std:    {np.std(p_values):.3f}\n"
        stats_text += f"  Median: {np.median(p_values):.3f}\n"
        stats_text += f"  Range:  [{np.min(p_values):.3f}, {np.max(p_values):.3f}]\n\n"

    # Add pairwise disagreement
    if num_methods >= 2:
        stats_text += "Pairwise Mean Absolute Differences:\n"
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1:]:
                diff = np.abs(
                    p_true_by_method[method1] - p_true_by_method[method2]
                ).mean()
                stats_text += f"  {method1} vs {method2}: {diff:.3f}\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment="top", fontfamily="monospace")

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_claim_by_claim_comparison(
    estimates_by_method: dict[str, list[CredenceEstimate]],
    max_claims: int = 20,
    title: str = "Claim-by-Claim Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """Plot P(True) estimates for each claim across methods.

    Args:
        estimates_by_method: Dict mapping method name to list of estimates
        max_claims: Maximum number of claims to show
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    method_names = list(estimates_by_method.keys())
    num_claims = len(next(iter(estimates_by_method.values())))
    num_claims_to_show = min(num_claims, max_claims)

    fig, ax = plt.subplots(figsize=figsize)

    # Get claims and P(True) values
    claims = [est.claim.statement for est in next(iter(estimates_by_method.values()))]
    claims = claims[:num_claims_to_show]

    x = np.arange(num_claims_to_show)
    width = 0.8 / len(method_names)

    # Plot bars for each method
    for i, method_name in enumerate(method_names):
        p_values = [est.p_true for est in estimates_by_method[method_name][:num_claims_to_show]]
        offset = width * (i - len(method_names) / 2 + 0.5)
        ax.bar(x + offset, p_values, width, label=method_name, alpha=0.8)

    # Add reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="P=0.5")
    ax.axhline(y=0.0, color="gray", linestyle="-", alpha=0.2)
    ax.axhline(y=1.0, color="gray", linestyle="-", alpha=0.2)

    # Formatting
    ax.set_xlabel("Claim", fontsize=12)
    ax.set_ylabel("P(True)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    # Truncate claim labels for readability
    claim_labels = [claim[:40] + "..." if len(claim) > 40 else claim for claim in claims]
    ax.set_xticklabels(claim_labels, rotation=45, ha="right", fontsize=8)

    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_calibration_comparison(
    estimates_by_method: dict[str, list[CredenceEstimate]],
    bins: int = 10,
    title: str = "Calibration Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot calibration curves showing how spread out credences are.

    Args:
        estimates_by_method: Dict mapping method name to list of estimates
        bins: Number of bins for histogram
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    method_names = list(estimates_by_method.keys())

    for method_name in method_names:
        p_values = np.array([est.p_true for est in estimates_by_method[method_name]])

        # Create bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist, _ = np.histogram(p_values, bins=bin_edges)

        ax.plot(bin_centers, hist, marker="o", label=method_name, linewidth=2)

    ax.set_xlabel("P(True) Bin", fontsize=12)
    ax.set_ylabel("Number of Claims", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_comparison_report(
    estimates_by_method: dict[str, list[CredenceEstimate]],
    output_dir: str | Path,
    report_name: str = "method_comparison",
) -> None:
    """Generate a complete comparison report with all plots.

    Args:
        estimates_by_method: Dict mapping method name to list of estimates
        output_dir: Directory to save plots
        report_name: Base name for saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison report in {output_dir}/")

    # 1. Comprehensive comparison
    print("  - Creating comprehensive comparison plot...")
    plot_method_comparison(
        estimates_by_method,
        save_path=output_dir / f"{report_name}_overview.png",
    )

    # 2. Claim-by-claim
    print("  - Creating claim-by-claim comparison...")
    plot_claim_by_claim_comparison(
        estimates_by_method,
        save_path=output_dir / f"{report_name}_claims.png",
    )

    # 3. Calibration
    print("  - Creating calibration comparison...")
    plot_calibration_comparison(
        estimates_by_method,
        save_path=output_dir / f"{report_name}_calibration.png",
    )

    print(f"\nReport complete! Saved to {output_dir}/")
