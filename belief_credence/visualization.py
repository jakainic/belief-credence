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
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Create a comprehensive comparison plot of multiple methods.

    Creates a figure with multiple subplots:
    - Scatter plot comparing each method pair
    - Distribution plot showing P(True) distributions
    - Correlation heatmap

    Args:
        estimates_by_method: Dict mapping method name to list of estimates
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    method_names = list(estimates_by_method.keys())
    num_methods = len(method_names)

    # Extract P(True) values
    p_true_by_method = {
        name: np.array([est.p_true for est in ests])
        for name, ests in estimates_by_method.items()
    }

    # 1. Pairwise scatter plots (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if num_methods == 2:
        method1, method2 = method_names
        ax1.scatter(
            p_true_by_method[method1],
            p_true_by_method[method2],
            alpha=0.6,
            s=50,
        )
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax1.set_xlabel(method1)
        ax1.set_ylabel(method2)
        ax1.set_title("Method Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    elif num_methods == 3:
        # Show first two methods
        method1, method2 = method_names[0], method_names[1]
        ax1.scatter(
            p_true_by_method[method1],
            p_true_by_method[method2],
            alpha=0.6,
            s=50,
            label=f"{method1} vs {method2}",
        )
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel(method1)
        ax1.set_ylabel(method2)
        ax1.set_title("Pairwise Comparison")
        ax1.grid(True, alpha=0.3)

    # 2. Distribution plots (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    for method_name in method_names:
        ax2.hist(
            p_true_by_method[method_name],
            bins=20,
            alpha=0.5,
            label=method_name,
            range=(0, 1),
        )
    ax2.set_xlabel("P(True)")
    ax2.set_ylabel("Count")
    ax2.set_title("P(True) Distributions")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Correlation heatmap (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    corr_matrix = np.corrcoef([p_true_by_method[name] for name in method_names])
    im = ax3.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(num_methods))
    ax3.set_yticks(range(num_methods))
    ax3.set_xticklabels(method_names, rotation=45, ha="right")
    ax3.set_yticklabels(method_names)
    ax3.set_title("Correlation Matrix")

    # Add correlation values
    for i in range(num_methods):
        for j in range(num_methods):
            text = ax3.text(
                j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black"
            )

    plt.colorbar(im, ax=ax3, label="Correlation")

    # 4. Agreement statistics (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

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
            for method2 in method_names[i + 1 :]:
                diff = np.abs(
                    p_true_by_method[method1] - p_true_by_method[method2]
                ).mean()
                stats_text += f"  {method1} vs {method2}: {diff:.3f}\n"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace")

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


def plot_agreement_heatmap(
    estimates_by_method: dict[str, list[CredenceEstimate]],
    title: str = "Agreement Heatmap",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Plot heatmap showing where methods agree/disagree.

    Args:
        estimates_by_method: Dict mapping method name to list of estimates
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    method_names = list(estimates_by_method.keys())
    num_claims = len(next(iter(estimates_by_method.values())))
    claims = [est.claim.statement for est in next(iter(estimates_by_method.values()))]

    # Create matrix of P(True) values
    p_matrix = np.array(
        [[est.p_true for est in estimates_by_method[name]] for name in method_names]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})

    # Main heatmap
    im1 = ax1.imshow(p_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax1.set_yticks(range(len(method_names)))
    ax1.set_yticklabels(method_names)
    ax1.set_xlabel("Claim Index", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im1, ax=ax1, label="P(True)")

    # Disagreement heatmap (standard deviation across methods)
    disagreement = np.std(p_matrix, axis=0)
    im2 = ax2.imshow(
        disagreement.reshape(-1, 1), cmap="Reds", vmin=0, vmax=0.5, aspect="auto"
    )
    ax2.set_xticks([])
    ax2.set_yticks(range(0, num_claims, max(1, num_claims // 20)))
    ax2.set_title("Disagreement\n(Std Dev)", fontsize=10)
    plt.colorbar(im2, ax=ax2, label="Std Dev")

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

    # 3. Agreement heatmap
    print("  - Creating agreement heatmap...")
    plot_agreement_heatmap(
        estimates_by_method,
        save_path=output_dir / f"{report_name}_heatmap.png",
    )

    # 4. Calibration
    print("  - Creating calibration comparison...")
    plot_calibration_comparison(
        estimates_by_method,
        save_path=output_dir / f"{report_name}_calibration.png",
    )

    print(f"\nReport complete! Saved to {output_dir}/")
