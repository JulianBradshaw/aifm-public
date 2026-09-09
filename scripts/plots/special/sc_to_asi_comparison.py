#!/usr/bin/env python3
"""
Plot PDF and CDF comparison for SC-to-ASI duration between AI 2027 model and new model.

Compares the AI 2027 model's SC-to-ASI transition duration distribution from a CSV file
against the new model's AI2027-SC to ASI milestone transition duration.

Usage examples:
  # Compare with a specific run directory
  python scripts/plot_sc_to_asi_comparison.py --run-dir outputs/251209_updated_params

  # Specify custom old model CSV path
  python scripts/plot_sc_to_asi_comparison.py --run-dir outputs/251209_updated_params \
    --old-model-csv inputs/old_model/sc_to_milestones.csv
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup path for imports - need scripts/ directory for plotting_utils
scripts_dir = Path(__file__).resolve().parent.parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Project root for resolving relative paths
project_root = scripts_dir.parent

# Use monospace font for consistency
matplotlib.rcParams["font.family"] = "monospace"

# Import utilities
from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.kde import make_gamma_kernel_kde


def load_old_model_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load old model SC-to-ASI PDF/CDF data from CSV.

    Args:
        csv_path: Path to CSV file with years_from_sc, sc_to_asi_pdf, sc_to_asi_cdf columns

    Returns:
        Tuple of (years_from_sc, pdf, cdf) arrays
    """
    df = pd.read_csv(csv_path)
    x = df['years_from_sc'].values
    pdf = df['sc_to_asi_pdf'].values
    cdf = df['sc_to_asi_cdf'].values
    return x, pdf, cdf


def load_new_model_data(rollouts_path: Path) -> tuple[np.ndarray, int, int, int]:
    """Load SC-to-ASI transition durations from new model rollouts.

    Args:
        rollouts_path: Path to rollouts.jsonl

    Returns:
        Tuple of (durations, num_censored, total_with_sc, num_out_of_order)
        Note: durations only includes positive durations (in-order transitions).
        Out-of-order cases (ASI before SC) are counted separately in num_out_of_order
        and should be treated as having duration 0 (already happened).
    """
    reader = RolloutsReader(rollouts_path)

    # Read transition durations from AI2027-SC to ASI
    # Only includes positive durations (in-order transitions)
    durations_list, num_censored, _, num_out_of_order = reader.read_transition_durations(
        "AI2027-SC", "ASI", include_censored=False
    )

    # Count total rollouts where SC was achieved
    sc_times, num_no_sc, _ = reader.read_milestone_times("AI2027-SC")
    total_with_sc = len(sc_times)

    return np.array(durations_list), num_censored, total_with_sc, num_out_of_order


def plot_pdf_comparison(
    old_x: np.ndarray,
    old_pdf: np.ndarray,
    old_cdf: np.ndarray,
    new_durations: np.ndarray,
    out_path: Path,
    x_min: float = 0,
    x_max: float = 30,
) -> None:
    """Plot PDF comparison for SC-to-ASI duration.

    Args:
        old_x: X values (years from SC) for old model
        old_pdf: PDF values for old model
        old_cdf: CDF values for old model (used to find median)
        new_durations: SC-to-ASI durations from new model
        out_path: Output file path
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter old model data to plot range
    mask = (old_x >= x_min) & (old_x <= x_max)

    # Find old model median from CDF
    old_median = float(np.interp(0.5, old_cdf, old_x))

    # Plot old model PDF
    ax.plot(old_x[mask], old_pdf[mask], linewidth=2.5, color='tab:blue',
            label='AI 2027 Takeoff Model (generated Apr 2025)', linestyle='--')
    ax.fill_between(old_x[mask], old_pdf[mask], alpha=0.15, color='tab:blue')

    # Create KDE for new model using gamma kernel with lower bound at 0
    if len(new_durations) >= 2:
        kde = make_gamma_kernel_kde(new_durations, lower_bound=0.0)

        # Create evaluation points starting from 0
        xs = np.linspace(
            max(x_min, 0),
            min(x_max, new_durations.max() + 2),
            512
        )

        # Calculate PDF values
        pdf_values = kde.pdf(xs)

        # Plot new model PDF
        ax.plot(xs, pdf_values, linewidth=2.5, color='tab:orange', label='AI Futures Model (Dec 2025)')
        ax.fill_between(xs, pdf_values, alpha=0.15, color='tab:orange')

        # Calculate new model statistics
        new_median = np.median(new_durations)

        # Calculate probability of new model by old median
        prob_by_old_median = np.mean(new_durations <= old_median)

        # Add median lines
        ax.axvline(old_median, color='tab:blue', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'AI 2027 Median: {old_median:.1f} yr')
        ax.axvline(new_median, color='tab:orange', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'AI Futures Median: {new_median:.1f} yr')

        # Add annotation for probability by old median
        y_pos = ax.get_ylim()[1] * 0.85
        ax.annotate(
            f'P(New ≤ {old_median:.1f}) = {prob_by_old_median:.1%}',
            xy=(old_median, y_pos),
            xytext=(old_median + 2, y_pos),
            fontsize=11,
            color='tab:green',
            arrowprops=dict(arrowstyle='->', color='tab:green', alpha=0.7),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='tab:green', alpha=0.8)
        )

        print(f"\nPDF statistics:")
        print(f"  AI 2027 median: {old_median:.1f} years")
        print(f"  New model median: {new_median:.1f} years")
        print(f"  P(New model SC→ASI ≤ AI 2027 median {old_median:.1f}): {prob_by_old_median:.1%}")

    ax.set_xlabel("Years from SC to ASI", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("SC to ASI Duration PDF: AI 2027 Takeoff Model vs AI Futures Model (Dec 2025)", fontsize=15, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(x_min, x_max)

    # Ensure y-axis starts at 0
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved PDF comparison to: {out_path}")


def plot_cdf_comparison(
    old_x: np.ndarray,
    old_cdf: np.ndarray,
    new_durations: np.ndarray,
    total_with_sc: int,
    num_out_of_order: int,
    out_path: Path,
    x_min: float = 0,
    x_max: float = 30,
) -> None:
    """Plot CDF comparison for SC-to-ASI duration.

    Args:
        old_x: X values (years from SC) for old model
        old_cdf: CDF values for old model
        new_durations: SC-to-ASI durations from new model (positive durations only)
        total_with_sc: Total rollouts that achieved SC (for proper censoring)
        num_out_of_order: Number of out-of-order transitions (ASI before SC, duration=0)
        out_path: Output file path
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter old model data to plot range
    mask = (old_x >= x_min) & (old_x <= x_max)

    # Plot old model CDF
    ax.plot(old_x[mask], old_cdf[mask], linewidth=2.5, color='tab:blue',
            label='AI 2027 Takeoff Model (generated Apr 2025)', linestyle='--')

    # Create empirical CDF for new model
    # Out-of-order cases (ASI before SC) have duration 0 - the CDF starts at that fraction
    # Normalize by total_with_sc to handle censoring: rollouts where SC was
    # achieved but ASI wasn't are treated as not happening at any finite duration
    if len(new_durations) >= 2 or num_out_of_order > 0:
        # Fraction that already reached ASI before SC (out-of-order, duration=0)
        start_frac = num_out_of_order / total_with_sc

        # Total achieved = in-order + out-of-order
        total_achieved = len(new_durations) + num_out_of_order
        achieved_frac = total_achieved / total_with_sc

        # Sort positive durations and build ECDF offset by out-of-order fraction
        sorted_durations = np.sort(new_durations)
        # Each positive duration adds 1/total_with_sc to the CDF, starting from start_frac
        ecdf_y = start_frac + np.arange(1, len(sorted_durations) + 1) / total_with_sc

        # Build plot arrays: start at (0, start_frac), then step through positive durations
        plot_x = np.concatenate([[0], sorted_durations])
        plot_y = np.concatenate([[start_frac], ecdf_y])

        # Filter to plot range
        cdf_mask = (plot_x >= x_min) & (plot_x <= x_max)

        # Plot as step function for proper CDF visualization
        ax.step(plot_x[cdf_mask], plot_y[cdf_mask], linewidth=2.5,
                color='tab:orange',
                label=f'AI Futures Model (Dec 2025)\n[{achieved_frac:.1%} reach ASI | SC]',
                where='post')

        # Calculate quantiles from the ECDF (unconditional, matching the plot)
        # Need to use the full ecdf including the start point
        full_ecdf_y = np.concatenate([[start_frac], ecdf_y])
        full_ecdf_x = np.concatenate([[0], sorted_durations])

        def ecdf_quantile(target_pct):
            if achieved_frac < target_pct:
                return np.inf  # Never reaches this percentile
            # If target is at or below start_frac, quantile is 0
            if target_pct <= start_frac:
                return 0.0
            return float(np.interp(target_pct, full_ecdf_y, full_ecdf_x))

        new_p10 = ecdf_quantile(0.1)
        new_median = ecdf_quantile(0.5)
        new_p90 = ecdf_quantile(0.9)

        # Find old model quantiles by interpolation
        old_p10 = np.interp(0.1, old_cdf, old_x)
        old_median = np.interp(0.5, old_cdf, old_x)
        old_p90 = np.interp(0.9, old_cdf, old_x)

        # Add vertical lines at medians
        ax.axvline(old_median, color='tab:blue', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'AI 2027 Median: {old_median:.1f} yr')
        if np.isfinite(new_median):
            ax.axvline(new_median, color='tab:orange', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'AI Futures Median: {new_median:.1f} yr')
        else:
            # Median not reached (achieved_frac < 50%)
            pass  # Don't draw a line for infinite median

        # Add horizontal line at median (0.5) - no legend entry
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

        # Add horizontal line at start_frac (probability ASI before SC)
        if start_frac > 0:
            ax.axhline(start_frac, color='tab:purple', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'P(ASI before SC) = {start_frac:.1%}')

        # Calculate probability of new model by old median (unconditional on ASI)
        # Include out-of-order cases (they all have duration 0 <= old_median)
        prob_by_old_median = (num_out_of_order + np.sum(new_durations <= old_median)) / total_with_sc
        ax.axhline(prob_by_old_median, color='tab:green', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'P(New ≤ AI 2027 median) = {prob_by_old_median:.1%}')

        def fmt_quantile(val):
            return f"{val:.1f}" if np.isfinite(val) else "∞"

        print(f"\nQuantile comparison (years from SC to ASI, unconditional):")
        print(f"  AI 2027 model: P10={old_p10:.1f}, P50={old_median:.1f}, P90={old_p90:.1f}")
        print(f"  New model: P10={fmt_quantile(new_p10)}, P50={fmt_quantile(new_median)}, P90={fmt_quantile(new_p90)}")
        print(f"  P(New model SC→ASI ≤ AI 2027 median {old_median:.1f}): {prob_by_old_median:.1%}")
        print(f"  New model CDF starts at {start_frac:.1%} (out-of-order: ASI before SC)")
        print(f"  New model CDF asymptotes to {achieved_frac:.1%} (fraction reaching ASI given SC)")

    ax.set_xlabel("Years from SC to ASI", fontsize=13)
    ax.set_ylabel("Cumulative Probability", fontsize=13)
    ax.set_title("SC to ASI Duration CDF: AI 2027 Takeoff Model vs AI Futures Model (Dec 2025)", fontsize=15, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved CDF comparison to: {out_path}")


def plot_combined_comparison(
    old_x: np.ndarray,
    old_pdf: np.ndarray,
    old_cdf: np.ndarray,
    new_durations: np.ndarray,
    total_with_sc: int,
    num_out_of_order: int,
    out_path: Path,
    x_min: float = 0,
    x_max: float = 30,
) -> None:
    """Plot combined PDF and CDF comparison as subplots.

    Args:
        old_x: X values (years from SC) for old model
        old_pdf: PDF values for old model
        old_cdf: CDF values for old model
        new_durations: SC-to-ASI durations from new model (positive durations only)
        total_with_sc: Total rollouts that achieved SC (for proper censoring)
        num_out_of_order: Number of out-of-order transitions (ASI before SC, duration=0)
        out_path: Output file path
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
    """
    fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(14, 12))

    # Filter old model data to plot range
    mask = (old_x >= x_min) & (old_x <= x_max)

    # Find old model median from CDF
    old_median = float(np.interp(0.5, old_cdf, old_x))

    # --- PDF subplot ---
    ax_pdf.plot(old_x[mask], old_pdf[mask], linewidth=2.5, color='tab:blue',
                label='AI 2027 Takeoff Model (generated Apr 2025)', linestyle='--')
    ax_pdf.fill_between(old_x[mask], old_pdf[mask], alpha=0.15, color='tab:blue')

    if len(new_durations) >= 2:
        kde = make_gamma_kernel_kde(new_durations, lower_bound=0.0)
        xs = np.linspace(
            max(x_min, 0),
            min(x_max, new_durations.max() + 2),
            512
        )
        pdf_values = kde.pdf(xs)

        ax_pdf.plot(xs, pdf_values, linewidth=2.5, color='tab:orange', label='AI Futures Model (Dec 2025)')
        ax_pdf.fill_between(xs, pdf_values, alpha=0.15, color='tab:orange')

        new_median = np.median(new_durations)
        prob_by_old_median = np.mean(new_durations <= old_median)

        ax_pdf.axvline(old_median, color='tab:blue', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'AI 2027 Median: {old_median:.1f} yr')
        ax_pdf.axvline(new_median, color='tab:orange', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'AI Futures Median: {new_median:.1f} yr')

        # Add annotation for probability by old median
        y_pos = ax_pdf.get_ylim()[1] * 0.85
        ax_pdf.annotate(
            f'P(New ≤ {old_median:.1f}) = {prob_by_old_median:.1%}',
            xy=(old_median, y_pos),
            xytext=(old_median + 2, y_pos),
            fontsize=10,
            color='tab:green',
            arrowprops=dict(arrowstyle='->', color='tab:green', alpha=0.7),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='tab:green', alpha=0.8)
        )

    ax_pdf.set_xlabel("Years from SC to ASI", fontsize=12)
    ax_pdf.set_ylabel("Probability Density", fontsize=12)
    ax_pdf.set_title("PDF Comparison", fontsize=14)
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.legend(loc='upper right', fontsize=11)
    ax_pdf.set_xlim(x_min, x_max)
    y_min, y_max = ax_pdf.get_ylim()
    ax_pdf.set_ylim(0, y_max)

    # --- CDF subplot ---
    ax_cdf.plot(old_x[mask], old_cdf[mask], linewidth=2.5, color='tab:blue',
                label='AI 2027 Takeoff Model (generated Apr 2025)', linestyle='--')

    if len(new_durations) >= 2 or num_out_of_order > 0:
        # Fraction that already reached ASI before SC (out-of-order, duration=0)
        start_frac = num_out_of_order / total_with_sc

        # Total achieved = in-order + out-of-order
        total_achieved = len(new_durations) + num_out_of_order
        achieved_frac = total_achieved / total_with_sc

        # Sort positive durations and build ECDF offset by out-of-order fraction
        sorted_durations = np.sort(new_durations)
        ecdf_y = start_frac + np.arange(1, len(sorted_durations) + 1) / total_with_sc

        # Build plot arrays: start at (0, start_frac), then step through positive durations
        plot_x = np.concatenate([[0], sorted_durations])
        plot_y = np.concatenate([[start_frac], ecdf_y])
        cdf_mask = (plot_x >= x_min) & (plot_x <= x_max)

        ax_cdf.step(plot_x[cdf_mask], plot_y[cdf_mask], linewidth=2.5,
                    color='tab:orange',
                    label=f'AI Futures Model (Dec 2025)\n[{achieved_frac:.1%} reach ASI | SC]',
                    where='post')

        # Compute median from ECDF (unconditional, matching the plot)
        full_ecdf_y = np.concatenate([[start_frac], ecdf_y])
        full_ecdf_x = np.concatenate([[0], sorted_durations])
        if achieved_frac >= 0.5:
            if 0.5 <= start_frac:
                new_median = 0.0
            else:
                new_median = float(np.interp(0.5, full_ecdf_y, full_ecdf_x))
        else:
            new_median = np.inf

        ax_cdf.axvline(old_median, color='tab:blue', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'AI 2027 Median: {old_median:.1f} yr')
        if np.isfinite(new_median):
            ax_cdf.axvline(new_median, color='tab:orange', linestyle=':', linewidth=1.5, alpha=0.7,
                           label=f'AI Futures Median: {new_median:.1f} yr')

    ax_cdf.set_xlabel("Years from SC to ASI", fontsize=12)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=12)
    ax_cdf.set_title("CDF Comparison", fontsize=14)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend(loc='lower right', fontsize=11)
    ax_cdf.set_xlim(x_min, x_max)
    ax_cdf.set_ylim(0, 1.05)

    fig.suptitle("SC to ASI Duration: AI 2027 Takeoff Model vs AI Futures Model (Dec 2025)", fontsize=16, y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined comparison to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AI 2027 SC-to-ASI duration with new model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory containing rollouts.jsonl"
    )
    parser.add_argument(
        "--old-model-csv",
        type=str,
        default="inputs/old_model/sc_to_milestones.csv",
        help="Path to old model CSV with sc_to_asi columns (default: inputs/old_model/sc_to_milestones.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <run-dir>/old_model_comparison/)"
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=0,
        help="Minimum x-axis value (years from SC) (default: 0)"
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=30,
        help="Maximum x-axis value (years from SC) (default: 30)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    rollouts_path = run_dir / "rollouts.jsonl"
    cache_path = run_dir / "rollouts.cache.json"

    # Check for either rollouts.jsonl or cache file
    if not rollouts_path.exists() and not cache_path.exists():
        raise FileNotFoundError(
            f"Neither rollouts file nor cache found: {rollouts_path}"
        )

    # Resolve old model CSV path relative to project root if not absolute
    old_model_csv = Path(args.old_model_csv)
    if not old_model_csv.is_absolute():
        old_model_csv = project_root / old_model_csv

    if not old_model_csv.exists():
        raise FileNotFoundError(f"Old model CSV not found: {old_model_csv}")

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = run_dir / "old_model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading AI 2027 model data from: {old_model_csv}")
    old_x, old_pdf, old_cdf = load_old_model_data(old_model_csv)
    print(f"  X range: {old_x.min():.1f} - {old_x.max():.1f} years from SC")

    print(f"\nLoading new model data from: {rollouts_path}")
    new_durations, num_censored, total_with_sc, num_out_of_order = load_new_model_data(rollouts_path)
    total_achieved = len(new_durations) + num_out_of_order
    achieved_pct = total_achieved / total_with_sc * 100 if total_with_sc > 0 else 0
    print(f"  Rollouts with SC achieved: {total_with_sc}")
    print(f"  SC→ASI transitions: {total_achieved} ({achieved_pct:.1f}%)")
    print(f"    - In order (SC before ASI): {len(new_durations)}")
    print(f"    - Out of order (ASI before SC, duration=0): {num_out_of_order}")
    print(f"  Censored (SC but no ASI): {num_censored}")

    if len(new_durations) < 2 and num_out_of_order == 0:
        print("Error: Not enough SC→ASI transitions in new model to create comparison")
        return

    # Generate plots
    print("\nGenerating plots...")

    plot_pdf_comparison(
        old_x, old_pdf, old_cdf, new_durations,
        out_dir / "sc_to_asi_pdf_comparison.png",
        x_min=args.x_min, x_max=args.x_max
    )

    plot_cdf_comparison(
        old_x, old_cdf, new_durations, total_with_sc, num_out_of_order,
        out_dir / "sc_to_asi_cdf_comparison.png",
        x_min=args.x_min, x_max=args.x_max
    )

    plot_combined_comparison(
        old_x, old_pdf, old_cdf, new_durations, total_with_sc, num_out_of_order,
        out_dir / "sc_to_asi_combined_comparison.png",
        x_min=args.x_min, x_max=args.x_max
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
