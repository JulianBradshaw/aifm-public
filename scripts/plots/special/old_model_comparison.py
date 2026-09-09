#!/usr/bin/env python3
"""
Plot CDF comparison between old model (from CSV) and new model AI2027-SC distribution.

Compares the old model's SC distribution from a CSV file (with x, pdf, cdf columns)
against the new model's AI2027-SC milestone distribution from rollouts using ECDFs.

Usage examples:
  # Compare with a specific run directory
  python scripts/plot_old_model_comparison.py --run-dir outputs/251209_updated_params

  # Specify custom old model CSV path
  python scripts/plot_old_model_comparison.py --run-dir outputs/251209_updated_params \
    --old-model-csv inputs/old_model/Eli_pdf_cdf.csv
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


def load_old_model_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load old model PDF/CDF data from CSV.

    Args:
        csv_path: Path to CSV file with x, pdf, cdf columns

    Returns:
        Tuple of (x, pdf, cdf) arrays
    """
    df = pd.read_csv(csv_path)
    x = df['x'].values
    pdf = df['pdf'].values
    cdf = df['cdf'].values
    return x, pdf, cdf


def load_new_model_data(rollouts_path: Path) -> tuple[np.ndarray, int, int]:
    """Load AI2027-SC milestone times from new model rollouts.

    Args:
        rollouts_path: Path to rollouts.jsonl

    Returns:
        Tuple of (times, num_not_achieved, total_rollouts)
    """
    reader = RolloutsReader(rollouts_path)
    times, num_not_achieved, _ = reader.read_milestone_times("AI2027-SC")
    total_rollouts = reader.count_rollouts()
    return np.array(times), num_not_achieved, total_rollouts


def plot_cdf_comparison(
    old_x: np.ndarray,
    old_cdf: np.ndarray,
    new_times: np.ndarray,
    total_rollouts: int,
    out_path: Path,
    x_min: float = 2020,
    x_max: float = 2050,
) -> None:
    """Plot CDF comparison between old and new models.

    Args:
        old_x: X values (years) for old model
        old_cdf: CDF values for old model
        new_times: SC arrival times from new model
        total_rollouts: Total number of rollouts (for proper censoring)
        out_path: Output file path
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter old model data to plot range
    mask = (old_x >= x_min) & (old_x <= x_max)

    # Plot old model CDF
    ax.plot(old_x[mask], old_cdf[mask], linewidth=2.5, color='tab:blue',
            label='AI 2027 Timelines Model (generated Apr 2025)\nSC dates in past due to internal/external gap', linestyle='--')

    # Create empirical CDF for new model
    # Normalize by total_rollouts to handle censoring: rollouts that never
    # achieved SC are treated as not happening at any finite time
    if len(new_times) >= 2:
        sorted_times = np.sort(new_times)
        ecdf_y = np.arange(1, len(sorted_times) + 1) / total_rollouts

        # Filter to plot range
        cdf_mask = (sorted_times >= x_min) & (sorted_times <= x_max)

        # Plot as step function for proper CDF visualization
        achieved_frac = len(new_times) / total_rollouts
        ax.step(sorted_times[cdf_mask], ecdf_y[cdf_mask], linewidth=2.5,
                color='tab:orange',
                label=f'AI Futures Model (Dec 2025)\n[{achieved_frac:.1%} achieve SC]',
                where='post')

        # Add percentile lines
        for pct in [0.1, 0.5, 0.9]:
            ax.axhline(pct, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # Calculate quantiles from the ECDF (unconditional, matching the plot)
        # Interpolate to find x where ecdf_y crosses the target percentile
        # If achieved_frac < target, the quantile is undefined (infinite)
        def ecdf_quantile(target_pct):
            if achieved_frac < target_pct:
                return np.inf  # Never reaches this percentile
            return float(np.interp(target_pct, ecdf_y, sorted_times))

        new_p10 = ecdf_quantile(0.1)
        new_median = ecdf_quantile(0.5)
        new_p90 = ecdf_quantile(0.9)

        # Find old model quantiles by interpolation
        old_p10 = np.interp(0.1, old_cdf, old_x)
        old_median = np.interp(0.5, old_cdf, old_x)
        old_p90 = np.interp(0.9, old_cdf, old_x)

        # Add vertical lines at medians
        ax.axvline(old_median, color='tab:blue', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'AI 2027 Median: {old_median:.1f}')
        if np.isfinite(new_median):
            ax.axvline(new_median, color='tab:orange', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'AI Futures Median: {new_median:.1f}')

        # Calculate probability of new model by old median (unconditional)
        prob_by_old_median = np.sum(new_times <= old_median) / total_rollouts

        # Add annotation for probability by old median
        # Find where to place annotation on CDF (at intersection with old median)
        ax.annotate(
            f'P(New ≤ {old_median:.1f}) = {prob_by_old_median:.1%}',
            xy=(old_median, prob_by_old_median),
            xytext=(old_median + 3, prob_by_old_median + 0.15),
            fontsize=11,
            color='tab:green',
            arrowprops=dict(arrowstyle='->', color='tab:green', alpha=0.7),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='tab:green', alpha=0.8)
        )

        def fmt_quantile(val):
            return f"{val:.1f}" if np.isfinite(val) else "∞"

        print(f"\nQuantile comparison (unconditional):")
        print(f"  AI 2027 model: P10={old_p10:.1f}, P50={old_median:.1f}, P90={old_p90:.1f}")
        print(f"  New model: P10={fmt_quantile(new_p10)}, P50={fmt_quantile(new_median)}, P90={fmt_quantile(new_p90)}")
        print(f"  P(New model SC ≤ AI 2027 median {old_median:.1f}): {prob_by_old_median:.1%}")
        print(f"  New model CDF asymptotes to {achieved_frac:.1%} (fraction achieving SC)")

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Cumulative Probability", fontsize=13)
    ax.set_title("SC Arrival Time CDF: AI 2027 Timelines Model vs AI Futures Model (Dec 2025)", fontsize=15, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved CDF comparison to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old model CDF with new model AI2027-SC distribution using ECDFs",
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
        default="inputs/old_model/Eli_pdf_cdf.csv",
        help="Path to old model CSV with x, pdf, cdf columns (default: inputs/old_model/Eli_pdf_cdf.csv)"
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
        default=2020,
        help="Minimum x-axis value (year) (default: 2020)"
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=2050,
        help="Maximum x-axis value (year) (default: 2050)"
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
    print(f"Loading old model data from: {old_model_csv}")
    old_x, _, old_cdf = load_old_model_data(old_model_csv)
    print(f"  X range: {old_x.min():.1f} - {old_x.max():.1f}")

    print(f"\nLoading new model data from: {rollouts_path}")
    new_times, num_not_achieved, total_rollouts = load_new_model_data(rollouts_path)
    achieved_pct = len(new_times) / total_rollouts * 100 if total_rollouts > 0 else 0
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  SC achieved: {len(new_times)} ({achieved_pct:.1f}%)")
    print(f"  SC not achieved: {num_not_achieved}")

    if len(new_times) < 2:
        print("Error: Not enough SC arrivals in new model to create comparison")
        return

    # Generate plots
    print("\nGenerating plots...")

    plot_cdf_comparison(
        old_x, old_cdf, new_times, total_rollouts,
        out_dir / "cdf_comparison.png",
        x_min=args.x_min, x_max=args.x_max
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
