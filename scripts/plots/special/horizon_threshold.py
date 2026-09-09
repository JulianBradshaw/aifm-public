#!/usr/bin/env python3
"""
Analyze and plot the distribution of times when trajectories reach 120,000 work-year horizon.

WARNING: This script contains bespoke data loading and plotting code that doesn't
follow the project's plotting utility conventions. It should be refactored to use
utilities from scripts/plotting/ and scripts/plotting_utils/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import sys


def read_horizon_threshold_times(
    rollouts_file: Path,
    threshold_years: float = 120000.0
) -> Tuple[List[float], List[Optional[float]], List[Optional[float]]]:
    """Extract the times when each trajectory first hits the threshold horizon.

    Args:
        rollouts_file: Path to rollouts.jsonl
        threshold_years: Threshold in work-years (default 120,000)

    Returns:
        threshold_times: List of times (in decimal years) when threshold is first reached
        aa_times: List of aa_time values for each trajectory
        mse_values: List of METR MSE values for each trajectory
    """
    threshold_minutes = threshold_years * 52 * 40 * 60  # Convert work-years to minutes
    threshold_times: List[float] = []
    aa_times: List[Optional[float]] = []
    mse_values: List[Optional[float]] = []

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            results = rec.get("results")
            if not isinstance(results, dict):
                continue

            times = results.get("times")
            horizon = results.get("horizon_lengths")
            aa_time_val = results.get("aa_time")
            mse_val = results.get("metr_mse")

            if times is None or horizon is None:
                continue

            try:
                times_arr = np.asarray(times, dtype=float)
                horizon_arr = np.asarray(horizon, dtype=float)
            except Exception:
                continue

            if times_arr.ndim != 1 or horizon_arr.ndim != 1 or times_arr.size != horizon_arr.size:
                continue

            # Find first time when horizon exceeds threshold
            exceeds_threshold = horizon_arr >= threshold_minutes
            if np.any(exceeds_threshold):
                first_idx = np.argmax(exceeds_threshold)
                threshold_time = times_arr[first_idx]
                threshold_times.append(float(threshold_time))

                # Store aa_time and MSE
                try:
                    aa_times.append(float(aa_time_val) if aa_time_val is not None and np.isfinite(float(aa_time_val)) else None)
                except Exception:
                    aa_times.append(None)

                try:
                    mse_values.append(float(mse_val) if mse_val is not None and np.isfinite(float(mse_val)) else None)
                except Exception:
                    mse_values.append(None)

    return threshold_times, aa_times, mse_values


def plot_threshold_distribution(
    threshold_times: List[float],
    out_path: Path,
    threshold_years: float = 120000.0,
    aa_times: Optional[List[Optional[float]]] = None,
    mse_values: Optional[List[Optional[float]]] = None,
    mse_threshold: Optional[float] = None,
    bins: int = 50
) -> None:
    """Create histogram of when trajectories reach the threshold.

    Args:
        threshold_times: List of times when threshold was reached
        out_path: Output path for the plot
        threshold_years: Threshold in work-years (for title)
        aa_times: Optional list of aa_time values
        mse_values: Optional list of MSE values for filtering
        mse_threshold: Optional MSE threshold for filtering
        bins: Number of histogram bins
    """
    times_arr = np.array(threshold_times)

    # Filter by MSE if requested
    if mse_threshold is not None and mse_values is not None:
        filtered_indices = []
        for i, mse in enumerate(mse_values):
            if mse is None or mse <= mse_threshold:
                filtered_indices.append(i)
        times_arr = times_arr[filtered_indices]
        print(f"Filtered to {len(times_arr)} trajectories with MSE <= {mse_threshold}")

    if len(times_arr) == 0:
        print("No trajectories reached the threshold")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Histogram
    counts, bin_edges, patches = ax1.hist(times_arr, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Trajectories', fontsize=12)
    ax1.set_title(f'Distribution of Times When Trajectories Reach {threshold_years:,.0f} Work-Year Horizon\n'
                  f'(n={len(times_arr)} trajectories reaching threshold)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.median(times_arr), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(times_arr):.1f}')
    ax1.legend()

    # Statistics text
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {np.mean(times_arr):.2f}\n'
    stats_text += f'Median: {np.median(times_arr):.2f}\n'
    stats_text += f'Std Dev: {np.std(times_arr):.2f}\n'
    stats_text += f'Min: {np.min(times_arr):.2f}\n'
    stats_text += f'Max: {np.max(times_arr):.2f}\n'
    stats_text += f'25th percentile: {np.percentile(times_arr, 25):.2f}\n'
    stats_text += f'75th percentile: {np.percentile(times_arr, 75):.2f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Cumulative distribution
    sorted_times = np.sort(times_arr)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
    ax2.plot(sorted_times, cumulative, color='darkgreen', linewidth=2)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution Function', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(np.median(times_arr), color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved threshold distribution plot to {out_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python plot_horizon_threshold_times.py <rollouts_dir> [threshold_years] [mse_threshold]")
        print("Example: python plot_horizon_threshold_times.py outputs/251110_eli_2200 120000 1.0")
        sys.exit(1)

    rollouts_dir = Path(sys.argv[1])
    threshold_years = float(sys.argv[2]) if len(sys.argv) > 2 else 120000.0
    mse_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    rollouts_file = rollouts_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        print(f"Error: {rollouts_file} not found")
        sys.exit(1)

    print(f"Reading threshold times from {rollouts_file}...")
    threshold_times, aa_times, mse_values = read_horizon_threshold_times(rollouts_file, threshold_years)

    print(f"Found {len(threshold_times)} trajectories that reached {threshold_years:,.0f} work-year horizon")

    if len(threshold_times) == 0:
        print("No trajectories reached the threshold. Exiting.")
        sys.exit(0)

    # Create output directory
    out_dir = rollouts_dir / "threshold_analysis"
    out_dir.mkdir(exist_ok=True)

    # Plot distribution
    if mse_threshold is not None:
        out_path = out_dir / f"threshold_{int(threshold_years)}_years_mse_{mse_threshold}.png"
    else:
        out_path = out_dir / f"threshold_{int(threshold_years)}_years.png"

    plot_threshold_distribution(
        threshold_times,
        out_path,
        threshold_years,
        aa_times=aa_times,
        mse_values=mse_values,
        mse_threshold=mse_threshold
    )

    # Print summary statistics
    times_arr = np.array(threshold_times)
    if mse_threshold is not None and mse_values is not None:
        filtered_indices = [i for i, mse in enumerate(mse_values) if mse is None or mse <= mse_threshold]
        times_arr = times_arr[filtered_indices]

    print(f"\nSummary Statistics:")
    print(f"  Mean: {np.mean(times_arr):.2f}")
    print(f"  Median: {np.median(times_arr):.2f}")
    print(f"  Std Dev: {np.std(times_arr):.2f}")
    print(f"  Min: {np.min(times_arr):.2f}")
    print(f"  Max: {np.max(times_arr):.2f}")
    print(f"  25th percentile: {np.percentile(times_arr, 25):.2f}")
    print(f"  75th percentile: {np.percentile(times_arr, 75):.2f}")


if __name__ == "__main__":
    main()
