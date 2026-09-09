#!/usr/bin/env python3
"""
Generate histogram plots for milestone arrival times.

This script reads rollout results and creates histogram plots showing when each
milestone is reached across all rollouts. It creates an overlaid histogram plot
for AC, SC, SAR, and SIAR milestones.

Usage examples:
  # Generate overlay histogram for milestones
  python scripts/plot_milestone_histograms.py --rollouts outputs/251110_eli_2200_50000_runs/rollouts.jsonl

  # Customize output path
  python scripts/plot_milestone_histograms.py --rollouts outputs/251110_eli_2200_50000_runs/rollouts.jsonl \
    --out-path outputs/251110_eli_2200_50000_runs/milestone_histograms_overlay.png

  # Adjust bin width (default: 1 year)
  python scripts/plot_milestone_histograms.py --rollouts outputs/251110_eli_2200_50000_runs/rollouts.jsonl \
    --bin-width 0.5
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

# Import utilities
from plotting_utils.rollouts_reader import RolloutsReader


def plot_milestone_histograms_overlay(
    rollouts_file: Path,
    milestone_names: List[str],
    out_path: Path,
    title: Optional[str] = None,
    bin_width: float = 1.0,
    max_year: float = 2050,
) -> None:
    """Plot overlaid histograms for multiple milestones.

    Args:
        rollouts_file: path to rollouts.jsonl
        milestone_names: list of milestone names to plot
        out_path: output file path
        title: optional custom title
        bin_width: width of histogram bins in years (default: 1.0)
        max_year: maximum year to plot (default: 2050)
    """
    # Filter to only keep AC, SC, SAR, SIAR
    allowed_milestones = {'AC', 'AI2027-SC', 'SAR-level-experiment-selection-skill', 'SIAR-level-experiment-selection-skill'}
    milestone_names = [m for m in milestone_names if m in allowed_milestones]

    if not milestone_names:
        print("Warning: No milestones matching AC, SC, SAR, or SIAR found")
        return

    plt.figure(figsize=(14, 8))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    stats_lines = []

    # Find global range for consistent binning
    all_times = []
    milestone_data = []

    # Helper function to clean milestone names for display
    def clean_milestone_name(name: str) -> str:
        """Remove 'level' and 'skill' from milestone names."""
        return name.replace("-level", "").replace("-skill", "")

    # First pass: collect all data
    reader = RolloutsReader(rollouts_file)
    for milestone_name in milestone_names:
        times, num_not_achieved, sim_end = reader.read_milestone_times(milestone_name)

        if len(times) < 2:
            print(f"Warning: Not enough data for {milestone_name}, skipping")
            continue

        all_times.extend(times)
        milestone_data.append({
            'name': milestone_name,
            'times': times,
            'num_not_achieved': num_not_achieved,
            'sim_end': sim_end
        })

    if not milestone_data:
        print("Error: No milestones with sufficient data")
        return

    # Create bins
    min_time = min(all_times)
    bins = np.arange(min_time, max_year + bin_width, bin_width)

    # Second pass: plot histograms
    for idx, data in enumerate(milestone_data):
        milestone_name = data['name']
        times = data['times']
        num_not_achieved = data['num_not_achieved']
        sim_end = data['sim_end']

        times_array = np.array(times)
        total_runs = len(times) + num_not_achieved

        # Calculate percentiles including not achieved as sim_end
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate([times_array, np.full(num_not_achieved, sim_end)])
        else:
            combined_data = times_array

        q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

        # Calculate mode as the bin with highest count
        hist, bin_edges = np.histogram(times_array, bins=bins)
        mode_bin_idx = np.argmax(hist)
        mode = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2

        # Calculate probability density (normalize by total runs and bin width)
        # This gives probability per year, scaled to percentage
        prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0
        density = hist / (total_runs * bin_width) * 100  # % per year

        # Plot as step histogram
        color = colors[idx % len(colors)]
        clean_name = clean_milestone_name(milestone_name)

        # Use bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.step(bin_centers, density, where='mid', linewidth=2.5,
                label=clean_name, color=color)

        # Calculate percentage after 2050
        num_after_2050 = sum(1 for t in times_array if t > 2050) + num_not_achieved
        pct_after_2050 = (num_after_2050 / total_runs * 100) if total_runs > 0 else 0

        stats_lines.append(
            f"{clean_name}: Mode={mode:.1f}, P10={q10:.1f}, P50={q50:.1f}, P90={q90:.1f}, {pct_after_2050:.0f}% > 2050"
        )

    # Count total trajectories
    if milestone_data:
        total_trajectories = len(milestone_data[0]['times']) + milestone_data[0]['num_not_achieved']
        plt.xlabel("Arrival Time (decimal year)", fontsize=12)
        plt.ylabel("Probability Density (% per year)", fontsize=12)
        plt.title(title or f"Milestone Arrival Time Distributions\n(n={total_trajectories} trajectories)", fontsize=14)
    else:
        plt.xlabel("Arrival Time (decimal year)", fontsize=12)
        plt.ylabel("Probability Density (% per year)", fontsize=12)
        plt.title(title or "Milestone Arrival Time Distributions", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)
    plt.xlim(right=max_year)

    # Add statistics text in top right
    if stats_lines:
        stats_text = "\n".join(stats_lines)
        plt.text(0.98, 0.98, stats_text,
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
                 family='monospace')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved overlay histogram plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate histogram plots for milestone arrival times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        required=True,
        help="Path to rollouts.jsonl file"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Output file path (default: <rollouts_dir>/milestone_histograms_overlay.png)"
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Width of histogram bins in years (default: 1.0)"
    )
    parser.add_argument(
        "--max-year",
        type=float,
        default=2050,
        help="Maximum year to plot (default: 2050)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # Determine output path
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = rollouts_path.parent / "milestone_histograms_overlay.png"

    # Fixed list of milestones (will be filtered to AC, SC, SAR, SIAR in the function)
    milestone_names = [
        'AC',
        'AI2027-SC',
        'SAR-level-experiment-selection-skill',
        'SIAR-level-experiment-selection-skill'
    ]

    # Generate plot
    plot_milestone_histograms_overlay(
        rollouts_path,
        milestone_names,
        out_path,
        title=args.title,
        bin_width=args.bin_width,
        max_year=args.max_year,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
