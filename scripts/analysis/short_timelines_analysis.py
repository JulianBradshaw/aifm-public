#!/usr/bin/env python3
"""
Generate short timelines analysis outputs for AC and SAR milestones.

This script creates a focused set of outputs in a short_timelines_outputs folder:
(a) Table with AC and SAR chance of arriving by 2027, 2030, 2035
(b) Filtered milestone PDF overlay containing only AC and SAR distributions
(c) PDF of the distribution from AC->SAR (including censored)

Usage:
  python scripts/short_timelines_analysis.py --run-dir outputs/20250813_020347
  python scripts/short_timelines_analysis.py --rollouts outputs/20250813_020347/rollouts.jsonl
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import RolloutsReader for efficient data loading (auto-uses cache when available)
# Add scripts/ directory to path for imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

from plotting_utils.rollouts_reader import RolloutsReader

# Import plotting functions and KDE utilities
from plots.pdfs.milestone_overlay import MilestonePDFOverlayPlot
from plotting_utils.kde import make_gamma_kernel_kde


def read_milestone_data(
    rollouts_file: Path,
    milestone_names: List[str]
) -> Tuple[Dict[str, List[float]], Dict[str, int], Optional[float]]:
    """Read milestone arrival times from rollouts.

    Uses RolloutsReader which automatically uses cache when available (100-1000x faster).

    Returns:
        times_dict: dict mapping milestone name to list of arrival times (only achieved)
        num_not_achieved_dict: dict mapping milestone name to count of non-achievements
        typical_sim_end: typical simulation end time (median), or None
    """
    reader = RolloutsReader(rollouts_file)
    times_dict, num_not_achieved_dict, typical_sim_end, _ = reader.read_milestone_times_batch(milestone_names)
    return times_dict, num_not_achieved_dict, typical_sim_end


def read_transition_data(
    rollouts_file: Path,
    from_milestone: str,
    to_milestone: str,
    include_censored: bool = True
) -> Tuple[List[float], Optional[float]]:
    """Read transition durations from one milestone to another.

    Uses RolloutsReader which automatically uses cache when available (100-1000x faster).

    Args:
        rollouts_file: path to rollouts.jsonl
        from_milestone: starting milestone name
        to_milestone: ending milestone name
        include_censored: if True, include transitions censored at sim end

    Returns:
        durations: list of transition durations (excludes cases where to_milestone comes before from_milestone)
        typical_sim_end: typical simulation end time (median), or None
    """
    reader = RolloutsReader(rollouts_file)
    durations, _, typical_sim_end, _ = reader.read_transition_durations(
        from_milestone,
        to_milestone,
        include_censored=include_censored
    )
    return durations, typical_sim_end


def generate_arrival_probability_table(
    times_dict: Dict[str, List[float]],
    num_not_achieved_dict: Dict[str, int],
    milestone_names: List[str],
    target_years: List[int],
    output_path: Path
) -> None:
    """Generate table showing probability of achieving milestones by end of target years.

    Args:
        times_dict: dict mapping milestone name to list of arrival times
        num_not_achieved_dict: dict mapping milestone name to count of non-achievements
        milestone_names: list of milestone names
        target_years: list of years to compute probabilities for (end of year, e.g., Dec 31)
        output_path: path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean milestone names for display
    def clean_name(name: str) -> str:
        return name.replace("-level-experiment-selection-skill", "")

    # Calculate probabilities
    rows = []
    for name in milestone_names:
        times = times_dict[name]
        num_not_achieved = num_not_achieved_dict[name]
        total_runs = len(times) + num_not_achieved

        row = {"Milestone": clean_name(name)}
        for year in target_years:
            # Count arrivals by end of year (i.e., <= Dec 31 = year + 1.0 - epsilon)
            # Since we use decimal years, end of year is approximately year + 0.9999
            # But to be safe and inclusive, we use <= year + 1.0 - 0.001
            end_of_year = year + 1.0 - 0.001
            num_achieved_by_year = sum(1 for t in times if t < end_of_year)
            prob = 100.0 * num_achieved_by_year / total_runs if total_runs > 0 else 0.0
            # Format with 2 significant figures
            row[f"By Dec {year}"] = f"{prob:.2g}%"

        rows.append(row)

    # Write to CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Milestone"] + [f"By Dec {year}" for year in target_years]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved arrival probability table to: {output_path}")


def plot_transition_duration_pdf(
    durations: List[float],
    from_milestone: str,
    to_milestone: str,
    output_path: Path,
    include_censored: bool = True,
    max_duration: float = 10.0
) -> None:
    """Plot PDF of transition duration from one milestone to another.

    Args:
        durations: list of transition durations
        from_milestone: starting milestone name
        to_milestone: ending milestone name
        output_path: output file path
        include_censored: whether censored transitions are included
        max_duration: maximum duration to plot (default: 30 years)
    """
    # Clean milestone names for display
    def clean_name(name: str) -> str:
        return name.replace("-level-experiment-selection-skill", "")

    raw_data = np.asarray(durations, dtype=float)

    # Filter out negative durations (shouldn't happen, but enforce boundary at 0)
    present_day = 0.0
    boundary = float(present_day)
    valid_mask = raw_data >= boundary
    data = raw_data[valid_mask]
    dropped = raw_data.size - data.size
    if dropped:
        print(f"  Dropped {dropped} samples before lower bound={boundary:.3f}.")

    if data.size < 2:
        print(f"Warning: Not enough transition data to create KDE (need at least 2 points, got {data.size})")
        return

    # Create KDE
    try:
        kde = make_gamma_kernel_kde(data, lower_bound=present_day)
        bw = float(kde.bandwidth)

        # Compute evaluation range respecting lower bound
        min_eval = max(present_day, float(np.min(data) - 5.0 * bw))
        max_eval = max(float(np.max(data) + 5.0 * bw), min_eval + bw)

        # Use 512 evenly-spaced points
        xs = np.linspace(min_eval, max_eval, 512)
        pdf_values = kde(xs) * 100  # Convert to percentage per year

        # Cut off plotting at max_duration
        plot_mask = xs <= max_duration
        xs_plot = xs[plot_mask]
        pdf_plot = pdf_values[plot_mask]
    except Exception as e:
        print(f"Warning: Could not create KDE for transition: {e}")
        return

    # Calculate percentiles
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])

    # Find mode (peak of KDE)
    mode_idx = np.argmax(pdf_values)
    mode = xs[mode_idx]

    # Create plot
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Plot PDF
    plt.plot(xs_plot, pdf_plot, linewidth=2.5, color='tab:purple', label='PDF')

    # Add percentile lines (no labels in legend since they're in the stats box)
    plt.axvline(q10, color='tab:gray', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axvline(q50, color='tab:green', linestyle='-', linewidth=2, alpha=0.7)
    plt.axvline(q90, color='tab:gray', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axvline(mode, color='tab:red', linestyle=':', linewidth=1.5, alpha=0.7)

    # Add statistics text box
    stats_text = (
        f"N = {len(durations)} trajectories\n"
        f"Mode: {mode:.1f} years\n"
        f"P10: {q10:.1f} years\n"
        f"P50: {q50:.1f} years\n"
        f"P90: {q90:.1f} years"
    )
    if include_censored:
        stats_text += "\n\n(includes censored)"

    plt.text(0.98, 0.98, stats_text,
             transform=ax.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
             family='monospace')

    from_clean = clean_name(from_milestone)
    to_clean = clean_name(to_milestone)
    censored_suffix = " (including censored)" if include_censored else ""
    plt.xlabel("Transition Duration (years)", fontsize=12)
    plt.ylabel("Probability Density (% per year)", fontsize=12)
    plt.title(f"Transition Duration: {from_clean} → {to_clean}{censored_suffix}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0, right=max_duration)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved transition duration PDF plot to: {output_path}")


def generate_short_timelines_analysis(
    rollouts_file: Path,
    output_dir: Path
) -> None:
    """Generate all short timelines analysis outputs.

    Args:
        rollouts_file: path to rollouts.jsonl
        output_dir: base directory for outputs (will create short_timelines_outputs subfolder)
    """
    # Create output directory
    short_timelines_dir = output_dir / "short_timelines_outputs"
    short_timelines_dir.mkdir(parents=True, exist_ok=True)

    # Define milestones
    milestone_names = ["AC", "SAR-level-experiment-selection-skill"]

    # Read milestone data
    print("Reading milestone data...")
    times_dict, num_not_achieved_dict, sim_end = read_milestone_data(rollouts_file, milestone_names)

    # (a) Generate arrival probability table
    print("\nGenerating arrival probability table...")
    target_years = [2027, 2030, 2035]
    table_path = short_timelines_dir / "arrival_probabilities.csv"
    generate_arrival_probability_table(
        times_dict,
        num_not_achieved_dict,
        milestone_names,
        target_years,
        table_path
    )

    # (b) Generate filtered milestone PDF overlay using new class-based API
    print("\nGenerating AC and SAR PDF overlay...")
    overlay_path = short_timelines_dir / "ac_sar_pdfs_overlay.png"
    overlay_plot = MilestonePDFOverlayPlot(output_dir)
    overlay_plot.generate(
        preset="AC_SAR",
        out_path=overlay_path,
        title="AC and SAR Arrival Time Distributions",
    )

    # (c) Generate AC->SAR transition duration PDF
    print("\nGenerating AC->SAR transition duration PDF...")
    durations, _ = read_transition_data(
        rollouts_file,
        "AC",
        "SAR-level-experiment-selection-skill",
        include_censored=True
    )
    transition_path = short_timelines_dir / "ac_to_sar_duration_pdf.png"
    plot_transition_duration_pdf(
        durations,
        "AC",
        "SAR-level-experiment-selection-skill",
        transition_path,
        include_censored=True,
        max_duration=10.0
    )

    print(f"\n✓ Short timelines analysis complete! All outputs saved to: {short_timelines_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate short timelines analysis for AC and SAR milestones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (will use rollouts.jsonl inside)"
    )
    group.add_argument(
        "--rollouts",
        type=str,
        help="Path to rollouts.jsonl file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine rollouts file and output directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        rollouts_file = run_dir / "rollouts.jsonl"
        output_dir = run_dir
    else:
        rollouts_file = Path(args.rollouts)
        output_dir = rollouts_file.parent

    if not rollouts_file.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    # Generate all outputs
    generate_short_timelines_analysis(rollouts_file, output_dir)


if __name__ == "__main__":
    main()
