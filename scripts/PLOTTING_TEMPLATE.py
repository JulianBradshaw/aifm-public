#!/usr/bin/env python3
"""
[TEMPLATE] Template for creating new plotting scripts following project conventions.

DO NOT USE THIS TEMPLATE DIRECTLY. Copy and modify it for your specific plotting needs.

This template demonstrates the correct way to:
1. Load data using RolloutsReader
2. Use high-level plotting functions from scripts/plotting/ when available
3. Use KDE utilities for custom density estimation (when needed)
4. Follow project conventions

Usage:
  python scripts/your_new_script.py --run-dir outputs/your_run/
  python scripts/your_new_script.py --rollouts outputs/your_run/rollouts.jsonl
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup path for imports (required for scripts to find plotting_utils)
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# REQUIRED: Import from plotting utilities, NOT custom implementations
from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.helpers import load_present_day

# Apply website-consistent styling (off-white background, monospace fonts)
# This sets figure/axes background to #fffff8 to match the website
from plotting_utils.style import apply_style
apply_style()

# Import KDE utilities ONLY if you need custom density estimation
# Choose the right KDE for your use case:
#
# 1. make_gamma_kernel_kde() - RECOMMENDED for milestone arrival times
#    - Gamma kernel KDE that properly handles lower bounds
#    - Density tapers to zero exactly at the boundary (e.g., present_day)
#    - Cross-validated bandwidth selection
#    - Example: from plotting_utils.kde import make_gamma_kernel_kde
#
# 2. make_gaussian_kde() - SIMPLEST, use for general-purpose distributions
#    - Standard Gaussian KDE with Scott's rule bandwidth
#    - Good for unbounded data or when boundary effects are not critical
#    - Example: from plotting_utils.kde import make_gaussian_kde
#
# 3. make_lower_bounded_kde() - Advanced log-space alternative
#    - Uses log transformation with Gaussian KDE
#    - Cross-validated bandwidth selection in log space
#    - Example: from plotting_utils.kde import make_lower_bounded_kde

# RECOMMENDED: Check if your plot type already exists in scripts/plotting/
# Available high-level plotting functions (import as needed):
#
# from plotting.histograms import (
#     plot_milestone_time_histogram,          # Standard histogram with KDE overlay
#     plot_milestone_time_histogram_cdf,      # CDF-style histogram
#     plot_milestone_effective_compute_histogram,  # Compute at milestone
#     plot_horizon_at_sc_histogram,           # Horizon values at SC time
#     plot_aa_time_histogram,                 # SC/AA arrival times
#     plot_aa_time_histogram_cdf,             # SC/AA CDF histogram
# )
# from plotting.trajectories import (
#     plot_horizon_trajectories,              # Horizon length over time
#     plot_uplift_trajectories,               # Progress uplift trajectories
#     plot_metric_trajectories,               # Generic metric trajectories
# )
# from plotting.boxplots import (
#     plot_milestone_transition_boxplot,      # Boxplot of transition durations
# )
# from plotting.scatter import (
#     plot_milestone_scatter,                 # Scatter plot of two milestones
#     plot_correlation_scatter,               # Parameter correlation scatter
# )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Description of what your script does",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--rollouts",
        type=str,
        help="Path to rollouts.jsonl file",
    )
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (expects rollouts.jsonl inside)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for plot "
            "(defaults to run-dir/descriptive_name.png)"
        ),
    )
    return parser.parse_args()


def load_data(rollouts_path: Path):
    """
    Load data using RolloutsReader utilities.

    DO NOT write custom JSON parsing loops. Use the provided methods.

    This simple example loads just milestone times. For more complex plots,
    you can return a dictionary with multiple data types - just make sure
    your plotting functions use the returned data consistently.
    """
    reader = RolloutsReader(rollouts_path)

    # Example 1: Load single milestone times
    # Returns: (times, num_not_achieved, typical_sim_end)
    milestone_times, num_not_achieved, _sim_end = reader.read_milestone_times("AC")

    # Print summary statistics
    total_rollouts = reader.count_rollouts()
    achieved_pct = (len(milestone_times) / total_rollouts * 100) if total_rollouts > 0 else 0
    print(f"\nTotal rollouts: {total_rollouts}")
    print(f"  AC achieved: {len(milestone_times)} ({achieved_pct:.1f}%)")
    print(f"  AC not achieved: {num_not_achieved}")

    # Example 2: Load multiple milestones efficiently (single pass through file)
    # milestone_names = ["AC", "AI2027-SC", "SAR-level-experiment-selection-skill"]
    # times_map, not_achieved_map, sim_end, total = reader.read_milestone_times_batch(
    #     milestone_names
    # )
    # # times_map is dict: {"AC": [list of times], "AI2027-SC": [...], ...}
    # # not_achieved_map is dict: {"AC": count, "AI2027-SC": count, ...}

    # Example 3: Load transition durations between two milestones
    # Returns: (durations, num_censored, typical_sim_end)
    # transition_durations, num_censored, sim_end = reader.read_transition_durations(
    #     "AC", "SAR-level-experiment-selection-skill"
    # )

    # Example 4: Load time-series trajectory data
    # Returns: (times_array, list_of_trajectory_arrays, list_of_aa_times)
    # times_array, progress_trajectories, aa_times = reader.read_trajectories("cumulative_progress")
    # times_array, automation_trajectories, aa_times = reader.read_trajectories("automation_fraction")

    # Example 5: Load metric trajectories with filtering and MSE values
    # times, trajectories, aa_times, mse_values = reader.read_metric_trajectories(
    #     "horizon_lengths",
    #     include_aa_times=True,
    #     include_mse=True,
    #     filter_milestone="AC",  # Optional: filter by milestone
    #     filter_year=2030.0,     # Optional: year threshold for filter
    # )

    # Example 6: Load parameter values from rollouts
    # parameter_values = []
    # for rollout in reader.iter_rollouts():
    #     params = rollout.get("parameters", {})
    #     param_val = params.get("your_parameter_name")
    #     if param_val is not None:
    #         parameter_values.append(float(param_val))
    # parameter_values = np.array(parameter_values)

    # Example 7: Load scatter plot data for two milestones
    # times1, times2 = reader.read_milestone_scatter_data("AC", "AI2027-SC")

    # For more complex analyses, return a dictionary:
    # return {
    #     "milestone_times": milestone_times,
    #     "num_not_achieved": num_not_achieved,
    #     "sim_end": sim_end,
    #     "transition_durations": transition_durations,
    #     "parameter_values": parameter_values,
    # }

    # Convert to numpy array for plotting
    return np.array(milestone_times), num_not_achieved


def create_custom_density_plot(
    milestone_times: np.ndarray,
    num_not_achieved: int,
    milestone_name: str,
    present_day: float,
    output_path: Path,
):
    """
    Example: Create a custom PDF plot using KDE utilities.

    IMPORTANT: Before writing custom plotting code, check if scripts/plotting/
    already has a function for your plot type! For example:
    - plot_milestone_time_histogram() already exists for milestone PDFs
    - plot_horizon_trajectories() exists for trajectory plots
    - See scripts/plotting/histograms.py, trajectories.py, etc.

    KDE Selection Guide:
    - For milestone arrival times: Use make_gamma_kernel_kde() (RECOMMENDED)
    - For general distributions: Use make_gaussian_kde() (simplest)
    - For advanced log-space KDE: Use make_lower_bounded_kde()

    DO NOT import scipy.stats.gaussian_kde directly for 1D distributions.
    """
    # Import KDE function (uncomment the one you need)
    from plotting_utils.kde import make_gamma_kernel_kde  # Recommended for milestone times
    # from plotting_utils.kde import make_gaussian_kde  # Simple alternative
    # from plotting_utils.kde import make_lower_bounded_kde  # Advanced alternative

    # Filter valid times
    valid_times = milestone_times[np.isfinite(milestone_times)]

    if len(valid_times) < 2:
        print(f"Not enough valid samples for KDE (found {len(valid_times)} samples)")
        return

    # Option 1: Gamma kernel KDE with proper boundary handling (RECOMMENDED for milestone times)
    kde = make_gamma_kernel_kde(valid_times, lower_bound=present_day)

    # Option 2: Simple Gaussian KDE (use for general distributions without boundaries)
    # kde = make_gaussian_kde(valid_times)

    # Option 3: Log-space KDE with cross-validation (advanced alternative for bounded data)
    # kde = make_lower_bounded_kde(valid_times, lower_bound=present_day)

    # Generate evaluation points
    x_min = valid_times.min()
    x_max = valid_times.max()
    x_range = x_max - x_min
    x_vals = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)

    # Evaluate density
    pdf_vals = kde.pdf(x_vals)

    # Create plot
    _fig, ax = plt.subplots(figsize=(12, 7))

    ax.fill_between(x_vals, pdf_vals, alpha=0.3, color='tab:blue')
    ax.plot(x_vals, pdf_vals, 'b-', linewidth=2)

    # Add statistics
    median_time = np.median(valid_times)
    mean_time = np.mean(valid_times)
    ax.axvline(median_time, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_time:.1f}')
    ax.axvline(mean_time, color='orange', linestyle=':', linewidth=2,
               label=f'Mean: {mean_time:.1f}')

    # Add title with achievement statistics
    total_samples = len(valid_times) + num_not_achieved
    achieved_pct = (len(valid_times) / total_samples * 100) if total_samples > 0 else 0
    title = f'{milestone_name} Arrival Time Distribution\n'
    title += f'({len(valid_times)}/{total_samples} rollouts achieved, {achieved_pct:.1f}%)'

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Determine paths
    if args.rollouts:
        rollouts_path = Path(args.rollouts)
        run_dir = rollouts_path.parent
    else:
        run_dir = Path(args.run_dir)
        rollouts_path = run_dir / "rollouts.jsonl"

    if not rollouts_path.exists():
        print(f"Error: {rollouts_path} not found")
        return

    # Load present_day from run directory
    present_day = load_present_day(run_dir)
    print(f"Present day: {present_day}")

    # Determine output path with descriptive name
    if args.output:
        output_path = Path(args.output)
    else:
        # Use descriptive filename that reflects what the plot shows
        output_path = run_dir / "milestone_arrival_density.png"

    # Load data using utilities
    milestone_times, num_not_achieved = load_data(rollouts_path)

    # Create your plots
    # Replace "AC" with your actual milestone name
    create_custom_density_plot(
        milestone_times,
        num_not_achieved=num_not_achieved,
        milestone_name="AC",
        present_day=present_day,
        output_path=output_path,
    )

    print("Done!")


if __name__ == "__main__":
    main()
