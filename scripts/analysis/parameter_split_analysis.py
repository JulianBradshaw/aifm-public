#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for parameter splits.

WARNING: This script contains bespoke data loading and analysis code that doesn't
follow the project's plotting utility conventions. It should be refactored to use
utilities from scripts/plotting/ and scripts/plotting_utils/.

For a given parameter split (above/below median), this script generates:
1. PDF overlay plots comparing AC and SAR distributions between splits
2. Timeline probability plots: P(milestone by date) vs parameter value
3. Takeoff speed plots: P(one-year takeoff) vs parameter value
4. "As fast as AI 2027" takeoff plots vs parameter value

Usage:
  # Analyze a specific parameter split directory
  python scripts/parameter_split_analysis.py \
    --split-dir outputs/251109_eli_2200/parameter_splits/ai-research-taste-at-coding-automation-anchor-sd_split

  # Analyze all splits in a run directory
  python scripts/parameter_split_analysis.py \
    --run-dir outputs/251109_eli_2200
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()


# Parameters that should be plotted on log scale (lognormal or shifted_lognormal distributions)
LOG_SCALE_PARAMETERS = {
    "ac_time_horizon_minutes",
    "gap_years",
    "present_doubling_time",
    "inf_compute_asymptote",
    "inf_labor_asymptote",
    "inv_compute_anchor_exp_cap",
    "swe_multiplier_at_present_day",
    "coding_automation_efficiency_slope",
    "max_serial_coding_labor_multiplier",
    "ai_research_taste_slope",
    "median_to_top_taste_multiplier",
}


def should_use_log_scale(parameter_name: str) -> bool:
    """Determine if a parameter should be plotted on log scale."""
    return parameter_name in LOG_SCALE_PARAMETERS


def read_rollouts_data(
    rollouts_file: Path,
    parameter_name: str
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read rollouts and extract parameter values and milestone times.

    Args:
        rollouts_file: Path to rollouts.jsonl
        parameter_name: Name of the parameter being analyzed

    Returns:
        Tuple of (parameter_values, milestone_times_dict)
        milestone_times_dict maps milestone name to array of times (NaN if not achieved)
    """
    param_values = []
    milestone_data = {}

    # Milestones to track
    milestones = ["AC", "SAR-level-experiment-selection-skill", "TED-AI", "ASI"]
    for ms in milestones:
        milestone_data[ms] = []

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                # Get parameter value
                param_val = all_params.get(parameter_name)
                if param_val is None or not isinstance(param_val, (int, float)) or not np.isfinite(param_val):
                    continue

                param_values.append(float(param_val))

                # Get milestone times
                ms_dict = results.get("milestones", {})
                for ms in milestones:
                    ms_info = ms_dict.get(ms)
                    if ms_info and isinstance(ms_info, dict):
                        time = ms_info.get("time")
                        if time is not None and np.isfinite(time):
                            milestone_data[ms].append(float(time))
                        else:
                            milestone_data[ms].append(np.nan)
                    else:
                        milestone_data[ms].append(np.nan)

            except Exception:
                continue

    # Convert to numpy arrays
    param_values = np.array(param_values)
    for ms in milestones:
        milestone_data[ms] = np.array(milestone_data[ms])

    return param_values, milestone_data


def plot_milestone_pdf_comparison(
    above_rollouts: Path,
    below_rollouts: Path,
    milestone_name: str,
    output_path: Path,
    parameter_display_name: str
) -> None:
    """Create histogram comparing a milestone's distribution for above vs below median.

    Args:
        above_rollouts: Path to above median rollouts
        below_rollouts: Path to below median rollouts
        milestone_name: Name of milestone to plot
        output_path: Output file path
        parameter_display_name: Display name for parameter
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {'above': 'tab:orange', 'below': 'tab:blue'}
    labels = {'above': f'Above median {parameter_display_name}',
              'below': f'Below median {parameter_display_name}'}

    # Collect all times to determine bin edges
    all_times = []
    split_times = {}

    for split_type, rollouts_file in [('below', below_rollouts), ('above', above_rollouts)]:
        times = []
        with rollouts_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    results = rec.get("results", {})
                    ms_dict = results.get("milestones", {})
                    ms_info = ms_dict.get(milestone_name)
                    if ms_info and isinstance(ms_info, dict):
                        time = ms_info.get("time")
                        if time is not None and np.isfinite(time):
                            times.append(float(time))
                except Exception:
                    continue

        if len(times) < 2:
            print(f"Warning: Not enough data for {milestone_name} {split_type} median")
            continue

        split_times[split_type] = np.array(times)
        all_times.extend(times)

    if len(all_times) < 2:
        print(f"Warning: Not enough total data for {milestone_name}")
        return

    # Create year bins (1-year bins from floor to ceiling, capped at 2050)
    all_times = np.array(all_times)
    year_min = int(np.floor(all_times.min()))
    year_max = min(2050, int(np.ceil(all_times.max())))
    year_bins = np.arange(year_min, year_max + 1, 1)  # 1-year bins
    year_centers = year_bins[:-1] + 0.5

    # Create histograms for each split
    for split_type in ['below', 'above']:
        if split_type not in split_times:
            continue

        times = split_times[split_type]
        color = colors[split_type]

        # Calculate histogram (counts per year)
        counts, _ = np.histogram(times, bins=year_bins)

        # Convert to percentage per year
        total = len(times)
        percentages = (counts / total) * 100

        # Plot as connected points
        ax.plot(year_centers, percentages, 'o-', linewidth=2, markersize=6,
               color=color, label=labels[split_type], alpha=0.8)

        # Add median line
        median = float(np.median(times))
        ax.axvline(median, color=color, linestyle='--', linewidth=1.5, alpha=0.5,
                  label=f'{labels[split_type]} median: {median:.1f}')

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Probability (% per year)", fontsize=12)
    ax.set_xlim(right=2050)  # Cap x-axis at 2050

    # Use display name mapping
    milestone_display = {
        "AC": "AC",
        "SAR-level-experiment-selection-skill": "SAR",
        "TED-AI": "TED-AI",
        "ASI": "ASI"
    }
    title = f"{milestone_display.get(milestone_name, milestone_name)} Arrival Time: Above vs Below Median {parameter_display_name}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_timeline_probabilities(
    rollouts_file: Path,
    parameter_name: str,
    milestone_name: str,
    target_years: List[int],
    output_path: Path,
    parameter_display_name: str,
    log_scale_x: bool = False
) -> None:
    """Plot P(milestone by year) vs parameter value.

    Args:
        rollouts_file: Path to combined rollouts file
        parameter_name: Name of parameter for x-axis
        milestone_name: Milestone to analyze
        target_years: List of target years (e.g., [2027, 2030, 2035])
        output_path: Output file path
        parameter_display_name: Display name for parameter
        log_scale_x: Whether to use log scale for x-axis
    """
    param_values, milestone_data = read_rollouts_data(rollouts_file, parameter_name)

    if len(param_values) < 20:
        print(f"Insufficient data for {milestone_name} timeline plot")
        return

    times = milestone_data[milestone_name]

    # Create bins for parameter values
    n_bins = 20
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(param_values, percentiles)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(target_years)))

    for target_year, color in zip(target_years, colors):
        bin_centers = []
        bin_probs = []
        bin_stds = []

        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                mask = (param_values >= bin_edges[i]) & (param_values <= bin_edges[i + 1])
            else:
                mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i + 1])

            if mask.sum() > 0:
                bin_center = float(np.mean(param_values[mask]))
                bin_times = times[mask]

                # Calculate P(achieved by target_year)
                achieved = np.isfinite(bin_times) & (bin_times <= target_year)
                prob = achieved.sum() / mask.sum() * 100

                # Standard error
                p = prob / 100
                se = np.sqrt(p * (1 - p) / mask.sum()) * 100 if mask.sum() > 0 else 0

                bin_centers.append(bin_center)
                bin_probs.append(prob)
                bin_stds.append(se)

        if len(bin_centers) > 2:
            bin_centers = np.array(bin_centers)
            bin_probs = np.array(bin_probs)
            bin_stds = np.array(bin_stds)

            ax.errorbar(bin_centers, bin_probs, yerr=bin_stds,
                       fmt='o-', linewidth=2, markersize=6, capsize=4,
                       color=color, label=f'By Dec {target_year}', alpha=0.8)

    ax.set_xlabel(parameter_display_name, fontsize=12)
    ax.set_ylabel("Probability (%)", fontsize=12)

    milestone_display = {
        "AC": "AC",
        "SAR-level-experiment-selection-skill": "SAR",
        "TED-AI": "TED-AI",
        "ASI": "ASI"
    }
    title = f"P({milestone_display.get(milestone_name, milestone_name)} by Date) vs {parameter_display_name}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(-5, 105)

    if log_scale_x:
        ax.set_xscale('log')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_one_year_takeoff_probabilities(
    rollouts_file: Path,
    parameter_name: str,
    from_milestone: str,
    to_milestones: List[str],
    output_path: Path,
    parameter_display_name: str,
    log_scale_x: bool = False
) -> None:
    """Plot P(≤1 year takeoff) vs parameter value.

    Args:
        rollouts_file: Path to combined rollouts file
        parameter_name: Name of parameter for x-axis
        from_milestone: Starting milestone (e.g., "SAR-level-experiment-selection-skill")
        to_milestones: List of ending milestones (e.g., ["TED-AI", "ASI"])
        output_path: Output file path
        parameter_display_name: Display name for parameter
        log_scale_x: Whether to use log scale for x-axis
    """
    # Read data
    param_values = []
    transitions = {ms: [] for ms in to_milestones}

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                param_val = all_params.get(parameter_name)
                if param_val is None or not isinstance(param_val, (int, float)) or not np.isfinite(param_val):
                    continue

                ms_dict = results.get("milestones", {})
                from_info = ms_dict.get(from_milestone)
                if not from_info or not isinstance(from_info, dict):
                    continue

                from_time = from_info.get("time")
                if from_time is None or not np.isfinite(from_time):
                    continue

                param_values.append(float(param_val))

                for to_ms in to_milestones:
                    to_info = ms_dict.get(to_ms)
                    if to_info and isinstance(to_info, dict):
                        to_time = to_info.get("time")
                        if to_time is not None and np.isfinite(to_time):
                            duration = float(to_time) - float(from_time)
                            transitions[to_ms].append(1.0 if duration <= 1.0 else 0.0)
                        else:
                            transitions[to_ms].append(np.nan)
                    else:
                        transitions[to_ms].append(np.nan)

            except Exception:
                continue

    if len(param_values) < 20:
        print(f"Insufficient data for {from_milestone} takeoff plot")
        return

    param_values = np.array(param_values)

    # Create bins
    n_bins = 20
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(param_values, percentiles)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {'TED-AI': 'tab:blue', 'ASI': 'tab:orange'}
    milestone_display = {
        "SAR-level-experiment-selection-skill": "SAR",
        "AC": "AC",
        "TED-AI": "TED-AI",
        "ASI": "ASI"
    }

    for to_ms in to_milestones:
        outcomes = np.array(transitions[to_ms])

        bin_centers = []
        bin_probs = []
        bin_stds = []

        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                mask = (param_values >= bin_edges[i]) & (param_values <= bin_edges[i + 1])
            else:
                mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i + 1])

            if mask.sum() > 0:
                bin_center = float(np.mean(param_values[mask]))
                bin_outcomes = outcomes[mask]

                # Only count cases where both milestones were achieved
                valid = np.isfinite(bin_outcomes)
                if valid.sum() > 0:
                    prob = bin_outcomes[valid].mean() * 100
                    p = prob / 100
                    se = np.sqrt(p * (1 - p) / valid.sum()) * 100 if valid.sum() > 0 else 0

                    bin_centers.append(bin_center)
                    bin_probs.append(prob)
                    bin_stds.append(se)

        if len(bin_centers) > 2:
            bin_centers = np.array(bin_centers)
            bin_probs = np.array(bin_probs)
            bin_stds = np.array(bin_stds)

            color = colors.get(to_ms, 'tab:gray')
            from_display = milestone_display.get(from_milestone, from_milestone)
            to_display = milestone_display.get(to_ms, to_ms)

            ax.errorbar(bin_centers, bin_probs, yerr=bin_stds,
                       fmt='o-', linewidth=2, markersize=6, capsize=4,
                       color=color, label=f'{from_display}→{to_display} ≤ 1yr', alpha=0.8)

    ax.set_xlabel(parameter_display_name, fontsize=12)
    ax.set_ylabel("Probability (%)", fontsize=12)

    from_display = milestone_display.get(from_milestone, from_milestone)
    title = f"P(≤ 1 Year Takeoff from {from_display}) vs {parameter_display_name}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(-5, 105)

    if log_scale_x:
        ax.set_xscale('log')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ai2027_speed_takeoff_probabilities(
    rollouts_file: Path,
    parameter_name: str,
    from_milestone: str,
    to_milestones_thresholds: Dict[str, float],
    output_path: Path,
    parameter_display_name: str,
    log_scale_x: bool = False
) -> None:
    """Plot P(as fast as AI 2027 takeoff) vs parameter value.

    Args:
        rollouts_file: Path to combined rollouts file
        parameter_name: Name of parameter for x-axis
        from_milestone: Starting milestone (e.g., "SAR-level-experiment-selection-skill")
        to_milestones_thresholds: Dict mapping milestone to threshold in years
                                   (e.g., {"TED-AI": 3/12, "ASI": 5/12})
        output_path: Output file path
        parameter_display_name: Display name for parameter
        log_scale_x: Whether to use log scale for x-axis
    """
    # Read data
    param_values = []
    transitions = {ms: [] for ms in to_milestones_thresholds.keys()}

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                param_val = all_params.get(parameter_name)
                if param_val is None or not isinstance(param_val, (int, float)) or not np.isfinite(param_val):
                    continue

                ms_dict = results.get("milestones", {})
                from_info = ms_dict.get(from_milestone)
                if not from_info or not isinstance(from_info, dict):
                    continue

                from_time = from_info.get("time")
                if from_time is None or not np.isfinite(from_time):
                    continue

                param_values.append(float(param_val))

                for to_ms, threshold in to_milestones_thresholds.items():
                    to_info = ms_dict.get(to_ms)
                    if to_info and isinstance(to_info, dict):
                        to_time = to_info.get("time")
                        if to_time is not None and np.isfinite(to_time):
                            duration = float(to_time) - float(from_time)
                            transitions[to_ms].append(1.0 if duration < threshold else 0.0)
                        else:
                            transitions[to_ms].append(np.nan)
                    else:
                        transitions[to_ms].append(np.nan)

            except Exception:
                continue

    if len(param_values) < 20:
        print(f"Insufficient data for {from_milestone} AI2027-speed takeoff plot")
        return

    param_values = np.array(param_values)

    # Create bins
    n_bins = 20
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(param_values, percentiles)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {'TED-AI': 'tab:blue', 'ASI': 'tab:orange', 'SIAR-level-experiment-selection-skill': 'tab:green'}
    milestone_display = {
        "SAR-level-experiment-selection-skill": "SAR",
        "SIAR-level-experiment-selection-skill": "SIAR",
        "AC": "AC",
        "TED-AI": "TED-AI",
        "ASI": "ASI"
    }

    for to_ms, threshold in to_milestones_thresholds.items():
        outcomes = np.array(transitions[to_ms])

        bin_centers = []
        bin_probs = []
        bin_stds = []

        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                mask = (param_values >= bin_edges[i]) & (param_values <= bin_edges[i + 1])
            else:
                mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i + 1])

            if mask.sum() > 0:
                bin_center = float(np.mean(param_values[mask]))
                bin_outcomes = outcomes[mask]

                valid = np.isfinite(bin_outcomes)
                if valid.sum() > 0:
                    prob = bin_outcomes[valid].mean() * 100
                    p = prob / 100
                    se = np.sqrt(p * (1 - p) / valid.sum()) * 100 if valid.sum() > 0 else 0

                    bin_centers.append(bin_center)
                    bin_probs.append(prob)
                    bin_stds.append(se)

        if len(bin_centers) > 2:
            bin_centers = np.array(bin_centers)
            bin_probs = np.array(bin_probs)
            bin_stds = np.array(bin_stds)

            color = colors.get(to_ms, 'tab:gray')
            from_display = milestone_display.get(from_milestone, from_milestone)
            to_display = milestone_display.get(to_ms, to_ms)

            # Format threshold label
            if threshold < 1:
                threshold_label = f"< {int(threshold * 12)}mo"
            else:
                threshold_label = f"< {threshold:.1f}yr"

            ax.errorbar(bin_centers, bin_probs, yerr=bin_stds,
                       fmt='o-', linewidth=2, markersize=6, capsize=4,
                       color=color, label=f'{from_display}→{to_display} {threshold_label}', alpha=0.8)

    ax.set_xlabel(parameter_display_name, fontsize=12)
    ax.set_ylabel("Probability (%)", fontsize=12)

    from_display = milestone_display.get(from_milestone, from_milestone)
    title = f"P(AI 2027 Speed Takeoff from {from_display}) vs {parameter_display_name}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(-5, 105)

    if log_scale_x:
        ax.set_xscale('log')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def analyze_parameter_split(split_dir: Path, parameter_name: str, log_scale_x: Optional[bool] = None) -> None:
    """Generate all analysis plots for a parameter split directory.

    Args:
        split_dir: Path to parameter split directory (contains rollouts_above/below_median.jsonl)
        parameter_name: Name of the parameter being split
        log_scale_x: Whether to use log scale for x-axis. If None, auto-detect based on parameter name.
    """
    # Auto-detect log scale if not specified
    if log_scale_x is None:
        log_scale_x = should_use_log_scale(parameter_name)

    print(f"\n{'='*60}")
    print(f"Analyzing parameter split: {parameter_name}")
    print(f"Split directory: {split_dir}")
    print(f"Log scale: {log_scale_x}")
    print(f"{'='*60}\n")

    # Check that required files exist
    above_rollouts = split_dir / "rollouts_above_median.jsonl"
    below_rollouts = split_dir / "rollouts_below_median.jsonl"

    if not above_rollouts.exists() or not below_rollouts.exists():
        print(f"Error: Missing rollouts files in {split_dir}")
        return

    # Create combined rollouts file for continuous plots
    combined_rollouts = split_dir / "rollouts_combined.jsonl"
    if not combined_rollouts.exists():
        print("Creating combined rollouts file...")
        with combined_rollouts.open("w") as f_out:
            for rollouts_file in [above_rollouts, below_rollouts]:
                with rollouts_file.open("r") as f_in:
                    f_out.write(f_in.read())

    # Output directly to split directory (not a subdirectory)
    analysis_dir = split_dir

    # Get parameter display name
    param_display = parameter_name.replace('_', ' ').title()

    # 1. PDF overlay plots for AC and SAR (above vs below median)
    print("1. Generating PDF overlay plots...")
    for milestone in ["AC", "SAR-level-experiment-selection-skill"]:
        milestone_short = "AC" if milestone == "AC" else "SAR"
        output_path = analysis_dir / f"{milestone_short.lower()}_pdf_comparison.png"
        plot_milestone_pdf_comparison(
            above_rollouts, below_rollouts, milestone, output_path, param_display
        )

    # 2. Timeline probability plots
    print("\n2. Generating timeline probability plots...")
    target_years = [2027, 2030, 2035]
    for milestone in ["AC", "SAR-level-experiment-selection-skill"]:
        milestone_short = "AC" if milestone == "AC" else "SAR"
        output_path = analysis_dir / f"{milestone_short.lower()}_timeline_probabilities.png"
        plot_timeline_probabilities(
            combined_rollouts, parameter_name, milestone, target_years,
            output_path, param_display, log_scale_x
        )

    # 3. One-year takeoff probability plots
    print("\n3. Generating one-year takeoff probability plots...")
    for from_ms in ["SAR-level-experiment-selection-skill", "AC"]:
        from_short = "SAR" if from_ms == "SAR-level-experiment-selection-skill" else "AC"
        output_path = analysis_dir / f"{from_short.lower()}_one_year_takeoff_probabilities.png"
        plot_one_year_takeoff_probabilities(
            combined_rollouts, parameter_name, from_ms, ["TED-AI", "ASI"],
            output_path, param_display, log_scale_x
        )

    # 4. AI 2027 speed takeoff probability plots
    print("\n4. Generating AI 2027 speed takeoff probability plots...")

    # SAR-based thresholds (from fast_takeoff_analysis.py)
    sar_thresholds = {
        'SIAR-level-experiment-selection-skill': 4/12,  # 4 months
        'TED-AI': 3/12,  # 3 months
        'ASI': 5/12  # 5 months
    }
    output_path = analysis_dir / f"sar_ai2027_speed_takeoff_probabilities.png"
    plot_ai2027_speed_takeoff_probabilities(
        combined_rollouts, parameter_name, "SAR-level-experiment-selection-skill",
        sar_thresholds, output_path, param_display, log_scale_x
    )

    # AC-based thresholds
    ac_thresholds = {
        'TED-AI': 9/12,  # 9 months
        'ASI': 1.0  # 1 year
    }
    output_path = analysis_dir / f"ac_ai2027_speed_takeoff_probabilities.png"
    plot_ai2027_speed_takeoff_probabilities(
        combined_rollouts, parameter_name, "AC",
        ac_thresholds, output_path, param_display, log_scale_x
    )

    print(f"\n✓ Analysis complete! All plots saved to: {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive analysis plots for parameter splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split-dir",
        type=str,
        help="Path to specific parameter split directory"
    )
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (will analyze all splits in parameter_splits/)"
    )

    parser.add_argument(
        "--parameter-name",
        type=str,
        help="Name of parameter (required if using --split-dir)"
    )
    parser.add_argument(
        "--log-scale-x",
        action="store_true",
        help="Use log scale for x-axis in continuous plots"
    )

    args = parser.parse_args()

    if args.split_dir:
        split_dir = Path(args.split_dir)
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        if not args.parameter_name:
            # Try to infer from directory name
            dir_name = split_dir.name
            if dir_name.endswith("_split"):
                parameter_name = dir_name[:-6].replace('-', '_')
            else:
                raise ValueError("--parameter-name required when using --split-dir")
        else:
            parameter_name = args.parameter_name

        analyze_parameter_split(split_dir, parameter_name, args.log_scale_x)

    else:  # run_dir
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        splits_dir = run_dir / "parameter_splits"
        if not splits_dir.exists():
            print(f"No parameter_splits directory found in {run_dir}")
            return

        # Find all split directories
        split_dirs = [d for d in splits_dir.iterdir() if d.is_dir() and d.name.endswith("_split")]

        if not split_dirs:
            print(f"No parameter split directories found in {splits_dir}")
            return

        print(f"Found {len(split_dirs)} parameter split(s) to analyze")

        for split_dir in split_dirs:
            # Infer parameter name from directory name
            parameter_name = split_dir.name[:-6].replace('-', '_')
            analyze_parameter_split(split_dir, parameter_name, args.log_scale_x)


if __name__ == "__main__":
    main()
