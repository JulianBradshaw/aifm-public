#!/usr/bin/env python3
"""
Generate fast takeoff analysis outputs.

NOTE: CENSORED DATA HANDLING DIFFERS FROM WEBSITE CDF
------------------------------------------------------
This script uses `include_censored=False` when reading transition data, meaning
the denominator only includes rollouts where BOTH milestones were achieved.
For example, P(AC→ASI < 1yr) = (transitions < 1yr) / (transitions completed).

The website empirical CDF (generate_conditional_timing.py) includes censored data
in the denominator: CDF(1yr) = (transitions ≤ 1yr) / (all rollouts where AC achieved).

This means values in this analysis will be HIGHER than the website CDF because
rollouts where ASI was never achieved are excluded from the denominator here.

TODO: Consider aligning with the website's approach for consistency, which is
arguably more correct as it captures "probability of fast takeoff given AC"
rather than "probability of fast takeoff given takeoff eventually happens."
------------------------------------------------------

This script creates outputs in a fast_takeoff_outputs folder:
(a) Plot showing SAR→TED-AI and SAR→ASI transition duration distributions,
    highlighting probability that each is faster than 1 year and 6 months
(b) Table showing conditional probabilities that takeoff is faster than AI 2027,
    conditioned on SAR arrival year (2027, 2030, 2035)
(c) Plots showing how probabilities change as a function of SAR arrival time

Usage:
  python scripts/fast_takeoff_analysis.py --run-dir outputs/20250813_020347
  python scripts/fast_takeoff_analysis.py --rollouts outputs/20250813_020347/rollouts.jsonl
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.special

# Add scripts/ directory to path for imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

# Import KDE utilities and rollouts reader
from plotting_utils.kde import make_gaussian_kde
from plotting_utils.rollouts_reader import RolloutsReader


def plot_transition_duration_pdfs(
    transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_path: Path,
    max_duration: float = 10.0,
    title: str = "Fast Takeoff: Transition Durations",
    run_label: str = ""
) -> None:
    """Plot PDFs for transition durations.

    Highlights probability that each transition is faster than 1 year and 6 months.

    Args:
        transitions_dict: dict mapping transition name to list of (from_time, to_time, duration)
        output_path: output file path
        max_duration: maximum duration to plot (default: 10 years)
        title: plot title (default: "Fast Takeoff: Transition Durations")
        run_label: optional label to add to title (e.g., "no correlation")
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        'SAR→TED-AI': 'tab:blue', 'SAR→ASI': 'tab:orange',
        'AC→TED-AI': 'tab:blue', 'AC→ASI': 'tab:orange',
        'SAR→SIAR': 'tab:green', 'AC→SIAR': 'tab:green'
    }

    stats_lines = []

    for transition_name, transitions in transitions_dict.items():
        if len(transitions) < 2:
            print(f"Warning: Not enough data for {transition_name} to create KDE, skipping")
            continue

        durations = np.array([d for _, _, d in transitions])

        # Calculate probabilities for key thresholds
        p_lt_6mo = (np.sum(durations < 0.5) / len(durations) * 100)
        p_lt_1yr = (np.sum(durations < 1.0) / len(durations) * 100)

        # Calculate percentiles
        q10, q50, q90 = np.quantile(durations, [0.1, 0.5, 0.9])

        # Create KDE
        try:
            kde = make_gaussian_kde(durations)
            bw = kde.factor * durations.std()
            xs = np.linspace(max(0, durations.min() - 2 * bw), min(durations.max() + 2 * bw, max_duration), 512)
            pdf_values = kde(xs) * 100  # Convert to percentage per year

            color = colors.get(transition_name, 'tab:gray')
            ax.plot(xs, pdf_values, linewidth=2.5, label=transition_name, color=color)

            # Find mode (peak of KDE)
            mode_idx = np.argmax(pdf_values)
            mode = xs[mode_idx]

            stats_lines.append(
                f"{transition_name}:\n"
                f"  P(< 6mo) = {p_lt_6mo:.2g}%,  P(< 1yr) = {p_lt_1yr:.2g}%\n"
                f"  Mode = {mode:.2f}yr,  P50 = {q50:.2f}yr"
            )
        except Exception as e:
            print(f"Warning: Could not create KDE for {transition_name}: {e}")
            continue

    # Add vertical lines at 6 months and 1 year
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='6 months')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='1 year')

    ax.set_xlabel("Transition Duration (years)", fontsize=12)
    ax.set_ylabel("Probability Density (% per year)", fontsize=12)
    full_title = f"{title}{' - ' + run_label if run_label else ''}"
    ax.set_title(full_title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(left=0, right=max_duration)

    # Add statistics text box
    if stats_lines:
        stats_text = "\n\n".join(stats_lines)
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
                family='monospace')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved transition duration PDFs to: {output_path}")


def plot_transition_duration_cdfs(
    transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_path: Path,
    max_duration: float = 10.0,
    title: str = "Fast Takeoff: Transition Durations (CDF)",
    run_label: str = ""
) -> None:
    """Plot CDFs for transition durations.

    Args:
        transitions_dict: dict mapping transition name to list of (from_time, to_time, duration)
        output_path: output file path
        max_duration: maximum duration to plot (default: 10 years)
        title: plot title
        run_label: optional label to add to title (e.g., "no correlation")
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        'SAR→TED-AI': 'tab:blue', 'SAR→ASI': 'tab:orange',
        'AC→TED-AI': 'tab:blue', 'AC→ASI': 'tab:orange',
        'SAR→SIAR': 'tab:green', 'AC→SIAR': 'tab:green'
    }

    stats_lines = []

    for transition_name, transitions in transitions_dict.items():
        if len(transitions) < 2:
            print(f"Warning: Not enough data for {transition_name} to create CDF, skipping")
            continue

        durations = np.array([d for _, _, d in transitions])
        durations_sorted = np.sort(durations)

        # Calculate CDF (empirical)
        cdf_values = np.arange(1, len(durations_sorted) + 1) / len(durations_sorted) * 100

        # Filter to max_duration
        mask = durations_sorted <= max_duration
        durations_plot = durations_sorted[mask]
        cdf_plot = cdf_values[mask]

        # Calculate key percentiles
        p_lt_6mo = (np.sum(durations < 0.5) / len(durations) * 100)
        p_lt_1yr = (np.sum(durations < 1.0) / len(durations) * 100)
        q10, q50, q90 = np.quantile(durations, [0.1, 0.5, 0.9])

        color = colors.get(transition_name, 'tab:gray')
        ax.plot(durations_plot, cdf_plot, linewidth=2.5, label=transition_name, color=color)

        stats_lines.append(
            f"{transition_name}:\n"
            f"  P(< 6mo) = {p_lt_6mo:.2g}%,  P(< 1yr) = {p_lt_1yr:.2g}%\n"
            f"  P10 = {q10:.2f}yr,  P50 = {q50:.2f}yr,  P90 = {q90:.2f}yr"
        )

    # Add vertical lines at 6 months and 1 year
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='6 months')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='1 year')

    ax.set_xlabel("Transition Duration (years)", fontsize=12)
    ax.set_ylabel("Cumulative Probability (%)", fontsize=12)
    full_title = f"{title}{' - ' + run_label if run_label else ''}"
    ax.set_title(full_title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(left=0, right=max_duration)
    ax.set_ylim(0, 100)

    # Add statistics text box
    if stats_lines:
        stats_text = "\n\n".join(stats_lines)
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
                family='monospace')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved transition duration CDFs to: {output_path}")


def generate_conditional_probability_table(
    transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_path: Path
) -> None:
    """Generate table showing probabilities conditioned on SAR arrival year.

    Shows probability that takeoff is faster than AI 2027, conditioned on:
    - Unconditional
    - SAR arrives in 2027
    - SAR arrives in 2030
    - SAR arrives in 2035

    Metrics:
    - p(SAR→SIAR) < 4 months
    - p(SAR→TED-AI) < 3 months
    - p(SAR→ASI) < 5 months
    - p(SAR→SIAR) ≤ 1 year
    - p(SAR→TED-AI) ≤ 1 year
    - p(SAR→ASI) ≤ 1 year

    Args:
        transitions_dict: dict with keys 'SAR→SIAR', 'SAR→TED-AI', 'SAR→ASI'
        output_path: path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define thresholds for each transition (in years)
    ai2027_thresholds = {
        'SAR→SIAR': 4/12,  # 4 months
        'SAR→TED-AI': 3/12,  # 3 months
        'SAR→ASI': 5/12  # 5 months
    }

    one_year_thresholds = {
        'SAR→SIAR': 1.0,  # 1 year
        'SAR→TED-AI': 1.0,  # 1 year
        'SAR→ASI': 1.0  # 1 year
    }

    # Define SAR arrival year bins (end of year cutoffs)
    sar_year_bins = {
        'Unconditional': (0, np.inf),
        'SAR in 2027': (2027.0, 2028.0),
        'SAR in 2030': (2030.0, 2031.0),
        'SAR in 2035': (2035.0, 2036.0)
    }

    # Calculate probabilities for each condition and transition
    rows = []

    for condition_name, (year_min, year_max) in sar_year_bins.items():
        row = {'Condition': condition_name}

        # Calculate AI 2027 threshold probabilities
        for transition_name, threshold in ai2027_thresholds.items():
            if transition_name not in transitions_dict:
                row[f'{transition_name} <= {int(threshold*12)}mo'] = 'N/A'
                continue

            transitions = transitions_dict[transition_name]

            # Filter by SAR arrival year
            if condition_name == 'Unconditional':
                filtered = transitions
            else:
                filtered = [(ft, tt, d) for ft, tt, d in transitions
                           if year_min <= ft < year_max]

            if len(filtered) == 0:
                row[f'{transition_name} <= {int(threshold*12)}mo'] = 'N/A (0 samples)'
                continue

            durations = np.array([d for _, _, d in filtered])
            prob = (np.sum(durations <= threshold) / len(durations) * 100)
            row[f'{transition_name} <= {int(threshold*12)}mo'] = f"{prob:.2g}%"

        # Calculate 1-year threshold probabilities
        for transition_name, threshold in one_year_thresholds.items():
            if transition_name not in transitions_dict:
                row[f'{transition_name} <= 1yr'] = 'N/A'
                continue

            transitions = transitions_dict[transition_name]

            # Filter by SAR arrival year
            if condition_name == 'Unconditional':
                filtered = transitions
            else:
                filtered = [(ft, tt, d) for ft, tt, d in transitions
                           if year_min <= ft < year_max]

            if len(filtered) == 0:
                row[f'{transition_name} <= 1yr'] = 'N/A (0 samples)'
                continue

            durations = np.array([d for _, _, d in filtered])
            prob = (np.sum(durations <= threshold) / len(durations) * 100)
            row[f'{transition_name} <= 1yr'] = f"{prob:.2g}%"

        rows.append(row)

    # Write to CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        # Create fieldnames with AI 2027 thresholds first, then 1-year thresholds
        ai2027_fields = [f'{name} <= {int(ai2027_thresholds[name]*12)}mo' for name in ai2027_thresholds.keys()]
        one_year_fields = [f'{name} <= 1yr' for name in one_year_thresholds.keys()]
        fieldnames = ['Condition'] + ai2027_fields + one_year_fields

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved conditional probability table to: {output_path}")


def plot_probability_vs_sar_arrival(
    transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_dir: Path,
    run_label: str = ""
) -> None:
    """Plot how probabilities change as a function of SAR arrival time.

    Creates two plots:
    1. P(faster than AI 2027) vs SAR arrival year for each transition
    2. P(≤ 1 year takeoff) vs SAR arrival year for each transition

    Args:
        transitions_dict: dict with keys 'SAR→SIAR', 'SAR→TED-AI', 'SAR→ASI'
        output_dir: directory for output files
        run_label: optional label to add to titles (e.g., "no correlation")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define thresholds for "faster than AI 2027" (varies by transition)
    ai2027_thresholds = {
        'SAR→SIAR': 4/12,  # 4 months
        'SAR→TED-AI': 3/12,  # 3 months
        'SAR→ASI': 5/12  # 5 months
    }

    # SAR arrival year bins (1-year windows)
    sar_years = np.arange(2025, 2051, 1)

    # Plot 1: P(faster than AI 2027) vs SAR arrival
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    colors = {'SAR→SIAR': 'tab:green', 'SAR→TED-AI': 'tab:blue', 'SAR→ASI': 'tab:orange'}

    for transition_name in ai2027_thresholds.keys():
        if transition_name not in transitions_dict:
            continue

        transitions = transitions_dict[transition_name]
        threshold = ai2027_thresholds[transition_name]

        probs = []
        counts = []

        for year in sar_years:
            # Filter transitions where SAR arrives in this year
            filtered = [(ft, tt, d) for ft, tt, d in transitions
                       if year <= ft < year + 1]

            if len(filtered) > 0:
                durations = np.array([d for _, _, d in filtered])
                prob = np.sum(durations < threshold) / len(durations) * 100
                probs.append(prob)
                counts.append(len(filtered))
            else:
                probs.append(np.nan)
                counts.append(0)

        # Plot with marker size proportional to sample count (scaled more aggressively for visibility)
        marker_sizes = [max(10, min(c / 2, 400)) for c in counts]
        valid_mask = ~np.isnan(probs)

        ax1.scatter(sar_years[valid_mask], np.array(probs)[valid_mask],
                   s=np.array(marker_sizes)[valid_mask], alpha=0.6,
                   color=colors.get(transition_name, 'tab:gray'),
                   label=f'{transition_name} (< {int(threshold*12)}mo)')
        ax1.plot(sar_years[valid_mask], np.array(probs)[valid_mask],
                linewidth=2, color=colors.get(transition_name, 'tab:gray'), alpha=0.5)

    ax1.set_xlabel("SAR Arrival Year", fontsize=12)
    ax1.set_ylabel("Probability (%)", fontsize=12)
    title1 = f"P(AI 2027 Takeoff or Faster) vs SAR Arrival Year{' - ' + run_label if run_label else ''}"
    ax1.set_title(title1, fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    ax1.set_xlim(sar_years[0] - 0.5, sar_years[-1] + 0.5)
    ax1.set_ylim(0, 100)

    output_path1 = output_dir / "prob_ai2027_takeoff_vs_sar_arrival.png"
    plt.tight_layout()
    plt.savefig(output_path1, dpi=150)
    plt.close()
    print(f"Saved P(AI 2027 takeoff or faster) vs SAR arrival plot to: {output_path1}")

    # Plot 2: P(≤ 1 year takeoff) vs SAR arrival
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    for transition_name in transitions_dict.keys():
        transitions = transitions_dict[transition_name]

        probs = []
        counts = []

        for year in sar_years:
            # Filter transitions where SAR arrives in this year
            filtered = [(ft, tt, d) for ft, tt, d in transitions
                       if year <= ft < year + 1]

            if len(filtered) > 0:
                durations = np.array([d for _, _, d in filtered])
                prob = np.sum(durations <= 1.0) / len(durations) * 100
                probs.append(prob)
                counts.append(len(filtered))
            else:
                probs.append(np.nan)
                counts.append(0)

        # Plot with marker size proportional to sample count (scaled more aggressively for visibility)
        marker_sizes = [max(10, min(c / 2, 400)) for c in counts]
        valid_mask = ~np.isnan(probs)

        ax2.scatter(sar_years[valid_mask], np.array(probs)[valid_mask],
                   s=np.array(marker_sizes)[valid_mask], alpha=0.6,
                   color=colors.get(transition_name, 'tab:gray'),
                   label=f'{transition_name} ≤ 1yr')
        ax2.plot(sar_years[valid_mask], np.array(probs)[valid_mask],
                linewidth=2, color=colors.get(transition_name, 'tab:gray'), alpha=0.5)

    ax2.set_xlabel("SAR Arrival Year", fontsize=12)
    ax2.set_ylabel("Probability (%)", fontsize=12)
    title2 = f"P(≤ 1 Year Takeoff) vs SAR Arrival Year{' - ' + run_label if run_label else ''}"
    ax2.set_title(title2, fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=11)
    ax2.set_xlim(sar_years[0] - 0.5, sar_years[-1] + 0.5)
    ax2.set_ylim(0, 100)

    output_path2 = output_dir / "prob_1yr_takeoff_vs_sar_arrival.png"
    plt.tight_layout()
    plt.savefig(output_path2, dpi=150)
    plt.close()
    print(f"Saved P(≤ 1yr takeoff) vs SAR arrival plot to: {output_path2}")


def generate_combined_conditional_probability_table(
    sar_transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    ac_transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_path: Path
) -> None:
    """Generate combined table showing SAR and AC probabilities.

    Combines SAR and AC transitions into one table with proper column ordering:
    - SAR→SIAR <= 4mo, SAR→TED-AI <= 3mo, SAR→ASI <= 5mo
    - AC→TED-AI <= 9mo, AC→ASI <= 1yr
    - SAR→SIAR <= 1yr, SAR→TED-AI <= 1yr, SAR→ASI <= 1yr

    Args:
        sar_transitions_dict: dict with keys 'SAR→SIAR', 'SAR→TED-AI', 'SAR→ASI'
        ac_transitions_dict: dict with keys 'AC→TED-AI', 'AC→ASI'
        output_path: path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define all conditions and thresholds
    sar_ai2027_thresholds = {
        'SAR→SIAR': 4/12,
        'SAR→TED-AI': 3/12,
        'SAR→ASI': 5/12
    }

    ac_thresholds = {
        'AC→TED-AI': 9/12,
        'AC→ASI': 1.0
    }

    sar_one_year_thresholds = {
        'SAR→SIAR': 1.0,
        'SAR→TED-AI': 1.0,
        'SAR→ASI': 1.0
    }

    # Define year bins (using SAR for SAR conditions, AC for AC conditions)
    year_bins = {
        'Unconditional': (0, np.inf),
        'SAR/AC in 2027': (2027.0, 2028.0),
        'SAR/AC in 2030': (2030.0, 2031.0),
        'SAR/AC in 2035': (2035.0, 2036.0)
    }

    rows = []

    for condition_name, (year_min, year_max) in year_bins.items():
        row = {'Condition': condition_name}

        # SAR AI 2027 thresholds
        for transition_name, threshold in sar_ai2027_thresholds.items():
            if transition_name not in sar_transitions_dict:
                row[f'{transition_name} <= {int(threshold*12)}mo'] = 'N/A'
                continue

            transitions = sar_transitions_dict[transition_name]

            if condition_name == 'Unconditional':
                filtered = transitions
            else:
                filtered = [(ft, tt, d) for ft, tt, d in transitions
                           if year_min <= ft < year_max]

            if len(filtered) == 0:
                row[f'{transition_name} <= {int(threshold*12)}mo'] = 'N/A (0 samples)'
                continue

            durations = np.array([d for _, _, d in filtered])
            prob = (np.sum(durations <= threshold) / len(durations) * 100)
            row[f'{transition_name} <= {int(threshold*12)}mo'] = f"{prob:.2g}%"

        # AC thresholds
        for transition_name, threshold in ac_thresholds.items():
            if transition_name not in ac_transitions_dict:
                row[f'{transition_name} <= {int(threshold*12)}mo' if threshold < 1 else f'{transition_name} <= {int(threshold)}yr'] = 'N/A'
                continue

            transitions = ac_transitions_dict[transition_name]

            if condition_name == 'Unconditional':
                filtered = transitions
            else:
                filtered = [(ft, tt, d) for ft, tt, d in transitions
                           if year_min <= ft < year_max]

            if len(filtered) == 0:
                row[f'{transition_name} <= {int(threshold*12)}mo' if threshold < 1 else f'{transition_name} <= {int(threshold)}yr'] = 'N/A (0 samples)'
                continue

            durations = np.array([d for _, _, d in filtered])
            prob = (np.sum(durations <= threshold) / len(durations) * 100)
            row[f'{transition_name} <= {int(threshold*12)}mo' if threshold < 1 else f'{transition_name} <= {int(threshold)}yr'] = f"{prob:.2g}%"

        # SAR 1-year thresholds
        for transition_name, threshold in sar_one_year_thresholds.items():
            if transition_name not in sar_transitions_dict:
                row[f'{transition_name} <= 1yr'] = 'N/A'
                continue

            transitions = sar_transitions_dict[transition_name]

            if condition_name == 'Unconditional':
                filtered = transitions
            else:
                filtered = [(ft, tt, d) for ft, tt, d in transitions
                           if year_min <= ft < year_max]

            if len(filtered) == 0:
                row[f'{transition_name} <= 1yr'] = 'N/A (0 samples)'
                continue

            durations = np.array([d for _, _, d in filtered])
            prob = (np.sum(durations <= threshold) / len(durations) * 100)
            row[f'{transition_name} <= 1yr'] = f"{prob:.2g}%"

        rows.append(row)

    # Write to CSV with proper column ordering
    with output_path.open("w", newline="", encoding="utf-8") as f:
        sar_ai2027_fields = [f'{name} <= {int(sar_ai2027_thresholds[name]*12)}mo' for name in sar_ai2027_thresholds.keys()]
        ac_fields = [f'{name} <= {int(ac_thresholds[name]*12)}mo' if ac_thresholds[name] < 1 else f'{name} <= {int(ac_thresholds[name])}yr' for name in ac_thresholds.keys()]
        sar_one_year_fields = [f'{name} <= 1yr' for name in sar_one_year_thresholds.keys()]

        fieldnames = ['Condition'] + sar_ai2027_fields + ac_fields + sar_one_year_fields

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved combined conditional probability table to: {output_path}")


def generate_conditional_probability_table_ac(
    transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_path: Path
) -> None:
    """Generate table showing probabilities conditioned on AC arrival year.

    Shows probability that takeoff is faster than AI 2027, conditioned on:
    - Unconditional
    - AC arrives in 2027
    - AC arrives in 2030
    - AC arrives in 2035

    Metrics:
    - p(AC→TED-AI) < 9 months
    - p(AC→ASI) < 1 year

    Args:
        transitions_dict: dict with keys 'AC→TED-AI', 'AC→ASI'
        output_path: path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define thresholds for each transition (in years)
    thresholds = {
        'AC→TED-AI': 9/12,  # 9 months
        'AC→ASI': 1.0  # 1 year
    }

    # Define AC arrival year bins (end of year cutoffs)
    ac_year_bins = {
        'Unconditional': (0, np.inf),
        'AC in 2027': (2027.0, 2028.0),
        'AC in 2030': (2030.0, 2031.0),
        'AC in 2035': (2035.0, 2036.0)
    }

    # Calculate probabilities for each condition and transition
    rows = []

    for condition_name, (year_min, year_max) in ac_year_bins.items():
        row = {'Condition': condition_name}

        for transition_name, threshold in thresholds.items():
            if transition_name not in transitions_dict:
                row[f'{transition_name} <= {int(threshold*12)}mo' if threshold < 1 else f'{transition_name} <= {int(threshold)}yr'] = 'N/A'
                continue

            transitions = transitions_dict[transition_name]

            # Filter by AC arrival year
            if condition_name == 'Unconditional':
                filtered = transitions
            else:
                filtered = [(ft, tt, d) for ft, tt, d in transitions
                           if year_min <= ft < year_max]

            if len(filtered) == 0:
                row[f'{transition_name} <= {int(threshold*12)}mo' if threshold < 1 else f'{transition_name} <= {int(threshold)}yr'] = 'N/A (0 samples)'
                continue

            durations = np.array([d for _, _, d in filtered])
            prob = (np.sum(durations <= threshold) / len(durations) * 100)
            row[f'{transition_name} <= {int(threshold*12)}mo' if threshold < 1 else f'{transition_name} <= {int(threshold)}yr'] = f"{prob:.2g}%"

        rows.append(row)

    # Write to CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ['Condition'] + [f'{name} <= {int(thresholds[name]*12)}mo' if thresholds[name] < 1 else f'{name} <= {int(thresholds[name])}yr' for name in thresholds.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved AC conditional probability table to: {output_path}")


def plot_probability_vs_ac_arrival(
    transitions_dict: Dict[str, List[Tuple[float, float, float]]],
    output_dir: Path,
    run_label: str = ""
) -> None:
    """Plot how probabilities change as a function of AC arrival time.

    Creates two plots:
    1. P(faster than AI 2027) vs AC arrival year for each transition
    2. P(≤ 1 year takeoff) vs AC arrival year for each transition

    Args:
        transitions_dict: dict with keys 'AC→TED-AI', 'AC→ASI'
        output_dir: directory for output files
        run_label: optional label to add to titles (e.g., "no correlation")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define thresholds for "faster than AI 2027" (varies by transition)
    ai2027_thresholds = {
        'AC→TED-AI': 9/12,  # 9 months
        'AC→ASI': 1.0  # 1 year
    }

    # AC arrival year bins (1-year windows)
    ac_years = np.arange(2025, 2051, 1)

    # Plot 1: P(faster than AI 2027) vs AC arrival
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    colors = {'AC→TED-AI': 'tab:blue', 'AC→ASI': 'tab:orange'}

    for transition_name in ai2027_thresholds.keys():
        if transition_name not in transitions_dict:
            continue

        transitions = transitions_dict[transition_name]
        threshold = ai2027_thresholds[transition_name]

        probs = []
        counts = []

        for year in ac_years:
            # Filter transitions where AC arrives in this year
            filtered = [(ft, tt, d) for ft, tt, d in transitions
                       if year <= ft < year + 1]

            if len(filtered) > 0:
                durations = np.array([d for _, _, d in filtered])
                prob = np.sum(durations < threshold) / len(durations) * 100
                probs.append(prob)
                counts.append(len(filtered))
            else:
                probs.append(np.nan)
                counts.append(0)

        # Plot with marker size proportional to sample count (scaled more aggressively for visibility)
        marker_sizes = [max(10, min(c / 2, 400)) for c in counts]
        valid_mask = ~np.isnan(probs)

        label = f'{transition_name} (< {int(threshold*12)}mo)' if threshold < 1 else f'{transition_name} (< {int(threshold)}yr)'
        ax1.scatter(ac_years[valid_mask], np.array(probs)[valid_mask],
                   s=np.array(marker_sizes)[valid_mask], alpha=0.6,
                   color=colors.get(transition_name, 'tab:gray'),
                   label=label)
        ax1.plot(ac_years[valid_mask], np.array(probs)[valid_mask],
                linewidth=2, color=colors.get(transition_name, 'tab:gray'), alpha=0.5)

    ax1.set_xlabel("AC Arrival Year", fontsize=12)
    ax1.set_ylabel("Probability (%)", fontsize=12)
    title1 = f"P(AI 2027 Takeoff or Faster) vs AC Arrival Year{' - ' + run_label if run_label else ''}"
    ax1.set_title(title1, fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    ax1.set_xlim(ac_years[0] - 0.5, ac_years[-1] + 0.5)
    ax1.set_ylim(0, 100)

    output_path1 = output_dir / "prob_ai2027_takeoff_vs_ac_arrival.png"
    plt.tight_layout()
    plt.savefig(output_path1, dpi=150)
    plt.close()
    print(f"Saved P(AI 2027 takeoff or faster) vs AC arrival plot to: {output_path1}")

    # Plot 2: P(≤ 1 year takeoff) vs AC arrival
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    for transition_name in transitions_dict.keys():
        transitions = transitions_dict[transition_name]

        probs = []
        counts = []

        for year in ac_years:
            # Filter transitions where AC arrives in this year
            filtered = [(ft, tt, d) for ft, tt, d in transitions
                       if year <= ft < year + 1]

            if len(filtered) > 0:
                durations = np.array([d for _, _, d in filtered])
                prob = np.sum(durations <= 1.0) / len(durations) * 100
                probs.append(prob)
                counts.append(len(filtered))
            else:
                probs.append(np.nan)
                counts.append(0)

        # Plot with marker size proportional to sample count (scaled more aggressively for visibility)
        marker_sizes = [max(10, min(c / 2, 400)) for c in counts]
        valid_mask = ~np.isnan(probs)

        ax2.scatter(ac_years[valid_mask], np.array(probs)[valid_mask],
                   s=np.array(marker_sizes)[valid_mask], alpha=0.6,
                   color=colors.get(transition_name, 'tab:gray'),
                   label=f'{transition_name} ≤ 1yr')
        ax2.plot(ac_years[valid_mask], np.array(probs)[valid_mask],
                linewidth=2, color=colors.get(transition_name, 'tab:gray'), alpha=0.5)

    ax2.set_xlabel("AC Arrival Year", fontsize=12)
    ax2.set_ylabel("Probability (%)", fontsize=12)
    title2 = f"P(≤ 1 Year Takeoff) vs AC Arrival Year{' - ' + run_label if run_label else ''}"
    ax2.set_title(title2, fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=11)
    ax2.set_xlim(ac_years[0] - 0.5, ac_years[-1] + 0.5)
    ax2.set_ylim(0, 100)

    output_path2 = output_dir / "prob_1yr_takeoff_vs_ac_arrival.png"
    plt.tight_layout()
    plt.savefig(output_path2, dpi=150)
    plt.close()
    print(f"Saved P(≤ 1yr takeoff) vs AC arrival plot to: {output_path2}")


def generate_fast_takeoff_analysis(
    rollouts_file: Path,
    output_dir: Path,
    run_label: str = ""
) -> None:
    """Generate all fast takeoff analysis outputs.

    Args:
        rollouts_file: path to rollouts.jsonl
        output_dir: base directory for outputs (will create fast_takeoff_outputs subfolder)
        run_label: optional label to add to plot titles (e.g., "no correlation")
    """
    # Create output directory
    fast_takeoff_dir = output_dir / "fast_takeoff_outputs"
    fast_takeoff_dir.mkdir(parents=True, exist_ok=True)

    # ========== SAR-BASED ANALYSIS ==========
    print("=" * 60)
    print("SAR-BASED FAST TAKEOFF ANALYSIS")
    print("=" * 60)

    # Define transitions to analyze
    sar_transitions_to_analyze = {
        'SAR→SIAR': ('SAR-level-experiment-selection-skill', 'SIAR-level-experiment-selection-skill'),
        'SAR→TED-AI': ('SAR-level-experiment-selection-skill', 'TED-AI'),
        'SAR→ASI': ('SAR-level-experiment-selection-skill', 'ASI')
    }

    # Read transition data
    print("\nReading SAR transition data...")
    reader = RolloutsReader(rollouts_file)
    sar_transitions_dict = {}
    for name, (from_ms, to_ms) in sar_transitions_to_analyze.items():
        transitions = reader.read_transition_data(from_ms, to_ms, include_censored=False)
        sar_transitions_dict[name] = transitions
        print(f"  {name}: {len(transitions)} complete transitions")

    # (a) Generate transition duration PDFs with highlights
    print("\nGenerating SAR transition duration PDFs...")
    sar_pdf_transitions = {
        'SAR→TED-AI': sar_transitions_dict['SAR→TED-AI'],
        'SAR→ASI': sar_transitions_dict['SAR→ASI']
    }
    plot_path = fast_takeoff_dir / "sar_to_advanced_ai_durations.png"
    plot_transition_duration_pdfs(sar_pdf_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: SAR to Advanced AI Transition Durations",
                                  run_label=run_label)

    # (a2) Generate transition duration CDFs
    print("\nGenerating SAR transition duration CDFs...")
    plot_path = fast_takeoff_dir / "sar_to_advanced_ai_durations_cdf.png"
    plot_transition_duration_cdfs(sar_pdf_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: SAR to Advanced AI Transition Durations (CDF)",
                                  run_label=run_label)

    # (a3) Generate SAR→SIAR PDF and CDF
    print("\nGenerating SAR→SIAR PDF and CDF...")
    sar_siar_transitions = {'SAR→SIAR': sar_transitions_dict['SAR→SIAR']}
    plot_path = fast_takeoff_dir / "sar_to_siar_durations.png"
    plot_transition_duration_pdfs(sar_siar_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: SAR to SIAR Transition Duration",
                                  run_label=run_label)
    plot_path = fast_takeoff_dir / "sar_to_siar_durations_cdf.png"
    plot_transition_duration_cdfs(sar_siar_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: SAR to SIAR Transition Duration (CDF)",
                                  run_label=run_label)

    # (b) Generate conditional probability table
    print("\nGenerating SAR conditional probability table...")
    table_path = fast_takeoff_dir / "conditional_fast_takeoff_probs_sar.csv"
    generate_conditional_probability_table(sar_transitions_dict, table_path)

    # (c) Generate probability vs SAR arrival plots
    print("\nGenerating probability vs SAR arrival plots...")
    plot_probability_vs_sar_arrival(sar_transitions_dict, fast_takeoff_dir, run_label=run_label)

    # ========== AC-BASED ANALYSIS ==========
    print("\n" + "=" * 60)
    print("AC-BASED FAST TAKEOFF ANALYSIS")
    print("=" * 60)

    # Define AC-based transitions to analyze
    ac_transitions_to_analyze = {
        'AC→SIAR': ('AC', 'SIAR-level-experiment-selection-skill'),
        'AC→TED-AI': ('AC', 'TED-AI'),
        'AC→ASI': ('AC', 'ASI')
    }

    # Read AC transition data
    print("\nReading AC transition data...")
    ac_transitions_dict = {}
    for name, (from_ms, to_ms) in ac_transitions_to_analyze.items():
        transitions = reader.read_transition_data(from_ms, to_ms, include_censored=False)
        ac_transitions_dict[name] = transitions
        print(f"  {name}: {len(transitions)} complete transitions")

    # (a) Generate AC transition duration PDFs with highlights
    print("\nGenerating AC transition duration PDFs...")
    ac_advanced_transitions = {
        'AC→SIAR': ac_transitions_dict['AC→SIAR'],
        'AC→TED-AI': ac_transitions_dict['AC→TED-AI'],
        'AC→ASI': ac_transitions_dict['AC→ASI']
    }
    plot_path = fast_takeoff_dir / "ac_to_advanced_ai_durations.png"
    plot_transition_duration_pdfs(ac_advanced_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: AC to Advanced AI Transition Durations",
                                  run_label=run_label)

    # (a2) Generate AC transition duration CDFs
    print("\nGenerating AC transition duration CDFs...")
    plot_path = fast_takeoff_dir / "ac_to_advanced_ai_durations_cdf.png"
    plot_transition_duration_cdfs(ac_advanced_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: AC to Advanced AI Transition Durations (CDF)",
                                  run_label=run_label)

    # (a3) Generate AC→SIAR PDF and CDF
    print("\nGenerating AC→SIAR PDF and CDF...")
    ac_siar_transitions = {'AC→SIAR': ac_transitions_dict['AC→SIAR']}
    plot_path = fast_takeoff_dir / "ac_to_siar_durations.png"
    plot_transition_duration_pdfs(ac_siar_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: AC to SIAR Transition Duration",
                                  run_label=run_label)
    plot_path = fast_takeoff_dir / "ac_to_siar_durations_cdf.png"
    plot_transition_duration_cdfs(ac_siar_transitions, plot_path, max_duration=10.0,
                                  title="Fast Takeoff: AC to SIAR Transition Duration (CDF)",
                                  run_label=run_label)

    # (b) Generate AC conditional probability table with AI 2027 thresholds
    print("\nGenerating AC conditional probability table...")
    table_path = fast_takeoff_dir / "conditional_fast_takeoff_probs_ac.csv"
    generate_conditional_probability_table_ac(ac_transitions_dict, table_path)

    # (c) Generate probability vs AC arrival plots
    print("\nGenerating probability vs AC arrival plots...")
    plot_probability_vs_ac_arrival(ac_transitions_dict, fast_takeoff_dir, run_label=run_label)

    # ========== COMBINED TABLE ==========
    print("\n" + "=" * 60)
    print("COMBINED SAR+AC CONDITIONAL PROBABILITY TABLE")
    print("=" * 60)

    print("\nGenerating combined conditional probability table...")
    combined_table_path = fast_takeoff_dir / "conditional_fast_takeoff_probs_combined.csv"
    generate_combined_conditional_probability_table(sar_transitions_dict, ac_transitions_dict, combined_table_path)

    print(f"\n✓ Fast takeoff analysis complete! All outputs saved to: {fast_takeoff_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fast takeoff analysis",
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
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label to add to plot titles (e.g., 'no correlation')"
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
    generate_fast_takeoff_analysis(rollouts_file, output_dir, run_label=args.label)


if __name__ == "__main__":
    main()
