#!/usr/bin/env python3
"""
Explain why the difference in median milestone arrival dates differs from
the median transition duration.

WARNING: This script is a huge mess of bespoke nonsense and shouldn't be used
or trusted at the moment. It uses outdated plotting techniques and doesn't
follow the project's plotting utility conventions.

This script creates visualizations that make it clear why:
  median(arrival_time_B) - median(arrival_time_A) != median(arrival_time_B - arrival_time_A)

This is a consequence of how medians behave with correlated variables and
non-linear relationships. The script generates:

1. Scatter plot showing correlation between arrival times at consecutive milestones
2. Visualization of how individual trajectories map from arrival_A to arrival_B
3. Histogram showing the distribution of transition durations vs arrival dates
4. Quantile-quantile plot explaining the difference

Usage:
  python scripts/explain_median_difference_paradox.py --rollouts outputs/run/rollouts.jsonl

  python scripts/explain_median_difference_paradox.py --rollouts outputs/run/rollouts.jsonl \
    --milestone-a AC --milestone-b SAR-level-experiment-selection-skill
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()


def extract_transition_data(
    rollouts_file: Path,
    milestone_a: str,
    milestone_b: str,
    include_censored: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float], int, int, int]:
    """Extract paired arrival times and transition durations.

    Args:
        rollouts_file: Path to rollouts.jsonl
        milestone_a: First milestone name
        milestone_b: Second milestone name
        include_censored: If True, include all trajectories, treating non-achieved
                         milestones as achieved at simulation end time (censored values)

    Returns:
        arrival_times_a: list of arrival times at milestone A (or sim end if censored)
        arrival_times_b: list of arrival times at milestone B (or sim end if censored)
        transition_durations: list of durations from A to B (or to sim end if censored)
        sim_end_times: list of simulation end times for each trajectory
        num_both_achieved: count of trajectories achieving both milestones
        num_only_a_achieved: count of trajectories achieving only A (not B)
        num_neither_achieved: count of trajectories achieving neither milestone
    """
    arrival_times_a: List[float] = []
    arrival_times_b: List[float] = []
    transition_durations: List[float] = []
    sim_end_times: List[float] = []
    num_both_achieved = 0
    num_only_a_achieved = 0
    num_neither_achieved = 0

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

            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue

            # Get simulation end time
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    sim_end = float(times_array[-1])
                except Exception:
                    sim_end = np.nan
            else:
                sim_end = np.nan

            if not np.isfinite(sim_end):
                continue

            # Get milestone A time
            m_a = milestones.get(milestone_a)
            time_a = None
            if isinstance(m_a, dict):
                time_a_raw = m_a.get("time")
                if time_a_raw is not None and np.isfinite(float(time_a_raw)):
                    time_a = float(time_a_raw)

            # Get milestone B time
            m_b = milestones.get(milestone_b)
            time_b = None
            if isinstance(m_b, dict):
                time_b_raw = m_b.get("time")
                if time_b_raw is not None and np.isfinite(float(time_b_raw)):
                    time_b = float(time_b_raw)

            # Categorize trajectory
            a_achieved = time_a is not None
            b_achieved = time_b is not None

            if include_censored:
                # Include all trajectories, using sim_end for non-achieved milestones
                use_time_a = time_a if a_achieved else sim_end
                use_time_b = time_b if b_achieved else sim_end

                # Skip trajectories where B was achieved before A (impossible ordering)
                if a_achieved and b_achieved and time_b <= time_a:
                    continue

                # Skip trajectories where first milestone (A) was not achieved
                # (can't have a transition without a starting point)
                if not a_achieved:
                    num_neither_achieved += 1
                    continue

                arrival_times_a.append(use_time_a)
                arrival_times_b.append(use_time_b)
                transition_durations.append(use_time_b - use_time_a)
                sim_end_times.append(sim_end)

                # Count categories
                if a_achieved and b_achieved:
                    num_both_achieved += 1
                elif a_achieved and not b_achieved:
                    num_only_a_achieved += 1
            else:
                # Only include trajectories where both are achieved
                if a_achieved and b_achieved and time_b > time_a:
                    arrival_times_a.append(time_a)
                    arrival_times_b.append(time_b)
                    transition_durations.append(time_b - time_a)
                    sim_end_times.append(sim_end)
                    num_both_achieved += 1
                elif a_achieved and not b_achieved:
                    num_only_a_achieved += 1
                elif not a_achieved:
                    num_neither_achieved += 1

    return arrival_times_a, arrival_times_b, transition_durations, sim_end_times, num_both_achieved, num_only_a_achieved, num_neither_achieved


def extract_unconditional_data(
    rollouts_file: Path,
    milestone_a: str,
    milestone_b: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Extract unconditional arrival times and transition durations.

    This includes ALL trajectories, treating non-achieved milestones as
    achieved at simulation end (censored values). This matches the behavior
    of milestone_summary_table.py.

    Returns:
        arrival_times_a: Array of arrival times at A (all trajectories, censored at sim end)
        arrival_times_b: Array of arrival times at B (all trajectories, censored at sim end)
        transition_durations: Array of transition durations (only where A achieved)
        transition_start_times: Array of A arrival times corresponding to transition_durations
        arrival_hit_flags: Boolean array indicating whether milestone A was achieved
        transition_hit_flags: Boolean array indicating whether milestone B was achieved (only where A achieved)
        sim_end_times: Array of simulation end times for each trajectory
        median_a: Median arrival time at milestone A
        median_b: Median arrival time at milestone B
        median_transition: Median transition duration
    """
    arrival_times_a: List[float] = []
    arrival_times_b: List[float] = []
    transition_durations: List[float] = []
    transition_start_times: List[float] = []
    arrival_hit_flags: List[bool] = []
    transition_hit_flags: List[bool] = []
    sim_end_times: List[float] = []

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

            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue

            # Get simulation end time
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    sim_end = float(times_array[-1])
                except Exception:
                    continue
            else:
                continue

            if not np.isfinite(sim_end):
                continue

            sim_end_times.append(sim_end)

            # Get milestone A time
            m_a = milestones.get(milestone_a)
            time_a = None
            if isinstance(m_a, dict):
                time_a_raw = m_a.get("time")
                if time_a_raw is not None and np.isfinite(float(time_a_raw)):
                    time_a = float(time_a_raw)

            a_achieved = time_a is not None

            # Get milestone B time
            m_b = milestones.get(milestone_b)
            time_b = None
            if isinstance(m_b, dict):
                time_b_raw = m_b.get("time")
                if time_b_raw is not None and np.isfinite(float(time_b_raw)):
                    time_b = float(time_b_raw)

            b_achieved = time_b is not None

            # For arrival times: use sim_end if not achieved (ALL trajectories)
            use_time_a = time_a if time_a is not None else sim_end
            use_time_b = time_b if time_b is not None else sim_end

            arrival_times_a.append(use_time_a)
            arrival_times_b.append(use_time_b)
            arrival_hit_flags.append(a_achieved)

            # For transition durations: only if A was achieved
            if time_a is not None:
                use_time_b_for_transition = time_b if time_b is not None else sim_end
                if use_time_b_for_transition > time_a:
                    transition_durations.append(use_time_b_for_transition - time_a)
                    transition_hit_flags.append(b_achieved)
                    transition_start_times.append(time_a)

    # Convert to numpy arrays
    arrival_times_a_arr = np.array(arrival_times_a)
    arrival_times_b_arr = np.array(arrival_times_b)
    transition_durations_arr = np.array(transition_durations)
    transition_start_times_arr = np.array(transition_start_times)
    arrival_hit_flags_arr = np.array(arrival_hit_flags, dtype=bool)
    transition_hit_flags_arr = np.array(transition_hit_flags, dtype=bool)
    sim_end_times_arr = np.array(sim_end_times)

    return (
        arrival_times_a_arr,
        arrival_times_b_arr,
        transition_durations_arr,
        transition_start_times_arr,
        arrival_hit_flags_arr,
        transition_hit_flags_arr,
        sim_end_times_arr,
        np.median(arrival_times_a_arr),
        np.median(arrival_times_b_arr),
        np.median(transition_durations_arr),
    )


def create_scatter_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
    diff_in_medians_cond: Optional[float] = None,
    diff_in_medians_uncond: Optional[float] = None,
) -> None:
    """Create scatter plot showing correlation between arrival times.

    Args:
        diff_in_medians_cond: Conditional difference (prev milestone achieved)
        diff_in_medians_uncond: Unconditional difference (all rollouts + censored)
    """

    median_a = np.median(arrival_times_a)
    median_b = np.median(arrival_times_b)
    median_duration = np.median(transition_durations)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot with color coding by transition duration
    scatter = ax.scatter(
        arrival_times_a,
        arrival_times_b,
        c=transition_durations,
        cmap='viridis',
        alpha=0.6,
        s=30,
        edgecolors='black',
        linewidths=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Transition Duration (years)', fontsize=11)

    # Add diagonal line y = x (no transition)
    min_val = min(arrival_times_a.min(), arrival_times_b.min())
    max_val = max(arrival_times_a.max(), arrival_times_b.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.3, linewidth=1, label='No transition (y=x)')

    # Add median lines
    ax.axvline(median_a, color='tab:blue', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_a}: {median_a:.2f}')
    ax.axhline(median_b, color='tab:orange', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_b}: {median_b:.2f}')

    # Mark the median point
    ax.plot(median_a, median_b, 'ro', markersize=12,
            markeredgewidth=2, markeredgecolor='white',
            label=f'Median pair: ({median_a:.2f}, {median_b:.2f})', zorder=10)

    # Add line showing difference in medians
    ax.plot([median_a, median_a], [median_a, median_b],
            'r-', linewidth=3, alpha=0.7, label=f'Δ Medians: {median_b - median_a:.2f} yrs')

    # Add line showing median transition duration
    # Find point on plot where we can show this
    q25_a = np.percentile(arrival_times_a, 25)
    q25_idx = np.argmin(np.abs(arrival_times_a - q25_a))
    example_a = arrival_times_a[q25_idx]
    example_b = example_a + median_duration
    ax.plot([example_a, example_a], [example_a, example_b],
            'g-', linewidth=3, alpha=0.7,
            label=f'Median Duration: {median_duration:.2f} yrs')

    # Add text annotation explaining the paradox
    paradox_text = (
        f"THE PARADOX:\n"
        f"Difference in medians (both achieved): {median_b - median_a:.2f} years\n"
        f"Median of differences: {median_duration:.2f} years\n"
        f"Gap: {abs((median_b - median_a) - median_duration):.2f} years\n"
    )

    if diff_in_medians_cond is not None:
        paradox_text += f"\nΔ (Prev Achieved): {diff_in_medians_cond:.2f} yrs\n"

    if diff_in_medians_uncond is not None:
        paradox_text += f"Δ (All+Censored): {diff_in_medians_uncond:.2f} yrs\n"

    paradox_text += (
        f"\n"
        f"WHY? The median of B minus median of A\n"
        f"doesn't equal the median transition duration\n"
        f"because each trajectory follows a different\n"
        f"path through this space."
    )

    ax.text(0.02, 0.02, paradox_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2),
            family='monospace')

    # Calculate correlation
    correlation = np.corrcoef(arrival_times_a, arrival_times_b)[0, 1]
    stats_text = (
        f"N = {len(arrival_times_a)} trajectories\n"
        f"Correlation: {correlation:.3f}\n"
        f"\n"
        f"Transition duration:\n"
        f"  P25: {np.percentile(transition_durations, 25):.2f}\n"
        f"  P50: {median_duration:.2f}\n"
        f"  P75: {np.percentile(transition_durations, 75):.2f}"
    )

    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            family='monospace')

    ax.set_xlabel(f'Arrival Time at {milestone_a} (years)', fontsize=12)
    ax.set_ylabel(f'Arrival Time at {milestone_b} (years)', fontsize=12)
    ax.set_title(
        f'Correlation Between Milestone Arrivals:\n{milestone_a} → {milestone_b}',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to: {out_path}")


def create_arrival_vs_duration_plot(
    arrival_times_a: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
) -> None:
    """Create plot showing arrival time at A vs transition duration A→B.

    This plot helps visualize whether trajectories that arrive early/late at milestone A
    tend to have shorter/longer transition times to milestone B.
    """

    median_a = np.median(arrival_times_a)
    median_duration = np.median(transition_durations)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    scatter = ax.scatter(
        arrival_times_a,
        transition_durations,
        alpha=0.5,
        s=30,
        c='tab:blue',
        edgecolors='black',
        linewidths=0.5
    )

    # Add median lines
    ax.axvline(median_a, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_a} arrival: {median_a:.2f}')
    ax.axhline(median_duration, color='green', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median transition duration: {median_duration:.2f} yrs')

    # Mark the median point
    ax.plot(median_a, median_duration, 'ro', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10,
            label=f'Median point')

    # Calculate correlation
    correlation = np.corrcoef(arrival_times_a, transition_durations)[0, 1]

    # Add trend line
    z = np.polyfit(arrival_times_a, transition_durations, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(arrival_times_a.min(), arrival_times_a.max(), 100)
    ax.plot(x_trend, p(x_trend), "purple", linestyle='-', linewidth=2, alpha=0.6,
            label=f'Linear trend (corr={correlation:.3f})')

    # Add stats text
    stats_text = (
        f"N = {len(arrival_times_a)} trajectories\n"
        f"Correlation: {correlation:.3f}\n"
        f"\n"
        f"Duration statistics:\n"
        f"  P25: {np.percentile(transition_durations, 25):.2f} yrs\n"
        f"  P50: {median_duration:.2f} yrs\n"
        f"  P75: {np.percentile(transition_durations, 75):.2f} yrs\n"
        f"\n"
        f"Key insight:\n"
        f"{'Positive' if correlation > 0 else 'Negative' if correlation < 0 else 'No'} correlation means\n"
        f"{'late arrivals at ' + milestone_a + ' tend to\nhave longer transitions' if correlation > 0 else 'late arrivals at ' + milestone_a + ' tend to\nhave shorter transitions' if correlation < 0 else 'arrival time at ' + milestone_a + ' does not\npredict transition time'}"
    )

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            family='monospace')

    ax.set_xlabel(f'Arrival Time at {milestone_a} (years)', fontsize=12)
    ax.set_ylabel(f'Transition Duration: {milestone_a} → {milestone_b} (years)', fontsize=12)
    ax.set_title(f'How Arrival Time Affects Transition Duration\n{milestone_a} → {milestone_b}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved arrival vs duration plot to: {out_path}")


def create_quantile_comparison_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
    sim_end_times: Optional[np.ndarray] = None,
) -> None:
    """Create plot showing quantile-by-quantile comparison."""

    percentiles = np.linspace(0, 100, 101)

    # Print diagnostic information about high percentiles
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: High Percentile Analysis")
    print("=" * 70)

    # Analyze trajectories at different percentiles
    for p in [50, 75, 90, 95, 99]:
        idx_a = int(len(arrival_times_a) * p / 100)

        # Get the threshold for this percentile in arrival_a
        sorted_a_indices = np.argsort(arrival_times_a)
        traj_idx = sorted_a_indices[idx_a]

        time_a = arrival_times_a[traj_idx]
        time_b = arrival_times_b[traj_idx]
        duration = transition_durations[traj_idx]

        # Calculate quantile difference
        q_a = np.percentile(arrival_times_a, p)
        q_b = np.percentile(arrival_times_b, p)
        q_diff = q_b - q_a
        q_duration = np.percentile(transition_durations, p)

        print(f"\nP{p}:")
        print(f"  Q_A({p}): {q_a:.2f}")
        print(f"  Q_B({p}): {q_b:.2f}")
        print(f"  Difference in quantiles [Q_B - Q_A]: {q_diff:.2f} yrs")
        print(f"  Quantile of durations [Q_duration({p})]: {q_duration:.2f} yrs")
        print(f"  Gap: {abs(q_diff - q_duration):.2f} yrs")

    # Analyze censoring effects
    print("\n" + "-" * 70)
    print("CENSORING ANALYSIS:")
    print("-" * 70)

    if sim_end_times is not None and len(sim_end_times) > 0:
        sim_end_times_arr = np.array(sim_end_times)
        valid_sim_ends = sim_end_times_arr[np.isfinite(sim_end_times_arr)]
        if len(valid_sim_ends) > 0:
            median_sim_end = np.median(valid_sim_ends)
            print(f"Median simulation end time: {median_sim_end:.2f}")

            # Check how close late arrivals are to sim end
            near_end_threshold = 5.0  # within 5 years of end
            late_ac = arrival_times_a > (median_sim_end - 10)
            if np.any(late_ac):
                print(f"Trajectories with AC arrival within 10 years of sim end: {np.sum(late_ac)}")
                late_b_times = arrival_times_b[late_ac]
                late_sim_ends = sim_end_times_arr[late_ac]
                near_end = np.abs(late_b_times - late_sim_ends) < near_end_threshold
                print(f"  Of those, {np.sum(near_end)} have SAR arrival within {near_end_threshold} years of sim end")
                print(f"  This suggests potential censoring effects on late trajectories")

    # Look specifically at the trajectories with latest AC arrivals
    print("\n" + "-" * 70)
    print("TOP 10 LATEST AC ARRIVALS:")
    print("-" * 70)
    latest_indices = np.argsort(arrival_times_a)[-10:]
    for rank, idx in enumerate(reversed(latest_indices), 1):
        sim_end_info = ""
        if sim_end_times is not None:
            sim_end = sim_end_times[idx]
            if np.isfinite(sim_end):
                margin = sim_end - arrival_times_b[idx]
                sim_end_info = f", Sim end: {sim_end:.2f} (margin: {margin:.2f})"
        print(f"{rank:2d}. AC: {arrival_times_a[idx]:.2f}, "
              f"SAR: {arrival_times_b[idx]:.2f}, "
              f"Duration: {transition_durations[idx]:.2f} yrs{sim_end_info}")

    # Look at trajectories with shortest durations
    print("\n" + "-" * 70)
    print("TOP 10 SHORTEST DURATIONS:")
    print("-" * 70)
    shortest_indices = np.argsort(transition_durations)[:10]
    for rank, idx in enumerate(shortest_indices, 1):
        print(f"{rank:2d}. AC: {arrival_times_a[idx]:.2f}, "
              f"SAR: {arrival_times_b[idx]:.2f}, "
              f"Duration: {transition_durations[idx]:.2f} yrs")

    # Look at trajectories with longest durations
    print("\n" + "-" * 70)
    print("TOP 10 LONGEST DURATIONS:")
    print("-" * 70)
    longest_indices = np.argsort(transition_durations)[-10:]
    for rank, idx in enumerate(reversed(longest_indices), 1):
        print(f"{rank:2d}. AC: {arrival_times_a[idx]:.2f}, "
              f"SAR: {arrival_times_b[idx]:.2f}, "
              f"Duration: {transition_durations[idx]:.2f} yrs")

    print("=" * 70 + "\n")

    # Calculate percentiles for each distribution
    quantiles_a = np.percentile(arrival_times_a, percentiles)
    quantiles_b = np.percentile(arrival_times_b, percentiles)
    quantiles_duration = np.percentile(transition_durations, percentiles)

    # Calculate differences in quantiles
    quantile_differences = quantiles_b - quantiles_a

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top plot: Show quantiles of arrival times
    ax1.plot(percentiles, quantiles_a, 'b-', linewidth=2.5, label=f'{milestone_a} arrival times')
    ax1.plot(percentiles, quantiles_b, 'orange', linewidth=2.5, label=f'{milestone_b} arrival times')

    # Highlight the median (50th percentile)
    median_idx = 50
    ax1.plot(50, quantiles_a[median_idx], 'bo', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10)
    ax1.plot(50, quantiles_b[median_idx], 'o', color='orange', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10)

    # Draw vertical line at median
    ax1.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Draw arrow showing difference
    ax1.annotate('', xy=(50, quantiles_b[median_idx]), xytext=(50, quantiles_a[median_idx]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=3))
    ax1.text(52, (quantiles_a[median_idx] + quantiles_b[median_idx])/2,
            f'Δ = {quantiles_b[median_idx] - quantiles_a[median_idx]:.2f} yrs',
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))

    ax1.set_xlabel('Percentile', fontsize=12)
    ax1.set_ylabel('Arrival Time (years)', fontsize=12)
    ax1.set_title('Quantile Functions: Arrival Times at Each Milestone', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)

    # Bottom plot: Compare quantile differences vs duration quantiles
    ax2.plot(percentiles, quantile_differences, 'r-', linewidth=2.5,
            label='Difference in quantiles: Q_b(p) - Q_a(p)')
    ax2.plot(percentiles, quantiles_duration, 'g-', linewidth=2.5,
            label='Quantiles of transition duration: Q_duration(p)')

    # Highlight the median (50th percentile)
    ax2.plot(50, quantile_differences[median_idx], 'ro', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10,
            label=f'Δ at P50: {quantile_differences[median_idx]:.2f} yrs')
    ax2.plot(50, quantiles_duration[median_idx], 'go', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10,
            label=f'Median duration: {quantiles_duration[median_idx]:.2f} yrs')

    # Draw vertical line at median
    ax2.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add shaded region showing where they differ
    ax2.fill_between(percentiles, quantile_differences, quantiles_duration,
                     alpha=0.3, color='purple',
                     label='Difference between curves')

    # Add explanation text
    explanation = (
        f"AT THE MEDIAN (50th percentile):\n"
        f"• Difference in medians = {quantile_differences[median_idx]:.2f} yrs\n"
        f"• Median duration = {quantiles_duration[median_idx]:.2f} yrs\n"
        f"• Gap = {abs(quantile_differences[median_idx] - quantiles_duration[median_idx]):.2f} yrs\n"
        f"\n"
        f"The red curve shows: median_B - median_A\n"
        f"The green curve shows: median(B - A) for each trajectory\n"
        f"\n"
        f"These differ because trajectories are correlated:\n"
        f"Runs that reach A early tend to reach B early too,\n"
        f"but not necessarily with the median transition time."
    )

    ax2.text(0.02, 0.98, explanation,
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2),
            family='monospace')

    ax2.set_xlabel('Percentile', fontsize=12)
    ax2.set_ylabel('Duration (years)', fontsize=12)
    ax2.set_title('The Paradox Explained: Why Median Difference ≠ Difference in Medians',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quantile comparison plot to: {out_path}")


def create_distribution_comparison_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
    diff_in_medians_cond: Optional[float] = None,
    diff_in_medians_uncond: Optional[float] = None,
) -> None:
    """Create histogram comparison of distributions.

    Args:
        diff_in_medians_cond: Conditional difference (prev milestone achieved)
        diff_in_medians_uncond: Unconditional difference (all rollouts + censored)
    """

    median_a = np.median(arrival_times_a)
    median_b = np.median(arrival_times_b)
    median_duration = np.median(transition_durations)
    diff_in_medians = median_b - median_a

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Histogram of arrival times at A
    ax = axes[0]
    ax.hist(arrival_times_a, bins=40, alpha=0.7, color='tab:blue', edgecolor='black')
    ax.axvline(median_a, color='darkblue', linestyle='--', linewidth=3,
              label=f'Median: {median_a:.2f}')
    ax.set_xlabel(f'Arrival Time at {milestone_a} (years)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distribution of Arrival Times: {milestone_a}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Plot 2: Histogram of arrival times at B
    ax = axes[1]
    ax.hist(arrival_times_b, bins=40, alpha=0.7, color='tab:orange', edgecolor='black')
    ax.axvline(median_b, color='darkorange', linestyle='--', linewidth=3,
              label=f'Median: {median_b:.2f}')
    ax.axvline(median_a + diff_in_medians, color='red', linestyle=':', linewidth=2,
              label=f'Median_A + Δ: {median_a + diff_in_medians:.2f}', alpha=0.7)
    ax.set_xlabel(f'Arrival Time at {milestone_b} (years)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distribution of Arrival Times: {milestone_b}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Plot 3: Histogram of transition durations
    ax = axes[2]
    ax.hist(transition_durations, bins=40, alpha=0.7, color='tab:green', edgecolor='black')
    ax.axvline(median_duration, color='darkgreen', linestyle='--', linewidth=3,
              label=f'Median duration: {median_duration:.2f}')
    ax.axvline(diff_in_medians, color='red', linestyle='--', linewidth=3,
              label=f'Δ in medians: {diff_in_medians:.2f}', alpha=0.8)

    # Add shaded region showing the gap
    if median_duration < diff_in_medians:
        ax.axvspan(median_duration, diff_in_medians, alpha=0.3, color='purple',
                  label=f'Gap: {abs(diff_in_medians - median_duration):.2f} yrs')
    else:
        ax.axvspan(diff_in_medians, median_duration, alpha=0.3, color='purple',
                  label=f'Gap: {abs(diff_in_medians - median_duration):.2f} yrs')

    ax.set_xlabel(f'Transition Duration: {milestone_a} → {milestone_b} (years)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Transition Durations', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Add overall explanation text
    explanation = (
        f"KEY INSIGHT:\n"
        f"The transition duration for each trajectory is B_i - A_i.\n"
        f"But median(B) - median(A) ≠ median(B_i - A_i).\n"
        f"\n"
        f"This happens because:\n"
        f"1. Trajectories are correlated (early A → early B)\n"
        f"2. The median of differences ≠ difference of medians\n"
        f"   (medians don't distribute over subtraction)\n"
        f"\n"
        f"Results (Both Achieved):\n"
        f"• Difference in medians: {diff_in_medians:.2f} yrs\n"
        f"• Median of differences: {median_duration:.2f} yrs\n"
        f"• Gap: {abs(diff_in_medians - median_duration):.2f} yrs"
    )

    if diff_in_medians_cond is not None:
        explanation += (
            f"\n\n"
            f"Δ (Prev Achieved): {diff_in_medians_cond:.2f} yrs\n"
            f"  (median difference when prev milestone achieved)"
        )

    if diff_in_medians_uncond is not None:
        explanation += (
            f"\n"
            f"Δ (All+Censored): {diff_in_medians_uncond:.2f} yrs\n"
            f"  (median difference including all rollouts)"
        )

    fig.text(0.98, 0.5, explanation,
            transform=fig.transFigure, fontsize=11,
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='red', linewidth=2),
            family='monospace')

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Make room for text box
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution comparison plot to: {out_path}")


def create_correlation_vs_median_difference_plot(
    arrival_times_a_uncond: np.ndarray,
    transition_durations_uncond: np.ndarray,
    transition_start_times_uncond: np.ndarray,
    arrival_hit_flags_uncond: np.ndarray,
    transition_hit_flags_uncond: np.ndarray,
    sim_end_times_uncond: np.ndarray,
    arrival_times_a_corr: np.ndarray,
    transition_durations_corr: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
    median_a_uncond: float,
    median_b_uncond: float,
    median_transition_uncond: float,
    diff_in_medians_achieved: float,
) -> None:
    """Create plot showing how difference between medians varies with correlation.

    This version respects the censoring behavior in the empirical data. Rather than
    treating the transition duration as a purely continuous draw, we jointly sample:
      - An unconditional SAR arrival time (including censored runs at the horizon)
      - A transition bucket (hit vs. censored) aligned with the duration distribution
    Trajectories where SAR is censored or the sampled transition is censored place the
    SIAR arrival at the simulation horizon, replicating the observed mass at 2050.
    """
    from scipy.stats import norm
    from scipy.optimize import brentq
    from scipy.ndimage import uniform_filter1d

    # Observed values from the data
    diff_medians_obs = median_b_uncond - median_a_uncond
    median_transition_obs = median_transition_uncond

    # Correlation using the provided dataset (includes censored durations)
    corr_obs = np.corrcoef(arrival_times_a_corr, transition_durations_corr)[0, 1]

    print(f"\n{'='*70}")
    print("CORRELATION vs MEDIAN DIFFERENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Observed correlation r({milestone_a}, {milestone_b}-{milestone_a}) = {corr_obs:.4f}")
    print(f"Observed Median({milestone_a}) = {median_a_uncond:.2f}")
    print(f"Observed Median({milestone_b}) = {median_b_uncond:.2f}")
    print(f"Observed difference in medians = {diff_medians_obs:.2f} years")
    print(f"Observed median transition duration = {median_transition_obs:.2f} years")
    print(f"Gap: {abs(diff_medians_obs - median_transition_obs):.2f} years")
    print(f"Achieved-only difference in medians (SAR & {milestone_b} both hit): {diff_in_medians_achieved:.2f} years")

    # Sort arrays once so we can sample by rank while keeping censor flags aligned
    arrival_order = np.argsort(arrival_times_a_uncond)
    arrival_sorted = arrival_times_a_uncond[arrival_order]
    arrival_hit_sorted = arrival_hit_flags_uncond[arrival_order]
    sim_end_sorted = sim_end_times_uncond[arrival_order]

    if transition_start_times_uncond.size == 0:
        raise ValueError("No achieved transitions available to estimate conditional probabilities.")

    num_bins = min(30, max(6, transition_start_times_uncond.size // 20))
    if num_bins < 6:
        num_bins = 6
    bin_edges = np.linspace(transition_start_times_uncond.min(), transition_start_times_uncond.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p_hit_bins = np.empty(num_bins)
    hit_duration_bins: List[np.ndarray] = []
    global_p_hit = transition_hit_flags_uncond.mean()

    for i in range(num_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == num_bins - 1:
            range_mask = (transition_start_times_uncond >= left) & (transition_start_times_uncond <= right)
        else:
            range_mask = (transition_start_times_uncond >= left) & (transition_start_times_uncond < right)

        if range_mask.any():
            p_hit_bins[i] = transition_hit_flags_uncond[range_mask].mean()
        else:
            p_hit_bins[i] = p_hit_bins[i - 1] if i > 0 else global_p_hit

        hit_mask = range_mask & transition_hit_flags_uncond
        durations_bin = transition_durations_uncond[hit_mask]
        hit_duration_bins.append(np.sort(durations_bin) if durations_bin.size > 0 else None)

    non_empty_bins = [idx for idx, arr in enumerate(hit_duration_bins) if arr is not None]
    if not non_empty_bins:
        raise ValueError("No successful transitions to sample duration distribution.")
    nearest_non_empty = np.empty(num_bins, dtype=int)
    for i in range(num_bins):
        if i in non_empty_bins:
            nearest_non_empty[i] = i
        else:
            nearest_non_empty[i] = min(non_empty_bins, key=lambda j: abs(j - i))

    # Simulate across correlation values
    print(f"\nSimulating difference between medians for varying correlations (with censoring)...")
    correlations = np.linspace(-0.95, 0.95, 100)
    correlations = np.unique(np.concatenate([correlations, [corr_obs, 0.0]]))
    median_differences = []

    n_arrivals = len(arrival_sorted)
    if n_arrivals == 0:
        raise ValueError("Insufficient data to run correlation analysis simulation.")

    for target_corr in correlations:
        n_samples = max(20000, n_arrivals * 4)

        mean = [0.0, 0.0]
        cov = [[1.0, target_corr], [target_corr, 1.0]]
        seed = 42 + abs(int(target_corr * 1000))
        samples = np.random.RandomState(seed).multivariate_normal(mean, cov, n_samples)

        u_a = norm.cdf(samples[:, 0])
        u_dur = norm.cdf(samples[:, 1])

        idx_a = np.minimum((u_a * n_arrivals).astype(int), n_arrivals - 1)
        simulated_a = arrival_sorted[idx_a]
        a_hit = arrival_hit_sorted[idx_a]
        sim_end = sim_end_sorted[idx_a]
        simulated_b = sim_end.copy()

        mask_a_hit = a_hit
        if np.any(mask_a_hit):
            idx_candidates = np.where(mask_a_hit)[0]
            p_hit_vals = np.interp(
                simulated_a[idx_candidates],
                bin_centers,
                p_hit_bins,
                left=p_hit_bins[0],
                right=p_hit_bins[-1],
            )
            p_hit_vals = np.clip(p_hit_vals, 0.0, 1.0)

            u_slice = u_dur[idx_candidates]
            hit_local = u_slice < p_hit_vals

            if np.any(hit_local):
                idx_global_hits = idx_candidates[hit_local]
                u_for_hits = u_slice[hit_local] / np.maximum(p_hit_vals[hit_local], 1e-6)
                u_for_hits = np.clip(u_for_hits, 0.0, 0.999999)

                bin_idx = np.clip(
                    np.searchsorted(bin_edges, simulated_a[idx_global_hits], side='right') - 1,
                    0,
                    num_bins - 1,
                )
                mapped_bins = nearest_non_empty[bin_idx]

                unique_bins = np.unique(mapped_bins)
                for bin_id in unique_bins:
                    bin_mask = mapped_bins == bin_id
                    durations_pool = hit_duration_bins[bin_id]
                    if durations_pool is None or durations_pool.size == 0:
                        continue
                    u_vals_bin = u_for_hits[bin_mask]
                    idx_global_bin = idx_global_hits[bin_mask]
                    n_pool = len(durations_pool)
                    duration_indices = np.minimum((u_vals_bin * n_pool).astype(int), n_pool - 1)
                    durations_selected = durations_pool[duration_indices]
                    simulated_b[idx_global_bin] = simulated_a[idx_global_bin] + durations_selected

        simulated_b = np.minimum(simulated_b, sim_end)  # guard against exceeding horizon

        diff = np.median(simulated_b) - np.median(simulated_a)
        median_differences.append(diff)

    median_differences = np.array(median_differences)
    median_differences_smoothed = uniform_filter1d(median_differences, size=3)

    # Find correlation where difference = median transition (using smoothed curve)
    def objective(r):
        idx = np.argmin(np.abs(correlations - r))
        return median_differences_smoothed[idx] - median_transition_obs

    try:
        corr_at_median_transition = brentq(objective, -0.95, 0.95)
        print(f"Correlation where Diff(medians) = Median(transition): r = {corr_at_median_transition:.4f}")
    except ValueError:
        corr_at_median_transition = None
        print(f"No correlation in range makes Diff(medians) = Median(transition)")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot the smoothed simulated curve
    ax.plot(correlations, median_differences_smoothed, 'b-', linewidth=3,
            label='Simulated: median(B) - median(A)', zorder=5)

    # Add horizontal line for observed difference
    ax.axhline(diff_medians_obs, color='red', linestyle='--', linewidth=2.5,
               label=f'Observed difference: {diff_medians_obs:.2f} yrs', alpha=0.8)

    # Add horizontal line for median transition
    ax.axhline(median_transition_obs, color='green', linestyle='--', linewidth=2.5,
               label=f'Median transition: {median_transition_obs:.2f} yrs', alpha=0.8)

    # Add achieved-only difference line (achieved trajectories only)
    ax.axhline(diff_in_medians_achieved, color='tab:gray', linestyle=':', linewidth=2.5,
               alpha=0.9,
               label=f'Achieved-only Δ medians: {diff_in_medians_achieved:.2f} yrs')

    # Mark observed correlation
    ax.axvline(corr_obs, color='purple', linestyle='--', linewidth=2,
               label=f'Observed correlation: r={corr_obs:.3f}', alpha=0.7)

    # Mark the observed point on the curve
    simulated_at_obs = median_differences_smoothed[np.argmin(np.abs(correlations - corr_obs))]
    zero_idx = np.argmin(np.abs(correlations))
    simulated_at_zero = median_differences_smoothed[zero_idx]

    raw_at_obs = median_differences[np.argmin(np.abs(correlations - corr_obs))]
    raw_at_zero = median_differences[zero_idx]

    print(f"Simulated difference at observed correlation: {simulated_at_obs:.2f} years (smoothed) / {raw_at_obs:.2f} years (raw)")
    print(f"Simulated difference at zero correlation: {simulated_at_zero:.2f} years (smoothed) / {raw_at_zero:.2f} years (raw)")

    ax.plot(corr_obs, simulated_at_obs, 'ro', markersize=14,
            markeredgewidth=2, markeredgecolor='white', zorder=15,
            label=f'Simulated at observed r: {simulated_at_obs:.2f} yrs')

    # Mark zero correlation point
    ax.plot(0, simulated_at_zero, 'go', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10,
            label=f'Independent (r=0): {simulated_at_zero:.2f} yrs')

    # Mark crossover point if it exists
    if corr_at_median_transition is not None:
        ax.plot(corr_at_median_transition, median_transition_obs, 'o', color='orange',
                markersize=12, markeredgewidth=2, markeredgecolor='white', zorder=10,
                label=f'Crossover at r={corr_at_median_transition:.3f}')

    ax.set_xlabel(f'Correlation between {milestone_a} arrival and ({milestone_b}-{milestone_a}) transition', fontsize=13)
    ax.set_ylabel('Difference between medians: median(B) - median(A) (years)', fontsize=13)
    ax.set_title(f'How Correlation Affects Difference Between Medians\n{milestone_a} → {milestone_b}',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved unconditional median analysis plot to: {out_path}")


def create_correlation_vs_median_difference_plot_v2(
    arrival_times_a_uncond: np.ndarray,
    transition_durations_uncond: np.ndarray,
    arrival_times_a_corr: np.ndarray,
    transition_durations_corr: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
    median_a_uncond: float,
    median_b_uncond: float,
    median_transition_uncond: float,
) -> None:
    """Create plot showing how difference between medians varies with correlation.

    This version clarifies that the Gaussian copula simulation is a theoretical model
    that may not match the observed data structure.
    """
    from scipy.stats import norm, rankdata
    from scipy.optimize import brentq

    # Observed values from the data
    diff_medians_obs = median_b_uncond - median_a_uncond
    median_transition_obs = median_transition_uncond

    # Correlation using the provided dataset (includes censored durations)
    corr_obs = np.corrcoef(arrival_times_a_corr, transition_durations_corr)[0, 1]

    print(f"\n{'='*70}")
    print("CORRELATION vs MEDIAN DIFFERENCE ANALYSIS V2")
    print(f"{'='*70}")
    print(f"Observed correlation r({milestone_a}, {milestone_b}-{milestone_a}) = {corr_obs:.4f}")
    print(f"Observed Median({milestone_a}) = {median_a_uncond:.2f}")
    print(f"Observed Median({milestone_b}) = {median_b_uncond:.2f}")
    print(f"Observed difference in medians = {diff_medians_obs:.2f} years")
    print(f"Observed median transition duration = {median_transition_obs:.2f} years")
    print(f"Gap: {abs(diff_medians_obs - median_transition_obs):.2f} years")

    # Simulate across correlation values using Gaussian copula
    print(f"\nSimulating difference between medians for varying correlations...")
    correlations = np.linspace(-0.95, 0.95, 100)
    median_differences = []

    for target_corr in correlations:
        n_samples = max(10000, len(arrival_times_a_uncond) * 2)

        # Generate correlated normal samples
        mean = [0, 0]
        cov = [[1, target_corr], [target_corr, 1]]
        seed = 42 + abs(int(target_corr * 1000))
        samples = np.random.RandomState(seed).multivariate_normal(mean, cov, n_samples)

        # Convert to uniform via normal CDF
        u_a = norm.cdf(samples[:, 0])
        u_dur = norm.cdf(samples[:, 1])

        # Map to original distributions via empirical quantiles
        simulated_a = np.percentile(arrival_times_a_uncond, u_a * 100)
        simulated_dur = np.percentile(transition_durations_uncond, u_dur * 100)

        # Calculate B = A + duration
        simulated_b = simulated_a + simulated_dur

        # Calculate difference in medians
        diff = np.median(simulated_b) - np.median(simulated_a)
        median_differences.append(diff)

    median_differences = np.array(median_differences)

    # Apply light smoothing
    from scipy.ndimage import uniform_filter1d
    median_differences_smoothed = uniform_filter1d(median_differences, size=5)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot the smoothed simulated curve
    ax.plot(correlations, median_differences_smoothed, 'b-', linewidth=3,
            label='Gaussian copula simulation: median(B) - median(A)', zorder=5)

    # Add horizontal line for median transition (what we'd expect at r=0 or r=1)
    ax.axhline(median_transition_obs, color='green', linestyle='--', linewidth=2.5,
               label=f'Median transition duration: {median_transition_obs:.2f} yrs', alpha=0.8)

    # Mark the ACTUAL observed point (not on the simulation curve)
    ax.plot(corr_obs, diff_medians_obs, 's', color='red', markersize=16,
            markeredgewidth=2.5, markeredgecolor='darkred', zorder=20,
            label=f'Actual observed data: r={corr_obs:.3f}, diff={diff_medians_obs:.2f} yrs')

    # Mark what Gaussian copula predicts at the observed correlation
    simulated_at_obs = median_differences_smoothed[np.argmin(np.abs(correlations - corr_obs))]
    ax.plot(corr_obs, simulated_at_obs, 'o', color='cyan', markersize=14,
            markeredgewidth=2, markeredgecolor='blue', zorder=15,
            label=f'Gaussian copula at r={corr_obs:.3f}: {simulated_at_obs:.2f} yrs')

    # Mark zero correlation point
    zero_idx = np.argmin(np.abs(correlations))
    ax.plot(0, median_differences_smoothed[zero_idx], 'o', color='purple', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10,
            label=f'Gaussian copula at r=0: {median_differences_smoothed[zero_idx]:.2f} yrs')

    # Mark r=1 point to check if it equals median transition
    r1_idx = -1
    ax.plot(correlations[r1_idx], median_differences_smoothed[r1_idx], 'o', color='orange', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10,
            label=f'Gaussian copula at r=1: {median_differences_smoothed[r1_idx]:.2f} yrs')

    # Draw vertical line from observed point to Gaussian prediction
    ax.plot([corr_obs, corr_obs], [simulated_at_obs, diff_medians_obs],
            'r--', linewidth=2, alpha=0.5, zorder=1)

    # Annotate the gap
    gap = diff_medians_obs - simulated_at_obs
    mid_y = (diff_medians_obs + simulated_at_obs) / 2
    ax.annotate(f'Gap: {gap:.2f} yrs',
                xy=(corr_obs + 0.05, mid_y),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    ax.set_xlabel(f'Correlation between {milestone_a} arrival and ({milestone_b}-{milestone_a}) transition', fontsize=13)
    ax.set_ylabel('Difference between medians: median(B) - median(A) (years)', fontsize=13)
    ax.set_title(f'Gaussian Copula vs Observed Data: Median Difference Analysis\n{milestone_a} → {milestone_b}',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.95)

    # Add explanation text box
    explanation = (
        f"INTERPRETATION:\n\n"
        f"The blue curve shows what a Gaussian\n"
        f"copula model predicts for different\n"
        f"correlation values.\n\n"
        f"The red square shows the actual observed\n"
        f"data point.\n\n"
        f"The GAP indicates that the real data has\n"
        f"a different joint distribution structure\n"
        f"than the Gaussian copula assumes.\n\n"
        f"At r=0 (independence), Gaussian copula\n"
        f"predicts {median_differences_smoothed[zero_idx]:.2f} yrs,\n"
        f"which should equal median transition\n"
        f"({median_transition_obs:.2f} yrs).\n\n"
        f"At r=1 (perfect correlation), Gaussian\n"
        f"copula predicts {median_differences_smoothed[r1_idx]:.2f} yrs."
    )
    ax.text(0.02, 0.98, explanation,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.95, edgecolor='blue', linewidth=2),
            family='monospace')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V2 correlation analysis plot to: {out_path}")


def create_ac_vs_sar_scatter_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
) -> None:
    """Create simple scatter plot with AC date on x-axis and SAR date on y-axis."""

    median_a = np.median(arrival_times_a)
    median_b = np.median(arrival_times_b)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot
    ax.scatter(
        arrival_times_a,
        arrival_times_b,
        alpha=0.5,
        s=40,
        c='tab:blue',
        edgecolors='black',
        linewidths=0.5
    )

    # Add diagonal line y = x (no time difference)
    min_val = min(arrival_times_a.min(), arrival_times_b.min())
    max_val = max(arrival_times_a.max(), arrival_times_b.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.3, linewidth=1.5, label='Same time (y=x)')

    # Add median lines
    ax.axvline(median_a, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_a}: {median_a:.2f}')
    ax.axhline(median_b, color='green', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_b}: {median_b:.2f}')

    # Mark the median point
    ax.plot(median_a, median_b, 'ro', markersize=14,
            markeredgewidth=2, markeredgecolor='white',
            label=f'Median pair: ({median_a:.2f}, {median_b:.2f})', zorder=10)

    # Calculate correlation
    correlation = np.corrcoef(arrival_times_a, arrival_times_b)[0, 1]

    # Add stats text
    stats_text = (
        f"N = {len(arrival_times_a)} trajectories\n"
        f"Correlation: {correlation:.3f}\n"
        f"\n"
        f"{milestone_a} statistics:\n"
        f"  P25: {np.percentile(arrival_times_a, 25):.2f}\n"
        f"  P50: {median_a:.2f}\n"
        f"  P75: {np.percentile(arrival_times_a, 75):.2f}\n"
        f"\n"
        f"{milestone_b} statistics:\n"
        f"  P25: {np.percentile(arrival_times_b, 25):.2f}\n"
        f"  P50: {median_b:.2f}\n"
        f"  P75: {np.percentile(arrival_times_b, 75):.2f}"
    )

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', linewidth=2),
            family='monospace')

    ax.set_xlabel(f'{milestone_a} Date (years)', fontsize=13)
    ax.set_ylabel(f'{milestone_b} Date (years)', fontsize=13)
    ax.set_title(
        f'{milestone_a} vs {milestone_b} Arrival Times',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    # Set equal aspect ratio to make comparison clearer
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AC vs SAR scatter plot to: {out_path}")


def extract_three_milestone_data(
    rollouts_file: Path,
    milestone_a: str,
    milestone_b: str,
    milestone_c: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract arrival times for three milestones.

    Returns:
        arrival_times_a: list of arrival times at milestone A
        arrival_times_b: list of arrival times at milestone B
        arrival_times_c: list of arrival times at milestone C
        bc_durations: list of durations from B to C
    """
    arrival_times_a: List[float] = []
    arrival_times_b: List[float] = []
    arrival_times_c: List[float] = []
    bc_durations: List[float] = []

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

            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue

            # Get milestone A time
            m_a = milestones.get(milestone_a)
            time_a = None
            if isinstance(m_a, dict):
                time_a_raw = m_a.get("time")
                if time_a_raw is not None and np.isfinite(float(time_a_raw)):
                    time_a = float(time_a_raw)

            # Get milestone B time
            m_b = milestones.get(milestone_b)
            time_b = None
            if isinstance(m_b, dict):
                time_b_raw = m_b.get("time")
                if time_b_raw is not None and np.isfinite(float(time_b_raw)):
                    time_b = float(time_b_raw)

            # Get milestone C time
            m_c = milestones.get(milestone_c)
            time_c = None
            if isinstance(m_c, dict):
                time_c_raw = m_c.get("time")
                if time_c_raw is not None and np.isfinite(float(time_c_raw)):
                    time_c = float(time_c_raw)

            # Only include trajectories where all three are achieved
            if time_a is not None and time_b is not None and time_c is not None:
                if time_c > time_b:  # Ensure C comes after B
                    arrival_times_a.append(time_a)
                    arrival_times_b.append(time_b)
                    arrival_times_c.append(time_c)
                    bc_durations.append(time_c - time_b)

    return arrival_times_a, arrival_times_b, arrival_times_c, bc_durations


def create_ac_vs_sar_siar_duration_plot(
    rollouts_file: Path,
    milestone_a: str,
    milestone_b: str,
    milestone_c: str,
    out_path: Path,
) -> None:
    """Create scatter plot with AC time on x-axis and SAR-to-SIAR duration on y-axis."""

    # Extract three milestone data
    arrival_times_a, arrival_times_b, arrival_times_c, bc_durations = extract_three_milestone_data(
        rollouts_file, milestone_a, milestone_b, milestone_c
    )

    if len(arrival_times_a) == 0:
        print(f"Warning: No trajectories found with all three milestones: {milestone_a}, {milestone_b}, {milestone_c}")
        return

    # Convert to numpy arrays
    arrival_times_a = np.array(arrival_times_a)
    bc_durations = np.array(bc_durations)

    median_a = np.median(arrival_times_a)
    median_duration = np.median(bc_durations)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot
    ax.scatter(
        arrival_times_a,
        bc_durations,
        alpha=0.5,
        s=40,
        c='tab:blue',
        edgecolors='black',
        linewidths=0.5
    )

    # Add median lines
    ax.axvline(median_a, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_a}: {median_a:.2f}')
    ax.axhline(median_duration, color='green', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Median {milestone_b}→{milestone_c} duration: {median_duration:.2f} yrs')

    # Mark the median point
    ax.plot(median_a, median_duration, 'ro', markersize=14,
            markeredgewidth=2, markeredgecolor='white',
            label=f'Median pair: ({median_a:.2f}, {median_duration:.2f})', zorder=10)

    # Calculate correlation
    correlation = np.corrcoef(arrival_times_a, bc_durations)[0, 1]

    # Add trend line if correlation is significant
    if abs(correlation) > 0.1:
        z = np.polyfit(arrival_times_a, bc_durations, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(arrival_times_a.min(), arrival_times_a.max(), 100)
        ax.plot(x_trend, p(x_trend), "purple", linestyle='-', linewidth=2, alpha=0.6,
                label=f'Linear trend')

    # Add stats text
    # Determine correlation interpretation
    if correlation > 0:
        corr_type = "Positive"
        corr_meaning = "later AC → longer transition"
    elif correlation < 0:
        corr_type = "Negative"
        corr_meaning = "later AC → shorter transition"
    else:
        corr_type = "No"
        corr_meaning = "AC time does not predict transition"

    stats_text = (
        f"N = {len(arrival_times_a)} trajectories\n"
        f"Correlation: {correlation:.3f}\n"
        f"\n"
        f"{milestone_a} arrival statistics:\n"
        f"  P25: {np.percentile(arrival_times_a, 25):.2f}\n"
        f"  P50: {median_a:.2f}\n"
        f"  P75: {np.percentile(arrival_times_a, 75):.2f}\n"
        f"\n"
        f"{milestone_b}→{milestone_c} duration:\n"
        f"  P25: {np.percentile(bc_durations, 25):.2f} yrs\n"
        f"  P50: {median_duration:.2f} yrs\n"
        f"  P75: {np.percentile(bc_durations, 75):.2f} yrs\n"
        f"\n"
        f"Interpretation:\n"
        f"{corr_type} correlation means\n"
        f"{corr_meaning}"
    )

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', linewidth=2),
            family='monospace')

    ax.set_xlabel(f'{milestone_a} Arrival Time (years)', fontsize=13)
    ax.set_ylabel(f'{milestone_b} → {milestone_c} Duration (years)', fontsize=13)
    ax.set_title(
        f'{milestone_a} Time vs {milestone_b}→{milestone_c} Transition Duration',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AC vs SAR-SIAR duration plot to: {out_path}")


def create_correlation_comparison_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
) -> None:
    """Create plot comparing transition duration distributions under different correlation assumptions.

    Shows three distributions:
    1. Actual (observed correlation from the data)
    2. Independent (randomly shuffled B times, breaking correlation)
    3. Perfect rank correlation (sorted A and B times paired together)
    """

    # Actual transition durations (already computed)
    actual_durations = transition_durations
    median_actual = np.median(actual_durations)

    # Independent: shuffle arrival_times_b to break correlation
    independent_b = np.random.RandomState(42).permutation(arrival_times_b)
    independent_durations = independent_b - arrival_times_a
    median_independent = np.median(independent_durations)

    # Perfect rank correlation: sort both arrays and pair them
    sorted_a = np.sort(arrival_times_a)
    sorted_b = np.sort(arrival_times_b)
    perfect_corr_durations = sorted_b - sorted_a
    median_perfect = np.median(perfect_corr_durations)

    # Calculate actual correlation for reference
    actual_correlation = np.corrcoef(arrival_times_a, arrival_times_b)[0, 1]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top plot: Overlaid histograms
    ax = ax1
    bins = np.linspace(
        min(actual_durations.min(), independent_durations.min(), perfect_corr_durations.min()),
        max(actual_durations.max(), independent_durations.max(), perfect_corr_durations.max()),
        50
    )

    ax.hist(independent_durations, bins=bins, alpha=0.5, color='gray',
            edgecolor='black', linewidth=0.5, label='Independent (r=0.0)')
    ax.hist(actual_durations, bins=bins, alpha=0.6, color='tab:blue',
            edgecolor='black', linewidth=0.5, label=f'Actual (r={actual_correlation:.3f})')
    ax.hist(perfect_corr_durations, bins=bins, alpha=0.5, color='tab:red',
            edgecolor='black', linewidth=0.5, label='Perfect Rank Corr (r≈1.0)')

    # Add median lines
    ax.axvline(median_independent, color='gray', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(median_actual, color='tab:blue', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(median_perfect, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.8)

    # Add median annotations
    y_max = ax.get_ylim()[1]
    ax.text(median_independent, y_max * 0.95, f'  {median_independent:.1f}',
            color='gray', fontsize=10, fontweight='bold', rotation=90,
            verticalalignment='top', horizontalalignment='right')
    ax.text(median_actual, y_max * 0.95, f'  {median_actual:.1f}',
            color='tab:blue', fontsize=10, fontweight='bold', rotation=90,
            verticalalignment='top', horizontalalignment='right')
    ax.text(median_perfect, y_max * 0.95, f'  {median_perfect:.1f}',
            color='tab:red', fontsize=10, fontweight='bold', rotation=90,
            verticalalignment='top', horizontalalignment='right')

    ax.set_xlabel(f'Transition Duration: {milestone_a} → {milestone_b} (years)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Transition Duration Distribution Under Different Correlation Assumptions',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    # Bottom plot: Cumulative distribution functions
    ax = ax2

    # Sort for CDF plotting
    sorted_independent = np.sort(independent_durations)
    sorted_actual = np.sort(actual_durations)
    sorted_perfect = np.sort(perfect_corr_durations)

    # Compute CDF values
    cdf_values = np.linspace(0, 1, len(sorted_actual))

    ax.plot(sorted_independent, cdf_values, color='gray', linewidth=2.5,
            alpha=0.8, label='Independent (r=0.0)')
    ax.plot(sorted_actual, cdf_values, color='tab:blue', linewidth=2.5,
            alpha=0.8, label=f'Actual (r={actual_correlation:.3f})')
    ax.plot(sorted_perfect, cdf_values, color='tab:red', linewidth=2.5,
            alpha=0.8, label='Perfect Rank Corr (r≈1.0)')

    # Add median line
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    # Mark medians on CDF
    ax.plot(median_independent, 0.5, 'o', color='gray', markersize=10,
            markeredgewidth=2, markeredgecolor='white', zorder=10)
    ax.plot(median_actual, 0.5, 'o', color='tab:blue', markersize=10,
            markeredgewidth=2, markeredgecolor='white', zorder=10)
    ax.plot(median_perfect, 0.5, 'o', color='tab:red', markersize=10,
            markeredgewidth=2, markeredgecolor='white', zorder=10)

    ax.set_xlabel(f'Transition Duration: {milestone_a} → {milestone_b} (years)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Functions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

    # Add explanation text
    explanation = (
        f"CORRELATION EFFECTS ON TRANSITION TIME:\n"
        f"\n"
        f"Independent (gray):\n"
        f"  If {milestone_a} and {milestone_b} times were\n"
        f"  uncorrelated, median duration = {median_independent:.2f} yrs\n"
        f"\n"
        f"Actual (blue):\n"
        f"  With observed correlation r={actual_correlation:.3f},\n"
        f"  median duration = {median_actual:.2f} yrs\n"
        f"\n"
        f"Perfect Rank Correlation (red):\n"
        f"  If trajectories perfectly preserved rank order,\n"
        f"  median duration = {median_perfect:.2f} yrs\n"
        f"\n"
        f"KEY INSIGHT:\n"
        f"Correlation affects the SPREAD more than the median.\n"
        f"Stronger correlation → narrower distribution.\n"
        f"{'Positive' if actual_correlation > 0 else 'Negative'} correlation means trajectories that\n"
        f"reach {milestone_a} {'early tend to reach ' + milestone_b + ' early' if actual_correlation > 0 else 'late tend to reach ' + milestone_b + ' early'}."
    )

    fig.text(0.98, 0.02, explanation,
            transform=fig.transFigure, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='lightyellow', alpha=0.95, edgecolor='purple', linewidth=2),
            family='monospace')

    plt.tight_layout()  # No need to reserve space anymore
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation comparison plot to: {out_path}")


def create_transition_pdf_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
) -> None:
    """Create PDF (probability density function) plot for transition durations.

    Shows the distribution of transition times as a smooth density curve with
    key statistics overlaid. Only includes positive durations (B after A).
    """
    from scipy.stats import gaussian_kde

    # Filter to only positive durations (following plot_rollouts.py logic)
    # Negative durations mean B was achieved before A (out of order)
    positive_mask = (np.isfinite(transition_durations)) & (transition_durations > 0)
    transition_durations_filtered = transition_durations[positive_mask]

    # Also filter the corresponding A and B times for consistency
    arrival_times_a_filtered = arrival_times_a[positive_mask]
    arrival_times_b_filtered = arrival_times_b[positive_mask]

    # Count how many were filtered out
    num_negative = np.sum(transition_durations < 0)
    num_total = len(transition_durations)

    if len(transition_durations_filtered) == 0:
        print(f"Warning: No positive transition durations found for {milestone_a} → {milestone_b}")
        return

    median_duration = np.median(transition_durations_filtered)
    mean_duration = np.mean(transition_durations_filtered)
    p25 = np.percentile(transition_durations_filtered, 25)
    p75 = np.percentile(transition_durations_filtered, 75)

    # Calculate median difference for comparison
    median_a = np.median(arrival_times_a_filtered)
    median_b = np.median(arrival_times_b_filtered)
    diff_medians = median_b - median_a

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create KDE for smooth density estimation
    kde = gaussian_kde(transition_durations_filtered, bw_method='scott')

    # Create x values for plotting the PDF
    x_min = transition_durations_filtered.min()
    x_max = transition_durations_filtered.max()
    x_range = x_max - x_min
    x_vals = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
    pdf_vals = kde(x_vals)

    # Plot the PDF
    ax.fill_between(x_vals, pdf_vals, alpha=0.3, color='tab:blue', label='Probability density')
    ax.plot(x_vals, pdf_vals, 'b-', linewidth=2.5, label='Density curve')

    # Add histogram in background for reference
    ax.hist(transition_durations_filtered, bins=50, density=True, alpha=0.2,
            color='gray', edgecolor='black', linewidth=0.5, label='Histogram (normalized)')

    # Mark key statistics
    y_max = pdf_vals.max()

    # Median
    ax.axvline(median_duration, color='red', linestyle='--', linewidth=2.5,
               label=f'Median: {median_duration:.2f} yrs', alpha=0.8)
    ax.plot(median_duration, kde(median_duration)[0], 'ro', markersize=12,
            markeredgewidth=2, markeredgecolor='white', zorder=10)

    # Mean
    ax.axvline(mean_duration, color='orange', linestyle=':', linewidth=2.5,
               label=f'Mean: {mean_duration:.2f} yrs', alpha=0.8)

    # Quartiles
    ax.axvline(p25, color='purple', linestyle='--', linewidth=1.5,
               label=f'25th percentile: {p25:.2f} yrs', alpha=0.6)
    ax.axvline(p75, color='purple', linestyle='--', linewidth=1.5,
               label=f'75th percentile: {p75:.2f} yrs', alpha=0.6)

    # Add shaded IQR region
    ax.axvspan(p25, p75, alpha=0.15, color='purple', label='Interquartile range')

    # Add difference in medians for comparison
    ax.axvline(diff_medians, color='green', linestyle='-', linewidth=2.5,
               label=f'Difference in medians: {diff_medians:.2f} yrs', alpha=0.8)

    # Add statistics text box
    stats_text = (
        f"DISTRIBUTION STATISTICS:\n"
        f"\n"
        f"Sample size: N = {len(transition_durations_filtered)}\n"
    )

    # Add note about filtered trajectories if any
    if num_negative > 0:
        pct_negative = 100.0 * num_negative / num_total
        stats_text += f"  ({num_negative} trajectories excluded\n"
        stats_text += f"   where {milestone_b} before {milestone_a},\n"
        stats_text += f"   {pct_negative:.1f}% of total)\n"

    stats_text += (
        f"\n"
        f"Central tendency:\n"
        f"  Median: {median_duration:.2f} yrs\n"
        f"  Mean: {mean_duration:.2f} yrs\n"
        f"\n"
        f"Spread:\n"
        f"  IQR: {p75 - p25:.2f} yrs\n"
        f"  Std dev: {np.std(transition_durations_filtered):.2f} yrs\n"
        f"  Min: {transition_durations_filtered.min():.2f} yrs\n"
        f"  Max: {transition_durations_filtered.max():.2f} yrs\n"
        f"\n"
        f"Comparison:\n"
        f"  Median transition: {median_duration:.2f} yrs\n"
        f"  Diff in medians: {diff_medians:.2f} yrs\n"
        f"  Gap: {abs(diff_medians - median_duration):.2f} yrs"
    )

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', linewidth=2),
            family='monospace')

    ax.set_xlabel(f'Transition Duration: {milestone_a} → {milestone_b} (years)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Probability Density Function of Transition Times\n{milestone_a} → {milestone_b}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved transition PDF plot to: {out_path}")


def create_trajectory_map_plot(
    arrival_times_a: np.ndarray,
    arrival_times_b: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    out_path: Path,
) -> None:
    """Create plot showing how individual trajectories map through the transition."""

    median_a = np.median(arrival_times_a)
    median_b = np.median(arrival_times_b)
    median_duration = np.median(transition_durations)

    # Sort by arrival time A for cleaner visualization
    sort_idx = np.argsort(arrival_times_a)
    sorted_a = arrival_times_a[sort_idx]
    sorted_b = arrival_times_b[sort_idx]
    sorted_durations = transition_durations[sort_idx]

    # Sample every nth trajectory to avoid overcrowding (show ~100 trajectories)
    n_show = min(100, len(sorted_a))
    step = max(1, len(sorted_a) // n_show)
    sample_idx = np.arange(0, len(sorted_a), step)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw individual trajectory lines
    for i in sample_idx:
        # Color by duration
        duration_percentile = (sorted_durations[i] - sorted_durations.min()) / (sorted_durations.max() - sorted_durations.min())
        color = plt.cm.viridis(duration_percentile)

        ax.plot([sorted_a[i], sorted_b[i]], [0, 1],
               color=color, alpha=0.3, linewidth=1)

    # Draw thick lines for median trajectory
    # Find trajectory closest to median values
    median_distances = np.abs(arrival_times_a - median_a) + np.abs(arrival_times_b - median_b)
    median_traj_idx = np.argmin(median_distances)

    ax.plot([arrival_times_a[median_traj_idx], arrival_times_b[median_traj_idx]],
           [0, 1], color='red', linewidth=4, alpha=0.8,
           label=f'Example median-ish trajectory\nDuration: {transition_durations[median_traj_idx]:.2f} yrs')

    # Draw vertical lines at medians
    ax.axvline(median_a, ymin=0, ymax=0.45, color='blue', linestyle='--',
              linewidth=3, alpha=0.7)
    ax.axvline(median_b, ymin=0.55, ymax=1, color='orange', linestyle='--',
              linewidth=3, alpha=0.7)

    # Add points at medians
    ax.plot(median_a, 0, 'o', color='blue', markersize=15,
           markeredgewidth=2, markeredgecolor='white', zorder=10,
           label=f'Median {milestone_a}: {median_a:.2f}')
    ax.plot(median_b, 1, 'o', color='orange', markersize=15,
           markeredgewidth=2, markeredgecolor='white', zorder=10,
           label=f'Median {milestone_b}: {median_b:.2f}')

    # Add arrow showing difference in medians
    ax.annotate('', xy=(median_b, 0.5), xytext=(median_a, 0.5),
               arrowprops=dict(arrowstyle='<->', color='red', lw=4, shrinkA=0, shrinkB=0))
    ax.text((median_a + median_b) / 2, 0.52,
           f'Difference in medians:\n{median_b - median_a:.2f} years',
           fontsize=12, color='red', fontweight='bold', ha='center',
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', linewidth=2))

    # Add explanation for median duration
    explanation = (
        f"WHAT THIS SHOWS:\n"
        f"Each line represents one trajectory moving from\n"
        f"{milestone_a} (bottom) to {milestone_b} (top).\n"
        f"\n"
        f"• Median arrival at {milestone_a}: {median_a:.2f}\n"
        f"• Median arrival at {milestone_b}: {median_b:.2f}\n"
        f"• Difference in medians: {median_b - median_a:.2f} yrs\n"
        f"\n"
        f"BUT:\n"
        f"• Median transition duration: {median_duration:.2f} yrs\n"
        f"\n"
        f"The median trajectory (red line) doesn't necessarily\n"
        f"connect the median points! The trajectory that starts\n"
        f"at median {milestone_a} is unlikely to end at median {milestone_b}.\n"
        f"\n"
        f"This is why median(B-A) ≠ median(B) - median(A)."
    )

    ax.text(0.02, 0.98, explanation,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='red', linewidth=2),
           family='monospace')

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f'{milestone_a}\nArrival', f'{milestone_b}\nArrival'], fontsize=11)
    ax.set_title(
        f'Trajectory Flow: How Individual Runs Transit from {milestone_a} to {milestone_b}',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=sorted_durations.min(),
                                                 vmax=sorted_durations.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Transition Duration (years)', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory map plot to: {out_path}")


def create_distribution_fit_comparison_plot(
    arrival_times_a: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    output_path: Path
) -> None:
    """Create side-by-side distributions with Lognormal fit for arrivals and Weibull for transitions.

    Shows distribution of milestone A arrival times (with lognormal fit) and transition durations
    (with Weibull fit), with goodness-of-fit statistics.
    """
    from scipy import stats
    from scipy.special import gamma as gamma_func

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Helper function to fit Lognormal and compute goodness-of-fit
    def fit_lognormal(data, ax, title, xlabel):
        # Fit Lognormal distribution (using MLE)
        # scipy.stats.lognorm uses shape (sigma), loc, scale (exp(mu)) parameterization
        shape, loc, scale = stats.lognorm.fit(data, floc=0)

        # Create histogram
        n_bins = 30
        counts, bin_edges, patches = ax.hist(data, bins=n_bins, density=True,
                                              alpha=0.7, color='steelblue',
                                              edgecolor='black', linewidth=0.5,
                                              label='Observed')

        # Plot fitted Lognormal PDF
        x = np.linspace(data.min(), data.max(), 200)
        pdf = stats.lognorm.pdf(x, shape, loc=0, scale=scale)
        ax.plot(x, pdf, 'r-', linewidth=2.5, label='Lognormal fit')

        # Compute goodness-of-fit statistics
        # 1. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.lognorm.cdf(x, shape, loc=0, scale=scale))

        # 2. R-squared (comparing empirical vs theoretical quantiles)
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.lognorm.ppf(np.linspace(0.01, 0.99, len(sorted_data)),
                                                   shape, loc=0, scale=scale)
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
        r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                        np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

        # 3. AIC/BIC
        log_likelihood = np.sum(stats.lognorm.logpdf(data, shape, loc=0, scale=scale))
        n = len(data)
        k = 2
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Calculate summary statistics
        median_obs = np.median(data)
        mean_obs = np.mean(data)
        # For lognormal: median = exp(mu) = scale, mean = exp(mu + sigma^2/2)
        median_fit = scale
        mean_fit = scale * np.exp(shape**2 / 2)

        # Add vertical lines for median
        ax.axvline(median_obs, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Observed median: {median_obs:.2f}')
        ax.axvline(median_fit, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Fitted median: {median_fit:.2f}')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add goodness-of-fit statistics box
        stats_text = (
            f"Lognormal Fit Quality:\n"
            f"\n"
            f"Parameters:\n"
            f"  μ (log-scale mean) = {np.log(scale):.3f}\n"
            f"  σ (log-scale std) = {shape:.3f}\n"
            f"\n"
            f"Goodness-of-fit:\n"
            f"  R² = {r_squared:.4f}\n"
            f"  KS stat = {ks_stat:.4f}\n"
            f"  KS p-value = {ks_pvalue:.4f}\n"
            f"  AIC = {aic:.1f}\n"
            f"  BIC = {bic:.1f}\n"
            f"\n"
            f"Interpretation:\n"
            f"  R² > 0.95: Excellent fit\n"
            f"  R² > 0.90: Good fit\n"
            f"  R² > 0.80: Fair fit\n"
            f"  R² < 0.80: Poor fit\n"
            f"\n"
            f"KS p-value > 0.05:\n"
            f"  {'Cannot reject' if ks_pvalue > 0.05 else 'Reject'} Lognormal\n"
        )

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1),
                family='monospace')

        return shape, scale, r_squared, ks_stat, ks_pvalue, aic, bic

    # Helper function to fit Weibull and compute goodness-of-fit (for transitions with log scale)
    def fit_weibull_logscale(data, ax, title, xlabel):
        # Fit Weibull distribution (using MLE)
        # scipy.stats.weibull_min uses shape (k), loc, scale (lambda) parameterization
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)

        # Create histogram with log-spaced bins for better visualization
        # Use logarithmically-spaced bins
        log_min = np.log10(max(data.min(), 1e-3))
        log_max = np.log10(data.max())
        bins = np.logspace(log_min, log_max, 25)

        counts, bin_edges, patches = ax.hist(data, bins=bins, density=True,
                                              alpha=0.7, color='steelblue',
                                              edgecolor='black', linewidth=0.5,
                                              label='Observed')

        # Plot fitted Weibull PDF with log-spaced x values
        x = np.logspace(log_min, log_max, 200)
        pdf = stats.weibull_min.pdf(x, shape, loc=0, scale=scale)
        ax.plot(x, pdf, 'r-', linewidth=2.5, label='Weibull fit')

        # Compute goodness-of-fit statistics
        # 1. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale))

        # 2. R-squared (comparing empirical vs theoretical quantiles)
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.weibull_min.ppf(np.linspace(0.01, 0.99, len(sorted_data)),
                                                       shape, loc=0, scale=scale)
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
        r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                        np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

        # 3. AIC/BIC
        log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, loc=0, scale=scale))
        n = len(data)
        k = 2
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Calculate summary statistics
        median_obs = np.median(data)
        mean_obs = np.mean(data)
        median_fit = scale * (np.log(2) ** (1/shape))
        mean_fit = scale * gamma_func(1 + 1/shape)

        # Add vertical lines for median
        ax.axvline(median_obs, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Observed median: {median_obs:.2f}')
        ax.axvline(median_fit, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Fitted median: {median_fit:.2f}')

        # Set log scale for x-axis
        ax.set_xscale('log')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Add goodness-of-fit statistics box
        stats_text = (
            f"Weibull Fit Quality:\n"
            f"\n"
            f"Parameters:\n"
            f"  k (shape) = {shape:.3f}\n"
            f"  λ (scale) = {scale:.3f}\n"
            f"\n"
            f"Goodness-of-fit:\n"
            f"  R² = {r_squared:.4f}\n"
            f"  KS stat = {ks_stat:.4f}\n"
            f"  KS p-value = {ks_pvalue:.4f}\n"
            f"  AIC = {aic:.1f}\n"
            f"  BIC = {bic:.1f}\n"
            f"\n"
            f"Interpretation:\n"
            f"  R² > 0.95: Excellent fit\n"
            f"  R² > 0.90: Good fit\n"
            f"  R² > 0.80: Fair fit\n"
            f"  R² < 0.80: Poor fit\n"
            f"\n"
            f"KS p-value > 0.05:\n"
            f"  {'Cannot reject' if ks_pvalue > 0.05 else 'Reject'} Weibull\n"
        )

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1),
                family='monospace')

        return shape, scale, r_squared, ks_stat, ks_pvalue, aic, bic

    # Plot 1: Milestone A arrival times (Lognormal fit)
    shape_a, scale_a, r2_a, ks_a, ksp_a, aic_a, bic_a = fit_lognormal(
        arrival_times_a, axes[0],
        f'Distribution of {milestone_a} Arrival Time',
        f'{milestone_a} arrival year'
    )

    # Plot 2: Transition durations (Weibull fit with log scale)
    shape_t, scale_t, r2_t, ks_t, ksp_t, aic_t, bic_t = fit_weibull_logscale(
        transition_durations, axes[1],
        f'Distribution of Transition Duration\n({milestone_a} → {milestone_b})',
        'Transition duration (years)'
    )

    # Add overall title
    fig.suptitle(f'Distribution Fit Analysis: {milestone_a} → {milestone_b}',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved distribution fit comparison plot to: {output_path}")


def create_best_fit_comparison_plot(
    arrival_times_a: np.ndarray,
    transition_durations: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    output_path: Path
) -> None:
    """Create side-by-side distributions with best-fit distributions.

    Automatically selects the best distribution for each (arrival time and transition duration)
    based on AIC, and displays that distribution.
    """
    from scipy import stats
    from scipy.special import gamma as gamma_func

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Helper function to fit multiple distributions and select the best
    def fit_best_distribution(data, ax, title, xlabel, use_log_scale=False):
        distributions_to_try = {
            'Weibull': lambda d: stats.weibull_min.fit(d, floc=0),
            'Gamma': lambda d: stats.gamma.fit(d, floc=0),
            'Lognormal': lambda d: stats.lognorm.fit(d, floc=0),
        }

        best_dist = None
        best_aic = np.inf
        best_params = None
        fit_results = {}

        # Fit all distributions
        for name, fit_func in distributions_to_try.items():
            try:
                if name == 'Lognormal':
                    shape, loc, scale = fit_func(data)
                    log_likelihood = np.sum(stats.lognorm.logpdf(data, shape, loc=0, scale=scale))
                    median = np.exp(np.log(scale))
                    params_str = f'μ={np.log(scale):.3f}, σ={shape:.3f}'
                    pdf_func = lambda x: stats.lognorm.pdf(x, shape, loc=0, scale=scale)
                elif name == 'Gamma':
                    shape, loc, scale = fit_func(data)
                    log_likelihood = np.sum(stats.gamma.logpdf(data, shape, loc=0, scale=scale))
                    median = stats.gamma.median(shape, loc=0, scale=scale)
                    params_str = f'k={shape:.3f}, θ={scale:.3f}'
                    pdf_func = lambda x: stats.gamma.pdf(x, shape, loc=0, scale=scale)
                elif name == 'Weibull':
                    shape, loc, scale = fit_func(data)
                    log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, loc=0, scale=scale))
                    median = scale * (np.log(2) ** (1/shape))
                    params_str = f'k={shape:.3f}, λ={scale:.3f}'
                    pdf_func = lambda x: stats.weibull_min.pdf(x, shape, loc=0, scale=scale)

                k = 2  # number of parameters
                aic = 2 * k - 2 * log_likelihood

                fit_results[name] = {
                    'aic': aic,
                    'params': (shape, loc, scale),
                    'params_str': params_str,
                    'median': median,
                    'pdf_func': pdf_func,
                    'log_likelihood': log_likelihood
                }

                if aic < best_aic:
                    best_aic = aic
                    best_dist = name
                    best_params = (shape, loc, scale)

            except Exception as e:
                print(f"Warning: Could not fit {name} distribution: {e}")
                continue

        if best_dist is None:
            print(f"Error: Could not fit any distribution to {title}")
            return None, None, None, None, None, None, None

        # Use the best distribution
        result = fit_results[best_dist]
        shape, loc, scale = result['params']

        # Create histogram
        n_bins = 30
        counts, bin_edges, patches = ax.hist(data, bins=n_bins, density=True,
                                              alpha=0.7, color='steelblue',
                                              edgecolor='black', linewidth=0.5,
                                              label='Observed')

        # Plot fitted PDF
        x_min = data.min()
        x_max = data.max()
        if use_log_scale:
            x = np.logspace(np.log10(max(x_min, 1e-3)), np.log10(x_max), 200)
        else:
            x = np.linspace(x_min, x_max, 200)
        pdf = result['pdf_func'](x)
        ax.plot(x, pdf, 'r-', linewidth=2.5, label=f'{best_dist} fit (best AIC)')

        # Compute goodness-of-fit statistics for best distribution
        if best_dist == 'Lognormal':
            ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.lognorm.cdf(x, shape, loc=0, scale=scale))
        elif best_dist == 'Gamma':
            ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.gamma.cdf(x, shape, loc=0, scale=scale))
        elif best_dist == 'Weibull':
            ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale))

        # Calculate R-squared
        sorted_data = np.sort(data)
        if best_dist == 'Lognormal':
            theoretical_quantiles = stats.lognorm.ppf(np.linspace(0.01, 0.99, len(sorted_data)), shape, loc=0, scale=scale)
        elif best_dist == 'Gamma':
            theoretical_quantiles = stats.gamma.ppf(np.linspace(0.01, 0.99, len(sorted_data)), shape, loc=0, scale=scale)
        elif best_dist == 'Weibull':
            theoretical_quantiles = stats.weibull_min.ppf(np.linspace(0.01, 0.99, len(sorted_data)), shape, loc=0, scale=scale)

        empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
        r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                        np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

        # Add vertical lines for median
        median_obs = np.median(data)
        median_fit = result['median']
        ax.axvline(median_obs, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Obs. median: {median_obs:.2f}')
        ax.axvline(median_fit, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Fit median: {median_fit:.2f}')

        if use_log_scale:
            ax.set_xscale('log')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add goodness-of-fit statistics box
        stats_text = (
            f"{best_dist} Fit (Best by AIC):\n"
            f"\n"
            f"Parameters:\n"
            f"  {result['params_str']}\n"
            f"\n"
            f"Goodness-of-fit:\n"
            f"  R² = {r_squared:.4f}\n"
            f"  KS stat = {ks_stat:.4f}\n"
            f"  KS p = {ks_pvalue:.4f}\n"
            f"  AIC = {result['aic']:.1f}\n"
            f"\n"
            f"All AICs:\n"
        )
        for dist_name in sorted(fit_results.keys(), key=lambda x: fit_results[x]['aic']):
            marker = "✓" if dist_name == best_dist else " "
            stats_text += f"  {marker} {dist_name}: {fit_results[dist_name]['aic']:.1f}\n"

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1),
                family='monospace')

        return shape, scale, r_squared, ks_stat, ks_pvalue, result['aic'], best_dist

    # Plot 1: Milestone A arrival times
    shape_a, scale_a, r2_a, ks_a, ksp_a, aic_a, dist_a = fit_best_distribution(
        arrival_times_a, axes[0],
        f'Distribution of {milestone_a} Arrival Time',
        f'{milestone_a} arrival year',
        use_log_scale=False
    )

    # Plot 2: Transition durations (with log scale)
    shape_t, scale_t, r2_t, ks_t, ksp_t, aic_t, dist_t = fit_best_distribution(
        transition_durations, axes[1],
        f'Distribution of Transition Duration\n({milestone_a} → {milestone_b})',
        'Transition duration (years)',
        use_log_scale=True
    )

    # Add overall title
    fig.suptitle(f'Best-Fit Distribution Analysis: {milestone_a} → {milestone_b}',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved best-fit comparison plot to: {output_path}")


def list_all_milestones(rollouts_file: Path) -> List[str]:
    """List all unique milestone names found in the rollouts file."""
    all_milestones = set()

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

            milestones = results.get("milestones")
            if milestones is not None and isinstance(milestones, dict):
                all_milestones.update(milestones.keys())

    return sorted(all_milestones)


def analyze_transition(rollouts_path: Path, milestone_a: str, milestone_b: str, milestone_c: str, out_dir: Path, create_subfolder: bool = False) -> None:
    """Run full analysis for a single transition pair.

    Args:
        rollouts_path: Path to rollouts.jsonl file
        milestone_a: First milestone name
        milestone_b: Second milestone name
        milestone_c: Third milestone name
        out_dir: Base output directory
        create_subfolder: If True, create a subfolder for this specific transition
    """
    # Create transition-specific subfolder if requested
    if create_subfolder:
        # Create safe folder name from milestone names
        safe_a = milestone_a.replace("-", "_")
        safe_b = milestone_b.replace("-", "_")
        transition_dir = out_dir / f"{safe_a}_to_{safe_b}"
        transition_dir.mkdir(parents=True, exist_ok=True)
        out_dir = transition_dir

    # Extract data - both with and without censored trajectories
    print(f"\nLoading trajectories from: {rollouts_path}")
    print(f"Analyzing transition: {milestone_a} → {milestone_b}")
    print(f"Output directory: {out_dir}")

    # Extract achieved-only data
    arrival_times_a, arrival_times_b, transition_durations, sim_end_times, num_both, num_only_a, num_neither = extract_transition_data(
        rollouts_path,
        milestone_a,
        milestone_b,
        include_censored=False
    )

    # Extract with censored data (includes trajectories where A was achieved but B may not be)
    arrival_times_a_cens, arrival_times_b_cens, transition_durations_cens, sim_end_times_cens, _, _, _ = extract_transition_data(
        rollouts_path,
        milestone_a,
        milestone_b,
        include_censored=True
    )

    # Extract UNCONDITIONAL data (includes ALL trajectories, even where A wasn't achieved)
    # This matches the milestone_summary_table.py calculation
    (arrival_times_a_uncond,
     arrival_times_b_uncond,
     transition_durations_uncond,
     transition_start_times_uncond,
     arrival_hit_flags_uncond,
     transition_hit_flags_uncond,
     sim_end_times_uncond,
     median_a_uncond,
     median_b_uncond,
     median_transition_uncond) = extract_unconditional_data(
        rollouts_path,
        milestone_a,
        milestone_b
    )

    total_trajectories = num_both + num_only_a + num_neither

    if len(arrival_times_a) == 0:
        print(f"ERROR: No trajectories found that achieve both milestones")
        print(f"Make sure both '{milestone_a}' and '{milestone_b}' exist in the data")
        return

    print(f"Total trajectories: {total_trajectories}")
    print(f"  Both milestones achieved: {num_both}")
    print(f"  Only {milestone_a} achieved: {num_only_a}")
    print(f"  Neither achieved: {num_neither}")

    # Convert to numpy arrays - achieved only
    arrival_times_a = np.array(arrival_times_a)
    arrival_times_b = np.array(arrival_times_b)
    transition_durations = np.array(transition_durations)
    sim_end_times = np.array(sim_end_times)

    # Convert to numpy arrays - with censored
    arrival_times_a_cens = np.array(arrival_times_a_cens)
    arrival_times_b_cens = np.array(arrival_times_b_cens)
    transition_durations_cens = np.array(transition_durations_cens)
    sim_end_times_cens = np.array(sim_end_times_cens)

    # Calculate and display the paradox - achieved only
    median_a = np.median(arrival_times_a)
    median_b = np.median(arrival_times_b)
    median_duration = np.median(transition_durations)
    diff_in_medians = median_b - median_a

    # Calculate and display the paradox - with censored (conditional on A achieved)
    median_a_cens = np.median(arrival_times_a_cens)
    median_b_cens = np.median(arrival_times_b_cens)
    median_duration_cens = np.median(transition_durations_cens)
    diff_in_medians_cens = median_b_cens - median_a_cens

    # Calculate unconditional difference (matches milestone_summary_table.py)
    diff_in_medians_uncond = median_b_uncond - median_a_uncond

    print("\n" + "=" * 70)
    print("THE MEDIAN DIFFERENCE PARADOX")
    print("=" * 70)
    print(f"\n--- ACHIEVED ONLY (both milestones reached) ---")
    print(f"N = {num_both} trajectories")
    print(f"Median arrival at {milestone_a}: {median_a:.2f}")
    print(f"Median arrival at {milestone_b}: {median_b:.2f}")
    print(f"Difference in medians: {diff_in_medians:.2f} years")
    print(f"Median transition duration: {median_duration:.2f} years")
    print(f"GAP: {abs(diff_in_medians - median_duration):.2f} years")

    if num_only_a > 0 or num_neither > 0:
        print(f"\n--- CONDITIONAL ON A ACHIEVED (A achieved, B may be censored) ---")
        print(f"N = {num_both + num_only_a} trajectories")
        print(f"  {num_both} have both milestones")
        print(f"  {num_only_a} have only {milestone_a} (censored {milestone_b} at sim end)")
        print(f"Median arrival at {milestone_a}: {median_a_cens:.2f}")
        print(f"Median arrival at {milestone_b}: {median_b_cens:.2f}")
        print(f"Difference in medians: {diff_in_medians_cens:.2f} years")
        print(f"Median transition duration: {median_duration_cens:.2f} years")
        print(f"GAP: {abs(diff_in_medians_cens - median_duration_cens):.2f} years")

        print(f"\n--- UNCONDITIONAL (all trajectories, matches milestone_summary_table.py) ---")
        print(f"N = {total_trajectories} trajectories (includes {num_neither} where neither achieved)")
        print(f"Median arrival at {milestone_a}: {median_a_uncond:.2f}")
        print(f"Median arrival at {milestone_b}: {median_b_uncond:.2f}")
        print(f"Difference in medians: {diff_in_medians_uncond:.2f} years")
        print(f"Median transition duration: {median_transition_uncond:.2f} years")
        print(f"GAP: {abs(diff_in_medians_uncond - median_transition_uncond):.2f} years")
    print("=" * 70)
    print("")

    # Create visualizations
    print("Generating visualizations for ACHIEVED ONLY...")

    # 1. Scatter plot
    scatter_path = out_dir / f"median_paradox_scatter_{milestone_a}_to_{milestone_b}.png"
    create_scatter_plot(
        arrival_times_a, arrival_times_b, transition_durations,
        milestone_a, milestone_b, scatter_path,
        diff_in_medians_cond=diff_in_medians_cens,
        diff_in_medians_uncond=diff_in_medians_uncond
    )

    # 2. Quantile comparison
    quantile_path = out_dir / f"median_paradox_quantiles_{milestone_a}_to_{milestone_b}.png"
    create_quantile_comparison_plot(
        arrival_times_a, arrival_times_b, transition_durations,
        milestone_a, milestone_b, quantile_path, sim_end_times
    )

    # 3. Distribution comparison
    dist_path = out_dir / f"median_paradox_distributions_{milestone_a}_to_{milestone_b}.png"
    create_distribution_comparison_plot(
        arrival_times_a, arrival_times_b, transition_durations,
        milestone_a, milestone_b, dist_path,
        diff_in_medians_cond=diff_in_medians_cens,
        diff_in_medians_uncond=diff_in_medians_uncond
    )

    # 4. Trajectory flow map
    trajectory_path = out_dir / f"median_paradox_trajectories_{milestone_a}_to_{milestone_b}.png"
    create_trajectory_map_plot(
        arrival_times_a, arrival_times_b, transition_durations,
        milestone_a, milestone_b, trajectory_path
    )

    # 5. Arrival vs duration plot - use CENSORED data
    arrival_duration_path = out_dir / f"median_paradox_arrival_vs_duration_{milestone_a}_to_{milestone_b}.png"
    create_arrival_vs_duration_plot(
        arrival_times_a_cens, transition_durations_cens,
        milestone_a, milestone_b, arrival_duration_path
    )

    # 6. Correlation comparison plot (independent vs actual vs perfect correlation) - use CENSORED data
    correlation_path = out_dir / f"median_paradox_correlation_comparison_{milestone_a}_to_{milestone_b}.png"
    create_correlation_comparison_plot(
        arrival_times_a_cens, arrival_times_b_cens, transition_durations_cens,
        milestone_a, milestone_b, correlation_path
    )

    # 7. Correlation vs median difference analysis (mix: unconditional medians, censored correlation)
    corr_analysis_path = out_dir / f"median_paradox_correlation_analysis_{milestone_a}_to_{milestone_b}.png"
    create_correlation_vs_median_difference_plot(
        arrival_times_a_uncond,
        transition_durations_uncond,
        transition_start_times_uncond,
        arrival_hit_flags_uncond,
        transition_hit_flags_uncond,
        sim_end_times_uncond,
        arrival_times_a_cens,
        transition_durations_cens,
        milestone_a,
        milestone_b,
        corr_analysis_path,
        median_a_uncond,
        median_b_uncond,
        median_transition_uncond,
        diff_in_medians_cens
    )

    # 7b. V2 version with clearer interpretation
    corr_analysis_v2_path = out_dir / f"median_paradox_correlation_analysis_v2_{milestone_a}_to_{milestone_b}.png"
    create_correlation_vs_median_difference_plot_v2(
        arrival_times_a_uncond, transition_durations_uncond,
        arrival_times_a_cens, transition_durations_cens,
        milestone_a, milestone_b, corr_analysis_v2_path,
        median_a_uncond, median_b_uncond, median_transition_uncond
    )

    # 8. AC vs SAR simple scatter plot
    ac_sar_path = out_dir / f"ac_vs_sar_{milestone_a}_to_{milestone_b}.png"
    create_ac_vs_sar_scatter_plot(
        arrival_times_a, arrival_times_b,
        milestone_a, milestone_b, ac_sar_path
    )

    # 9. AC time vs SAR→SIAR duration plot
    ac_sar_siar_path = out_dir / f"ac_vs_sar_siar_duration_{milestone_a}_{milestone_b}_{milestone_c}.png"
    create_ac_vs_sar_siar_duration_plot(
        rollouts_path,
        milestone_a, milestone_b, milestone_c,
        ac_sar_siar_path
    )

    # 10. Transition time PDF
    pdf_path = out_dir / f"median_paradox_transition_pdf_{milestone_a}_to_{milestone_b}.png"
    create_transition_pdf_plot(
        arrival_times_a, arrival_times_b, transition_durations,
        milestone_a, milestone_b, pdf_path
    )

    # 11. Distribution fit comparison plot (saved in parent directory)
    parent_dir = out_dir.parent if create_subfolder else out_dir
    dist_fit_path = parent_dir / f"distribution_fit_comparison_{milestone_a}_to_{milestone_b}.png"
    create_distribution_fit_comparison_plot(
        arrival_times_a, transition_durations,
        milestone_a, milestone_b, dist_fit_path
    )

    # 12. Best-fit distribution comparison plot
    best_fit_path = out_dir / f"best_fit_comparison_{milestone_a}_to_{milestone_b}.png"
    create_best_fit_comparison_plot(
        arrival_times_a, transition_durations,
        milestone_a, milestone_b, best_fit_path
    )

    # Generate plots for censored data if there are any censored trajectories
    if num_only_a > 0 or num_neither > 0:
        print(f"\nGenerating visualizations for INCLUDING CENSORED data...")

        # 1. Scatter plot (censored)
        scatter_path_cens = out_dir / f"median_paradox_scatter_{milestone_a}_to_{milestone_b}_censored.png"
        create_scatter_plot(
            arrival_times_a_cens, arrival_times_b_cens, transition_durations_cens,
            milestone_a, milestone_b + " (censored)", scatter_path_cens
        )

        # 2. Quantile comparison (censored)
        quantile_path_cens = out_dir / f"median_paradox_quantiles_{milestone_a}_to_{milestone_b}_censored.png"
        create_quantile_comparison_plot(
            arrival_times_a_cens, arrival_times_b_cens, transition_durations_cens,
            milestone_a, milestone_b + " (censored)", quantile_path_cens, sim_end_times_cens
        )

        # 3. Distribution comparison (censored)
        dist_path_cens = out_dir / f"median_paradox_distributions_{milestone_a}_to_{milestone_b}_censored.png"
        create_distribution_comparison_plot(
            arrival_times_a_cens, arrival_times_b_cens, transition_durations_cens,
            milestone_a, milestone_b + " (censored)", dist_path_cens
        )

        # 4. Trajectory flow map (censored)
        trajectory_path_cens = out_dir / f"median_paradox_trajectories_{milestone_a}_to_{milestone_b}_censored.png"
        create_trajectory_map_plot(
            arrival_times_a_cens, arrival_times_b_cens, transition_durations_cens,
            milestone_a, milestone_b + " (censored)", trajectory_path_cens
        )

        # 5. Arrival vs duration plot (censored)
        arrival_duration_path_cens = out_dir / f"median_paradox_arrival_vs_duration_{milestone_a}_to_{milestone_b}_censored.png"
        create_arrival_vs_duration_plot(
            arrival_times_a_cens, transition_durations_cens,
            milestone_a, milestone_b + " (censored)", arrival_duration_path_cens
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if num_only_a > 0 or num_neither > 0:
        print(f"Generated 17 visualization plots in: {out_dir}")
        print(f"  - 12 plots for achieved-only data ({num_both} trajectories)")
        print(f"  - 5 plots for data including censored trajectories ({total_trajectories} total)")
        print(f"    ({num_only_a + num_neither} censored: {num_only_a} only-A, {num_neither} neither)")
    else:
        print(f"Generated 12 visualization plots in: {out_dir}")
    print(f"")
    print(f"These plots explain why:")
    print(f"  median({milestone_b}) - median({milestone_a}) = {diff_in_medians:.2f}")
    print(f"  ≠")
    print(f"  median({milestone_b} - {milestone_a}) = {median_duration:.2f}")
    print(f"")
    print(f"The key insight: Medians don't distribute over subtraction when")
    print(f"the variables are correlated. Trajectories that reach {milestone_a}")
    print(f"early tend to reach {milestone_b} early too, but the trajectory")
    print(f"starting at median({milestone_a}) doesn't necessarily end at")
    print(f"median({milestone_b}).")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Explain why median(B) - median(A) ≠ median(B - A) with visualizations",
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
        "--milestone-a",
        type=str,
        default=None,
        help="First milestone name (if not specified, runs both AC→SAR and SAR→SIAR)"
    )
    parser.add_argument(
        "--milestone-b",
        type=str,
        default=None,
        help="Second milestone name (if not specified, runs both AC→SAR and SAR→SIAR)"
    )
    parser.add_argument(
        "--milestone-c",
        type=str,
        default="SIAR-level-experiment-selection-skill",
        help="Third milestone name for AC vs SAR→SIAR plot (default: SIAR-level-experiment-selection-skill)"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run analysis for both AC→SAR and SAR→SIAR transitions (default behavior if no milestones specified)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as rollouts file)"
    )
    parser.add_argument(
        "--list-milestones",
        action="store_true",
        help="List available milestones and exit"
    )

    args = parser.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # List milestones if requested
    if args.list_milestones:
        print(f"Scanning {rollouts_path} for milestones...")
        milestones = list_all_milestones(rollouts_path)
        if milestones:
            print(f"\nFound {len(milestones)} unique milestones:")
            for m in milestones:
                print(f"  - {m}")
        else:
            print("No milestones found in rollouts file")
        return

    # Determine output directory - ALWAYS use median_paradox_plots subfolder
    if args.out_dir:
        base_dir = Path(args.out_dir)
    else:
        base_dir = rollouts_path.parent

    # Always create median_paradox_plots subfolder
    out_dir = base_dir / "median_paradox_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which transitions to run
    if args.milestone_a is None or args.milestone_b is None or args.run_all:
        # Run both AC→SAR and SAR→SIAR
        transitions = [
            ("AC", "SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            ("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill")
        ]
        print("\n" + "=" * 70)
        print("RUNNING ANALYSIS FOR MULTIPLE TRANSITIONS")
        print("=" * 70)
        print("Will analyze:")
        for ma, mb, _ in transitions:
            print(f"  - {ma} → {mb}")
        print("=" * 70)

        # Create separate subfolders for each transition
        for milestone_a, milestone_b, milestone_c in transitions:
            analyze_transition(rollouts_path, milestone_a, milestone_b, milestone_c, out_dir, create_subfolder=True)
    else:
        # Run single specified transition (no subfolder)
        analyze_transition(rollouts_path, args.milestone_a, args.milestone_b, args.milestone_c, out_dir, create_subfolder=False)


if __name__ == "__main__":
    main()
