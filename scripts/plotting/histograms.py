"""
Histogram plotting functions for rollout analysis.

This module contains all histogram-related plotting functions extracted from
plot_rollouts.py, including milestone time histograms, effective compute
histograms, and other distribution plots.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import utilities
import sys
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

from plotting_utils.kde import make_gaussian_kde, make_gamma_kernel_kde
from plotting_utils.helpers import (
    decimal_year_to_date_string as _decimal_year_to_date_string,
    format_time_duration as _format_time_duration,
    get_time_tick_values_and_labels as _get_time_tick_values_and_labels,
    load_present_day,
)


def plot_milestone_time_histogram_cdf(
    times: List[float],
    num_not_achieved: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    milestone_label: Optional[str] = None,
    sim_end: Optional[float] = None,
    max_year: Optional[float] = None
) -> None:
    """Plot CDF-style histogram of milestone arrival times.

    Bar heights represent cumulative count of rollouts reaching milestone BY that year.

    Args:
        times: list of arrival times (only for rollouts that achieved the milestone)
        num_not_achieved: count of rollouts where milestone was not achieved
        milestone_label: name of the milestone for labeling
        sim_end: typical simulation end time for labeling "not achieved" bar
        max_year: maximum year to display on x-axis (default: no limit)
    """
    total_n = int(len(times) + max(0, num_not_achieved))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(times, dtype=float) if len(times) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    bin_width = 1.0

    if len(data) > 0:
        # Create bins for the histogram
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.0,  # Invisible, just to get bin edges
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0

        # Compute cumulative counts for each bin
        cumulative_counts = np.cumsum(counts)

        # Plot bars with cumulative heights
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, cumulative_counts, width=bin_width, edgecolor="black", alpha=0.6, label="CDF Histogram")

        # Place the Not Achieved bar at the simulation end time if available
        if sim_end is not None:
            no_x = float(sim_end)
        else:
            no_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only Not Achieved data: use sim_end if available, else arbitrary position
        if sim_end is not None:
            no_x = float(sim_end)
            bin_width = 1.0
        else:
            no_x = 1.0
            bin_width = 1.0

    # Calculate ymax for annotations
    ymax = float(len(data)) if len(data) > 0 else float(num_not_achieved)

    # Draw the Not Achieved bar if needed (same height as if all achieved)
    sim_end_year = int(np.round(sim_end)) if sim_end is not None else None
    if num_not_achieved > 0 and len(data) > 0:
        # The "not achieved" bar shows total rollouts (all achieved by sim_end or not achieved at all)
        ax.bar(no_x, total_n, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0])
        right = float(no_x + bin_width / 2.0)
        # Apply max_year limit if specified
        if max_year is not None:
            right = min(right, max_year)
        ax.set_xlim(left, right)
    elif num_not_achieved > 0:
        # Only not achieved data
        ax.bar(no_x, num_not_achieved, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = no_x - bin_width
        right = float(no_x + bin_width / 2.0)
        # Apply max_year limit if specified
        if max_year is not None:
            right = min(right, max_year)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "not achieved" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        # Apply max_year limit if specified
        if max_year is not None:
            right = min(right, max_year)
        ax.set_xlim(left, right)

    # Percentiles and annotations (including not achieved as simulation end)
    if total_n > 0:
        y_annot = max(ymax, total_n) * 0.95

        # Create combined data including "not achieved" at simulation end
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_not_achieved, sim_end)])
        else:
            combined_data = data

        if len(combined_data) > 0:
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Determine position and label for each percentile
            def _percentile_pos_label(qv: float) -> Tuple[float, str]:
                # If percentile is at or beyond simulation end, it's "not achieved"
                if sim_end is not None and qv >= sim_end - 0.01:  # small tolerance
                    sim_end_year = int(np.round(sim_end))
                    return float(no_x), f"Not achieved by {sim_end_year}"
                return float(qv), _decimal_year_to_date_string(float(qv))

            x10, lbl10 = _percentile_pos_label(q10)
            x50, lbl50 = _percentile_pos_label(q50)
            x90, lbl90 = _percentile_pos_label(q90)

            plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
            plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
            plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

            plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("Arrival Time (decimal year)")
    plt.ylabel("Cumulative Count")
    ttl = title or (f"Cumulative Distribution of Arrival Time: {milestone_label}" if milestone_label else "Cumulative Distribution of Milestone Arrival Time")
    plt.title(ttl)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_milestone_time_histogram(
    times: List[float],
    num_not_achieved: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    milestone_label: Optional[str] = None,
    sim_end: Optional[float] = None,
    max_year: Optional[float] = None,
    present_day: Optional[float] = None
) -> None:
    """Plot histogram of milestone arrival times.

    Args:
        times: list of arrival times (only for rollouts that achieved the milestone)
        num_not_achieved: count of rollouts where milestone was not achieved
        milestone_label: name of the milestone for labeling
        sim_end: typical simulation end time for labeling "not achieved" bar
        max_year: maximum year to display on x-axis (default: no limit)
        present_day: lower bound for gamma kernel KDE (defaults to min(times) if not provided)
    """
    total_n = int(len(times) + max(0, num_not_achieved))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(times, dtype=float) if len(times) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    kde_counts = np.asarray([], dtype=float)
    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    counts = np.asarray([], dtype=float)
    bin_width = 1.0
    no_x = None

    if len(data) > 0:
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.6,
            label="Histogram",
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        if len(data) >= 2:
            # Determine lower bound for gamma kernel KDE
            lower_bound = present_day if present_day is not None else data.min()

            xs = np.linspace(data.min(), data.max(), 512)
            try:
                # Use gamma kernel KDE for milestone times (proper boundary handling)
                kde = make_gamma_kernel_kde(data, lower_bound=lower_bound)
                kde_counts = kde(xs) * len(data) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gamma Kernel KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the Not Achieved bar at the simulation end time if available
        if sim_end is not None:
            no_x = float(sim_end)
        else:
            no_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only Not Achieved data: use sim_end if available, else arbitrary position
        if sim_end is not None:
            no_x = float(sim_end)
            bin_width = 1.0
        else:
            no_x = 1.0
            bin_width = 1.0

    # Calculate ymax for annotations
    ymax_hist = float(np.max(counts) if counts.size else 0.0)
    ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
    ymax = max(ymax_hist, ymax_kde, float(num_not_achieved), 1.0)

    # Draw the Not Achieved bar if needed
    sim_end_year = int(np.round(sim_end)) if sim_end is not None else None
    if num_not_achieved > 0:
        ax.bar(no_x, num_not_achieved, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0]) if len(data) > 0 else (no_x - bin_width)
        right = float(no_x + bin_width / 2.0)
        # Apply max_year limit if specified
        if max_year is not None:
            right = min(right, max_year)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "not achieved" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        # Apply max_year limit if specified
        if max_year is not None:
            right = min(right, max_year)
        ax.set_xlim(left, right)

    # Percentiles and annotations (including not achieved as simulation end)
    if total_n > 0:
        y_annot = ymax * 0.95

        # Create combined data including "not achieved" at simulation end
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_not_achieved, sim_end)])
        else:
            combined_data = data

        if len(combined_data) > 0:
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Determine position and label for each percentile
            def _percentile_pos_label(qv: float) -> Tuple[float, str]:
                # If percentile is at or beyond simulation end, it's "not achieved"
                if sim_end is not None and qv >= sim_end - 0.01:  # small tolerance
                    sim_end_year = int(np.round(sim_end))
                    return float(no_x), f"Not achieved by {sim_end_year}"
                return float(qv), _decimal_year_to_date_string(float(qv))

            x10, lbl10 = _percentile_pos_label(q10)
            x50, lbl50 = _percentile_pos_label(q50)
            x90, lbl90 = _percentile_pos_label(q90)

            plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
            plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
            plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

            plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("Arrival Time (decimal year)")
    plt.ylabel("Count")
    ttl = title or (f"Distribution of Arrival Time: {milestone_label}" if milestone_label else "Distribution of Milestone Arrival Time")
    plt.title(ttl)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_milestone_effective_compute_histogram(
    compute_ooms: List[float],
    num_not_achieved: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    milestone_label: Optional[str] = None
) -> None:
    """Plot histogram of effective compute (in OOMs) required to achieve a milestone.

    Args:
        compute_ooms: list of effective compute values in OOMs (only for rollouts that achieved the milestone)
        num_not_achieved: count of rollouts where milestone was not achieved
        milestone_label: name of the milestone for labeling
    """
    total_n = int(len(compute_ooms) + max(0, num_not_achieved))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(compute_ooms, dtype=float) if len(compute_ooms) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    kde_counts = np.asarray([], dtype=float)
    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    counts = np.asarray([], dtype=float)
    bin_width = 1.0
    no_x = None

    if len(data) > 0:
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.6,
            label="Histogram",
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        if len(data) >= 2:
            xs = np.linspace(data.min(), data.max(), 512)
            try:
                kde = make_gaussian_kde(data)
                kde_counts = kde(xs) * len(data) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the Not Achieved bar at the right edge
        no_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only Not Achieved data: use arbitrary position
        no_x = 1.0
        bin_width = 1.0

    # Calculate ymax for annotations
    ymax_hist = float(np.max(counts) if counts.size else 0.0)
    ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
    ymax = max(ymax_hist, ymax_kde, float(num_not_achieved), 1.0)

    # Draw the Not Achieved bar if needed
    if num_not_achieved > 0:
        ax.bar(no_x, num_not_achieved, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0]) if len(data) > 0 else (no_x - bin_width)
        right = float(no_x + bin_width / 2.0)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "not achieved" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        ax.set_xlim(left, right)

    # Percentiles and annotations
    if len(data) > 0:
        y_annot = ymax * 0.95
        q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])

        plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
        plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
        plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

        plt.text(q10, y_annot, f"P10: {q10:.1f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
        plt.text(q50, y_annot, f"Median: {q50:.1f}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
        plt.text(q90, y_annot, f"P90: {q90:.1f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("log(2025-FLOP)")
    plt.ylabel("Count")
    ttl = title or (f"Distribution of Effective Compute at {milestone_label}" if milestone_label else "Distribution of Effective Compute at Milestone")
    plt.title(ttl)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_horizon_at_sc_histogram(
    values: List[float],
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None
) -> None:
    """Plot histogram of horizon length at AC (in minutes, log scale).

    Args:
        values: list of horizon length values in minutes
        out_path: output file path
        bins: number of histogram bins
        title: optional plot title
    """
    if len(values) == 0:
        raise ValueError("No horizon_at_sc values found to plot")
    data = np.asarray(values, dtype=float)
    # Define log-spaced bins between min and cap
    cap = float(120000 * 52 * 40 * 60)
    data = np.clip(data, 0.001, cap)
    xmin = max(data[data > 0].min(), 0.001)
    xmax = max(data.max(), xmin * 1.01)
    bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), int(bins))

    plt.figure(figsize=(10, 6))
    counts, _, _ = plt.hist(data, bins=bin_edges, edgecolor="black", alpha=0.6, label="Histogram")

    # Percentiles and annotations
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])
    ymax = float(np.max(counts) if counts.size else 1.0)
    y_annot = ymax * 0.95
    plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
    plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
    plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")
    plt.text(q10, y_annot, f"P10: {_format_time_duration(q10)}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q50, y_annot, f"Median: {_format_time_duration(q50)}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q90, y_annot, f"P90: {_format_time_duration(q90)}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Axis formatting: log x with custom ticks
    plt.xscale("log")
    ticks, labels = _get_time_tick_values_and_labels()
    plt.xticks(ticks, labels, rotation=0)
    plt.xlabel("Horizon at AC (minutes)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of Horizon Length at AC")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_aa_time_histogram_cdf(
    aa_times: List[float],
    num_no_sc: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    sim_end: Optional[float] = None
) -> None:
    """Plot CDF-style histogram of AC arrival times.

    Bar heights represent cumulative count of rollouts reaching AC BY that year.

    Args:
        aa_times: list of SC arrival times (only for rollouts that achieved SC)
        num_no_sc: count of rollouts where SC was not achieved
        sim_end: typical simulation end time for labeling "not achieved" bar
    """
    total_n = int(len(aa_times) + max(0, num_no_sc))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(aa_times, dtype=float) if len(aa_times) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    bin_width = 1.0
    nosc_x = None

    if len(data) > 0:
        # Create bins for the histogram
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.0,  # Invisible, just to get bin edges
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0

        # Compute cumulative counts for each bin
        cumulative_counts = np.cumsum(counts)

        # Plot bars with cumulative heights
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, cumulative_counts, width=bin_width, edgecolor="black", alpha=0.6, label="CDF Histogram")

        # Place the No SC bar at the simulation end time if available
        if sim_end is not None:
            nosc_x = float(sim_end)
        else:
            nosc_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only No SC data: use sim_end if available, else arbitrary position
        if sim_end is not None:
            nosc_x = float(sim_end)
            bin_width = 1.0
        else:
            nosc_x = 1.0
            bin_width = 1.0

    # Calculate ymax for annotations
    ymax = float(len(data)) if len(data) > 0 else float(num_no_sc)

    # Draw the No SC bar if needed (same height as if all achieved)
    sim_end_year = int(np.round(sim_end)) if sim_end is not None else None
    if num_no_sc > 0 and len(data) > 0:
        # The "no SC" bar shows total rollouts (all achieved by sim_end or not achieved at all)
        ax.bar(nosc_x, total_n, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0])
        right = float(nosc_x + bin_width / 2.0)
        ax.set_xlim(left, right)
    elif num_no_sc > 0:
        # Only No SC data
        ax.bar(nosc_x, num_no_sc, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = nosc_x - bin_width
        right = float(nosc_x + bin_width / 2.0)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "no SC" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        ax.set_xlim(left, right)

    # Percentiles and annotations (including not achieved as simulation end)
    if total_n > 0:
        y_annot = max(ymax, total_n) * 0.95

        # Create combined data including "not achieved" at simulation end
        if num_no_sc > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_no_sc, sim_end)])
        else:
            combined_data = data

        if len(combined_data) > 0:
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Determine position and label for each percentile
            def _percentile_pos_label(qv: float) -> Tuple[float, str]:
                # If percentile is at or beyond simulation end, it's "not achieved"
                if sim_end is not None and qv >= sim_end - 0.01:  # small tolerance
                    sim_end_year = int(np.round(sim_end))
                    return float(nosc_x), f"Not achieved by {sim_end_year}"
                return float(qv), _decimal_year_to_date_string(float(qv))

            x10, lbl10 = _percentile_pos_label(q10)
            x50, lbl50 = _percentile_pos_label(q50)
            x90, lbl90 = _percentile_pos_label(q90)

            plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
            plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
            plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

            plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("AC Time (decimal year)")
    plt.ylabel("Cumulative Count")
    plt.title(title or "Cumulative Distribution of AC Times")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_aa_time_histogram(
    aa_times: List[float],
    num_no_sc: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    sim_end: Optional[float] = None
) -> None:
    """Plot histogram of AC arrival times.

    Args:
        aa_times: list of SC arrival times (only for rollouts that achieved SC)
        num_no_sc: count of rollouts where SC was not achieved
        sim_end: typical simulation end time for labeling "not achieved" bar
    """
    total_n = int(len(aa_times) + max(0, num_no_sc))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(aa_times, dtype=float) if len(aa_times) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    kde_counts = np.asarray([], dtype=float)
    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    counts = np.asarray([], dtype=float)
    bin_width = 1.0
    nosc_x = None

    if len(data) > 0:
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.6,
            label="Histogram",
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        if len(data) >= 2:
            xs = np.linspace(data.min(), data.max(), 512)
            try:
                kde = make_gaussian_kde(data)
                kde_counts = kde(xs) * len(data) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the No SC bar at the simulation end time if available
        if sim_end is not None:
            nosc_x = float(sim_end)
        else:
            nosc_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only No SC data: use sim_end if available, else arbitrary position
        if sim_end is not None:
            nosc_x = float(sim_end)
            bin_width = 1.0
        else:
            nosc_x = 1.0
            bin_width = 1.0

    # Calculate ymax for annotations
    ymax_hist = float(np.max(counts) if counts.size else 0.0)
    ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
    ymax = max(ymax_hist, ymax_kde, float(num_no_sc), 1.0)

    # Draw the No SC bar if needed
    sim_end_year = int(np.round(sim_end)) if sim_end is not None else None
    if num_no_sc > 0:
        ax.bar(nosc_x, num_no_sc, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0]) if len(data) > 0 else (nosc_x - bin_width)
        right = float(nosc_x + bin_width / 2.0)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "no SC" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        ax.set_xlim(left, right)

    # Percentiles and annotations (including not achieved as simulation end)
    if total_n > 0:
        y_annot = ymax * 0.95

        # Create combined data including "not achieved" at simulation end
        if num_no_sc > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_no_sc, sim_end)])
        else:
            combined_data = data

        if len(combined_data) > 0:
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Determine position and label for each percentile
            def _percentile_pos_label(qv: float) -> Tuple[float, str]:
                # If percentile is at or beyond simulation end, it's "not achieved"
                if sim_end is not None and qv >= sim_end - 0.01:  # small tolerance
                    sim_end_year = int(np.round(sim_end))
                    return float(nosc_x), f"Not achieved by {sim_end_year}"
                return float(qv), _decimal_year_to_date_string(float(qv))

            x10, lbl10 = _percentile_pos_label(q10)
            x50, lbl50 = _percentile_pos_label(q50)
            x90, lbl90 = _percentile_pos_label(q90)

            plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
            plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
            plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

            plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("AC Time (decimal year)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of AC Times")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_x_years_in_1_year_histogram(
    x_values: List[float],
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None
) -> None:
    """Plot histogram of 'X years in 1 year' values.

    Args:
        x_values: list of X values (ratio of max OOMs/year to current OOMs/year)
        out_path: output file path
        bins: number of histogram bins
        title: optional plot title
    """
    if len(x_values) == 0:
        raise ValueError("No X values found to plot")

    data = np.asarray(x_values, dtype=float)

    # Separate values ≤50 and >50
    data_in_range = data[data <= 50]
    num_above_50 = int(np.sum(data > 50))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Create log-spaced bins from min to 50
    xmin = max(data_in_range[data_in_range > 0].min(), 1.0) if len(data_in_range) > 0 else 1.0
    xmax = 50.0
    bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), int(bins))

    if len(data_in_range) > 0:
        counts, _, _ = plt.hist(data_in_range, bins=bin_edges, edgecolor="black", alpha=0.6, label="Histogram")
        bin_width_log = (np.log10(xmax) - np.log10(xmin)) / bins

        # Add ">50" bar at the right edge if needed
        if num_above_50 > 0:
            # Place at 50 * 1.2 to be visually distinct
            above_50_x = xmax * 1.2
            above_50_width_log = bin_width_log  # Match the bin width in log space
            ax.bar(above_50_x, num_above_50, width=above_50_x * (10**above_50_width_log - 1), edgecolor="black", alpha=0.6, color="tab:red", label=f">50 (n={num_above_50})")

        # Percentiles (only for data ≤50)
        if len(data_in_range) > 0:
            q10, q50, q90 = np.quantile(data_in_range, [0.1, 0.5, 0.9])
            ymax = float(np.max(counts) if counts.size else 1.0)
            y_annot = ymax * 0.95

            plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
            plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
            plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

            plt.text(q10, y_annot, f"P10: {q10:.1f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(q50, y_annot, f"Median: {q50:.1f}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(q90, y_annot, f"P90: {q90:.1f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
    elif num_above_50 > 0:
        # Only >50 data
        above_50_x = xmax * 1.2
        ax.bar(above_50_x, num_above_50, width=above_50_x * 0.2, edgecolor="black", alpha=0.6, color="tab:red", label=f">50 (n={num_above_50})")

    plt.xscale("log")
    plt.xlabel("X years of progress in 1 year")
    plt.ylabel("Count")
    plt.title(title or "Distribution of 'X years in 1 year'")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_takeoff_exponent_histogram(
    takeoff_exponent_values: List[float],
    num_skipped: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None
) -> None:
    """Plot histogram of takeoff exponent 2^(1 - beta/m) values.

    Args:
        takeoff_exponent_values: list of 2^(1 - beta/m) values
        num_skipped: count of rollouts where the exponent could not be computed
        out_path: output file path
        bins: number of histogram bins
        title: optional plot title
    """
    if len(takeoff_exponent_values) == 0:
        raise ValueError("No takeoff exponent values found to plot")

    data = np.asarray(takeoff_exponent_values, dtype=float)
    # Filter to positive finite values
    data = data[(data > 0) & np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("No positive finite takeoff exponent values found to plot")

    # Use log-spaced bins for log scale x-axis
    log_data = np.log10(data)
    log_bins = np.linspace(log_data.min(), log_data.max(), bins + 1)
    bin_edges_linear = 10 ** log_bins

    plt.figure(figsize=(10, 6))

    counts, bin_edges, _ = plt.hist(data, bins=bin_edges_linear, edgecolor="black", alpha=0.6, label="Histogram")
    plt.xscale('log')

    # Add KDE on log-transformed data
    if len(data) >= 2:
        xs_log = np.linspace(log_data.min(), log_data.max(), 512)
        xs = 10 ** xs_log
        try:
            kde = make_gaussian_kde(log_data)
            # Scale KDE to match histogram counts (approximate)
            kde_vals = kde(xs_log)
            # Normalize to match histogram area
            bin_widths = np.diff(bin_edges_linear)
            hist_area = np.sum(counts * bin_widths)
            kde_area = np.trapezoid(kde_vals, xs)
            kde_counts = kde_vals * hist_area / kde_area if kde_area > 0 else kde_vals
            plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE (log)")
        except Exception:
            pass

    # Percentile lines
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])
    ymax = float(np.max(counts) if counts.size else 1.0)
    y_annot = ymax * 0.95

    plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
    plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
    plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

    plt.text(q10, y_annot, f"P10: {q10:.3f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q50, y_annot, f"Median: {q50:.3f}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q90, y_annot, f"P90: {q90:.3f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Add vertical line at 1.0 (corresponds to m/beta = 1, i.e., 2^0 = 1)
    pct_at_1 = np.mean(data <= 1.0) * 100
    plt.axvline(1.0, color="tab:red", linestyle=":", linewidth=2, label=f"P{round(pct_at_1)}: 1.0")
    plt.text(1.0, y_annot * 0.7, f"P{round(pct_at_1)}: 1.0", rotation=90, va="top", ha="right", color="tab:red", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Add vertical line at 2.0 (corresponds to m/beta = 2, i.e., 2^0.5 ≈ 1.41)
    pct_at_2 = np.mean(data <= 2.0) * 100
    plt.axvline(2.0, color="tab:purple", linestyle=":", linewidth=2, label=f"P{round(pct_at_2)}: 2.0")
    plt.text(2.0, y_annot * 0.7, f"P{round(pct_at_2)}: 2.0", rotation=90, va="top", ha="right", color="tab:purple", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("Takeoff Exponent 2^(1 - β/m)")
    plt.ylabel("Count")
    plt.title(title or f"Distribution of Takeoff Exponent 2^(1 - β/m) (n={len(data)}, skipped={num_skipped})")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_uplift_ratio_histogram(
    uplift_ratio_values: List[float],
    num_skipped: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    max_x: float = 10.0
) -> None:
    """Plot histogram of uplift ratio 2^(beta/m - 1) values.

    This represents the ratio of successive uplift doublings during software
    intelligence explosion.

    Args:
        uplift_ratio_values: list of 2^(beta/m - 1) values
        num_skipped: count of rollouts where the ratio could not be computed
        out_path: output file path
        bins: number of histogram bins
        title: optional plot title
        max_x: maximum x value to display (values above this are noted separately)
    """
    if len(uplift_ratio_values) == 0:
        raise ValueError("No uplift ratio values found to plot")

    data_all = np.asarray(uplift_ratio_values, dtype=float)
    # Filter to positive finite values
    data_all = data_all[(data_all > 0) & np.isfinite(data_all)]
    if len(data_all) == 0:
        raise ValueError("No positive finite uplift ratio values found to plot")

    # Count values above max_x
    num_above_max = int(np.sum(data_all > max_x))
    pct_above_max = 100.0 * num_above_max / len(data_all)

    # Filter data for plotting (only values <= max_x)
    data = data_all[data_all <= max_x]

    # Use log-spaced bins for log scale x-axis
    log_data = np.log10(data)
    log_bins = np.linspace(log_data.min(), np.log10(max_x), bins + 1)
    bin_edges_linear = 10 ** log_bins

    plt.figure(figsize=(10, 6))

    counts, bin_edges, _ = plt.hist(data, bins=bin_edges_linear, edgecolor="black", alpha=0.6, label="Histogram")
    plt.xscale('log')
    plt.xlim(right=max_x)

    # Percentile lines (using all data for accurate percentiles)
    q10, q50, q90 = np.quantile(data_all, [0.1, 0.5, 0.9])
    ymax = float(np.max(counts) if counts.size else 1.0)
    y_annot = ymax * 0.95

    # Include p10 and p90 values in legend labels
    plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label=f"P10: {q10:.2f}")
    plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label=f"Median: {q50:.2f}")
    # Only show P90 line if it's within the visible range
    if q90 <= max_x:
        plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label=f"P90: {q90:.2f}")
    else:
        # Add P90 to legend even if not visible
        plt.plot([], [], color="tab:gray", linestyle="--", linewidth=1.5, label=f"P90: {q90:.2f} (off chart)")

    plt.text(q10, y_annot, f"P10: {q10:.2f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q50, y_annot, f"Median: {q50:.2f}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
    if q90 <= max_x:
        plt.text(q90, y_annot, f"P90: {q90:.2f}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Add vertical line at 1.0 (corresponds to beta/m = 1, i.e., 2^0 = 1)
    pct_at_1 = np.mean(data_all <= 1.0) * 100
    plt.axvline(1.0, color="tab:red", linestyle=":", linewidth=2, label=f"P{round(pct_at_1)}: 1.0")
    plt.text(1.0, y_annot * 0.7, f"P{round(pct_at_1)}: 1.0", rotation=90, va="top", ha="right", color="tab:red", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Add vertical line at 0.7
    pct_at_07 = np.mean(data_all <= 0.7) * 100
    plt.axvline(0.7, color="tab:purple", linestyle=":", linewidth=2, label=f"P{round(pct_at_07)}: 0.7")
    plt.text(0.7, y_annot * 0.7, f"P{round(pct_at_07)}: 0.7", rotation=90, va="top", ha="right", color="tab:purple", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Add legend entry for values above max_x (if any)
    if num_above_max > 0:
        plt.plot([], [], ' ', label=f"{pct_above_max:.1f}% > {max_x} (not shown)")

    plt.xlabel("Ratio of successive uplift doubling times")
    plt.ylabel("Count")

    default_title = "Ratio of successive uplift doubling times when attempting a taste-only singularity"
    plt.title(title or default_title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Alias for backwards compatibility
def plot_inverse_takeoff_exponent_histogram(
    inverse_takeoff_exponent_values: List[float],
    num_skipped: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None,
    max_x: float = 10.0
) -> None:
    """Alias for plot_uplift_ratio_histogram for backwards compatibility."""
    plot_uplift_ratio_histogram(
        inverse_takeoff_exponent_values, num_skipped, out_path, bins, title, max_x
    )


def plot_m_over_beta_histogram(
    m_over_beta_values: List[float],
    num_skipped: int,
    out_path: Path,
    bins: int = 50,
    title: Optional[str] = None
) -> None:
    """Plot histogram of m/beta (automation_fraction_a / automation_fraction_b) values.

    Args:
        m_over_beta_values: list of m/beta ratio values
        num_skipped: count of rollouts where m/beta could not be computed
        out_path: output file path
        bins: number of histogram bins
        title: optional plot title
    """
    if len(m_over_beta_values) == 0:
        raise ValueError("No m/beta values found to plot")

    data = np.asarray(m_over_beta_values, dtype=float)
    # Filter to positive values for log scale
    data = data[data > 0]
    if len(data) == 0:
        raise ValueError("No positive m/beta values found to plot")

    # Use log-spaced bins for log scale x-axis
    log_data = np.log10(data)
    log_bins = np.linspace(log_data.min(), log_data.max(), bins + 1)
    bin_edges_linear = 10 ** log_bins

    plt.figure(figsize=(10, 6))

    counts, bin_edges, _ = plt.hist(data, bins=bin_edges_linear, edgecolor="black", alpha=0.6, label="Histogram")
    plt.xscale('log')

    # Add KDE on log-transformed data
    if len(data) >= 2:
        xs_log = np.linspace(log_data.min(), log_data.max(), 512)
        xs = 10 ** xs_log
        try:
            kde = make_gaussian_kde(log_data)
            # Scale KDE to match histogram counts (approximate)
            kde_vals = kde(xs_log)
            # Normalize to match histogram area
            bin_widths = np.diff(bin_edges_linear)
            hist_area = np.sum(counts * bin_widths)
            kde_area = np.trapezoid(kde_vals, xs)
            kde_counts = kde_vals * hist_area / kde_area if kde_area > 0 else kde_vals
            plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE (log)")
        except Exception:
            pass

    # Median line only
    q50 = np.quantile(data, 0.5)
    ymax = float(np.max(counts) if counts.size else 1.0)
    y_annot = ymax * 0.95

    plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
    plt.text(q50, y_annot, f"Median: {q50:.3f}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Add vertical line and percentile annotation for m/β = 1.0
    # Show as "P{rounded_percentile}: {value}"
    pct_at_1 = np.mean(data <= 1.0) * 100

    plt.axvline(1.0, color="tab:red", linestyle=":", linewidth=2, label=f"P{round(pct_at_1)}: 1.0")

    plt.text(1.0, y_annot * 0.7, f"P{round(pct_at_1)}: 1.0", rotation=90, va="top", ha="right", color="tab:red", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("m/β ratio")
    plt.ylabel("Count")
    plt.title(title or f"Distribution of m/beta (n={len(data)}, skipped={num_skipped})")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
