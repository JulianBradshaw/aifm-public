#!/usr/bin/env python3
"""
Plot a histogram of a metric's value at a specified calendar year.

Useful for inspecting distributions like automation fraction at 2025.0 or
progress rates at a milestone-independent snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

# Force non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_utils.rollouts_reader import RolloutsReader

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of metric values at a specific year",
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
        "--metric",
        type=str,
        default="automation_fraction",
        help="Metric trajectory name in rollouts.jsonl",
    )
    parser.add_argument(
        "--year",
        type=float,
        default=2025.0,
        help="Decimal year to sample",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Optional threshold to report fraction above",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional minimum x-axis value to display",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional maximum x-axis value to display",
    )
    parser.add_argument(
        "--x-scale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="Scale for the x-axis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for histogram PNG (default <run-dir>/<metric>_<year>.png)",
    )
    return parser.parse_args()


def _sample_metric_at_year(
    reader: RolloutsReader,
    metric_name: str,
    sample_year: float,
) -> np.ndarray:
    times, trajectories, _ = reader.read_trajectories(metric_name)
    if times.size == 0 or not trajectories:
        raise ValueError(f"No trajectories found for metric '{metric_name}'.")

    if sample_year < times.min() or sample_year > times.max():
        raise ValueError(
            f"Requested year {sample_year} outside available time range "
            f"[{times.min()}, {times.max()}]."
        )

    # Stack all trajectories into 2D array for vectorized interpolation
    traj_2d = np.vstack(trajectories)  # Shape: (n_rollouts, n_times)

    # Find interpolation index once (same for all trajectories)
    idx = np.searchsorted(times, sample_year)

    if idx == 0:
        values = traj_2d[:, 0]
    elif idx >= len(times):
        values = traj_2d[:, -1]
    else:
        # Linear interpolation: v = v0 + (v1 - v0) * weight
        t0, t1 = times[idx - 1], times[idx]
        weight = (sample_year - t0) / (t1 - t0)
        values = traj_2d[:, idx - 1] * (1 - weight) + traj_2d[:, idx] * weight

    # Filter out non-finite values
    finite_mask = np.isfinite(values)
    values = values[finite_mask]

    if values.size == 0:
        raise ValueError("No valid samples available after filtering non-finite values.")

    return values


def plot_histogram(
    values: np.ndarray,
    *,
    metric: str,
    year: float,
    threshold: float,
    bins: int,
    x_min: float | None,
    x_max: float | None,
    x_scale: str,
    out_path: Path,
) -> None:
    display_values = values
    if x_min is not None:
        display_values = display_values[display_values >= x_min]
    if x_max is not None:
        display_values = display_values[display_values <= x_max]

    if display_values.size == 0:
        raise ValueError("No samples fall within the requested x-range.")

    if x_scale == "log" and np.any(display_values <= 0):
        raise ValueError("Log-scale x-axis requires positive values within the display range.")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use logarithmically-spaced bins for log scale to avoid one giant bar
    if x_scale == "log":
        bin_edges = np.logspace(
            np.log10(display_values.min()),
            np.log10(display_values.max()),
            bins + 1,
        )
    else:
        bin_edges = bins

    counts, _, patches = ax.hist(
        display_values,
        bins=bin_edges,
        color="skyblue",
        edgecolor="white",
        alpha=0.9,
    )

    frac_above = np.mean(values > threshold)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel(f"{metric} at year {year}")
    ax.set_ylabel("Count")
    ax.set_title(f"{metric} distribution at {year:.2f}\n{len(values)} samples, {frac_above:.1%} above threshold")
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    ax.set_xscale(x_scale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved histogram to {out_path}")
    print(f"Samples: {len(values)}, mean: {values.mean():.4f}, median: {np.median(values):.4f}")
    print(f"Fraction above {threshold:.3f}: {frac_above:.2%}")


def main() -> None:
    args = _parse_args()
    if args.rollouts:
        rollouts_path = Path(args.rollouts)
        run_dir = rollouts_path.parent
    else:
        run_dir = Path(args.run_dir)
        rollouts_path = run_dir / "rollouts.jsonl"

    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    reader = RolloutsReader(rollouts_path)
    values = _sample_metric_at_year(reader, args.metric, args.year)

    out_path = (
        Path(args.output)
        if args.output
        else run_dir / f"{args.metric}_hist_{args.year:.1f}.png"
    )

    plot_histogram(
        values,
        metric=args.metric,
        year=args.year,
        threshold=args.threshold,
        bins=args.bins,
        x_min=args.x_min,
        x_max=args.x_max,
        x_scale=args.x_scale,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
