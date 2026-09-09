#!/usr/bin/env python3
"""
Generate Slack-formatted summary messages for milestone and horizon statistics.

WARNING: This script contains bespoke data loading code that doesn't follow the
project's plotting utility conventions. It should be refactored to use utilities
from scripts/plotting_utils/rollouts_reader.py.

Usage:
  python scripts/slack_summary.py --rollouts outputs/run/rollouts.jsonl --mse-threshold 1.5
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def read_rollout_data(rollouts_file: Path):
    """Read milestone times, horizon trajectories, and MSE values from rollouts.jsonl.

    Returns:
        times: common time array (decimal years)
        horizon_trajectories: list of horizon arrays (minutes)
        milestone_times: dict mapping milestone name to list of times
        mse_values: list of METR MSE values
    """
    horizon_trajectories: List[np.ndarray] = []
    milestone_times_dict = {}
    mse_values: List[Optional[float]] = []
    common_times: Optional[np.ndarray] = None

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

            # Read horizon trajectory
            times = results.get("times")
            horizon = results.get("horizon_lengths")
            if times is not None and horizon is not None:
                try:
                    times_arr = np.asarray(times, dtype=float)
                    horizon_arr = np.asarray(horizon, dtype=float)
                    if times_arr.ndim == 1 and horizon_arr.ndim == 1 and times_arr.size == horizon_arr.size:
                        if common_times is None:
                            common_times = times_arr
                        horizon_trajectories.append(horizon_arr)
                    else:
                        horizon_trajectories.append(None)
                except Exception:
                    horizon_trajectories.append(None)
            else:
                horizon_trajectories.append(None)

            # Read milestones
            milestones = results.get("milestones")
            if milestones is not None and isinstance(milestones, dict):
                for milestone_name, milestone_data in milestones.items():
                    if milestone_name not in milestone_times_dict:
                        milestone_times_dict[milestone_name] = []

                    if isinstance(milestone_data, dict):
                        time_val = milestone_data.get("time")
                        try:
                            if time_val is not None and np.isfinite(float(time_val)):
                                milestone_times_dict[milestone_name].append(float(time_val))
                            else:
                                milestone_times_dict[milestone_name].append(None)
                        except (ValueError, TypeError):
                            milestone_times_dict[milestone_name].append(None)
                    else:
                        milestone_times_dict[milestone_name].append(None)

            # Read MSE
            mse_val = results.get("metr_mse")
            try:
                mse_values.append(float(mse_val) if mse_val is not None and np.isfinite(float(mse_val)) else None)
            except Exception:
                mse_values.append(None)

    return common_times, horizon_trajectories, milestone_times_dict, mse_values


def find_crossing_year(times: np.ndarray, horizon: np.ndarray, target_minutes: float) -> Optional[float]:
    """Find the year when horizon first reaches or exceeds target_minutes."""
    valid_mask = np.isfinite(horizon) & (horizon > 0)
    if not np.any(valid_mask):
        return None

    valid_times = times[valid_mask]
    valid_horizon = horizon[valid_mask]

    crossing_indices = np.where(valid_horizon >= target_minutes)[0]
    if len(crossing_indices) == 0:
        return None

    first_idx = crossing_indices[0]
    if first_idx == 0:
        return float(valid_times[0])

    t0, t1 = valid_times[first_idx - 1], valid_times[first_idx]
    h0, h1 = valid_horizon[first_idx - 1], valid_horizon[first_idx]

    if h1 == h0:
        return float(t1)

    if h0 > 0 and h1 > 0 and target_minutes > 0:
        log_h0, log_h1, log_target = np.log(h0), np.log(h1), np.log(target_minutes)
        frac = (log_target - log_h0) / (log_h1 - log_h0)
    else:
        frac = (target_minutes - h0) / (h1 - h0)

    frac = np.clip(frac, 0.0, 1.0)
    crossing_year = t0 + frac * (t1 - t0)
    return float(crossing_year)


def compute_statistics(values: List[float]) -> Tuple[float, float, float]:
    """Compute median, P10, and P90."""
    arr = np.array(values)
    median = np.median(arr)
    p10 = np.percentile(arr, 10)
    p90 = np.percentile(arr, 90)
    return median, p10, p90


def format_year(year: float) -> str:
    """Format year, displaying '>2050' for censored values."""
    if year >= 2050.0:
        return ">2050"
    return f"{year:.2f}"


def compute_slack_summary(rollouts_path: Path, mse_threshold: float, milestone: str = "AC") -> dict:
    """Compute summary statistics for a given MSE threshold.

    Returns:
        dict with keys: mse_threshold, num_trajectories, milestone_name,
        milestone_median, milestone_p10, milestone_p90,
        horizon_10y_median, horizon_10y_p10, horizon_10y_p90,
        horizon_100y_median, horizon_100y_p10, horizon_100y_p90
    """
    # Read data
    times, horizon_trajectories, milestone_times_dict, mse_values = read_rollout_data(rollouts_path)

    if times is None:
        raise ValueError("No valid horizon trajectories found")

    # Filter by MSE threshold
    filtered_indices = []
    for i, mse in enumerate(mse_values):
        if mse is None or mse <= mse_threshold:
            filtered_indices.append(i)

    num_trajectories = len(filtered_indices)

    # === Compute milestone statistics ===
    milestone_values = []
    if milestone in milestone_times_dict:
        milestone_list = milestone_times_dict[milestone]
        for idx in filtered_indices:
            if idx < len(milestone_list):
                time_val = milestone_list[idx]
                if time_val is not None:
                    milestone_values.append(time_val)
                else:
                    milestone_values.append(2050.0)  # Censored

    milestone_median, milestone_p10, milestone_p90 = compute_statistics(milestone_values) if milestone_values else (None, None, None)

    # === Compute 10-year horizon statistics ===
    horizon_10y_minutes = 10 * 124560  # 10 work-years in minutes
    horizon_10y_values = []
    for idx in filtered_indices:
        if idx < len(horizon_trajectories) and horizon_trajectories[idx] is not None:
            crossing_year = find_crossing_year(times, horizon_trajectories[idx], horizon_10y_minutes)
            if crossing_year is not None:
                horizon_10y_values.append(crossing_year)
            else:
                horizon_10y_values.append(2050.0)  # Censored

    horizon_10y_median, horizon_10y_p10, horizon_10y_p90 = compute_statistics(horizon_10y_values) if horizon_10y_values else (None, None, None)

    # === Compute 100-year horizon statistics ===
    horizon_100y_minutes = 100 * 124560  # 100 work-years in minutes
    horizon_100y_values = []
    for idx in filtered_indices:
        if idx < len(horizon_trajectories) and horizon_trajectories[idx] is not None:
            crossing_year = find_crossing_year(times, horizon_trajectories[idx], horizon_100y_minutes)
            if crossing_year is not None:
                horizon_100y_values.append(crossing_year)
            else:
                horizon_100y_values.append(2050.0)  # Censored

    horizon_100y_median, horizon_100y_p10, horizon_100y_p90 = compute_statistics(horizon_100y_values) if horizon_100y_values else (None, None, None)

    return {
        "mse_threshold": mse_threshold,
        "num_trajectories": num_trajectories,
        "milestone_name": milestone,
        "milestone_median": milestone_median,
        "milestone_p10": milestone_p10,
        "milestone_p90": milestone_p90,
        "horizon_10y_median": horizon_10y_median,
        "horizon_10y_p10": horizon_10y_p10,
        "horizon_10y_p90": horizon_10y_p90,
        "horizon_100y_median": horizon_100y_median,
        "horizon_100y_p10": horizon_100y_p10,
        "horizon_100y_p90": horizon_100y_p90,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Slack-formatted summary of milestone and horizon statistics"
    )
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts.jsonl file")
    parser.add_argument("--mse-threshold", type=float, required=True, help="Maximum MSE threshold for inclusion")
    parser.add_argument("--milestone", type=str, default="AC", help="Milestone to analyze (default: AC)")

    args = parser.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # Compute summary
    result = compute_slack_summary(rollouts_path, args.mse_threshold, args.milestone)

    # === Generate Slack message ===
    print(f"MSE < {result['mse_threshold']} ({result['num_trajectories']} trajectories)")
    print()

    if result['milestone_median'] is not None:
        print(f"Median for {result['milestone_name']}: {format_year(result['milestone_median'])}")
        print(f"[{format_year(result['milestone_p10'])}, {format_year(result['milestone_p90'])}]")
    else:
        print(f"Median for {result['milestone_name']}: N/A")
    print()

    if result['horizon_10y_median'] is not None:
        print(f"Median for 10y horizons: {format_year(result['horizon_10y_median'])}")
        print(f"[{format_year(result['horizon_10y_p10'])}, {format_year(result['horizon_10y_p90'])}]")
    else:
        print(f"Median for 10y horizons: N/A")
    print()

    if result['horizon_100y_median'] is not None:
        print(f"Median for 100y horizons: {format_year(result['horizon_100y_median'])}")
        print(f"[{format_year(result['horizon_100y_p10'])}, {format_year(result['horizon_100y_p90'])}]")
    else:
        print(f"Median for 100y horizons: N/A")


if __name__ == "__main__":
    main()
