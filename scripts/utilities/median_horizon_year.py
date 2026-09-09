#!/usr/bin/env python3
"""
Ad-hoc script to compute the median year at which a given horizon length is reached,
filtering trajectories by METR MSE threshold.

WARNING: This script contains bespoke data loading code that doesn't follow the
project's plotting utility conventions. It should be refactored to use utilities
from scripts/plotting_utils/rollouts_reader.py.

Usage:
  python scripts/median_horizon_year.py --rollouts outputs/run/rollouts.jsonl \
    --horizon-minutes 120 --mse-threshold 0.5

  python scripts/median_horizon_year.py --rollouts outputs/run/rollouts.jsonl \
    --horizon-minutes 2400 --mse-threshold 1.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RolloutsReader for unified rollout parsing
from plotting_utils.rollouts_reader import RolloutsReader


def read_horizon_data(rollouts_file: Path) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]]]:
    """Read horizon trajectories and MSE values from rollouts.jsonl.

    Returns:
        times: common time array (decimal years)
        trajectories: list of horizon arrays (minutes)
        mse_values: list of METR MSE values
    """
    trajectories: List[np.ndarray] = []
    mse_values: List[Optional[float]] = []
    common_times: Optional[np.ndarray] = None

    reader = RolloutsReader(rollouts_file)

    for rollout in reader.iter_rollouts():
        results = rollout.get("results", {})

        times = results.get("times")
        horizon = results.get("horizon_lengths")
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

        if common_times is None:
            common_times = times_arr

        trajectories.append(horizon_arr)

        try:
            mse_values.append(float(mse_val) if mse_val is not None and np.isfinite(float(mse_val)) else None)
        except Exception:
            mse_values.append(None)

    if common_times is None or len(trajectories) == 0:
        raise ValueError("No horizon trajectories found in rollouts file")

    return common_times, trajectories, mse_values


def find_crossing_year(times: np.ndarray, horizon: np.ndarray, target_minutes: float) -> Optional[float]:
    """Find the year when horizon first reaches or exceeds target_minutes.

    Returns None if target is never reached.
    """
    # Filter out non-finite values
    valid_mask = np.isfinite(horizon) & (horizon > 0)
    if not np.any(valid_mask):
        return None

    valid_times = times[valid_mask]
    valid_horizon = horizon[valid_mask]

    # Find first crossing
    crossing_indices = np.where(valid_horizon >= target_minutes)[0]

    if len(crossing_indices) == 0:
        return None

    first_idx = crossing_indices[0]

    # If first point already exceeds target, return that time
    if first_idx == 0:
        return float(valid_times[0])

    # Linear interpolation between the point before and after crossing
    t0, t1 = valid_times[first_idx - 1], valid_times[first_idx]
    h0, h1 = valid_horizon[first_idx - 1], valid_horizon[first_idx]

    if h1 == h0:
        return float(t1)

    # Linear interpolation in log-space for horizon (since it's exponential)
    if h0 > 0 and h1 > 0 and target_minutes > 0:
        log_h0, log_h1, log_target = np.log(h0), np.log(h1), np.log(target_minutes)
        frac = (log_target - log_h0) / (log_h1 - log_h0)
    else:
        # Fallback to linear interpolation
        frac = (target_minutes - h0) / (h1 - h0)

    frac = np.clip(frac, 0.0, 1.0)
    crossing_year = t0 + frac * (t1 - t0)

    return float(crossing_year)


def main():
    parser = argparse.ArgumentParser(
        description="Find median year when a target horizon is reached, filtering by MSE threshold"
    )
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts.jsonl file")
    parser.add_argument("--horizon-minutes", type=float, required=True, help="Target horizon length in minutes")
    parser.add_argument("--mse-threshold", type=float, required=True, help="Maximum MSE threshold for inclusion")
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics")

    args = parser.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    print(f"Loading trajectories from: {rollouts_path}")
    times, trajectories, mse_values = read_horizon_data(rollouts_path)
    print(f"Loaded {len(trajectories)} total trajectories")

    # Filter by MSE threshold
    filtered_indices = []
    for i, mse in enumerate(mse_values):
        if mse is None or mse <= args.mse_threshold:
            filtered_indices.append(i)

    print(f"Filtering to {len(filtered_indices)} trajectories with MSE ≤ {args.mse_threshold}")

    if len(filtered_indices) == 0:
        print(f"ERROR: No trajectories pass MSE threshold of {args.mse_threshold}")
        return

    # Find crossing years for filtered trajectories
    # Treat non-achieving trajectories as ">2050"
    crossing_years = []
    num_not_achieved = 0
    for idx in filtered_indices:
        crossing_year = find_crossing_year(times, trajectories[idx], args.horizon_minutes)
        if crossing_year is not None:
            crossing_years.append(crossing_year)
        else:
            crossing_years.append(2050.0)  # Treat as ">2050"
            num_not_achieved += 1

    print(f"Found {len(crossing_years) - num_not_achieved} trajectories that reach {args.horizon_minutes} minutes")
    print(f"  ({num_not_achieved} trajectories never reach target, treated as '>2050')")

    # Compute statistics
    crossing_years = np.array(crossing_years)
    median_year = np.median(crossing_years)
    p10_year = np.percentile(crossing_years, 10)
    p90_year = np.percentile(crossing_years, 90)
    mean_year = np.mean(crossing_years)

    # Format horizon for display
    if args.horizon_minutes < 60:
        horizon_label = f"{args.horizon_minutes:.1f} minutes"
    elif args.horizon_minutes < 2400:  # Less than 1 work-week
        horizon_label = f"{args.horizon_minutes / 60:.1f} hours"
    elif args.horizon_minutes < 10380:  # Less than 1 work-month
        horizon_label = f"{args.horizon_minutes / 2400:.1f} work-weeks"
    elif args.horizon_minutes < 124560:  # Less than 1 work-year
        horizon_label = f"{args.horizon_minutes / 10380:.1f} work-months"
    else:
        horizon_label = f"{args.horizon_minutes / 124560:.1f} work-years"

    print("\n" + "=" * 70)
    print(f"Target Horizon: {horizon_label} ({args.horizon_minutes} minutes)")
    print(f"MSE Threshold: {args.mse_threshold}")
    print(f"Trajectories analyzed: {len(crossing_years) - num_not_achieved} / {len(filtered_indices)} achieve target")
    if num_not_achieved > 0:
        print(f"  + {num_not_achieved} trajectories censored at >2050")
    print("=" * 70)

    # Format output with ">2050" indicator if applicable
    def format_year(year):
        if year >= 2050.0:
            return ">2050"
        return f"{year:.2f}"

    print(f"Median year: {format_year(median_year)}")
    print(f"Mean year:   {format_year(mean_year)}")
    print(f"P10 year:    {format_year(p10_year)}")
    print(f"P90 year:    {format_year(p90_year)}")
    print("=" * 70)

    if args.verbose:
        print("\nDetailed statistics:")
        print(f"  Min year:  {np.min(crossing_years):.2f}")
        print(f"  Max year:  {np.max(crossing_years):.2f}")
        print(f"  Std dev:   {np.std(crossing_years):.2f} years")
        print(f"  P25 year:  {np.percentile(crossing_years, 25):.2f}")
        print(f"  P75 year:  {np.percentile(crossing_years, 75):.2f}")

        # Show MSE statistics for filtered set
        filtered_mse = [mse_values[i] for i in filtered_indices if mse_values[i] is not None]
        if len(filtered_mse) > 0:
            print(f"\nMSE statistics (filtered set):")
            print(f"  Mean MSE:   {np.mean(filtered_mse):.4f}")
            print(f"  Median MSE: {np.median(filtered_mse):.4f}")
            print(f"  Min MSE:    {np.min(filtered_mse):.4f}")
            print(f"  Max MSE:    {np.max(filtered_mse):.4f}")


if __name__ == "__main__":
    main()
