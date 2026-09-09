#!/usr/bin/env python3
"""
Ad-hoc script to compute the median year at which a milestone is reached,
filtering trajectories by METR MSE threshold.

WARNING: This script contains bespoke data loading code that doesn't follow the
project's plotting utility conventions. It should be refactored to use utilities
from scripts/plotting_utils/rollouts_reader.py.

Usage:
  python scripts/median_milestone_year.py --rollouts outputs/run/rollouts.jsonl \
    --milestone AC-95 --mse-threshold 0.5

  python scripts/median_milestone_year.py --rollouts outputs/run/rollouts.jsonl \
    --milestone AIR-5x --mse-threshold 1.0 --list-milestones
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RolloutsReader for unified rollout parsing
from plotting_utils.rollouts_reader import RolloutsReader


def read_milestone_data(rollouts_file: Path, milestone_name: str) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Read milestone times and MSE values from rollouts.jsonl.

    Returns:
        milestone_times: list of milestone arrival times (decimal years, None if not achieved)
        mse_values: list of METR MSE values
    """
    milestone_times: List[Optional[float]] = []
    mse_values: List[Optional[float]] = []

    reader = RolloutsReader(rollouts_file)

    for rollout in reader.iter_rollouts():
        results = rollout.get("results", {})
        milestones = results.get("milestones", {})
        mse_val = results.get("metr_mse")

        # Get the specific milestone time
        milestone_data = milestones.get(milestone_name)
        if milestone_data is not None and isinstance(milestone_data, dict):
            time_val = milestone_data.get("time")
            try:
                if time_val is not None and np.isfinite(float(time_val)):
                    milestone_times.append(float(time_val))
                else:
                    milestone_times.append(None)
            except (ValueError, TypeError):
                milestone_times.append(None)
        else:
            milestone_times.append(None)

        # Get MSE value
        try:
            mse_values.append(float(mse_val) if mse_val is not None and np.isfinite(float(mse_val)) else None)
        except Exception:
            mse_values.append(None)

    if len(milestone_times) == 0:
        raise ValueError(f"No data found in rollouts file")

    return milestone_times, mse_values


def list_all_milestones(rollouts_file: Path) -> Set[str]:
    """List all unique milestone names found in the rollouts file."""
    all_milestones: Set[str] = set()

    reader = RolloutsReader(rollouts_file)

    for rollout in reader.iter_rollouts():
        results = rollout.get("results", {})
        milestones = results.get("milestones", {})
        if isinstance(milestones, dict):
            all_milestones.update(milestones.keys())

    return all_milestones


def main():
    parser = argparse.ArgumentParser(
        description="Find median year when a milestone is reached, filtering by MSE threshold"
    )
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts.jsonl file")
    parser.add_argument("--milestone", type=str, help="Milestone name (e.g., 'AC-95', 'AIR-5x')")
    parser.add_argument("--mse-threshold", type=float, help="Maximum MSE threshold for inclusion")
    parser.add_argument("--list-milestones", action="store_true", help="List available milestones and exit")
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics")

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
            for m in sorted(milestones):
                print(f"  - {m}")
        else:
            print("No milestones found in rollouts file")
        return

    # Validate required arguments
    if args.milestone is None:
        parser.error("--milestone is required (or use --list-milestones to see available milestones)")
    if args.mse_threshold is None:
        parser.error("--mse-threshold is required")

    print(f"Loading trajectories from: {rollouts_path}")
    milestone_times, mse_values = read_milestone_data(rollouts_path, args.milestone)
    print(f"Loaded {len(milestone_times)} total trajectories")

    # Filter by MSE threshold
    # Treat non-achieving trajectories as ">2050"
    filtered_times = []
    num_achieved = 0
    num_not_achieved = 0
    num_achieved_before_filter = sum(1 for t in milestone_times if t is not None)

    for i, mse in enumerate(mse_values):
        if mse is None or mse <= args.mse_threshold:
            if milestone_times[i] is not None:
                filtered_times.append(milestone_times[i])
                num_achieved += 1
            else:
                filtered_times.append(2050.0)  # Treat as ">2050"
                num_not_achieved += 1

    num_passed_mse = sum(1 for mse in mse_values if mse is None or mse <= args.mse_threshold)

    print(f"Filtering to {num_passed_mse} trajectories with MSE ≤ {args.mse_threshold}")
    print(f"Of those, {num_achieved} achieve milestone '{args.milestone}'")
    print(f"  ({num_not_achieved} trajectories never achieve milestone, treated as '>2050')")

    if num_passed_mse == 0:
        print(f"ERROR: No trajectories pass MSE threshold of {args.mse_threshold}")
        return

    # Compute statistics
    filtered_times = np.array(filtered_times)
    median_year = np.median(filtered_times)
    p10_year = np.percentile(filtered_times, 10)
    p90_year = np.percentile(filtered_times, 90)
    mean_year = np.mean(filtered_times)

    print("\n" + "=" * 70)
    print(f"Milestone: {args.milestone}")
    print(f"MSE Threshold: {args.mse_threshold}")
    print(f"Trajectories achieving milestone: {num_achieved} / {num_passed_mse}")
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
        print(f"  Min year:  {np.min(filtered_times):.2f}")
        print(f"  Max year:  {np.max(filtered_times):.2f}")
        print(f"  Std dev:   {np.std(filtered_times):.2f} years")
        print(f"  P25 year:  {np.percentile(filtered_times, 25):.2f}")
        print(f"  P75 year:  {np.percentile(filtered_times, 75):.2f}")

        # Show MSE statistics for filtered set
        filtered_mse = [mse_values[i] for i in range(len(mse_values))
                       if (mse_values[i] is None or mse_values[i] <= args.mse_threshold)
                       and milestone_times[i] is not None]
        filtered_mse = [m for m in filtered_mse if m is not None]

        if len(filtered_mse) > 0:
            print(f"\nMSE statistics (filtered set achieving milestone):")
            print(f"  Mean MSE:   {np.mean(filtered_mse):.4f}")
            print(f"  Median MSE: {np.median(filtered_mse):.4f}")
            print(f"  Min MSE:    {np.min(filtered_mse):.4f}")
            print(f"  Max MSE:    {np.max(filtered_mse):.4f}")


if __name__ == "__main__":
    main()
