#!/usr/bin/env python3
"""
Generate conditional and unconditional timing distribution CSVs for the Forecast page.

This script filters rollouts where AC (Automated Coder) was achieved in a specific
calendar year, then computes time-until distributions for subsequent milestones.
It also generates unconditional distributions for all rollouts that achieved AC.

Output files match the format used by app/forecast/page.tsx:
  - ac_{year}_time_until_distributions.csv - PDF values over time (conditional)
  - ac_{year}_time_until_statistics.csv - Summary statistics (conditional)
  - ac_{year}_time_until_cdf.csv - Empirical CDF values (conditional)
  - ac_unconditional_time_until_distributions.csv - PDF values (unconditional)
  - ac_unconditional_time_until_statistics.csv - Statistics (unconditional)
  - ac_unconditional_time_until_cdf.csv - Empirical CDF (unconditional)

Usage:
  # Generate for all default years (2027, 2030, 2035) + unconditional
  python scripts/generate_conditional_timing.py --rollouts outputs/20241210/rollouts.jsonl

  # Generate for specific years
  python scripts/generate_conditional_timing.py --rollouts outputs/20241210/rollouts.jsonl \\
    --years 2027 2028 2029

  # Skip unconditional generation
  python scripts/generate_conditional_timing.py --rollouts outputs/20241210/rollouts.jsonl \\
    --no-unconditional

  # Specify output directory
  python scripts/generate_conditional_timing.py --rollouts outputs/20241210/rollouts.jsonl \\
    --out-dir app/forecast/data/simulations/my-simulation/
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import utilities from plotting_utils
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # scripts/ directory
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.kde import make_gamma_kernel_kde
from plotting_utils.helpers import load_present_day


def compute_empirical_cdf(
    times: np.ndarray,
    xs: np.ndarray,
    total_rollouts: int,
) -> np.ndarray:
    """Compute empirical CDF at given evaluation points.

    For each x in xs, computes the fraction of samples <= x.
    Unachieved rollouts are treated as not yet achieved at any finite time.

    Args:
        times: Array of achieved milestone times (time-until values)
        xs: Evaluation points (time values)
        total_rollouts: Total number of rollouts (including not achieved)

    Returns:
        Array of CDF values at each x point
    """
    if len(times) == 0 or total_rollouts == 0:
        return np.zeros_like(xs)

    sorted_times = np.sort(times)
    cdf_values = np.zeros_like(xs, dtype=float)

    for i, x in enumerate(xs):
        # Count how many achieved times are <= x
        count = np.searchsorted(sorted_times, x, side='right')
        cdf_values[i] = count / total_rollouts

    return cdf_values


# Default target milestones for conditional timing (all milestones after AC)
DEFAULT_TARGET_MILESTONES = [
    "AI2027-SC",
    "AIR-5x",
    "AIR-25x",
    "AIR-250x",
    "AIR-2000x",
    "AIR-10000x",
    "SAR-level-experiment-selection-skill",
    "SIAR-level-experiment-selection-skill",
    "STRAT-AI",
    "TED-AI",
    "ASI",
]

# Default years for conditional timing analysis
DEFAULT_FILTER_YEARS = [2027, 2030, 2035]


def filter_rollouts_by_ac_year(
    reader: RolloutsReader,
    year: int,
) -> Tuple[List[Dict], int]:
    """
    Filter rollouts where AC was achieved within a specific calendar year.

    Args:
        reader: RolloutsReader instance
        year: Calendar year to filter (e.g., 2027 means 2027.0 <= AC < 2028.0)

    Returns:
        Tuple of (list of filtered rollout data, total matching count)
    """
    filtered = []

    for rollout in reader.iter_normalized_rollouts():
        # Resolve AC milestone
        ac_data = reader._resolve_milestone(rollout, "AC")
        if ac_data is None:
            continue  # AC not achieved

        ac_time = ac_data['time']

        # Check if AC achieved in target year
        if float(year) <= ac_time < float(year + 1):
            # Store rollout data with AC time for later processing
            filtered.append({
                'rollout': rollout,
                'ac_time': ac_time,
            })

    return filtered, len(filtered)


def filter_all_rollouts_with_ac(
    reader: RolloutsReader,
) -> Tuple[List[Dict], int]:
    """
    Filter all rollouts where AC was achieved (no year constraint).

    Args:
        reader: RolloutsReader instance

    Returns:
        Tuple of (list of filtered rollout data, total matching count)
    """
    filtered = []

    for rollout in reader.iter_normalized_rollouts():
        # Resolve AC milestone
        ac_data = reader._resolve_milestone(rollout, "AC")
        if ac_data is None:
            continue  # AC not achieved

        ac_time = ac_data['time']

        # Store rollout data with AC time for later processing
        filtered.append({
            'rollout': rollout,
            'ac_time': ac_time,
        })

    return filtered, len(filtered)


def compute_time_until_distributions(
    reader: RolloutsReader,
    filtered_rollouts: List[Dict],
    target_milestones: List[str],
    monthly_sampling: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict], Dict[str, List[float]], Dict[str, int]]:
    """
    Compute KDE distributions for time from AC to each target milestone.

    Args:
        reader: RolloutsReader instance (for milestone resolution)
        filtered_rollouts: List of dicts with 'rollout' and 'ac_time' keys
        target_milestones: List of milestone names to compute distributions for
        monthly_sampling: If True, sample at monthly intervals

    Returns:
        Tuple of:
          - xs: Common time array (years from AC)
          - pdf_dict: Dict mapping milestone name to PDF values
          - stats_dict: Dict mapping milestone name to statistics
          - time_until_data: Dict mapping milestone name to list of time-until values
          - not_achieved_count: Dict mapping milestone name to count of not achieved
    """
    # First pass: collect all time-until values for each milestone
    time_until_data: Dict[str, List[float]] = {m: [] for m in target_milestones}
    not_achieved_count: Dict[str, int] = {m: 0 for m in target_milestones}

    for item in filtered_rollouts:
        rollout = item['rollout']
        ac_time = item['ac_time']

        for milestone_name in target_milestones:
            milestone_data = reader._resolve_milestone(rollout, milestone_name)

            if milestone_data is None:
                not_achieved_count[milestone_name] += 1
                continue

            milestone_time = milestone_data['time']
            time_until = milestone_time - ac_time

            # Include all achieved milestones
            # For milestones achieved at or before AC, treat as time=0
            if time_until >= 0:
                time_until_data[milestone_name].append(time_until)
            else:
                # Achieved before AC - treat as achieved at time 0
                time_until_data[milestone_name].append(0.0)

    # Determine common time grid
    # Find max duration across all milestones
    max_duration = 0.0
    for milestone_name, times in time_until_data.items():
        if times:
            max_duration = max(max_duration, max(times))

    # Add some padding
    max_duration = max(max_duration * 1.1, 30.0)  # At least 30 years

    # Create time grid
    if monthly_sampling:
        # Monthly intervals (1/12 year)
        steps = max(int(np.ceil(max_duration * 12.0)) + 1, 2)
        xs = np.linspace(0, max_duration, steps)
    else:
        xs = np.linspace(0, max_duration, 2048)

    # Compute KDE for each milestone
    pdf_dict: Dict[str, np.ndarray] = {}
    stats_dict: Dict[str, Dict] = {}
    total_rollouts = len(filtered_rollouts)

    for milestone_name in target_milestones:
        times = time_until_data[milestone_name]
        num_not_achieved = not_achieved_count[milestone_name]

        if len(times) < 2:
            # Not enough data for KDE - fill with zeros
            pdf_dict[milestone_name] = np.zeros_like(xs)
            stats_dict[milestone_name] = {
                'num_achieved': len(times),
                'num_not_achieved': num_not_achieved,
                'total_rollouts': total_rollouts,
                'achievement_rate_pct': (len(times) / total_rollouts * 100) if total_rollouts > 0 else 0,
                'mode': None,
                'p10': None,
                'p50': None,
                'p90': None,
            }
            continue

        data = np.asarray(times, dtype=float)

        # Calculate achievement probability for scaling
        prob_achieved = len(times) / total_rollouts if total_rollouts > 0 else 0.0

        # Create KDE with lower bound at 0 (durations can't be negative)
        try:
            kde = make_gamma_kernel_kde(data, lower_bound=0.0)
            pdf_values = kde.pdf(xs) * prob_achieved * 100.0  # Scale by achievement rate, convert to %
        except Exception as e:
            print(f"  Warning: Could not create KDE for {milestone_name}: {e}")
            pdf_dict[milestone_name] = np.zeros_like(xs)
            stats_dict[milestone_name] = {
                'num_achieved': len(times),
                'num_not_achieved': num_not_achieved,
                'total_rollouts': total_rollouts,
                'achievement_rate_pct': prob_achieved * 100,
                'mode': None,
                'p10': None,
                'p50': None,
                'p90': None,
            }
            continue

        pdf_dict[milestone_name] = pdf_values

        # Calculate statistics
        # For percentiles, include not-achieved as very large values (effectively censored)
        if num_not_achieved > 0:
            combined_data = np.concatenate([data, np.full(num_not_achieved, 1e6)])
        else:
            combined_data = data

        q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])
        mode_idx = int(np.argmax(pdf_values))
        mode = float(xs[mode_idx]) if pdf_values[mode_idx] > 0 else None

        stats_dict[milestone_name] = {
            'num_achieved': len(times),
            'num_not_achieved': num_not_achieved,
            'total_rollouts': total_rollouts,
            'achievement_rate_pct': prob_achieved * 100,
            'mode': mode,
            'p10': float(q10),
            'p50': float(q50),
            'p90': float(q90),
        }

    return xs, pdf_dict, stats_dict, time_until_data, not_achieved_count


def save_distributions_csv(
    xs: np.ndarray,
    pdf_dict: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Save combined distribution data in the same format as milestone_pdfs_overlay_distributions.csv.

    Args:
        xs: Time array (years from AC)
        pdf_dict: Dict mapping milestone name to PDF values
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get milestone names in consistent order
    milestone_names = list(pdf_dict.keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header: time_decimal_year, then one column per milestone
        header = ["time_decimal_year"] + milestone_names
        writer.writerow(header)

        # Data rows
        for i, t in enumerate(xs):
            row = [f"{t:.6f}"]
            for milestone_name in milestone_names:
                pdf_val = pdf_dict[milestone_name][i]
                row.append(f"{pdf_val:.10f}")
            writer.writerow(row)

    print(f"  Saved distributions to: {output_path}")


def save_statistics_csv(
    stats_dict: Dict[str, Dict],
    output_path: Path,
) -> None:
    """
    Save statistics in the same format as milestone_pdfs_overlay_statistics.csv.

    Args:
        stats_dict: Dict mapping milestone name to statistics dict
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "milestone_name",
        "num_achieved",
        "num_not_achieved",
        "total_rollouts",
        "achievement_rate_pct",
        "mode",
        "p10",
        "p50",
        "p90",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for milestone_name, stats in stats_dict.items():
            row = {
                "milestone_name": milestone_name,
                "num_achieved": stats['num_achieved'],
                "num_not_achieved": stats['num_not_achieved'],
                "total_rollouts": stats['total_rollouts'],
                "achievement_rate_pct": f"{stats['achievement_rate_pct']:.2f}",
                "mode": f"{stats['mode']:.4f}" if stats['mode'] is not None else "",
                "p10": f"{stats['p10']:.4f}" if stats['p10'] is not None else "",
                "p50": f"{stats['p50']:.4f}" if stats['p50'] is not None else "",
                "p90": f"{stats['p90']:.4f}" if stats['p90'] is not None else "",
            }
            writer.writerow(row)

    print(f"  Saved statistics to: {output_path}")


def save_cdf_csv(
    xs: np.ndarray,
    time_until_data: Dict[str, List[float]],
    not_achieved_count: Dict[str, int],
    total_rollouts: int,
    output_path: Path,
) -> None:
    """
    Save empirical CDF data to CSV.

    Args:
        xs: Time array (years from AC)
        time_until_data: Dict mapping milestone name to list of time-until values
        not_achieved_count: Dict mapping milestone name to count of not achieved
        total_rollouts: Total number of filtered rollouts
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get milestone names in consistent order
    milestone_names = list(time_until_data.keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header: time_decimal_year, then one column per milestone
        header = ["time_decimal_year"] + milestone_names
        writer.writerow(header)

        # Data rows
        for i, t in enumerate(xs):
            row = [f"{t:.6f}"]
            for milestone_name in milestone_names:
                times = time_until_data.get(milestone_name, [])
                times_array = np.asarray(times, dtype=float)
                cdf_val = compute_empirical_cdf(times_array, np.array([t]), total_rollouts)[0]
                row.append(f"{cdf_val:.10f}")
            writer.writerow(row)

    print(f"  Saved empirical CDF to: {output_path}")


def generate_conditional_timing_csvs(
    rollouts_file: Path,
    output_dir: Path,
    filter_years: List[int] = None,
    target_milestones: List[str] = None,
    monthly_sampling: bool = True,
    include_unconditional: bool = True,
) -> None:
    """
    Generate conditional timing distribution CSVs for specified AC arrival years.

    For each filter_year, filters rollouts where AC was achieved within that year,
    then computes time-until distributions for each subsequent milestone.

    Args:
        rollouts_file: Path to rollouts.jsonl file
        output_dir: Output directory for CSV files
        filter_years: List of years to filter (default: [2027, 2030, 2035])
        target_milestones: List of milestone names to include (default: all post-AC milestones)
        monthly_sampling: If True, sample at monthly intervals
        include_unconditional: If True, also generate unconditional timing files
    """
    if filter_years is None:
        filter_years = DEFAULT_FILTER_YEARS

    if target_milestones is None:
        target_milestones = DEFAULT_TARGET_MILESTONES

    print(f"Loading rollouts from: {rollouts_file}")
    reader = RolloutsReader(rollouts_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in filter_years:
        print(f"\nProcessing AC year {year}...")

        # Filter rollouts
        filtered, count = filter_rollouts_by_ac_year(reader, year)
        print(f"  Found {count} rollouts with AC in {year}")

        if count < 2:
            print(f"  Warning: Not enough rollouts for year {year}, skipping")
            continue

        # Compute distributions
        xs, pdf_dict, stats_dict, time_until_data, not_achieved_count = compute_time_until_distributions(
            reader,
            filtered,
            target_milestones,
            monthly_sampling=monthly_sampling,
        )

        # Save CSVs
        dist_path = output_dir / f"ac_{year}_time_until_distributions.csv"
        stats_path = output_dir / f"ac_{year}_time_until_statistics.csv"
        cdf_path = output_dir / f"ac_{year}_time_until_cdf.csv"

        save_distributions_csv(xs, pdf_dict, dist_path)
        save_statistics_csv(stats_dict, stats_path)
        save_cdf_csv(xs, time_until_data, not_achieved_count, count, cdf_path)

    # Generate unconditional timing distributions
    if include_unconditional:
        print("\nProcessing unconditional (all rollouts with AC)...")

        # Filter all rollouts that achieved AC
        filtered, count = filter_all_rollouts_with_ac(reader)
        print(f"  Found {count} rollouts with AC achieved")

        if count < 2:
            print("  Warning: Not enough rollouts with AC, skipping unconditional")
        else:
            # Compute distributions
            xs, pdf_dict, stats_dict, time_until_data, not_achieved_count = compute_time_until_distributions(
                reader,
                filtered,
                target_milestones,
                monthly_sampling=monthly_sampling,
            )

            # Save unconditional CSVs
            dist_path = output_dir / "ac_unconditional_time_until_distributions.csv"
            stats_path = output_dir / "ac_unconditional_time_until_statistics.csv"
            cdf_path = output_dir / "ac_unconditional_time_until_cdf.csv"

            save_distributions_csv(xs, pdf_dict, dist_path)
            save_statistics_csv(stats_dict, stats_path)
            save_cdf_csv(xs, time_until_data, not_achieved_count, count, cdf_path)

    print("\nDone!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conditional timing distribution CSVs for the Forecast page",
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
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (default: same directory as rollouts file)"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Years to generate conditional timing for (default: 2027 2030 2035)"
    )
    parser.add_argument(
        "--milestones",
        type=str,
        default=None,
        help="Comma-separated list of target milestones (default: all post-AC milestones)"
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        default=True,
        help="Sample KDE at monthly intervals (default: True)"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense sampling (2048 points) instead of monthly"
    )
    parser.add_argument(
        "--include-unconditional",
        action="store_true",
        default=True,
        dest="include_unconditional",
        help="Generate unconditional timing files (default: True)"
    )
    parser.add_argument(
        "--no-unconditional",
        action="store_false",
        dest="include_unconditional",
        help="Skip generating unconditional timing files"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = rollouts_path.parent

    # Parse milestone filter
    target_milestones = None
    if args.milestones:
        target_milestones = [m.strip() for m in args.milestones.split(",")]

    # Generate CSVs
    generate_conditional_timing_csvs(
        rollouts_path,
        out_dir,
        filter_years=args.years,
        target_milestones=target_milestones,
        monthly_sampling=not args.dense,
        include_unconditional=args.include_unconditional,
    )


if __name__ == "__main__":
    main()
