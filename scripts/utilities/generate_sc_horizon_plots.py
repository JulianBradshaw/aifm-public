#!/usr/bin/env python3
"""
Generate horizon trajectory plots filtered by SC (AI2027-SC) arrival time.

Usage:
    python scripts/generate_sc_horizon_plots.py outputs/251209_eli_th_preview --sc-dates "2028-06" "2030-06"
    python scripts/generate_sc_horizon_plots.py outputs/run_dir --auto-median --include-mar-2027 --output-manifest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

# Add scripts directory to path for imports
scripts_dir = Path(__file__).resolve().parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.helpers import load_present_day, load_central_trajectory_selection_start
from plotting.trajectories import plot_horizon_trajectories


def compute_median_sc_date(run_dir: Path) -> Tuple[str, float]:
    """Compute median SC date from rollouts.

    Args:
        run_dir: Path to run directory containing rollouts.jsonl

    Returns:
        Tuple of (date_str, decimal_year) e.g., ("2028-06", 2028.458)

    Raises:
        ValueError: If no SC times found in rollouts
    """
    reader = RolloutsReader(run_dir / "rollouts.jsonl")
    times, num_not_achieved, _ = reader.read_milestone_times("AI2027-SC")

    if len(times) == 0:
        raise ValueError("No SC times found in rollouts")

    total = len(times) + num_not_achieved
    achievement_rate = len(times) / total if total > 0 else 0

    if achievement_rate < 0.1:
        print(f"Warning: Only {achievement_rate:.1%} of rollouts achieved SC")

    median_decimal = float(np.median(times))
    year = int(median_decimal)
    # Convert fractional year to month (1-12)
    month = int((median_decimal - year) * 12) + 1
    # Clamp month to valid range
    month = max(1, min(12, month))

    date_str = f"{year}-{month:02d}"
    print(f"Computed median SC: {date_str} (decimal: {median_decimal:.3f})")
    print(f"  Achievement rate: {achievement_rate:.1%} ({len(times)}/{total})")

    return date_str, median_decimal


def select_central_trajectory_with_tolerance(
    times: np.ndarray,
    trajectories: list,
    sc_times: list,
    present_day: float,
    ac_times: list,
    ac_tolerance: float = 0.15,
    selection_start: float = None,
):
    """Select central trajectory using distance-from-median method with configurable tolerance.

    Args:
        times: Time array in decimal years
        trajectories: List of horizon arrays (minutes)
        sc_times: List of SC arrival times (unused, kept for API compatibility)
        present_day: Present day in decimal years (fallback for selection_start)
        ac_times: List of AC arrival times (used for filtering and distance cutoff)
        ac_tolerance: Tolerance for AC time filtering (default: 0.15 years)
        selection_start: Start year for trajectory selection grid (defaults to present_day)

    Returns:
        Tuple of (best_index, central_trajectory_array, median_ac_time)
    """
    if ac_times is None:
        raise ValueError("ac_times is required for central trajectory selection")

    # Use selection_start if provided, otherwise fall back to present_day
    grid_start = selection_start if selection_start is not None else present_day

    # Get finite AC times
    finite_ac = [(i, t) for i, t in enumerate(ac_times) if t is not None]
    if len(finite_ac) == 0:
        raise ValueError("No trajectories with finite AC times")

    # Compute median AC time
    ac_vals = [t for _, t in finite_ac]
    median_ac = float(np.median(ac_vals))

    # Filter to trajectories with AC within tolerance of median
    filtered = [(i, t) for i, t in finite_ac if abs(t - median_ac) <= ac_tolerance]
    if len(filtered) == 0:
        # Fall back: use all finite AC trajectories
        filtered = finite_ac

    filtered_indices = [i for i, _ in filtered]
    ac_range_end = median_ac + ac_tolerance
    print(f"Central trajectory selection: {len(filtered_indices)} trajectories with AC within ±{ac_tolerance} of median AC={median_ac:.3f}")
    print(f"  Grid start: {grid_start:.3f}, Grid end: {ac_range_end:.3f}")

    # Create monthly grid from grid_start to end of AC tolerance range
    x_grid = np.arange(grid_start, ac_range_end + 1e-6, 1/12)

    # Build matrix of log10(horizon) values interpolated to monthly grid
    matrix = np.full((len(filtered_indices), len(x_grid)), np.nan)
    for row_idx, traj_idx in enumerate(filtered_indices):
        traj = trajectories[traj_idx]
        valid = np.isfinite(traj) & (traj > 0)
        if np.sum(valid) < 2:
            continue
        valid_times = times[valid]
        valid_vals = traj[valid]

        mask = (x_grid >= valid_times[0]) & (x_grid <= valid_times[-1])
        if mask.any():
            interp_vals = np.interp(x_grid[mask], valid_times, valid_vals)
            matrix[row_idx, mask] = np.log10(np.clip(interp_vals, 1e-10, None))

    # Compute median curve
    median_curve = np.nanmedian(matrix, axis=0)

    # Compute mean distance from median for each trajectory
    distances = np.zeros(len(filtered_indices))
    for row_idx, traj_idx in enumerate(filtered_indices):
        ac_time = ac_times[traj_idx] if traj_idx < len(ac_times) else None
        if ac_time is not None:
            time_mask = x_grid <= ac_time
            row_diffs = np.abs(matrix[row_idx, :] - median_curve)
            row_diffs[~time_mask] = np.nan
            distances[row_idx] = np.nanmean(row_diffs)
        else:
            distances[row_idx] = np.nanmean(np.abs(matrix[row_idx, :] - median_curve))

    # Find trajectory with minimum mean distance
    best_row_idx = int(np.nanargmin(distances))
    best_traj_idx = filtered_indices[best_row_idx]

    print(f"Selected central trajectory index {best_traj_idx} with mean distance {distances[best_row_idx]:.4f}")

    return best_traj_idx, trajectories[best_traj_idx], median_ac


def date_to_decimal_year(date_str: str) -> float:
    """Convert date string like '2028-06' to decimal year like 2028.458."""
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1])
    # Middle of the month
    day_of_year = (month - 1) * 30.44 + 15
    return year + day_of_year / 365.25


def decimal_year_to_label(decimal_year: float) -> str:
    """Convert decimal year to human-readable label like 'Jun 2028'."""
    year = int(decimal_year)
    frac = decimal_year - year
    month = int(frac * 12) + 1
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{month_names[month-1]} {year}"


def generate_sc_filtered_plot(
    run_dir: Path,
    target_date: str,
    tolerance: float = 0.1,
):
    """Generate horizon trajectory plot filtered by SC arrival time."""
    rollouts_path = run_dir / "rollouts.jsonl"
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # Parse target date
    target_year = date_to_decimal_year(target_date)
    label = decimal_year_to_label(target_year)
    safe_label = label.lower().replace(" ", "_")

    print(f"\n=== Generating SC {label} filtered plot ===")
    print(f"Target: {target_year:.3f} ± {tolerance}")

    # Read rollouts directly to get both SC and AC times in a single pass
    # IMPORTANT: Track file line numbers since some rollouts may be skipped
    times = None
    trajectories = []
    aa_times = []
    mse_values = []
    sc_times = []
    ac_times = []
    file_line_numbers = []  # Maps internal index to file line number

    with open(rollouts_path, 'r') as f:
        for line_num, line in enumerate(f):
            rollout = json.loads(line)
            results = rollout.get('results')
            if results is None:
                continue

            # Get times array
            if times is None:
                times = np.array(results.get('times', []))

            # Get horizon lengths
            horizons = results.get('horizon_lengths')
            if horizons is None:
                continue

            # Track file line number for this valid rollout
            file_line_numbers.append(line_num)
            trajectories.append(np.array(horizons))

            # Get aa_time
            aa_time = results.get('aa_time')
            if aa_time is not None and np.isfinite(aa_time):
                aa_times.append(float(aa_time))
            else:
                aa_times.append(None)

            # Get MSE
            mse = results.get('metr_mse')
            if mse is not None and np.isfinite(mse):
                mse_values.append(float(mse))
            else:
                mse_values.append(None)

            # Get milestone times
            milestones = results.get('milestones', {})

            # SC time (AI2027-SC)
            sc_data = milestones.get('AI2027-SC')
            if sc_data and sc_data.get('time') is not None:
                sc_times.append(float(sc_data['time']))
            else:
                sc_times.append(None)

            # AC time
            ac_data = milestones.get('AC')
            if ac_data and ac_data.get('time') is not None:
                ac_times.append(float(ac_data['time']))
            else:
                ac_times.append(None)

    if times is None or len(trajectories) == 0:
        raise ValueError("No valid trajectories found")

    print(f"Loaded {len(trajectories)} trajectories")

    # Get present day
    present_day = load_present_day(run_dir)
    if present_day is None:
        present_day = 2024.92  # Default fallback

    # Get central trajectory selection start (if configured)
    selection_start = load_central_trajectory_selection_start(run_dir)
    if selection_start is not None:
        print(f"Central trajectory selection start: {selection_start}")

    # Filter to trajectories with SC within tolerance of target
    sc_filtered_indices = [
        i for i, t in enumerate(sc_times)
        if t is not None and abs(t - target_year) <= tolerance
    ]

    print(f"Found {len(sc_filtered_indices)} trajectories with SC within ±{tolerance} of {target_year:.3f}")

    if len(sc_filtered_indices) == 0:
        print(f"Warning: No trajectories within tolerance. Trying 2x tolerance...")
        tolerance *= 2
        sc_filtered_indices = [
            i for i, t in enumerate(sc_times)
            if t is not None and abs(t - target_year) <= tolerance
        ]
        print(f"Found {len(sc_filtered_indices)} trajectories with SC within ±{tolerance} of {target_year:.3f}")

    if len(sc_filtered_indices) == 0:
        print(f"Error: No trajectories found even with expanded tolerance")
        return None

    # Filter all arrays
    trajectories_filtered = [trajectories[i] for i in sc_filtered_indices]
    aa_times_filtered = [aa_times[i] for i in sc_filtered_indices] if aa_times else None
    mse_filtered = [mse_values[i] for i in sc_filtered_indices] if mse_values else None
    sc_times_filtered = [sc_times[i] for i in sc_filtered_indices]
    ac_times_filtered = [ac_times[i] for i in sc_filtered_indices] if ac_times else None

    # Create output directory
    horizon_dir = run_dir / "horizon_trajectories"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    # Generate plot
    out_path = horizon_dir / f"horizon_trajectories_sc_{safe_label}.png"

    # Compute stats for title
    sc_times_arr = np.array([t for t in sc_times_filtered if t is not None])
    sc_mean = np.mean(sc_times_arr)
    sc_std = np.std(sc_times_arr)

    # Find central trajectory FIRST (using broader tolerance of 0.15 years)
    central_idx_in_filtered = None
    internal_idx = None
    file_line_num = None
    try:
        central_idx_in_filtered, _, _ = select_central_trajectory_with_tolerance(
            times, trajectories_filtered, sc_times_filtered, present_day,
            ac_times=ac_times_filtered, ac_tolerance=0.15, selection_start=selection_start
        )
        # Map: central_idx_in_filtered -> index in SC-filtered arrays
        #       -> sc_filtered_indices[...] -> index in internal arrays
        #       -> file_line_numbers[...] -> actual file line number
        internal_idx = sc_filtered_indices[central_idx_in_filtered]
        file_line_num = file_line_numbers[internal_idx]
        print(f"Central trajectory: filtered index {central_idx_in_filtered}, internal index {internal_idx}, file line {file_line_num}")
        print(f"  Verification - SC time: {sc_times[internal_idx]:.3f}, AC time: {ac_times[internal_idx]:.3f}")
    except Exception as e:
        print(f"Warning: Could not select central trajectory: {e}")

    # Generate plot with the selected central trajectory
    plot_horizon_trajectories(
        times,
        trajectories_filtered,
        out_path,
        max_trajectories=1000,
        stop_at_sc=False,
        aa_times=aa_times_filtered,
        mse_values=mse_filtered,
        shade_by_mse=True,
        mse_threshold=None,
        cutoff_year=2040.0,
        title=f"Trajectories with SC in {label} ({sc_mean:.3f} ± {sc_std:.4f}y)\n(n={len(sc_filtered_indices)} trajectories)",
        sc_times=sc_times_filtered,
        present_day=present_day,
        ac_times=ac_times_filtered,
        central_trajectory_idx=central_idx_in_filtered,
        selection_start=selection_start,
        show_both_central_methods=True,
    )
    print(f"Saved plot: {out_path}")

    # Save CSV and parameters for central trajectory
    if central_idx_in_filtered is not None:
        with open(rollouts_path, 'r') as f:
            for i, line in enumerate(f):
                if i == file_line_num:
                    rollout = json.loads(line)
                    results = rollout['results']
                    params = rollout.get('parameters', {})
                    traj_times_central = np.array(results['times'])
                    traj_horizons_central = np.array(results['horizon_lengths'])

                    # Save trajectory CSV
                    csv_path = horizon_dir / f"horizon_trajectories_sc_{safe_label}.csv"
                    with open(csv_path, 'w') as csv_f:
                        csv_f.write("year,horizon_minutes\n")
                        for t, h in zip(traj_times_central, traj_horizons_central):
                            if np.isfinite(h) and h > 0:
                                csv_f.write(f"{t},{h}\n")
                    print(f"Saved CSV: {csv_path}")

                    # Save parameters JSON
                    params_path = horizon_dir / f"central_trajectory_params_sc_{safe_label}.json"
                    with open(params_path, 'w') as pf:
                        json.dump(params, pf, indent=2)
                    print(f"Saved parameters: {params_path}")
                    break

    return out_path


def write_manifest(
    run_dir: Path,
    generated_csvs: List[str],
    median_sc_date: Optional[str] = None,
    median_sc_decimal_year: Optional[float] = None,
) -> Path:
    """Write manifest JSON with generated file info.

    Args:
        run_dir: Run directory
        generated_csvs: List of generated CSV filenames (relative to horizon_trajectories/)
        median_sc_date: Median SC date string (e.g., "2028-06")
        median_sc_decimal_year: Median SC decimal year (e.g., 2028.458)

    Returns:
        Path to manifest file
    """
    horizon_dir = run_dir / "horizon_trajectories"
    manifest_path = horizon_dir / "sc_horizon_manifest.json"

    manifest = {
        "median_sc_date": median_sc_date,
        "median_sc_decimal_year": median_sc_decimal_year,
        "median_sc_label": decimal_year_to_label(median_sc_decimal_year).lower().replace(" ", "_") if median_sc_decimal_year else None,
        "generated_csvs": generated_csvs,
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate SC-filtered horizon trajectory plots")
    parser.add_argument("run_dir", type=Path, help="Run directory containing rollouts.jsonl")
    parser.add_argument("--sc-dates", nargs="+", default=None,
                       help="SC dates to filter by (format: YYYY-MM). Ignored if --auto-median is set.")
    parser.add_argument("--tolerance", type=float, default=0.1,
                       help="Tolerance in years for SC filtering (default: 0.1)")
    parser.add_argument("--auto-median", action="store_true",
                       help="Compute median SC date from rollouts and generate trajectory for it")
    parser.add_argument("--include-mar-2027", action="store_true",
                       help="Always include Mar 2027 as fixed reference point")
    parser.add_argument("--output-manifest", action="store_true",
                       help="Write JSON manifest with computed dates and file paths")

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Determine which dates to generate
    dates_to_generate = []
    median_sc_date = None
    median_sc_decimal_year = None

    if args.auto_median:
        # Compute median SC date from rollouts
        try:
            median_sc_date, median_sc_decimal_year = compute_median_sc_date(args.run_dir)
            dates_to_generate.append(median_sc_date)
        except Exception as e:
            print(f"Error computing median SC date: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if args.include_mar_2027:
        # Add March 2027 as fixed reference (AI2027 scenario baseline)
        if "2027-03" not in dates_to_generate:
            dates_to_generate.insert(0, "2027-03")

    # If not using auto-median and no dates specified, use defaults
    if not args.auto_median and args.sc_dates is None:
        dates_to_generate = ["2028-06", "2030-06"]
    elif args.sc_dates and not args.auto_median:
        dates_to_generate = args.sc_dates

    print(f"\nGenerating SC-filtered plots for dates: {dates_to_generate}")

    # Generate plots
    generated_csvs = []
    for date_str in dates_to_generate:
        try:
            out_path = generate_sc_filtered_plot(args.run_dir, date_str, args.tolerance)
            if out_path:
                # The plot_horizon_trajectories function generates CSVs with _dist_incl_backcasts suffix
                label = decimal_year_to_label(date_to_decimal_year(date_str))
                safe_label = label.lower().replace(" ", "_")
                csv_filename = f"horizon_trajectories_sc_{safe_label}_dist_incl_backcasts.csv"
                generated_csvs.append(csv_filename)
        except Exception as e:
            print(f"Error generating plot for {date_str}: {e}")
            import traceback
            traceback.print_exc()

    # Write manifest if requested
    if args.output_manifest and generated_csvs:
        write_manifest(
            args.run_dir,
            generated_csvs,
            median_sc_date=median_sc_date,
            median_sc_decimal_year=median_sc_decimal_year,
        )


if __name__ == "__main__":
    main()
