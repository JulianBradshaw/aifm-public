#!/usr/bin/env python3
"""
Generate overlay distribution CSV files for milestone PDFs.

This script generates the CSV files needed by forecast_export.py:
- milestone_pdfs_overlay_distributions.csv - KDE distribution data
- milestone_pdfs_overlay_statistics.csv - Summary statistics
- milestone_pdfs_overlay_cdf.csv - Empirical CDF data

The interface matches the old milestone_pdfs.py script for backwards compatibility.

Usage:
    python -m export.generate_overlay_distributions --rollouts outputs/20251222/rollouts.jsonl --out-dir outputs/20251222 --overlay --monthly
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plotting_utils.helpers import load_present_day
from plotting_utils.kde import make_gamma_kernel_kde
from plotting_utils.rollouts_reader import RolloutsReader


@dataclass(frozen=True)
class MilestoneBatch:
    times_map: Dict[str, List[float]]
    not_achieved_map: Dict[str, int]
    typical_sim_end: Optional[float]
    total_rollouts: int


# Milestones allowed in overlay mode (matches old milestone_pdfs.py)
OVERLAY_ALLOWED_MILESTONES = (
    "AC",
    "AI2027-SC",
    "SAR-level-experiment-selection-skill",
    "SIAR-level-experiment-selection-skill",
    "TED-AI",
    "ASI",
)


def compute_empirical_cdf(
    times: np.ndarray,
    xs: np.ndarray,
    total_rollouts: int,
) -> np.ndarray:
    """Compute empirical CDF at given evaluation points.

    For each x in xs, computes the fraction of samples <= x.
    Unachieved rollouts are treated as not yet achieved at any finite time.

    Args:
        times: Array of achieved milestone times
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
        count = np.searchsorted(sorted_times, x, side="right")
        cdf_values[i] = count / total_rollouts

    return cdf_values


def prepare_gamma_pdf(
    data: np.ndarray,
    present_day: float,
    monthly_sampling: bool,
    scale_factor: float,
    plot_max_year: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a gamma-kernel KDE and sample it on a grid suitable for plotting and CSV export.

    Returns:
        Tuple of (xs_full, pdf_full, xs_plot, pdf_plot, mode_year)
    """
    if data.size < 2:
        raise ValueError("Need at least two samples to build KDE.")

    kde = make_gamma_kernel_kde(data, lower_bound=present_day)
    bandwidth = float(kde.bandwidth)

    min_eval = max(present_day, float(np.min(data) - 5.0 * bandwidth))
    max_eval = max(float(np.max(data) + 5.0 * bandwidth), min_eval + bandwidth)

    if monthly_sampling:
        min_year = max(present_day, np.floor(min_eval * 12.0) / 12.0)
        max_year_full = np.ceil(max_eval * 12.0) / 12.0
        steps = max(int(np.ceil((max_year_full - min_year) * 12.0)) + 1, 2)
        xs = np.linspace(min_year, max_year_full, steps)
    else:
        xs = np.linspace(min_eval, max_eval, 2048)

    pdf_values = kde(xs) * scale_factor

    if plot_max_year is not None:
        mask = xs <= plot_max_year
        if not np.any(mask):
            xs_plot = xs
            pdf_plot = pdf_values
        else:
            xs_plot = xs[mask]
            pdf_plot = pdf_values[mask]
    else:
        xs_plot = xs
        pdf_plot = pdf_values

    mode_idx = int(np.argmax(pdf_values))
    mode = float(xs[mode_idx])
    return xs, pdf_values, xs_plot, pdf_plot, mode


def generate_overlay_distributions(
    rollouts_file: Path,
    out_dir: Path,
    monthly_sampling: bool = False,
) -> bool:
    """Generate overlay distribution CSV files.

    Args:
        rollouts_file: Path to rollouts.jsonl
        out_dir: Output directory for CSV files
        monthly_sampling: If True, sample at monthly intervals for smaller files

    Returns:
        True if successful, False otherwise
    """
    if not rollouts_file.exists():
        print(f"Error: rollouts.jsonl not found: {rollouts_file}")
        return False

    # Load present_day from the rollouts directory
    present_day = load_present_day(rollouts_file.parent)
    print(f"Using present_day: {present_day:.4f}")

    # Read milestone data
    reader = RolloutsReader(rollouts_file, use_cache=True)

    # Get all milestone names first
    all_milestones = list(OVERLAY_ALLOWED_MILESTONES)
    times_map, not_achieved_map, typical_sim_end, total_rollouts = (
        reader.read_milestone_times_batch(all_milestones)
    )

    batch = MilestoneBatch(times_map, not_achieved_map, typical_sim_end, total_rollouts)

    print(f"\nTotal rollouts: {total_rollouts}")

    # Filter to milestones with sufficient data
    valid_milestones = []
    for milestone in OVERLAY_ALLOWED_MILESTONES:
        times = batch.times_map.get(milestone, [])
        if len(times) >= 2:
            valid_milestones.append(milestone)
            not_achieved = batch.not_achieved_map.get(milestone, 0)
            pct = (len(times) / total_rollouts * 100) if total_rollouts > 0 else 0
            print(f"  {milestone}: {len(times)} achieved ({pct:.1f}%)")
        else:
            print(f"  {milestone}: insufficient data ({len(times)} achieved), skipping")

    if not valid_milestones:
        print("Error: No milestones with sufficient data")
        return False

    # Generate KDE data for each milestone
    all_kde_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    stats_list: List[Dict] = []
    boundary = float(present_day)

    for milestone in valid_milestones:
        times = batch.times_map.get(milestone, [])
        num_not_achieved = batch.not_achieved_map.get(milestone, 0)
        sim_end = batch.typical_sim_end

        raw_data = np.asarray(times, dtype=float)

        # Filter for present_day
        valid_mask = raw_data >= boundary
        data = raw_data[valid_mask]
        dropped = raw_data.size - data.size
        if dropped:
            print(f"  {milestone}: dropped {dropped} samples before present_day")

        if data.size < 2:
            print(f"  {milestone}: insufficient data after filtering, skipping")
            continue

        # Calculate statistics
        total_runs = len(times) + num_not_achieved
        prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0

        # Include not achieved as sim_end for percentile calculation
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate(
                [raw_data, np.full(num_not_achieved, sim_end)]
            )
        else:
            combined_data = raw_data

        q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

        # Generate KDE
        try:
            xs, pdf_values, xs_plot, pdf_plot, mode = prepare_gamma_pdf(
                data,
                boundary,
                monthly_sampling,
                prob_achieved * 100.0,
                plot_max_year=2050,
            )
            all_kde_data[milestone] = (xs, pdf_values)

            stats_list.append(
                {
                    "milestone_name": milestone,
                    "num_achieved": len(times),
                    "num_not_achieved": num_not_achieved,
                    "total_rollouts": total_runs,
                    "achievement_rate_pct": prob_achieved * 100,
                    "mode": mode,
                    "p10": q10,
                    "p50": q50,
                    "p90": q90,
                }
            )
        except Exception as e:
            print(f"  {milestone}: error generating KDE: {e}")
            continue

    if not all_kde_data:
        print("Error: No KDE data generated")
        return False

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create common time grid from first milestone's xs array
    first_xs, _ = next(iter(all_kde_data.values()))
    common_times = first_xs

    # Save distributions CSV
    dist_path = out_dir / "milestone_pdfs_overlay_distributions.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_decimal_year"] + list(all_kde_data.keys())
        writer.writerow(header)

        for t in common_times:
            row = [f"{t:.6f}"]
            for milestone_name in header[1:]:
                xs, pdf_vals = all_kde_data[milestone_name]
                pdf_val = float(np.interp(t, xs, pdf_vals, left=0.0, right=0.0))
                row.append(f"{pdf_val:.10f}")
            writer.writerow(row)

    print(f"\nSaved distributions to: {dist_path}")

    # Save statistics CSV
    stats_path = out_dir / "milestone_pdfs_overlay_statistics.csv"
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

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats_list:
            row = {
                "milestone_name": stat["milestone_name"],
                "num_achieved": stat["num_achieved"],
                "num_not_achieved": stat["num_not_achieved"],
                "total_rollouts": stat["total_rollouts"],
                "achievement_rate_pct": f"{stat['achievement_rate_pct']:.2f}",
                "mode": f"{stat['mode']:.4f}",
                "p10": f"{stat['p10']:.4f}",
                "p50": f"{stat['p50']:.4f}",
                "p90": f"{stat['p90']:.4f}",
            }
            writer.writerow(row)

    print(f"Saved statistics to: {stats_path}")

    # Save CDF CSV
    cdf_path = out_dir / "milestone_pdfs_overlay_cdf.csv"
    with cdf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_decimal_year"] + list(all_kde_data.keys())
        writer.writerow(header)

        for t in common_times:
            row = [f"{t:.6f}"]
            for milestone_name in header[1:]:
                times = batch.times_map.get(milestone_name, [])
                num_not_achieved = batch.not_achieved_map.get(milestone_name, 0)
                total_runs = len(times) + num_not_achieved
                times_array = np.asarray(times, dtype=float)
                cdf_val = compute_empirical_cdf(times_array, np.array([t]), total_runs)[0]
                row.append(f"{cdf_val:.10f}")
            writer.writerow(row)

    print(f"Saved CDF to: {cdf_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate overlay distribution CSV files for milestone PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        required=True,
        help="Path to rollouts.jsonl file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Generate overlay mode (for compatibility with old interface)",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Use monthly sampling for smaller CSV files",
    )

    args = parser.parse_args()

    rollouts_file = Path(args.rollouts)
    out_dir = Path(args.out_dir)

    if not rollouts_file.exists():
        print(f"Error: Rollouts file not found: {rollouts_file}")
        sys.exit(1)

    success = generate_overlay_distributions(
        rollouts_file,
        out_dir,
        monthly_sampling=args.monthly,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
