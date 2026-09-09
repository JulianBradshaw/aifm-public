#!/usr/bin/env python3
"""
Split rollouts by doubling_difficulty_growth_factor and generate plots for each subset.

Usage:
  python scripts/split_by_growth_factor.py --run-dir outputs/eli_1103_w_correlation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add scripts directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from plot_rollouts import (
    plot_milestone_pdfs_overlay,
    plot_horizon_trajectories,
    _read_horizon_trajectories
)


def split_rollouts_by_growth_factor(
    rollouts_file: Path,
    output_dir: Path,
    threshold: float = 1.0
) -> Tuple[Path, Path, int, int]:
    """Split rollouts into two files based on doubling_difficulty_growth_factor.

    Args:
        rollouts_file: Path to original rollouts.jsonl
        output_dir: Directory to save split rollouts
        threshold: Threshold value for splitting (default: 1.0)

    Returns:
        Tuple of (below_threshold_file, above_threshold_file, count_below, count_above)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    below_file = output_dir / f"rollouts_growth_factor_below_{threshold}.jsonl"
    above_file = output_dir / f"rollouts_growth_factor_above_{threshold}.jsonl"

    count_below = 0
    count_above = 0

    with rollouts_file.open("r", encoding="utf-8") as f_in, \
         below_file.open("w", encoding="utf-8") as f_below, \
         above_file.open("w", encoding="utf-8") as f_above:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters")

                if params is None or not isinstance(params, dict):
                    print(f"Warning: No parameters for sample {rec.get('sample_id')}, skipping")
                    continue

                growth_factor = params.get("doubling_difficulty_growth_factor")

                if growth_factor is None:
                    print(f"Warning: No growth_factor for sample {rec.get('sample_id')}, skipping")
                    continue

                if growth_factor < threshold:
                    f_below.write(line + "\n")
                    count_below += 1
                else:
                    f_above.write(line + "\n")
                    count_above += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON line: {e}")
                continue

    print(f"Split {count_below + count_above} rollouts:")
    print(f"  Growth factor < {threshold}: {count_below} rollouts -> {below_file}")
    print(f"  Growth factor >= {threshold}: {count_above} rollouts -> {above_file}")

    return below_file, above_file, count_below, count_above


def generate_plots_for_subset(
    rollouts_file: Path,
    output_dir: Path,
    subset_name: str,
    mse_threshold: float = None
) -> None:
    """Generate milestone PDFs and horizon trajectories for a subset of rollouts.

    Args:
        rollouts_file: Path to rollouts.jsonl file for this subset
        output_dir: Directory to save plots
        subset_name: Name for this subset (used in filenames and titles)
        mse_threshold: Optional MSE threshold for filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Milestone PDF overlays
    print(f"\nGenerating milestone PDF overlay for {subset_name}...")
    overlay_milestones = [
        "AC",
        "AI2027-SC",
        "SAR-level-experiment-selection-skill",
        "SIAR-level-experiment-selection-skill"
    ]

    pdf_out = output_dir / f"milestone_pdfs_overlay_{subset_name}.png"
    try:
        plot_milestone_pdfs_overlay(
            rollouts_file,
            overlay_milestones,
            pdf_out,
            title=f"Milestone Arrival Time Distributions - {subset_name}"
        )
        print(f"  Saved {pdf_out}")
    except Exception as e:
        print(f"  Warning: Could not generate milestone PDF overlay: {e}")

    # 2. Horizon trajectories
    print(f"\nGenerating horizon trajectories for {subset_name}...")
    try:
        times, trajectories, aa_times, mse_values = _read_horizon_trajectories(rollouts_file)

        # Generate base horizon trajectory plot
        horizon_out = output_dir / f"horizon_trajectories_{subset_name}.png"
        plot_horizon_trajectories(
            times,
            trajectories,
            horizon_out,
            max_trajectories=1000,
            stop_at_sc=False,
            aa_times=aa_times,
            mse_values=mse_values,
            shade_by_mse=True,
            mse_threshold=mse_threshold,
            title=f"Time Horizon Extension Trajectories - {subset_name}"
        )
        print(f"  Saved {horizon_out}")

    except Exception as e:
        print(f"  Warning: Could not generate horizon trajectories: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Split rollouts by doubling_difficulty_growth_factor and generate plots"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to rollout directory containing rollouts.jsonl"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Growth factor threshold for splitting (default: 1.0)"
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=None,
        help="Optional MSE threshold for filtering trajectories"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    rollouts_file = run_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    # Create output directory for split analysis
    split_dir = run_dir / "growth_factor_split"

    # Step 1: Split rollouts
    print(f"Splitting rollouts from {rollouts_file}...")
    below_file, above_file, count_below, count_above = split_rollouts_by_growth_factor(
        rollouts_file,
        split_dir,
        threshold=args.threshold
    )

    # Step 2: Generate plots for growth_factor < threshold
    if count_below > 0:
        print(f"\n{'='*60}")
        print(f"Processing subset: growth_factor < {args.threshold}")
        print(f"{'='*60}")
        generate_plots_for_subset(
            below_file,
            split_dir,
            f"growth_factor_below_{args.threshold}",
            mse_threshold=args.mse_threshold
        )
    else:
        print(f"\nNo rollouts with growth_factor < {args.threshold}, skipping plots")

    # Step 3: Generate plots for growth_factor >= threshold
    if count_above > 0:
        print(f"\n{'='*60}")
        print(f"Processing subset: growth_factor >= {args.threshold}")
        print(f"{'='*60}")
        generate_plots_for_subset(
            above_file,
            split_dir,
            f"growth_factor_above_{args.threshold}",
            mse_threshold=args.mse_threshold
        )
    else:
        print(f"\nNo rollouts with growth_factor >= {args.threshold}, skipping plots")

    print(f"\n{'='*60}")
    print("Complete! All plots saved to:")
    print(f"  {split_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
