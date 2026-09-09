#!/usr/bin/env python3
"""
Create special threshold-based splits for specific parameters in addition to median splits.

For certain parameters, threshold-based splits are more meaningful than median splits:
- doubling_difficulty_growth_factor: Split at 1.0 (easier vs harder doubling)
- gap_years: Split at ~0 (has gap vs no gap)

Usage:
  # Process all special splits for a run directory
  python scripts/create_special_parameter_splits.py --run-dir outputs/251110_eli_2200

  # Process only specific parameter
  python scripts/create_special_parameter_splits.py \
    --run-dir outputs/251110_eli_2200 \
    --parameter doubling_difficulty_growth_factor
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from plot_rollouts import plot_milestone_pdfs_overlay


# Special parameter configurations
SPECIAL_PARAMETERS = {
    'doubling_difficulty_growth_factor': {
        'threshold': 1.0,
        'below_label': 'below_1.0',
        'above_label': 'above_1.0',
        'display_name': 'Doubling Difficulty Growth Factor',
        'description': 'Split at 1.0 (easier vs harder doubling)'
    },
    'gap_years': {
        'threshold': 0.01,  # Effectively zero
        'below_label': 'no_gap',
        'above_label': 'has_gap',
        'display_name': 'Effective Compute Gap',
        'description': 'Split at ~0 (has gap vs no gap)'
    },
    'r_software': {
        'threshold': 1.0,
        'below_label': 'below_1.0',
        'above_label': 'above_1.0',
        'display_name': 'R Software (m/β)',
        'description': 'Split at 1.0 (superlinear vs sublinear returns)'
    }
}


def split_rollouts_by_threshold(
    rollouts_file: Path,
    output_dir: Path,
    parameter_name: str,
    threshold: float,
    below_label: str,
    above_label: str
) -> Tuple[Path, Path, int, int]:
    """Split rollouts into two files based on parameter threshold.

    Args:
        rollouts_file: Path to original rollouts.jsonl
        output_dir: Directory to save split rollouts
        parameter_name: Name of parameter to split on
        threshold: Threshold value for splitting
        below_label: Label for below threshold file (e.g., 'below_1.0', 'no_gap')
        above_label: Label for above threshold file (e.g., 'above_1.0', 'has_gap')

    Returns:
        Tuple of (below_file, above_file, count_below, count_above)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    below_file = output_dir / f"rollouts_{below_label}.jsonl"
    above_file = output_dir / f"rollouts_{above_label}.jsonl"

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
                params = rec.get("parameters", {})

                if not isinstance(params, dict):
                    print(f"Warning: No parameters for sample {rec.get('sample_id')}, skipping")
                    continue

                param_value = params.get(parameter_name)

                if param_value is None:
                    print(f"Warning: No {parameter_name} for sample {rec.get('sample_id')}, skipping")
                    continue

                if param_value < threshold:
                    f_below.write(line + "\n")
                    count_below += 1
                else:
                    f_above.write(line + "\n")
                    count_above += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON line: {e}")
                continue

    print(f"Split {count_below + count_above} rollouts:")
    print(f"  {parameter_name} < {threshold}: {count_below} rollouts -> {below_file.name}")
    print(f"  {parameter_name} >= {threshold}: {count_above} rollouts -> {above_file.name}")

    return below_file, above_file, count_below, count_above


def generate_milestone_plots(
    rollouts_file: Path,
    output_dir: Path,
    subset_name: str,
    max_year: float = 2050
) -> None:
    """Generate key milestone PDF plots for a subset of rollouts.

    Args:
        rollouts_file: Path to rollouts.jsonl file for this subset
        output_dir: Directory to save plots
        subset_name: Name for this subset (used in filenames)
        max_year: Maximum year to plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating milestone PDFs for {subset_name}...")
    pdf_out = output_dir / f"milestone_pdfs_{subset_name}.png"

    try:
        milestone_names = ["AC", "SAR-level-experiment-selection-skill", "TED-AI", "ASI"]
        plot_milestone_pdfs_overlay(
            rollouts_file,
            milestone_names,
            pdf_out,
            title=f"Milestone Arrival Times - {subset_name}"
        )
        print(f"    Saved {pdf_out.name}")
    except Exception as e:
        print(f"    Warning: Could not generate milestone PDF: {e}")


def create_special_split(
    run_dir: Path,
    parameter_name: str,
    config: Dict
) -> None:
    """Create special threshold-based split for a parameter.

    Args:
        run_dir: Path to run directory
        parameter_name: Name of parameter to split
        config: Configuration dict with threshold, labels, etc.
    """
    print(f"\n{'='*70}")
    print(f"Creating special split for: {parameter_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")

    # Find the existing parameter split directory
    split_base_dir = run_dir / "parameter_splits" / f"{parameter_name.replace('_', '-')}_split"

    if not split_base_dir.exists():
        print(f"Error: Parameter split directory not found: {split_base_dir}")
        return

    # Check for median split files
    median_above = split_base_dir / "rollouts_above_median.jsonl"
    median_below = split_base_dir / "rollouts_below_median.jsonl"

    # Use combined file if it exists, otherwise use one of the median files as source
    if (split_base_dir / "rollouts_combined.jsonl").exists():
        source_file = split_base_dir / "rollouts_combined.jsonl"
        print(f"Using combined rollouts file as source")
    elif median_above.exists() and median_below.exists():
        # Create combined file from median splits
        source_file = split_base_dir / "rollouts_combined.jsonl"
        print(f"Creating combined rollouts file from median splits...")
        with source_file.open("w") as f_out:
            for rollouts_file in [median_above, median_below]:
                with rollouts_file.open("r") as f_in:
                    f_out.write(f_in.read())
    else:
        print(f"Error: No median split files found in {split_base_dir}")
        return

    # Create threshold-based split
    print(f"\nSplitting by threshold: {config['threshold']}")
    below_file, above_file, count_below, count_above = split_rollouts_by_threshold(
        source_file,
        split_base_dir,
        parameter_name,
        config['threshold'],
        config['below_label'],
        config['above_label']
    )

    # Generate plots for each split
    if count_below > 0:
        print(f"\nGenerating plots for {config['below_label']}...")
        generate_milestone_plots(
            below_file,
            split_base_dir,
            config['below_label']
        )
    else:
        print(f"\nNo rollouts for {config['below_label']}, skipping plots")

    if count_above > 0:
        print(f"\nGenerating plots for {config['above_label']}...")
        generate_milestone_plots(
            above_file,
            split_base_dir,
            config['above_label']
        )
    else:
        print(f"\nNo rollouts for {config['above_label']}, skipping plots")

    print(f"\n✓ Special split complete for {parameter_name}!")
    print(f"  Files saved to: {split_base_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create special threshold-based splits for specific parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory containing parameter_splits/"
    )
    parser.add_argument(
        "--parameter",
        type=str,
        choices=list(SPECIAL_PARAMETERS.keys()),
        help="Specific parameter to process (default: all special parameters)"
    )
    parser.add_argument(
        "--max-year",
        type=float,
        default=2050,
        help="Maximum year for plots (default: 2050)"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    splits_dir = run_dir / "parameter_splits"
    if not splits_dir.exists():
        print(f"Error: No parameter_splits directory found in {run_dir}")
        return

    # Determine which parameters to process
    if args.parameter:
        parameters_to_process = {args.parameter: SPECIAL_PARAMETERS[args.parameter]}
    else:
        parameters_to_process = SPECIAL_PARAMETERS

    print(f"\nProcessing {len(parameters_to_process)} special parameter split(s)")

    # Process each parameter
    for param_name, config in parameters_to_process.items():
        try:
            create_special_split(run_dir, param_name, config)
        except Exception as e:
            print(f"\nError processing {param_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print("All special splits complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
