#!/usr/bin/env python3
"""
Create lightweight cache from rollouts.jsonl for faster plotting.

This utility exports a cache that excludes trajectory arrays, making it
100-1000x smaller and faster to load than the full rollouts.jsonl file.
Use this for plotting scripts that only need milestone and scalar data.

The cache includes:
- sample_id, parameters, time_series_parameters
- milestones dict, aa_time, simulation_end
- All scalar values from results dict
- Error information for failed rollouts

Usage:
    python scripts/create_milestones_cache.py outputs/run_20250113_120000/rollouts.jsonl

    # Or process all runs in outputs/
    python scripts/create_milestones_cache.py --all

This creates rollouts.cache.json in the same directory as rollouts.jsonl.

RolloutsReader will automatically use the cache when available:
    from plotting_utils.rollouts_reader import RolloutsReader

    # Automatically uses cache if available (no code changes needed!)
    reader = RolloutsReader("outputs/latest/rollouts.jsonl")
    times, not_achieved, sim_end = reader.read_milestone_times("AC")
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from plotting_utils.rollouts_reader import RolloutsReader


def create_cache(rollouts_path: Path, force: bool = False) -> None:
    """Create cache for a single rollouts file.

    Args:
        rollouts_path: Path to rollouts.jsonl file
        force: If True, overwrite existing cache
    """
    rollouts_path = Path(rollouts_path)

    if not rollouts_path.exists():
        print(f"Error: {rollouts_path} not found")
        return

    cache_path = rollouts_path.parent / "rollouts.cache.json"

    if cache_path.exists() and not force:
        print(f"Cache already exists: {cache_path}")
        print("Use --force to overwrite")
        return

    print(f"Creating cache from {rollouts_path}...")

    try:
        reader = RolloutsReader(rollouts_path, use_cache=False)
        output_path = reader.export_cache()

        # Get file sizes for comparison
        original_size = rollouts_path.stat().st_size / (1024 * 1024)  # MB
        cache_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / cache_size if cache_size > 0 else 0

        print(f"✓ Created: {output_path}")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Cache size: {cache_size:.2f} MB")
        print(f"  Compression: {compression_ratio:.1f}x smaller")

    except Exception as e:
        print(f"Error creating cache: {e}")
        raise


def process_all_runs(outputs_dir: Path, force: bool = False) -> None:
    """Create caches for all runs in outputs directory.

    Args:
        outputs_dir: Path to outputs/ directory
        force: If True, overwrite existing caches
    """
    outputs_dir = Path(outputs_dir)

    if not outputs_dir.exists():
        print(f"Error: {outputs_dir} not found")
        return

    # Find all rollouts.jsonl files
    rollouts_files = list(outputs_dir.glob("*/rollouts.jsonl"))

    if not rollouts_files:
        print(f"No rollouts.jsonl files found in {outputs_dir}")
        return

    print(f"Found {len(rollouts_files)} rollout files")
    print()

    for rollouts_path in sorted(rollouts_files):
        create_cache(rollouts_path, force=force)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Create milestone-only NPZ cache from rollouts.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "rollouts_file",
        nargs="?",
        type=Path,
        help="Path to rollouts.jsonl file"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all runs in outputs/ directory"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache files"
    )

    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run outputs (default: outputs/)"
    )

    args = parser.parse_args()

    if args.all:
        process_all_runs(args.outputs_dir, force=args.force)
    elif args.rollouts_file:
        create_cache(args.rollouts_file, force=args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
