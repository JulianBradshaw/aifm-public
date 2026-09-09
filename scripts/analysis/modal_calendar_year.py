#!/usr/bin/env python3
"""
Find the modal (most common) calendar year for a given milestone.

This script analyzes rollouts to find the single most likely calendar year
for milestone achievement. Only rollouts where the milestone was actually
achieved are considered (filler values like simulation end year are excluded).

Usage:
    python scripts/analysis/modal_calendar_year.py --run-dir outputs/your_run/ --milestone AC
    python scripts/analysis/modal_calendar_year.py --rollouts outputs/your_run/rollouts.jsonl --milestone "AI2027-SC"
    python scripts/analysis/modal_calendar_year.py --run-dir outputs/your_run/ --list-milestones
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from plotting_utils.rollouts_reader import RolloutsReader


def get_calendar_year(decimal_year: float) -> int:
    """Convert decimal year to calendar year (floor).

    Args:
        decimal_year: Year as decimal (e.g., 2027.9)

    Returns:
        Calendar year (e.g., 2027)
    """
    return int(np.floor(decimal_year))


def find_modal_calendar_year(
    milestone_times: List[float],
) -> Tuple[Optional[int], int, Counter]:
    """Find the modal calendar year from a list of milestone times.

    Args:
        milestone_times: List of decimal years when milestone was achieved

    Returns:
        Tuple of:
          - modal_year: The most common calendar year (None if no data)
          - modal_count: Number of rollouts achieving milestone in modal year
          - year_counts: Counter with all year counts
    """
    if not milestone_times:
        return None, 0, Counter()

    calendar_years = [get_calendar_year(t) for t in milestone_times]
    year_counts = Counter(calendar_years)

    modal_year, modal_count = year_counts.most_common(1)[0]
    return modal_year, modal_count, year_counts


def format_year_distribution(
    year_counts: Counter,
    modal_year: Optional[int],
    total_achieved: int,
    total_rollouts: int,
    top_n: int = 10,
) -> str:
    """Format year distribution for display.

    Args:
        year_counts: Counter with year counts
        modal_year: The modal year
        total_achieved: Total rollouts that achieved milestone
        total_rollouts: Total rollouts in dataset
        top_n: Number of top years to display

    Returns:
        Formatted string
    """
    lines = []

    if not year_counts:
        lines.append("No rollouts achieved this milestone.")
        return "\n".join(lines)

    # Summary stats
    all_years = list(year_counts.elements())
    median_year = get_calendar_year(float(np.median(all_years)))
    min_year = min(year_counts.keys())
    max_year = max(year_counts.keys())

    lines.append(f"Modal year:   {modal_year}")
    lines.append(f"Median year:  {median_year}")
    lines.append(f"Range:        {min_year} - {max_year}")
    lines.append(f"Achieved:     {total_achieved}/{total_rollouts} "
                 f"({100 * total_achieved / total_rollouts:.1f}%)")
    lines.append("")

    # Top years distribution
    lines.append(f"Top {min(top_n, len(year_counts))} years by frequency:")
    lines.append("-" * 40)

    for year, count in year_counts.most_common(top_n):
        pct = 100 * count / total_rollouts
        bar_len = int(pct / 2)  # Scale bar to fit
        bar = "#" * bar_len
        marker = " <-- MODAL" if year == modal_year else ""
        lines.append(f"  {year}: {count:5d} ({pct:5.1f}%) {bar}{marker}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find the modal calendar year for a milestone",
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
        "--milestone",
        type=str,
        default=None,
        help="Milestone name (e.g., 'AC', 'AI2027-SC', 'TED-AI')",
    )
    parser.add_argument(
        "--list-milestones",
        action="store_true",
        help="List all available milestone names in the rollouts file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show modal year for all milestones",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    # Determine paths
    if args.rollouts:
        rollouts_path = Path(args.rollouts)
        run_dir = rollouts_path.parent
    else:
        run_dir = Path(args.run_dir)
        rollouts_path = run_dir / "rollouts.jsonl"

    # RolloutsReader automatically uses cache if available, even if JSONL is missing.
    # Check for either rollouts.jsonl or rollouts.cache.json
    cache_path = run_dir / "rollouts.cache.json"
    if not rollouts_path.exists() and not cache_path.exists():
        print(f"Error: Neither {rollouts_path} nor {cache_path} found", file=sys.stderr)
        sys.exit(1)

    # RolloutsReader will use cache if JSONL is missing but cache exists
    try:
        reader = RolloutsReader(rollouts_path)
    except FileNotFoundError:
        # Try with cache path directly if JSONL doesn't exist
        if cache_path.exists():
            reader = RolloutsReader(cache_path)
        else:
            print(f"Error: {rollouts_path} not found", file=sys.stderr)
            sys.exit(1)
    total_rollouts = reader.count_rollouts()

    # Handle --list-milestones
    if args.list_milestones:
        milestone_names = reader.list_milestone_names()
        print("Available milestones:")
        for name in milestone_names:
            print(f"  - {name}")
        return

    # Handle --all flag
    if args.all:
        milestone_names = reader.list_milestone_names()
        results = []

        print("=" * 60)
        print("MODAL CALENDAR YEAR FOR ALL MILESTONES")
        print("=" * 60)
        print(f"Total rollouts: {total_rollouts}")
        print()
        print(f"{'Milestone':<45} {'Modal':<6} {'Count':<7} {'% of Total'}")
        print("-" * 70)

        for milestone_name in milestone_names:
            times, num_not_achieved, _ = reader.read_milestone_times(milestone_name)
            modal_year, modal_count, _ = find_modal_calendar_year(times)

            total_achieved = len(times)
            if total_achieved > 0 and modal_year is not None:
                pct_of_total = 100 * modal_count / total_rollouts
                print(f"{milestone_name:<45} {modal_year:<6} {modal_count:<7} {pct_of_total:.1f}%")
                results.append({
                    "milestone": milestone_name,
                    "modal_year": modal_year,
                    "modal_count": modal_count,
                    "total_achieved": total_achieved,
                    "total_rollouts": total_rollouts,
                })
            else:
                print(f"{milestone_name:<45} {'N/A':<6} {'0':<7} -")

        if args.json:
            import json
            print()
            print(json.dumps(results, indent=2))
        return

    # Single milestone mode
    if not args.milestone:
        print("Error: --milestone is required (or use --list-milestones or --all)",
              file=sys.stderr)
        sys.exit(1)

    milestone_name = args.milestone

    # Read milestone times (only achieved rollouts)
    times, num_not_achieved, typical_sim_end = reader.read_milestone_times(milestone_name)

    if not times:
        print(f"Error: No rollouts achieved milestone '{milestone_name}'", file=sys.stderr)
        print(f"  Total rollouts: {total_rollouts}")
        print(f"  Not achieved: {num_not_achieved}")
        if typical_sim_end:
            print(f"  Simulation end: {typical_sim_end:.1f}")
        sys.exit(1)

    # Find modal year
    modal_year, modal_count, year_counts = find_modal_calendar_year(times)

    if args.json:
        import json
        result = {
            "milestone": milestone_name,
            "modal_year": modal_year,
            "modal_count": modal_count,
            "total_achieved": len(times),
            "total_rollouts": total_rollouts,
            "year_distribution": dict(year_counts.most_common()),
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print(f"MODAL CALENDAR YEAR: {milestone_name}")
        print("=" * 60)
        print()
        print(format_year_distribution(
            year_counts,
            modal_year,
            len(times),
            total_rollouts,
        ))
        print()
        print("=" * 60)


if __name__ == "__main__":
    main()