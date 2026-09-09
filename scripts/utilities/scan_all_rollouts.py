#!/usr/bin/env python3
"""
Scan all output directories and generate a summary table of rollout statistics.

This script scans all subdirectories in the outputs folder and generates a
quick overview table showing success/failure rates for each run.

Usage:
    python scripts/scan_all_rollouts.py
    python scripts/scan_all_rollouts.py --output-dir outputs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RolloutsReader for unified rollout parsing
from plotting_utils.rollouts_reader import RolloutsReader


def get_rollout_stats(rollouts_path: Path) -> Dict[str, Any]:
    """Quickly analyze a rollouts.jsonl file and return basic statistics.

    Args:
        rollouts_path: Path to rollouts.jsonl file

    Returns:
        Dictionary with basic statistics
    """
    total = 0
    successful = 0
    failed = 0
    timeouts = 0

    # Use RolloutsReader to iterate over all records (including errors)
    reader = RolloutsReader(rollouts_path)

    for record in reader.iter_all_records():
        total += 1

        if "error" in record:
            failed += 1
            if "timeout" in record["error"].lower() or "timed out" in record["error"].lower():
                timeouts += 1
        else:
            results = record.get("results")
            if results is not None and isinstance(results, dict):
                successful += 1
            else:
                failed += 1

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "timeouts": timeouts,
        "success_rate": successful / total if total > 0 else 0.0
    }


def format_table(run_stats: List[Dict[str, Any]]) -> str:
    """Format run statistics as a table.

    Args:
        run_stats: List of statistics dictionaries

    Returns:
        Formatted table string
    """
    if not run_stats:
        return "No rollout runs found."

    lines = [
        "=" * 100,
        "ROLLOUT STATISTICS - ALL RUNS",
        "=" * 100,
        "",
        f"{'Run Directory':<30} {'Total':>8} {'Success':>8} {'Failed':>8} {'Timeout':>8} {'Success %':>10}",
        "-" * 100
    ]

    # Sort by directory name (most recent first if using timestamp format)
    run_stats_sorted = sorted(run_stats, key=lambda x: x["run_dir"], reverse=True)

    for stats in run_stats_sorted:
        lines.append(
            f"{stats['run_dir']:<30} "
            f"{stats['total']:>8d} "
            f"{stats['successful']:>8d} "
            f"{stats['failed']:>8d} "
            f"{stats['timeouts']:>8d} "
            f"{stats['success_rate']:>9.1%}"
        )

    # Add summary statistics
    total_all = sum(s["total"] for s in run_stats)
    success_all = sum(s["successful"] for s in run_stats)
    failed_all = sum(s["failed"] for s in run_stats)
    timeout_all = sum(s["timeouts"] for s in run_stats)
    success_rate_all = success_all / total_all if total_all > 0 else 0.0

    lines.extend([
        "-" * 100,
        f"{'TOTAL':<30} "
        f"{total_all:>8d} "
        f"{success_all:>8d} "
        f"{failed_all:>8d} "
        f"{timeout_all:>8d} "
        f"{success_rate_all:>9.1%}",
        "=" * 100
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan all rollout directories and generate summary table")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory to scan")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: stdout only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {output_dir}...")

    run_stats = []

    # Scan all subdirectories
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        rollouts_path = run_dir / "rollouts.jsonl"
        if not rollouts_path.exists():
            continue

        try:
            stats = get_rollout_stats(rollouts_path)
            stats["run_dir"] = run_dir.name
            run_stats.append(stats)
        except Exception as e:
            print(f"Warning: Error processing {run_dir.name}: {e}", file=sys.stderr)
            continue

    # Generate and display table
    table = format_table(run_stats)
    print("\n" + table)

    # Optionally write to file
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(table)
        print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
