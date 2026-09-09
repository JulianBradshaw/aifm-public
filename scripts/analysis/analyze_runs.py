#!/usr/bin/env python3
"""
Utility to analyze and compare run directories in the outputs folder.

Usage:
    python scripts/analyze_runs.py                    # Show summary table
    python scripts/analyze_runs.py --detailed         # Show detailed view
    python scripts/analyze_runs.py --compare ID1 ID2  # Compare two runs
    python scripts/analyze_runs.py --group-by-config  # Group runs by config
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import argparse


class RunAnalyzer:
    """Analyzes run directories in the outputs folder."""

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.runs: List[Dict] = []
        self._scan_runs()

    def _scan_runs(self):
        """Scan all run directories and extract metadata."""
        if not self.outputs_dir.exists():
            print(f"Error: {self.outputs_dir} does not exist")
            return

        # Get all directories that look like run directories (timestamp format)
        run_dirs = sorted([d for d in self.outputs_dir.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            run_info = self._extract_run_info(run_dir)
            if run_info:
                self.runs.append(run_info)

    def _extract_run_info(self, run_dir: Path) -> Optional[Dict]:
        """Extract information from a single run directory."""
        metadata_path = run_dir / "metadata.json"
        config_path = run_dir / "model_config_snapshot.json"
        input_dist_path = run_dir / "input_distributions.yaml"
        samples_path = run_dir / "samples.jsonl"
        rollouts_path = run_dir / "rollouts.jsonl"

        info = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "num_samples": None,
            "num_rollouts": None,
            "seed": None,
            "created_at": None,
            "input_data": None,
            "config_hash": None,
            "input_dist_hash": None,
            "has_metadata": metadata_path.exists(),
            "has_config": config_path.exists(),
            "has_input_dist": input_dist_path.exists(),
            "has_samples": samples_path.exists(),
            "has_rollouts": rollouts_path.exists(),
        }

        # Read metadata if available
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                info["num_samples"] = metadata.get("num_samples")
                info["seed"] = metadata.get("seed")
                info["created_at"] = metadata.get("created_at")
                info["input_data"] = Path(metadata.get("input_data_path", "")).name
            except Exception as e:
                print(f"Warning: Could not read {metadata_path}: {e}", file=sys.stderr)

        # Count actual samples/rollouts if files exist
        if samples_path.exists():
            try:
                with open(samples_path) as f:
                    info["num_samples"] = sum(1 for _ in f)
            except Exception:
                pass

        if rollouts_path.exists():
            try:
                with open(rollouts_path) as f:
                    info["num_rollouts"] = sum(1 for _ in f)
            except Exception:
                pass

        # Hash config for comparison
        if config_path.exists():
            try:
                with open(config_path, 'rb') as f:
                    info["config_hash"] = hashlib.md5(f.read()).hexdigest()[:8]
            except Exception:
                pass

        # Hash input distributions for comparison
        if input_dist_path.exists():
            try:
                with open(input_dist_path, 'rb') as f:
                    info["input_dist_hash"] = hashlib.md5(f.read()).hexdigest()[:8]
            except Exception:
                pass

        return info

    def print_summary_table(self):
        """Print a summary table of all runs."""
        if not self.runs:
            print("No runs found.")
            return

        # Print header
        print("\n" + "="*150)
        print(f"{'Run ID':<20} {'Date':<18} {'Samples':<10} {'Rollouts':<10} {'Seed':<15} {'Config':<10} {'Input':<10} {'Input Data':<25}")
        print("="*150)

        # Print each run
        for run in self.runs:
            samples = str(run["num_samples"]) if run["num_samples"] is not None else "?"
            rollouts = str(run["num_rollouts"]) if run["num_rollouts"] is not None else "?"
            seed = str(run["seed"]) if run["seed"] is not None else "?"
            config = run["config_hash"] or "?"
            input_dist = run["input_dist_hash"] or "?"
            input_data = run["input_data"] or "?"

            # Extract date from created_at (format: 2025-09-30T11:26:06.079791)
            date_str = "?"
            if run["created_at"]:
                try:
                    # Extract just the date and time without microseconds
                    date_str = run["created_at"].split('.')[0].replace('T', ' ')
                except Exception:
                    date_str = "?"

            print(f"{run['run_id']:<20} {date_str:<18} {samples:<10} {rollouts:<10} {seed:<15} {config:<10} {input_dist:<10} {input_data:<25}")

        print("="*150)
        print(f"Total runs: {len(self.runs)}\n")

    def print_detailed(self):
        """Print detailed information for all runs."""
        if not self.runs:
            print("No runs found.")
            return

        for i, run in enumerate(self.runs):
            print(f"\n{'='*80}")
            print(f"Run {i+1}/{len(self.runs)}: {run['run_id']}")
            print(f"{'='*80}")
            print(f"Path:           {run['path']}")
            print(f"Created:        {run['created_at'] or 'Unknown'}")
            print(f"Samples:        {run['num_samples'] or 'Unknown'}")
            print(f"Rollouts:       {run['num_rollouts'] or 'Unknown'}")
            print(f"Seed:           {run['seed'] or 'Unknown'}")
            print(f"Input Data:     {run['input_data'] or 'Unknown'}")
            print(f"Config Hash:    {run['config_hash'] or 'Unknown'}")
            print(f"Input Dist:     {run['input_dist_hash'] or 'Unknown'}")
            print(f"\nFiles present:")
            print(f"  Metadata:     {'✓' if run['has_metadata'] else '✗'}")
            print(f"  Config:       {'✓' if run['has_config'] else '✗'}")
            print(f"  Input Dist:   {'✓' if run['has_input_dist'] else '✗'}")
            print(f"  Samples:      {'✓' if run['has_samples'] else '✗'}")
            print(f"  Rollouts:     {'✓' if run['has_rollouts'] else '✗'}")

    def group_by_config(self):
        """Group runs by their config hash."""
        if not self.runs:
            print("No runs found.")
            return

        # Group by config hash
        groups = defaultdict(list)
        for run in self.runs:
            key = (run["config_hash"], run["input_dist_hash"])
            groups[key].append(run)

        print(f"\n{'='*80}")
        print(f"Found {len(groups)} unique config combinations")
        print(f"{'='*80}\n")

        for i, ((config_hash, input_dist_hash), runs) in enumerate(sorted(groups.items()), 1):
            config_str = config_hash or "Unknown"
            input_str = input_dist_hash or "Unknown"
            print(f"Group {i}: Config={config_str}, Input={input_str} ({len(runs)} runs)")
            print("-" * 80)

            for run in runs:
                samples = run["num_samples"] or "?"
                rollouts = run["num_rollouts"] or "?"
                seed = run["seed"] or "?"
                print(f"  {run['run_id']:<20} Samples: {samples:<6} Rollouts: {rollouts:<6} Seed: {seed}")

            print()

    def compare_runs(self, run_id1: str, run_id2: str):
        """Compare two runs in detail."""
        run1 = next((r for r in self.runs if r["run_id"] == run_id1), None)
        run2 = next((r for r in self.runs if r["run_id"] == run_id2), None)

        if not run1:
            print(f"Error: Run {run_id1} not found")
            return
        if not run2:
            print(f"Error: Run {run_id2} not found")
            return

        print(f"\n{'='*80}")
        print(f"Comparing: {run_id1} vs {run_id2}")
        print(f"{'='*80}\n")

        # Compare basic info
        print(f"{'Attribute':<20} {'Run 1':<30} {'Run 2':<30} {'Same?':<10}")
        print("-" * 90)

        attrs = [
            ("Samples", "num_samples"),
            ("Rollouts", "num_rollouts"),
            ("Seed", "seed"),
            ("Input Data", "input_data"),
            ("Config Hash", "config_hash"),
            ("Input Dist Hash", "input_dist_hash"),
        ]

        for label, attr in attrs:
            val1 = str(run1[attr]) if run1[attr] is not None else "?"
            val2 = str(run2[attr]) if run2[attr] is not None else "?"
            same = "✓" if val1 == val2 else "✗"
            print(f"{label:<20} {val1:<30} {val2:<30} {same:<10}")

        print()

        # Check if configs are identical
        if run1["config_hash"] == run2["config_hash"] and run1["config_hash"] is not None:
            print("✓ These runs use the SAME model configuration")
        else:
            print("✗ These runs use DIFFERENT model configurations")

        if run1["input_dist_hash"] == run2["input_dist_hash"] and run1["input_dist_hash"] is not None:
            print("✓ These runs use the SAME input distributions")
        else:
            print("✗ These runs use DIFFERENT input distributions")

        print()

    def get_stats(self):
        """Get overall statistics."""
        if not self.runs:
            return {}

        total_samples = sum(r["num_samples"] for r in self.runs if r["num_samples"])
        total_rollouts = sum(r["num_rollouts"] for r in self.runs if r["num_rollouts"])
        unique_configs = len(set(r["config_hash"] for r in self.runs if r["config_hash"]))
        unique_inputs = len(set(r["input_dist_hash"] for r in self.runs if r["input_dist_hash"]))

        return {
            "total_runs": len(self.runs),
            "total_samples": total_samples,
            "total_rollouts": total_rollouts,
            "unique_configs": unique_configs,
            "unique_input_dists": unique_inputs,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze run directories in the outputs folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed information for each run"
    )
    parser.add_argument(
        "--group-by-config", "-g",
        action="store_true",
        help="Group runs by their configuration"
    )
    parser.add_argument(
        "--compare", "-c",
        nargs=2,
        metavar=("RUN1", "RUN2"),
        help="Compare two runs by their IDs"
    )
    parser.add_argument(
        "--outputs-dir", "-o",
        default="outputs",
        help="Path to outputs directory (default: outputs)"
    )

    args = parser.parse_args()

    analyzer = RunAnalyzer(args.outputs_dir)

    if args.compare:
        analyzer.compare_runs(args.compare[0], args.compare[1])
    elif args.group_by_config:
        analyzer.group_by_config()
    elif args.detailed:
        analyzer.print_detailed()
    else:
        analyzer.print_summary_table()

        # Print stats
        stats = analyzer.get_stats()
        if stats:
            print("Summary Statistics:")
            print(f"  Total runs:              {stats['total_runs']}")
            print(f"  Total samples:           {stats['total_samples']}")
            print(f"  Total rollouts:          {stats['total_rollouts']}")
            print(f"  Unique configurations:   {stats['unique_configs']}")
            print(f"  Unique input dists:      {stats['unique_input_dists']}")
            print()


if __name__ == "__main__":
    main()
