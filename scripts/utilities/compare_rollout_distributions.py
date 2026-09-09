#!/usr/bin/env python3
"""
Compare milestone time distributions between two rollouts using the Kolmogorov-Smirnov test.

Usage:
    python scripts/compare_rollout_distributions.py <new_rollout_dir> <baseline_rollout_dir>

Example:
    python scripts/compare_rollout_distributions.py outputs/20251215_054024 outputs/20251213_231204
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from plotting_utils.rollouts_reader import RolloutsReader


# Key milestones to compare
MILESTONES_TO_COMPARE = [
    "AC",
    "AI2027-SC",
    "SAR-level-experiment-selection-skill",
    "SIAR-level-experiment-selection-skill",
    "TED-AI",
    "ASI",
    "AIR-5x",
    "AIR-25x",
    "AIR-250x",
]


def load_milestone_times(rollout_dir: Path) -> Dict[str, List[float]]:
    """Load milestone times from a rollout directory."""
    rollouts_file = rollout_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    reader = RolloutsReader(rollouts_file)

    # Use batch read for efficiency
    times_map, not_achieved, _, total = reader.read_milestone_times_batch(MILESTONES_TO_COMPARE)

    print(f"\nLoaded {total} rollouts from {rollout_dir.name}")
    for milestone in MILESTONES_TO_COMPARE:
        n_achieved = len(times_map[milestone])
        n_missed = not_achieved[milestone]
        pct = 100 * n_achieved / total if total > 0 else 0
        print(f"  {milestone}: {n_achieved}/{total} achieved ({pct:.1f}%)")

    return times_map


def compute_ks_statistics(
    times_new: Dict[str, List[float]],
    times_baseline: Dict[str, List[float]]
) -> List[Dict]:
    """Compute KS statistics for each milestone."""
    results = []

    for milestone in MILESTONES_TO_COMPARE:
        new = np.array(times_new.get(milestone, []))
        baseline = np.array(times_baseline.get(milestone, []))

        result = {
            "milestone": milestone,
            "n_new": len(new),
            "n_baseline": len(baseline),
        }

        # Need at least 2 samples in each to compute KS
        if len(new) >= 2 and len(baseline) >= 2:
            # Two-sample KS test
            ks_stat, p_value = stats.ks_2samp(new, baseline)

            result["ks_statistic"] = ks_stat
            result["p_value"] = p_value
            result["significant_001"] = p_value < 0.001
            result["significant_01"] = p_value < 0.01
            result["significant_05"] = p_value < 0.05

            # Descriptive statistics
            result["mean_new"] = np.mean(new)
            result["mean_baseline"] = np.mean(baseline)
            result["mean_diff"] = result["mean_new"] - result["mean_baseline"]
            result["median_new"] = np.median(new)
            result["median_baseline"] = np.median(baseline)
            result["median_diff"] = result["median_new"] - result["median_baseline"]
            result["std_new"] = np.std(new)
            result["std_baseline"] = np.std(baseline)
        else:
            result["ks_statistic"] = None
            result["p_value"] = None
            result["significant_001"] = None
            result["significant_01"] = None
            result["significant_05"] = None

        results.append(result)

    return results


def print_results(results: List[Dict], new_name: str, baseline_name: str):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print(f"KS Test Results: {new_name} vs {baseline_name}")
    print("=" * 100)

    # Header
    print(f"\n{'Milestone':<45} {'KS Stat':>8} {'p-value':>12} {'Sig?':>6} {'Mean Diff':>12} {'Med Diff':>10}")
    print("-" * 100)

    for r in results:
        milestone = r["milestone"]

        if r["ks_statistic"] is not None:
            ks_stat = f"{r['ks_statistic']:.4f}"

            # Format p-value with scientific notation for small values
            if r['p_value'] < 0.0001:
                p_value = f"{r['p_value']:.2e}"
            else:
                p_value = f"{r['p_value']:.4f}"

            # Significance indicator
            if r["significant_001"]:
                sig = "***"
            elif r["significant_01"]:
                sig = "**"
            elif r["significant_05"]:
                sig = "*"
            else:
                sig = ""

            mean_diff = f"{r['mean_diff']:+.3f}" if r.get('mean_diff') is not None else "N/A"
            med_diff = f"{r['median_diff']:+.3f}" if r.get('median_diff') is not None else "N/A"

            print(f"{milestone:<45} {ks_stat:>8} {p_value:>12} {sig:>6} {mean_diff:>12} {med_diff:>10}")
        else:
            print(f"{milestone:<45} {'N/A':>8} {'N/A':>12} {'':>6} {'N/A':>12} {'N/A':>10}")

    print("-" * 100)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    print("Mean/Median Diff: New - Baseline (positive = later in new rollout)")

    # Summary
    significant_count = sum(1 for r in results if r.get("significant_05"))
    testable_count = sum(1 for r in results if r["ks_statistic"] is not None)

    print(f"\nSummary: {significant_count}/{testable_count} milestones show significant differences (p<0.05)")

    # Highlight major differences
    print("\n" + "=" * 100)
    print("Milestones with SIGNIFICANT differences (p<0.05):")
    print("=" * 100)

    has_significant = False
    for r in results:
        if r.get("significant_05"):
            has_significant = True
            milestone = r["milestone"]
            direction = "LATER" if r["mean_diff"] > 0 else "EARLIER"
            print(f"\n  {milestone}:")
            print(f"    KS statistic: {r['ks_statistic']:.4f}, p-value: {r['p_value']:.2e}")
            print(f"    New mean: {r['mean_new']:.3f}, Baseline mean: {r['mean_baseline']:.3f}")
            print(f"    Difference: {r['mean_diff']:+.3f} years ({direction} in new rollout)")

    if not has_significant:
        print("\n  No significant differences found!")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    new_dir = Path(sys.argv[1])
    baseline_dir = Path(sys.argv[2])

    if not new_dir.exists():
        print(f"Error: New rollout directory not found: {new_dir}")
        sys.exit(1)

    if not baseline_dir.exists():
        print(f"Error: Baseline rollout directory not found: {baseline_dir}")
        sys.exit(1)

    print("Loading milestone times...")

    times_new = load_milestone_times(new_dir)
    times_baseline = load_milestone_times(baseline_dir)

    print("\nComputing KS statistics...")
    results = compute_ks_statistics(times_new, times_baseline)

    print_results(results, new_dir.name, baseline_dir.name)


if __name__ == "__main__":
    main()
