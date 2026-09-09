#!/usr/bin/env python3
"""
Generate a summary table for key milestones showing:
- Median arrival date
- Difference between median arrival dates
- Median transition duration (assuming achieved at simulation cutoff)
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import RolloutsReader for efficient data loading (auto-uses cache when available)
sys.path.insert(0, str(Path(__file__).parent.parent))
from plotting_utils.rollouts_reader import RolloutsReader


def clean_milestone_name(name: str) -> str:
    """Remove 'level' and 'skill' from milestone names for cleaner display."""
    return name.replace("-level", "").replace("-skill", "")


def extract_milestone_data(
    rollouts_file: Path,
    milestone_names: List[str]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[Tuple[str, str], Tuple[List[float], List[float]]]]:
    """
    Extract arrival times and transition durations for given milestones.

    Uses RolloutsReader which automatically uses cache when available (100-1000x faster).

    Returns:
        - arrival_times: dict mapping milestone name to list of arrival times (only achieved)
        - arrival_times_with_censored: dict mapping milestone name to list of arrival times (including censored at sim end)
        - transitions_with_censored: dict mapping (from, to) pair to list of durations (including censored)
        - conditional_arrival_times: dict mapping (prev, curr) pair to tuple of (prev_times, curr_times) where prev was achieved
    """
    arrival_times: Dict[str, List[float]] = {name: [] for name in milestone_names}
    arrival_times_with_censored: Dict[str, List[float]] = {name: [] for name in milestone_names}

    # For transitions, we'll track consecutive pairs
    transitions_with_censored: Dict[Tuple[str, str], List[float]] = {}
    # For conditional arrival times (for delta calculation)
    conditional_arrival_times: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    for i in range(len(milestone_names) - 1):
        transitions_with_censored[(milestone_names[i], milestone_names[i+1])] = []
        conditional_arrival_times[(milestone_names[i], milestone_names[i+1])] = ([], [])

    # Use RolloutsReader for efficient iteration (auto-uses cache if available)
    reader = RolloutsReader(rollouts_file)

    for rollout in reader.iter_normalized_rollouts():
        simulation_end = rollout.simulation_end

        # Extract arrival times for each milestone
        milestone_times: Dict[str, Optional[float]] = {}
        for name in milestone_names:
            m = rollout.milestones.get(name)
            if isinstance(m, dict) and m.get("time") is not None:
                try:
                    t = float(m["time"])
                    if np.isfinite(t):
                        milestone_times[name] = t
                        arrival_times[name].append(t)
                        arrival_times_with_censored[name].append(t)
                    else:
                        milestone_times[name] = None
                        # Not achieved - use simulation end
                        arrival_times_with_censored[name].append(simulation_end)
                except Exception:
                    milestone_times[name] = None
                    # Not achieved - use simulation end
                    arrival_times_with_censored[name].append(simulation_end)
            else:
                milestone_times[name] = None
                # Not achieved - use simulation end
                arrival_times_with_censored[name].append(simulation_end)

        # Calculate transition durations (with censored data)
        for i in range(len(milestone_names) - 1):
            from_name = milestone_names[i]
            to_name = milestone_names[i + 1]

            t_from = milestone_times.get(from_name)
            t_to = milestone_times.get(to_name)

            if t_from is None:
                # If first milestone not achieved, skip
                continue

            # For conditional arrival times: track both milestone times when prev is achieved
            use_t_to = t_to if t_to is not None else simulation_end
            prev_list, curr_list = conditional_arrival_times[(from_name, to_name)]
            prev_list.append(t_from)
            curr_list.append(use_t_to)

            if t_to is None:
                # Second milestone not achieved - use simulation end as censored value
                if simulation_end > t_from:
                    duration = simulation_end - t_from
                    transitions_with_censored[(from_name, to_name)].append(duration)
            elif t_to > t_from:
                # Both achieved, normal case
                duration = t_to - t_from
                transitions_with_censored[(from_name, to_name)].append(duration)

    return arrival_times, arrival_times_with_censored, transitions_with_censored, conditional_arrival_times


def format_year(value: float) -> str:
    """Format a year value."""
    return f"{value:.1f}"


def format_duration(value: float) -> str:
    """Format a duration in years."""
    return f"{value:.1f}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate milestone summary table")
    parser.add_argument("rollouts_file", type=Path, help="Path to rollouts.jsonl file")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV file path (default: same directory as rollouts file)")
    args = parser.parse_args()

    if not args.rollouts_file.exists():
        print(f"Error: File not found: {args.rollouts_file}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.rollouts_file.parent / "milestone_summary.csv"

    # Milestones to analyze
    milestone_names = [
        "AC",
        "SAR-level-experiment-selection-skill",
        "SIAR-level-experiment-selection-skill"
    ]

    print("Extracting data from rollouts...")
    arrival_times, arrival_times_with_censored, transitions_with_censored, conditional_arrival_times = extract_milestone_data(
        args.rollouts_file,
        milestone_names
    )

    # Prepare data for CSV
    rows = []
    prev_median_arrival = None

    for i, name in enumerate(milestone_names):
        times = arrival_times[name]
        times_censored = arrival_times_with_censored[name]

        # Median arrival date (including censored data)
        if len(times_censored) > 0:
            median_arrival = np.median(times_censored)
            median_arrival_str = format_year(median_arrival)
        else:
            median_arrival = None
            median_arrival_str = "N/A"

        # (a) Difference from previous median arrival (all rollouts + censored)
        if i == 0 or prev_median_arrival is None or median_arrival is None:
            delta_all_str = "-"
        else:
            delta_all = median_arrival - prev_median_arrival
            delta_all_str = format_duration(delta_all)

        # (b) Difference from previous median arrival (conditional on previous achieved)
        if i == 0:
            delta_conditional_str = "-"
        else:
            prev_name = milestone_names[i - 1]
            prev_times, curr_times = conditional_arrival_times.get((prev_name, name), ([], []))
            if len(prev_times) > 0 and len(curr_times) > 0:
                median_prev_cond = np.median(prev_times)
                median_curr_cond = np.median(curr_times)
                delta_conditional = median_curr_cond - median_prev_cond
                delta_conditional_str = format_duration(delta_conditional)
            else:
                delta_conditional_str = "N/A"

        # Median transition from previous (with censored data)
        if i == 0:
            median_transition_str = "-"
        else:
            prev_name = milestone_names[i - 1]
            trans_durations = transitions_with_censored.get((prev_name, name), [])
            if len(trans_durations) > 0:
                median_transition = np.median(trans_durations)
                median_transition_str = format_duration(median_transition)
            else:
                median_transition_str = "N/A"

        # Achievement statistics
        total_runs = len(arrival_times["AC"]) if "AC" in arrival_times else len(times)
        achieved_count = len(times)
        achieved_pct = 100.0 * achieved_count / total_runs if total_runs > 0 else 0.0

        rows.append({
            'Milestone': clean_milestone_name(name),
            'Median Arrival': median_arrival_str,
            'Delta (All+Censored)': delta_all_str,
            'Delta (Prev Achieved)': delta_conditional_str,
            'Median Transition': median_transition_str,
            'Achieved Count': achieved_count,
            'Total Runs': total_runs,
            'Achievement Rate (%)': f"{achieved_pct:.1f}"
        })

        prev_median_arrival = median_arrival

    # Write to CSV
    with output_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['Milestone', 'Median Arrival', 'Delta (All+Censored)', 'Delta (Prev Achieved)',
                      'Median Transition', 'Achieved Count', 'Total Runs', 'Achievement Rate (%)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTable written to: {output_path}")
    print("\nSummary:")
    for row in rows:
        print(f"  {row['Milestone']}: Median arrival {row['Median Arrival']}, "
              f"Δ(All)={row['Delta (All+Censored)']}, "
              f"Δ(Cond)={row['Delta (Prev Achieved)']}, "
              f"Transition={row['Median Transition']}, "
              f"Achieved {row['Achievement Rate (%)']}%")


if __name__ == "__main__":
    main()
