#!/usr/bin/env python3
"""
Filter rollouts.jsonl to create a subset matching specified conditions.

Supports filtering by:
- Parameter values (e.g., --param "r_software > 1.0")
- Metric values at a specific time (e.g., --metric "horizon_lengths > 5 @ 2030")
- Milestone timing (e.g., --milestone "AC < 2028")

Examples:
  # Filter by parameter value
  python scripts/filter_rollouts.py outputs/run_dir --param "r_software > 1.0"

  # Filter by milestone timing
  python scripts/filter_rollouts.py outputs/run_dir --milestone "AC < 2028"

  # Filter by metric value at a specific year
  python scripts/filter_rollouts.py outputs/run_dir --metric "horizon_lengths > 5 @ 2030"

  # Combine multiple conditions (AND logic)
  python scripts/filter_rollouts.py outputs/run_dir \\
    --param "r_software > 1.0" \\
    --milestone "AC < 2030" \\
    --output filtered_fast_takeoff.jsonl

  # Filter where milestone was NOT achieved (use 'none' keyword)
  python scripts/filter_rollouts.py outputs/run_dir --milestone "ASI == none"

  # Filter where milestone WAS achieved (any time)
  python scripts/filter_rollouts.py outputs/run_dir --milestone "ASI != none"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def parse_comparison(expr: str) -> Tuple[str, str, str]:
    """Parse a comparison expression like 'param > 1.0' or 'milestone == none'.

    Returns:
        Tuple of (name, operator, value_str)
    """
    # Match: name operator value
    # Operators: >, <, >=, <=, ==, !=
    pattern = r'^([a-zA-Z_][a-zA-Z0-9_\-]*)\s*(>=|<=|==|!=|>|<)\s*(.+)$'
    match = re.match(pattern, expr.strip())
    if not match:
        raise ValueError(f"Invalid comparison expression: '{expr}'. "
                        f"Expected format: 'name operator value' (e.g., 'r_software > 1.0')")
    return match.group(1), match.group(2), match.group(3).strip()


def parse_metric_expr(expr: str) -> Tuple[str, str, float, float]:
    """Parse a metric-at-time expression like 'horizon_lengths > 5 @ 2030'.

    Returns:
        Tuple of (metric_name, operator, threshold, year)
    """
    # Match: metric_name operator value @ year
    pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*([\d.eE+-]+)\s*@\s*([\d.]+)$'
    match = re.match(pattern, expr.strip())
    if not match:
        raise ValueError(f"Invalid metric expression: '{expr}'. "
                        f"Expected format: 'metric > value @ year' (e.g., 'horizon_lengths > 5 @ 2030')")
    return match.group(1), match.group(2), float(match.group(3)), float(match.group(4))


def make_comparator(op: str) -> Callable[[float, float], bool]:
    """Return a comparison function for the given operator."""
    ops = {
        '>': lambda a, b: a > b,
        '<': lambda a, b: a < b,
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: abs(a - b) < 1e-9,
        '!=': lambda a, b: abs(a - b) >= 1e-9,
    }
    return ops[op]


def get_param_value(rec: Dict[str, Any], param_name: str) -> Optional[float]:
    """Extract a parameter value from a rollout record.

    Searches in: parameters, time_series_parameters, results
    """
    params = rec.get("parameters") or {}
    ts_params = rec.get("time_series_parameters") or {}
    results = rec.get("results") or {}

    for source in [params, ts_params, results]:
        if not isinstance(source, dict):
            continue
        if param_name in source:
            val = source[param_name]
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)
    return None


def get_milestone_time(rec: Dict[str, Any], milestone_name: str) -> Optional[float]:
    """Extract milestone arrival time from a rollout record.

    Returns None if milestone not achieved.
    """
    results = rec.get("results")
    if not isinstance(results, dict):
        return None
    milestones = results.get("milestones", {})

    milestone_info = milestones.get(milestone_name)
    if isinstance(milestone_info, dict):
        time_val = milestone_info.get("time")
        if time_val is not None:
            try:
                t = float(time_val)
                if np.isfinite(t):
                    return t
            except (TypeError, ValueError):
                pass
    return None


def get_metric_at_time(
    rec: Dict[str, Any],
    metric_name: str,
    target_year: float
) -> Optional[float]:
    """Get interpolated metric value at a specific time.

    Returns None if metric data not available.
    """
    results = rec.get("results")
    if not isinstance(results, dict):
        return None
    times = results.get("times")
    metric = results.get(metric_name)

    if times is None or metric is None:
        return None

    try:
        times_arr = np.asarray(times, dtype=float)
        metric_arr = np.asarray(metric, dtype=float)
    except (TypeError, ValueError):
        return None

    if times_arr.ndim != 1 or metric_arr.ndim != 1:
        return None
    if times_arr.size != metric_arr.size or times_arr.size == 0:
        return None

    # Check if target year is within range
    if target_year < times_arr[0] or target_year > times_arr[-1]:
        return None

    # Interpolate
    value = float(np.interp(target_year, times_arr, metric_arr))
    return value if np.isfinite(value) else None


class RolloutFilter:
    """Composable filter for rollout records."""

    def __init__(self):
        self.param_filters: List[Tuple[str, str, float]] = []
        self.milestone_filters: List[Tuple[str, str, str]] = []  # (name, op, value_str)
        self.metric_filters: List[Tuple[str, str, float, float]] = []  # (name, op, threshold, year)

    def add_param_filter(self, expr: str) -> None:
        """Add a parameter filter from expression string."""
        name, op, value_str = parse_comparison(expr)
        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"Parameter filter value must be numeric: '{value_str}'")
        self.param_filters.append((name, op, value))

    def add_milestone_filter(self, expr: str) -> None:
        """Add a milestone timing filter from expression string."""
        name, op, value_str = parse_comparison(expr)
        self.milestone_filters.append((name, op, value_str))

    def add_metric_filter(self, expr: str) -> None:
        """Add a metric-at-time filter from expression string."""
        name, op, threshold, year = parse_metric_expr(expr)
        self.metric_filters.append((name, op, threshold, year))

    def matches(self, rec: Dict[str, Any]) -> bool:
        """Check if a rollout record matches all filter conditions."""
        # Check parameter filters
        for param_name, op, threshold in self.param_filters:
            value = get_param_value(rec, param_name)
            if value is None:
                return False
            if not make_comparator(op)(value, threshold):
                return False

        # Check milestone filters
        for milestone_name, op, value_str in self.milestone_filters:
            milestone_time = get_milestone_time(rec, milestone_name)

            # Handle 'none' keyword for not-achieved milestones
            if value_str.lower() == 'none':
                if op == '==':
                    if milestone_time is not None:
                        return False
                elif op == '!=':
                    if milestone_time is None:
                        return False
                else:
                    raise ValueError(f"Only == and != operators supported with 'none': "
                                   f"'{milestone_name} {op} none'")
            else:
                # Numeric comparison
                try:
                    threshold = float(value_str)
                except ValueError:
                    raise ValueError(f"Milestone filter value must be numeric or 'none': "
                                   f"'{value_str}'")

                if milestone_time is None:
                    return False
                if not make_comparator(op)(milestone_time, threshold):
                    return False

        # Check metric-at-time filters
        for metric_name, op, threshold, year in self.metric_filters:
            value = get_metric_at_time(rec, metric_name, year)
            if value is None:
                return False
            if not make_comparator(op)(value, threshold):
                return False

        return True

    def describe(self) -> str:
        """Return human-readable description of filter conditions."""
        conditions = []
        for name, op, val in self.param_filters:
            conditions.append(f"param:{name} {op} {val}")
        for name, op, val in self.milestone_filters:
            conditions.append(f"milestone:{name} {op} {val}")
        for name, op, threshold, year in self.metric_filters:
            conditions.append(f"metric:{name} {op} {threshold} @ {year}")
        return " AND ".join(conditions) if conditions else "(no filters)"


def filter_rollouts(
    input_file: Path,
    output_file: Path,
    filter_obj: RolloutFilter,
    verbose: bool = True
) -> Tuple[int, int]:
    """Filter rollouts from input file and write matches to output file.

    Args:
        input_file: Path to input rollouts.jsonl
        output_file: Path to output filtered rollouts.jsonl
        filter_obj: RolloutFilter with conditions to apply
        verbose: Print progress messages

    Returns:
        Tuple of (total_count, matched_count)
    """
    total = 0
    matched = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as f_in, \
         output_file.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            if filter_obj.matches(rec):
                f_out.write(line + "\n")
                matched += 1

    if verbose:
        print(f"Filtered {total} rollouts -> {matched} matched ({100*matched/total:.1f}%)")
        print(f"Output: {output_file}")

    return total, matched


def main():
    parser = argparse.ArgumentParser(
        description="Filter rollouts.jsonl to create a subset matching specified conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to rollouts.jsonl file or run directory containing it"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: filtered_rollouts.jsonl in same directory)"
    )
    parser.add_argument(
        "--param", "-p",
        type=str,
        action="append",
        default=[],
        metavar="EXPR",
        help="Parameter filter: 'param_name op value' (e.g., 'r_software > 1.0')"
    )
    parser.add_argument(
        "--milestone", "-m",
        type=str,
        action="append",
        default=[],
        metavar="EXPR",
        help="Milestone filter: 'milestone op year' or 'milestone == none' (e.g., 'AC < 2028')"
    )
    parser.add_argument(
        "--metric",
        type=str,
        action="append",
        default=[],
        metavar="EXPR",
        help="Metric-at-time filter: 'metric op value @ year' (e.g., 'horizon_lengths > 5 @ 2030')"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Resolve input file
    input_path = Path(args.input_path)
    if input_path.is_dir():
        input_file = input_path / "rollouts.jsonl"
    else:
        input_file = input_path

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Resolve output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / "filtered_rollouts.jsonl"

    # Build filter
    filter_obj = RolloutFilter()

    for expr in args.param:
        filter_obj.add_param_filter(expr)

    for expr in args.milestone:
        filter_obj.add_milestone_filter(expr)

    for expr in args.metric:
        filter_obj.add_metric_filter(expr)

    if not args.quiet:
        print(f"Input: {input_file}")
        print(f"Filter: {filter_obj.describe()}")
        print()

    # Run filter
    total, matched = filter_rollouts(
        input_file,
        output_file,
        filter_obj,
        verbose=not args.quiet
    )

    if matched == 0:
        print("Warning: No rollouts matched the filter criteria", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
