#!/usr/bin/env python3
"""
Generate a samples.jsonl file from a rollouts.jsonl file.

Extracts sample_id, parameters, and time_series_parameters from each rollout,
discarding the results field.

Usage:
  python scripts/rollouts_to_samples.py outputs/run_dir/rollouts.jsonl
  python scripts/rollouts_to_samples.py outputs/run_dir/rollouts.jsonl -o samples_filtered.jsonl
  python scripts/rollouts_to_samples.py outputs/run_dir  # auto-detects rollouts.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def rollouts_to_samples(
    input_file: Path,
    output_file: Path,
    verbose: bool = True
) -> int:
    """Convert rollouts.jsonl to samples.jsonl format.

    Args:
        input_file: Path to input rollouts.jsonl
        output_file: Path to output samples.jsonl
        verbose: Print progress messages

    Returns:
        Number of samples written
    """
    count = 0

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

            # Extract only the sample fields (exclude results)
            sample = {
                "sample_id": rec.get("sample_id"),
                "parameters": rec.get("parameters"),
                "time_series_parameters": rec.get("time_series_parameters"),
            }

            # Write compact JSON
            f_out.write(json.dumps(sample, separators=(",", ":")) + "\n")
            count += 1

    if verbose:
        print(f"Converted {count} rollouts to samples")
        print(f"Output: {output_file}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples.jsonl from rollouts.jsonl",
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
        help="Output file path (default: samples.jsonl in same directory)"
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
        output_file = input_file.parent / "samples.jsonl"

    if not args.quiet:
        print(f"Input: {input_file}")

    count = rollouts_to_samples(
        input_file,
        output_file,
        verbose=not args.quiet
    )

    if count == 0:
        print("Warning: No rollouts found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()