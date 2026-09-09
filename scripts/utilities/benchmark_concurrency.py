#!/usr/bin/env python3
"""
Benchmark script to determine optimal concurrency setting for batch_rollout.py.

This script tests different concurrency levels and measures throughput (rollouts/second)
to help identify the optimal --num-workers setting for your hardware.

Usage:
  python scripts/benchmark_concurrency.py
  python scripts/benchmark_concurrency.py --num-rollouts 200 --max-workers 16
  python scripts/benchmark_concurrency.py --config config/sampling_config.yaml
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark concurrency settings for batch_rollout.py"
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=100,
        help="Number of rollouts to run for each concurrency test (default: 100)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sampling_config.yaml",
        help="Path to sampling config YAML (default: config/sampling_config.yaml)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of workers to test (default: CPU count)",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers to test (default: 1)",
    )
    parser.add_argument(
        "--worker-step",
        type=int,
        default=None,
        help="Step size between worker counts (default: auto-calculated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for reproducibility (default: 12345)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for benchmark results (default: auto-generated in outputs/)",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep rollout output directories (default: clean up after each run)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-sample timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="pipelines/batch_rollout.py",
        choices=["pipelines/batch_rollout.py", "pipelines/batch_rollout_legacy.py"],
        help="Which batch rollout script to benchmark (default: pipelines/batch_rollout.py)",
    )
    return parser.parse_args()


def determine_worker_counts(
    min_workers: int, max_workers: int, step: int = None
) -> List[int]:
    """Determine which worker counts to test."""
    cpu_count = os.cpu_count() or 4

    if max_workers is None:
        max_workers = cpu_count

    # Ensure max_workers doesn't exceed CPU count
    max_workers = min(max_workers, cpu_count)

    if step is None:
        # Auto-calculate step based on range
        if max_workers <= 8:
            step = 1
        elif max_workers <= 16:
            step = 2
        else:
            step = 4

    # Generate worker counts
    worker_counts = []

    # Always test sequential (1 worker)
    if min_workers <= 1:
        worker_counts.append(1)

    # Generate counts from min_workers to max_workers
    current = max(2, min_workers) if min_workers > 1 else 2
    while current <= max_workers:
        if current not in worker_counts:
            worker_counts.append(current)
        current += step

    # Always test max_workers if not already included
    if max_workers not in worker_counts and max_workers >= min_workers:
        worker_counts.append(max_workers)

    # Sort and return
    worker_counts.sort()
    return worker_counts


def run_benchmark(
    num_workers: int,
    num_rollouts: int,
    config_path: str = None,
    seed: int = 12345,
    timeout: float = 5.0,
    output_base: Path = None,
    script_name: str = "pipelines/batch_rollout.py",
) -> Dict[str, Any]:
    """Run a single benchmark with specified number of workers.

    Returns:
        Dict with timing results and metadata
    """
    print(f"\n{'='*80}")
    print(f"Testing {num_workers} worker(s)...")
    print(f"{'='*80}")

    # Prepare arguments for batch_rollout script
    python_exe = sys.executable
    rollout_script = SCRIPTS_DIR / script_name

    cmd = [
        python_exe,
        str(rollout_script),
        "--num-samples", str(num_rollouts),
        "--seed", str(seed),
        "--per-sample-timeout", str(timeout),
        "--num-workers", str(num_workers),
        "--skip-summary",  # Skip post-processing for speed
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    if output_base:
        cmd.extend(["--output-dir", str(output_base)])

    # Run and time
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed_time = time.perf_counter() - start_time
        success = True
        error = None
        stdout = result.stdout
    except subprocess.CalledProcessError as e:
        elapsed_time = time.perf_counter() - start_time
        success = False
        error = str(e)
        stdout = e.stdout
        print(f"ERROR: Benchmark failed: {error}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")

    # Calculate metrics
    if success and elapsed_time > 0:
        throughput = num_rollouts / elapsed_time
    else:
        throughput = 0.0

    return {
        "num_workers": num_workers,
        "num_rollouts": num_rollouts,
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "success": success,
        "error": error,
        "stdout": stdout,
    }


def print_summary_table(results: List[Dict[str, Any]], script_name: str = "pipelines/batch_rollout.py") -> None:
    """Print a formatted summary table of benchmark results."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - {script_name}")
    print(f"{'='*80}")

    # Find baseline (sequential) for speedup calculation
    baseline_time = None
    for r in results:
        if r["num_workers"] == 1 and r["success"]:
            baseline_time = r["elapsed_time"]
            break

    # Print header
    print(f"{'Workers':>8} | {'Time (s)':>10} | {'Rollouts/s':>12} | {'Speedup':>8} | {'Efficiency':>10} | {'Status':>10}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    # Print results
    best_throughput = 0
    best_workers = None

    for r in results:
        if not r["success"]:
            print(f"{r['num_workers']:>8} | {'N/A':>10} | {'N/A':>12} | {'N/A':>8} | {'N/A':>10} | {'FAILED':>10}")
            continue

        time_str = f"{r['elapsed_time']:.2f}"
        throughput_str = f"{r['throughput']:.2f}"

        if baseline_time and baseline_time > 0:
            speedup = baseline_time / r["elapsed_time"]
            speedup_str = f"{speedup:.2f}x"

            # Efficiency = speedup / num_workers (ideal is 1.0)
            efficiency = speedup / r["num_workers"]
            efficiency_str = f"{efficiency*100:.1f}%"
        else:
            speedup_str = "N/A"
            efficiency_str = "N/A"

        print(f"{r['num_workers']:>8} | {time_str:>10} | {throughput_str:>12} | {speedup_str:>8} | {efficiency_str:>10} | {'OK':>10}")

        # Track best
        if r["throughput"] > best_throughput:
            best_throughput = r["throughput"]
            best_workers = r["num_workers"]

    print(f"{'='*80}")

    if best_workers:
        print(f"\nOPTIMAL CONCURRENCY: {best_workers} workers ({best_throughput:.2f} rollouts/s)")

    # Print efficiency analysis
    print(f"\n{'='*80}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    print("\nParallel efficiency measures how well the speedup scales with worker count.")
    print("100% efficiency means perfect linear scaling (2x workers = 2x speed).")
    print("Diminishing returns typically occur due to:")
    print("  - CPU contention (more workers than cores)")
    print("  - Memory bandwidth limits")
    print("  - I/O bottlenecks")
    print("  - Python GIL (though multiprocessing uses separate processes)")
    print("\nRecommendation: Choose the worker count with highest throughput, or")
    print("                slightly lower if efficiency drops significantly.")


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save benchmark results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Workers",
            "Rollouts",
            "Time (s)",
            "Throughput (rollouts/s)",
            "Speedup",
            "Efficiency (%)",
            "Success",
            "Error",
        ])

        # Find baseline for speedup
        baseline_time = None
        for r in results:
            if r["num_workers"] == 1 and r["success"]:
                baseline_time = r["elapsed_time"]
                break

        # Data rows
        for r in results:
            if r["success"]:
                speedup = baseline_time / r["elapsed_time"] if baseline_time else None
                efficiency = (speedup / r["num_workers"] * 100) if speedup else None

                writer.writerow([
                    r["num_workers"],
                    r["num_rollouts"],
                    f"{r['elapsed_time']:.2f}",
                    f"{r['throughput']:.2f}",
                    f"{speedup:.2f}" if speedup else "N/A",
                    f"{efficiency:.1f}" if efficiency else "N/A",
                    "Success",
                    "",
                ])
            else:
                writer.writerow([
                    r["num_workers"],
                    r["num_rollouts"],
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "Failed",
                    r["error"] or "",
                ])

    print(f"\nResults saved to: {output_path}")


def cleanup_outputs(output_dir: Path) -> None:
    """Remove temporary output directories."""
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
            print(f"Cleaned up: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")


def main() -> None:
    args = parse_args()

    # Print configuration
    print(f"{'='*80}")
    print("CONCURRENCY BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Script: {args.script}")
    print(f"Number of rollouts per test: {args.num_rollouts}")
    print(f"Seed: {args.seed}")
    print(f"Per-sample timeout: {args.timeout}s")
    print(f"Config: {args.config}")
    print(f"CPU count: {os.cpu_count()}")

    # Determine worker counts to test
    worker_counts = determine_worker_counts(
        args.min_workers,
        args.max_workers,
        args.worker_step,
    )
    print(f"Testing worker counts: {worker_counts}")

    # Prepare output directory
    output_base = REPO_ROOT / "outputs" / "benchmark_concurrency"
    output_base.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = []
    for num_workers in worker_counts:
        result = run_benchmark(
            num_workers=num_workers,
            num_rollouts=args.num_rollouts,
            config_path=args.config,
            seed=args.seed,
            timeout=args.timeout,
            output_base=output_base,
            script_name=args.script,
        )
        results.append(result)

        # Clean up output directory if requested
        if not args.keep_outputs:
            # Find the most recent directory in output_base
            subdirs = sorted(output_base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if subdirs:
                cleanup_outputs(subdirs[0])

    # Print summary
    print_summary_table(results, args.script)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        script_label = "legacy" if "legacy" in args.script else "optimized"
        output_path = REPO_ROOT / "outputs" / f"benchmark_concurrency_{script_label}_{timestamp}.csv"

    save_results(results, output_path)

    # Final cleanup
    if not args.keep_outputs:
        cleanup_outputs(output_base)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
