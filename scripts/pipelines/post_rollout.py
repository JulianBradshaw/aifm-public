#!/usr/bin/env python3
"""
Post-rollout processing pipeline.

Orchestrates the generation of plots and analyses after a batch rollout completes.
Uses explicit lists of plots (not auto-discovery) for predictable behavior.

Usage:
    # Run all standard post-processing
    python -m pipelines.post_rollout --run-dir outputs/251209_updated_params

    # Skip slow trajectory plots
    python -m pipelines.post_rollout --run-dir outputs/251209_updated_params --fast-only

    # Run only histogram plots
    python -m pipelines.post_rollout --run-dir outputs/251209_updated_params --histograms-only
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Plot Lists - Explicit configuration of what runs in each phase
# These are the canonical lists of plots generated after each rollout
# ---------------------------------------------------------------------------

# Phase 1: Fast plots (use cache, can run in parallel, < 5 seconds each)
FAST_PLOTS: List[Dict] = [
    # Milestone time histograms
    {"name": "milestone_time_hist", "args": {"milestone": "AC"}},
    {"name": "milestone_time_hist", "args": {"milestone": "AI2027-SC"}},
    {"name": "milestone_time_hist", "args": {"milestone": "ASI"}},
    # PDF overlays
    {"name": "milestone_pdfs_overlay", "args": {"preset": "AC_SAR"}},
    {"name": "milestone_pdfs_overlay", "args": {"preset": "AC_SC_SAR"}},
    {"name": "milestone_pdfs_overlay", "args": {"preset": "POST_SAR"}},
]

# Phase 2: Medium plots (may need full JSONL but not too slow)
MEDIUM_PLOTS: List[Dict] = [
    {"name": "m_over_beta_hist", "args": {}},
    {"name": "takeoff_exponent_hist", "args": {}},
    {"name": "uplift_ratio_hist", "args": {}},
]

# Phase 3: Slow plots (full JSONL parsing, run sequentially)
SLOW_PLOTS: List[Dict] = [
    {"name": "metric_trajectories", "args": {"metric": "horizon_lengths", "max_trajectories": 100}},
]

# External scripts that are run but not (yet) converted to registry pattern
EXTERNAL_SCRIPTS: List[Dict] = [
    {"name": "Rollout summary", "script": "analysis/rollout_summary.py", "args": ["{run_dir}"]},
    {"name": "Milestone summary table", "script": "utilities/milestone_summary_table.py", "args": ["{run_dir}/rollouts.jsonl"]},
    {"name": "Short timelines analysis", "script": "analysis/short_timelines_analysis.py", "args": ["--run-dir", "{run_dir}"]},
    {"name": "Fast takeoff analysis", "script": "analysis/fast_takeoff_analysis.py", "args": ["--run-dir", "{run_dir}"]},
]


def _run_script(name: str, cmd: List[str]) -> Tuple[str, bool, Optional[str]]:
    """Run a script and return success/failure status."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            return name, False, result.stderr[:500] if result.stderr else "Unknown error"
        return name, True, None
    except subprocess.TimeoutExpired:
        return name, False, "Timeout after 600s"
    except Exception as e:
        return name, False, str(e)


def create_cache(run_dir: Path) -> bool:
    """Create the milestone cache for faster post-processing."""
    scripts_dir = Path(__file__).parent.parent
    python_exe = sys.executable

    try:
        print("\nCreating milestone cache for faster post-processing...")
        subprocess.run(
            [
                python_exe,
                str(scripts_dir / "utilities" / "create_milestones_cache.py"),
                str(run_dir / "rollouts.jsonl"),
                "--force",
            ],
            check=True,
            capture_output=True,
        )
        print("Cache created successfully!")
        return True
    except Exception as e:
        print(f"WARNING: Cache creation failed: {e}")
        return False


def run_fast_plots(run_dir: Path, max_workers: int = 4) -> None:
    """Run fast plots using the registry pattern."""
    # Import here to avoid circular imports and ensure registry is populated
    from plots.histograms.milestone_time import MilestoneTimeHistogramPlot
    from plots.pdfs.milestone_overlay import MilestonePDFOverlayPlot
    from plots.registry import PlotRegistry

    print(f"\nPhase 1: Running {len(FAST_PLOTS)} fast plots...")

    for plot_config in FAST_PLOTS:
        plot_name = plot_config["name"]
        args = plot_config["args"]

        try:
            plot_class = PlotRegistry.get_class(plot_name)
            if plot_class is None:
                print(f"  WARNING: Plot '{plot_name}' not found in registry")
                continue

            plot = plot_class(run_dir)
            output = plot.generate(**args)
            print(f"  ✓ {plot_name}: {output.name}")
        except Exception as e:
            print(f"  ✗ {plot_name}: {e}")


def run_medium_plots(run_dir: Path) -> None:
    """Run medium-speed plots."""
    from plots.histograms.m_over_beta import MOverBetaHistogramPlot
    from plots.histograms.takeoff_exponent import TakeoffExponentHistogramPlot
    from plots.histograms.uplift_ratio import UpliftRatioHistogramPlot
    from plots.registry import PlotRegistry

    print(f"\nPhase 2: Running {len(MEDIUM_PLOTS)} medium plots...")

    for plot_config in MEDIUM_PLOTS:
        plot_name = plot_config["name"]
        args = plot_config["args"]

        try:
            plot_class = PlotRegistry.get_class(plot_name)
            if plot_class is None:
                print(f"  WARNING: Plot '{plot_name}' not found in registry")
                continue

            plot = plot_class(run_dir)
            output = plot.generate(**args)
            print(f"  ✓ {plot_name}: {output.name}")
        except Exception as e:
            print(f"  ✗ {plot_name}: {e}")


def run_slow_plots(run_dir: Path) -> None:
    """Run slow plots that require full JSONL parsing."""
    from plots.registry import PlotRegistry
    from plots.trajectories.metric import MetricTrajectoryPlot

    print(f"\nPhase 3: Running {len(SLOW_PLOTS)} slow plots...")

    for plot_config in SLOW_PLOTS:
        plot_name = plot_config["name"]
        args = plot_config["args"]

        try:
            plot_class = PlotRegistry.get_class(plot_name)
            if plot_class is None:
                print(f"  WARNING: Plot '{plot_name}' not found in registry")
                continue

            plot = plot_class(run_dir)
            output = plot.generate(**args)
            print(f"  ✓ {plot_name}: {output.name}")
        except Exception as e:
            print(f"  ✗ {plot_name}: {e}")


def run_external_scripts(run_dir: Path, max_workers: int = 4) -> None:
    """Run external scripts not yet converted to registry pattern."""
    scripts_dir = Path(__file__).parent.parent
    python_exe = sys.executable

    print(f"\nRunning {len(EXTERNAL_SCRIPTS)} external scripts in parallel...")

    # Build commands with run_dir substituted
    commands = {}
    for script_config in EXTERNAL_SCRIPTS:
        name = script_config["name"]
        script = script_config["script"]
        args = [
            arg.format(run_dir=str(run_dir)) for arg in script_config["args"]
        ]
        cmd = [python_exe, str(scripts_dir / script)] + args
        commands[name] = cmd

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_script, name, cmd): name
            for name, cmd in commands.items()
        }

        completed = 0
        for future in as_completed(futures):
            name, success, error_msg = future.result()
            completed += 1
            if success:
                print(f"  [{completed}/{len(commands)}] {name}: complete!")
            else:
                print(f"  [{completed}/{len(commands)}] WARNING: {name} failed: {error_msg}")


def run_post_rollout(
    run_dir: Path,
    fast_only: bool = False,
    histograms_only: bool = False,
    skip_external: bool = False,
) -> None:
    """Run the complete post-rollout pipeline.

    Args:
        run_dir: Path to the run directory containing rollouts.jsonl
        fast_only: Skip slow trajectory plots
        histograms_only: Only run histogram plots
        skip_external: Skip external scripts (rollout_summary, etc.)
    """
    print(f"Running post-rollout pipeline for: {run_dir}")

    # Step 0: Create cache
    create_cache(run_dir)

    # Step 1: Fast plots (always run)
    run_fast_plots(run_dir)

    if not histograms_only:
        # Step 2: Medium plots
        run_medium_plots(run_dir)

        # Step 3: Slow plots (unless fast_only)
        if not fast_only:
            run_slow_plots(run_dir)

        # Step 4: External scripts
        if not skip_external:
            run_external_scripts(run_dir)

    print("\n" + "=" * 60)
    print("Post-rollout pipeline complete!")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-rollout processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory containing rollouts.jsonl",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Skip slow trajectory plots",
    )
    parser.add_argument(
        "--histograms-only",
        action="store_true",
        help="Only run histogram plots",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip external scripts (rollout_summary, etc.)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if not (run_dir / "rollouts.jsonl").exists():
        raise FileNotFoundError(f"rollouts.jsonl not found in: {run_dir}")

    run_post_rollout(
        run_dir,
        fast_only=args.fast_only,
        histograms_only=args.histograms_only,
        skip_external=args.skip_external,
    )


if __name__ == "__main__":
    main()
