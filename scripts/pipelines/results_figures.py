#!/usr/bin/env python3
"""
Generate all figures for the results document.

This script automates the complete pipeline:
1. Run Monte Carlo batch rollouts (Eli, Daniel, and no-correlation configs)
2. Generate all analysis figures
3. Create a Markdown report with embedded figures

Usage:
  python scripts/generate_results_figures.py                     # Default (1000 samples)
  python scripts/generate_results_figures.py --num-samples 5000  # Custom sample count
  python scripts/generate_results_figures.py --skip-rollout --eli-dir outputs/xxx  # Use existing data
  python scripts/generate_results_figures.py --eli-only          # Only Eli's config
  python scripts/generate_results_figures.py --daniel-only       # Only Daniel's config
  python scripts/generate_results_figures.py --regenerate-plots outputs/results_xxx  # Regenerate plots only
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from numpy.random import SeedSequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_DIR = REPO_ROOT / "config"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Config files
ELI_CONFIG = CONFIG_DIR / "sampling_config.yaml"
DANIEL_CONFIG = CONFIG_DIR / "sampling_config_daniel.yaml"
NO_CORR_CONFIG = CONFIG_DIR / "sampling_config_2200_no_correlations.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all figures for the results document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Number of samples per rollout (default: 1000)"
    )
    parser.add_argument(
        "--skip-rollout", action="store_true",
        help="Skip rollout generation, use existing directories"
    )
    parser.add_argument(
        "--eli-dir", type=str, default=None,
        help="Existing Eli run directory (requires --skip-rollout)"
    )
    parser.add_argument(
        "--daniel-dir", type=str, default=None,
        help="Existing Daniel run directory (requires --skip-rollout)"
    )
    parser.add_argument(
        "--no-corr-dir", type=str, default=None,
        help="Existing no-correlation run directory (requires --skip-rollout)"
    )
    parser.add_argument(
        "--regenerate-plots", type=str, default=None,
        metavar="DIR",
        help="Regenerate plots for an existing output directory (auto-detects eli/daniel/eli_no_corr subdirs)"
    )
    parser.add_argument(
        "--eli-only", action="store_true",
        help="Only run Eli's config (skip Daniel and no-correlation)"
    )
    parser.add_argument(
        "--daniel-only", action="store_true",
        help="Only run Daniel's config (skip Eli and no-correlation)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/results_<timestamp>)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for rollouts (default: 42)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=16,
        help="Number of parallel workers (default: 16)"
    )
    parser.add_argument(
        "--skip-extended", action="store_true",
        help="Skip extended analyses (parameter splits) for faster execution"
    )
    parser.add_argument(
        "--website-only", action="store_true",
        help="Only generate plots needed for the website (faster horizon trajectories)"
    )
    return parser.parse_args()


def get_python_env() -> Dict[str, str]:
    """Get environment with PYTHONPATH set for imports."""
    import os
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if str(REPO_ROOT) not in pythonpath:
        env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)
    return env


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return (success, error_message)."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=False,  # Let output stream to console
            env=get_python_env()
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"Command failed with return code {e.returncode}"
    except Exception as e:
        return False, str(e)


def generate_rollouts_cache(run_dir: Path) -> bool:
    """Generate lightweight cache for faster analysis script execution.

    The cache excludes trajectory arrays, making it 100-1000x smaller and faster
    to load than the full rollouts.jsonl. Scripts using RolloutsReader will
    automatically use the cache when available.

    Returns True if cache was successfully created.
    """
    rollouts_file = run_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        print(f"WARNING: Cannot generate cache - {rollouts_file} not found")
        return False

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "utilities" / "create_milestones_cache.py"),
        str(rollouts_file),
        "--force"
    ]
    success, error = run_command(cmd, f"Generate cache: {run_dir.name}")
    if not success:
        print(f"WARNING: Cache generation failed: {error}")
    return success


def run_batch_rollout(
    config_path: Path,
    output_subdir: str,
    base_output_dir: Path,
    num_samples: int,
    seed: int,
    num_workers: int = 16
) -> Optional[Path]:
    """Run batch_rollout.py and return the run directory."""

    # The batch rollout creates its own timestamped directory
    # Use --skip-summary to avoid redundant analyses (generate_figures() runs them)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "pipelines" / "batch_rollout.py"),
        "--config", str(config_path),
        "--num-samples", str(num_samples),
        "--seed", str(seed),
        "--output-dir", str(base_output_dir / output_subdir),
        "--num-workers", str(num_workers),
        "--skip-summary",  # Skip auto post-processing; we run analyses separately
    ]

    # Get directories before running
    target_dir = base_output_dir / output_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    dirs_before = set(target_dir.iterdir()) if target_dir.exists() else set()

    success, error = run_command(cmd, f"Batch rollout: {config_path.name}")

    if not success:
        print(f"ERROR: Batch rollout failed: {error}")
        return None

    # Find the new directory
    dirs_after = set(target_dir.iterdir()) if target_dir.exists() else set()
    new_dirs = dirs_after - dirs_before

    run_dir = None
    if new_dirs:
        # Return the newest directory
        run_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
    elif (target_dir / "rollouts.jsonl").exists():
        # Fallback: return the target_dir itself if rollouts.jsonl exists there
        run_dir = target_dir
    else:
        # Last resort: find most recent subdirectory
        subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
        if subdirs:
            run_dir = max(subdirs, key=lambda p: p.stat().st_mtime)

    if run_dir is None:
        print(f"WARNING: Could not find run directory in {target_dir}")
        return None

    # Generate cache for fast analysis (100-1000x faster for milestone-only scripts)
    generate_rollouts_cache(run_dir)

    return run_dir


def find_run_dir_in_subdir(base_dir: Path, subdir_name: str) -> Optional[Path]:
    """Find the run directory within a subdirectory (e.g., eli/, daniel/, eli_no_corr/).

    The run directory is the one containing rollouts.jsonl or rollouts.cache.json.
    """
    subdir = base_dir / subdir_name
    if not subdir.exists():
        return None

    # Check if rollouts.jsonl exists directly in the subdir
    if (subdir / "rollouts.jsonl").exists() or (subdir / "rollouts.cache.json").exists():
        return subdir

    # Otherwise, look for timestamped subdirectories
    subdirs = [d for d in subdir.iterdir() if d.is_dir()]
    for d in sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "rollouts.jsonl").exists() or (d / "rollouts.cache.json").exists():
            return d

    return None


def generate_figures(run_dir: Path, run_name: str, skip_extended: bool = False, website_only: bool = False) -> Dict[str, bool]:
    """Generate all analysis figures for a run directory."""
    results = {}

    scripts = [
        ("analysis/short_timelines_analysis.py", "Short timelines analysis"),
        ("analysis/fast_takeoff_analysis.py", "Fast takeoff analysis"),
        ("analysis/timeline_takeoff_correlation_analysis.py", "Timeline-takeoff correlation"),
    ]

    for script_name, description in scripts:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / script_name),
            "--run-dir", str(run_dir)
        ]
        success, error = run_command(cmd, f"{description} ({run_name})")
        results[f"{run_name}/{script_name}"] = success
        if not success:
            print(f"WARNING: {script_name} failed: {error}")

    # run_extended_analyses.py (optional)
    # Use --results-page to only generate the 4 plots needed for /results (much faster)
    if not skip_extended:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "analysis" / "run_extended_analyses.py"),
            "--run-dir", str(run_dir),
            "--results-page"  # Only generate the 4 continuous plots for /results
        ]
        success, error = run_command(cmd, f"Extended analyses - results page ({run_name})")
        results[f"{run_name}/run_extended_analyses.py"] = success
    else:
        print(f"Skipping extended analyses for {run_name} (--skip-extended)")

    # plot_rollouts.py - horizon trajectories with MSE shading and filtering
    # Use --batch-horizon to generate all horizon trajectory variants:
    #   - horizon_trajectories_all_shaded.png (all trajectories with MSE shading)
    #   - horizon_trajectories_mse_1.0.png (MSE <= 1.0 filtered)
    #   - horizon_trajectories_mse_0.7.png, etc. (unless --website-only)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "plot_rollouts.py"),
        "--run-dir", str(run_dir),
        "--batch-horizon"
    ]
    if website_only:
        cmd.append("--website-only")
    success, error = run_command(cmd, f"Horizon trajectories batch ({run_name})")
    results[f"{run_name}/horizon_trajectories_batch"] = success

    # generate_sc_horizon_plots.py - SC-filtered trajectories with auto-computed median
    # Generates CSVs for:
    #   - Mar 2027 (fixed reference point)
    #   - Median SC date (dynamically computed from rollouts)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "utilities" / "generate_sc_horizon_plots.py"),
        str(run_dir),
        "--auto-median",
        "--include-mar-2027",
        "--output-manifest"
    ]
    success, error = run_command(cmd, f"SC horizon trajectories ({run_name})")
    results[f"{run_name}/sc_horizon_trajectories"] = success

    # plot_rollouts.py - m/beta histogram
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "plot_rollouts.py"),
        "--run-dir", str(run_dir),
        "--mode", "m_over_beta"
    ]
    success, error = run_command(cmd, f"m/beta histogram ({run_name})")
    results[f"{run_name}/m_over_beta_hist"] = success

    # plot_rollouts.py - takeoff exponent 2^(1-beta/m) histogram (output to fast_takeoff_outputs)
    fast_takeoff_dir = run_dir / "fast_takeoff_outputs"
    fast_takeoff_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "plot_rollouts.py"),
        "--run-dir", str(run_dir),
        "--mode", "takeoff_exponent",
        "--out", str(fast_takeoff_dir / "takeoff_exponent_hist.png")
    ]
    success, error = run_command(cmd, f"takeoff exponent histogram ({run_name})")
    results[f"{run_name}/takeoff_exponent_hist"] = success

    # plot_rollouts.py - uplift ratio 2^(beta/m-1) histogram (output to fast_takeoff_outputs)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "plot_rollouts.py"),
        "--run-dir", str(run_dir),
        "--mode", "uplift_ratio",
        "--out", str(fast_takeoff_dir / "uplift_ratio_hist.png")
    ]
    success, error = run_command(cmd, f"uplift ratio histogram ({run_name})")
    results[f"{run_name}/uplift_ratio_hist"] = success

    # sensitivity_analysis.py - parameter sensitivity bar charts
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "analysis" / "sensitivity_analysis.py"),
        "--run-dir", str(run_dir),
        "--plot"
    ]
    success, error = run_command(cmd, f"Sensitivity analysis ({run_name})")
    results[f"{run_name}/sensitivity_analysis"] = success

    # tornado_plot.py - one-at-a-time parameter sensitivity tornado plots
    # Find config file (input_distributions.yaml from batch_rollout, or sampling_config.yaml from forecast_export)
    config_path = run_dir / "input_distributions.yaml"
    if not config_path.exists():
        config_path = run_dir / "sampling_config.yaml"
    if config_path.exists():
        sensitivity_dir = run_dir / "sensitivity_analysis"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)

        # Tornado plot for AC arrival time
        tornado_output = sensitivity_dir / "tornado_ac.png"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "analysis" / "tornado_plot.py"),
            "--config", str(config_path),
            "--milestone", "AC",
            "--top-k", "15",
            "--show-vals",
            "--output", str(tornado_output)
        ]
        success, error = run_command(cmd, f"Tornado plot - AC ({run_name})")
        results[f"{run_name}/tornado_plot_ac"] = success

        # Tornado plot for AC to ASI takeoff duration
        tornado_takeoff_output = sensitivity_dir / "tornado_ac_asi_takeoff.png"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "analysis" / "tornado_plot.py"),
            "--config", str(config_path),
            "--transition", "AC:ASI",
            "--top-k", "15",
            "--show-vals",
            "--output", str(tornado_takeoff_output)
        ]
        success, error = run_command(cmd, f"Tornado plot - AC to ASI ({run_name})")
        results[f"{run_name}/tornado_plot_ac_asi"] = success
    else:
        print(f"WARNING: No config file found for tornado plot in {run_dir}")
        results[f"{run_name}/tornado_plot_ac"] = False
        results[f"{run_name}/tornado_plot_ac_asi"] = False

    return results


def find_figures(run_dir: Path) -> List[Tuple[str, Path]]:
    """Find the specific figures needed for the results PDF.

    Based on outputs/figure_generation_pipeline.md Figure-to-Script Mapping.
    Only includes figures that appear in the actual PDF, not auxiliary outputs.
    """
    figures = []

    # Define the specific figures from the PDF (see figure_generation_pipeline.md)
    # Format: (relative_path_pattern, description)
    pdf_figures = [
        # Short timelines analysis (Pages 3-4)
        ("short_timelines_outputs/ac_sar_pdfs_overlay.png", "AC and SAR Arrival Time Distributions (Page 3)"),
        ("short_timelines_outputs/ac_to_sar_duration_pdf.png", "Transition Duration: AC → SAR (Page 4)"),

        # Fast takeoff analysis (Pages 6-7)
        ("fast_takeoff_outputs/prob_ai2027_takeoff_vs_sar_arrival.png", "P(AI 2027 Takeoff or Faster) vs SAR Arrival Year (Page 6)"),
        ("fast_takeoff_outputs/prob_1yr_takeoff_vs_sar_arrival.png", "P(≤ 1 Year Takeoff) vs SAR Arrival Year (Page 6)"),
        ("fast_takeoff_outputs/prob_ai2027_takeoff_vs_ac_arrival.png", "P(AI 2027 Takeoff or Faster) vs AC Arrival Year (Page 7)"),

        # Timeline-takeoff correlation (Pages 8-9)
        ("timeline_takeoff_correlation/parameter_sensitivity_scatter_sar.png", "Parameter Effects: SAR Timeline vs Takeoff Speed (Page 8)"),
        ("timeline_takeoff_correlation/parameter_sensitivity_scatter_ac.png", "Parameter Effects: AC Timeline vs Takeoff Speed (Page 8)"),

        # Horizon trajectories (Pages 10-11)
        ("horizon_trajectories/horizon_trajectories_all_shaded.png", "Complete Time Horizon Trajectories - All (Page 10)"),
        ("horizon_trajectories/horizon_trajectories_mse_1.0.png", "Time Horizon Trajectories (MSE ≤ 1.0) (Page 11)"),

        # m/β distribution (Page 18)
        ("m_over_beta_hist.png", "Distribution of m/β (Page 18)"),

        # Takeoff exponent and uplift ratio distributions (in fast_takeoff_outputs)
        ("fast_takeoff_outputs/takeoff_exponent_hist.png", "Distribution of Takeoff Exponent 2^(1-β/m)"),
        ("fast_takeoff_outputs/uplift_ratio_hist.png", "Ratio of Successive Uplift Doublings 2^(β/m-1)"),

        # Extended analyses - continuous plots (Pages 12-14)
        # Note: These are generated by run_extended_analyses.py, which is slow
        # Paths may vary based on parameter naming conventions
        ("continuous_plots/doubling-difficulty-growth-factor/aa-time_vs_doubling-difficulty-growth-factor.png",
         "P(AC by Date) vs Doubling Difficulty Growth Factor (Page 12)"),
        ("continuous_plots/present-doubling-time/aa-time_vs_present-doubling-time.png",
         "P(AC by Date) vs Present Doubling Time (Page 12)"),
        ("continuous_plots/ai-research-taste-slope/p-1yr-takeoff-AC_vs_ai-research-taste-slope.png",
         "P(AI 2027 Speed Takeoff) vs Automated Research Taste Slope (Page 13)"),
        ("continuous_plots/median-to-top-taste-multiplier/p-1yr-takeoff-AC_vs_median-to-top-taste-multiplier.png",
         "P(AI 2027 Speed Takeoff) vs Median To Top Taste Multiplier (Page 14)"),

        # Sensitivity analysis - tornado plots (one-at-a-time) and correlation bar charts
        ("sensitivity_analysis/tornado_ac.png",
         "Parameter Sensitivity: AC Arrival Time (One-at-a-Time Analysis)"),
        ("sensitivity_analysis/tornado_ac_asi_takeoff.png",
         "Parameter Sensitivity: AC→ASI Takeoff Duration (One-at-a-Time Analysis)"),
        ("sensitivity_analysis/sensitivity_spearman_top_aa_time.png",
         "Parameter Sensitivity: Spearman Correlations with AC Time"),
    ]

    for rel_path, description in pdf_figures:
        full_path = run_dir / rel_path
        if full_path.exists():
            figures.append((description, full_path))
        else:
            # Try finding by filename only (in case directory structure varies)
            filename = Path(rel_path).name
            alt_paths = list(run_dir.rglob(filename))
            if alt_paths:
                figures.append((description, alt_paths[0]))

    return figures


def generate_markdown_report(
    output_dir: Path,
    eli_dir: Optional[Path],
    daniel_dir: Optional[Path],
    no_corr_dir: Optional[Path]
) -> Path:
    """Generate the Markdown report with all figures."""

    report_path = output_dir / "results_figures.md"

    lines = [
        "# AI Futures Model Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    def add_section(title: str, run_dir: Optional[Path], run_name: str):
        if run_dir is None:
            return

        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Run directory: `{run_dir}`")
        lines.append("")

        figures = find_figures(run_dir)

        if not figures:
            lines.append("*No figures found*")
            lines.append("")
            return

        for description, fig_path in figures:
            # Use relative path from output_dir
            try:
                rel_path = fig_path.relative_to(output_dir)
            except ValueError:
                rel_path = fig_path

            lines.append(f"### {description}")
            lines.append("")
            lines.append(f"![{description}]({rel_path})")
            lines.append("")

        lines.append("---")
        lines.append("")

    add_section("Results with Eli's Parameters", eli_dir, "eli")
    add_section("Results with No Correlations", no_corr_dir, "eli_no_corr")
    add_section("Results with Daniel's Parameters", daniel_dir, "daniel")

    # Write the report
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report saved to: {report_path}")

    return report_path


def main():
    args = parse_args()

    # Track results
    all_results: Dict[str, bool] = {}
    eli_run_dir: Optional[Path] = None
    daniel_run_dir: Optional[Path] = None
    no_corr_run_dir: Optional[Path] = None

    # Handle --regenerate-plots option (auto-detect subdirectories)
    if args.regenerate_plots:
        output_dir = Path(args.regenerate_plots)
        if not output_dir.exists():
            print(f"ERROR: Directory does not exist: {output_dir}")
            sys.exit(1)

        print(f"\nRegenerating plots for: {output_dir}")
        print("Auto-detecting run directories...")

        eli_run_dir = find_run_dir_in_subdir(output_dir, "eli")
        daniel_run_dir = find_run_dir_in_subdir(output_dir, "daniel")
        no_corr_run_dir = find_run_dir_in_subdir(output_dir, "eli_no_corr")

        if eli_run_dir:
            print(f"  Found eli: {eli_run_dir}")
        if daniel_run_dir:
            print(f"  Found daniel: {daniel_run_dir}")
        if no_corr_run_dir:
            print(f"  Found eli_no_corr: {no_corr_run_dir}")

        if not any([eli_run_dir, daniel_run_dir, no_corr_run_dir]):
            print("ERROR: No run directories found in the specified directory.")
            print("Expected subdirectories: eli/, daniel/, or eli_no_corr/")
            sys.exit(1)

        # Ensure caches exist
        for run_dir, name in [(eli_run_dir, "eli"), (daniel_run_dir, "daniel"), (no_corr_run_dir, "eli_no_corr")]:
            if run_dir and run_dir.exists():
                cache_file = run_dir / "rollouts.cache.json"
                if not cache_file.exists():
                    print(f"Generating cache for {name}...")
                    generate_rollouts_cache(run_dir)

        # Skip to figure generation (no rollouts)
        print("\n" + "="*60)
        print("Regenerating Analysis Figures")
        print("="*60)

        if eli_run_dir and eli_run_dir.exists():
            results = generate_figures(eli_run_dir, "eli", args.skip_extended, args.website_only)
            all_results.update(results)

        if daniel_run_dir and daniel_run_dir.exists():
            results = generate_figures(daniel_run_dir, "daniel", args.skip_extended, args.website_only)
            all_results.update(results)

        if no_corr_run_dir and no_corr_run_dir.exists():
            results = generate_figures(no_corr_run_dir, "eli_no_corr", args.skip_extended, args.website_only)
            all_results.update(results)

        # Generate Markdown report
        print("\n" + "="*60)
        print("Generating Markdown Report")
        print("="*60)

        report_path = generate_markdown_report(
            output_dir, eli_run_dir, daniel_run_dir, no_corr_run_dir
        )

        # Print summary and exit
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        success_count = sum(1 for v in all_results.values() if v)
        fail_count = sum(1 for v in all_results.values() if not v)

        print(f"\nTotal scripts run: {len(all_results)}")
        print(f"Succeeded: {success_count}")
        print(f"Failed: {fail_count}")

        if fail_count > 0:
            print("\nFailed scripts:")
            for name, success in all_results.items():
                if not success:
                    print(f"  - {name}")

        print(f"\nOutput directory: {output_dir}")
        print(f"Markdown report: {report_path}")
        print("\nDone!")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUTS_DIR / f"results_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Determine which configs to run
    run_eli = not args.daniel_only
    run_daniel = not args.eli_only
    run_no_corr = not args.daniel_only  # No-corr is a variant of Eli's

    if args.skip_rollout:
        # Use provided directories (resolve to absolute paths)
        if args.eli_dir:
            eli_run_dir = Path(args.eli_dir).resolve()
        if args.daniel_dir:
            daniel_run_dir = Path(args.daniel_dir).resolve()
        if args.no_corr_dir:
            no_corr_run_dir = Path(args.no_corr_dir).resolve()

        # Ensure cache exists for provided directories (if not already present)
        for run_dir, name in [(eli_run_dir, "eli"), (daniel_run_dir, "daniel"), (no_corr_run_dir, "eli_no_corr")]:
            if run_dir and run_dir.exists():
                cache_file = run_dir / "rollouts.cache.json"
                if not cache_file.exists():
                    print(f"Generating cache for {name}...")
                    generate_rollouts_cache(run_dir)
    else:
        # Run batch rollouts
        print("\n" + "="*60)
        print("PHASE 1: Running Monte Carlo Simulations")
        print("="*60)

        # Use SeedSequence to spawn independent random streams for each config.
        # This guarantees statistical independence regardless of sample counts.
        master_ss = SeedSequence(args.seed)
        eli_ss, daniel_ss, no_corr_ss = master_ss.spawn(3)
        # Pass entropy as seed; batch_rollout.py will create its own SeedSequence
        eli_seed = eli_ss.entropy
        daniel_seed = daniel_ss.entropy
        no_corr_seed = no_corr_ss.entropy

        if run_eli:
            eli_run_dir = run_batch_rollout(
                ELI_CONFIG, "eli", output_dir, args.num_samples, eli_seed, args.num_workers
            )
            if eli_run_dir:
                print(f"Eli run directory: {eli_run_dir}")

        if run_daniel:
            daniel_run_dir = run_batch_rollout(
                DANIEL_CONFIG, "daniel", output_dir, args.num_samples, daniel_seed, args.num_workers
            )
            if daniel_run_dir:
                print(f"Daniel run directory: {daniel_run_dir}")

        if run_no_corr:
            no_corr_run_dir = run_batch_rollout(
                NO_CORR_CONFIG, "eli_no_corr", output_dir, args.num_samples, no_corr_seed, args.num_workers
            )
            if no_corr_run_dir:
                print(f"No-correlation run directory: {no_corr_run_dir}")

    # Generate figures
    print("\n" + "="*60)
    print("PHASE 2: Generating Analysis Figures")
    print("="*60)

    if eli_run_dir and eli_run_dir.exists():
        results = generate_figures(eli_run_dir, "eli", args.skip_extended, args.website_only)
        all_results.update(results)

    if daniel_run_dir and daniel_run_dir.exists():
        results = generate_figures(daniel_run_dir, "daniel", args.skip_extended, args.website_only)
        all_results.update(results)

    if no_corr_run_dir and no_corr_run_dir.exists():
        results = generate_figures(no_corr_run_dir, "eli_no_corr", args.skip_extended, args.website_only)
        all_results.update(results)

    # Generate Markdown report
    print("\n" + "="*60)
    print("PHASE 3: Generating Markdown Report")
    print("="*60)

    report_path = generate_markdown_report(
        output_dir, eli_run_dir, daniel_run_dir, no_corr_run_dir
    )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    success_count = sum(1 for v in all_results.values() if v)
    fail_count = sum(1 for v in all_results.values() if not v)

    print(f"\nTotal scripts run: {len(all_results)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {fail_count}")

    if fail_count > 0:
        print("\nFailed scripts:")
        for name, success in all_results.items():
            if not success:
                print(f"  - {name}")

    print(f"\nOutput directory: {output_dir}")
    print(f"Markdown report: {report_path}")

    if eli_run_dir:
        print(f"Eli figures: {eli_run_dir}")
    if daniel_run_dir:
        print(f"Daniel figures: {daniel_run_dir}")
    if no_corr_run_dir:
        print(f"No-correlation figures: {no_corr_run_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
