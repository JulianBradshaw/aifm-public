#!/usr/bin/env python3
"""
Plot utilities for analyzing batch rollout results.

Currently supported:
- Plot distribution (histogram) of AC times from a rollouts.jsonl file
- Plot distribution (histogram) of horizon length at AC across rollouts
- Plot distribution (histogram) of arrival time for arbitrary milestones
- Plot distribution (histogram) of effective compute (OOMs) required to achieve milestones

Usage examples:
  python scripts/plot_rollouts.py --run-dir outputs/20250813_020347 \
    --out outputs/20250813_020347/aa_time_hist.png

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl \
    --mode milestone_compute_hist --milestone AC
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr, norm
from datetime import datetime, timedelta
import yaml

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

# Import utilities
from plotting_utils.kde import make_gaussian_kde
from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.helpers import (
    decimal_year_to_date_string,
    format_time_duration,
    format_years_value,
    get_year_tick_values_and_labels,
    now_decimal_year,
    get_time_tick_values_and_labels,
    load_metr_p80_points,
    parse_milestone_pairs,
    simplify_milestone_name,
)
from plotting.histograms import (
    plot_aa_time_histogram,
    plot_aa_time_histogram_cdf,
    plot_milestone_effective_compute_histogram,
    plot_milestone_time_histogram,
    plot_milestone_time_histogram_cdf,
    plot_m_over_beta_histogram,
    plot_x_years_in_1_year_histogram,
)
from plotting.trajectories import (
    plot_horizon_trajectories,
    plot_uplift_trajectories,
)
from analysis.sensitivity_analysis import plot_all_correlations
from plotting.boxplots import plot_milestone_transition_boxplot
from plots.pdfs.milestone_overlay import MilestonePDFOverlayPlot
from plots.histograms.m_over_beta import MOverBetaHistogramPlot
from plots.histograms.takeoff_exponent import TakeoffExponentHistogramPlot
from plots.histograms.uplift_ratio import UpliftRatioHistogramPlot
from plots.histograms.horizon_at_sc import HorizonAtSCHistogramPlot
from plots.histograms.milestone_time import MilestoneTimeHistogramPlot
from plots.histograms.milestone_compute import MilestoneComputeHistogramPlot
from plots.scatter.milestone_scatter import MilestoneScatterPlot
from plots.scatter.transition_boxplot import TransitionBoxplotPlot
from plots.trajectories.metric import MetricTrajectoryPlot


def _resolve_rollouts_path(run_dir: Optional[str], rollouts_path: Optional[str]) -> Path:
    if rollouts_path:
        p = Path(rollouts_path)
        if not p.exists():
            raise FileNotFoundError(f"rollouts file does not exist: {p}")
        return p
    if not run_dir:
        raise ValueError("Either --run-dir or --rollouts must be provided")
    d = Path(run_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"run directory does not exist or is not a directory: {d}")
    p = d / "rollouts.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"rollouts.jsonl not found in run directory: {d}")
    return p


def generate_horizon_plots(
    rollouts_file: Path,
    output_dir: Path,
    present_day: float,
    selection_start: Optional[float] = None,
    website_only: bool = False,
) -> None:
    """Generate all horizon trajectory plots for a rollout run.

    This function is used by both batch_plot_all() and --batch-horizon to avoid
    code duplication.

    Args:
        rollouts_file: Path to rollouts.jsonl
        output_dir: Base output directory (horizon_trajectories subfolder will be created)
        present_day: Present day in decimal years
        selection_start: Optional start year for central trajectory selection
        website_only: If True, only generate the 2 plots needed for the website
                     (horizon_trajectories_all_shaded.png and horizon_trajectories_mse_1.0.png)
    """
    horizon_dir = output_dir / "horizon_trajectories"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    reader = RolloutsReader(rollouts_file)
    # When website_only, only generate mse_1.0 (the only MSE plot used by the website)
    mse_thresholds = [1.0] if website_only else [10.0, 1.5, 1.0, 0.7, 0.5]

    # Read horizon data with SC times for central trajectory selection
    times, trajectories, aa_times, mse_values, sc_times = reader.read_metric_trajectories_with_milestone(
        "horizon_lengths", "AI2027-SC"
    )
    # Also read AC times for central trajectory selection
    _, _, _, _, ac_times = reader.read_metric_trajectories_with_milestone(
        "horizon_lengths", "AC"
    )
    print(f"Loaded {len(trajectories)} trajectories")

    # Generate main plot with all trajectories and MSE shading
    out_path = horizon_dir / "horizon_trajectories_all_shaded.png"
    try:
        plot_horizon_trajectories(
            times,
            trajectories,
            out_path,
            max_trajectories=1000,
            stop_at_sc=False,
            aa_times=aa_times,
            mse_values=mse_values,
            shade_by_mse=True,
            mse_threshold=None,
            cutoff_year=2040.0,
            sc_times=sc_times,
            present_day=present_day,
            ac_times=ac_times,
            selection_start=selection_start,
            show_both_central_methods=True,
        )
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Warning: Could not generate all trajectories with shading: {e}")

    # Generate MSE-filtered plots
    for mse_threshold in mse_thresholds:
        out_path = horizon_dir / f"horizon_trajectories_mse_{mse_threshold:.1f}.png"
        try:
            plot_horizon_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
                mse_values=mse_values,
                shade_by_mse=True,
                mse_threshold=mse_threshold,
                cutoff_year=2040.0,
                sc_times=sc_times,
                present_day=present_day,
                ac_times=ac_times,
                selection_start=selection_start,
                show_both_central_methods=True,
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate MSE <= {mse_threshold}: {e}")

    # Skip remaining plots in website_only mode (only needed all_shaded and mse_1.0)
    if website_only:
        print("Website-only mode: skipping additional horizon plots")
        return

    # Generate AC-median filtered plot
    try:
        reader_ac = RolloutsReader(rollouts_file)
        times_ac, trajectories_ac, aa_times_ac, mse_values_ac, ac_times_for_filter = reader_ac.read_metric_trajectories_with_milestone(
            "horizon_lengths", "AC"
        )
        _, _, _, _, sc_times_ac = reader_ac.read_metric_trajectories_with_milestone(
            "horizon_lengths", "AI2027-SC"
        )

        finite_ac_times = [t for t in ac_times_for_filter if t is not None]
        if len(finite_ac_times) > 0:
            median_ac = np.median(finite_ac_times)
            print(f"Median AC time: {median_ac:.2f}")

            tolerance = 0.0415
            filtered_indices = [
                i for i, t in enumerate(ac_times_for_filter)
                if t is not None and abs(t - median_ac) <= tolerance
            ]

            if len(filtered_indices) > 0:
                trajectories_filtered = [trajectories_ac[i] for i in filtered_indices]
                aa_times_filtered = [aa_times_ac[i] for i in filtered_indices] if aa_times_ac else None
                mse_filtered = [mse_values_ac[i] for i in filtered_indices] if mse_values_ac else None
                sc_times_filtered = [sc_times_ac[i] for i in filtered_indices] if sc_times_ac else None
                ac_times_filtered = [ac_times_for_filter[i] for i in filtered_indices]

                out_path = horizon_dir / "horizon_trajectories_ac_median.png"
                plot_horizon_trajectories(
                    times_ac,
                    trajectories_filtered,
                    out_path,
                    max_trajectories=1000,
                    stop_at_sc=False,
                    aa_times=aa_times_filtered,
                    mse_values=mse_filtered,
                    shade_by_mse=True,
                    mse_threshold=None,
                    cutoff_year=2040.0,
                    title=f"Trajectories with AC at Median Time ({median_ac:.2f} ± {tolerance}y)",
                    sc_times=sc_times_filtered,
                    present_day=present_day,
                    ac_times=ac_times_filtered,
                    selection_start=selection_start,
                    show_both_central_methods=True,
                )
                print(f"Saved AC-median filtered plot: {out_path} (n={len(filtered_indices)})")
            else:
                print("Warning: No trajectories within tolerance of median AC time")
        else:
            print("Warning: No finite AC times found")
    except Exception as e:
        print(f"Warning: Could not generate AC-median filtered: {e}")

    # Generate AI2027-SC + ASI Dec 2027 trajectory plot
    try:
        import math
        ai2027_sc_target = 2027.208  # mid-March
        asi_target = 2027.958  # mid-December

        best_idx = None
        best_distance = float('inf')
        best_sc = None
        best_asi = None
        best_present_horizon = 26.0

        with open(rollouts_file, 'r') as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                results = r.get('results', {})
                milestones = results.get('milestones', {})
                sc_data = milestones.get('AI2027-SC', {})
                asi_data = milestones.get('ASI', {})
                sc_time = sc_data.get('time')
                asi_time = asi_data.get('time')
                if sc_time is None or asi_time is None:
                    continue
                distance = math.sqrt((sc_time - ai2027_sc_target)**2 + (asi_time - asi_target)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = i
                    best_sc = sc_time
                    best_asi = asi_time
                    best_present_horizon = r.get('parameters', {}).get('present_horizon', 26.0)

        if best_idx is not None:
            print(f"\nAI2027-SC + ASI trajectory: index {best_idx}, SC={best_sc:.4f}, ASI={best_asi:.4f}, dist={best_distance:.4f}")

            with open(rollouts_file, 'r') as f:
                for i, line in enumerate(f):
                    if i == best_idx:
                        r = json.loads(line)
                        break

            results = r['results']
            traj_times = np.array(results['times'])
            traj_horizons = np.array(results['horizon_lengths'])

            csv_path = horizon_dir / "horizon_trajectories_ai2027_sc_asi_dec2027.csv"
            with open(csv_path, 'w') as f:
                f.write("year,horizon_minutes\n")
                for t, h in zip(traj_times, traj_horizons):
                    if np.isfinite(h) and h > 0:
                        f.write(f"{t},{h}\n")
            print(f"Saved {csv_path}")

            tick_values, tick_labels = get_time_tick_values_and_labels()
            metr_points = load_metr_p80_points()

            plt.figure(figsize=(14, 8))
            valid = np.isfinite(traj_horizons) & (traj_horizons > 0)
            plt.plot(traj_times[valid], traj_horizons[valid], color="tab:green", linewidth=2.0,
                    label="AI2027 SC Mar 2027, ASI Dec 2027")

            plt.axhline(best_present_horizon, color="red", linewidth=2.0, label=f"Current Horizon ({int(best_present_horizon)} min)")
            plt.axvline(present_day, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

            if metr_points:
                xs = [p[0] for p in metr_points]
                ys = [p[1] for p in metr_points]
                plt.scatter(xs, ys, color="black", s=18, label="External Benchmarks (p80)", zorder=10)

            plt.axvline(best_sc, color="orange", linestyle=":", linewidth=1.5, label=f"AI2027-SC ({best_sc:.2f})")
            plt.axvline(best_asi, color="purple", linestyle=":", linewidth=1.5, label=f"ASI ({best_asi:.2f})")

            plt.yscale("log")
            plt.yticks(tick_values, tick_labels)
            plt.ylim(min(tick_values), max(traj_horizons[valid]) * 1.5)
            plt.xlim(2017, 2032)
            plt.xlabel("Year")
            plt.ylabel("Time Horizon")
            plt.title("AI2027 SC March 2027, ASI December 2027 Trajectory\n(Closest match from Monte Carlo)")
            plt.grid(True, which="both", axis="y", alpha=0.25)
            plt.legend(loc="upper left")
            plt.tight_layout()

            out_path = horizon_dir / "horizon_trajectories_ai2027_sc_asi_dec2027.png"
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {out_path}")
        else:
            print("Warning: No trajectory found with both AI2027-SC and ASI milestones")
    except Exception as e:
        print(f"Warning: Could not generate AI2027-SC + ASI trajectory: {e}")
        import traceback
        traceback.print_exc()

    # Generate AI2027-SC March 2027 filtered plot
    try:
        ai2027_sc_march_target = 2027.208
        sc_tolerance = 0.0415 * 2

        filtered_indices = [
            i for i, t in enumerate(sc_times)
            if t is not None and abs(t - ai2027_sc_march_target) <= sc_tolerance
        ]

        if len(filtered_indices) > 0:
            trajectories_filtered = [trajectories[i] for i in filtered_indices]
            aa_times_filtered = [aa_times[i] for i in filtered_indices] if aa_times else None
            mse_filtered = [mse_values[i] for i in filtered_indices] if mse_values else None
            sc_times_filtered = [sc_times[i] for i in filtered_indices]
            ac_times_filtered = [ac_times[i] for i in filtered_indices] if ac_times else None

            out_path = horizon_dir / "horizon_trajectories_sc_march2027.png"
            plot_horizon_trajectories(
                times,
                trajectories_filtered,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times_filtered,
                mse_values=mse_filtered,
                shade_by_mse=True,
                mse_threshold=None,
                cutoff_year=2040.0,
                title=f"Trajectories with AI2027-SC in March 2027 ({ai2027_sc_march_target:.2f} ± {sc_tolerance:.3f}y)",
                sc_times=sc_times_filtered,
                present_day=present_day,
                ac_times=ac_times_filtered,
                selection_start=selection_start,
                show_both_central_methods=True,
            )
            print(f"Saved AI2027-SC March 2027 filtered plot: {out_path} (n={len(filtered_indices)})")

            # Save CSV for central trajectory
            from plotting.trajectories import select_central_trajectory
            central_idx_in_filtered, _, _ = select_central_trajectory(
                times, trajectories_filtered, sc_times_filtered, present_day, ac_times=ac_times_filtered
            )
            original_idx = filtered_indices[central_idx_in_filtered]
            print(f"SC March 2027 central trajectory: filtered index {central_idx_in_filtered}, original index {original_idx}")

            with open(rollouts_file, 'r') as f:
                for i, line in enumerate(f):
                    if i == original_idx:
                        rollout = json.loads(line)
                        results = rollout['results']
                        traj_times_central = np.array(results['times'])
                        traj_horizons_central = np.array(results['horizon_lengths'])

                        csv_path = horizon_dir / "horizon_trajectories_sc_march2027.csv"
                        with open(csv_path, 'w') as csv_f:
                            csv_f.write("year,horizon_minutes\n")
                            for t, h in zip(traj_times_central, traj_horizons_central):
                                if np.isfinite(h) and h > 0:
                                    csv_f.write(f"{t},{h}\n")
                        print(f"Saved {csv_path}")
                        break
        else:
            print("Warning: No trajectories within tolerance of March 2027 AI2027-SC time")
    except Exception as e:
        print(f"Warning: Could not generate AI2027-SC March 2027 filtered trajectories: {e}")
        import traceback
        traceback.print_exc()


def batch_plot_all(rollouts_file: Path, output_dir: Path, histograms_only: bool = False) -> None:
    """Generate all standard plots for a batch rollout.

    Args:
        rollouts_file: Path to rollouts.jsonl file
        output_dir: Directory to write output plots
        histograms_only: If True, only generate fast cache-based plots (histograms, boxplots, PDFs)
                        and skip slow trajectory plots that require parsing the full JSONL

    Creates:
    - Milestone time histograms for key milestones
    - Milestone transition boxplot for key transitions
    - Overlaid PDF plot showing arrival distributions for key milestones
    - (unless histograms_only) Horizon trajectory plots
    - (unless histograms_only) Uplift trajectory plots
    - (unless histograms_only) Sensitivity analysis plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load present_day from run directory for gamma kernel KDE
    from plotting_utils.helpers import load_present_day, load_central_trajectory_selection_start
    present_day = load_present_day(output_dir)
    print(f"Present day: {present_day}")
    selection_start = load_central_trajectory_selection_start(output_dir)
    if selection_start is not None:
        print(f"Central trajectory selection start: {selection_start}")

    # Create milestone_time_hists subfolder
    milestone_hist_dir = output_dir / "milestone_time_hists"
    milestone_hist_dir.mkdir(parents=True, exist_ok=True)

    # Create a single RolloutsReader instance to reuse (avoids reloading cache)
    print(f"Loading rollouts data from {rollouts_file}...")
    reader = RolloutsReader(rollouts_file)
    print("Rollouts data loaded.")

    # Milestone time histograms (both PDF and CDF style)
    milestones = [
        "AC",
        "AIR-5x",
        "AI2027-SC",
        "SAR-level-experiment-selection-skill",
        "AIR-25x",
        "SIAR-level-experiment-selection-skill",
        "AIR-250x"
    ]

    for milestone in milestones:
        # Create safe filename from milestone name
        safe_name = milestone.replace("(Expensive, threshold only considers taste) ", "").replace("-", "_").replace("x", "x")

        # Reuse the reader instance
        times, num_not_achieved, sim_end = reader.read_milestone_times(milestone)

        if not times and num_not_achieved == 0:
            print(f"Warning: No data found for milestone '{milestone}', skipping histograms")
            continue

        # PDF-style histogram - 2050 cutoff version
        out_path_pdf = milestone_hist_dir / f"milestone_time_hist_{safe_name}.png"
        plot_milestone_time_histogram(
            times,
            num_not_achieved,
            out_path_pdf,
            bins=50,
            title=f"Distribution of {milestone} Times",
            sim_end=sim_end,
            max_year=2050,
            present_day=present_day
        )
        print(f"Saved {out_path_pdf}")

        # PDF-style histogram - full range version
        out_path_pdf_full = milestone_hist_dir / f"milestone_time_hist_{safe_name}_full.png"
        plot_milestone_time_histogram(
            times,
            num_not_achieved,
            out_path_pdf_full,
            bins=50,
            title=f"Distribution of {milestone} Times (Full Range)",
            sim_end=sim_end,
            max_year=None,
            present_day=present_day
        )
        print(f"Saved {out_path_pdf_full}")

        # CDF-style histogram - 2050 cutoff version (100 bins for finer granularity)
        out_path_cdf = milestone_hist_dir / f"milestone_time_hist_{safe_name}_cdf.png"
        plot_milestone_time_histogram_cdf(
            times,
            num_not_achieved,
            out_path_cdf,
            bins=100,
            title=f"Cumulative Distribution of {milestone} Times",
            sim_end=sim_end,
            max_year=2050
        )
        print(f"Saved {out_path_cdf}")

        # CDF-style histogram - full range version (100 bins for finer granularity)
        out_path_cdf_full = milestone_hist_dir / f"milestone_time_hist_{safe_name}_cdf_full.png"
        plot_milestone_time_histogram_cdf(
            times,
            num_not_achieved,
            out_path_cdf_full,
            bins=100,
            title=f"Cumulative Distribution of {milestone} Times (Full Range)",
            sim_end=sim_end,
            max_year=None
        )
        print(f"Saved {out_path_cdf_full}")

        # Effective compute histogram - save in milestone_time_hists subfolder
        compute_ooms, num_compute_not_achieved = reader.read_milestone_compute(milestone)
        if compute_ooms or num_compute_not_achieved > 0:
            out_path_compute = milestone_hist_dir / f"milestone_compute_hist_{safe_name}.png"
            plot_milestone_effective_compute_histogram(
                compute_ooms,
                num_compute_not_achieved,
                out_path_compute,
                bins=50,
                title=f"Distribution of Effective Compute at {milestone}",
                milestone_label=milestone
            )
            print(f"Saved {out_path_compute}")

    # Milestone transition boxplot
    pairs_str = "AC:SAR-level-experiment-selection-skill,SAR-level-experiment-selection-skill:SIAR-level-experiment-selection-skill,SIAR-level-experiment-selection-skill:STRAT-AI,STRAT-AI:TED-AI,TED-AI:ASI"
    pairs = parse_milestone_pairs(pairs_str)
    out_path = output_dir / "milestone_transition_box.png"

    # Reuse the reader instance
    labels, durations, durations_censored, num_b_not_achieved, num_b_before_a, total_per_pair, typical_max, simulation_cutoff = reader.read_multiple_transition_durations(
        pairs,
        filter_milestone=None,
        filter_by_year=None
    )

    if labels:
        plot_milestone_transition_boxplot(
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            out_path,
            ymin_years=None,
            ymax_years=None,
            exclude_inf_from_stats=False,
            inf_years_cap=typical_max,
            inf_years_display=100.0,
            title="Milestone Transition Durations",
            simulation_cutoff=simulation_cutoff
        )
        print(f"Saved {out_path}")
    else:
        print("Warning: No transition data found, skipping boxplot")

    # Overlaid PDF plot for key milestones using new class-based API
    overlay_plot = MilestonePDFOverlayPlot(output_dir, reader=reader)
    out_path = overlay_plot.generate(preset="ALL", out_path=output_dir / "milestone_pdfs_overlay.png")
    print(f"Saved {out_path}")

    # Post-SAR milestone PDFs (SIAR, TED-AI, ASI) using new class-based API
    print("\nGenerating post-SAR milestone PDFs (SIAR, TED-AI, ASI)...")
    try:
        post_sar_plot = MilestonePDFOverlayPlot(output_dir, reader=reader)
        out_path = post_sar_plot.generate(preset="POST_SAR", out_path=output_dir / "post_sar_milestone_pdfs.png")
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Warning: Could not generate post-SAR milestone PDFs: {e}")

    # Early exit if histograms_only mode - skip slow trajectory plots
    if histograms_only:
        print("\n[histograms_only mode] Skipping trajectory plots (horizon, uplift, sensitivity)")
        return

    # Horizon trajectory plots with MSE filtering
    print("\nGenerating horizon trajectory plots with MSE filtering...")
    try:
        generate_horizon_plots(rollouts_file, output_dir, present_day, selection_start)
    except Exception as e:
        print(f"Warning: Could not generate horizon trajectory plots: {e}")

    # Uplift trajectory plots
    print("\nGenerating AI software R&D uplift trajectory plots...")

    # Create subfolder for uplift trajectories
    uplift_dir = output_dir / "uplift_trajectories"
    uplift_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read uplift data (all trajectories)
        reader = RolloutsReader(rollouts_file)
        times, trajectories, aa_times, _ = reader.read_metric_trajectories(
            "ai_sw_progress_mult_ref_present_day",
            include_mse=False
        )

        # Generate basic uplift trajectory plot
        out_path = uplift_dir / "uplift_trajectories.png"
        try:
            plot_uplift_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate uplift trajectories: {e}")

        # Generate variant with 2050 cutoff
        out_path = uplift_dir / "uplift_trajectories_2050.png"
        try:
            plot_uplift_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
                max_year=2050,
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate 2050 uplift trajectories: {e}")
    except Exception as e:
        print(f"Warning: Could not load uplift trajectories: {e}")

    # Uplift trajectories filtered for SAR in 2027
    try:
        # Read uplift data filtered for SAR-level-experiment-selection-skill in 2027
        reader = RolloutsReader(rollouts_file)
        times, trajectories, aa_times, _ = reader.read_metric_trajectories(
            "ai_sw_progress_mult_ref_present_day",
            include_mse=False,
            filter_milestone="SAR-level-experiment-selection-skill",
            filter_year=2027.0
        )

        # Generate variant with 2050 cutoff and SAR filter
        out_path = uplift_dir / "uplift_trajectories_2050_sar2027.png"
        try:
            plot_uplift_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
                max_year=2050,
                title="AI Software R&D Uplift Trajectories\n(Filtered for SAR-level in 2027)"
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate SAR 2027 uplift trajectories: {e}")
    except Exception as e:
        print(f"Warning: Could not load SAR 2027 uplift trajectories: {e}")

    # Parameter sensitivity analysis for milestone transitions
    try:
        import sys
        sys.path.insert(0, str(rollouts_file.parent.parent / "scripts"))
        from sensitivity_analysis import analyze_milestone_transitions

        # Sensitivity analysis: AC to SAR-level
        print("Running parameter sensitivity analysis for AC to SAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for AC to SAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SAR-level-experiment-selection-skill"),
            include_censored=True
        )

        # Sensitivity analysis: SAR-level to SIAR-level
        print("Running parameter sensitivity analysis for SAR-level-experiment-selection-skill to SIAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for SAR-level-experiment-selection-skill to SIAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            include_censored=True
        )

        # Sensitivity analysis: AC to SIAR-level
        print("Running parameter sensitivity analysis for AC to SIAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SIAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for AC to SIAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SIAR-level-experiment-selection-skill"),
            include_censored=True
        )

    except Exception as e:
        print(f"Warning: Could not run parameter sensitivity analysis: {e}")

    # X years in 1 year distribution
    out_path = output_dir / "x_years_in_1_year_hist.png"
    x_values = reader.read_x_years_in_1_year()

    if len(x_values) > 0:
        plot_x_years_in_1_year_histogram(
            x_values,
            out_path,
            bins=50
        )
        # Print statistics
        arr = np.asarray(x_values, dtype=float)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"Saved {out_path}")
        print(f"  X years in 1 year - P10/Median/P90: {q10:.1f}x / {q50:.1f}x / {q90:.1f}x")
    else:
        print(f"Warning: No valid X values found, skipping X years in 1 year plot")

    # m/beta distribution
    out_path = output_dir / "m_over_beta_hist.png"
    reader = RolloutsReader(rollouts_file)
    m_over_beta_values, num_skipped = reader.read_m_over_beta()

    if len(m_over_beta_values) > 0:
        plot_m_over_beta_histogram(
            m_over_beta_values,
            num_skipped,
            out_path,
            bins=50,
            title="Distribution of m/β"
        )
        # Print statistics
        arr = np.asarray(m_over_beta_values, dtype=float)
        pct_above_one = 100.0 * np.sum(arr > 1) / len(arr)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"Saved {out_path}")
        print(f"  m/β - Percentage > 1: {pct_above_one:.1f}% | P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f}")
        if num_skipped > 0:
            print(f"  Note: {num_skipped} rollouts skipped due to missing r_software")
    else:
        print(f"Warning: No valid m/β values found, skipping m/β plot")
        if num_skipped > 0:
            print(f"  (All {num_skipped} rollouts were missing r_software - run a new Monte Carlo with updated code)")

    # Generate short timelines analysis (AC and SAR)
    print("\nGenerating short timelines analysis (AC and SAR)...")
    try:
        import sys
        import subprocess
        scripts_dir = rollouts_file.parent.parent / "scripts"
        python_exe = sys.executable
        subprocess.run(
            [python_exe, str(scripts_dir / "analysis" / "short_timelines_analysis.py"), "--rollouts", str(rollouts_file)],
            check=True
        )
        print("Short timelines analysis complete!")
    except Exception as e:
        print(f"Warning: Could not generate short timelines analysis: {e}")

    # Generate slack summary table for all MSE thresholds
    print("\nGenerating slack summary table for all MSE thresholds...")
    try:
        import sys
        sys.path.insert(0, str(rollouts_file.parent.parent / "scripts"))
        from slack_summary import compute_slack_summary, format_year
        import csv

        mse_thresholds_for_summary = [10.0, 1.5, 1.0, 0.7, 0.5]
        milestone = "AC"

        # Collect results
        summary_results = []
        for mse_threshold in mse_thresholds_for_summary:
            try:
                result = compute_slack_summary(rollouts_file, mse_threshold, milestone)
                summary_results.append(result)
            except Exception as e:
                print(f"Warning: Could not compute summary for MSE <= {mse_threshold}: {e}")

        if summary_results:
            # Write to CSV
            csv_path = output_dir / "slack_summary.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    "MSE Threshold",
                    "N Trajectories",
                    f"{milestone} Median",
                    f"{milestone} P10",
                    f"{milestone} P90",
                    "10y Horizon Median",
                    "10y Horizon P10",
                    "10y Horizon P90",
                    "100y Horizon Median",
                    "100y Horizon P10",
                    "100y Horizon P90"
                ])
                # Rows
                for result in summary_results:
                    writer.writerow([
                        f"<= {result['mse_threshold']}",
                        result['num_trajectories'],
                        format_year(result['milestone_median']) if result['milestone_median'] is not None else "N/A",
                        format_year(result['milestone_p10']) if result['milestone_p10'] is not None else "N/A",
                        format_year(result['milestone_p90']) if result['milestone_p90'] is not None else "N/A",
                        format_year(result['horizon_10y_median']) if result['horizon_10y_median'] is not None else "N/A",
                        format_year(result['horizon_10y_p10']) if result['horizon_10y_p10'] is not None else "N/A",
                        format_year(result['horizon_10y_p90']) if result['horizon_10y_p90'] is not None else "N/A",
                        format_year(result['horizon_100y_median']) if result['horizon_100y_median'] is not None else "N/A",
                        format_year(result['horizon_100y_p10']) if result['horizon_100y_p10'] is not None else "N/A",
                        format_year(result['horizon_100y_p90']) if result['horizon_100y_p90'] is not None else "N/A",
                    ])
            print(f"Saved slack summary table to {csv_path}")

            # Also print to console
            print("\n=== Slack Summary Table ===")
            for result in summary_results:
                print(f"\nMSE <= {result['mse_threshold']} ({result['num_trajectories']} trajectories):")
                if result['milestone_median'] is not None:
                    print(f"  {milestone}: {format_year(result['milestone_median'])} [{format_year(result['milestone_p10'])}, {format_year(result['milestone_p90'])}]")
                else:
                    print(f"  {milestone}: N/A")
                if result['horizon_10y_median'] is not None:
                    print(f"  10y horizons: {format_year(result['horizon_10y_median'])} [{format_year(result['horizon_10y_p10'])}, {format_year(result['horizon_10y_p90'])}]")
                else:
                    print(f"  10y horizons: N/A")
                if result['horizon_100y_median'] is not None:
                    print(f"  100y horizons: {format_year(result['horizon_100y_median'])} [{format_year(result['horizon_100y_p10'])}, {format_year(result['horizon_100y_p90'])}]")
                else:
                    print(f"  100y horizons: N/A")
    except Exception as e:
        print(f"Warning: Could not generate slack summary table: {e}")

    # Fast takeoff analysis
    try:
        print("\nGenerating fast takeoff analysis (SAR transitions)...")
        import subprocess
        import sys
        from pathlib import Path as PathLib
        python_exe = sys.executable
        # Get the repository root (where scripts/ is located)
        repo_root = PathLib(__file__).resolve().parent.parent
        scripts_dir = repo_root / "scripts"
        subprocess.run(
            [python_exe, str(scripts_dir / "analysis" / "fast_takeoff_analysis.py"), "--rollouts", str(rollouts_file)],
            cwd=str(repo_root),
            check=True
        )
        print("Fast takeoff analysis complete!")
    except Exception as e:
        print(f"Warning: Could not generate fast takeoff analysis: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tools for batch rollout results")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None, help="Path to a single rollout run directory containing rollouts.jsonl")
    g.add_argument("--rollouts", type=str, default=None, help="Path directly to a rollouts.jsonl file")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). Defaults vary by --mode")
    parser.add_argument("--mode", type=str, choices=["sc_hist", "horizon_trajectories", "horizon_at_sc_hist", "milestone_time_hist", "milestone_compute_hist", "milestone_transition_box", "milestone_scatter", "correlations", "m_over_beta", "takeoff_exponent", "uplift_ratio", "metric_trajectories"], default="sc_hist", help="Which plot to generate")
    parser.add_argument("--batch-all", action="store_true", help="Generate all standard plots (ignores --mode and --out)")
    parser.add_argument("--batch-horizon", action="store_true", help="Generate only horizon trajectory plots (faster than --batch-all)")
    parser.add_argument("--website-only", action="store_true", help="With --batch-horizon, only generate the 2 plots needed for the website (all_shaded and mse_1.0)")
    parser.add_argument("--histograms-only", action="store_true", help="With --batch-all, only generate fast cache-based plots (histograms, boxplots, PDFs), skip slow trajectory plots")
    parser.add_argument("--extract-central-params", action="store_true", help="Also generate SC March 2027 filtered plot and output central trajectory parameters for PRESET_PARAMETERS")
    parser.add_argument("--plot-correlations", action="store_true", help="Generate correlation scatter plots for all parameter pairs with non-zero correlation")
    # Histogram options
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins for sc_hist mode")
    parser.add_argument("--cdf-hist", action="store_true", help="Generate CDF-style histograms where bar heights show cumulative counts (rollouts reaching milestone BY that year)")
    # Milestone options
    parser.add_argument("--milestone", type=str, default=None, help="Milestone name for milestone_time_hist mode (e.g., 'AC')")
    parser.add_argument("--list-milestones", action="store_true", help="List milestone names found in the rollouts file and exit (works for milestone_time_hist and milestone_transition_box)")
    # Metric trajectories options
    parser.add_argument("--metric", type=str, default=None, help="Metric name for metric_trajectories mode (e.g., 'ai_coding_labor_mult_ref_present_day')")
    # Milestone transition boxplot options
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated milestone pairs for transition durations, formatted as FROM:TO (e.g., 'SC:SAR,SAR:SIAR,SIAR:ASI')")
    parser.add_argument("--ymin-years", type=float, default=None, help="Minimum y-axis (years, log scale) for transition boxplot")
    parser.add_argument("--ymax-years", type=float, default=None, help="Maximum y-axis (years, log scale) for transition boxplot")
    parser.add_argument("--exclude-inf-from-stats", action="store_true", help="Exclude 'not achieved' (treated as +inf) from the stats panel and the plot")
    parser.add_argument("--inf-years", type=float, default=100.0, help="Where to plot 'not achieved' as points on the y-axis (years)")
    parser.add_argument("--filter-milestone", type=str, default=None, help="Only include rollouts where this milestone was achieved by --filter-by-year (e.g., 'AIR-5x')")
    parser.add_argument("--filter-by-year", type=float, default=None, help="Decimal year cutoff for --filter-milestone (e.g., 2029.5)")
    # Milestone scatter/heatmap options
    parser.add_argument("--scatter-pair", type=str, default=None, help="Pair for scatter/heatmap FROM:TO (e.g., 'AIR-5x:AIR-2000x')")
    parser.add_argument("--scatter-kind", type=str, choices=["hex", "hist2d", "scatter"], default="hex", help="Density visualization type")
    parser.add_argument("--gridsize", type=int, default=50, help="Grid size for hex/hist2d density")
    parser.add_argument("--point-size", type=float, default=8.0, help="Point size for scatter overlay")
    parser.add_argument("--no-scatter-overlay", action="store_true", help="Disable scatter overlay when using hex/hist2d")
    # Horizon trajectories options
    parser.add_argument("--current-horizon-minutes", type=float, default=15.0, help="Horizontal reference line for current horizon in minutes")
    parser.add_argument("--alpha", type=float, default=0.08, help="Transparency for individual trajectories")
    parser.add_argument("--max-trajectories", type=int, default=2000, help="Maximum number of trajectories to draw")
    parser.add_argument("--no-metr", action="store_true", help="Disable overlay of METR p80 benchmark points")
    parser.add_argument("--stop-at-sc", action="store_true", help="Mask each trajectory after its own aa_time")
    parser.add_argument("--shade-by-mse", action="store_true", help="Color trajectories by their METR MSE (low MSE=green, high MSE=red)")
    parser.add_argument("--mse-threshold", type=float, default=None, help="Exclude trajectories with METR MSE above this threshold")
    parser.add_argument("--cutoff-year", type=float, default=None, help="Cut off all trajectories at this year")
    parser.add_argument("--yscale", type=str, choices=["auto", "log", "linear"], default="auto", help="Y-axis scale: auto (default), log, or linear")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Handle correlations mode or flag (doesn't need rollouts_path)
    if args.mode == "correlations" or args.plot_correlations:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            # If --rollouts was provided, use its parent directory
            rollouts_path = Path(args.rollouts)
            run_dir = rollouts_path.parent
        
        plot_all_correlations(run_dir)
        return
    
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)

    default_dir = rollouts_path.parent

    # Handle batch-all mode
    if args.batch_all:
        mode_desc = "histogram/PDF plots only" if args.histograms_only else "all standard plots"
        print(f"Generating {mode_desc} from {rollouts_path}")
        batch_plot_all(rollouts_path, default_dir, histograms_only=args.histograms_only)
        print("Batch plotting complete.")
        return

    # Handle batch-horizon mode (just horizon trajectory plots)
    if args.batch_horizon:
        mode_desc = "website-only horizon plots" if args.website_only else "horizon trajectory plots"
        print(f"Generating {mode_desc} from {rollouts_path}")
        from plotting_utils.helpers import load_present_day, load_central_trajectory_selection_start
        present_day = load_present_day(default_dir)
        print(f"Present day: {present_day}")
        selection_start = load_central_trajectory_selection_start(default_dir)
        if selection_start is not None:
            print(f"Central trajectory selection start: {selection_start}")

        try:
            generate_horizon_plots(rollouts_path, default_dir, present_day, selection_start, website_only=args.website_only)

            # Generate SC March 2027 filtered plot and extract central trajectory parameters
            if args.extract_central_params:
                try:
                    # March 2027 is approximately 2027.167 to 2027.25 (mid-March ~ 2027.208)
                    sc_march_2027_mid = 2027.208
                    sc_tolerance = 0.0415

                    # Filter to trajectories with SC in March 2027
                    sc_filtered_indices = [
                        i for i, t in enumerate(sc_times)
                        if t is not None and abs(t - sc_march_2027_mid) <= sc_tolerance
                    ]

                    if len(sc_filtered_indices) > 0:
                        trajectories_sc_filtered = [trajectories[i] for i in sc_filtered_indices]
                        aa_times_sc_filtered = [aa_times[i] for i in sc_filtered_indices] if aa_times else None
                        mse_sc_filtered = [mse_values[i] for i in sc_filtered_indices] if mse_values else None
                        sc_times_sc_filtered = [sc_times[i] for i in sc_filtered_indices]
                        ac_times_sc_filtered = [ac_times[i] for i in sc_filtered_indices] if ac_times else None

                        print(f"\nSC March 2027 filter: {len(sc_filtered_indices)} trajectories with SC within ±{sc_tolerance} of {sc_march_2027_mid}")

                        out_path = horizon_dir / "horizon_trajectories_sc_march2027.png"
                        plot_horizon_trajectories(
                            times,
                            trajectories_sc_filtered,
                            out_path,
                            max_trajectories=1000,
                            stop_at_sc=False,
                            aa_times=aa_times_sc_filtered,
                            mse_values=mse_sc_filtered,
                            shade_by_mse=True,
                            mse_threshold=None,
                            cutoff_year=2040.0,
                            title=f"Trajectories with SC in March 2027 ({sc_march_2027_mid:.3f} ± {sc_tolerance}y)",
                            sc_times=sc_times_sc_filtered,
                            present_day=present_day,
                            ac_times=ac_times_sc_filtered,
                            selection_start=selection_start,
                            show_both_central_methods=True,
                        )
                        print(f"Saved SC March 2027 filtered plot: {out_path} (n={len(sc_filtered_indices)})")

                        # Now find the central trajectory index and extract parameters
                        from plotting.trajectories import select_central_trajectory
                        central_idx_in_filtered, _, _ = select_central_trajectory(
                            times, trajectories_sc_filtered, sc_times_sc_filtered, present_day, ac_times=ac_times_sc_filtered
                        )

                        # Map back to original index
                        original_idx = sc_filtered_indices[central_idx_in_filtered]
                        print(f"Central trajectory: filtered index {central_idx_in_filtered}, original index {original_idx}")

                        # Read parameters for this rollout
                        import json
                        with open(rollouts_path) as f:
                            for i, line in enumerate(f):
                                if i == original_idx:
                                    rollout = json.loads(line)
                                    params = rollout.get('parameters', {})

                                    print(f"\n=== PRESET_PARAMETERS for central trajectory (SC March 2027) ===")
                                    print("'eli-th-preview-sc-march-2027': {")
                                    # Output in TypeScript format
                                    for key in sorted(params.keys()):
                                        val = params[key]
                                        if isinstance(val, str):
                                            print(f"    {key}: '{val}',")
                                        elif isinstance(val, bool):
                                            print(f"    {key}: {'true' if val else 'false'},")
                                        elif isinstance(val, (int, float)):
                                            print(f"    {key}: {val},")
                                    print("},")

                                    # Also save to a JSON file
                                    params_path = horizon_dir / "central_trajectory_params_sc_march_2027.json"
                                    with open(params_path, 'w') as pf:
                                        json.dump(params, pf, indent=2)
                                    print(f"\nSaved parameters to {params_path}")
                                    break
                    else:
                        print(f"Warning: No trajectories with SC in March 2027 (±{sc_tolerance})")
                except Exception as e:
                    print(f"Warning: Could not generate SC March 2027 filtered plot: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return

    if args.out is not None:
        out_path = Path(args.out)
    else:
        if args.mode == "sc_hist":
            out_path = default_dir / ("aa_time_hist_cdf.png" if args.cdf_hist else "aa_time_hist.png")
        elif args.mode == "horizon_at_sc_hist":
            out_path = default_dir / "horizon_at_sc_hist.png"
        elif args.mode == "milestone_time_hist":
            name = args.milestone or "milestone"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in name])
            suffix = "_cdf" if args.cdf_hist else ""
            out_path = default_dir / f"milestone_{safe}_hist{suffix}.png"
        elif args.mode == "milestone_compute_hist":
            name = args.milestone or "milestone"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in name])
            out_path = default_dir / f"milestone_{safe}_compute_hist.png"
        elif args.mode == "milestone_scatter":
            out_path = default_dir / "milestone_scatter.png"
        elif args.mode == "m_over_beta":
            out_path = default_dir / "m_over_beta_hist.png"
        elif args.mode == "takeoff_exponent":
            out_path = default_dir / "takeoff_exponent_hist.png"
        elif args.mode == "uplift_ratio":
            out_path = default_dir / "uplift_ratio_hist.png"
        elif args.mode == "metric_trajectories":
            metric_name = args.metric or "metric"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in metric_name])
            suffix = "_stop_at_sc" if args.stop_at_sc else ""
            out_path = default_dir / f"{safe}_trajectories{suffix}.png"
        else:
            if args.mode == "milestone_transition_box":
                out_path = default_dir / "milestone_transition_box.png"
            else:
                # Create horizon_trajectories subfolder for horizon plots
                horizon_dir = default_dir / "horizon_trajectories"
                horizon_dir.mkdir(parents=True, exist_ok=True)
                out_path = horizon_dir / ("horizon_trajectories_stop_at_sc.png" if args.stop_at_sc else "horizon_trajectories.png")

    if args.mode == "sc_hist":
        # sc_hist uses dedicated AA histogram functions (different from milestone_time_hist)
        reader = RolloutsReader(rollouts_path)
        aa_times, num_no_sc, sim_end = reader.read_aa_times()
        if (len(aa_times) + num_no_sc) > 0 and len(aa_times) > 0:
            arr = np.asarray(aa_times, dtype=float)
            print(f"Loaded {len(arr)} finite AC times (+{num_no_sc} No AC) from {rollouts_path}")
            print(f"Finite Min/Median/Max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
            with_inf = np.concatenate([arr, np.full(int(num_no_sc), np.inf)]) if num_no_sc > 0 else arr
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "No AC"
            print(f"P10/Median/P90 (incl. No AC): {_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}")
        elif (len(aa_times) + num_no_sc) > 0:
            print(f"Loaded 0 finite AC times (+{num_no_sc} No AC) from {rollouts_path}")

        if args.cdf_hist:
            plot_aa_time_histogram_cdf(aa_times, num_no_sc=num_no_sc, out_path=out_path, bins=int(args.bins), sim_end=sim_end)
        else:
            plot_aa_time_histogram(aa_times, num_no_sc=num_no_sc, out_path=out_path, bins=int(args.bins), sim_end=sim_end)
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "horizon_at_sc_hist":
        plot = HorizonAtSCHistogramPlot(run_dir=default_dir)
        result = plot.generate(bins=args.bins, out_path=out_path)
        print(f"Saved horizon at SC histogram to: {result}")
        return

    if args.mode == "milestone_time_hist":
        if not args.milestone and not args.list_milestones:
            raise ValueError("--milestone is required for milestone_time_hist mode (use --list-milestones to inspect names)")
        plot = MilestoneTimeHistogramPlot(run_dir=default_dir)
        result = plot.generate(
            milestone=args.milestone,
            bins=args.bins,
            cdf_hist=args.cdf_hist,
            list_milestones=args.list_milestones,
            out_path=out_path,
        )
        if result and result != Path():
            print(f"Saved histogram to: {result}")
        return

    if args.mode == "milestone_compute_hist":
        if not args.milestone and not args.list_milestones:
            raise ValueError("--milestone is required for milestone_compute_hist mode (use --list-milestones to inspect names)")
        plot = MilestoneComputeHistogramPlot(run_dir=default_dir)
        result = plot.generate(
            milestone=args.milestone,
            bins=args.bins,
            list_milestones=args.list_milestones,
            out_path=out_path,
        )
        if result and result != Path():
            print(f"Saved histogram to: {result}")
        return

    if args.mode == "milestone_transition_box":
        if not args.pairs and not args.list_milestones:
            raise ValueError("--pairs is required for milestone_transition_box mode (use --list-milestones to inspect names)")
        plot = TransitionBoxplotPlot(run_dir=default_dir)
        result = plot.generate(
            pairs=args.pairs,
            filter_milestone=args.filter_milestone,
            filter_by_year=float(args.filter_by_year) if args.filter_by_year is not None else None,
            ymin_years=float(args.ymin_years) if args.ymin_years is not None else None,
            ymax_years=float(args.ymax_years) if args.ymax_years is not None else None,
            exclude_inf_from_stats=args.exclude_inf_from_stats,
            inf_years=float(args.inf_years),
            list_milestones=args.list_milestones,
            out_path=out_path,
        )
        if result and result != Path():
            print(f"Saved transition boxplot to: {result}")
        return

    if args.mode == "milestone_scatter":
        if not args.scatter_pair:
            raise ValueError("--scatter-pair is required for milestone_scatter mode (e.g., 'AIR-5x:AIR-2000x')")
        plot = MilestoneScatterPlot(run_dir=default_dir)
        result = plot.generate(
            scatter_pair=args.scatter_pair,
            scatter_kind=args.scatter_kind,
            gridsize=int(args.gridsize),
            point_size=float(args.point_size),
            no_scatter_overlay=args.no_scatter_overlay,
            ymin_years=float(args.ymin_years) if args.ymin_years is not None else None,
            ymax_years=float(args.ymax_years) if args.ymax_years is not None else None,
            exclude_inf_from_stats=args.exclude_inf_from_stats,
            inf_years=float(args.inf_years),
            out_path=out_path,
        )
        print(f"Saved milestone scatter to: {result}")
        return

    if args.mode == "m_over_beta":
        plot = MOverBetaHistogramPlot(run_dir=default_dir)
        result = plot.generate(bins=args.bins, out_path=out_path)
        print(f"Saved m/β histogram to: {result}")
        return

    if args.mode == "takeoff_exponent":
        plot = TakeoffExponentHistogramPlot(run_dir=default_dir)
        result = plot.generate(bins=args.bins, out_path=out_path)
        print(f"Saved takeoff exponent histogram to: {result}")
        return

    if args.mode == "uplift_ratio":
        plot = UpliftRatioHistogramPlot(run_dir=default_dir)
        result = plot.generate(bins=args.bins, out_path=out_path)
        print(f"Saved uplift ratio histogram to: {result}")
        return

    if args.mode == "metric_trajectories":
        if not args.metric:
            raise ValueError("--metric is required for metric_trajectories mode")
        plot = MetricTrajectoryPlot(run_dir=default_dir)
        result = plot.generate(
            metric=args.metric,
            alpha=float(args.alpha),
            max_trajectories=int(args.max_trajectories),
            stop_at_sc=args.stop_at_sc,
            shade_by_mse=args.shade_by_mse,
            mse_threshold=args.mse_threshold,
            cutoff_year=args.cutoff_year,
            yscale=args.yscale,
            out_path=out_path,
        )
        print(f"Saved {args.metric} trajectories to: {result}")
        return

    # horizon_trajectories mode
    plot = MetricTrajectoryPlot(run_dir=default_dir)
    result = plot.generate(
        metric="horizon_lengths",
        current_horizon_minutes=float(args.current_horizon_minutes),
        alpha=float(args.alpha),
        max_trajectories=int(args.max_trajectories),
        no_metr=args.no_metr,
        stop_at_sc=args.stop_at_sc,
        shade_by_mse=args.shade_by_mse,
        mse_threshold=args.mse_threshold,
        cutoff_year=args.cutoff_year,
        out_path=out_path,
    )
    print(f"Saved horizon trajectories to: {result}")


if __name__ == "__main__":
    main()
