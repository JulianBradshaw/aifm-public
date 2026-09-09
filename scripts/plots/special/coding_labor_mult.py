#!/usr/bin/env python3
"""
Ad-hoc script to plot ai_coding_labor_mult_ref_present_day trajectories
with 80% confidence interval for AI-2027 SC target overlaid.

WARNING: This script contains bespoke data loading code that doesn't follow the
project's plotting utility conventions. It should be refactored to use utilities
from scripts/plotting_utils/rollouts_reader.py.

Also supports plotting histograms of:
- parallel_penalty parameter values
- AI2027-SC target values
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import functions from plot_rollouts.py (same directory)
from plot_rollouts import _read_metric_trajectories, _now_decimal_year


def read_ai2027_sc_targets(rollouts_file: Path) -> List[float]:
    """Extract AI2027-SC target values from milestones.

    Returns:
        List of target values (one per rollout)
    """
    targets: List[float] = []
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            results = rec.get("results")
            if not isinstance(results, dict):
                continue
            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue
            ai2027_sc = milestones.get("AI2027-SC")
            if ai2027_sc is not None and isinstance(ai2027_sc, dict):
                target = ai2027_sc.get("target")
                if target is not None:
                    try:
                        targets.append(float(target))
                    except (ValueError, TypeError):
                        pass
    return targets


def read_parallel_penalty_values(rollouts_file: Path) -> List[float]:
    """Extract parallel_penalty parameter values from rollouts.

    Returns:
        List of parallel_penalty values (one per rollout)
    """
    values: List[float] = []
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = rec.get("parameters")
            if not isinstance(params, dict):
                continue
            parallel_penalty = params.get("parallel_penalty")
            if parallel_penalty is not None:
                try:
                    values.append(float(parallel_penalty))
                except (ValueError, TypeError):
                    pass
    return values


def read_serial_coding_speedup_trajectories(rollouts_file: Path) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]], List[Optional[float]], List[float]]:
    """Compute serial coding speedup trajectories from rollouts.

    Serial coding speedup = ai_coding_labor_mult_ref_present_day ^ parallel_penalty

    Returns:
        times: common time array in decimal years
        trajectories: list of serial speedup arrays (one per rollout)
        aa_times: list of aa_time values (one per rollout)
        mse_values: list of METR MSE values (one per rollout)
        sc_targets: list of AI2027-SC serial speedup targets (one per rollout)
    """
    trajectories: List[np.ndarray] = []
    aa_times: List[Optional[float]] = []
    mse_values: List[Optional[float]] = []
    sc_targets: List[float] = []
    common_times: Optional[np.ndarray] = None

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get parallel_penalty from parameters
            params = rec.get("parameters")
            if not isinstance(params, dict):
                continue
            parallel_penalty = params.get("parallel_penalty")
            if parallel_penalty is None:
                continue

            # Get ai_coding_labor_mult trajectory from results
            results = rec.get("results")
            if not isinstance(results, dict):
                continue
            times = results.get("times")
            ai_coding_labor_mult = results.get("ai_coding_labor_mult_ref_present_day")
            aa_time_val = results.get("aa_time")
            mse_val = results.get("metr_mse")

            if times is None or ai_coding_labor_mult is None:
                continue

            try:
                times_arr = np.asarray(times, dtype=float)
                mult_arr = np.asarray(ai_coding_labor_mult, dtype=float)
                penalty = float(parallel_penalty)
            except Exception:
                continue

            if times_arr.ndim != 1 or mult_arr.ndim != 1 or times_arr.size != mult_arr.size:
                continue

            # Compute serial speedup trajectory = mult ^ penalty
            try:
                serial_speedup_arr = np.power(mult_arr, penalty)
            except (ValueError, OverflowError):
                continue

            if common_times is None:
                common_times = times_arr

            trajectories.append(serial_speedup_arr)

            # Get AI2027-SC target and compute serial speedup requirement
            milestones = results.get("milestones")
            if isinstance(milestones, dict):
                ai2027_sc = milestones.get("AI2027-SC")
                if ai2027_sc is not None and isinstance(ai2027_sc, dict):
                    target = ai2027_sc.get("target")
                    if target is not None:
                        try:
                            ai_coding_labor_mult_target = float(target)
                            # Serial speedup requirement = target ^ penalty
                            sc_speedup_target = ai_coding_labor_mult_target ** penalty
                            sc_targets.append(sc_speedup_target)
                        except (ValueError, TypeError, OverflowError):
                            pass

            try:
                aa_times.append(float(aa_time_val) if aa_time_val is not None and np.isfinite(float(aa_time_val)) else None)
            except Exception:
                aa_times.append(None)

            try:
                mse_values.append(float(mse_val) if mse_val is not None and np.isfinite(float(mse_val)) else None)
            except Exception:
                mse_values.append(None)

    if common_times is None or len(trajectories) == 0:
        raise ValueError("No serial coding speedup trajectories found in rollouts file")

    return common_times, trajectories, aa_times, mse_values, sc_targets


def plot_histogram(values: List[float], out_path: Path, title: str, xlabel: str, bins: int = 50, use_log_scale: bool = False, crop_to_p95: bool = False):
    """Plot a histogram of values."""
    if len(values) == 0:
        print(f"Warning: No values to plot for {title}")
        return

    arr = np.asarray(values, dtype=float)

    # Filter out non-positive values if using log scale
    if use_log_scale:
        arr = arr[arr > 0]
        if len(arr) == 0:
            print(f"Warning: No positive values to plot for {title}")
            return

    # Crop to P95 if requested
    p95 = None
    if crop_to_p95:
        p95 = np.percentile(arr, 95)
        arr_for_bins = arr[arr <= p95]
        print(f"  Cropping to P95: {p95:.2e} ({len(arr_for_bins)}/{len(arr)} values)")
    else:
        arr_for_bins = arr

    plt.figure(figsize=(12, 7))

    if use_log_scale:
        # Use log-spaced bins
        if len(arr_for_bins) > 0:
            log_bins = np.logspace(np.log10(arr_for_bins.min()), np.log10(arr_for_bins.max()), bins)
            plt.hist(arr_for_bins, bins=log_bins, edgecolor="black", alpha=0.7)
        plt.xscale("log")
    else:
        plt.hist(arr_for_bins, bins=bins, edgecolor="black", alpha=0.7)

    # Add statistics (computed on full dataset)
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    p10, p90 = np.percentile(arr, [10, 90])

    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2e}")
    plt.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.2e}")

    # Set x-axis limits if cropping
    if crop_to_p95 and p95 is not None:
        if use_log_scale:
            plt.xlim(right=p95)
        else:
            plt.xlim(right=p95)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    title_text = f"{title}\n(n={len(arr)} samples, P10={p10:.2e}, P90={p90:.2e}"
    if crop_to_p95 and p95 is not None:
        title_text += f", P95={p95:.2e})"
    else:
        title_text += ")"
    plt.title(title_text)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved histogram to: {out_path}")
    print(f"  Mean/Median: {mean_val:.2e} / {median_val:.2e}")
    print(f"  P10/P90: {p10:.2e} / {p90:.2e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot AI coding labor multiplier trajectories with AI2027-SC target CI, or plot histograms"
    )
    parser.add_argument("rollouts", type=str, help="Path to rollouts.jsonl file or run directory")
    parser.add_argument("--mode", type=str, choices=["trajectories", "parallel_penalty_hist", "ai2027sc_target_hist", "serial_speedup_hist"],
                        default="trajectories", help="What to plot")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histograms")
    parser.add_argument("--crop-to-p95", action="store_true", help="Crop x-axis to 95th percentile (for histograms)")
    args = parser.parse_args()

    # Resolve path to rollouts.jsonl
    input_path = Path(args.rollouts)
    if input_path.is_dir():
        rollouts_path = input_path / "rollouts.jsonl"
    else:
        rollouts_path = input_path

    if not rollouts_path.exists():
        print(f"Error: {rollouts_path} not found")
        sys.exit(1)

    # Handle different modes
    if args.mode == "parallel_penalty_hist":
        print("Reading parallel_penalty values...")
        values = read_parallel_penalty_values(rollouts_path)
        print(f"Loaded {len(values)} parallel_penalty values")
        out_path = rollouts_path.parent / "parallel_penalty_hist.png"
        plot_histogram(values, out_path, "Parallel Penalty Distribution", "parallel_penalty", bins=args.bins)
        return

    if args.mode == "ai2027sc_target_hist":
        print("Reading AI2027-SC target values...")
        values = read_ai2027_sc_targets(rollouts_path)
        print(f"Loaded {len(values)} AI2027-SC target values")
        out_path = rollouts_path.parent / "ai2027sc_target_hist.png"
        plot_histogram(values, out_path, "AI2027-SC Target Distribution", "AI2027-SC Target Value",
                      bins=args.bins, use_log_scale=True, crop_to_p95=args.crop_to_p95)
        return

    if args.mode == "serial_speedup_hist":
        print("Reading serial coding speedup trajectories (ai_coding_labor_mult ^ parallel_penalty)...")
        times, trajectories, aa_times, mse_values, sc_targets = read_serial_coding_speedup_trajectories(rollouts_path)
        print(f"Loaded {len(trajectories)} serial speedup trajectories with {len(times)} time points each")
        print(f"Loaded {len(sc_targets)} AI2027-SC serial speedup target values")
        out_path = rollouts_path.parent / "serial_speedup_trajectories.png"

        # Compute percentiles of AI2027-SC serial speedup targets
        if len(sc_targets) > 0:
            sc_targets_arr = np.asarray(sc_targets, dtype=float)
            # Filter out non-finite values
            sc_targets_arr = sc_targets_arr[np.isfinite(sc_targets_arr)]
            if len(sc_targets_arr) > 0:
                p10_sc = np.percentile(sc_targets_arr, 10)
                p50_sc = np.percentile(sc_targets_arr, 50)
                p60_sc = np.percentile(sc_targets_arr, 60)
                p70_sc = np.percentile(sc_targets_arr, 70)
                p80_sc = np.percentile(sc_targets_arr, 80)
                p90_sc = np.percentile(sc_targets_arr, 90)
                print(f"AI2027-SC serial speedup target P10/Median/P60/P70/P80/P90: {p10_sc:.1f} / {p50_sc:.1f} / {p60_sc:.1f} / {p70_sc:.1f} / {p80_sc:.1f} / {p90_sc:.1f}")
            else:
                p10_sc = p50_sc = p60_sc = p70_sc = p80_sc = p90_sc = None
        else:
            p10_sc = p50_sc = p60_sc = p70_sc = p80_sc = p90_sc = None

        # Plot trajectories with CI overlay

        # Clean trajectories
        cleaned: List[np.ndarray] = []
        for t in trajectories:
            arr = t.astype(float)
            arr[~np.isfinite(arr)] = np.nan
            cleaned.append(arr)

        # Compute median trajectory
        stacked = np.vstack(cleaned)
        median_traj = np.nanmedian(stacked, axis=0)

        # Create plot
        plt.figure(figsize=(14, 8))

        # Draw all trajectories
        alpha = 0.08
        max_trajectories = 2000
        num_plot = min(len(cleaned), max_trajectories)
        for i in range(num_plot):
            plt.plot(times, cleaned[i], color=(0.2, 0.5, 0.7, alpha), linewidth=1.0)

        # Central trajectory
        plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Median Trajectory")

        # Vertical line for current time
        now_year = _now_decimal_year()
        plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

        # Add AI2027-SC serial speedup target CI as shaded region
        if p10_sc is not None and p90_sc is not None:
            plt.axhspan(p10_sc, p90_sc, color="red", alpha=0.15,
                        label="AI2027-SC Serial Speedup Target 80% CI (P10-P90)")
            # Add percentile lines
            plt.axhline(p50_sc, color="red", linestyle="-", linewidth=2.0,
                        label=f"AI2027-SC Target P50 ({p50_sc:.0f})")
            plt.axhline(p60_sc, color="red", linestyle=":", linewidth=1.5,
                        label=f"AI2027-SC Target P60 ({p60_sc:.0f})")
            plt.axhline(p70_sc, color="red", linestyle=":", linewidth=1.5,
                        label=f"AI2027-SC Target P70 ({p70_sc:.0f})")
            plt.axhline(p80_sc, color="red", linestyle=":", linewidth=1.5,
                        label=f"AI2027-SC Target P80 ({p80_sc:.0f})")

        # Determine if we should use log scale
        finite_vals = stacked[np.isfinite(stacked)]
        use_log_scale = False
        if len(finite_vals) > 0:
            val_min = np.min(finite_vals)
            val_max = np.max(finite_vals)
            if val_min > 0 and val_max / val_min > 100:
                use_log_scale = True

        if use_log_scale:
            plt.yscale("log")

        plt.xlabel("Year")
        plt.ylabel("Serial Coding Speedup (ai_coding_labor_mult ^ parallel_penalty)")

        title = f"Serial Coding Speedup Trajectories with AI2027-SC Target\n"
        title += f"(n={len(trajectories)} trajectories)"
        plt.title(title)

        plt.grid(True, which="both", alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        print(f"Saved serial speedup trajectories to: {out_path}")
        return

    # Default mode: trajectories
    # Read metric trajectories
    metric_name = "ai_coding_labor_mult_ref_present_day"
    print(f"Reading {metric_name} trajectories from {rollouts_path}...")
    times, trajectories, aa_times, mse_values = _read_metric_trajectories(
        rollouts_path, metric_name
    )
    print(f"Loaded {len(trajectories)} trajectories with {len(times)} time points each")

    # Read AI2027-SC targets
    print("Reading AI2027-SC targets from milestones...")
    targets = read_ai2027_sc_targets(rollouts_path)
    print(f"Loaded {len(targets)} AI2027-SC target values")

    if len(targets) == 0:
        print("Warning: No AI2027-SC targets found in rollouts")

    # Compute percentiles (P10, P50, P60, P70, P80, P90)
    if len(targets) > 0:
        targets_arr = np.asarray(targets, dtype=float)
        p10_target = np.percentile(targets_arr, 10)
        p50_target = np.percentile(targets_arr, 50)
        p60_target = np.percentile(targets_arr, 60)
        p70_target = np.percentile(targets_arr, 70)
        p80_target = np.percentile(targets_arr, 80)
        p90_target = np.percentile(targets_arr, 90)
        print(f"AI2027-SC target P10/Median/P60/P70/P80/P90: {p10_target:.1f} / {p50_target:.1f} / {p60_target:.1f} / {p70_target:.1f} / {p80_target:.1f} / {p90_target:.1f}")
    else:
        p10_target = p50_target = p60_target = p70_target = p80_target = p90_target = None

    # Clean trajectories: replace non-finite values with NaN
    cleaned: List[np.ndarray] = []
    for t in trajectories:
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory
    stacked = np.vstack(cleaned)
    median_traj = np.nanmedian(stacked, axis=0)

    # Create plot
    plt.figure(figsize=(14, 8))

    # Draw all trajectories with low alpha
    alpha = 0.08
    max_trajectories = 2000
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        plt.plot(times, cleaned[i], color=(0.2, 0.5, 0.7, alpha), linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0,
             label="Median Trajectory")

    # Vertical line for current time
    now_year = _now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75,
                label="Current Time")

    # Add AI2027-SC target CI as shaded region
    if p10_target is not None and p90_target is not None:
        plt.axhspan(p10_target, p90_target, color="red", alpha=0.15,
                    label="AI2027-SC Target 80% CI (P10-P90)")
        # Add percentile lines
        plt.axhline(p50_target, color="red", linestyle="-", linewidth=2.0,
                    label=f"AI2027-SC Target P50 ({p50_target:.0f})")
        plt.axhline(p60_target, color="red", linestyle=":", linewidth=1.5,
                    label=f"AI2027-SC Target P60 ({p60_target:.0f})")
        plt.axhline(p70_target, color="red", linestyle=":", linewidth=1.5,
                    label=f"AI2027-SC Target P70 ({p70_target:.0f})")
        plt.axhline(p80_target, color="red", linestyle=":", linewidth=1.5,
                    label=f"AI2027-SC Target P80 ({p80_target:.0f})")

    # Determine if we should use log scale
    finite_vals = stacked[np.isfinite(stacked)]
    use_log_scale = False
    if len(finite_vals) > 0:
        val_min = np.min(finite_vals)
        val_max = np.max(finite_vals)
        if val_min > 0 and val_max / val_min > 100:
            use_log_scale = True

    if use_log_scale:
        plt.yscale("log")

    plt.xlabel("Year")
    plt.ylabel("AI Coding Labor Multiplier (ref: present day)")

    title = f"AI Coding Labor Multiplier Trajectories with AI2027-SC Target\n"
    title += f"(n={len(trajectories)} trajectories)"
    plt.title(title)

    plt.grid(True, which="both", alpha=0.25)
    plt.legend(loc="best")

    # Save plot
    out_path = rollouts_path.parent / "ai_coding_labor_mult_with_ci.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
