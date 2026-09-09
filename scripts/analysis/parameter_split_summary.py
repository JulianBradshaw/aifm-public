#!/usr/bin/env python3
"""
Generate summary tables and multi-parameter comparison plots for parameter splits.

This script:
1. Creates summary tables showing effect sizes (above vs below median) for all parameters
2. Generates multi-parameter comparison plots showing top N parameters for each outcome
3. Provides flexible interface for custom parameter comparisons

Usage:
  # Generate summary for all splits in a run
  python scripts/parameter_split_summary.py --run-dir outputs/251109_eli_2200

  # Generate custom comparison plot for specific parameters
  python scripts/parameter_split_summary.py --run-dir outputs/251109_eli_2200 \
    --compare-parameters ai_research_taste_at_coding_automation_anchor_sd,median_to_top_taste_multiplier \
    --outcome ac_timeline_2030 \
    --output custom_comparison.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()


def read_split_statistics(
    split_dir: Path,
    parameter_name: str
) -> Dict[str, Dict[str, float]]:
    """Read statistics from above and below median rollouts.

    Returns:
        Dict with keys 'above' and 'below', each containing statistics
    """
    stats = {'above': {}, 'below': {}}

    for split_type in ['above', 'below']:
        rollouts_file = split_dir / f"rollouts_{split_type}_median.jsonl"
        if not rollouts_file.exists():
            continue

        # Track milestone times and transitions
        ac_times = []
        sar_times = []
        ac_to_ted_durations = []
        ac_to_asi_durations = []
        sar_to_ted_durations = []
        sar_to_asi_durations = []
        sar_to_siar_durations = []

        with rollouts_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                    results = rec.get("results", {})
                    ms_dict = results.get("milestones", {})

                    # AC times
                    ac_info = ms_dict.get("AC")
                    if ac_info and isinstance(ac_info, dict):
                        ac_time = ac_info.get("time")
                        if ac_time is not None and np.isfinite(ac_time):
                            ac_times.append(float(ac_time))

                            # AC to TED-AI
                            ted_info = ms_dict.get("TED-AI")
                            if ted_info and isinstance(ted_info, dict):
                                ted_time = ted_info.get("time")
                                if ted_time is not None and np.isfinite(ted_time):
                                    ac_to_ted_durations.append(float(ted_time) - float(ac_time))

                            # AC to ASI
                            asi_info = ms_dict.get("ASI")
                            if asi_info and isinstance(asi_info, dict):
                                asi_time = asi_info.get("time")
                                if asi_time is not None and np.isfinite(asi_time):
                                    ac_to_asi_durations.append(float(asi_time) - float(ac_time))

                    # SAR times
                    sar_info = ms_dict.get("SAR-level-experiment-selection-skill")
                    if sar_info and isinstance(sar_info, dict):
                        sar_time = sar_info.get("time")
                        if sar_time is not None and np.isfinite(sar_time):
                            sar_times.append(float(sar_time))

                            # SAR to SIAR
                            siar_info = ms_dict.get("SIAR-level-experiment-selection-skill")
                            if siar_info and isinstance(siar_info, dict):
                                siar_time = siar_info.get("time")
                                if siar_time is not None and np.isfinite(siar_time):
                                    sar_to_siar_durations.append(float(siar_time) - float(sar_time))

                            # SAR to TED-AI
                            ted_info = ms_dict.get("TED-AI")
                            if ted_info and isinstance(ted_info, dict):
                                ted_time = ted_info.get("time")
                                if ted_time is not None and np.isfinite(ted_time):
                                    sar_to_ted_durations.append(float(ted_time) - float(sar_time))

                            # SAR to ASI
                            asi_info = ms_dict.get("ASI")
                            if asi_info and isinstance(asi_info, dict):
                                asi_time = asi_info.get("time")
                                if asi_time is not None and np.isfinite(asi_time):
                                    sar_to_asi_durations.append(float(asi_time) - float(sar_time))

                except Exception:
                    continue

        # Calculate statistics
        if ac_times:
            ac_times = np.array(ac_times)
            stats[split_type]['ac_median'] = float(np.median(ac_times))
            stats[split_type]['ac_p10'] = float(np.percentile(ac_times, 10))
            stats[split_type]['p_ac_by_2027'] = float(np.mean(ac_times <= 2027) * 100)
            stats[split_type]['p_ac_by_2030'] = float(np.mean(ac_times <= 2030) * 100)
            stats[split_type]['p_ac_by_2035'] = float(np.mean(ac_times <= 2035) * 100)

        if sar_times:
            sar_times = np.array(sar_times)
            stats[split_type]['sar_median'] = float(np.median(sar_times))
            stats[split_type]['sar_p10'] = float(np.percentile(sar_times, 10))
            stats[split_type]['p_sar_by_2027'] = float(np.mean(sar_times <= 2027) * 100)
            stats[split_type]['p_sar_by_2030'] = float(np.mean(sar_times <= 2030) * 100)
            stats[split_type]['p_sar_by_2035'] = float(np.mean(sar_times <= 2035) * 100)

        if ac_to_ted_durations:
            ac_to_ted = np.array(ac_to_ted_durations)
            stats[split_type]['p_ac_to_ted_1yr'] = float(np.mean(ac_to_ted <= 1.0) * 100)
            stats[split_type]['p_ac_to_ted_ai2027'] = float(np.mean(ac_to_ted < 9/12) * 100)

        if ac_to_asi_durations:
            ac_to_asi = np.array(ac_to_asi_durations)
            stats[split_type]['p_ac_to_asi_1yr'] = float(np.mean(ac_to_asi <= 1.0) * 100)
            stats[split_type]['p_ac_to_asi_ai2027'] = float(np.mean(ac_to_asi < 1.0) * 100)

        if sar_to_siar_durations:
            sar_to_siar = np.array(sar_to_siar_durations)
            stats[split_type]['p_sar_to_siar_ai2027'] = float(np.mean(sar_to_siar < 4/12) * 100)

        if sar_to_ted_durations:
            sar_to_ted = np.array(sar_to_ted_durations)
            stats[split_type]['p_sar_to_ted_1yr'] = float(np.mean(sar_to_ted <= 1.0) * 100)
            stats[split_type]['p_sar_to_ted_ai2027'] = float(np.mean(sar_to_ted < 3/12) * 100)

        if sar_to_asi_durations:
            sar_to_asi = np.array(sar_to_asi_durations)
            stats[split_type]['p_sar_to_asi_1yr'] = float(np.mean(sar_to_asi <= 1.0) * 100)
            stats[split_type]['p_sar_to_asi_ai2027'] = float(np.mean(sar_to_asi < 5/12) * 100)

    return stats


def generate_summary_tables(run_dir: Path) -> None:
    """Generate summary tables comparing above vs below median for all parameters."""
    splits_dir = run_dir / "parameter_splits"
    if not splits_dir.exists():
        print(f"No parameter_splits directory found in {run_dir}")
        return

    # Find all split directories
    split_dirs = [d for d in splits_dir.iterdir() if d.is_dir() and d.name.endswith("_split")]

    if not split_dirs:
        print(f"No parameter split directories found in {splits_dir}")
        return

    # Collect statistics for all parameters
    all_stats = {}
    for split_dir in split_dirs:
        parameter_name = split_dir.name[:-6].replace('-', '_')
        stats = read_split_statistics(split_dir, parameter_name)
        all_stats[parameter_name] = stats

    # Generate timeline summary table
    timeline_rows = []
    for param_name, stats in sorted(all_stats.items()):
        row = {'Parameter': param_name}

        # AC timelines
        for metric in ['ac_median', 'ac_p10', 'p_ac_by_2027', 'p_ac_by_2030', 'p_ac_by_2035']:
            below = stats['below'].get(metric)
            above = stats['above'].get(metric)
            if below is not None and above is not None:
                diff = above - below
                row[f'{metric}_diff'] = f"{diff:+.2f}"
            else:
                row[f'{metric}_diff'] = "N/A"

        # SAR timelines
        for metric in ['sar_median', 'sar_p10', 'p_sar_by_2027', 'p_sar_by_2030', 'p_sar_by_2035']:
            below = stats['below'].get(metric)
            above = stats['above'].get(metric)
            if below is not None and above is not None:
                diff = above - below
                row[f'{metric}_diff'] = f"{diff:+.2f}"
            else:
                row[f'{metric}_diff'] = "N/A"

        timeline_rows.append(row)

    # Write timeline table
    timeline_output = run_dir / "parameter_effects_timelines.csv"
    timeline_fields = ['Parameter',
                      'ac_median_diff', 'ac_p10_diff',
                      'p_ac_by_2027_diff', 'p_ac_by_2030_diff', 'p_ac_by_2035_diff',
                      'sar_median_diff', 'sar_p10_diff',
                      'p_sar_by_2027_diff', 'p_sar_by_2030_diff', 'p_sar_by_2035_diff']

    with timeline_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timeline_fields)
        writer.writeheader()
        writer.writerows(timeline_rows)

    print(f"Saved timeline effects table: {timeline_output}")

    # Generate takeoff summary table
    takeoff_rows = []
    for param_name, stats in sorted(all_stats.items()):
        row = {'Parameter': param_name}

        # One-year takeoff probabilities
        for metric in ['p_ac_to_ted_1yr', 'p_ac_to_asi_1yr', 'p_sar_to_ted_1yr', 'p_sar_to_asi_1yr']:
            below = stats['below'].get(metric)
            above = stats['above'].get(metric)
            if below is not None and above is not None:
                diff = above - below
                row[f'{metric}_diff'] = f"{diff:+.2f}"
            else:
                row[f'{metric}_diff'] = "N/A"

        # AI 2027 speed takeoff probabilities
        for metric in ['p_ac_to_ted_ai2027', 'p_ac_to_asi_ai2027',
                      'p_sar_to_siar_ai2027', 'p_sar_to_ted_ai2027', 'p_sar_to_asi_ai2027']:
            below = stats['below'].get(metric)
            above = stats['above'].get(metric)
            if below is not None and above is not None:
                diff = above - below
                row[f'{metric}_diff'] = f"{diff:+.2f}"
            else:
                row[f'{metric}_diff'] = "N/A"

        takeoff_rows.append(row)

    # Write takeoff table
    takeoff_output = run_dir / "parameter_effects_takeoff.csv"
    takeoff_fields = ['Parameter',
                     'p_ac_to_ted_1yr_diff', 'p_ac_to_asi_1yr_diff',
                     'p_sar_to_ted_1yr_diff', 'p_sar_to_asi_1yr_diff',
                     'p_ac_to_ted_ai2027_diff', 'p_ac_to_asi_ai2027_diff',
                     'p_sar_to_siar_ai2027_diff', 'p_sar_to_ted_ai2027_diff', 'p_sar_to_asi_ai2027_diff']

    with takeoff_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=takeoff_fields)
        writer.writeheader()
        writer.writerows(takeoff_rows)

    print(f"Saved takeoff effects table: {takeoff_output}")


def read_parameter_percentiles(
    run_dir: Path,
    parameter_names: List[str]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read percentile data for parameters from combined rollouts.

    Returns:
        Dict mapping parameter name to (percentiles, values) arrays
    """
    rollouts_file = run_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    param_data = {p: [] for p in parameter_names}

    with rollouts_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                for param_name in parameter_names:
                    val = all_params.get(param_name)
                    if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
                        param_data[param_name].append(float(val))

            except Exception:
                continue

    # Convert to percentiles
    percentile_data = {}
    for param_name, values in param_data.items():
        if len(values) > 0:
            values = np.array(values)
            # Calculate percentile for each value
            percentiles = np.array([
                (values <= v).sum() / len(values) * 100 for v in values
            ])
            percentile_data[param_name] = (percentiles, values)

    return percentile_data


def plot_multi_parameter_comparison(
    run_dir: Path,
    parameter_names: List[str],
    outcome_metric: str,
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """Plot outcome vs parameter percentile for multiple parameters.

    Args:
        run_dir: Run directory containing rollouts.jsonl
        parameter_names: List of parameter names to compare
        outcome_metric: Outcome metric to plot (e.g., 'p_ac_by_2030')
        output_path: Output file path
        title: Optional custom title
    """
    rollouts_file = run_dir / "rollouts.jsonl"

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(parameter_names)))

    for param_name, color in zip(parameter_names, colors):
        # Read data
        param_values = []
        outcome_values = []

        with rollouts_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                    params = rec.get("parameters", {})
                    ts_params = rec.get("time_series_parameters", {})
                    results = rec.get("results", {})
                    all_params = {**params, **ts_params, **results}

                    param_val = all_params.get(param_name)
                    if param_val is None or not isinstance(param_val, (int, float)) or not np.isfinite(param_val):
                        continue

                    # Calculate outcome metric
                    outcome_val = None
                    ms_dict = results.get("milestones", {})

                    if outcome_metric.startswith('p_ac_by_'):
                        year = int(outcome_metric.split('_')[-1])
                        ac_info = ms_dict.get("AC")
                        if ac_info and isinstance(ac_info, dict):
                            ac_time = ac_info.get("time")
                            if ac_time is not None and np.isfinite(ac_time):
                                outcome_val = 1.0 if ac_time <= year else 0.0

                    elif outcome_metric.startswith('p_sar_by_'):
                        year = int(outcome_metric.split('_')[-1])
                        sar_info = ms_dict.get("SAR-level-experiment-selection-skill")
                        if sar_info and isinstance(sar_info, dict):
                            sar_time = sar_info.get("time")
                            if sar_time is not None and np.isfinite(sar_time):
                                outcome_val = 1.0 if sar_time <= year else 0.0

                    elif outcome_metric.startswith('p_') and '_to_' in outcome_metric and '_1yr' in outcome_metric:
                        # One-year takeoff metrics
                        parts = outcome_metric.split('_')
                        from_ms = parts[1]  # e.g., 'ac' or 'sar'
                        to_ms = parts[3]  # e.g., 'ted' or 'asi'

                        from_milestone = "AC" if from_ms == "ac" else "SAR-level-experiment-selection-skill"
                        to_milestone = "TED-AI" if to_ms == "ted" else "ASI"

                        from_info = ms_dict.get(from_milestone)
                        to_info = ms_dict.get(to_milestone)

                        if from_info and to_info:
                            from_time = from_info.get("time")
                            to_time = to_info.get("time")
                            if from_time is not None and to_time is not None and np.isfinite(from_time) and np.isfinite(to_time):
                                duration = to_time - from_time
                                outcome_val = 1.0 if duration <= 1.0 else 0.0

                    if outcome_val is not None:
                        param_values.append(float(param_val))
                        outcome_values.append(float(outcome_val))

                except Exception:
                    continue

        if len(param_values) < 20:
            print(f"Insufficient data for {param_name}: only {len(param_values)} samples")
            continue

        param_values = np.array(param_values)
        outcome_values = np.array(outcome_values)

        # Convert to percentiles
        percentiles = np.array([
            (param_values <= v).sum() / len(param_values) * 100 for v in param_values
        ])

        # Bin by percentile
        n_bins = 20
        percentile_bins = np.linspace(0, 100, n_bins + 1)
        bin_centers = []
        bin_means = []
        bin_stds = []

        for i in range(len(percentile_bins) - 1):
            if i == len(percentile_bins) - 2:
                mask = (percentiles >= percentile_bins[i]) & (percentiles <= percentile_bins[i + 1])
            else:
                mask = (percentiles >= percentile_bins[i]) & (percentiles < percentile_bins[i + 1])

            if mask.sum() > 0:
                bin_centers.append((percentile_bins[i] + percentile_bins[i + 1]) / 2)
                bin_means.append(float(np.mean(outcome_values[mask]) * 100))
                p = np.mean(outcome_values[mask])
                se = np.sqrt(p * (1 - p) / mask.sum()) * 100 if mask.sum() > 0 else 0
                bin_stds.append(se)

        if len(bin_centers) > 2:
            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)

            label = param_name.replace('_', ' ').title().replace('Ai ', 'AI ')
            ax.plot(bin_centers, bin_means, 'o-', linewidth=2, markersize=6,
                   color=color, label=label, alpha=0.8)

    ax.set_xlabel("Parameter Percentile", fontsize=12)
    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)

    if title is None:
        title = f"{outcome_metric.replace('_', ' ').title()} vs Parameter Percentile"
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved multi-parameter comparison: {output_path}")


def generate_top_parameter_plots(run_dir: Path, top_n: int = 5) -> None:
    """Generate plots showing top N parameters for key outcomes.

    Uses permutation importance from sensitivity analysis to identify the most impactful parameters.
    Falls back to effect size differences if sensitivity analysis is not available.
    """
    # Try to read permutation importance from sensitivity analysis
    sensitivity_dir = run_dir / "sensitivity_analysis"

    # Map outcome metrics to their corresponding sensitivity analysis files
    # Note: This assumes sensitivity analyses have been run for each transition
    # Uses censored versions which treat non-achieved milestones as achieved at simulation cutoff
    outcome_to_sensitivity = {
        'p_ac_by_2027': 'sensitivity_SAR-level-experiment-selection-skill_arrival.json',
        'p_ac_by_2030': 'sensitivity_SAR-level-experiment-selection-skill_arrival.json',
        'p_sar_by_2027': 'sensitivity_SAR-level-experiment-selection-skill_arrival.json',
        'p_sar_by_2030': 'sensitivity_SAR-level-experiment-selection-skill_arrival.json',
        'p_ac_to_ted_1yr': 'sensitivity_AC_to_TED-AI_censored.json',
        'p_ac_to_asi_1yr': 'sensitivity_AC_to_ASI_censored.json',
        'p_sar_to_ted_1yr': 'sensitivity_SAR-level-experiment-selection-skill_to_TED-AI_censored.json',
        'p_sar_to_asi_1yr': 'sensitivity_SAR-level-experiment-selection-skill_to_ASI_censored.json',
    }

    # Read permutation importance for all available analyses
    permutation_importance = {}
    if sensitivity_dir.exists():
        for outcome_metric, sensitivity_file in outcome_to_sensitivity.items():
            sensitivity_path = sensitivity_dir / sensitivity_file
            if sensitivity_path.exists():
                try:
                    with sensitivity_path.open("r") as f:
                        data = json.load(f)

                        # Try permutation importance first (transition duration analyses)
                        perm_imp = data.get("permutation_importance", {})
                        if perm_imp:
                            # Extract mean_drop_in_r2 for each parameter
                            permutation_importance[outcome_metric] = {
                                param: info.get("mean_drop_in_r2", 0.0)
                                for param, info in perm_imp.items()
                            }
                            print(f"Loaded permutation importance for {outcome_metric} from {sensitivity_file}")
                        else:
                            # Fall back to Spearman correlations (arrival time analyses)
                            spearman_corrs = data.get("spearman_correlations", [])
                            if spearman_corrs:
                                # Use absolute Spearman rho as importance score
                                # Negate because we want parameters that make arrival earlier (negative correlation)
                                # to show as positive importance for "probability of arriving by year X"
                                permutation_importance[outcome_metric] = {
                                    item["parameter"]: -item["rho"]  # Negate: negative rho = earlier arrival = higher probability
                                    for item in spearman_corrs
                                }
                                print(f"Loaded Spearman correlations for {outcome_metric} from {sensitivity_file}")
                except Exception as e:
                    print(f"Warning: Could not read {sensitivity_path}: {e}")

    # Fallback: read effect sizes from summary tables
    timeline_table = run_dir / "parameter_effects_timelines.csv"
    takeoff_table = run_dir / "parameter_effects_takeoff.csv"

    timeline_effects = {}
    takeoff_effects = {}

    if timeline_table.exists():
        with timeline_table.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row['Parameter']
                timeline_effects[param] = {k: float(v) if v not in ['N/A', ''] else 0.0
                                          for k, v in row.items() if k != 'Parameter'}

    if takeoff_table.exists():
        with takeoff_table.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row['Parameter']
                takeoff_effects[param] = {k: float(v) if v not in ['N/A', ''] else 0.0
                                         for k, v in row.items() if k != 'Parameter'}

    # Create output directory
    comparison_dir = run_dir / "top_parameter_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Key outcomes to plot
    outcomes = [
        ('p_ac_by_2027_diff', 'P(AC by Dec 2027)', 'p_ac_by_2027'),
        ('p_ac_by_2030_diff', 'P(AC by Dec 2030)', 'p_ac_by_2030'),
        ('p_sar_by_2027_diff', 'P(SAR by Dec 2027)', 'p_sar_by_2027'),
        ('p_sar_by_2030_diff', 'P(SAR by Dec 2030)', 'p_sar_by_2030'),
        ('p_ac_to_ted_1yr_diff', 'P(AC→TED-AI ≤ 1yr)', 'p_ac_to_ted_1yr'),
        ('p_ac_to_asi_1yr_diff', 'P(AC→ASI ≤ 1yr)', 'p_ac_to_asi_1yr'),
        ('p_sar_to_ted_1yr_diff', 'P(SAR→TED-AI ≤ 1yr)', 'p_sar_to_ted_1yr'),
        ('p_sar_to_asi_1yr_diff', 'P(SAR→ASI ≤ 1yr)', 'p_sar_to_asi_1yr'),
    ]

    for effect_metric, display_name, outcome_metric in outcomes:
        # Prefer permutation importance if available
        if outcome_metric in permutation_importance:
            effects = permutation_importance[outcome_metric]
            print(f"\nUsing permutation importance for {outcome_metric}")
        else:
            # Fallback to simple effect size difference
            if effect_metric in (list(timeline_effects.values())[0] if timeline_effects else {}):
                effects = {p: abs(timeline_effects[p].get(effect_metric, 0.0))
                          for p in timeline_effects.keys()}
            else:
                effects = {p: abs(takeoff_effects[p].get(effect_metric, 0.0))
                          for p in takeoff_effects.keys()}
            print(f"\nUsing effect size differences for {outcome_metric} (no sensitivity analysis found)")

        # Sort by absolute value of effect and take top N
        top_params = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        top_param_names = [p[0] for p in top_params]

        print(f"Top {top_n} parameters for {outcome_metric}:")
        for i, (param, effect) in enumerate(top_params, 1):
            print(f"  {i}. {param}: {effect:.4f}")

        if not top_param_names:
            print(f"  No parameters found for {outcome_metric}")
            continue

        # Generate plot
        output_path = comparison_dir / f"top{top_n}_{outcome_metric}.png"
        plot_multi_parameter_comparison(
            run_dir,
            top_param_names,
            outcome_metric,
            output_path,
            title=f"Top {top_n} Parameters for {display_name} (by permutation importance)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter split summaries and comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    parser.add_argument(
        "--compare-parameters",
        type=str,
        help="Comma-separated list of parameters to compare (for custom plots)"
    )
    parser.add_argument(
        "--outcome",
        type=str,
        help="Outcome metric for custom comparison (e.g., 'p_ac_by_2030')"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for custom comparison plot"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top parameters to include in comparison plots (default: 5)"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Generate summary tables
    print("Generating summary tables...")
    generate_summary_tables(run_dir)

    # Generate top parameter comparison plots
    print(f"\nGenerating top {args.top_n} parameter comparison plots...")
    generate_top_parameter_plots(run_dir, top_n=args.top_n)

    # Generate custom comparison if requested
    if args.compare_parameters and args.outcome and args.output:
        print("\nGenerating custom comparison plot...")
        param_names = [p.strip() for p in args.compare_parameters.split(',')]
        output_path = Path(args.output)
        plot_multi_parameter_comparison(run_dir, param_names, args.outcome, output_path)

    print("\n✓ Summary generation complete!")


if __name__ == "__main__":
    main()
