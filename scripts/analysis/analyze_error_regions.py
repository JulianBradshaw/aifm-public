#!/usr/bin/env python3
"""
Analyze error regions in quantile space to identify parameter ranges causing failures.

WARNING: This script contains bespoke data loading and plotting code that doesn't
follow the project's plotting utility conventions. It should be refactored to use
utilities from scripts/plotting/ and scripts/plotting_utils/.

This script:
1. Loads samples_quantile.jsonl and identifies failed vs successful samples
2. Compares parameter distributions between failures and successes
3. Identifies parameter ranges with high error rates
4. Generates visualizations and a summary report

Usage:
  python scripts/analyze_error_regions.py --run-dir outputs/20251023_184450/ --plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze error regions in quantile space")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--plot", action="store_true", help="Generate visualizations")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for quantile ranges")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum samples per bin to report")
    return parser.parse_args()


def load_quantile_samples(run_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and separate successful vs failed samples (both quantile and actual values).

    Returns:
        Tuple of (successful_quantiles, failed_quantiles, successful_actuals, failed_actuals)
    """
    quantile_path = run_dir / "samples_quantile.jsonl"
    actual_path = run_dir / "samples.jsonl"

    if not actual_path.exists():
        print(f"Error: samples.jsonl not found in {run_dir}")
        sys.exit(1)

    if not quantile_path.exists():
        print(f"samples_quantile.jsonl not found. Running samples_to_quantiles.py...")
        import subprocess

        # Run samples_to_quantiles.py
        script_path = REPO_ROOT / "scripts" / "samples_to_quantiles.py"
        if not script_path.exists():
            print(f"Error: samples_to_quantiles.py not found at {script_path}")
            sys.exit(1)

        result = subprocess.run(
            ["python3", str(script_path), "--run-dir", str(run_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error running samples_to_quantiles.py:")
            print(result.stderr)
            sys.exit(1)

        print(result.stdout)

        # Verify the file was created
        if not quantile_path.exists():
            print(f"Error: samples_quantile.jsonl was not created successfully")
            sys.exit(1)

        print(f"Successfully created {quantile_path}")
        print()

    successful_quantiles = []
    failed_quantiles = []
    successful_actuals = []
    failed_actuals = []

    # Load quantile samples
    with quantile_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            if "error" in sample:
                failed_quantiles.append(sample)
            else:
                successful_quantiles.append(sample)

    # Load actual samples
    with actual_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            if "error" in sample:
                failed_actuals.append(sample)
            else:
                successful_actuals.append(sample)

    return successful_quantiles, failed_quantiles, successful_actuals, failed_actuals


def extract_parameter_values(samples: List[Dict[str, Any]], param_name: str, 
                             group: str = "parameters") -> List[float]:
    """Extract all non-None values for a parameter from samples."""
    values = []
    for sample in samples:
        if sample.get(group) is not None:
            val = sample[group].get(param_name)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))
    return values


def analyze_parameter_distributions(successful: List[Dict[str, Any]],
                                    failed: List[Dict[str, Any]],
                                    successful_actuals: List[Dict[str, Any]] = None,
                                    failed_actuals: List[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Compare parameter distributions between successful and failed samples."""

    # Collect all parameter names
    all_param_names = set()
    for sample in successful + failed:
        if sample.get("parameters"):
            all_param_names.update(sample["parameters"].keys())
        if sample.get("time_series_parameters"):
            all_param_names.update(sample["time_series_parameters"].keys())

    results = {}

    for param_name in sorted(all_param_names):
        # Try both parameter groups (quantile values)
        success_vals = extract_parameter_values(successful, param_name, "parameters")
        fail_vals = extract_parameter_values(failed, param_name, "parameters")

        if not success_vals:
            success_vals = extract_parameter_values(successful, param_name, "time_series_parameters")
        if not fail_vals:
            fail_vals = extract_parameter_values(failed, param_name, "time_series_parameters")

        if not success_vals or not fail_vals:
            continue

        # Compute statistics
        success_mean = np.mean(success_vals)
        success_std = np.std(success_vals)
        fail_mean = np.mean(fail_vals)
        fail_std = np.std(fail_vals)

        # Compute percentiles (quantile space)
        success_percentiles = np.percentile(success_vals, [10, 25, 50, 75, 90])
        fail_percentiles = np.percentile(fail_vals, [10, 25, 50, 75, 90])

        # Two-sample t-test (if we have enough samples)
        from scipy import stats
        if len(success_vals) >= 3 and len(fail_vals) >= 3:
            t_stat, p_value = stats.ttest_ind(success_vals, fail_vals)
        else:
            t_stat, p_value = None, None

        # Kolmogorov-Smirnov test (distribution difference)
        if len(success_vals) >= 3 and len(fail_vals) >= 3:
            ks_stat, ks_p = stats.ks_2samp(success_vals, fail_vals)
        else:
            ks_stat, ks_p = None, None

        result_dict = {
            "success_count": len(success_vals),
            "fail_count": len(fail_vals),
            "success_mean": success_mean,
            "success_std": success_std,
            "fail_mean": fail_mean,
            "fail_std": fail_std,
            "mean_difference": fail_mean - success_mean,
            "success_percentiles": success_percentiles,
            "fail_percentiles": fail_percentiles,
            "t_statistic": t_stat,
            "t_p_value": p_value,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
        }

        # Also compute percentiles from actual parameter values if provided
        if successful_actuals is not None and failed_actuals is not None:
            success_actual_vals = extract_parameter_values(successful_actuals, param_name, "parameters")
            fail_actual_vals = extract_parameter_values(failed_actuals, param_name, "parameters")

            if not success_actual_vals:
                success_actual_vals = extract_parameter_values(successful_actuals, param_name, "time_series_parameters")
            if not fail_actual_vals:
                fail_actual_vals = extract_parameter_values(failed_actuals, param_name, "time_series_parameters")

            if success_actual_vals and fail_actual_vals:
                success_actual_percentiles = np.percentile(success_actual_vals, [10, 25, 50, 75, 90])
                fail_actual_percentiles = np.percentile(fail_actual_vals, [10, 25, 50, 75, 90])
                result_dict["success_actual_percentiles"] = success_actual_percentiles
                result_dict["fail_actual_percentiles"] = fail_actual_percentiles

        results[param_name] = result_dict

    return results


def find_high_error_regions(successful: List[Dict[str, Any]], 
                            failed: List[Dict[str, Any]], 
                            bins: int = 10,
                            min_samples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Find quantile ranges with high error rates for each parameter."""
    
    # Collect all parameter names
    all_param_names = set()
    for sample in successful + failed:
        if sample.get("parameters"):
            all_param_names.update(sample["parameters"].keys())
        if sample.get("time_series_parameters"):
            all_param_names.update(sample["time_series_parameters"].keys())
    
    high_error_regions = {}
    
    for param_name in sorted(all_param_names):
        # Get values from both groups
        success_vals = extract_parameter_values(successful, param_name, "parameters")
        fail_vals = extract_parameter_values(failed, param_name, "parameters")
        
        if not success_vals:
            success_vals = extract_parameter_values(successful, param_name, "time_series_parameters")
        if not fail_vals:
            fail_vals = extract_parameter_values(failed, param_name, "time_series_parameters")
        
        if not success_vals and not fail_vals:
            continue
        
        # Create bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Count successes and failures in each bin
        success_counts = np.histogram(success_vals, bins=bin_edges)[0]
        fail_counts = np.histogram(fail_vals, bins=bin_edges)[0]
        total_counts = success_counts + fail_counts
        
        # Calculate error rates
        error_rates = np.zeros(bins)
        for i in range(bins):
            if total_counts[i] >= min_samples:
                error_rates[i] = fail_counts[i] / total_counts[i]
            else:
                error_rates[i] = np.nan
        
        # Find bins with high error rates (>50% and sufficient samples)
        high_error_bins = []
        for i in range(bins):
            if total_counts[i] >= min_samples and error_rates[i] > 0.5:
                high_error_bins.append({
                    "bin_index": i,
                    "range": (bin_edges[i], bin_edges[i + 1]),
                    "center": bin_centers[i],
                    "total_samples": int(total_counts[i]),
                    "failures": int(fail_counts[i]),
                    "successes": int(success_counts[i]),
                    "error_rate": float(error_rates[i]),
                })
        
        if high_error_bins:
            high_error_regions[param_name] = high_error_bins
    
    return high_error_regions


def plot_parameter_comparison(param_name: str, 
                              successful: List[Dict[str, Any]], 
                              failed: List[Dict[str, Any]],
                              output_path: Path) -> None:
    """Plot histogram comparison of a parameter between successful and failed samples."""
    
    success_vals = extract_parameter_values(successful, param_name, "parameters")
    fail_vals = extract_parameter_values(failed, param_name, "parameters")
    
    if not success_vals:
        success_vals = extract_parameter_values(successful, param_name, "time_series_parameters")
    if not fail_vals:
        fail_vals = extract_parameter_values(failed, param_name, "time_series_parameters")
    
    if not success_vals or not fail_vals:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)
    
    ax.hist(success_vals, bins=bins, alpha=0.5, label=f"Success (n={len(success_vals)})", 
            color="green", edgecolor="black")
    ax.hist(fail_vals, bins=bins, alpha=0.5, label=f"Failed (n={len(fail_vals)})", 
            color="red", edgecolor="black")
    
    ax.set_xlabel("Quantile", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of {param_name}\n(Success vs Failed Samples)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_rate_heatmap(successful: List[Dict[str, Any]], 
                            failed: List[Dict[str, Any]],
                            output_path: Path,
                            bins: int = 10,
                            min_samples: int = 3,
                            top_n: int = 20) -> None:
    """Plot heatmap of error rates across parameter ranges."""
    
    # Collect all parameter names
    all_param_names = set()
    for sample in successful + failed:
        if sample.get("parameters"):
            all_param_names.update(sample["parameters"].keys())
        if sample.get("time_series_parameters"):
            all_param_names.update(sample["time_series_parameters"].keys())
    
    # Calculate error rates for each parameter and bin
    param_error_data = {}
    
    for param_name in sorted(all_param_names):
        success_vals = extract_parameter_values(successful, param_name, "parameters")
        fail_vals = extract_parameter_values(failed, param_name, "parameters")
        
        if not success_vals:
            success_vals = extract_parameter_values(successful, param_name, "time_series_parameters")
        if not fail_vals:
            fail_vals = extract_parameter_values(failed, param_name, "time_series_parameters")
        
        if not success_vals and not fail_vals:
            continue
        
        bin_edges = np.linspace(0, 1, bins + 1)
        success_counts = np.histogram(success_vals, bins=bin_edges)[0]
        fail_counts = np.histogram(fail_vals, bins=bin_edges)[0]
        total_counts = success_counts + fail_counts
        
        error_rates = np.zeros(bins)
        for i in range(bins):
            if total_counts[i] >= min_samples:
                error_rates[i] = fail_counts[i] / total_counts[i]
            else:
                error_rates[i] = np.nan
        
        # Calculate max error rate for sorting
        valid_rates = error_rates[~np.isnan(error_rates)]
        if len(valid_rates) > 0:
            max_error_rate = np.max(valid_rates)
            param_error_data[param_name] = (error_rates, max_error_rate)
    
    # Select top parameters by max error rate
    sorted_params = sorted(param_error_data.items(), key=lambda x: x[1][1], reverse=True)[:top_n]
    
    if not sorted_params:
        print("Not enough data to generate error rate heatmap")
        return
    
    param_names = [p[0] for p in sorted_params]
    error_matrix = np.array([p[1][0] for p in sorted_params])
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(param_names) * 0.4)))
    
    # Plot heatmap
    im = ax.imshow(error_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1, 
                   interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(bins))
    ax.set_yticks(np.arange(len(param_names)))
    
    # Set tick labels
    bin_labels = [f"{i*10}-{(i+1)*10}%" for i in range(bins)]
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Error Rate", fontsize=12)
    
    # Add title and labels
    ax.set_xlabel("Quantile Range", fontsize=12)
    ax.set_ylabel("Parameter", fontsize=12)
    ax.set_title(f"Error Rates by Parameter and Quantile Range\n(Top {len(param_names)} parameters by max error rate)", 
                 fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(successful: List[Dict[str, Any]],
                   failed: List[Dict[str, Any]],
                   successful_actuals: List[Dict[str, Any]],
                   failed_actuals: List[Dict[str, Any]],
                   dist_analysis: Dict[str, Dict[str, Any]],
                   high_error_regions: Dict[str, List[Dict[str, Any]]],
                   output_path: Path) -> None:
    """Generate a text report summarizing error regions."""

    with output_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("ERROR REGION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total successful samples: {len(successful)}\n")
        f.write(f"Total failed samples: {len(failed)}\n")
        f.write(f"Overall error rate: {len(failed) / (len(successful) + len(failed)) * 100:.1f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("HIGH ERROR RATE REGIONS (>50% error rate)\n")
        f.write("=" * 80 + "\n\n")

        if not high_error_regions:
            f.write("No high-error regions found.\n\n")
        else:
            for param_name in sorted(high_error_regions.keys()):
                f.write(f"\n{param_name}:\n")
                f.write("-" * 60 + "\n")

                for region in high_error_regions[param_name]:
                    f.write(f"  Quantile range: {region['range'][0]:.2f} - {region['range'][1]:.2f}\n")
                    f.write(f"    Total samples: {region['total_samples']}\n")
                    f.write(f"    Failures: {region['failures']}\n")
                    f.write(f"    Successes: {region['successes']}\n")
                    f.write(f"    Error rate: {region['error_rate'] * 100:.1f}%\n")
                    f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("PARAMETER DISTRIBUTION COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        # Sort by KS statistic (largest difference in distributions)
        sorted_params = sorted(dist_analysis.items(),
                              key=lambda x: x[1]['ks_statistic'] if x[1]['ks_statistic'] is not None else 0,
                              reverse=True)

        f.write("Parameters sorted by distribution difference (KS statistic):\n\n")

        for param_name, stats in sorted_params[:20]:  # Top 20
            f.write(f"\n{param_name}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Success mean: {stats['success_mean']:.4f} ± {stats['success_std']:.4f}\n")
            f.write(f"  Failure mean: {stats['fail_mean']:.4f} ± {stats['fail_std']:.4f}\n")
            f.write(f"  Mean difference: {stats['mean_difference']:.4f}\n")

            if stats['ks_statistic'] is not None:
                f.write(f"  KS statistic: {stats['ks_statistic']:.4f}\n")
                f.write(f"  KS p-value: {stats['ks_p_value']:.4e}\n")
                significance = "***" if stats['ks_p_value'] < 0.001 else ("**" if stats['ks_p_value'] < 0.01 else ("*" if stats['ks_p_value'] < 0.05 else "n.s."))
                f.write(f"  Significance: {significance}\n")

            f.write(f"\n  QUANTILE PERCENTILES (10,25,50,75,90):\n")
            f.write(f"    Success: {', '.join(f'{p:.3f}' for p in stats['success_percentiles'])}\n")
            f.write(f"    Failure: {', '.join(f'{p:.3f}' for p in stats['fail_percentiles'])}\n")

            # Add actual parameter values at these percentiles
            if 'success_actual_percentiles' in stats and 'fail_actual_percentiles' in stats:
                f.write(f"\n  ACTUAL PARAMETER VALUES (10,25,50,75,90):\n")
                f.write(f"    Success: {', '.join(f'{p:.4g}' for p in stats['success_actual_percentiles'])}\n")
                f.write(f"    Failure: {', '.join(f'{p:.4g}' for p in stats['fail_actual_percentiles'])}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n\n")
        f.write("- Quantile percentiles show the distribution in [0, 1] CDF space\n")
        f.write("- Actual parameter values show the corresponding real parameter values\n")
        f.write("- High error regions indicate parameter ranges prone to failures\n")
        f.write("- KS statistic measures distribution difference (0=identical, 1=completely different)\n")
        f.write("- p-values < 0.05 indicate statistically significant differences\n")
        f.write("- ***: p<0.001, **: p<0.01, *: p<0.05, n.s.: not significant\n\n")


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Analyzing error regions for: {run_dir}")
    print()

    # Load samples (both quantile and actual values)
    print("Loading samples...")
    successful, failed, successful_actuals, failed_actuals = load_quantile_samples(run_dir)

    print(f"  Successful samples: {len(successful)}")
    print(f"  Failed samples: {len(failed)}")
    print(f"  Error rate: {len(failed) / (len(successful) + len(failed)) * 100:.1f}%")
    print()

    if len(failed) == 0:
        print("No failed samples found. Nothing to analyze.")
        return

    # Analyze distributions
    print("Analyzing parameter distributions...")
    dist_analysis = analyze_parameter_distributions(successful, failed, successful_actuals, failed_actuals)
    print(f"  Analyzed {len(dist_analysis)} parameters")
    print()

    # Find high-error regions
    print(f"Finding high-error regions (bins={args.bins}, min_samples={args.min_samples})...")
    high_error_regions = find_high_error_regions(successful, failed, args.bins, args.min_samples)
    print(f"  Found {len(high_error_regions)} parameters with high-error regions")
    print()

    # Generate report
    report_path = run_dir / "error_region_analysis.txt"
    print(f"Generating report: {report_path}")
    generate_report(successful, failed, successful_actuals, failed_actuals,
                    dist_analysis, high_error_regions, report_path)
    print()

    # Generate plots if requested
    if args.plot:
        print("Generating visualizations...")

        # Error rate heatmap
        heatmap_path = run_dir / "error_rate_heatmap.png"
        print(f"  - Error rate heatmap: {heatmap_path}")
        plot_error_rate_heatmap(successful, failed, heatmap_path, args.bins, args.min_samples)

        # Individual parameter comparisons for high-error parameters
        if high_error_regions:
            comparison_dir = run_dir / "error_region_comparisons"
            comparison_dir.mkdir(exist_ok=True)

            print(f"  - Individual parameter comparisons in: {comparison_dir}/")
            for param_name in sorted(high_error_regions.keys())[:10]:  # Top 10
                plot_path = comparison_dir / f"{param_name}_comparison.png"
                plot_parameter_comparison(param_name, successful, failed, plot_path)

        print()

    print("Analysis complete!")
    print(f"\nView the full report at: {report_path}")


if __name__ == "__main__":
    main()




