#!/usr/bin/env python3
"""
Fit alternative distributions to transition duration data.

WARNING: This script contains bespoke data loading functions that don't follow
the project's plotting utility conventions. It should be refactored to use
utilities from scripts/plotting_utils/.

This script fits multiple distributions (Gamma, Weibull, Exponential, etc.)
to transition duration data and compares them to the lognormal fit.
Handles censored data (where the second milestone was not achieved).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()


def read_transition_data(
    rollouts_file: Path,
    milestone_a: str,
    milestone_b: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read transition duration data including censored observations.

    Returns:
        durations: Array of observed transition durations (both achieved)
        censored_durations: Array of censored durations (B not achieved, use sim_end - A)
        is_censored: Boolean array indicating which observations are censored
    """
    durations_list = []
    censored_list = []

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

            # Get simulation end time
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    simulation_end = float(times_array[-1])
                except Exception:
                    simulation_end = None
            else:
                simulation_end = None

            # Get milestone times
            m_a = milestones.get(milestone_a)
            m_b = milestones.get(milestone_b)

            ta = None
            if isinstance(m_a, dict) and m_a.get("time") is not None:
                try:
                    ta = float(m_a["time"])
                    if not np.isfinite(ta):
                        ta = None
                except Exception:
                    ta = None

            if ta is None:
                # If first milestone not achieved, skip
                continue

            tb = None
            if isinstance(m_b, dict) and m_b.get("time") is not None:
                try:
                    tb = float(m_b["time"])
                    if not np.isfinite(tb):
                        tb = None
                except Exception:
                    tb = None

            if tb is not None and tb > ta:
                # Both achieved
                durations_list.append(tb - ta)
            elif simulation_end is not None and simulation_end > ta:
                # B not achieved, censored observation
                censored_list.append(simulation_end - ta)

    durations = np.array(durations_list)
    censored_durations = np.array(censored_list)

    # Combine for full dataset with censoring indicator
    all_durations = np.concatenate([durations, censored_durations])
    is_censored = np.concatenate([
        np.zeros(len(durations), dtype=bool),
        np.ones(len(censored_durations), dtype=bool)
    ])

    return durations, censored_durations, is_censored


def fit_lognormal(data: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """Fit lognormal distribution using MLE.

    Returns:
        mu: Mean of log(X)
        sigma: Std of log(X)
        stats: Dictionary of goodness-of-fit statistics
    """
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    mu = np.log(scale)
    sigma = shape

    # Compute goodness-of-fit
    ks_stat, ks_pvalue = stats.kstest(
        data, lambda x: stats.lognorm.cdf(x, shape, loc=0, scale=scale)
    )

    # Anderson-Darling on log-transformed data
    log_data = np.log(data)
    ad_result = stats.anderson(log_data, dist='norm')
    ad_stat = ad_result.statistic

    # R-squared
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.lognorm.ppf(
        np.linspace(0.01, 0.99, len(sorted_data)),
        shape, loc=0, scale=scale
    )
    empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
    r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                    np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

    # AIC/BIC
    log_likelihood = np.sum(stats.lognorm.logpdf(data, shape, loc=0, scale=scale))
    n = len(data)
    k = 2  # number of parameters
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return mu, sigma, {
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'ad_stat': ad_stat,
        'r_squared': r_squared,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'median': np.exp(mu),
        'mean': np.exp(mu + sigma**2 / 2)
    }


def fit_gamma(data: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """Fit Gamma distribution using MLE.

    Returns:
        shape: Shape parameter (k)
        scale: Scale parameter (theta)
        stats: Dictionary of goodness-of-fit statistics
    """
    shape, loc, scale = stats.gamma.fit(data, floc=0)

    # Compute goodness-of-fit
    ks_stat, ks_pvalue = stats.kstest(
        data, lambda x: stats.gamma.cdf(x, shape, loc=0, scale=scale)
    )

    # R-squared
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.gamma.ppf(
        np.linspace(0.01, 0.99, len(sorted_data)),
        shape, loc=0, scale=scale
    )
    empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
    r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                    np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

    # AIC/BIC
    log_likelihood = np.sum(stats.gamma.logpdf(data, shape, loc=0, scale=scale))
    n = len(data)
    k = 2
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return shape, scale, {
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'r_squared': r_squared,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'median': stats.gamma.median(shape, loc=0, scale=scale),
        'mean': shape * scale
    }


def fit_weibull(data: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """Fit Weibull distribution using MLE.

    Returns:
        shape: Shape parameter (k)
        scale: Scale parameter (lambda)
        stats: Dictionary of goodness-of-fit statistics
    """
    shape, loc, scale = stats.weibull_min.fit(data, floc=0)

    # Compute goodness-of-fit
    ks_stat, ks_pvalue = stats.kstest(
        data, lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale)
    )

    # R-squared
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.weibull_min.ppf(
        np.linspace(0.01, 0.99, len(sorted_data)),
        shape, loc=0, scale=scale
    )
    empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
    r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                    np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

    # AIC/BIC
    log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, loc=0, scale=scale))
    n = len(data)
    k = 2
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return shape, scale, {
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'r_squared': r_squared,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'median': scale * (np.log(2) ** (1/shape)),
        'mean': scale * gamma_func(1 + 1/shape)
    }


def fit_exponential(data: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Fit Exponential distribution using MLE.

    Returns:
        scale: Scale parameter (1/lambda)
        stats: Dictionary of goodness-of-fit statistics
    """
    loc, scale = stats.expon.fit(data, floc=0)

    # Compute goodness-of-fit
    ks_stat, ks_pvalue = stats.kstest(
        data, lambda x: stats.expon.cdf(x, loc=0, scale=scale)
    )

    # R-squared
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.expon.ppf(
        np.linspace(0.01, 0.99, len(sorted_data)),
        loc=0, scale=scale
    )
    empirical_quantiles = np.percentile(data, np.linspace(1, 99, len(sorted_data)))
    r_squared = 1 - np.sum((empirical_quantiles - theoretical_quantiles)**2) / \
                    np.sum((empirical_quantiles - empirical_quantiles.mean())**2)

    # AIC/BIC
    log_likelihood = np.sum(stats.expon.logpdf(data, loc=0, scale=scale))
    n = len(data)
    k = 1
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return scale, {
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'r_squared': r_squared,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'median': scale * np.log(2),
        'mean': scale
    }


def create_comparison_plot(
    data: np.ndarray,
    censored_data: np.ndarray,
    milestone_a: str,
    milestone_b: str,
    output_path: Path
) -> None:
    """Create comparison plot of different distribution fits.

    Args:
        data: Observed (uncensored) transition durations
        censored_data: Censored transition durations
        milestone_a: First milestone name
        milestone_b: Second milestone name
        output_path: Path to save plot
    """
    # Fit all distributions
    print(f"\nFitting distributions to {len(data)} uncensored observations...")
    print(f"  ({len(censored_data)} censored observations)")

    lognorm_mu, lognorm_sigma, lognorm_stats = fit_lognormal(data)
    gamma_shape, gamma_scale, gamma_stats = fit_gamma(data)
    weibull_shape, weibull_scale, weibull_stats = fit_weibull(data)
    expon_scale, expon_stats = fit_exponential(data)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    distributions = [
        ('Lognormal', lognorm_stats,
         lambda x: stats.lognorm.pdf(x, lognorm_sigma, loc=0, scale=np.exp(lognorm_mu)),
         f'μ={lognorm_mu:.3f}, σ={lognorm_sigma:.3f}'),
        ('Gamma', gamma_stats,
         lambda x: stats.gamma.pdf(x, gamma_shape, loc=0, scale=gamma_scale),
         f'k={gamma_shape:.3f}, θ={gamma_scale:.3f}'),
        ('Weibull', weibull_stats,
         lambda x: stats.weibull_min.pdf(x, weibull_shape, loc=0, scale=weibull_scale),
         f'k={weibull_shape:.3f}, λ={weibull_scale:.3f}'),
        ('Exponential', expon_stats,
         lambda x: stats.expon.pdf(x, loc=0, scale=expon_scale),
         f'λ={1/expon_scale:.3f}')
    ]

    x_min = min(data.min(), censored_data.min() if len(censored_data) > 0 else data.min())
    x_max = max(data.max(), censored_data.max() if len(censored_data) > 0 else data.max())
    x = np.linspace(x_min, x_max, 500)

    for idx, (name, dist_stats, pdf_func, params_str) in enumerate(distributions):
        ax = axes[idx]

        # Create histogram with both censored and uncensored data
        # Show uncensored in blue, censored in lighter blue
        n_bins = 30
        ax.hist(data, bins=n_bins, density=True,
                alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.5,
                label=f'Uncensored (n={len(data)})')

        if len(censored_data) > 0:
            ax.hist(censored_data, bins=n_bins, density=True,
                    alpha=0.4, color='lightsteelblue',
                    edgecolor='black', linewidth=0.5,
                    label=f'Censored (n={len(censored_data)})')

        # Plot fitted PDF (only uses uncensored data)
        pdf = pdf_func(x)
        ax.plot(x, pdf, 'r-', linewidth=2.5, label=f'{name} fit')

        # Add median line
        median_obs = np.median(data)
        median_fit = dist_stats['median']
        ax.axvline(median_obs, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Obs. median: {median_obs:.2f}')
        ax.axvline(median_fit, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Fit median: {median_fit:.2f}')

        ax.set_xlabel('Transition duration (years)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = (
            f"{name} Fit:\n"
            f"\n"
            f"Parameters:\n"
            f"  {params_str}\n"
            f"\n"
            f"Goodness-of-fit:\n"
            f"  R² = {dist_stats['r_squared']:.4f}\n"
            f"  KS stat = {dist_stats['ks_stat']:.4f}\n"
            f"  KS p = {dist_stats['ks_pvalue']:.4f}\n"
            f"\n"
            f"Model selection:\n"
            f"  AIC = {dist_stats['aic']:.1f}\n"
            f"  BIC = {dist_stats['bic']:.1f}\n"
            f"  LogLik = {dist_stats['log_likelihood']:.1f}\n"
        )

        quality = 'Excellent' if dist_stats['r_squared'] > 0.95 else \
                  'Good' if dist_stats['r_squared'] > 0.90 else \
                  'Fair' if dist_stats['r_squared'] > 0.80 else 'Poor'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='lightyellow', alpha=0.9,
                         edgecolor='gray', linewidth=1),
                family='monospace')

    # Add overall comparison
    fig.suptitle(
        f'Distribution Fit Comparison: {milestone_a} → {milestone_b}\n'
        f'Transition Duration Analysis (Censored data shown in light blue)',
        fontsize=15, fontweight='bold', y=0.98
    )

    # Create comparison table
    comparison_text = (
        "MODEL COMPARISON (lower AIC/BIC is better):\n"
        "\n"
        f"Distribution     R²      KS-p    AIC      BIC      Quality\n"
        f"{'='*65}\n"
        f"Lognormal     {lognorm_stats['r_squared']:6.4f}  {lognorm_stats['ks_pvalue']:6.4f}  "
        f"{lognorm_stats['aic']:7.1f}  {lognorm_stats['bic']:7.1f}  "
        f"{'Excellent' if lognorm_stats['r_squared'] > 0.95 else 'Good' if lognorm_stats['r_squared'] > 0.90 else 'Fair' if lognorm_stats['r_squared'] > 0.80 else 'Poor'}\n"
        f"Gamma         {gamma_stats['r_squared']:6.4f}  {gamma_stats['ks_pvalue']:6.4f}  "
        f"{gamma_stats['aic']:7.1f}  {gamma_stats['bic']:7.1f}  "
        f"{'Excellent' if gamma_stats['r_squared'] > 0.95 else 'Good' if gamma_stats['r_squared'] > 0.90 else 'Fair' if gamma_stats['r_squared'] > 0.80 else 'Poor'}\n"
        f"Weibull       {weibull_stats['r_squared']:6.4f}  {weibull_stats['ks_pvalue']:6.4f}  "
        f"{weibull_stats['aic']:7.1f}  {weibull_stats['bic']:7.1f}  "
        f"{'Excellent' if weibull_stats['r_squared'] > 0.95 else 'Good' if weibull_stats['r_squared'] > 0.90 else 'Fair' if weibull_stats['r_squared'] > 0.80 else 'Poor'}\n"
        f"Exponential   {expon_stats['r_squared']:6.4f}  {expon_stats['ks_pvalue']:6.4f}  "
        f"{expon_stats['aic']:7.1f}  {expon_stats['bic']:7.1f}  "
        f"{'Excellent' if expon_stats['r_squared'] > 0.95 else 'Good' if expon_stats['r_squared'] > 0.90 else 'Fair' if expon_stats['r_squared'] > 0.80 else 'Poor'}\n"
        f"\n"
        f"Best by AIC: {min([('Lognormal', lognorm_stats['aic']), ('Gamma', gamma_stats['aic']), ('Weibull', weibull_stats['aic']), ('Exponential', expon_stats['aic'])], key=lambda x: x[1])[0]}\n"
        f"Best by BIC: {min([('Lognormal', lognorm_stats['bic']), ('Gamma', gamma_stats['bic']), ('Weibull', weibull_stats['bic']), ('Exponential', expon_stats['bic'])], key=lambda x: x[1])[0]}\n"
        f"Best by R²:  {max([('Lognormal', lognorm_stats['r_squared']), ('Gamma', gamma_stats['r_squared']), ('Weibull', weibull_stats['r_squared']), ('Exponential', expon_stats['r_squared'])], key=lambda x: x[1])[0]}\n"
    )

    fig.text(0.5, 0.01, comparison_text,
             ha='center', va='bottom', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='blue', linewidth=2),
             family='monospace')

    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison plot to: {output_path}")

    # Print summary to console
    print("\n" + "="*70)
    print("DISTRIBUTION FIT SUMMARY")
    print("="*70)
    print(comparison_text)
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Fit alternative distributions to transition duration data"
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        required=True,
        help="Path to rollouts.jsonl file"
    )
    parser.add_argument(
        "--milestone-a",
        type=str,
        default="AC",
        help="First milestone name (default: AC)"
    )
    parser.add_argument(
        "--milestone-b",
        type=str,
        default="SAR-level-experiment-selection-skill",
        help="Second milestone name (default: SAR-level-experiment-selection-skill)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for plot (default: <rollouts_dir>/distribution_comparison.png)"
    )

    args = parser.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        print(f"Error: Rollouts file not found: {rollouts_path}")
        sys.exit(1)

    # Read data
    print(f"Reading transition data for {args.milestone_a} → {args.milestone_b}...")
    durations, censored_durations, is_censored = read_transition_data(
        rollouts_path,
        args.milestone_a,
        args.milestone_b
    )

    if len(durations) == 0:
        print("Error: No transition data found")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = rollouts_path.parent / "distribution_comparison.png"

    # Create comparison plot
    create_comparison_plot(
        durations,
        censored_durations,
        args.milestone_a,
        args.milestone_b,
        output_path
    )


if __name__ == "__main__":
    main()
