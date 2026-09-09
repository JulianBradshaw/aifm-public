#!/usr/bin/env python3
"""
Visualize near-present milestone arrival density to diagnose boundary spikes.

This script focuses on a narrow time window (default 2017–2027) and plots a
fine-grained histogram of milestone arrival samples along with the
lower-bounded KDE we use elsewhere. It helps confirm whether the KDE spike
around `present_day` is driven by genuine samples clustered just after today.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

# Non-interactive backend for automated runs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_utils.helpers import load_present_day
from plotting_utils.kde import (
    make_gamma_kernel_kde,
    make_lower_bounded_kde,
)
from plotting_utils.rollouts_reader import RolloutsReader

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect near-present milestone arrival densities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--rollouts",
        type=str,
        help="Path to rollouts.jsonl file",
    )
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (expects rollouts.jsonl inside)",
    )
    parser.add_argument(
        "--milestones",
        type=str,
        nargs="+",
        default=["AC", "SAR-level-experiment-selection-skill"],
        help="Milestone names to plot",
    )
    parser.add_argument(
        "--year-min",
        type=float,
        default=2017.0,
        help="Lower bound of the diagnostic window",
    )
    parser.add_argument(
        "--year-max",
        type=float,
        default=2027.0,
        help="Upper bound of the diagnostic window",
    )
    parser.add_argument(
        "--bins-per-year",
        type=int,
        default=40,
        help="Histogram resolution (bins per calendar year)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path (PNG). Defaults to <run-dir>/present_day_spike.png",
    )
    parser.add_argument(
        "--present-day",
        type=float,
        default=None,
        help="Override the present_day boundary (defaults to value recorded in run dir)",
    )
    parser.add_argument(
        "--kde-method",
        type=str,
        choices=["log", "gamma"],
        default="gamma",
        help=(
            "Choose between the gamma kernel KDE ('gamma', default) and the log-transform KDE ('log')."
        ),
    )
    parser.add_argument(
        "--report-scores",
        action="store_true",
        help="Compute LOO/K-fold mean log predictive density (and CRPS) for the selected KDE.",
    )
    parser.add_argument(
        "--score-folds",
        type=int,
        default=None,
        help="Optional K-fold setting for scoring (defaults to leave-one-out when omitted).",
    )
    parser.add_argument(
        "--score-grid-points",
        type=int,
        default=400,
        help="Grid resolution for CRPS integration when reporting scores.",
    )
    parser.add_argument(
        "--score-tail-years",
        type=float,
        default=5.0,
        help="Extend the CRPS grid this many years beyond the max training point.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Log timing information for expensive sections (diagnostic only).",
    )
    return parser.parse_args()


def _load_times(
    rollouts_path: Path,
    milestones: List[str],
    profiler: "Profiler" | None = None,
) -> Dict[str, List[float]]:
    reader = RolloutsReader(rollouts_path)
    data: Dict[str, List[float]] = {}
    prof = profiler or Profiler(False)
    with prof.section("load milestone times"):
        times_map, not_achieved_map, _, total_rollouts = reader.read_milestone_times_batch(milestones)

    print(f"\nTotal rollouts: {total_rollouts}")
    for milestone in milestones:
        times = times_map.get(milestone, [])
        num_not_achieved = not_achieved_map.get(milestone, 0)
        achieved = len(times)
        denom = achieved + num_not_achieved
        pct = (achieved / denom * 100) if denom else 0.0
        print(f"  {milestone}: {achieved} achieved ({pct:.1f}%)")
        data[milestone] = times
    return data


def _build_bins(year_min: float, year_max: float, bins_per_year: int) -> np.ndarray:
    total_years = max(year_max - year_min, 1e-6)
    num_bins = max(int(np.ceil(total_years * bins_per_year)), 10)
    return np.linspace(year_min, year_max, num_bins + 1)


@dataclass(frozen=True)
class ScoreOptions:
    folds: int | None
    grid_points: int
    tail_padding: float


class Profiler:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    @contextmanager
    def section(self, label: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            print(f"[profile] {label}: {duration:.3f}s")

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[profile] {message}")


def _fit_kde(
    data: Sequence[float],
    method: str,
    lower_bound: float,
    *,
    bandwidth_hint: float | None = None,
):
    if method == "log":
        return make_lower_bounded_kde(
            data,
            lower_bound=lower_bound,
            bandwidth=bandwidth_hint,
        )
    if method == "gamma":
        return make_gamma_kernel_kde(
            data,
            lower_bound=lower_bound,
            bandwidth=bandwidth_hint,
        )
    raise ValueError(f"Unknown kde method: {method}")


def _extract_bandwidth_hint(kde, method: str) -> float | None:
    try:
        if method == "log":
            estimator = getattr(kde, "estimator", None)
            value = getattr(estimator, "bandwidth", None)
            if value is not None and np.isfinite(value) and value > 0:
                return float(value)
        elif method == "gamma":
            value = getattr(kde, "bandwidth", None)
            if value is not None and np.isfinite(value) and value > 0:
                return float(value)
    except Exception:
        return None
    return None


def _make_score_grid(
    train_data: np.ndarray,
    holdout: np.ndarray,
    lower_bound: float,
    *,
    num_points: int,
    tail_padding: float,
) -> np.ndarray:
    max_train = np.max(train_data) if train_data.size else lower_bound
    max_holdout = np.max(holdout) if holdout.size else lower_bound
    upper = max(max_train, max_holdout)
    span = max(upper - lower_bound, 1.0)
    upper += max(tail_padding, 0.1 * span)
    upper = max(upper, lower_bound + span)
    return np.linspace(lower_bound, upper, num_points)


def _cdf_from_estimator(
    estimator, grid: np.ndarray
) -> np.ndarray | None:
    pdf = estimator.pdf(grid)
    if pdf.ndim != 1:
        pdf = np.asarray(pdf).ravel()
    pdf = np.clip(pdf, 0.0, None)
    if not np.any(pdf):
        return None
    delta = np.diff(grid)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * delta))
    )
    total = cumulative[-1]
    if total <= 0:
        return None
    cdf = np.clip(cumulative / total, 0.0, 1.0)
    return cdf


def _crps_from_cdf(grid: np.ndarray, cdf: np.ndarray, value: float) -> float:
    indicator = (grid >= value).astype(float)
    integrand = (cdf - indicator) ** 2
    return float(np.trapezoid(integrand, x=grid))


def _compute_scores(
    data: np.ndarray,
    *,
    lower_bound: float,
    builder: Callable[[np.ndarray], object],
    options: ScoreOptions,
    profiler: "Profiler" | None = None,
) -> Dict[str, float] | None:
    if options.grid_points < 2:
        raise ValueError("score_grid_points must be at least 2.")
    if options.tail_padding <= 0:
        raise ValueError("score_tail_years must be positive.")
    samples = np.asarray(data, dtype=float)
    samples = samples[samples > lower_bound]
    n = samples.size
    if n < 3:
        return None

    prof = profiler or Profiler(False)

    if options.folds is not None and options.folds < 2:
        raise ValueError("score-folds must be at least 2.")

    from sklearn.model_selection import KFold, LeaveOneOut

    if options.folds is None or options.folds >= n:
        splitter = LeaveOneOut()
        fold_label = "LOO"
    else:
        splitter = KFold(n_splits=options.folds, shuffle=True, random_state=0)
        fold_label = f"{options.folds}-fold"

    log_scores: List[float] = []
    crps_scores: List[float] = []

    fold_desc = "LOO" if isinstance(splitter, LeaveOneOut) else f"{options.folds}-fold"
    with prof.section(f"score computation ({fold_desc}, n={n})"):
        for train_idx, test_idx in splitter.split(samples):
            train = samples[train_idx]
            test = samples[test_idx]
            if train.size < 2:
                continue
            estimator = builder(train)
            log_vals = estimator.logpdf(test)
            finite_mask = np.isfinite(log_vals)
            log_scores.extend(log_vals[finite_mask])

            grid = _make_score_grid(
                train,
                test,
                lower_bound,
                num_points=options.grid_points,
                tail_padding=options.tail_padding,
            )
            cdf = _cdf_from_estimator(estimator, grid)
            if cdf is None:
                continue
            for value in test:
                crps_scores.append(_crps_from_cdf(grid, cdf, value))

    if not log_scores:
        return None

    mlpd = float(np.mean(log_scores))
    crps = float(np.mean(crps_scores)) if crps_scores else float("nan")
    return {
        "mlpd": mlpd,
        "crps": crps,
        "fold_label": fold_label,
        "n": n,
    }


def plot_near_present_density(
    milestone_data: Dict[str, List[float]],
    out_path: Path,
    *,
    year_min: float,
    year_max: float,
    bins_per_year: int,
    present_day: float,
    kde_method: str,
    score_options: ScoreOptions | None = None,
    profiler: "Profiler" | None = None,
) -> None:
    bins = _build_bins(year_min, year_max, bins_per_year)
    colors = {
        "AC": "tab:blue",
        "SAR-level-experiment-selection-skill": "tab:green",
        "AI2027-SC": "tab:orange",
    }

    prof = profiler or Profiler(False)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax_density = ax.twinx()
    eps = 1e-6
    support_start = max(year_min, present_day + eps)

    for milestone, times in milestone_data.items():
        arr = np.asarray(times, dtype=float)
        window_mask = (arr >= year_min) & (arr <= year_max)
        window_data = arr[window_mask]
        if window_data.size == 0:
            print(f"Skipping {milestone}: no samples within [{year_min}, {year_max}].")
            continue

        color = colors.get(milestone, None)
        with prof.section(f"{milestone}: histogram"):
            hist = ax.hist(
                window_data,
                bins=bins,
                density=False,
                alpha=0.35,
                color=color,
                label=f"{milestone} histogram",
            )
        print(
            f"{milestone}: {window_data.size} samples in window; "
            f"bin width ≈ {(bins[1]-bins[0]):.3f} years"
        )

        # Use ALL data >= present_day for KDE (not just window), matching plot_ac_sc_sar_pdfs.py
        kde_data = arr[arr >= present_day]
        if kde_data.size < 2:
            print(f"  Not enough samples after present_day for KDE (n={kde_data.size}).")
            continue

        try:
            with prof.section(f"{milestone}: fit KDE"):
                kde = _fit_kde(kde_data, kde_method, present_day)
            bandwidth_hint = _extract_bandwidth_hint(kde, kde_method)
            xs = np.linspace(support_start, year_max, 2000)
            with prof.section(f"{milestone}: evaluate KDE"):
                ax_density.plot(xs, kde(xs), color=color, linewidth=2.5, label=f"{milestone} KDE")
            print(f"  KDE trained on {kde_data.size} samples (full range >= present_day)")

            if score_options is not None:
                def _builder(train: np.ndarray):
                    return _fit_kde(
                        train,
                        kde_method,
                        present_day,
                        bandwidth_hint=bandwidth_hint,
                    )

                with prof.section(f"{milestone}: scoring"):
                    scores = _compute_scores(
                        kde_data,
                        lower_bound=present_day,
                        builder=_builder,
                        options=score_options,
                        profiler=prof,
                    )
                if scores:
                    crps_val = scores["crps"]
                    crps_text = (
                        f"{crps_val:.4f}" if np.isfinite(crps_val) else "nan"
                    )
                    print(
                        f"  {scores['fold_label']} scores (n={scores['n']}): "
                        f"MLPD={scores['mlpd']:.4f}, CRPS={crps_text}"
                    )
        except Exception as exc:
            print(f"  Failed to build KDE for {milestone}: {exc}")

    ax.axvline(present_day, color="red", linestyle="--", linewidth=1.5, label="present_day")
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(bottom=0)
    ax_density.set_ylim(bottom=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax_density.set_ylabel("Probability Density")
    ax.set_title("Near-present milestone arrival density")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_density.get_legend_handles_labels()
    ax_density.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with prof.section("render & save figure"):
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"\nSaved diagnostic plot to {out_path}")


def main() -> None:
    args = _parse_args()
    profiler = Profiler(args.profile_timings)
    if profiler.enabled:
        profiler.log("Profiling enabled")
    if args.rollouts:
        rollouts_path = Path(args.rollouts)
        run_dir = rollouts_path.parent
    else:
        run_dir = Path(args.run_dir)
        rollouts_path = run_dir / "rollouts.jsonl"

    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    present_day = (
        float(args.present_day)
        if args.present_day is not None
        else load_present_day(run_dir)
    )
    print(f"Using present_day = {present_day:.3f}")

    milestone_data = _load_times(rollouts_path, args.milestones, profiler=profiler)
    out_path = Path(args.output) if args.output else run_dir / "present_day_spike.png"

    score_options = None
    if args.report_scores:
        score_options = ScoreOptions(
            folds=args.score_folds,
            grid_points=args.score_grid_points,
            tail_padding=args.score_tail_years,
        )

    plot_near_present_density(
        milestone_data,
        out_path,
        year_min=args.year_min,
        year_max=args.year_max,
        bins_per_year=args.bins_per_year,
        present_day=present_day,
        kde_method=args.kde_method,
        score_options=score_options,
        profiler=profiler,
    )


if __name__ == "__main__":
    main()
