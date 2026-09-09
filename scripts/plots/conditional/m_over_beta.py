#!/usr/bin/env python3
"""
m/β Conditional Histogram Plot.

Plots the distribution of m/β values conditional on SC (Superhuman Coder)
being achieved in a specific year.

Usage:
    python -m plots.conditional.m_over_beta --run-dir outputs/251209_updated_params --year 2027
    python -m plots.conditional.m_over_beta --run-dir outputs/251209_updated_params --year 2028
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot
from plotting_utils.rollouts_reader import RolloutsReader


@register_plot(
    PlotSpec(
        name="m_over_beta_conditional",
        display_name="m/β Conditional Distribution",
        category=PlotCategory.CONDITIONAL,
        speed=PlotSpeed.SLOW,
        output_filename="m_over_beta_conditional.png",
        requires_cache=False,
        requires_full_jsonl=True,
        description="Distribution of m/β conditional on SC in a specific year",
    )
)
class MOverBetaConditionalPlot(BasePlot):
    """Histogram of m/β values conditional on SC achievement year."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add conditional-specific arguments."""
        parser.add_argument(
            "--year",
            type=int,
            default=2027,
            help="Year to filter SC achievement (default: 2027)",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the conditional m/β histogram plot.

        Args:
            **kwargs: Arguments including year, bins, etc.

        Returns:
            Path to the generated plot
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from plotting_utils.kde import make_gaussian_kde
        from plotting_utils.style import apply_style

        apply_style()

        filter_year = kwargs.get("year", 2027)
        bins = kwargs.get("bins", 50)
        rollouts_file = self.run_dir / "rollouts.jsonl"

        # Read data using RolloutsReader
        reader = RolloutsReader(rollouts_file)
        (
            m_over_beta_values,
            _,  # takeoff_exponent_values - not used
            _,  # uplift_ratio_values - not used
            num_skipped,
            total_sc_in_year,
            total_rollouts,
        ) = reader.read_takeoff_metrics_conditional("AI2027-SC", filter_year)

        print(f"Total rollouts: {total_rollouts}")
        print(f"Rollouts with SC in {filter_year}: {total_sc_in_year}")
        print(f"Rollouts with valid m/β: {len(m_over_beta_values)}")
        print(f"Rollouts skipped (no m/β data): {num_skipped}")

        if len(m_over_beta_values) == 0:
            raise ValueError(
                f"No m/β values found for rollouts with SC in {filter_year}"
            )

        data = np.asarray(m_over_beta_values, dtype=float)
        data = data[data > 0]  # Filter to positive for log scale
        if len(data) == 0:
            raise ValueError("No positive m/β values found to plot")

        # Log-spaced bins
        log_data = np.log10(data)
        log_bins = np.linspace(log_data.min(), log_data.max(), bins + 1)
        bin_edges_linear = 10**log_bins

        plt.figure(figsize=(10, 6))
        counts, bin_edges, _ = plt.hist(
            data, bins=bin_edges_linear, edgecolor="black", alpha=0.6, label="Histogram"
        )
        plt.xscale("log")

        # Add KDE on log-transformed data
        if len(data) >= 2:
            xs_log = np.linspace(log_data.min(), log_data.max(), 512)
            xs = 10**xs_log
            try:
                kde = make_gaussian_kde(log_data)
                kde_vals = kde(xs_log)
                bin_widths = np.diff(bin_edges_linear)
                hist_area = np.sum(counts * bin_widths)
                kde_area = np.trapezoid(kde_vals, xs)
                kde_counts = (
                    kde_vals * hist_area / kde_area if kde_area > 0 else kde_vals
                )
                plt.plot(
                    xs,
                    kde_counts,
                    color="tab:orange",
                    linewidth=2.25,
                    label="Gaussian KDE (log)",
                )
            except Exception:
                pass

        # Median line
        q50 = np.quantile(data, 0.5)
        ymax = float(np.max(counts) if counts.size else 1.0)
        y_annot = ymax * 0.95

        plt.axvline(
            q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median"
        )
        plt.text(
            q50,
            y_annot,
            f"Median: {q50:.3f}",
            rotation=90,
            va="top",
            ha="right",
            color="tab:green",
            fontsize=9,
            backgroundcolor=(1, 1, 1, 0.6),
        )

        # Vertical line for m/β = 1.0
        pct_at_1 = np.mean(data <= 1.0) * 100
        plt.axvline(
            1.0,
            color="tab:red",
            linestyle=":",
            linewidth=2,
            label=f"P{round(pct_at_1)}: 1.0",
        )
        plt.text(
            1.0,
            y_annot * 0.7,
            f"P{round(pct_at_1)}: 1.0",
            rotation=90,
            va="top",
            ha="right",
            color="tab:red",
            fontsize=9,
            backgroundcolor=(1, 1, 1, 0.6),
        )

        plt.xlabel("m/β ratio")
        plt.ylabel("Count")
        plt.title(
            f"Distribution of m/β | SC achieved in {filter_year} "
            f"(n={len(data)}, skipped={num_skipped})"
        )
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(loc="upper right")

        # Determine output path
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / f"m_over_beta_conditional_{filter_year}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path


if __name__ == "__main__":
    MOverBetaConditionalPlot.main()
