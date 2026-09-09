#!/usr/bin/env python3
"""
Uplift Ratio Conditional Histogram Plot.

Plots the distribution of 2^(β/m - 1) (ratio of successive uplift doublings)
conditional on SC being achieved in a specific year.

Usage:
    python -m plots.conditional.uplift_ratio --run-dir outputs/251209_updated_params --year 2027
    python -m plots.conditional.uplift_ratio --run-dir outputs/251209_updated_params --year 2028 --max-x 5.0
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot
from plotting_utils.rollouts_reader import RolloutsReader


@register_plot(
    PlotSpec(
        name="uplift_ratio_conditional",
        display_name="Uplift Ratio (Conditional)",
        category=PlotCategory.CONDITIONAL,
        speed=PlotSpeed.SLOW,
        output_filename="uplift_ratio_conditional.png",
        output_subdir="fast_takeoff_outputs",
        requires_cache=False,
        requires_full_jsonl=True,
        description="Distribution of uplift ratio conditional on SC achievement year",
    )
)
class UpliftRatioConditionalPlot(BasePlot):
    """Histogram of uplift ratio 2^(β/m - 1) conditional on SC achievement year."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add conditional-specific arguments."""
        parser.add_argument(
            "--year",
            type=int,
            default=2027,
            help="Year to filter SC achievement (default: 2027)",
        )
        parser.add_argument(
            "--max-x",
            type=float,
            default=10.0,
            help="Maximum x value to display (default: 10.0)",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the conditional uplift ratio histogram plot.

        Args:
            **kwargs: Arguments including year, bins, max_x, etc.

        Returns:
            Path to the generated plot
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = "monospace"

        filter_year = kwargs.get("year", 2027)
        bins = kwargs.get("bins", 50)
        max_x = kwargs.get("max_x", 10.0)
        rollouts_file = self.run_dir / "rollouts.jsonl"

        # Read data using RolloutsReader
        reader = RolloutsReader(rollouts_file)
        (
            _,  # m_over_beta_values - not used
            _,  # takeoff_exponent_values - not used
            uplift_ratio_values,
            num_skipped,
            total_sc_in_year,
            total_rollouts,
        ) = reader.read_takeoff_metrics_conditional("AI2027-SC", filter_year)

        print(f"Total rollouts: {total_rollouts}")
        print(f"Rollouts with SC in {filter_year}: {total_sc_in_year}")
        print(f"Rollouts with valid uplift ratio: {len(uplift_ratio_values)}")
        print(f"Rollouts skipped (no data): {num_skipped}")

        if len(uplift_ratio_values) == 0:
            raise ValueError(
                f"No uplift ratio values found for rollouts with SC in {filter_year}"
            )

        data_all = np.asarray(uplift_ratio_values, dtype=float)
        data_all = data_all[data_all > 0]
        if len(data_all) == 0:
            raise ValueError("No positive uplift ratio values found to plot")

        # Count values above max_x
        num_above_max = int(np.sum(data_all > max_x))
        pct_above_max = 100.0 * num_above_max / len(data_all)

        # Filter data for plotting
        data = data_all[data_all <= max_x]

        # Log-spaced bins
        log_data = np.log10(data)
        log_bins = np.linspace(log_data.min(), np.log10(max_x), bins + 1)
        bin_edges_linear = 10**log_bins

        plt.figure(figsize=(10, 6))
        counts, bin_edges, _ = plt.hist(
            data, bins=bin_edges_linear, edgecolor="black", alpha=0.6, label="Histogram"
        )
        plt.xscale("log")
        plt.xlim(right=max_x)

        # Percentile lines (using all data for accurate percentiles)
        q10, q50, q90 = np.quantile(data_all, [0.1, 0.5, 0.9])
        ymax = float(np.max(counts) if counts.size else 1.0)
        y_annot = ymax * 0.95

        plt.axvline(
            q10,
            color="tab:gray",
            linestyle="--",
            linewidth=1.5,
            label=f"P10: {q10:.2f}",
        )
        plt.axvline(
            q50,
            color="tab:green",
            linestyle="-",
            linewidth=1.75,
            label=f"Median: {q50:.2f}",
        )
        # Only show P90 line if within visible range
        if q90 <= max_x:
            plt.axvline(
                q90,
                color="tab:gray",
                linestyle="--",
                linewidth=1.5,
                label=f"P90: {q90:.2f}",
            )
        else:
            plt.plot(
                [],
                [],
                color="tab:gray",
                linestyle="--",
                linewidth=1.5,
                label=f"P90: {q90:.2f} (off chart)",
            )

        plt.text(
            q10,
            y_annot,
            f"P10: {q10:.2f}",
            rotation=90,
            va="top",
            ha="right",
            color="tab:gray",
            fontsize=9,
            backgroundcolor=(1, 1, 1, 0.6),
        )
        plt.text(
            q50,
            y_annot,
            f"Median: {q50:.2f}",
            rotation=90,
            va="top",
            ha="right",
            color="tab:green",
            fontsize=9,
            backgroundcolor=(1, 1, 1, 0.6),
        )
        if q90 <= max_x:
            plt.text(
                q90,
                y_annot,
                f"P90: {q90:.2f}",
                rotation=90,
                va="top",
                ha="right",
                color="tab:gray",
                fontsize=9,
                backgroundcolor=(1, 1, 1, 0.6),
            )

        # Vertical line at 1.0 (beta/m = 1)
        pct_at_1 = np.mean(data_all <= 1.0) * 100
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

        # Vertical line at 0.7
        pct_at_07 = np.mean(data_all <= 0.7) * 100
        plt.axvline(
            0.7,
            color="tab:purple",
            linestyle=":",
            linewidth=2,
            label=f"P{round(pct_at_07)}: 0.7",
        )
        plt.text(
            0.7,
            y_annot * 0.7,
            f"P{round(pct_at_07)}: 0.7",
            rotation=90,
            va="top",
            ha="right",
            color="tab:purple",
            fontsize=9,
            backgroundcolor=(1, 1, 1, 0.6),
        )

        # Legend entry for values above max_x
        if num_above_max > 0:
            plt.plot([], [], " ", label=f"{pct_above_max:.1f}% > {max_x} (not shown)")

        plt.xlabel("Ratio of successive uplift doublings")
        plt.ylabel("Count")
        plt.title(f"Ratio of successive uplift doublings | SC in {filter_year}")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(loc="upper right")

        # Determine output path
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            fast_takeoff_dir = self.run_dir / "fast_takeoff_outputs"
            fast_takeoff_dir.mkdir(parents=True, exist_ok=True)
            out_path = fast_takeoff_dir / f"uplift_ratio_conditional_{filter_year}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path


if __name__ == "__main__":
    UpliftRatioConditionalPlot.main()
