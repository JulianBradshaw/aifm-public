#!/usr/bin/env python3
"""
Takeoff Exponent Histogram Plot.

Plots the distribution of 2^(1 - β/m) values across rollouts. This exponent
represents the asymptotic exponent in the progress growth rate. When m/β > 1,
the exponent > 1 indicating superexponential takeoff.

Usage:
    python -m plots.histograms.takeoff_exponent --run-dir outputs/251209_updated_params
"""

from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot
from plotting_utils.rollouts_reader import RolloutsReader


@register_plot(
    PlotSpec(
        name="takeoff_exponent_hist",
        display_name="Takeoff Exponent Distribution",
        category=PlotCategory.HISTOGRAM,
        speed=PlotSpeed.SLOW,
        output_filename="takeoff_exponent_hist.png",
        requires_cache=False,
        requires_full_jsonl=True,
        description="Distribution of 2^(1 - β/m) takeoff exponent across rollouts",
    )
)
class TakeoffExponentHistogramPlot(BasePlot):
    """Histogram of takeoff exponent 2^(1 - β/m) values."""

    def generate(self, **kwargs) -> Path:
        """Generate the takeoff exponent histogram plot.

        Args:
            **kwargs: Additional arguments (bins, etc.)

        Returns:
            Path to the generated plot
        """
        from plotting.histograms import plot_takeoff_exponent_histogram

        bins = kwargs.get("bins", 50)
        rollouts_file = self.run_dir / "rollouts.jsonl"

        # Read data using RolloutsReader
        reader = RolloutsReader(rollouts_file)
        takeoff_exponent_values, num_skipped = reader.read_takeoff_exponent()

        if len(takeoff_exponent_values) == 0:
            raise ValueError("No valid takeoff exponent values found in rollouts")

        # Get output path
        out_path = self.get_output_path(kwargs.get("out_path"))
        self.ensure_output_dir()

        # Generate plot using existing function
        plot_takeoff_exponent_histogram(
            takeoff_exponent_values,
            num_skipped,
            out_path,
            bins=bins,
            title="Distribution of Takeoff Exponent 2^(1 - β/m)",
        )

        # Print statistics
        arr = np.asarray(takeoff_exponent_values, dtype=float)
        pct_above_one = 100.0 * np.sum(arr > 1) / len(arr)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"  Percentage > 1 (superexponential): {pct_above_one:.1f}%")
        print(f"  P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f}")
        if num_skipped > 0:
            print(f"  Note: {num_skipped} rollouts skipped due to missing data")

        return out_path


if __name__ == "__main__":
    TakeoffExponentHistogramPlot.main()
