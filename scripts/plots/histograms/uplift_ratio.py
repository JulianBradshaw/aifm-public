#!/usr/bin/env python3
"""
Uplift Ratio Histogram Plot.

Plots the distribution of 2^(β/m - 1) values across rollouts. This ratio
represents the ratio of successive uplift doublings during software
intelligence explosion. When β/m < 1, this value < 1, indicating
superexponential growth.

Usage:
    python -m plots.histograms.uplift_ratio --run-dir outputs/251209_updated_params
"""

from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot
from plotting_utils.rollouts_reader import RolloutsReader


@register_plot(
    PlotSpec(
        name="uplift_ratio_hist",
        display_name="Uplift Ratio Distribution",
        category=PlotCategory.HISTOGRAM,
        speed=PlotSpeed.SLOW,
        output_filename="uplift_ratio_hist.png",
        requires_cache=False,
        requires_full_jsonl=True,
        description="Distribution of 2^(β/m - 1) uplift ratio across rollouts",
    )
)
class UpliftRatioHistogramPlot(BasePlot):
    """Histogram of uplift ratio 2^(β/m - 1) values."""

    def generate(self, **kwargs) -> Path:
        """Generate the uplift ratio histogram plot.

        Args:
            **kwargs: Additional arguments (bins, etc.)

        Returns:
            Path to the generated plot
        """
        from plotting.histograms import plot_uplift_ratio_histogram

        bins = kwargs.get("bins", 50)
        rollouts_file = self.run_dir / "rollouts.jsonl"

        # Read data using RolloutsReader
        reader = RolloutsReader(rollouts_file)
        uplift_ratio_values, num_skipped = reader.read_uplift_ratio()

        if len(uplift_ratio_values) == 0:
            raise ValueError("No valid uplift ratio values found in rollouts")

        # Get output path
        out_path = self.get_output_path(kwargs.get("out_path"))
        self.ensure_output_dir()

        # Generate plot using existing function (use default title)
        plot_uplift_ratio_histogram(
            uplift_ratio_values,
            num_skipped,
            out_path,
            bins=bins,
        )

        # Print statistics
        arr = np.asarray(uplift_ratio_values, dtype=float)
        pct_below_one = 100.0 * np.sum(arr < 1) / len(arr)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"  Percentage < 1 (superexponential): {pct_below_one:.1f}%")
        print(f"  P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f}")
        if num_skipped > 0:
            print(f"  Note: {num_skipped} rollouts skipped due to missing data")

        return out_path


if __name__ == "__main__":
    UpliftRatioHistogramPlot.main()
