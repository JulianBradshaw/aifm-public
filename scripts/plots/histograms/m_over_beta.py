#!/usr/bin/env python3
"""
m/β Ratio Histogram Plot.

Plots the distribution of m/β (slope_times_log_f / beta_software) values
across all rollouts. This ratio is important for understanding takeoff dynamics.

Usage:
    python -m plots.histograms.m_over_beta --run-dir outputs/251209_updated_params
"""

from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot
from plotting_utils.rollouts_reader import RolloutsReader


@register_plot(
    PlotSpec(
        name="m_over_beta_hist",
        display_name="m/β Ratio Distribution",
        category=PlotCategory.HISTOGRAM,
        speed=PlotSpeed.SLOW,  # Requires full JSONL parsing
        output_filename="m_over_beta_hist.png",
        requires_cache=False,
        requires_full_jsonl=True,
        description="Distribution of m/β ratio across rollouts",
    )
)
class MOverBetaHistogramPlot(BasePlot):
    """Histogram of m/β ratio values across all rollouts."""

    def generate(self, **kwargs) -> Path:
        """Generate the m/β histogram plot.

        Args:
            **kwargs: Additional arguments (bins, etc.)

        Returns:
            Path to the generated plot
        """
        from plotting.histograms import plot_m_over_beta_histogram

        bins = kwargs.get("bins", 50)
        rollouts_file = self.run_dir / "rollouts.jsonl"

        # Read data using RolloutsReader
        reader = RolloutsReader(rollouts_file)
        m_over_beta_values, num_skipped = reader.read_m_over_beta()

        if len(m_over_beta_values) == 0:
            raise ValueError("No valid m/β values found in rollouts")

        # Get output path
        out_path = self.get_output_path(kwargs.get("out_path"))
        self.ensure_output_dir()

        # Generate plot using existing function
        plot_m_over_beta_histogram(
            m_over_beta_values,
            num_skipped,
            out_path,
            bins=bins,
            title="Distribution of m/β",
        )

        # Print statistics
        arr = np.asarray(m_over_beta_values, dtype=float)
        pct_above_one = 100.0 * np.sum(arr > 1) / len(arr)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"  Percentage m/β > 1: {pct_above_one:.1f}%")
        print(f"  P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f}")
        if num_skipped > 0:
            print(f"  Note: {num_skipped} rollouts skipped due to missing data")

        return out_path


if __name__ == "__main__":
    MOverBetaHistogramPlot.main()
