#!/usr/bin/env python3
"""
Horizon at SC Histogram Plot.

Plots the distribution of horizon lengths at the time of AC (Autonomous Coding)
achievement across rollouts.

Usage:
    python -m plots.histograms.horizon_at_sc --run-dir outputs/251209_updated_params
"""

from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


def format_time_duration(value: float) -> str:
    """Format a time duration in seconds to human-readable form."""
    if value < 60:
        return f"{value:.1f}s"
    elif value < 3600:
        return f"{value / 60:.1f}m"
    elif value < 86400:
        return f"{value / 3600:.1f}h"
    elif value < 86400 * 365:
        return f"{value / 86400:.1f}d"
    else:
        return f"{value / (86400 * 365):.1f}y"


@register_plot(
    PlotSpec(
        name="horizon_at_sc_hist",
        display_name="Horizon at SC Distribution",
        category=PlotCategory.HISTOGRAM,
        speed=PlotSpeed.FAST,
        output_filename="horizon_at_sc_hist.png",
        requires_cache=True,
        requires_full_jsonl=False,
        milestones=["AC"],
        description="Distribution of horizon lengths at AC achievement time",
    )
)
class HorizonAtSCHistogramPlot(BasePlot):
    """Histogram of horizon lengths at SC (AC) achievement time."""

    def generate(self, **kwargs) -> Path:
        """Generate the horizon at SC histogram plot.

        Args:
            **kwargs: Additional arguments (bins, etc.)

        Returns:
            Path to the generated plot
        """
        from plotting.histograms import plot_horizon_at_sc_histogram

        bins = kwargs.get("bins", 50)

        # Read horizon lengths at AC time with appropriate clipping
        # Cap to avoid absurd tails: 120000 engineers * 52 weeks * 40 hours * 60 minutes
        cap = float(120000 * 52 * 40 * 60)
        values = self.reader.read_metric_at_milestone(
            "horizon_lengths", "aa_time", clip_min=0.001, clip_max=cap
        )

        if len(values) > 0:
            arr = np.asarray(values, dtype=float)
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"Loaded {len(arr)} horizon_at_sc values")
            print(
                f"P10/Median/P90: {format_time_duration(q10)} / "
                f"{format_time_duration(q50)} / {format_time_duration(q90)}"
            )
        else:
            print("Warning: No horizon_at_sc values found")

        # Get output path
        out_path = self.get_output_path(kwargs.get("out_path"))
        self.ensure_output_dir()

        # Generate plot
        plot_horizon_at_sc_histogram(values, out_path=out_path, bins=bins)

        return out_path


if __name__ == "__main__":
    HorizonAtSCHistogramPlot.main()
