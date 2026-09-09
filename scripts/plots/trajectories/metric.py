#!/usr/bin/env python3
"""
Metric Trajectory Plot.

Plots time series trajectories for any metric across multiple rollouts.
Supports horizon_lengths, effective_compute, and other metrics.

Usage:
    # Plot horizon trajectories (default)
    python -m plots.trajectories.metric --run-dir outputs/251209_updated_params --metric horizon_lengths

    # Plot effective compute trajectories
    python -m plots.trajectories.metric --run-dir outputs/251209_updated_params --metric effective_compute

    # With MSE shading
    python -m plots.trajectories.metric --run-dir outputs/251209_updated_params --metric horizon_lengths --shade-by-mse
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


@register_plot(
    PlotSpec(
        name="metric_trajectories",
        display_name="Metric Trajectories",
        category=PlotCategory.TRAJECTORY,
        speed=PlotSpeed.SLOW,  # Requires full JSONL parsing
        output_filename="metric_trajectories.png",
        requires_cache=False,
        requires_full_jsonl=True,
        description="Time series trajectories for a specified metric",
    )
)
class MetricTrajectoryPlot(BasePlot):
    """Plot trajectories for any metric across rollouts."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add metric-specific arguments."""
        parser.add_argument(
            "--metric",
            type=str,
            default="horizon_lengths",
            help="Metric name to plot (e.g., 'horizon_lengths', 'effective_compute')",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.3,
            help="Transparency for trajectory lines (default: 0.3)",
        )
        parser.add_argument(
            "--max-trajectories",
            type=int,
            default=100,
            help="Maximum number of trajectories to plot (default: 100)",
        )
        parser.add_argument(
            "--stop-at-sc",
            action="store_true",
            help="Stop trajectories at SC achievement time",
        )
        parser.add_argument(
            "--shade-by-mse",
            action="store_true",
            help="Color trajectories by MSE value",
        )
        parser.add_argument(
            "--mse-threshold",
            type=float,
            default=None,
            help="MSE threshold for shading (default: auto)",
        )
        parser.add_argument(
            "--cutoff-year",
            type=float,
            default=None,
            help="Cut off trajectories at this year",
        )
        parser.add_argument(
            "--yscale",
            type=str,
            default="log",
            choices=["log", "linear"],
            help="Y-axis scale (default: log)",
        )
        # Horizon-specific arguments
        parser.add_argument(
            "--current-horizon-minutes",
            type=float,
            default=1.0,
            help="Current horizon in minutes (for horizon_lengths metric)",
        )
        parser.add_argument(
            "--no-metr",
            action="store_true",
            help="Don't overlay METR data (for horizon_lengths metric)",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the metric trajectory plot.

        Args:
            **kwargs: Arguments including metric, alpha, max_trajectories, etc.

        Returns:
            Path to the generated plot
        """
        from plotting.trajectories import (
            plot_horizon_trajectories,
            plot_metric_trajectories,
        )

        metric = kwargs.get("metric", "horizon_lengths")
        alpha = kwargs.get("alpha", 0.3)
        max_trajectories = kwargs.get("max_trajectories", 100)
        stop_at_sc = kwargs.get("stop_at_sc", False)
        shade_by_mse = kwargs.get("shade_by_mse", False)
        mse_threshold = kwargs.get("mse_threshold")
        cutoff_year = kwargs.get("cutoff_year")
        yscale = kwargs.get("yscale", "log")

        # Read trajectories
        times, trajectories, aa_times, mse_values = self.reader.read_metric_trajectories(
            metric
        )
        print(
            f"Loaded {len(trajectories)} {metric} trajectories "
            f"with {len(times)} time points each"
        )

        # Determine output path with metric name
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / f"{metric}_trajectories.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Use specialized function for horizon_lengths
        if metric == "horizon_lengths":
            current_horizon_minutes = kwargs.get("current_horizon_minutes", 1.0)
            overlay_metr = not kwargs.get("no_metr", False)

            plot_horizon_trajectories(
                times,
                trajectories,
                out_path=out_path,
                current_horizon_minutes=current_horizon_minutes,
                alpha=alpha,
                max_trajectories=max_trajectories,
                overlay_metr=overlay_metr,
                stop_at_sc=stop_at_sc,
                aa_times=aa_times,
                mse_values=mse_values,
                shade_by_mse=shade_by_mse,
                mse_threshold=mse_threshold,
                cutoff_year=cutoff_year,
            )
        else:
            plot_metric_trajectories(
                times,
                trajectories,
                out_path=out_path,
                metric_name=metric,
                alpha=alpha,
                max_trajectories=max_trajectories,
                stop_at_sc=stop_at_sc,
                aa_times=aa_times,
                mse_values=mse_values,
                shade_by_mse=shade_by_mse,
                mse_threshold=mse_threshold,
                cutoff_year=cutoff_year,
                yscale=yscale,
            )

        return out_path


if __name__ == "__main__":
    MetricTrajectoryPlot.main()
