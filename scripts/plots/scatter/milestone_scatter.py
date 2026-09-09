#!/usr/bin/env python3
"""
Milestone Scatter Plot.

Plots a scatter/hexbin of milestone achievement times for two milestones.
Shows the relationship between when different milestones are achieved.

Usage:
    python -m plots.scatter.milestone_scatter --run-dir outputs/251209_updated_params --scatter-pair AIR-5x:AIR-2000x
    python -m plots.scatter.milestone_scatter --run-dir outputs/251209_updated_params --scatter-pair AC:ASI --scatter-kind hexbin
"""

from argparse import ArgumentParser
from pathlib import Path

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


@register_plot(
    PlotSpec(
        name="milestone_scatter",
        display_name="Milestone Scatter Plot",
        category=PlotCategory.SCATTER,
        speed=PlotSpeed.FAST,
        output_filename="milestone_scatter.png",
        requires_cache=True,
        requires_full_jsonl=False,
        description="Scatter plot of milestone achievement times",
    )
)
class MilestoneScatterPlot(BasePlot):
    """Scatter plot of milestone achievement times."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add scatter-specific arguments."""
        parser.add_argument(
            "--scatter-pair",
            type=str,
            required=True,
            help="Milestone pair to plot (e.g., 'AC:ASI', 'AIR-5x:AIR-2000x')",
        )
        parser.add_argument(
            "--scatter-kind",
            type=str,
            default="scatter",
            choices=["scatter", "hexbin", "density"],
            help="Type of scatter plot (default: scatter)",
        )
        parser.add_argument(
            "--gridsize",
            type=int,
            default=30,
            help="Grid size for hexbin plots (default: 30)",
        )
        parser.add_argument(
            "--point-size",
            type=float,
            default=10.0,
            help="Point size for scatter plots (default: 10.0)",
        )
        parser.add_argument(
            "--no-scatter-overlay",
            action="store_true",
            help="Don't overlay individual points on hexbin",
        )
        parser.add_argument(
            "--ymin-years",
            type=float,
            default=None,
            help="Minimum y-axis value in years",
        )
        parser.add_argument(
            "--ymax-years",
            type=float,
            default=None,
            help="Maximum y-axis value in years",
        )
        parser.add_argument(
            "--exclude-inf-from-stats",
            action="store_true",
            help="Exclude non-achieved milestones from statistics",
        )
        parser.add_argument(
            "--inf-years",
            type=float,
            default=10.0,
            help="Cap for non-achieved milestones in years (default: 10.0)",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the milestone scatter plot.

        Args:
            **kwargs: Arguments including scatter_pair, scatter_kind, etc.

        Returns:
            Path to the generated plot
        """
        from plotting.scatter import plot_milestone_scatter

        scatter_pair = kwargs.get("scatter_pair", "")
        if ":" not in scatter_pair:
            raise ValueError("--scatter-pair must be formatted as FROM:TO")

        from_name, to_name = [s.strip() for s in scatter_pair.split(":", 1)]

        scatter_kind = kwargs.get("scatter_kind", "scatter")
        gridsize = kwargs.get("gridsize", 30)
        point_size = kwargs.get("point_size", 10.0)
        scatter_overlay = not kwargs.get("no_scatter_overlay", False)
        ymin_years = kwargs.get("ymin_years")
        ymax_years = kwargs.get("ymax_years")
        exclude_inf = kwargs.get("exclude_inf_from_stats", False)
        inf_years = kwargs.get("inf_years", 10.0)

        # Read transition data
        xs, ys = self.reader.read_transition_data(
            from_name,
            to_name,
            include_censored=(not exclude_inf),
            inf_years_cap=inf_years if not exclude_inf else None,
            return_arrays=True,
        )
        print(f"Loaded {xs.size} points for scatter {from_name} -> {to_name}")

        # Determine output path
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / f"scatter_{from_name}_to_{to_name}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate plot
        plot_milestone_scatter(
            xs,
            ys,
            out_path=out_path,
            kind=scatter_kind,
            gridsize=gridsize,
            point_size=point_size,
            scatter_overlay=scatter_overlay,
            ymin_years=ymin_years,
            ymax_years=ymax_years,
            condition_text=None,
        )

        return out_path


if __name__ == "__main__":
    MilestoneScatterPlot.main()
