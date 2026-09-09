#!/usr/bin/env python3
"""
Milestone Time Histogram Plot.

Plots the distribution of milestone achievement times across rollouts.
Supports any milestone name (AC, AI2027-SC, ASI, etc.)

Usage:
    # Plot AC time histogram
    python -m plots.histograms.milestone_time --run-dir outputs/251209_updated_params --milestone AC

    # Plot any milestone histogram
    python -m plots.histograms.milestone_time --run-dir outputs/251209_updated_params --milestone AI2027-SC

    # Use CDF instead of PDF histogram
    python -m plots.histograms.milestone_time --run-dir outputs/251209_updated_params --milestone AC --cdf

    # List available milestone names
    python -m plots.histograms.milestone_time --run-dir outputs/251209_updated_params --list-milestones
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


@register_plot(
    PlotSpec(
        name="milestone_time_hist",
        display_name="Milestone Time Distribution",
        category=PlotCategory.HISTOGRAM,
        speed=PlotSpeed.FAST,  # Uses cache
        output_filename="milestone_time_hist.png",
        requires_cache=True,
        requires_full_jsonl=False,
        description="Distribution of milestone achievement times",
    )
)
class MilestoneTimeHistogramPlot(BasePlot):
    """Histogram of milestone achievement times."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add milestone-specific arguments."""
        parser.add_argument(
            "--milestone",
            type=str,
            required=True,
            help="Milestone name to plot (e.g., 'AC', 'AI2027-SC', 'ASI')",
        )
        parser.add_argument(
            "--cdf",
            action="store_true",
            dest="cdf_hist",
            help="Plot CDF instead of histogram",
        )
        parser.add_argument(
            "--list-milestones",
            action="store_true",
            help="List available milestone names and exit",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the milestone time histogram plot.

        Args:
            **kwargs: Arguments including milestone, bins, cdf_hist

        Returns:
            Path to the generated plot
        """
        from plotting.histograms import (
            plot_milestone_time_histogram,
            plot_milestone_time_histogram_cdf,
        )
        from plotting_utils.helpers import load_present_day

        milestone = kwargs.get("milestone", "AC")
        bins = kwargs.get("bins", 50)
        cdf_hist = kwargs.get("cdf_hist", False)

        # Handle list-milestones request
        if kwargs.get("list_milestones"):
            self._list_milestones()
            return Path()

        # Load present_day for gamma kernel KDE
        present_day = load_present_day(self.run_dir)

        # Read milestone times using RolloutsReader
        times, num_not_achieved, sim_end = self.reader.read_milestone_times(milestone)

        # Print statistics
        if (len(times) + num_not_achieved) > 0 and len(times) > 0:
            arr = np.asarray(times, dtype=float)
            print(
                f"Loaded {len(arr)} finite '{milestone}' arrival times "
                f"(+{num_not_achieved} Not achieved)"
            )
            print(
                f"Finite Min/Median/Max: {arr.min():.3f} / "
                f"{np.median(arr):.3f} / {arr.max():.3f}"
            )
            with_inf = (
                np.concatenate([arr, np.full(int(num_not_achieved), np.inf)])
                if num_not_achieved > 0
                else arr
            )
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])

            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "Not achieved"

            print(
                f"P10/Median/P90 (incl. Not achieved): "
                f"{_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}"
            )
        elif (len(times) + num_not_achieved) > 0:
            print(
                f"Loaded 0 finite '{milestone}' arrival times "
                f"(+{num_not_achieved} Not achieved)"
            )

        # Determine output path with milestone name
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            suffix = "_cdf" if cdf_hist else ""
            out_path = self.run_dir / f"{milestone}_time_hist{suffix}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate plot
        if cdf_hist:
            plot_milestone_time_histogram_cdf(
                times,
                num_not_achieved=num_not_achieved,
                out_path=out_path,
                bins=bins,
                milestone_label=milestone,
                sim_end=sim_end,
            )
        else:
            plot_milestone_time_histogram(
                times,
                num_not_achieved=num_not_achieved,
                out_path=out_path,
                bins=bins,
                milestone_label=milestone,
                present_day=present_day,
                sim_end=sim_end,
            )

        return out_path

    def _list_milestones(self) -> None:
        """List available milestone names from rollouts."""
        import json

        rollouts_file = self.run_dir / "rollouts.jsonl"
        milestone_names: set = set()

        with rollouts_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                results = rec.get("results", {})
                if not isinstance(results, dict):
                    continue

                milestones = results.get("milestones", {})
                if isinstance(milestones, dict):
                    milestone_names.update(milestones.keys())

        if milestone_names:
            print("Milestones found:")
            for name in sorted(milestone_names):
                print(f"  - {name}")
        else:
            print("No milestones found in file.")


if __name__ == "__main__":
    MilestoneTimeHistogramPlot.main()
