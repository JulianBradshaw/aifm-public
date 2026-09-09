#!/usr/bin/env python3
"""
Milestone Effective Compute Histogram Plot.

Plots the distribution of effective compute (in log OOMs) at the time
of milestone achievement across rollouts.

Usage:
    python -m plots.histograms.milestone_compute --run-dir outputs/251209_updated_params --milestone AC
    python -m plots.histograms.milestone_compute --run-dir outputs/251209_updated_params --milestone AI2027-SC
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


@register_plot(
    PlotSpec(
        name="milestone_compute_hist",
        display_name="Milestone Effective Compute Distribution",
        category=PlotCategory.HISTOGRAM,
        speed=PlotSpeed.FAST,
        output_filename="milestone_compute_hist.png",
        requires_cache=True,
        requires_full_jsonl=False,
        description="Distribution of effective compute at milestone achievement",
    )
)
class MilestoneComputeHistogramPlot(BasePlot):
    """Histogram of effective compute at milestone achievement."""

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
            "--list-milestones",
            action="store_true",
            help="List available milestone names and exit",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the milestone compute histogram plot.

        Args:
            **kwargs: Arguments including milestone, bins

        Returns:
            Path to the generated plot
        """
        from plotting.histograms import plot_milestone_effective_compute_histogram

        milestone = kwargs.get("milestone", "AC")
        bins = kwargs.get("bins", 50)

        # Handle list-milestones request
        if kwargs.get("list_milestones"):
            self._list_milestones()
            return Path()

        # Read milestone compute values
        compute_ooms, num_not_achieved = self.reader.read_milestone_compute(milestone)

        if (len(compute_ooms) + num_not_achieved) > 0 and len(compute_ooms) > 0:
            arr = np.asarray(compute_ooms, dtype=float)
            print(
                f"Loaded {len(arr)} effective compute values for '{milestone}' "
                f"(+{num_not_achieved} Not achieved)"
            )
            print(
                f"Min/Median/Max: {arr.min():.2f} / {np.median(arr):.2f} / "
                f"{arr.max():.2f} log(2025-FLOP)"
            )
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f} log(2025-FLOP)")
        elif (len(compute_ooms) + num_not_achieved) > 0:
            print(
                f"Loaded 0 effective compute values for '{milestone}' "
                f"(+{num_not_achieved} Not achieved)"
            )

        # Determine output path with milestone name
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / f"{milestone}_compute_hist.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate plot
        plot_milestone_effective_compute_histogram(
            compute_ooms,
            num_not_achieved=num_not_achieved,
            out_path=out_path,
            bins=bins,
            milestone_label=milestone,
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
    MilestoneComputeHistogramPlot.main()
