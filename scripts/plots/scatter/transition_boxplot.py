#!/usr/bin/env python3
"""
Milestone Transition Boxplot.

Plots boxplots of transition durations between milestone pairs.
Shows the distribution of time between achieving different milestones.

Usage:
    python -m plots.scatter.transition_boxplot --run-dir outputs/251209_updated_params --pairs "AC:AI2027-SC,AI2027-SC:ASI"
    python -m plots.scatter.transition_boxplot --run-dir outputs/251209_updated_params --pairs "AC:ASI" --filter-milestone AC --filter-by-year 2027
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


def _parse_milestone_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    """Parse comma-separated milestone pairs like 'A:B,C:D'."""
    if not pairs_str:
        return []
    pairs = []
    for pair in pairs_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            a, b = pair.split(":", 1)
            pairs.append((a.strip(), b.strip()))
    return pairs


@register_plot(
    PlotSpec(
        name="transition_boxplot",
        display_name="Milestone Transition Boxplot",
        category=PlotCategory.SCATTER,
        speed=PlotSpeed.FAST,
        output_filename="transition_boxplot.png",
        requires_cache=True,
        requires_full_jsonl=False,
        description="Boxplot of transition durations between milestones",
    )
)
class TransitionBoxplotPlot(BasePlot):
    """Boxplot of milestone transition durations."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add boxplot-specific arguments."""
        parser.add_argument(
            "--pairs",
            type=str,
            required=True,
            help="Comma-separated milestone pairs (e.g., 'AC:AI2027-SC,AI2027-SC:ASI')",
        )
        parser.add_argument(
            "--filter-milestone",
            type=str,
            default=None,
            help="Filter to rollouts where this milestone was achieved",
        )
        parser.add_argument(
            "--filter-by-year",
            type=float,
            default=None,
            help="Filter to rollouts where filter-milestone was achieved by this year",
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
            help="Display value for non-achieved milestones (default: 10.0)",
        )
        parser.add_argument(
            "--list-milestones",
            action="store_true",
            help="List available milestone names and exit",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the transition boxplot.

        Args:
            **kwargs: Arguments including pairs, filter_milestone, etc.

        Returns:
            Path to the generated plot
        """
        from plotting.boxplots import plot_milestone_transition_boxplot

        # Handle list-milestones request
        if kwargs.get("list_milestones"):
            self._list_milestones()
            return Path()

        pairs_str = kwargs.get("pairs", "")
        pairs = _parse_milestone_pairs(pairs_str)
        if not pairs:
            raise ValueError(
                "--pairs is required (use --list-milestones to inspect names)"
            )

        filter_milestone = kwargs.get("filter_milestone")
        filter_by_year = kwargs.get("filter_by_year")
        ymin_years = kwargs.get("ymin_years")
        ymax_years = kwargs.get("ymax_years")
        exclude_inf = kwargs.get("exclude_inf_from_stats", False)
        inf_years = kwargs.get("inf_years", 10.0)

        # Read transition durations
        (
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            typical_max,
            simulation_cutoff,
        ) = self.reader.read_multiple_transition_durations(
            pairs,
            filter_milestone=filter_milestone,
            filter_by_year=filter_by_year,
        )

        # Print statistics
        for lbl, arr, n_not_achieved, n_before, total_a in zip(
            labels, durations, num_b_not_achieved, num_b_before_a, total_per_pair
        ):
            arr_np = np.asarray(arr, dtype=float)
            if arr_np.size:
                q10, q50, q90 = np.quantile(arr_np, [0.1, 0.5, 0.9])
                print(
                    f"{lbl}: n={arr_np.size} achieved in order | "
                    f"P10/Median/P90 (years): {q10:.3f} / {q50:.3f} / {q90:.3f}"
                )
                if n_not_achieved > 0:
                    milestone_b = lbl.split(" to ")[1] if " to " in lbl else "B"
                    print(f"  +{n_not_achieved}/{total_a} {milestone_b} not achieved")
                if n_before > 0:
                    print(f"  +{n_before}/{total_a} out of order")
            else:
                print(
                    f"{lbl}: n=0 achieved in order (total where A achieved: {total_a})"
                )

        # Determine output path
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / "transition_boxplot.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build condition text
        condition_text = None
        if filter_milestone and filter_by_year is not None:
            condition_text = f"{filter_milestone} achieved by {filter_by_year}"

        # Generate plot
        plot_milestone_transition_boxplot(
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            out_path=out_path,
            title=None,
            ymin_years=ymin_years,
            ymax_years=ymax_years,
            exclude_inf_from_stats=exclude_inf,
            inf_years_cap=typical_max,
            inf_years_display=inf_years,
            condition_text=condition_text,
            simulation_cutoff=simulation_cutoff,
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
    TransitionBoxplotPlot.main()
