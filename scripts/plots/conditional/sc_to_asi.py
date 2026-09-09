#!/usr/bin/env python3
"""
SC to ASI Duration Conditional Plot.

Plots the probability density function of the duration from SC (Superhuman Coder)
to ASI, conditional on SC being achieved in a specific year.

Usage:
    python -m plots.conditional.sc_to_asi --run-dir outputs/251209_updated_params --year 2027
    python -m plots.conditional.sc_to_asi --run-dir outputs/251209_updated_params --year 2028 --max-duration 40
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


@register_plot(
    PlotSpec(
        name="sc_to_asi_conditional",
        display_name="SC→ASI Duration (Conditional)",
        category=PlotCategory.CONDITIONAL,
        speed=PlotSpeed.SLOW,
        output_filename="sc_to_asi_conditional.png",
        requires_cache=False,
        requires_full_jsonl=True,
        milestones=["AI2027-SC", "ASI"],
        description="Duration from SC to ASI conditional on SC achievement year",
    )
)
class SCToASIConditionalPlot(BasePlot):
    """PDF plot of SC→ASI duration conditional on SC achievement year."""

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
            "--max-duration",
            type=float,
            default=30.0,
            help="Maximum duration to plot in years (default: 30)",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the SC→ASI conditional duration plot.

        Args:
            **kwargs: Arguments including year, max_duration, etc.

        Returns:
            Path to the generated plot
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from plotting_utils.kde import make_gamma_kernel_kde
        from plotting_utils.rollouts_reader import RolloutsReader
        from plotting_utils.style import apply_style

        apply_style()

        filter_year = kwargs.get("year", 2027)
        max_duration = kwargs.get("max_duration", 30.0)

        rollouts_file = self.run_dir / "rollouts.jsonl"
        reader = RolloutsReader(rollouts_file)

        durations = []
        total_sc_in_year = 0
        asi_not_achieved = 0

        # Iterate through rollouts and filter
        for rollout in reader.iter_normalized_rollouts():
            # Check SC achievement
            sc_data = reader._resolve_milestone(rollout, "AI2027-SC")
            if sc_data is None:
                continue

            sc_time = sc_data["time"]

            # Filter: Only include if SC achieved in the target year
            if not (float(filter_year) <= sc_time < float(filter_year + 1)):
                continue

            total_sc_in_year += 1

            # Check ASI achievement
            asi_data = reader._resolve_milestone(rollout, "ASI")
            if asi_data is None:
                asi_not_achieved += 1
                continue

            asi_time = asi_data["time"]
            duration = asi_time - sc_time

            if duration > 0:
                durations.append(duration)

        if len(durations) < 2:
            raise ValueError(
                f"Not enough data: only {len(durations)} samples with SC in {filter_year}"
            )

        durations = np.array(durations)
        print(f"Found {len(durations)} rollouts with SC in {filter_year} and ASI achieved")
        print(
            f"  ({asi_not_achieved} rollouts had SC in {filter_year} but ASI not achieved)"
        )

        # Create KDE with lower bound at 0
        kde = make_gamma_kernel_kde(durations, lower_bound=0.0)

        # Evaluate KDE
        xs = np.linspace(0, max_duration, 512)
        pdf_values = kde.pdf(xs) * 100  # Convert to percentage per year

        # Calculate statistics
        q10, q25, q50, q75, q90 = np.quantile(durations, [0.1, 0.25, 0.5, 0.75, 0.9])
        mode_idx = np.argmax(pdf_values)
        mode = xs[mode_idx]
        mean_dur = np.mean(durations)

        # Calculate percentile for 1.1 years
        pct_at_1_1 = np.mean(durations <= 1.1) * 100

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(xs, pdf_values, linewidth=2.5, color="tab:blue")
        ax.fill_between(xs, pdf_values, alpha=0.3, color="tab:blue")

        # Add vertical lines for statistics
        ax.axvline(
            q50,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Median: {q50:.1f} years",
        )
        ax.axvline(
            mode,
            color="orange",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Mode: {mode:.1f} years",
        )
        ax.axvline(
            1.1,
            color="tab:purple",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"P{round(pct_at_1_1)}: 1.1 years",
        )

        # Statistics box
        pct_asi_achieved = (
            len(durations) / total_sc_in_year * 100 if total_sc_in_year > 0 else 0
        )
        stats_text = (
            f"N = {len(durations)} rollouts (SC in {filter_year})\n"
            f"ASI achieved: {pct_asi_achieved:.1f}%\n"
            f"Mode: {mode:.1f} years\n"
            f"Mean: {mean_dur:.1f} years\n"
            f"P{round(pct_at_1_1)}: 1.1 years\n"
            f"P10: {q10:.1f} years\n"
            f"P25: {q25:.1f} years\n"
            f"P50: {q50:.1f} years\n"
            f"P75: {q75:.1f} years\n"
            f"P90: {q90:.1f} years"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
            family="monospace",
        )

        ax.set_xlabel("Duration from SC to ASI (years)", fontsize=12)
        ax.set_ylabel("Probability Density (% per year)", fontsize=12)
        ax.set_title(
            f"SC → ASI Duration Distribution | SC achieved in {filter_year}", fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=11)
        ax.set_xlim(0, max_duration)

        # Determine output path
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / f"sc_to_asi_conditional_{filter_year}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path


if __name__ == "__main__":
    SCToASIConditionalPlot.main()
