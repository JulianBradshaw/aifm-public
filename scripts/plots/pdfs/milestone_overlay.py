#!/usr/bin/env python3
"""
Milestone PDF Overlay Plot.

Plots probability density functions for milestone arrival times as overlaid curves.
Supports multiple presets for different milestone combinations and KDE methods.

Consolidates the following scripts:
- milestone_pdfs.py
- plot_ac_sar_pdfs.py
- plot_ac_sc_sar_pdfs.py
- plot_post_sar_milestone_pdfs.py

Usage:
    # Use a preset
    python -m plots.pdfs.milestone_overlay --run-dir outputs/251209_updated_params --preset AC_SAR

    # Custom milestones
    python -m plots.pdfs.milestone_overlay --run-dir outputs/251209_updated_params \
        --milestones "AC,AI2027-SC,ASI"

    # Use Gaussian KDE instead of gamma
    python -m plots.pdfs.milestone_overlay --run-dir outputs/251209_updated_params \
        --preset AC_SC_SAR --kde-mode gaussian
"""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..base import BasePlot
from ..registry import PlotCategory, PlotSpeed, PlotSpec, register_plot


@dataclass
class PresetConfig:
    """Configuration for a milestone overlay preset."""

    milestones: List[str]
    kde_mode: str  # "gamma", "gaussian", or "lower_bounded"
    display_names: Dict[str, str] = field(default_factory=dict)
    colors: Dict[str, str] = field(default_factory=dict)
    output_filename: str = "milestone_pdfs_overlay.png"
    x_min: float = 2025
    x_max: float = 2050
    scale_by_achievement: bool = False


# Default colors for milestones
DEFAULT_COLORS = {
    "AC": "tab:blue",
    "AI2027-SC": "tab:green",
    "SAR-level-experiment-selection-skill": "tab:orange",
    "SIAR-level-experiment-selection-skill": "tab:purple",
    "TED-AI": "tab:red",
    "ASI": "tab:brown",
}

DEFAULT_DISPLAY_NAMES = {
    "SAR-level-experiment-selection-skill": "SAR",
    "SIAR-level-experiment-selection-skill": "SIAR",
}

PRESETS: Dict[str, PresetConfig] = {
    "AC_SAR": PresetConfig(
        milestones=["AC", "SAR-level-experiment-selection-skill"],
        kde_mode="gamma",
        output_filename="ac_sar_pdfs.png",
    ),
    "AC_SC_SAR": PresetConfig(
        milestones=["AC", "AI2027-SC", "SAR-level-experiment-selection-skill"],
        kde_mode="gaussian",
        output_filename="ac_sc_sar_pdfs.png",
    ),
    "POST_SAR": PresetConfig(
        milestones=["SIAR-level-experiment-selection-skill", "TED-AI", "ASI"],
        kde_mode="gamma",
        output_filename="post_sar_pdfs.png",
        scale_by_achievement=True,
    ),
    "ALL": PresetConfig(
        milestones=[
            "AC",
            "AI2027-SC",
            "SAR-level-experiment-selection-skill",
            "SIAR-level-experiment-selection-skill",
            "TED-AI",
            "ASI",
        ],
        kde_mode="gamma",
        output_filename="all_milestone_pdfs.png",
    ),
}


@register_plot(
    PlotSpec(
        name="milestone_pdfs_overlay",
        display_name="Milestone PDF Overlay",
        category=PlotCategory.PDF,
        speed=PlotSpeed.FAST,
        output_filename="milestone_pdfs_overlay.png",
        requires_cache=True,
        requires_full_jsonl=False,
        description="Overlaid probability density functions for milestone arrival times",
    )
)
class MilestonePDFOverlayPlot(BasePlot):
    """Overlaid PDF plot for milestone arrival times."""

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add PDF-specific arguments."""
        parser.add_argument(
            "--preset",
            type=str,
            choices=list(PRESETS.keys()),
            default=None,
            help="Use a preset configuration (AC_SAR, AC_SC_SAR, POST_SAR, ALL)",
        )
        parser.add_argument(
            "--milestones",
            type=str,
            default=None,
            help="Comma-separated list of milestones (overrides preset)",
        )
        parser.add_argument(
            "--kde-mode",
            type=str,
            choices=["gamma", "gaussian", "lower_bounded"],
            default=None,
            help="KDE method (overrides preset)",
        )
        parser.add_argument(
            "--x-min",
            type=float,
            default=None,
            help="Minimum x-axis value (year)",
        )
        parser.add_argument(
            "--x-max",
            type=float,
            default=None,
            help="Maximum x-axis value (year)",
        )
        parser.add_argument(
            "--list-milestones",
            action="store_true",
            help="List available milestone names and exit",
        )

    def generate(self, **kwargs) -> Path:
        """Generate the milestone PDF overlay plot.

        Args:
            **kwargs: Arguments including preset, milestones, kde_mode, etc.

        Returns:
            Path to the generated plot
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from plotting_utils.helpers import load_present_day
        from plotting_utils.kde import (
            make_gamma_kernel_kde,
            make_gaussian_kde,
            make_lower_bounded_kde,
        )
        from plotting_utils.style import apply_style

        apply_style()

        # Handle list-milestones request
        if kwargs.get("list_milestones"):
            self._list_milestones()
            return Path()

        # Get configuration from preset or arguments
        preset_name = kwargs.get("preset")
        if preset_name:
            config = PRESETS[preset_name]
        else:
            config = PRESETS["AC_SAR"]  # Default

        # Override with explicit arguments
        milestones_str = kwargs.get("milestones")
        if milestones_str:
            milestones = [m.strip() for m in milestones_str.split(",")]
        else:
            milestones = config.milestones

        kde_mode = kwargs.get("kde_mode") or config.kde_mode
        x_min = kwargs.get("x_min") or config.x_min
        x_max = kwargs.get("x_max") or config.x_max

        # Load present_day for gamma KDE
        present_day = load_present_day(self.run_dir)

        # Read milestone times
        (
            times_map,
            not_achieved_map,
            typical_sim_end,
            total_rollouts,
        ) = self.reader.read_milestone_times_batch(milestones)

        # Print summary
        print(f"\nTotal rollouts: {total_rollouts}")
        for milestone in milestones:
            count = len(times_map.get(milestone, []))
            not_achieved = not_achieved_map.get(milestone, 0)
            pct = (count / total_rollouts * 100) if total_rollouts > 0 else 0
            print(f"  {milestone}: {count} achieved ({pct:.1f}%)")

        def clean_milestone_name(name: str) -> str:
            """Remove 'level' and 'skill' from milestone names."""
            return name.replace("-level", "").replace("-skill", "")

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use standard color cycle like old code
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

        stats_lines = []

        for idx, milestone in enumerate(milestones):
            times = times_map.get(milestone, [])
            num_not_achieved = not_achieved_map.get(milestone, 0)

            if len(times) < 2:
                print(
                    f"Warning: Not enough data for {milestone} to create KDE, skipping"
                )
                continue

            raw_data = np.asarray(times, dtype=float)
            boundary = float(present_day)
            valid_mask = raw_data >= boundary
            data = raw_data[valid_mask]
            dropped = raw_data.size - data.size
            if dropped:
                print(
                    f"  {milestone}: dropped {dropped} samples before present_day={boundary:.3f}."
                )
            if data.size < 2:
                print(
                    f"Warning: Not enough data for {milestone} after enforcing present_day boundary, skipping."
                )
                continue

            # Calculate percentiles including not achieved as sim_end
            if num_not_achieved > 0 and typical_sim_end is not None:
                combined_data = np.concatenate(
                    [raw_data, np.full(num_not_achieved, typical_sim_end)]
                )
            else:
                combined_data = raw_data

            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Calculate achievement probability
            total_runs = len(times) + num_not_achieved
            prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0

            # Select KDE method and prepare PDF (matching old _prepare_gamma_pdf behavior)
            if kde_mode == "gamma":
                kde = make_gamma_kernel_kde(data, lower_bound=boundary)
            elif kde_mode == "lower_bounded":
                kde = make_lower_bounded_kde(data, lower_bound=boundary)
            else:  # gaussian
                kde = make_gaussian_kde(data)

            # Sample at 2048 points like old code (not 512)
            bandwidth = float(kde.bandwidth) if hasattr(kde, "bandwidth") else 1.0
            min_eval = max(boundary, float(np.min(data) - 5.0 * bandwidth))
            max_eval = max(float(np.max(data) + 5.0 * bandwidth), min_eval + bandwidth)
            xs = np.linspace(min_eval, max_eval, 2048)

            # Scale by prob_achieved * 100 like old code
            scale_factor = prob_achieved * 100.0
            pdf_values = kde(xs) * scale_factor

            # Filter to plot_max_year
            mask = xs <= x_max
            if np.any(mask):
                xs_plot = xs[mask]
                pdf_plot = pdf_values[mask]
            else:
                xs_plot = xs
                pdf_plot = pdf_values

            # Calculate mode
            mode_idx = int(np.argmax(pdf_values))
            mode = float(xs[mode_idx])

            color = colors[idx % len(colors)]
            clean_name = clean_milestone_name(milestone)
            ax.plot(xs_plot, pdf_plot, linewidth=2.5, label=clean_name, color=color)

            # Calculate percentage after 2050
            num_after_2050 = sum(1 for t in raw_data if t > 2050) + num_not_achieved
            pct_after_2050 = (num_after_2050 / total_runs * 100) if total_runs > 0 else 0

            stats_lines.append(
                f"{clean_name}: Mode={mode:.1f}, P10={q10:.1f}, P50={q50:.1f}, P90={q90:.1f}, {pct_after_2050:.0f}% > 2050"
            )

        ax.set_xlabel("Arrival Time (decimal year)", fontsize=12)
        ax.set_ylabel("Probability Density (% per year)", fontsize=12)

        # Use custom title if provided via kwargs, otherwise use default
        custom_title = kwargs.get("title")
        ax.set_title(
            custom_title or "Milestone Arrival Time Distributions", fontsize=14
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        if x_max is not None:
            ax.set_xlim(right=x_max)

        # Add statistics text in top right
        if stats_lines:
            stats_text = "\n".join(stats_lines)
            plt.text(
                0.98,
                0.98,
                stats_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
                family="monospace",
            )

        # Determine output path
        out_path = kwargs.get("out_path")
        if out_path:
            out_path = Path(out_path)
        else:
            out_path = self.run_dir / config.output_filename

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

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
            print("\nAvailable presets:")
            for name, config in PRESETS.items():
                print(f"  - {name}: {', '.join(config.milestones)}")
        else:
            print("No milestones found in file.")


if __name__ == "__main__":
    MilestonePDFOverlayPlot.main()
