"""Boxplot utilities for milestone transition analysis."""

from pathlib import Path
from typing import List, Optional
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

from plotting_utils.helpers import format_years_value, get_year_tick_values_and_labels


def plot_milestone_transition_boxplot(
    labels: List[str],
    durations_per_pair: List[List[float]],
    durations_with_censored_per_pair: List[List[float]],
    num_b_not_achieved_per_pair: List[int],
    num_b_before_a_per_pair: List[int],
    total_per_pair: List[int],
    out_path: Path,
    title: Optional[str] = None,
    ymin_years: Optional[float] = None,
    ymax_years: Optional[float] = None,
    exclude_inf_from_stats: bool = False,
    inf_years_cap: Optional[float] = None,
    inf_years_display: float = 100.0,
    condition_text: Optional[str] = None,
    simulation_cutoff: Optional[float] = None,
) -> None:
    """Plot milestone transition boxplot comparing achieved vs censored durations.

    Args:
        labels: Milestone transition labels (e.g., "AC to SAR")
        durations_per_pair: List of duration arrays (achieved only)
        durations_with_censored_per_pair: List of duration arrays (including censored)
        num_b_not_achieved_per_pair: Count of trajectories where B not achieved
        num_b_before_a_per_pair: Count of trajectories where B before A
        total_per_pair: Total trajectories per pair
        out_path: Output file path
        title: Plot title
        ymin_years: Minimum Y value (years)
        ymax_years: Maximum Y value (years)
        exclude_inf_from_stats: Whether to exclude infinite values from statistics
        inf_years_cap: Cap for infinite year values
        inf_years_display: Display value for infinite years
        condition_text: Optional condition text to display
        simulation_cutoff: Simulation cutoff year for labeling
    """
    if len(labels) == 0:
        raise ValueError("No milestone pairs provided")
    if len(labels) != len(durations_per_pair) or len(labels) != len(num_b_not_achieved_per_pair) or len(labels) != len(num_b_before_a_per_pair):
        raise ValueError("Mismatched inputs for boxplot")

    # Prepare data groups. Finite groups are the true finite durations.
    # Censored groups include both achieved and censored (not achieved -> sim_end)
    finite_groups: List[np.ndarray] = []
    censored_groups: List[np.ndarray] = []
    global_min = np.inf
    global_max = 0.0

    for arr, arr_censored in zip(durations_per_pair, durations_with_censored_per_pair):
        # Finite only
        a = np.asarray(arr, dtype=float)
        if a.size:
            a = a[np.isfinite(a) & (a > 0)]
        finite_groups.append(a)
        if a.size:
            global_min = min(global_min, float(a.min()))
            global_max = max(global_max, float(a.max()))

        # Including censored
        ac = np.asarray(arr_censored, dtype=float)
        if ac.size:
            ac = ac[np.isfinite(ac) & (ac > 0)]
        censored_groups.append(ac)
        if ac.size:
            global_min = min(global_min, float(ac.min()))
            global_max = max(global_max, float(ac.max()))

    if not np.isfinite(global_min):
        raise ValueError("No finite durations found to plot")

    # Axis limits
    ymin = float(ymin_years) if ymin_years is not None else 10 ** np.floor(np.log10(global_min)) / 10.0
    ymax_candidate = float(ymax_years) if ymax_years is not None else 10 ** np.ceil(np.log10(max(global_max, global_min * 10)))
    # Ensure y-axis includes the display point for 'Not achieved' points
    ymax_candidate = max(ymax_candidate, float(inf_years_display))
    ymax = max(ymax_candidate, ymin * 10.0)

    plt.figure(figsize=(22, 9))
    ax = plt.gca()

    # Adjust plot area to leave room for stats panel on right and labels at bottom
    plt.subplots_adjust(right=0.68, bottom=0.15)

    # Interleave the two sets of boxes: achieved only and including censored
    # For each pair, we'll have two boxes side by side
    all_groups = []
    all_positions = []
    all_labels = []
    colors = []

    for i, (label, fg, cg) in enumerate(zip(labels, finite_groups, censored_groups)):
        # Position boxes with spacing: pairs at 3*i and 3*i+1, with gap to next pair
        pos_achieved = 3 * i + 0.6
        pos_censored = 3 * i + 1.4

        all_groups.append(fg)
        all_positions.append(pos_achieved)
        colors.append((0.5, 0.8, 0.5, 0.7))  # Green for achieved only

        all_groups.append(cg)
        all_positions.append(pos_censored)
        colors.append((0.5, 0.5, 0.8, 0.7))  # Blue for including censored

        # Label in the middle of the two boxes
        all_labels.append(label if i == 0 else "")
        all_labels.append("")

    # Create boxplot with custom positions
    bp = plt.boxplot(
        all_groups,
        positions=all_positions,
        showfliers=True,
        patch_artist=True,
        widths=0.7,
        whis=1.5,
    )

    # Styling with different colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, edgecolor='black')
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        for line in bp[element]:
            line.set(color='black', linewidth=1.2)

    # Set custom x-tick labels at the center of each pair
    tick_positions = [3 * i + 1.0 for i in range(len(labels))]
    ax.set_xticks(tick_positions)

    # Wrap long labels to multiple lines
    wrapped_labels = []
    for label in labels:
        # Replace " to " with "\nto\n" for line break
        wrapped = label.replace(" to ", "\nto\n")
        wrapped_labels.append(wrapped)

    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=20)

    # Add legend
    cutoff_text = f'Assuming achieved at {simulation_cutoff:.0f}' if simulation_cutoff is not None else 'Assuming achieved at cutoff'
    legend_elements = [
        Patch(facecolor=(0.5, 0.8, 0.5, 0.7), edgecolor='black', label='Both achieved'),
        Patch(facecolor=(0.5, 0.5, 0.8, 0.7), edgecolor='black', label=cutoff_text)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=16)

    plt.yscale('log')
    ticks, tick_labels = get_year_tick_values_and_labels(ymin, ymax)
    plt.yticks(ticks, tick_labels, fontsize=18)
    plt.ylim(ymin, ymax)

    # Set x-axis limits to show all boxes with padding
    ax.set_xlim(-0.5, 3 * len(labels) - 0.5)

    plt.xlabel("Milestone Transition", fontsize=20)
    plt.ylabel("Calendar Years (log scale)", fontsize=20)
    plt.grid(True, which="both", axis="y", alpha=0.25)

    plt.title(title or "Time Spent in Each Milestone Transition (calendar years)", fontsize=22)

    # Stats panel
    x_text = 1.03
    y_text = 0.98
    panel_lines: List[str] = ["Statistics (years):", ""]
    if condition_text:
        panel_lines.append(f"Condition: {condition_text}")
        panel_lines.append("")

    for lbl, arr, arr_c, n_not_achieved, n_before, total_a in zip(labels, finite_groups, censored_groups, num_b_not_achieved_per_pair, num_b_before_a_per_pair, total_per_pair):
        # Wrap long milestone transition labels
        if len(lbl) > 40 and " to " in lbl:
            parts = lbl.split(" to ")
            panel_lines.append(f"{parts[0]} to")
            panel_lines.append(f"  {parts[1]}")
        else:
            panel_lines.append(lbl)

        # Stats for achieved only
        panel_lines.append("  Both achieved:")
        if arr.size == 0:
            panel_lines.append("    (none)")
        else:
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            panel_lines.append(f"    10/50/90: {format_years_value(float(q10))}/{format_years_value(float(q50))}/{format_years_value(float(q90))}")

        # Show not achieved count before "Assuming..." section
        if n_not_achieved > 0:
            milestone_b = lbl.split(" to ")[1] if " to " in lbl else "second"
            pct_not_achieved = 100.0 * n_not_achieved / total_a if total_a > 0 else 0.0
            panel_lines.append(f"  ({pct_not_achieved:.1f}% {milestone_b} not achieved)")

        # Stats for including censored
        milestone_b = lbl.split(" to ")[1] if " to " in lbl else "second"
        # Wrap if the full line would be too long
        cutoff_year = f"{simulation_cutoff:.0f}" if simulation_cutoff is not None else "simulation cutoff"
        full_line = f"  Assuming {milestone_b} achieved at {cutoff_year}:"
        if len(full_line) > 50:
            panel_lines.append(f"  Assuming {milestone_b}")
            panel_lines.append(f"    achieved at {cutoff_year}:")
        else:
            panel_lines.append(full_line)
        if arr_c.size == 0:
            panel_lines.append("    (none)")
        else:
            q10_c, q50_c, q90_c = np.quantile(arr_c, [0.1, 0.5, 0.9])
            panel_lines.append(f"    10/50/90: {format_years_value(float(q10_c))}/{format_years_value(float(q50_c))}/{format_years_value(float(q90_c))}")

        # Show out of order count at the end
        if n_before > 0:
            milestone_b = lbl.split(" to ")[1] if " to " in lbl else "second"
            milestone_a = lbl.split(" to ")[0] if " to " in lbl else "first"
            pct_before = 100.0 * n_before / total_a if total_a > 0 else 0.0
            panel_lines.append(f"  ({pct_before:.1f}% {milestone_b} before {milestone_a})")
        panel_lines.append("")

    txt = "\n".join(panel_lines)
    ax_inset = plt.gcf().add_axes([0.70, 0.12, 0.28, 0.76])
    ax_inset.axis('off')
    ax_inset.text(0.0, 1.0, txt, va='top', ha='left', fontsize=16, family='monospace', bbox=dict(facecolor=(1,1,1,0.7), edgecolor='0.7'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
