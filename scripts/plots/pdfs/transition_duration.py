#!/usr/bin/env python3
"""
Plot PDF of transition duration between two milestones (e.g., AC -> SC).

This script reads rollout results and creates a probability density function plot
showing the distribution of time between two milestones. It handles censored data
(where the second milestone is not achieved) by treating them as occurring at the
simulation end time.

Usage examples:
  # Plot AC to AI2027-SC duration
  python scripts/plot_transition_duration_pdf.py --rollouts outputs/251109_eli_2200/rollouts.jsonl \
    --from-milestone AC --to-milestone AI2027-SC

  # Plot AC to ASI duration with custom output
  python scripts/plot_transition_duration_pdf.py --rollouts outputs/251109_eli_2200/rollouts.jsonl \
    --from-milestone AC --to-milestone ASI --output ac_to_asi_duration.png

  # Exclude censored data
  python scripts/plot_transition_duration_pdf.py --rollouts outputs/251109_eli_2200/rollouts.jsonl \
    --from-milestone AC --to-milestone AI2027-SC --no-censored
"""

import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

# Import utilities
from plotting_utils.kde import make_gamma_kernel_kde
from plotting_utils.rollouts_reader import RolloutsReader


def plot_transition_duration_pdf(
    durations: List[float],
    num_censored: int,
    from_milestone: str,
    to_milestone: str,
    out_path: Path,
    include_censored: bool = True,
    max_duration: Optional[float] = None,
    present_day: Optional[float] = None,
) -> None:
    """Plot PDF for transition duration using Gamma kernel KDE.

    Args:
        durations: List of transition durations (years)
        num_censored: Number of censored transitions included in durations
        from_milestone: Name of the starting milestone
        to_milestone: Name of the ending milestone
        out_path: Output file path
        include_censored: Whether censored data was included
        max_duration: Maximum duration to plot (default: auto from data)
        present_day: Lower bound for the gamma KDE support (defaults to 0 for durations)
    """
    # Auto-load present_day if not provided (use 0 for durations since they're relative)
    if present_day is None:
        present_day = 0.0

    raw_data = np.asarray(durations, dtype=float)

    # Filter out negative durations (shouldn't happen, but enforce boundary)
    boundary = float(present_day)
    valid_mask = raw_data >= boundary
    data = raw_data[valid_mask]
    dropped = raw_data.size - data.size
    if dropped:
        print(f"  Dropped {dropped} samples before lower bound={boundary:.3f}.")

    if data.size < 2:
        print(f"Error: Not enough data to create KDE after enforcing boundary (need at least 2 points, got {data.size})")
        return

    # Create KDE
    try:
        kde = make_gamma_kernel_kde(data, lower_bound=present_day)

        # Get bandwidth from gamma kernel KDE
        bw = float(kde.bandwidth)

        # Compute evaluation range respecting present_day lower bound
        min_eval = max(present_day, float(np.min(data) - 5.0 * bw))
        max_eval = max(float(np.max(data) + 5.0 * bw), min_eval + bw)

        # Use 512 evenly-spaced points
        xs = np.linspace(min_eval, max_eval, 512)

        # Calculate PDF values
        pdf_values = kde(xs)

        # Cut off plotting at max_duration if specified
        if max_duration is not None:
            plot_mask = xs <= max_duration
            xs_plot = xs[plot_mask]
            pdf_plot = pdf_values[plot_mask]
        else:
            xs_plot = xs
            pdf_plot = pdf_values
    except Exception as e:
        print(f"Error creating KDE: {e}")
        return

    # Calculate statistics
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])
    mean = np.mean(data)

    # Find mode (peak of KDE)
    mode_idx = np.argmax(pdf_values)
    mode = xs[mode_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot PDF
    ax.plot(xs_plot, pdf_plot, linewidth=2.5, color='tab:blue', label='PDF')
    ax.fill_between(xs_plot, pdf_plot, alpha=0.3, color='tab:blue')

    # Add percentile lines
    ax.axvline(q10, color='tab:gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P10: {q10:.3f} yr')
    ax.axvline(q50, color='tab:green', linestyle='-', linewidth=2, alpha=0.7, label=f'Median: {q50:.3f} yr')
    ax.axvline(q90, color='tab:gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P90: {q90:.3f} yr')
    ax.axvline(mode, color='tab:red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mode: {mode:.3f} yr')
    ax.axvline(mean, color='tab:orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mean: {mean:.3f} yr')

    # Add statistics text
    achieved_transitions = (
        len(durations) - num_censored if include_censored else len(durations)
    )
    total_transition_attempts = achieved_transitions + num_censored
    censored_pct = (
        num_censored / total_transition_attempts * 100
        if total_transition_attempts > 0
        else 0
    )
    stats_text = (
        f"Total plotted samples: {len(durations)}\n"
        f"Censored: {num_censored} ({censored_pct:.1f}%)\n"
        f"\n"
        f"Mode: {mode:.3f} yr\n"
        f"Mean: {mean:.3f} yr\n"
        f"P10: {q10:.3f} yr\n"
        f"P50: {q50:.3f} yr\n"
        f"P90: {q90:.3f} yr\n"
    )

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            family='monospace')

    # Labels and title
    from_clean = from_milestone.replace("-level", "").replace("-skill", "")
    to_clean = to_milestone.replace("-level", "").replace("-skill", "")

    censored_label = " (including censored)" if include_censored else " (achieved only)"

    ax.set_xlabel("Duration (years)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"Transition Duration: {from_clean} → {to_clean}{censored_label}", fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)

    if max_duration is not None:
        ax.set_xlim(right=max_duration)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved PDF plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PDF of transition duration between two milestones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        required=True,
        help="Path to rollouts.jsonl file"
    )
    parser.add_argument(
        "--from-milestone",
        type=str,
        required=True,
        help="Starting milestone name (e.g., 'AC')"
    )
    parser.add_argument(
        "--to-milestone",
        type=str,
        required=True,
        help="Ending milestone name (e.g., 'AI2027-SC', 'ASI')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: <rollouts_dir>/<from>_to_<to>_duration.png)"
    )
    parser.add_argument(
        "--no-censored",
        action="store_true",
        help="Exclude censored transitions (only include when both milestones achieved)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum duration to plot (default: auto from data)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    include_censored = not args.no_censored

    # Read transition durations using RolloutsReader
    reader = RolloutsReader(rollouts_path)
    durations, num_censored, sim_end, num_out_of_order = reader.read_transition_durations(
        args.from_milestone,
        args.to_milestone,
        include_censored
    )

    # Print statistics
    total_rollouts = reader.count_rollouts()
    achieved_transitions = len(durations) - num_censored if include_censored else len(durations)
    num_from_not_achieved = total_rollouts - achieved_transitions - num_censored
    print(f"From '{args.from_milestone}' to '{args.to_milestone}':")
    print(f"  - Achieved transitions: {achieved_transitions}")
    print(f"  - Censored transitions: {num_censored}")
    print(f"  - '{args.from_milestone}' not achieved: {num_from_not_achieved}")
    print(f"  - Total valid durations: {len(durations)}")
    if num_out_of_order:
        print(
            f"  - Warning: {num_out_of_order} rollouts reached '{args.to_milestone}' before '{args.from_milestone}' "
            "and were excluded."
        )

    if not durations:
        print("No valid transition durations found!")
        return

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        from_safe = args.from_milestone.replace("/", "_").replace(" ", "_")
        to_safe = args.to_milestone.replace("/", "_").replace(" ", "_")
        censored_suffix = "_censored" if include_censored else ""
        out_path = rollouts_path.parent / f"{from_safe}_to_{to_safe}_duration{censored_suffix}.png"

    # Generate plot
    plot_transition_duration_pdf(
        durations,
        num_censored,
        args.from_milestone,
        args.to_milestone,
        out_path,
        include_censored=include_censored,
        max_duration=args.max_duration
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
