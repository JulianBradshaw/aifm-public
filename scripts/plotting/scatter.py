"""Scatter plot utilities for milestone analysis."""

import json
from pathlib import Path
from typing import Optional
import sys

import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load parameter display names
_DISPLAY_NAMES_PATH = Path(__file__).parent.parent.parent / "config" / "parameter_display_names.json"
_DISPLAY_NAMES: dict[str, str] = {}
if _DISPLAY_NAMES_PATH.exists():
    with open(_DISPLAY_NAMES_PATH, "r") as f:
        _DISPLAY_NAMES = json.load(f)


def _get_display_name(param: str) -> str:
    """Get display name for a parameter, falling back to the parameter name itself."""
    return _DISPLAY_NAMES.get(param, param)

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

from plotting_utils.helpers import get_year_tick_values_and_labels


def plot_milestone_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    out_path: Path,
    title: Optional[str] = None,
    kind: str = "hex",
    gridsize: int = 50,
    point_size: float = 8.0,
    scatter_overlay: bool = True,
    ymin_years: Optional[float] = None,
    ymax_years: Optional[float] = None,
    condition_text: Optional[str] = None,
) -> None:
    """Plot milestone scatter plot showing date vs transition duration.

    Args:
        xs: X values (milestone dates in decimal years)
        ys: Y values (transition durations in years)
        out_path: Output file path
        title: Plot title
        kind: Plot kind ("hex", "hist2d", or "scatter")
        gridsize: Grid size for hex/hist2d
        point_size: Point size for scatter overlay
        scatter_overlay: Whether to overlay scatter on hex/hist2d
        ymin_years: Minimum Y value (years)
        ymax_years: Maximum Y value (years)
        condition_text: Optional condition text to display
    """
    if xs.size == 0 or ys.size == 0:
        raise ValueError("No data to plot for milestone scatter")

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    if kind == "hex":
        hb = ax.hexbin(xs, ys, gridsize=int(gridsize), xscale='linear', yscale='log', cmap='viridis', mincnt=1)
        cb = plt.colorbar(hb)
        cb.set_label('Count')
    elif kind == "hist2d":
        h, xedges, yedges, img = ax.hist2d(xs, ys, bins=int(gridsize), cmap='viridis', norm=None)
        ax.set_yscale('log')
        cb = plt.colorbar(img)
        cb.set_label('Count')
    else:
        ax.set_yscale('log')
        ax.scatter(xs, ys, s=float(point_size), alpha=0.6, color='tab:blue')

    if kind in ("hex", "hist2d") and scatter_overlay:
        ax.scatter(xs, ys, s=float(point_size), alpha=0.3, color='black')

    # Y-axis bounds and ticks similar to other plots
    ymin = float(ymin_years) if ymin_years is not None else max(ys[ys > 0].min(), 1e-3)
    ymax = float(ymax_years) if ymax_years is not None else max(ys.max(), ymin * 10)
    ax.set_ylim(ymin, ymax)
    ticks, labels = get_year_tick_values_and_labels(ymin, ymax)
    ax.set_yscale('log')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.set_xlabel("From milestone date (decimal year)")
    ax.set_ylabel("Duration to target milestone (years, log scale)")
    ax.grid(True, which='both', axis='y', alpha=0.25)
    ax.set_title(title or "Milestone date vs transition duration")

    if condition_text:
        ax_inset = plt.gcf().add_axes([0.72, 0.12, 0.26, 0.2])
        ax_inset.axis('off')
        ax_inset.text(0.0, 1.0, f"Condition: {condition_text}", va='top', ha='left', fontsize=12, family='monospace', bbox=dict(facecolor=(1,1,1,0.7), edgecolor='0.7'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    out_path: Path,
    param_x: str,
    param_y: str,
    target_corr: float,
    use_ranks: bool = False,
    corr_type: str = "pearson",
    show_actual_corr: bool = True,
    x_dist: Optional[str] = None,
    y_dist: Optional[str] = None,
) -> None:
    """Create scatter plot for two correlated parameters.

    Args:
        x_values: array of x parameter values
        y_values: array of y parameter values
        out_path: output file path
        param_x: name of x parameter
        param_y: name of y parameter
        target_corr: target correlation coefficient
        use_ranks: if True, plot ranks instead of values
        corr_type: correlation type from config (e.g., 'spearman')
        show_actual_corr: if True, display actual correlation in text box
        x_dist: distribution type for x parameter (for log scale)
        y_dist: distribution type for y parameter (for log scale)
    """
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("No data to plot")

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have same length")

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Get display names for parameters
    display_x = _get_display_name(param_x)
    display_y = _get_display_name(param_y)

    if use_ranks:
        # Convert to ranks for Spearman-style visualization
        x_plot = rankdata(x_values)
        y_plot = rankdata(y_values)
        xlabel = f"{display_x} (rank)"
        ylabel = f"{display_y} (rank)"
        title_suffix = "Ranks"
    else:
        x_plot = x_values
        y_plot = y_values
        xlabel = display_x
        ylabel = display_y
        title_suffix = "Values"

    # Create scatter plot with some transparency
    ax.scatter(x_plot, y_plot, alpha=0.5, s=20, color='tab:blue')

    # Calculate actual correlation (only if needed for display)
    if show_actual_corr:
        if use_ranks:
            actual_corr, _ = spearmanr(x_values, y_values)
        else:
            actual_corr = float(np.corrcoef(x_values, y_values)[0, 1])

        # Add text box with correlation info
        textstr = (
            f'Target {corr_type} correlation: {target_corr:.2f}\n'
            f'Actual correlation: {actual_corr:.3f}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace'
        )
    else:
        # Just show target correlation
        textstr = f'Target {corr_type} correlation: {target_corr:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace'
        )

    # Set log scale for lognormal parameters
    if not use_ranks:
        if x_dist in ['lognormal', 'shifted_lognormal'] and np.all(x_plot > 0):
            ax.set_xscale('log')
        if y_dist in ['lognormal', 'shifted_lognormal'] and np.all(y_plot > 0):
            ax.set_yscale('log')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{display_x} vs {display_y}\n({title_suffix})", fontsize=14)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
