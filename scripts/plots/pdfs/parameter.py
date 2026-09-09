#!/usr/bin/env python3
"""
Interactive parameter distribution viewer.

Displays a grid of PDFs for all sampled parameters from a samples.jsonl file.
Supports interactive zoom, pan, and click-to-expand individual plots.

Usage:
    python scripts/parameter_pdfs.py [path/to/samples.jsonl]

If no path is provided, uses the most recent run from outputs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.stats import gaussian_kde


def load_samples(filepath: Path) -> list[dict[str, Any]]:
    """Load samples from a JSONL file."""
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_parameters(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extract all numeric parameters from samples into arrays."""
    if not samples:
        return {}

    # Collect all parameter keys from the first sample
    param_keys: list[str] = []
    ts_keys: list[str] = []

    first = samples[0]
    if "parameters" in first:
        for k, v in first["parameters"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                param_keys.append(k)

    if "time_series_parameters" in first:
        for k, v in first["time_series_parameters"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ts_keys.append(k)

    # Extract values
    result: dict[str, list[float]] = {k: [] for k in param_keys + ts_keys}

    for sample in samples:
        params = sample.get("parameters") or {}
        ts_params = sample.get("time_series_parameters") or {}

        for k in param_keys:
            val = params.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                result[k].append(float(val))

        for k in ts_keys:
            val = ts_params.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                result[k].append(float(val))

    # Convert to arrays, filtering out empty ones
    return {k: np.array(v) for k, v in result.items() if len(v) > 0}


def should_use_log_scale(values: np.ndarray) -> bool:
    """Determine if log scale is appropriate for these values."""
    if values.min() <= 0:
        return False
    return values.max() / values.min() > 100


def compute_histogram_bins(values: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Compute histogram bins, using log scale if appropriate."""
    if should_use_log_scale(values):
        return np.logspace(np.log10(values.min()), np.log10(values.max()), n_bins + 1)
    else:
        return np.linspace(values.min(), values.max(), n_bins + 1)


def compute_kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute KDE for a parameter. Returns (x, density) or None if failed."""
    if len(values) < 2:
        return None

    # Handle constant values
    if np.std(values) < 1e-10:
        return None

    try:
        kde = gaussian_kde(values, bw_method="scott")
        # Use log scale for parameters that span multiple orders of magnitude
        val_range = values.max() - values.min()
        val_min = values.min()

        # Check if log scale is appropriate
        if val_min > 0 and values.max() / val_min > 100:
            # Log scale
            log_vals = np.log10(values)
            kde_log = gaussian_kde(log_vals, bw_method="scott")
            x_log = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 200)
            x = 10**x_log
            # Transform density back
            density = kde_log(x_log) / (x * np.log(10))
        else:
            # Linear scale
            padding = val_range * 0.1 if val_range > 0 else 1.0
            x = np.linspace(val_min - padding, values.max() + padding, 200)
            density = kde(x)

        return x, density
    except (ValueError, np.linalg.LinAlgError):
        return None


class ParameterPDFViewer:
    """Interactive viewer for parameter PDFs."""

    def __init__(self, parameters: dict[str, np.ndarray], filepath: Path):
        self.parameters = parameters
        self.filepath = filepath
        self.param_names = sorted(parameters.keys())
        self.n_params = len(self.param_names)

        # Calculate grid dimensions
        self.ncols = min(4, self.n_params)
        self.nrows = (self.n_params + self.ncols - 1) // self.ncols

        # Create main figure
        self.fig, self.axes = plt.subplots(
            self.nrows,
            self.ncols,
            figsize=(4 * self.ncols, 3 * self.nrows),
            squeeze=False,
        )
        self.fig.suptitle(
            f"Parameter Distributions ({len(next(iter(parameters.values())))} samples)\n{filepath.name}",
            fontsize=12,
        )

        # Track current zoom state
        self.zoomed_param = None
        self.zoomed_fig = None

        # Plot all parameters
        self._plot_all()

        # Connect click handler
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.tight_layout()

    def _format_param_name(self, name: str) -> str:
        """Format parameter name for display."""
        # Truncate long names
        if len(name) > 25:
            return name[:22] + "..."
        return name

    def _plot_single(self, ax: plt.Axes, name: str, show_stats: bool = True):
        """Plot a single parameter distribution with histogram."""
        values = self.parameters[name]
        ax.clear()

        # Determine if we should use log scale on x-axis
        use_log = should_use_log_scale(values)

        # Plot histogram with narrow buckets (50 bins)
        bins = compute_histogram_bins(values, n_bins=50)
        ax.hist(values, bins=bins, density=True, alpha=0.7, color="steelblue",
                edgecolor="white", linewidth=0.3)

        if use_log:
            ax.set_xscale("log")

        ax.set_title(self._format_param_name(name), fontsize=9)
        ax.set_ylabel("Density", fontsize=8)

        if show_stats:
            # Add statistics
            stats_text = (
                f"μ={np.mean(values):.3g}\nσ={np.std(values):.3g}\n"
                f"min={np.min(values):.3g}\nmax={np.max(values):.3g}"
            )
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        ax.tick_params(axis="both", labelsize=7)

    def _plot_all(self):
        """Plot all parameters in the grid."""
        for idx, name in enumerate(self.param_names):
            row = idx // self.ncols
            col = idx % self.ncols
            ax = self.axes[row, col]
            self._plot_single(ax, name)

        # Hide unused axes
        for idx in range(self.n_params, self.nrows * self.ncols):
            row = idx // self.ncols
            col = idx % self.ncols
            self.axes[row, col].set_visible(False)

    def _on_click(self, event):
        """Handle click events to zoom into a single parameter."""
        if event.inaxes is None:
            return

        # Find which parameter was clicked
        for idx, name in enumerate(self.param_names):
            row = idx // self.ncols
            col = idx % self.ncols
            if event.inaxes == self.axes[row, col]:
                self._show_zoomed(name)
                break

    def _show_zoomed(self, name: str):
        """Show a zoomed view of a single parameter."""
        if self.zoomed_fig is not None:
            plt.close(self.zoomed_fig)

        self.zoomed_fig, ax = plt.subplots(figsize=(10, 6))
        self.zoomed_param = name

        values = self.parameters[name]
        use_log = should_use_log_scale(values)

        # Show histogram with narrow buckets (80 bins for zoomed view)
        bins = compute_histogram_bins(values, n_bins=80)
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
            label="Histogram",
        )

        # Optionally overlay KDE
        kde_result = compute_kde(values)
        if kde_result is not None:
            x, density = kde_result
            ax.plot(x, density, color="darkorange", linewidth=2, label="KDE")

        # Add rug plot
        y_min, y_max = ax.get_ylim()
        rug_y = y_min - 0.03 * (y_max - y_min)
        ax.scatter(
            values,
            np.full_like(values, rug_y),
            marker="|",
            color="darkblue",
            alpha=0.3,
            s=30,
        )

        if use_log:
            ax.set_xscale("log")

        ax.set_title(f"{name}", fontsize=14)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()

        # Add detailed statistics
        stats = [
            f"n = {len(values)}",
            f"mean = {np.mean(values):.6g}",
            f"std = {np.std(values):.6g}",
            f"min = {np.min(values):.6g}",
            f"max = {np.max(values):.6g}",
            f"median = {np.median(values):.6g}",
            f"p5 = {np.percentile(values, 5):.6g}",
            f"p95 = {np.percentile(values, 95):.6g}",
        ]
        stats_text = "\n".join(stats)
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

        # Add close button
        ax_button = plt.axes((0.85, 0.01, 0.1, 0.04))
        btn = Button(ax_button, "Close")
        btn.on_clicked(lambda _: plt.close(self.zoomed_fig))

        plt.tight_layout()
        plt.show(block=False)

    def show(self):
        """Display the viewer."""
        plt.show()


def find_latest_samples() -> Path | None:
    """Find the most recent samples.jsonl in outputs/."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    if not outputs_dir.exists():
        return None

    # Find all run directories
    run_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    for run_dir in run_dirs:
        samples_file = run_dir / "samples.jsonl"
        if samples_file.exists():
            return samples_file

    return None


def main():
    """Entry point for the parameter PDF viewer."""
    parser = argparse.ArgumentParser(
        description="Interactive viewer for parameter distributions from Monte Carlo samples."
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to samples.jsonl file. If not provided, uses the most recent run.",
    )
    args = parser.parse_args()

    if args.filepath:
        filepath = Path(args.filepath)
    else:
        filepath = find_latest_samples()
        if filepath is None:
            print("No samples.jsonl found in outputs/. Please provide a path.")
            sys.exit(1)
        print(f"Using: {filepath}")

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    # Load and process samples
    print(f"Loading samples from {filepath}...")
    samples = load_samples(filepath)
    print(f"Loaded {len(samples)} samples")

    parameters = extract_parameters(samples)
    print(f"Found {len(parameters)} numeric parameters")

    if not parameters:
        print("No numeric parameters found in samples.")
        sys.exit(1)

    # Create and show viewer
    viewer = ParameterPDFViewer(parameters, filepath)
    print("\nClick on any plot to see an expanded view with detailed statistics.")
    print("Close the main window to exit.")
    viewer.show()


if __name__ == "__main__":
    main()
