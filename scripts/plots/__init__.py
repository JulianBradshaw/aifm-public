"""
Plot modules for the AI Futures Calculator.

This package contains registry-based plot classes organized by type:
- histograms/: Histogram-based distribution plots
- trajectories/: Time series trajectory plots
- pdfs/: PDF/KDE distribution plots
- scatter/: Scatter and boxplot visualizations
- conditional/: Conditional probability plots
- special/: One-off specialized plots

Usage:
    from plots.registry import PlotRegistry
    from plots.histograms.m_over_beta import MOverBetaHistogramPlot

    # Run a specific plot
    plot = MOverBetaHistogramPlot(run_dir)
    plot.generate()

    # Or via registry
    plot_class = PlotRegistry.get_class("m_over_beta_hist")
    plot_class(run_dir).generate()
"""

from .registry import PlotRegistry, PlotSpec, PlotCategory, PlotSpeed, register_plot
from .base import BasePlot

__all__ = [
    "PlotRegistry",
    "PlotSpec",
    "PlotCategory",
    "PlotSpeed",
    "register_plot",
    "BasePlot",
]
