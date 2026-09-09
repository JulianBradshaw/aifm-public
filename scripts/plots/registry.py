"""
Plot registry for organizing and discovering plot classes.

This module provides a registry pattern for plot classes, enabling:
- Centralized plot organization with metadata
- Easy discovery and listing of available plots
- Speed/category-based filtering for pipeline optimization
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Type

if TYPE_CHECKING:
    from .base import BasePlot


class PlotCategory(Enum):
    """Category of plot for organization purposes."""

    HISTOGRAM = auto()
    PDF = auto()
    TRAJECTORY = auto()
    SCATTER = auto()
    CONDITIONAL = auto()
    SPECIAL = auto()


class PlotSpeed(Enum):
    """Expected execution speed of the plot."""

    FAST = auto()  # Uses cache, < 5 seconds
    MEDIUM = auto()  # 5-30 seconds
    SLOW = auto()  # Full JSONL parsing, > 30 seconds


class CensoredMode(Enum):
    """How the plot handles non-achieved milestones."""

    INCLUDE = auto()  # Include censored data (non-achieved → sim_end)
    EXCLUDE = auto()  # Exclude censored data (only achieved milestones)
    BOTH = auto()  # Generate both variants with different suffixes


@dataclass
class PlotSpec:
    """Metadata specification for a plot class.

    Attributes:
        name: Unique identifier for the plot (e.g., "m_over_beta_hist")
        display_name: Human-readable name for display
        category: Type of plot (histogram, trajectory, etc.)
        speed: Expected execution speed
        output_filename: Default output filename
        output_subdir: Optional subdirectory within run_dir for output
        requires_cache: Whether the plot requires milestones cache
        requires_full_jsonl: Whether the plot needs to parse full rollouts.jsonl
        milestones: List of milestone names this plot uses
        censored_mode: How non-achieved milestones are handled
        description: Optional description of what the plot shows
    """

    name: str
    display_name: str
    category: PlotCategory
    speed: PlotSpeed
    output_filename: str
    output_subdir: Optional[str] = None
    requires_cache: bool = True
    requires_full_jsonl: bool = False
    milestones: List[str] = field(default_factory=list)
    censored_mode: CensoredMode = CensoredMode.INCLUDE
    description: str = ""


class PlotRegistry:
    """Central registry for all plot classes.

    Usage:
        # Register a plot class (usually via @register_plot decorator)
        PlotRegistry.register(spec, MOverBetaHistogramPlot)

        # Get a plot class by name
        plot_class = PlotRegistry.get_class("m_over_beta_hist")
        plot = plot_class(run_dir)
        plot.generate()

        # List plots by category
        histogram_specs = PlotRegistry.list_by_category(PlotCategory.HISTOGRAM)

        # List fast plots for parallel execution
        fast_specs = PlotRegistry.list_by_speed(PlotSpeed.FAST)
    """

    _plots: Dict[str, PlotSpec] = {}
    _plot_classes: Dict[str, Type["BasePlot"]] = {}

    @classmethod
    def register(cls, spec: PlotSpec, plot_class: Type["BasePlot"]) -> None:
        """Register a plot class with its specification.

        Args:
            spec: PlotSpec containing metadata
            plot_class: The plot class to register
        """
        if spec.name in cls._plots:
            raise ValueError(f"Plot '{spec.name}' is already registered")
        cls._plots[spec.name] = spec
        cls._plot_classes[spec.name] = plot_class

    @classmethod
    def get(cls, name: str) -> Optional[PlotSpec]:
        """Get a plot specification by name.

        Args:
            name: The plot identifier

        Returns:
            PlotSpec if found, None otherwise
        """
        return cls._plots.get(name)

    @classmethod
    def get_class(cls, name: str) -> Optional[Type["BasePlot"]]:
        """Get a plot class by name.

        Args:
            name: The plot identifier

        Returns:
            Plot class if found, None otherwise
        """
        return cls._plot_classes.get(name)

    @classmethod
    def list_all(cls) -> List[PlotSpec]:
        """List all registered plot specifications.

        Returns:
            List of all PlotSpec instances
        """
        return list(cls._plots.values())

    @classmethod
    def list_by_category(cls, category: PlotCategory) -> List[PlotSpec]:
        """List plots by category.

        Args:
            category: The PlotCategory to filter by

        Returns:
            List of PlotSpec instances matching the category
        """
        return [spec for spec in cls._plots.values() if spec.category == category]

    @classmethod
    def list_by_speed(cls, speed: PlotSpeed) -> List[PlotSpec]:
        """List plots by speed classification.

        Args:
            speed: The PlotSpeed to filter by

        Returns:
            List of PlotSpec instances matching the speed
        """
        return [spec for spec in cls._plots.values() if spec.speed == speed]

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered plot names.

        Returns:
            List of plot identifiers
        """
        return list(cls._plots.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plots. Useful for testing."""
        cls._plots.clear()
        cls._plot_classes.clear()


def register_plot(spec: PlotSpec):
    """Decorator to register a plot class with the registry.

    Usage:
        @register_plot(PlotSpec(
            name="m_over_beta_hist",
            display_name="m/β Ratio Distribution",
            category=PlotCategory.HISTOGRAM,
            speed=PlotSpeed.FAST,
            output_filename="m_over_beta_hist.png",
        ))
        class MOverBetaHistogramPlot(BasePlot):
            def generate(self, **kwargs) -> Path:
                ...
    """

    def decorator(cls: Type["BasePlot"]) -> Type["BasePlot"]:
        cls.SPEC = spec
        PlotRegistry.register(spec, cls)
        return cls

    return decorator
