"""
Base class for all plot implementations.

Provides common functionality for:
- Run directory and output path management
- Shared RolloutsReader lifecycle
- Standard CLI entry point
- Argument parsing with common options
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from plotting_utils.rollouts_reader import RolloutsReader

    from .registry import PlotSpec


class BasePlot(ABC):
    """Abstract base class for all plot implementations.

    Subclasses must:
    1. Be decorated with @register_plot(PlotSpec(...))
    2. Implement the generate() method

    Example:
        @register_plot(PlotSpec(
            name="m_over_beta_hist",
            display_name="m/β Ratio Distribution",
            category=PlotCategory.HISTOGRAM,
            speed=PlotSpeed.FAST,
            output_filename="m_over_beta_hist.png",
        ))
        class MOverBetaHistogramPlot(BasePlot):
            def generate(self, **kwargs) -> Path:
                data = self.reader.read_m_over_beta_values()
                # ... generate plot ...
                return self.output_path

    Attributes:
        SPEC: PlotSpec set by the @register_plot decorator
        run_dir: Path to the run directory containing rollouts
        _reader: Optional shared RolloutsReader instance
    """

    SPEC: ClassVar["PlotSpec"]

    def __init__(
        self,
        run_dir: Path,
        reader: Optional["RolloutsReader"] = None,
    ):
        """Initialize the plot.

        Args:
            run_dir: Path to run directory containing rollouts.jsonl
            reader: Optional shared RolloutsReader for efficiency.
                    If not provided, a reader will be created lazily.
        """
        self.run_dir = Path(run_dir)
        self._reader = reader
        self._validate_run_dir()

    def _validate_run_dir(self) -> None:
        """Validate that run_dir exists and contains expected files."""
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        rollouts_file = self.run_dir / "rollouts.jsonl"
        cache_file = self.run_dir / "rollouts.cache.json"

        if not rollouts_file.exists() and not cache_file.exists():
            raise FileNotFoundError(
                f"Neither rollouts.jsonl nor rollouts.cache.json found in {self.run_dir}"
            )

    @property
    def reader(self) -> "RolloutsReader":
        """Get or create a RolloutsReader for this plot.

        Returns:
            RolloutsReader instance
        """
        if self._reader is None:
            from plotting_utils.rollouts_reader import RolloutsReader

            rollouts_file = self.run_dir / "rollouts.jsonl"
            self._reader = RolloutsReader(rollouts_file, use_cache=True)
        return self._reader

    @property
    def output_path(self) -> Path:
        """Get the output path for this plot.

        Returns:
            Path where the plot will be saved
        """
        if self.SPEC.output_subdir:
            return self.run_dir / self.SPEC.output_subdir / self.SPEC.output_filename
        return self.run_dir / self.SPEC.output_filename

    def ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(self, **kwargs) -> Path:
        """Generate the plot and save it to output_path.

        Args:
            **kwargs: Plot-specific arguments from CLI or pipeline

        Returns:
            Path to the generated plot file
        """
        ...

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add plot-specific arguments to the parser.

        Override in subclasses to add custom arguments like --milestone,
        --max-year, etc.

        Args:
            parser: ArgumentParser to add arguments to
        """
        pass

    @classmethod
    def create_parser(cls) -> ArgumentParser:
        """Create an argument parser with common and plot-specific args.

        Returns:
            Configured ArgumentParser
        """
        parser = ArgumentParser(
            description=cls.SPEC.description or f"Generate {cls.SPEC.display_name}",
        )

        # Common arguments
        parser.add_argument(
            "--run-dir",
            type=str,
            required=True,
            help="Path to run directory containing rollouts.jsonl",
        )
        parser.add_argument(
            "--out-path",
            type=str,
            default=None,
            help=f"Output path (default: <run-dir>/{cls.SPEC.output_filename})",
        )
        parser.add_argument(
            "--bins",
            type=int,
            default=50,
            help="Number of histogram bins (default: 50)",
        )

        # Allow subclasses to add custom arguments
        cls.add_arguments(parser)

        return parser

    @classmethod
    def main(cls) -> None:
        """Standard CLI entry point for standalone execution.

        Usage in script:
            if __name__ == "__main__":
                MOverBetaHistogramPlot.main()
        """
        parser = cls.create_parser()
        args = parser.parse_args()

        # Create plot instance
        plot = cls(Path(args.run_dir))

        # Override output path if provided
        if args.out_path:
            # Store custom output path for use in generate()
            plot._custom_output_path = Path(args.out_path)

        # Generate the plot
        output = plot.generate(**vars(args))
        print(f"Saved plot to: {output}")

    def get_output_path(self, custom_path: Optional[str | Path] = None) -> Path:
        """Get output path, considering custom override.

        Args:
            custom_path: Optional custom path from CLI args (string or Path)

        Returns:
            Final output path to use
        """
        if custom_path:
            return Path(custom_path)
        if hasattr(self, "_custom_output_path"):
            return self._custom_output_path
        return self.output_path
