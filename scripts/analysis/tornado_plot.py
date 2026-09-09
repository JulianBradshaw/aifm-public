#!/usr/bin/env python3
"""
Generate tornado plots for parameter sensitivity analysis.

This script shows how individual parameters affect a target statistic when varied
from their 10th to 90th percentile values while all other parameters are held at
their median values.

Usage:
  # Milestone arrival time sensitivity
  python scripts/analysis/tornado_plot.py \
    --config config/sampling_config.yaml \
    --milestone AC \
    --top-k 15

  # Metric at specific time
  python scripts/analysis/tornado_plot.py \
    --config config/sampling_config.yaml \
    --metric horizon_lengths \
    --at-time 2030

  # Transition duration between milestones
  python scripts/analysis/tornado_plot.py \
    --config config/sampling_config.yaml \
    --transition AC:SAR-level-experiment-selection-skill
"""

import argparse
import contextlib
import io
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import sys

import numpy as np
import scipy.special

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

scripts_dir = Path(__file__).resolve().parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from plotting_utils.style import apply_style
apply_style()

from progress_model import config as cfg
from progress_model import ProgressModel, Parameters, load_time_series_data

# Try to import YAML, fall back gracefully
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Distribution types that are numeric/continuous (can be sampled at quantiles)
NUMERIC_DIST_TYPES = {"lognormal", "normal", "uniform", "beta", "shifted_lognormal"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate tornado plots for parameter sensitivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sampling config file (YAML or JSON)",
    )

    # Statistic type - mutually exclusive group
    stat_group = parser.add_argument_group("target statistic (choose one)")
    stat_group.add_argument(
        "--milestone",
        type=str,
        default=None,
        help="Milestone name for arrival time sensitivity (e.g., AC, SAR-level-experiment-selection-skill)",
    )
    stat_group.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric name for value-at-time sensitivity (requires --at-time)",
    )
    stat_group.add_argument(
        "--at-time",
        type=float,
        default=None,
        help="Year at which to evaluate metric (requires --metric)",
    )
    stat_group.add_argument(
        "--transition",
        type=str,
        default=None,
        help="Transition duration between milestones, format: START:END (e.g., AC:SAR-level-experiment-selection-skill)",
    )

    # Output options
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only show top K parameters by impact",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (defaults to tornado_<statistic>.png in config directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--show-vals",
        action="store_true",
        help="Show parameter values (10th/90th percentile) at the end of each bar",
    )

    args = parser.parse_args()

    # Validate statistic arguments
    num_stats = sum([
        args.milestone is not None,
        args.metric is not None,
        args.transition is not None,
    ])

    if num_stats == 0:
        parser.error("Must specify one of: --milestone, --metric (with --at-time), or --transition")
    if num_stats > 1:
        parser.error("Cannot specify more than one statistic type")

    if args.metric is not None and args.at_time is None:
        parser.error("--metric requires --at-time to specify the year")
    if args.at_time is not None and args.metric is None:
        parser.error("--at-time requires --metric to specify which metric")

    if args.transition is not None:
        if ":" not in args.transition:
            parser.error("--transition must be in format START:END (e.g., AC:SAR-level-experiment-selection-skill)")

    return args


def _deep_merge_configs(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dicts, with child values overriding parent."""
    result = parent.copy()
    for key, child_val in child.items():
        if child_val is None:
            # None in child means "keep parent value"
            continue
        if key in result and isinstance(result[key], dict) and isinstance(child_val, dict):
            # Special case: empty dict in child replaces parent
            if len(child_val) == 0:
                result[key] = {}
            else:
                result[key] = _deep_merge_configs(result[key], child_val)
        else:
            result[key] = child_val
    return result


def load_config(config_path: str, _visited: Optional[Set[Path]] = None) -> Dict[str, Any]:
    """Load config with support for inheritance via 'parent' field."""
    p = Path(config_path).resolve()

    if _visited is None:
        _visited = set()
    if p in _visited:
        raise ValueError(f"Circular config inheritance detected: {p}")
    _visited.add(p)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML configs. Install with: pip install pyyaml")
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)

    # Handle inheritance
    if config and "parent" in config:
        parent_path = config.pop("parent")
        if not Path(parent_path).is_absolute():
            parent_path = (p.parent / parent_path).resolve()
        else:
            parent_path = Path(parent_path).resolve()

        parent_config = load_config(str(parent_path), _visited)
        config = _deep_merge_configs(parent_config, config)

    return config


def get_numeric_parameters(param_dists: Dict[str, Any]) -> List[str]:
    """Get list of parameters with numeric (non-categorical) distributions."""
    numeric_params = []
    for name, spec in param_dists.items():
        if not isinstance(spec, dict):
            continue
        dist_type = spec.get("dist", "fixed")
        if dist_type in NUMERIC_DIST_TYPES:
            numeric_params.append(name)
    return sorted(numeric_params)


def sample_at_quantile(dist_spec: Dict[str, Any], quantile: float, param_name: str) -> float:
    """Sample from a distribution at a specific quantile."""
    kind = dist_spec.get("dist", "fixed")

    if kind == "fixed":
        return float(dist_spec.get("value", 0.0))

    if kind == "uniform":
        a = float(dist_spec["min"])
        b = float(dist_spec["max"])
        return a + quantile * (b - a)

    if kind == "normal":
        # Support ci80 or mean/sd parameterization
        if "ci80" in dist_spec:
            ci = dist_spec["ci80"]
            q10, q90 = float(ci[0]), float(ci[1])
            if q10 > q90:
                q10, q90 = q90, q10
        elif "ci80_low" in dist_spec and "ci80_high" in dist_spec:
            q10 = float(dist_spec["ci80_low"])
            q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
        else:
            mu = float(dist_spec["mean"])
            sigma = float(dist_spec.get("sd", dist_spec.get("sigma", 1.0)))
            x = mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1)
            if "min" in dist_spec and "max" in dist_spec:
                x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
            return x

        z = 1.2815515655446004  # 80% CI z-score
        mu = 0.5 * (q10 + q90)
        sigma = (q90 - q10) / (2.0 * z)
        x = mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1)
        if "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "lognormal":
        if "ci80" in dist_spec:
            ci = dist_spec["ci80"]
            q10, q90 = float(ci[0]), float(ci[1])
            if q10 > q90:
                q10, q90 = q90, q10
        elif "ci80_low" in dist_spec and "ci80_high" in dist_spec:
            q10 = float(dist_spec["ci80_low"])
            q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])
            x = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
            if "min" in dist_spec and "max" in dist_spec:
                x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
            return x

        z = 1.2815515655446004
        ln_q10, ln_q90 = float(np.log(q10)), float(np.log(q90))
        mu = 0.5 * (ln_q10 + ln_q90)
        sigma = (ln_q90 - ln_q10) / (2.0 * z)
        x = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        if "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "shifted_lognormal":
        shift = float(dist_spec.get("shift", 0.0))
        if "ci80" in dist_spec:
            ci = dist_spec["ci80"]
            q10, q90 = float(ci[0]), float(ci[1])
            if q10 > q90:
                q10, q90 = q90, q10
        elif "ci80_low" in dist_spec and "ci80_high" in dist_spec:
            q10 = float(dist_spec["ci80_low"])
            q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])
            x_core = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
            x = shift + x_core
            if "min" in dist_spec and "max" in dist_spec:
                x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
            return x

        z = 1.2815515655446004
        ln_q10, ln_q90 = float(np.log(q10)), float(np.log(q90))
        mu = 0.5 * (ln_q10 + ln_q90)
        sigma = (ln_q90 - ln_q10) / (2.0 * z)
        x_core = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        x = shift + x_core
        if "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "beta":
        a = float(dist_spec["alpha"])
        b = float(dist_spec["beta"])
        lo = float(dist_spec.get("min", 0.0))
        hi = float(dist_spec.get("max", 1.0))
        x01 = scipy.special.betaincinv(a, b, quantile)
        return lo + (hi - lo) * x01

    raise ValueError(f"Unknown distribution kind: {kind} for parameter {param_name}")


def clip_to_bounds(param_name: str, value: float) -> float:
    """Clip parameter value to model bounds if defined."""
    bounds = cfg.PARAMETER_BOUNDS.get(param_name)
    if bounds is None:
        return value
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


def build_parameter_dict(
    param_dists: Dict[str, Any],
    quantiles: Dict[str, float],
    default_quantile: float = 0.5,
    apply_model_defaults: bool = True,
) -> Dict[str, Any]:
    """Build parameter dict with specified quantiles for each parameter.

    apply_model_defaults injects the categorical MODEL defaults. It must be False
    when building a *time series* parameter dict: those two keys are model-level,
    and injecting them made an empty time_series_parameters section produce a
    non-empty dict, which then read as truthy in run_trajectory and routed into
    generate_time_series_from_dict -- which promptly KeyError'd on the time
    series keys that were never there. run_trajectory swallows the exception, so
    this surfaced only as "Could not compute baseline statistic".
    """
    params = {}
    for name, spec in param_dists.items():
        if not isinstance(spec, dict):
            continue

        dist_type = spec.get("dist", "fixed")

        if dist_type == "fixed":
            params[name] = spec.get("value")
        elif dist_type == "choice":
            # For categorical, use most probable value (mode)
            values = spec["values"]
            probs = spec.get("p")
            if probs is None:
                params[name] = values[0]  # First value if uniform
            else:
                params[name] = values[np.argmax(probs)]
        elif dist_type in NUMERIC_DIST_TYPES:
            q = quantiles.get(name, default_quantile)
            val = sample_at_quantile(spec, q, name)
            val = clip_to_bounds(name, val)
            params[name] = val
        else:
            # Unknown distribution, skip
            continue

    # Ensure categorical defaults (model params only -- see docstring)
    if apply_model_defaults:
        params.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
        params.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)

    return params


@contextlib.contextmanager
def suppress_noise():
    """Suppress warnings, logging, and stdout/stderr."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            logging.disable(prev_disable)


def run_trajectory(
    model_params: Dict[str, Any],
    ts_params: Dict[str, Any],
    input_data_path: str,
    time_range: List[float],
    initial_progress: float,
) -> Optional[Dict[str, Any]]:
    """Run a single trajectory and return results."""
    try:
        # Generate time series with parameters if provided
        if ts_params:
            from progress_model.time_series import generate_time_series_from_dict
            base_data = load_time_series_data(input_data_path)
            rollout_data = generate_time_series_from_dict(ts_params, input_data_path, base_data)
        else:
            rollout_data = load_time_series_data(input_data_path)

        params_obj = Parameters(**model_params)
        model = ProgressModel(params_obj, rollout_data)

        with suppress_noise():
            model.compute_progress_trajectory(time_range, initial_progress)

        return model.results
    except Exception:
        return None


def extract_statistic(
    results: Optional[Dict[str, Any]],
    stat_type: str,
    stat_args: Dict[str, Any],
) -> Optional[float]:
    """Extract the target statistic from results.

    Args:
        results: Model results dict
        stat_type: One of 'milestone', 'metric', 'transition'
        stat_args: Dict with keys like 'milestone_name', 'metric_name', 'at_time',
                   'start_milestone', 'end_milestone'

    Returns:
        Extracted statistic value, or None if not available
    """
    if results is None:
        return None

    if stat_type == "milestone":
        milestone_name = stat_args["milestone_name"]
        milestones = results.get("milestones", {})
        if milestone_name not in milestones:
            return None
        milestone = milestones[milestone_name]
        return milestone.get("time")

    elif stat_type == "metric":
        metric_name = stat_args["metric_name"]
        at_time = stat_args["at_time"]

        times = results.get("times")
        metric_vals = results.get(metric_name)

        if times is None or metric_vals is None:
            return None

        times = np.array(times)
        metric_vals = np.array(metric_vals)

        if at_time < times[0] or at_time > times[-1]:
            return None

        return float(np.interp(at_time, times, metric_vals))

    elif stat_type == "transition":
        start_name = stat_args["start_milestone"]
        end_name = stat_args["end_milestone"]
        milestones = results.get("milestones", {})

        if start_name not in milestones or end_name not in milestones:
            return None

        start_time = milestones[start_name].get("time")
        end_time = milestones[end_name].get("time")

        if start_time is None or end_time is None:
            return None

        return end_time - start_time

    return None


def resolve_input_data_path(config_input_data: str, config_path: Path) -> str:
    """Resolve input_data path, handling absolute paths from different machines.

    Tries in order:
    1. Original path if it exists and is accessible
    2. Path relative to config file's directory
    3. Same filename in repo's inputs/ directory
    4. Default inputs/input_data.csv
    """
    repo_root = Path(__file__).resolve().parent.parent.parent

    def path_exists_safely(p: Path) -> bool:
        """Check if path exists without raising permission errors."""
        try:
            return p.exists()
        except (PermissionError, OSError):
            return False

    # Try original path
    if path_exists_safely(Path(config_input_data)):
        return config_input_data

    # Try relative to config directory
    config_dir = config_path.parent
    relative_path = config_dir / Path(config_input_data).name
    if path_exists_safely(relative_path):
        return str(relative_path)

    # Try in repo's inputs/ directory
    filename = Path(config_input_data).name
    inputs_path = repo_root / "inputs" / filename
    if path_exists_safely(inputs_path):
        return str(inputs_path)

    # Fall back to default
    default_path = repo_root / "inputs" / "input_data.csv"
    if path_exists_safely(default_path):
        return str(default_path)

    # Return original and let it fail with a clear error
    return config_input_data


def compute_sensitivities(
    config: Dict[str, Any],
    stat_type: str,
    stat_args: Dict[str, Any],
    config_path: Path,
    verbose: bool = False,
) -> Tuple[float, List[Tuple[str, float, float, float, float]]]:
    """Compute parameter sensitivities.

    Returns:
        Tuple of (baseline_value, list of (param_name, low_val, high_val, param_low, param_high))
    """
    param_dists = config.get("parameters", {})
    ts_param_dists = config.get("time_series_parameters", {})

    raw_input_data_path = config.get("input_data", "inputs/input_data.csv")
    input_data_path = resolve_input_data_path(raw_input_data_path, config_path)
    if verbose:
        print(f"Using input data: {input_data_path}")

    time_range = config.get("time_range", [2017.0, 2050.0])
    initial_progress = config.get("initial_progress", 0.0)

    # Get numeric parameters from both regular and time series params
    numeric_params = get_numeric_parameters(param_dists)
    numeric_ts_params = get_numeric_parameters(ts_param_dists)

    if verbose:
        print(f"Found {len(numeric_params)} numeric model parameters")
        print(f"Found {len(numeric_ts_params)} numeric time series parameters")

    # Compute baseline with all medians
    if verbose:
        print("\nComputing baseline trajectory (all parameters at median)...")

    baseline_model_params = build_parameter_dict(param_dists, {}, default_quantile=0.5)
    baseline_ts_params = build_parameter_dict(ts_param_dists, {}, default_quantile=0.5, apply_model_defaults=False)

    baseline_results = run_trajectory(
        baseline_model_params, baseline_ts_params, input_data_path, time_range, initial_progress
    )
    baseline_value = extract_statistic(baseline_results, stat_type, stat_args)

    if baseline_value is None:
        raise ValueError(f"Could not compute baseline statistic. Check that the milestone/metric exists.")

    if verbose:
        print(f"Baseline value: {baseline_value:.4f}")

    sensitivities = []

    # Compute sensitivities for model parameters
    total_params = len(numeric_params) + len(numeric_ts_params)
    processed = 0

    for param_name in numeric_params:
        processed += 1
        if verbose:
            print(f"  [{processed}/{total_params}] Testing {param_name}...")

        spec = param_dists[param_name]

        # Get parameter values at 10th and 90th percentile
        param_low = clip_to_bounds(param_name, sample_at_quantile(spec, 0.1, param_name))
        param_high = clip_to_bounds(param_name, sample_at_quantile(spec, 0.9, param_name))

        # Run with low value
        low_model_params = build_parameter_dict(param_dists, {param_name: 0.1}, default_quantile=0.5)
        low_results = run_trajectory(
            low_model_params, baseline_ts_params, input_data_path, time_range, initial_progress
        )
        low_val = extract_statistic(low_results, stat_type, stat_args)

        # Run with high value
        high_model_params = build_parameter_dict(param_dists, {param_name: 0.9}, default_quantile=0.5)
        high_results = run_trajectory(
            high_model_params, baseline_ts_params, input_data_path, time_range, initial_progress
        )
        high_val = extract_statistic(high_results, stat_type, stat_args)

        if low_val is not None and high_val is not None:
            sensitivities.append((param_name, low_val, high_val, param_low, param_high))

    # Compute sensitivities for time series parameters
    for param_name in numeric_ts_params:
        processed += 1
        if verbose:
            print(f"  [{processed}/{total_params}] Testing ts:{param_name}...")

        spec = ts_param_dists[param_name]

        # Get parameter values at 10th and 90th percentile
        param_low = sample_at_quantile(spec, 0.1, param_name)
        param_high = sample_at_quantile(spec, 0.9, param_name)

        # Run with low value
        low_ts_params = build_parameter_dict(ts_param_dists, {param_name: 0.1}, default_quantile=0.5, apply_model_defaults=False)
        low_results = run_trajectory(
            baseline_model_params, low_ts_params, input_data_path, time_range, initial_progress
        )
        low_val = extract_statistic(low_results, stat_type, stat_args)

        # Run with high value
        high_ts_params = build_parameter_dict(ts_param_dists, {param_name: 0.9}, default_quantile=0.5, apply_model_defaults=False)
        high_results = run_trajectory(
            baseline_model_params, high_ts_params, input_data_path, time_range, initial_progress
        )
        high_val = extract_statistic(high_results, stat_type, stat_args)

        if low_val is not None and high_val is not None:
            sensitivities.append((f"ts:{param_name}", low_val, high_val, param_low, param_high))

    return baseline_value, sensitivities


# Cache for display names
_display_names_cache: Optional[Dict[str, str]] = None


def load_display_names() -> Dict[str, str]:
    """Load parameter display names from config file."""
    global _display_names_cache
    if _display_names_cache is not None:
        return _display_names_cache

    display_names_path = Path(__file__).resolve().parent.parent.parent / "config" / "parameter_display_names.json"
    if display_names_path.exists():
        with open(display_names_path) as f:
            _display_names_cache = json.load(f)
    else:
        _display_names_cache = {}

    return _display_names_cache


def simplify_param_name(name: str, max_len: int = 45) -> str:
    """Get display name for parameter, falling back to simplified version."""
    display_names = load_display_names()

    # Check for display name first
    if name in display_names:
        display_name = display_names[name]
        if len(display_name) > max_len:
            return display_name[:max_len-2] + ".."
        return display_name

    # Fall back to simplified name
    simplified = name.replace("_at_present_day", "@pd")
    simplified = simplified.replace("_at_coding_automation_anchor", "@ac")
    simplified = simplified.replace("_experiment_selection_skill", "_ess")
    simplified = simplified.replace("experiment_", "exp_")
    simplified = simplified.replace("automation_", "auto_")
    simplified = simplified.replace("multiplier", "mult")
    simplified = simplified.replace("training_compute", "train_comp")

    if len(simplified) > max_len:
        simplified = simplified[:max_len-2] + ".."

    return simplified


def create_tornado_plot(
    baseline: float,
    sensitivities: List[Tuple[str, float, float, float, float]],
    stat_label: str,
    stat_title: str,
    output_path: Path,
    top_k: Optional[int] = None,
    show_vals: bool = False,
) -> None:
    """Create and save tornado plot.

    Args:
        baseline: Baseline statistic value
        sensitivities: List of (param_name, low_val, high_val, param_low, param_high)
        stat_label: Label for x-axis (e.g., "AC Arrival Time (years)")
        stat_title: Chart title (e.g., "How does each parameter change the AC date?")
        output_path: Path to save figure
        top_k: If set, only show top K parameters by impact
        show_vals: If True, show parameter values at the end of each bar
    """
    if not sensitivities:
        print("No sensitivities to plot")
        return

    # Sort by absolute spread (largest impact first)
    sensitivities_sorted = sorted(
        sensitivities,
        key=lambda x: abs(x[2] - x[1]),  # |high_val - low_val|
        reverse=True,
    )

    # Limit to top K if specified
    if top_k is not None and top_k > 0:
        sensitivities_sorted = sensitivities_sorted[:top_k]

    # Reverse for plotting (largest at top)
    sensitivities_sorted = list(reversed(sensitivities_sorted))

    n_params = len(sensitivities_sorted)

    # Create figure
    fig_height = max(6, 0.4 * n_params + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = np.arange(n_params)

    # Colors: blue = 10th percentile param value, orange = 90th percentile param value
    color_p10 = "#3b82f6"   # Blue for 10th percentile
    color_p90 = "#f97316"   # Orange for 90th percentile

    def format_param_val(val: float) -> str:
        """Format parameter value for display."""
        if abs(val) >= 10000:
            return f"{val:.1e}"
        elif abs(val) >= 100:
            return f"{val:.0f}"
        elif abs(val) >= 1:
            return f"{val:.1f}"
        elif abs(val) >= 0.01:
            return f"{val:.2f}"
        else:
            return f"{val:.1e}"

    for i, (param_name, low_val, high_val, param_low, param_high) in enumerate(sensitivities_sorted):
        y = y_positions[i]
        # low_val = statistic when param is at 10th percentile (param_low)
        # high_val = statistic when param is at 90th percentile (param_high)
        # Colors are based on percentile, not direction

        # Draw bar for 10th percentile result (blue)
        ax.barh(y, low_val - baseline, left=baseline, height=0.6, color=color_p10, alpha=0.8)
        # Draw bar for 90th percentile result (orange)
        ax.barh(y, high_val - baseline, left=baseline, height=0.6, color=color_p90, alpha=0.8)

        # Add value annotations if requested
        if show_vals:
            # Position text at the end of each bar
            if low_val <= baseline:
                ax.text(low_val, y, f"{format_param_val(param_low)} ", va="center", ha="right", fontsize=7, color=color_p10)
            else:
                ax.text(low_val, y, f" {format_param_val(param_low)}", va="center", ha="left", fontsize=7, color=color_p10)

            if high_val <= baseline:
                ax.text(high_val, y, f"{format_param_val(param_high)} ", va="center", ha="right", fontsize=7, color=color_p90)
            else:
                ax.text(high_val, y, f" {format_param_val(param_high)}", va="center", ha="left", fontsize=7, color=color_p90)

    # Draw baseline vertical line
    ax.axvline(baseline, color="black", linewidth=2, linestyle="-", zorder=10)

    # Set y-axis labels
    param_labels = [simplify_param_name(s[0]) for s in sensitivities_sorted]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(param_labels, fontsize=9)

    # Add parameter value annotations on right side
    x_max = ax.get_xlim()[1]
    for i, (param_name, low_val, high_val, param_low, param_high) in enumerate(sensitivities_sorted):
        # Format parameter values
        if abs(param_low) >= 1000 or abs(param_high) >= 1000:
            annotation = f"[{param_low:.1e}, {param_high:.1e}]"
        elif abs(param_low) < 0.01 or abs(param_high) < 0.01:
            annotation = f"[{param_low:.2e}, {param_high:.2e}]"
        else:
            annotation = f"[{param_low:.2f}, {param_high:.2f}]"

    # Labels and title
    ax.set_xlabel(stat_label, fontsize=11)
    ax.set_ylabel("")
    ax.set_title(stat_title, fontsize=13, fontweight="bold", pad=35)

    # Add baseline value annotation
    ax.annotate(
        f"Baseline: {baseline:.2f}",
        xy=(baseline, n_params - 0.5),
        xytext=(baseline, n_params + 0.3),
        fontsize=10,
        ha="center",
        va="bottom",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_p10, alpha=0.8, label="10th percentile"),
        Patch(facecolor=color_p90, alpha=0.8, label="90th percentile"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved tornado plot to: {output_path}")


def get_stat_label(stat_type: str, stat_args: Dict[str, Any]) -> str:
    """Get human-readable label for the statistic (used for x-axis)."""
    if stat_type == "milestone":
        return f"{stat_args['milestone_name']} Arrival Time (year)"
    elif stat_type == "metric":
        return f"{stat_args['metric_name']} @ {stat_args['at_time']:.1f}"
    elif stat_type == "transition":
        return f"{stat_args['start_milestone']} to {stat_args['end_milestone']} Duration (years)"
    return "Statistic"


def get_stat_title(stat_type: str, stat_args: Dict[str, Any]) -> str:
    """Get chart title describing the statistic being analyzed."""
    if stat_type == "milestone":
        return f"How does the {stat_args['milestone_name']} date change as each parameter is individually varied?"
    elif stat_type == "metric":
        return f"How does {stat_args['metric_name']} at {stat_args['at_time']:.0f} change as each parameter is individually varied?"
    elif stat_type == "transition":
        return f"How does the {stat_args['start_milestone']} to {stat_args['end_milestone']} duration change as each parameter is individually varied?"
    return "Parameter Sensitivity Analysis"


def get_output_filename(stat_type: str, stat_args: Dict[str, Any]) -> str:
    """Get default output filename based on statistic type."""
    if stat_type == "milestone":
        name = stat_args['milestone_name'].replace("-", "_").lower()
        return f"tornado_{name}.png"
    elif stat_type == "metric":
        name = stat_args['metric_name']
        time = int(stat_args['at_time'])
        return f"tornado_{name}_at_{time}.png"
    elif stat_type == "transition":
        start = stat_args['start_milestone'].replace("-", "_").lower()[:10]
        end = stat_args['end_milestone'].replace("-", "_").lower()[:10]
        return f"tornado_{start}_to_{end}.png"
    return "tornado_plot.png"


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    # Determine statistic type and arguments
    if args.milestone:
        stat_type = "milestone"
        stat_args = {"milestone_name": args.milestone}
    elif args.metric:
        stat_type = "metric"
        stat_args = {"metric_name": args.metric, "at_time": args.at_time}
    else:  # args.transition
        parts = args.transition.split(":", 1)
        stat_type = "transition"
        stat_args = {"start_milestone": parts[0], "end_milestone": parts[1]}

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config_path.parent / get_output_filename(stat_type, stat_args)

    stat_label = get_stat_label(stat_type, stat_args)
    stat_title = get_stat_title(stat_type, stat_args)
    print(f"Target statistic: {stat_label}")

    # Compute sensitivities
    print(f"\nComputing parameter sensitivities...")
    baseline, sensitivities = compute_sensitivities(
        config, stat_type, stat_args, config_path, verbose=args.verbose
    )

    print(f"\nBaseline value: {baseline:.4f}")
    print(f"Parameters with valid sensitivities: {len(sensitivities)}")

    # Filter out parameters with zero spread
    sensitivities = [(n, l, h, pl, ph) for n, l, h, pl, ph in sensitivities if abs(h - l) > 1e-9]
    print(f"Parameters with non-zero impact: {len(sensitivities)}")

    if not sensitivities:
        print("No parameters have measurable impact on this statistic.")
        return

    # Create tornado plot
    create_tornado_plot(
        baseline=baseline,
        sensitivities=sensitivities,
        stat_label=stat_label,
        stat_title=stat_title,
        output_path=output_path,
        top_k=args.top_k,
        show_vals=args.show_vals,
    )

    # Print top impacts
    sorted_sens = sorted(sensitivities, key=lambda x: abs(x[2] - x[1]), reverse=True)
    print(f"\nTop parameter impacts:")
    for i, (name, low, high, pl, ph) in enumerate(sorted_sens[:10]):
        spread = abs(high - low)
        print(f"  {i+1:2d}. {name:40s}: spread = {spread:.4f} ({low:.3f} to {high:.3f})")

    print("\nDone!")


if __name__ == "__main__":
    main()
