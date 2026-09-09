"""
Shared helper functions for plotting scripts.

This module contains utility functions used across multiple plotting modules
including histograms, trajectories, and other visualization functions.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml


def decimal_year_to_date_string(decimal_year: float) -> str:
    """Convert a decimal year (e.g., 2031.5) to calendar date string YYYY-MM-DD.

    Handles leap years by interpolating between Jan 1 of the year and Jan 1 of the next year.
    """
    year = int(np.floor(decimal_year))
    frac = float(decimal_year - year)
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    total_seconds = (end - start).total_seconds()
    dt = start + timedelta(seconds=frac * total_seconds)
    return dt.date().isoformat()


def format_time_duration(minutes: float) -> str:
    """Pretty format a duration given in minutes."""
    if not np.isfinite(minutes):
        return "NaN"
    if minutes < 1.0:
        seconds = minutes * 60.0
        return f"{seconds:.0f} sec"
    if minutes < 60.0:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 24.0:
        return f"{hours:.0f} hrs"
    days = hours / 24.0
    if days < 7:
        return f"{days:.0f} days"
    weeks = days / 7.0
    if weeks < 4:
        return f"{weeks:.0f} weeks"
    months = days / 30.44
    if months < 12:
        return f"{months:.0f} months"
    years = days / 365.25
    if years < 1000:
        return f"{years:.0f} years"
    # Work-years not distinguished in label; show with comma grouping
    return f"{years:,.0f} years"


def now_decimal_year() -> float:
    """Return the present day from model config (for METR reference point)."""
    return _load_default_present_day()


def load_present_day(run_dir: Path | None, *, fallback: Optional[float] = None) -> float:
    """
    Load the present_day value saved alongside a run directory.

    Args:
        run_dir: Path to the run directory that contains model_config_snapshot files.
        fallback: Optional override if snapshot parsing fails.

    Returns:
        Present day as a decimal year.
    """
    if fallback is None:
        fallback = _load_default_present_day()

    if not run_dir:
        return fallback

    run_dir = Path(run_dir)

    # Prioritize the sampling configuration dump because it reflects the exact
    # inputs used for the run, even if DEFAULT_present_day gets updated later.
    yaml_path = run_dir / "input_distributions.yaml"
    value = _load_present_day_from_input_distributions(yaml_path)
    if value is not None:
        return value

    json_path = run_dir / "model_config_snapshot.json"
    value = _load_present_day_from_json(json_path)
    if value is not None:
        return value

    py_path = run_dir / "model_config_snapshot.py"
    value = _load_present_day_from_python(py_path)
    if value is not None:
        return value

    return fallback


def _load_present_day_from_json(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with path.open("r") as fh:
            payload = json.load(fh)
    except Exception:
        return None

    for key in ("present_day", "DEFAULT_present_day"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    default_params = payload.get("DEFAULT_PARAMETERS")
    if isinstance(default_params, dict):
        value = default_params.get("present_day")
        if isinstance(value, (int, float)):
            return float(value)

    return None


def _load_present_day_from_input_distributions(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with path.open("r") as fh:
            payload = yaml.safe_load(fh)
    except Exception:
        return None

    params = payload.get("parameters") if isinstance(payload, dict) else None
    if not isinstance(params, dict):
        return None

    entry = params.get("present_day")
    if isinstance(entry, dict):
        for key in ("value", "mean", "median"):
            val = entry.get(key)
            if isinstance(val, (int, float)):
                return float(val)

    return None


def _load_present_day_from_python(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except Exception:
        return None

    pattern = re.compile(r"DEFAULT_present_day\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    match = pattern.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    pattern_param = re.compile(
        r"'present_day'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    )
    match = pattern_param.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _load_default_present_day() -> float:
    try:
        from progress_model import config as cfg

        return float(getattr(cfg, "DEFAULT_present_day"))
    except Exception:
        # Fallback that roughly matches historical defaults if import fails
        return 2025.0


def load_central_trajectory_selection_start(run_dir: Path | None, *, fallback: Optional[float] = None) -> Optional[float]:
    """
    Load the central_trajectory_selection_start value from run directory.

    Args:
        run_dir: Path to the run directory that contains input_distributions.yaml.
        fallback: Optional override if parsing fails.

    Returns:
        Central trajectory selection start year as a decimal year, or None if not set.
    """
    if not run_dir:
        return fallback

    run_dir = Path(run_dir)
    yaml_path = run_dir / "input_distributions.yaml"
    if not yaml_path.exists():
        return fallback

    try:
        with yaml_path.open("r") as fh:
            payload = yaml.safe_load(fh)
    except Exception:
        return fallback

    if isinstance(payload, dict):
        value = payload.get("central_trajectory_selection_start")
        if isinstance(value, (int, float)):
            return float(value)

    return fallback


def get_time_tick_values_and_labels() -> Tuple[List[float], List[str]]:
    """Tick values and labels using work-time units (minutes, log scale).

    Mirrors the tick logic the removed app.py dashboard used
    (e.g., 1 week = 5 work days = 2,400 minutes; 1 year = work-year minutes).
    """
    tick_values = [
        0.033333,   # 2 sec
        0.5,        # 30 sec
        2,          # 2 min
        8,          # 8 min
        30,         # 30 min
        120,        # 2 hrs
        480,        # 8 hrs
        2400,       # 1 week (work week)
        10380,      # 1 month (work month)
        41520,      # 4 months (work months)
        124560,     # 1 year (work year)
        622800,     # 5 years (work years)
        2491200,    # 20 years (work years)
        12456000,   # 100 years (work years)
        49824000,   # 400 years (work years)
        199296000,  # 1,600 years (work years)
        797184000,  # 6,400 years (work years)
        3188736000, # 25,600 years (work years)
        14947200000,# 120,000 years (work years)
    ]
    tick_labels = [
        "2 sec",
        "30 sec",
        "2 min",
        "8 min",
        "30 min",
        "2 hrs",
        "8 hrs",
        "1 week",
        "1 month",
        "4 months",
        "1 year",
        "5 years",
        "20 years",
        "100 years",
        "400 years",
        "1,600 years",
        "6,400 years",
        "25,600 years",
        "120,000 years",
    ]
    return tick_values, tick_labels


def format_years_value(y: float) -> str:
    """Format a year value for display.

    Args:
        y: Year value to format

    Returns:
        Formatted string representation
    """
    if not np.isfinite(y):
        return "inf"
    if y < 0.01:
        return f"{y:.3f}"
    if y < 0.1:
        return f"{y:.2f}"
    if y < 10:
        return f"{y:.2f}".rstrip("0").rstrip(".")
    if y < 1000:
        return f"{y:.2f}".rstrip("0").rstrip(".")
    return f"{y:,.0f}"


def get_year_tick_values_and_labels(ymin: float, ymax: float) -> Tuple[List[float], List[str]]:
    """Generate log-scale tick values and labels for year axis.

    Args:
        ymin: Minimum year value
        ymax: Maximum year value

    Returns:
        Tuple of (tick values, tick labels)
    """
    if ymin <= 0:
        ymin = 1e-3
    lo = int(np.floor(np.log10(ymin)))
    hi = int(np.ceil(np.log10(ymax)))
    values: List[float] = []
    labels: List[str] = []
    for p in range(lo, hi + 1):
        v = float(10 ** p)
        values.append(v)
        if v >= 1000:
            labels.append(f"{int(v):,}")
        elif v >= 1:
            labels.append(str(int(v)))
        else:
            labels.append(f"{v:g}")
    return values, labels


def simplify_milestone_name(name: str) -> str:
    """Simplify milestone names for display."""
    simplifications = {
        "SAR-level-experiment-selection-skill": "SAR",
        "SIAR-level-experiment-selection-skill": "SIAR",
        "STRAT-AI": "STRAT-AI",
        "TED-AI": "TED-AI",
        "ASI": "ASI",
    }
    return simplifications.get(name, name)


def parse_milestone_pairs(pairs_arg: Optional[str]) -> List[Tuple[str, str]]:
    """Parse a pairs string like "SC:SAR,SAR:SIAR" into [("SC","SAR"), ...].

    Args:
        pairs_arg: Comma-separated pairs in FROM:TO format

    Returns:
        List of (from_milestone, to_milestone) tuples
    """
    if not pairs_arg:
        return []
    pairs: List[Tuple[str, str]] = []
    for chunk in pairs_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        left, right = chunk.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            pairs.append((left, right))
    return pairs


def load_metr_p80_points() -> Optional[List[Tuple[float, float]]]:
    """Load SOTA METR p80 horizon points as (decimal_year, p80_minutes).

    Returns None if file missing or malformed.
    """
    try:
        metr_yaml = Path(__file__).resolve().parents[2] / "inputs" / "metr" / "benchmark_results_1_1.yaml"
        with open(metr_yaml, "r") as f:
            bench = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    results = bench.get("results") if isinstance(bench, dict) else None
    if not isinstance(results, dict):
        return None
    points: List[Tuple[float, float]] = []
    for _model_name, model_info in results.items():
        try:
            release = model_info.get("release_date")
            if isinstance(release, str):
                d = datetime.strptime(release, "%Y-%m-%d").date()
            else:
                # If already a date-like, try year/month/day attrs
                d = release
            decimal_year = float(d.year + (d.timetuple().tm_yday - 1) / 365.25)
        except Exception:
            continue
        # New format: metrics at top level of model_info
        metrics = model_info.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        # Only include points marked SOTA
        is_sota = metrics.get("is_sota", False)
        p80 = metrics.get("p80_horizon_length", {})
        p80_est = None
        if isinstance(p80, dict) and p80.get("estimate") is not None:
            p80_est = float(p80["estimate"])
        if is_sota and p80_est is not None and p80_est > 0:
            points.append((decimal_year, float(p80_est)))
    return points if points else None
