"""
Trajectory plotting functions for rollout analysis.

This module contains functions for plotting time series trajectories including
horizon length trajectories, uplift trajectories, and generic metric trajectories.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import utilities
import sys
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

from plotting_utils.helpers import (
    now_decimal_year,
    get_time_tick_values_and_labels,
    load_metr_p80_points,
)


def select_central_trajectory(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    sc_times: List[Optional[float]],
    present_day: float,
    ac_times: Optional[List[Optional[float]]] = None,
    selection_start: Optional[float] = None,
) -> tuple[int, np.ndarray, float]:
    """Select central trajectory using distance-from-median method.

    Algorithm:
    1. Filter to trajectories that achieved AC within ±0.0415 years of median AC
    2. Create monthly grid from selection_start (or present_day) to AC median + tolerance
    3. Interpolate all trajectories to log10(horizon) space on monthly grid
    4. Compute median curve across trajectories
    5. For each trajectory, sum distances from median (from selection_start to AC time)
    6. Return trajectory with minimum cumulative distance

    Args:
        times: Time array in decimal years
        trajectories: List of horizon arrays (minutes)
        sc_times: List of SC arrival times (unused, kept for API compatibility)
        present_day: Present day in decimal years (used as fallback for selection_start)
        ac_times: List of AC arrival times (used for filtering and distance cutoff)
        selection_start: Start year for trajectory selection grid (defaults to present_day)

    Returns:
        Tuple of (best_index, central_trajectory_array, median_ac_time)
    """
    if ac_times is None:
        raise ValueError("ac_times is required for central trajectory selection")

    # Use selection_start if provided, otherwise fall back to present_day
    grid_start = selection_start if selection_start is not None else present_day

    # Get finite AC times
    finite_ac = [(i, t) for i, t in enumerate(ac_times) if t is not None]
    if len(finite_ac) == 0:
        raise ValueError("No trajectories with finite AC times")

    # Compute median AC time
    ac_vals = [t for _, t in finite_ac]
    median_ac = float(np.median(ac_vals))

    # Filter to trajectories with AC within ±0.0415 years (~half month) of median
    tolerance = 0.0415
    filtered = [(i, t) for i, t in finite_ac if abs(t - median_ac) <= tolerance]
    if len(filtered) == 0:
        # Fall back: allow wider tolerance (±2 months)
        filtered = [(i, t) for i, t in finite_ac if abs(t - median_ac) <= 1/6]
        if len(filtered) == 0:
            filtered = finite_ac[:100]  # Last resort: take first 100

    filtered_indices = [i for i, _ in filtered]
    ac_range_end = median_ac + tolerance
    print(f"Central trajectory selection: {len(filtered_indices)} trajectories with AC within ±{tolerance} of median AC={median_ac:.3f}")
    print(f"  Grid start: {grid_start:.3f}, Grid end: {ac_range_end:.3f}")

    # Create monthly grid from grid_start to end of AC tolerance range
    x_grid = np.arange(grid_start, ac_range_end + 1e-6, 1/12)

    # Build matrix of log10(horizon) values interpolated to monthly grid
    matrix = np.full((len(filtered_indices), len(x_grid)), np.nan)
    for row_idx, traj_idx in enumerate(filtered_indices):
        traj = trajectories[traj_idx]
        # Find valid (positive, finite) values
        valid = np.isfinite(traj) & (traj > 0)
        if np.sum(valid) < 2:
            continue
        valid_times = times[valid]
        valid_vals = traj[valid]

        # Interpolate to grid points within trajectory time range
        mask = (x_grid >= valid_times[0]) & (x_grid <= valid_times[-1])
        if mask.any():
            interp_vals = np.interp(x_grid[mask], valid_times, valid_vals)
            # Convert to log10 space
            matrix[row_idx, mask] = np.log10(np.clip(interp_vals, 1e-10, None))

    # Compute median curve
    median_curve = np.nanmedian(matrix, axis=0)

    # Compute mean distance from median for each trajectory
    # Mean of absolute distances in log space, stopping at AC time for each trajectory
    if ac_times is not None:
        # Create per-trajectory masks that stop at AC time
        distances = np.zeros(len(filtered_indices))
        for row_idx, traj_idx in enumerate(filtered_indices):
            ac_time = ac_times[traj_idx] if traj_idx < len(ac_times) else None
            if ac_time is not None:
                # Only average distances up to AC time
                time_mask = x_grid <= ac_time
                row_diffs = np.abs(matrix[row_idx, :] - median_curve)
                row_diffs[~time_mask] = np.nan  # Ignore points after AC
                distances[row_idx] = np.nanmean(row_diffs)
            else:
                # No AC time, use full range
                distances[row_idx] = np.nanmean(np.abs(matrix[row_idx, :] - median_curve))
    else:
        # No AC times provided, use full range for all
        distances = np.nanmean(np.abs(matrix - median_curve), axis=1)

    # Find trajectory with minimum mean distance
    best_row_idx = int(np.nanargmin(distances))
    best_traj_idx = filtered_indices[best_row_idx]

    print(f"Selected central trajectory index {best_traj_idx} with mean distance {distances[best_row_idx]:.4f}")

    return best_traj_idx, trajectories[best_traj_idx], median_ac


def plot_horizon_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    current_horizon_minutes: float = 15.0,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    overlay_metr: bool = True,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    aa_times: Optional[List[Optional[float]]] = None,
    mse_values: Optional[List[Optional[float]]] = None,
    shade_by_mse: bool = False,
    mse_threshold: Optional[float] = None,
    cutoff_year: Optional[float] = None,
    sc_times: Optional[List[Optional[float]]] = None,
    present_day: Optional[float] = None,
    ac_times: Optional[List[Optional[float]]] = None,
    central_trajectory_idx: Optional[int] = None,
    selection_start: Optional[float] = None,
    show_both_central_methods: bool = False,
) -> None:
    """Render horizon length trajectories similar to the reference figure.

    Assumes times are decimal years and horizons are in minutes.

    If central_trajectory_idx is provided, uses that trajectory as the central one.
    Otherwise, if sc_times and present_day are provided, uses distance-from-median method
    to select central trajectory (filters to SC median month, finds closest to median).
    Distance calculation stops at AC time for each trajectory if ac_times is provided.
    Otherwise falls back to simple element-wise median.

    Args:
        selection_start: Start year for central trajectory selection grid (defaults to present_day)
        show_both_central_methods: If True, plot both distance-from-median and element-wise median trajectories
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Filter trajectories by MSE threshold if specified
    if mse_threshold is not None and mse_values is not None:
        filtered_indices = []
        for i, mse in enumerate(mse_values):
            if mse is None or mse <= mse_threshold:
                filtered_indices.append(i)
        trajectories = [trajectories[i] for i in filtered_indices]
        if aa_times is not None:
            aa_times = [aa_times[i] for i in filtered_indices]
        if mse_values is not None:
            mse_values = [mse_values[i] for i in filtered_indices]
        if sc_times is not None:
            sc_times = [sc_times[i] for i in filtered_indices]
        if ac_times is not None:
            ac_times = [ac_times[i] for i in filtered_indices]
        print(f"Filtered to {len(trajectories)} trajectories with MSE <= {mse_threshold}")

    if len(trajectories) == 0:
        raise ValueError("No trajectories remain after MSE filtering")

    # Store original times and trajectories before interpolation (for central trajectory selection)
    original_times = times.copy()
    original_trajectories = [t.copy() for t in trajectories]

    # Clip extremely large values to avoid distorting the log scale
    min_horizon_minutes = 0.001
    # Set a very high cap that won't artificially bunch trajectories
    # Cap at 10 million work-years instead of 120,000 to avoid bunching
    max_work_year_minutes = float(10_000_000 * 52 * 40 * 60)
    # Replace non-positive and non-finite values with NaN so they don't draw lines to zero
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_horizon_minutes, max_work_year_minutes)
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Optionally interpolate trajectories for smoother plotting
    # Skip interpolation if many trajectories are at the cap (causes visual artifacts)
    num_at_cap = sum(1 for arr in cleaned if np.any(arr >= max_work_year_minutes * 0.99))
    skip_interpolation = num_at_cap > len(cleaned) * 0.5  # Skip if >50% hit cap

    if not skip_interpolation:
        if cutoff_year is not None:
            # Only interpolate the visible range
            time_range = cutoff_year - times[0]
            num_interp_points = min(len(times) * 10, int(time_range * 50))  # ~50 points per year in visible range
            times_interp = np.linspace(times[0], cutoff_year, num_interp_points)
            print(f"Interpolating from {len(times)} to {num_interp_points} points ({num_interp_points/time_range:.1f} pts/year)")
        else:
            num_interp_points = len(times) * 10
            times_interp = np.linspace(times[0], times[-1], num_interp_points)
            print(f"Interpolating from {len(times)} to {num_interp_points} points")

        from scipy.interpolate import interp1d
        cleaned_interp: List[np.ndarray] = []
        for arr in cleaned:
            # Only interpolate over valid (non-NaN) regions
            valid_mask = ~np.isnan(arr)
            if np.sum(valid_mask) > 1:  # Need at least 2 points to interpolate
                try:
                    f = interp1d(times[valid_mask], arr[valid_mask], kind='linear',
                               bounds_error=False, fill_value=np.nan)
                    arr_interp = f(times_interp)
                except Exception:
                    # If interpolation fails, fall back to original
                    arr_interp = np.interp(times_interp, times, arr)
            else:
                arr_interp = np.full(len(times_interp), np.nan)
            cleaned_interp.append(arr_interp)

        # Use interpolated data for plotting
        cleaned = cleaned_interp
        times = times_interp
    else:
        print(f"Skipping interpolation ({num_at_cap}/{len(cleaned)} trajectories at cap - would cause artifacts)")

    # Compute element-wise median for fallback
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    elementwise_median_traj = np.nanmedian(stacked, axis=0)

    # Select central trajectory
    # If central_trajectory_idx provided, use that
    # Else if sc_times and present_day provided, use distance-from-median method
    # Otherwise fall back to element-wise median
    central_traj_idx = central_trajectory_idx
    median_sc_time = None
    distance_method_traj = None
    if central_traj_idx is not None:
        # Use the provided central trajectory index
        central_traj = cleaned[central_traj_idx]
        distance_method_traj = central_traj
        print(f"Using provided central trajectory index: {central_traj_idx}")
    elif sc_times is not None and present_day is not None:
        try:
            central_traj_idx, central_traj_raw, median_sc_time = select_central_trajectory(
                original_times, original_trajectories, sc_times, present_day, ac_times=ac_times,
                selection_start=selection_start
            )
            # Get the cleaned/interpolated version for plotting
            central_traj = cleaned[central_traj_idx]
            distance_method_traj = central_traj
        except Exception as e:
            print(f"Warning: Could not select central trajectory via distance method: {e}")
            central_traj_idx = None

    if central_traj_idx is None:
        # Fall back to element-wise median
        central_traj = elementwise_median_traj
        print("Using element-wise median for central trajectory")

    plt.figure(figsize=(14, 8))

    # Determine colors for trajectories
    if shade_by_mse and mse_values is not None:
        # Compute color mapping based on MSE values
        # Filter out None values for computing percentiles
        valid_mse = [mse for mse in mse_values if mse is not None]
        if len(valid_mse) > 0:
            mse_min = np.percentile(valid_mse, 5)  # Use 5th percentile as min
            mse_max = np.percentile(valid_mse, 95)  # Use 95th percentile as max
            print(f"MSE range for coloring: {mse_min:.4f} to {mse_max:.4f}")
        else:
            mse_min = 0.0
            mse_max = 1.0
    else:
        valid_mse = []

    # Draw all trajectories
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        if shade_by_mse and mse_values is not None and i < len(mse_values) and mse_values[i] is not None:
            # Color based on MSE: green (low) to red (high)
            mse = mse_values[i]
            # Normalize MSE to [0, 1]
            if len(valid_mse) > 0 and mse_max > mse_min:
                norm_mse = (mse - mse_min) / (mse_max - mse_min)
                norm_mse = np.clip(norm_mse, 0.0, 1.0)
            else:
                norm_mse = 0.5
            # Green (0.0) to Red (1.0) via Yellow
            # RGB: (0,1,0) -> (1,1,0) -> (1,0,0)
            if norm_mse < 0.5:
                # Green to Yellow
                r = norm_mse * 2
                g = 1.0
                b = 0.0
            else:
                # Yellow to Red
                r = 1.0
                g = 2.0 * (1.0 - norm_mse)
                b = 0.0
            color = (r, g, b, alpha)
        else:
            # Default blue color
            color = (0.2, 0.5, 0.7, alpha)
        plt.plot(times, cleaned[i], color=color, linewidth=1.0)

    # Central trajectory - only use dist-from-median incl. backcasts method
    if show_both_central_methods and distance_method_traj is not None:
        # Use distance-from-median including backcasts method
        plt.plot(times, distance_method_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")
        # Also show element-wise median
        plt.plot(times, elementwise_median_traj, color="tab:orange", linestyle=":", linewidth=2.0, label="Median (synthetic, not a real traj)")
    else:
        # Just plot the selected central trajectory
        plt.plot(times, central_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")

    # Horizontal line for current horizon
    plt.axhline(current_horizon_minutes, color="red", linewidth=2.0, label=f"Current Horizon ({int(current_horizon_minutes)} min)")

    # Vertical line for current time
    now_year = now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Optional METR p80 scatter
    if overlay_metr:
        points = load_metr_p80_points()
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, color="black", s=18, label="External Benchmarks (p80)", zorder=10)

    # Axes and formatting
    plt.yscale("log")
    tick_values, tick_labels = get_time_tick_values_and_labels()
    plt.yticks(tick_values, tick_labels)
    # Expand y-limit to cover data range
    finite_max = np.nanmax(stacked)
    if np.isfinite(finite_max):
        ymin = min(tick_values)
        ymax = min(max(finite_max, max(tick_values)), max_work_year_minutes)
        plt.ylim(ymin, ymax)

    # Set x-axis limit if cutoff_year is specified
    if cutoff_year is not None:
        plt.xlim(times[0], cutoff_year)

    plt.xlabel("Year")
    plt.ylabel("Time Horizon")

    # Update title to indicate MSE shading/filtering
    base_title = title or "Complete Time Horizon Extension Trajectories\n(Historical development and future projections)"
    base_title += f"\n(n={len(trajectories)} trajectories"
    if mse_threshold is not None:
        base_title += f", MSE ≤ {mse_threshold:.3f}"
    base_title += ")"
    if shade_by_mse and len(valid_mse) > 0:
        base_title += f"\nColor: Green (low MSE) → Yellow → Red (high MSE)"
    plt.title(base_title)

    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Export central trajectory to CSV
    if show_both_central_methods and distance_method_traj is not None:
        # Export the dist-from-median incl. backcasts trajectory
        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w") as f:
            f.write("year,horizon_minutes\n")
            for t, h in zip(times, distance_method_traj):
                if np.isfinite(h):
                    f.write(f"{t},{h}\n")
        print(f"Saved central trajectory CSV to {csv_path}")
    else:
        # Just export the single central trajectory
        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w") as f:
            f.write("year,horizon_minutes\n")
            for t, h in zip(times, central_traj):
                if np.isfinite(h):
                    f.write(f"{t},{h}\n")
        print(f"Saved central trajectory CSV to {csv_path}")


def plot_uplift_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    aa_times: Optional[List[Optional[float]]] = None,
    max_year: Optional[float] = None,
) -> None:
    """Render AI software R&D uplift (ai_sw_progress_mult_ref_present_day) trajectories.

    Assumes times are decimal years and uplift values are dimensionless multipliers.

    Args:
        max_year: Optional maximum year for x-axis. Trajectories continue to edge if they extend beyond.
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Replace non-positive and non-finite values with NaN
    min_uplift = 0.1
    max_uplift = 1e6
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_uplift, max_uplift)
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory across rollouts
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    median_traj = np.nanmedian(stacked, axis=0)

    plt.figure(figsize=(14, 8))

    # Draw all trajectories in blue
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        color = (0.2, 0.5, 0.7, alpha)
        plt.plot(times, cleaned[i], color=color, linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")

    # Horizontal line for current uplift (assuming ~1x at present day)
    plt.axhline(1.0, color="red", linewidth=2.0, label="Current Uplift (1x)")

    # Vertical line for current time
    now_year = now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Axes and formatting
    plt.yscale("log")
    plt.xlabel("Year")
    plt.ylabel("AI Software R&D Uplift (Multiplier)")

    # Set x-axis limits if max_year is specified
    if max_year is not None:
        plt.xlim(right=max_year)

    # Update title
    base_title = title or "AI Software R&D Uplift Trajectories\n(Historical development and future projections)"
    base_title += f"\n(n={len(trajectories)} trajectories)"
    plt.title(base_title)

    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    metric_name: str,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    aa_times: Optional[List[Optional[float]]] = None,
    mse_values: Optional[List[Optional[float]]] = None,
    shade_by_mse: bool = False,
    mse_threshold: Optional[float] = None,
    cutoff_year: Optional[float] = None,
    yscale: str = "auto",
) -> None:
    """Render arbitrary metric trajectories over time.

    Args:
        times: Array of decimal years
        trajectories: List of metric value arrays (one per rollout)
        out_path: Output PNG path
        metric_name: Name of the metric being plotted
        alpha: Transparency for individual trajectories
        max_trajectories: Maximum number of trajectories to draw
        title: Optional custom title
        stop_at_sc: Mask each trajectory after its own aa_time
        aa_times: List of aa_time values (one per rollout)
        mse_values: List of METR MSE values (one per rollout)
        shade_by_mse: Color trajectories by their METR MSE
        mse_threshold: Exclude trajectories with METR MSE above this threshold
        cutoff_year: Cut off x-axis at this year
        yscale: Y-axis scale - "auto" (default), "log", or "linear"
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Filter trajectories by MSE threshold if specified
    if mse_threshold is not None and mse_values is not None:
        filtered_indices = []
        for i, mse in enumerate(mse_values):
            if mse is None or mse <= mse_threshold:
                filtered_indices.append(i)
        trajectories = [trajectories[i] for i in filtered_indices]
        if aa_times is not None:
            aa_times = [aa_times[i] for i in filtered_indices]
        if mse_values is not None:
            mse_values = [mse_values[i] for i in filtered_indices]
        print(f"Filtered to {len(trajectories)} trajectories with MSE <= {mse_threshold}")

    if len(trajectories) == 0:
        raise ValueError("No trajectories remain after MSE filtering")

    # Clean trajectories: replace non-finite values with NaN
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory across rollouts
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    median_traj = np.nanmedian(stacked, axis=0)

    plt.figure(figsize=(14, 8))

    # Determine colors for trajectories
    if shade_by_mse and mse_values is not None:
        # Compute color mapping based on MSE values
        valid_mse = [mse for mse in mse_values if mse is not None]
        if len(valid_mse) > 0:
            mse_min = np.percentile(valid_mse, 5)
            mse_max = np.percentile(valid_mse, 95)
            print(f"MSE range for coloring: {mse_min:.4f} to {mse_max:.4f}")
        else:
            mse_min = 0.0
            mse_max = 1.0
    else:
        valid_mse = []

    # Draw all trajectories
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        if shade_by_mse and mse_values is not None and i < len(mse_values) and mse_values[i] is not None:
            # Color based on MSE: green (low) to red (high)
            mse = mse_values[i]
            if len(valid_mse) > 0 and mse_max > mse_min:
                norm_mse = (mse - mse_min) / (mse_max - mse_min)
                norm_mse = np.clip(norm_mse, 0.0, 1.0)
            else:
                norm_mse = 0.5
            # Green (0.0) to Red (1.0) via Yellow
            if norm_mse < 0.5:
                r = norm_mse * 2
                g = 1.0
                b = 0.0
            else:
                r = 1.0
                g = 2.0 * (1.0 - norm_mse)
                b = 0.0
            color = (r, g, b, alpha)
        else:
            # Default blue color
            color = (0.2, 0.5, 0.7, alpha)
        plt.plot(times, cleaned[i], color=color, linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Median Trajectory")

    # Vertical line for current time
    now_year = now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Determine if we should use log scale
    if yscale == "log":
        use_log_scale = True
    elif yscale == "linear":
        use_log_scale = False
    else:  # auto
        finite_vals = stacked[np.isfinite(stacked)]
        use_log_scale = False
        if len(finite_vals) > 0:
            val_min = np.min(finite_vals)
            val_max = np.max(finite_vals)
            # Use log scale if data spans more than 2 orders of magnitude and all values are positive
            if val_min > 0 and val_max / val_min > 100:
                use_log_scale = True

    if use_log_scale:
        plt.yscale("log")

    plt.xlabel("Year")
    plt.ylabel(metric_name.replace("_", " ").title())

    # Build title
    base_title = title or f"{metric_name.replace('_', ' ').title()} Trajectories"
    base_title += f"\n(n={len(trajectories)} trajectories"
    if mse_threshold is not None:
        base_title += f", MSE ≤ {mse_threshold:.3f}"
    base_title += ")"
    if shade_by_mse and len(valid_mse) > 0:
        base_title += f"\nColor: Green (low MSE) → Yellow → Red (high MSE)"
    plt.title(base_title)

    plt.grid(True, which="both", alpha=0.25)
    plt.legend(loc="best")

    # Set x-axis limit if cutoff_year is specified
    if cutoff_year is not None:
        plt.xlim(times[0], cutoff_year)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
