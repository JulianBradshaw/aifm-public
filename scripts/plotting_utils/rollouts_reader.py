"""
Unified JSONL rollouts reader for eliminating parsing duplication.

This module provides a clean abstraction for reading rollout data from
rollouts.jsonl files, replacing dozens of duplicated reading functions
across the plotting scripts.

Usage:
    from plotting_utils.rollouts_reader import RolloutsReader

    reader = RolloutsReader(rollouts_file)

    # Read milestone times
    times, not_achieved, sim_end = reader.read_milestone_times("AC")

    # Read trajectories
    times_array, trajectories, aa_times = reader.read_trajectories("horizon")
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterator, Any, Sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class _RolloutRecord:
    """Internal normalized rollout record wrapper.

    Stores parsed and validated rollout data with derived fields to eliminate
    redundant JSON-to-NumPy parsing across methods.
    """
    times: np.ndarray
    milestones: Dict[str, Dict[str, Any]]
    aa_time: Optional[float]
    ai_research_taste: Optional[np.ndarray]
    effective_compute: Optional[np.ndarray]
    simulation_end: float
    results: Dict[str, Any]  # Full results dict for accessing other metrics


def _is_cache_path(path: Optional[Path]) -> bool:
    """True for a rollouts.cache.json-style path (compact single-line JSON, not JSONL)."""
    if path is None:
        return False
    path = Path(path)
    return path.suffix == ".json" and ".cache" in path.name


class RolloutsReader:
    """Reader for rollouts.jsonl files with common data extraction patterns.

    Automatically detects and uses lightweight cache files when available.
    """

    def __init__(self, rollouts_file: Path, use_cache: bool = True):
        """Initialize reader with path to rollouts.jsonl file.

        Args:
            rollouts_file: Path to rollouts.jsonl file or cache directory
            use_cache: If True, automatically use cache if available (default: True)
        """
        self.rollouts_file = Path(rollouts_file)
        self._use_cache = use_cache
        self._cache_data = None  # Loaded cache data if using cache
        self._jsonl_file = None  # Original JSONL file path (when cache is used)

        # Check if this is a cache file or if cache exists
        if use_cache:
            cache_file = self._find_cache_file()
            if cache_file:
                self._load_cache(cache_file)
                return

        # Fall back to regular JSONL
        if not self.rollouts_file.exists():
            raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    def _find_cache_file(self) -> Optional[Path]:
        """Find cache file if it exists.

        Returns:
            Path to cache file, or None if not found
        """
        # If given path ends with .cache.json, use it directly
        if self.rollouts_file.suffix == '.json' and '.cache' in self.rollouts_file.name:
            if self.rollouts_file.exists():
                return self.rollouts_file
            return None

        # Look for cache in same directory as rollouts.jsonl
        cache_path = self.rollouts_file.parent / "rollouts.cache.json"
        if cache_path.exists():
            return cache_path

        return None

    def _load_cache(self, cache_file: Path) -> None:
        """Load data from cache file.

        Args:
            cache_file: Path to cache JSON file
        """
        with open(cache_file, 'r', encoding='utf-8') as f:
            self._cache_data = json.load(f)
        # Store original JSONL path for trajectory reading
        self._jsonl_file = self.rollouts_file
        # Update rollouts_file to point to cache for reference
        self.rollouts_file = cache_file

    def _parse_rollout_record(self, rec: Dict[str, Any]) -> Optional[_RolloutRecord]:
        """Parse raw JSON record into normalized _RolloutRecord.

        Args:
            rec: Raw JSON record

        Returns:
            Normalized rollout record, or None if parsing fails
        """
        results = rec.get("results")
        if not isinstance(results, dict):
            return None

        # Parse times array (may not exist in cached records)
        times_list = results.get("times")
        if times_list is None or len(times_list) == 0:
            # Cache records don't have times array, use simulation_end
            simulation_end_val = results.get("simulation_end")
            if simulation_end_val is not None:
                try:
                    simulation_end = float(simulation_end_val)
                    times = np.array([simulation_end])  # Dummy array for cache
                except (TypeError, ValueError):
                    return None
            else:
                return None
        else:
            times = np.asarray(times_list, dtype=float)
            simulation_end = float(times[-1])

        # Parse milestones
        milestones = results.get("milestones", {})
        if not isinstance(milestones, dict):
            milestones = {}

        # Parse aa_time
        aa_time_val = results.get("aa_time")
        try:
            aa_time = (
                float(aa_time_val)
                if aa_time_val is not None and np.isfinite(float(aa_time_val))
                else None
            )
        except (TypeError, ValueError):
            aa_time = None

        # Parse ai_research_taste trajectory (only in full JSONL)
        taste_list = results.get("ai_research_taste")
        ai_research_taste = (
            np.asarray(taste_list, dtype=float) if taste_list is not None else None
        )

        # Parse effective_compute trajectory (only in full JSONL)
        compute_list = results.get("effective_compute")
        effective_compute = (
            np.asarray(compute_list, dtype=float) if compute_list is not None else None
        )

        return _RolloutRecord(
            times=times,
            milestones=milestones,
            aa_time=aa_time,
            ai_research_taste=ai_research_taste,
            effective_compute=effective_compute,
            simulation_end=simulation_end,
            results=results,
        )

    def iter_rollouts(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all valid rollout records.

        Yields:
            Dictionaries containing rollout data with 'results' key
        """
        # Use cache if available
        if self._cache_data is not None:
            for rec in self._cache_data['rollouts']:
                yield rec
            return

        # Read from JSONL
        with self.rollouts_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results = rec.get("results")
                if not isinstance(results, dict):
                    continue
                yield rec

    def iter_normalized_rollouts(self) -> Iterator[_RolloutRecord]:
        """Iterate over normalized rollout records.

        Yields:
            Normalized _RolloutRecord instances with parsed data
        """
        for rec in self.iter_rollouts():
            normalized = self._parse_rollout_record(rec)
            if normalized is not None:
                yield normalized

    def _iter_rollouts_from_jsonl(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rollouts directly from JSONL, bypassing cache.

        This is used for reading trajectory data which is not stored in cache.

        Yields:
            Dictionaries containing rollout data with 'results' key
        """
        # Determine JSONL file path
        jsonl_path = self._jsonl_file if self._jsonl_file is not None else self.rollouts_file

        # Read from JSONL
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results = rec.get("results")
                if not isinstance(results, dict):
                    continue
                yield rec

    def _iter_normalized_rollouts_from_jsonl(self) -> Iterator[_RolloutRecord]:
        """Iterate over normalized rollouts from JSONL, bypassing cache.

        This is used for reading trajectory data which is not stored in cache.

        Yields:
            Normalized _RolloutRecord instances with parsed data
        """
        for rec in self._iter_rollouts_from_jsonl():
            normalized = self._parse_rollout_record(rec)
            if normalized is not None:
                yield normalized

    def iter_all_records(self) -> Iterator[Dict[str, Any]]:
        """Iterate over ALL rollout records, including those with errors.

        Unlike iter_rollouts(), this includes records with errors or missing results.
        Useful for generating summary statistics or analyzing failures.

        Yields:
            All dictionaries from the JSONL file (valid JSON lines)
        """
        # _load_cache() repoints self.rollouts_file at rollouts.cache.json, which is a single
        # compact JSON object (json.dump with separators=(',',':') -> zero newlines). Reading
        # that path line-by-line as JSONL used to parse the whole cache as ONE record, so every
        # caller saw "1 rollout, 0 successful". Always read the real JSONL instead, and fall
        # back to the cached records only when the JSONL is genuinely unavailable.
        jsonl_path = self._jsonl_file if self._jsonl_file is not None else self.rollouts_file
        if self._cache_data is not None and (
            _is_cache_path(jsonl_path) or not jsonl_path.exists()
        ):
            # NOTE: build_cache() stores results={} for failed rollouts, so failures are not
            # distinguishable by results alone here — use the 'error' key (which it preserves).
            for rec in self._cache_data.get("rollouts", []):
                yield rec
            return

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    yield rec
                except json.JSONDecodeError:
                    continue

    def _resolve_milestone(
        self,
        rollout: _RolloutRecord,
        milestone_name: str,
        *,
        include_compute: bool = False
    ) -> Optional[Dict[str, float]]:
        """Resolve milestone time and optionally compute from a rollout.

        Centralizes milestone resolution logic from explicit rollout data.

        Args:
            rollout: Normalized rollout record
            milestone_name: Name of milestone to resolve
            include_compute: If True, include effective_compute_ooms in result

        Returns:
            Dictionary with 'time' and optionally 'effective_compute_ooms',
            or None if milestone not found/achieved
        """
        # Try to get milestone from explicit milestones dict
        milestone_info = rollout.milestones.get(milestone_name)
        if isinstance(milestone_info, dict):
            time_val = milestone_info.get("time")
            try:
                time_float = float(time_val) if time_val is not None else np.nan
            except (TypeError, ValueError):
                time_float = np.nan

            if np.isfinite(time_float):
                result = {'time': float(time_float)}
                if include_compute:
                    compute_val = milestone_info.get("effective_compute_ooms")
                    try:
                        compute_float = (
                            float(compute_val) if compute_val is not None else np.nan
                        )
                    except (TypeError, ValueError):
                        compute_float = np.nan
                    if np.isfinite(compute_float):
                        result['effective_compute_ooms'] = float(compute_float)
                return result

        return None

    def read_milestone_times(
        self,
        milestone_name: str
    ) -> Tuple[List[float], int, Optional[float]]:
        """Read milestone arrival times from rollouts.

        Args:
            milestone_name: Name of milestone (e.g., "AC", "SAR")

        Returns:
            times: List of arrival times (only for rollouts that achieved milestone)
            num_not_achieved: Count of rollouts where milestone was not achieved
            typical_sim_end: Typical simulation end time (median), or None
        """
        times: List[float] = []
        num_not_achieved: int = 0
        sim_end_times: List[float] = []

        for rollout in self.iter_normalized_rollouts():
            # Track simulation end time
            sim_end_times.append(rollout.simulation_end)

            # Resolve milestone using centralized helper
            milestone_data = self._resolve_milestone(rollout, milestone_name)

            if milestone_data is not None:
                times.append(milestone_data['time'])
            else:
                num_not_achieved += 1

        typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
        return times, num_not_achieved, typical_sim_end

    def read_milestone_compute(
        self,
        milestone_name: str,
    ) -> Tuple[List[float], int]:
        """Read milestone effective compute values.

        Args:
            milestone_name: Name of milestone

        Returns:
            compute_ooms: List of effective compute in OOMs
            num_not_achieved: Count of rollouts where milestone was not achieved
        """
        compute_ooms: List[float] = []
        num_not_achieved: int = 0

        for rollout in self.iter_normalized_rollouts():
            # Resolve milestone with compute using centralized helper
            milestone_data = self._resolve_milestone(
                rollout,
                milestone_name,
                include_compute=True
            )

            if milestone_data is not None and 'effective_compute_ooms' in milestone_data:
                compute_ooms.append(milestone_data['effective_compute_ooms'])
            else:
                num_not_achieved += 1

        return compute_ooms, num_not_achieved

    def _iter_transition(
        self,
        from_milestone: str,
        to_milestone: str,
        *,
        allow_censored: bool = False,
        stats: Optional[Dict[str, int]] = None
    ) -> Iterator[Tuple[float, Optional[float], float]]:
        """Core generator for transition data between milestones.

        Yields transition tuples for each rollout where from_milestone is achieved.

        Args:
            from_milestone: Starting milestone name
            to_milestone: Ending milestone name
            allow_censored: If True, yield censored transitions (to_time=None)

        Yields:
            Tuples of (from_time, to_time, simulation_end)
            where to_time is None for censored transitions
        """
        for rollout in self.iter_normalized_rollouts():
            # Resolve from_milestone
            from_data = self._resolve_milestone(rollout, from_milestone)
            if from_data is None:
                continue  # First milestone not achieved, skip

            from_time = from_data['time']

            # Resolve to_milestone
            to_data = self._resolve_milestone(rollout, to_milestone)
            if to_data is None:
                # Second milestone not achieved (censored)
                if allow_censored:
                    yield (from_time, None, rollout.simulation_end)
                continue

            to_time = to_data['time']

            # Check if transition is valid (to comes after from)
            if to_time <= from_time:
                if stats is not None:
                    stats["out_of_order"] = stats.get("out_of_order", 0) + 1
                continue  # B before or at same time as A

            yield (from_time, to_time, rollout.simulation_end)

    def iter_transition_with_params(
        self,
        from_milestone: str,
        to_milestone: str,
        *,
        include_censored: bool = False,
        censored_value: float = 1e10,
    ) -> Iterator[Tuple[float, float, Dict[str, Any]]]:
        """Iterate over transitions yielding timeline, duration, and parameters.

        Efficient single-pass extraction for correlation analysis. Works with cache.

        Args:
            from_milestone: Starting milestone name
            to_milestone: Ending milestone name
            include_censored: If True, include censored transitions
            censored_value: Duration value for censored transitions (default 1e10)

        Yields:
            Tuples of (from_time, duration, params_dict) where:
            - from_time: arrival time of from_milestone
            - duration: to_time - from_time, or censored_value if censored
            - params_dict: merged parameters and time_series_parameters
        """
        for rec in self.iter_rollouts():
            results = rec.get("results")
            if not isinstance(results, dict):
                continue

            milestones = results.get("milestones", {})

            # Get from_milestone time
            from_info = milestones.get(from_milestone)
            if not isinstance(from_info, dict):
                continue
            from_time_val = from_info.get("time")
            try:
                from_time = float(from_time_val) if from_time_val is not None else np.nan
            except (TypeError, ValueError):
                continue
            if not np.isfinite(from_time):
                continue

            # Get to_milestone time
            to_info = milestones.get(to_milestone)
            to_time: Optional[float] = None
            if isinstance(to_info, dict):
                to_time_val = to_info.get("time")
                try:
                    to_time = float(to_time_val) if to_time_val is not None else None
                    if to_time is not None and not np.isfinite(to_time):
                        to_time = None
                except (TypeError, ValueError):
                    to_time = None

            # Build merged params dict
            params = {
                **rec.get("parameters", {}),
                **rec.get("time_series_parameters", {})
            }

            if to_time is None:
                # Censored: from_milestone achieved but to_milestone not
                if include_censored:
                    yield (from_time, censored_value, params)
            else:
                # Both milestones achieved
                duration = to_time - from_time
                if duration > 0:
                    yield (from_time, duration, params)

    def read_trajectories(
        self,
        metric_name: str
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]]]:
        """Read time series trajectories for a metric.

        NOTE: This method requires trajectory arrays and will automatically
        read from the full JSONL file (bypassing cache) when needed.

        Note: For more features (filtering, MSE values), use read_metric_trajectories().

        Args:
            metric_name: Name of metric (e.g., "horizon", "automation_fraction")

        Returns:
            times: Common time array in decimal years
            trajectories: List of metric arrays (one per rollout)
            aa_times: List of aa_time values (one per rollout)
        """
        times_arrays: List[np.ndarray] = []
        metric_arrays: List[np.ndarray] = []
        aa_times: List[Optional[float]] = []

        # Trajectories require full JSONL data (not in cache)
        rollouts_iter = (
            self._iter_normalized_rollouts_from_jsonl()
            if self._cache_data is not None
            else self.iter_normalized_rollouts()
        )

        for rollout in rollouts_iter:
            metric_array = rollout.results.get(metric_name)
            if metric_array is None:
                continue

            try:
                metric_arr = np.asarray(metric_array, dtype=float)
            except Exception:
                continue

            if len(rollout.times) != len(metric_arr):
                continue

            times_arrays.append(rollout.times)
            metric_arrays.append(metric_arr)
            aa_times.append(rollout.aa_time)

        if not times_arrays:
            return np.array([]), [], []

        # Find common time array (use the first one)
        common_times = times_arrays[0]

        # Interpolate all trajectories to common times if needed
        interpolated_trajectories = []
        for times_arr, metric_arr in zip(times_arrays, metric_arrays):
            if not np.array_equal(times_arr, common_times):
                # Interpolate to common times
                interpolated = np.interp(common_times, times_arr, metric_arr)
                interpolated_trajectories.append(interpolated)
            else:
                interpolated_trajectories.append(metric_arr)

        return common_times, interpolated_trajectories, aa_times

    def read_transition_data(
        self,
        from_milestone: str,
        to_milestone: str,
        include_censored: bool = False,
        inf_years_cap: Optional[float] = None,
        return_arrays: bool = False
    ) -> Tuple[np.ndarray, np.ndarray] | List[Tuple[float, float, float]]:
        """Read transition data from one milestone to another.

        Args:
            from_milestone: Starting milestone name
            to_milestone: Ending milestone name
            include_censored: If True, include censored transitions with inf_years_cap
            inf_years_cap: Duration value for censored transitions (default None)
            return_arrays: If True, return (from_times, durations) arrays

        Returns:
            If return_arrays=True: Tuple of (from_times, durations) as numpy arrays
            If return_arrays=False: List of (from_time, to_time, duration) tuples
        """
        if return_arrays:
            # Return arrays for scatter plots
            xs: List[float] = []
            ys: List[float] = []

            for from_time, to_time, _ in self._iter_transition(
                from_milestone, to_milestone, allow_censored=include_censored
            ):
                if to_time is None:
                    # Censored transition
                    if inf_years_cap is not None:
                        xs.append(from_time)
                        ys.append(inf_years_cap)
                else:
                    # Both milestones achieved
                    duration = to_time - from_time
                    xs.append(from_time)
                    ys.append(duration)

            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        # Return list of tuples (original behavior)
        transitions: List[Tuple[float, float, float]] = []
        for from_time, to_time, _ in self._iter_transition(
            from_milestone, to_milestone, allow_censored=False
        ):
            if to_time is not None:  # Only achieved transitions
                duration = to_time - from_time
                transitions.append((from_time, to_time, duration))

        return transitions

    def read_transition_durations(
        self,
        from_milestone: str,
        to_milestone: str,
        include_censored: bool = True,
        censored_value: Optional[float] = None,
    ) -> Tuple[List[float], int, Optional[float], int]:
        """Read transition durations between two milestones.

        Args:
            from_milestone: Starting milestone name
            to_milestone: Ending milestone name
            include_censored: If True, include censored transitions in output
            censored_value: If provided, use this fixed value for censored durations.
                           If None and include_censored=True, use sim_end - from_time.
                           For sensitivity analysis, use censored_value=1e10 so Spearman
                           correlation ranks censored cases at the end.

        Returns:
            durations: List of transition durations (years)
            num_censored: Number of censored transitions
            typical_sim_end: Typical simulation end time (median)
            num_out_of_order: Number of rollouts where milestones were achieved out of order
        """
        durations: List[float] = []
        num_censored: int = 0
        sim_end_times: List[float] = []

        stats: Dict[str, int] = {}
        for from_time, to_time, sim_end in self._iter_transition(
            from_milestone, to_milestone, allow_censored=True, stats=stats
        ):
            sim_end_times.append(sim_end)

            if to_time is None:
                # Censored transition
                num_censored += 1
                if include_censored:
                    if censored_value is not None:
                        duration = censored_value
                    else:
                        duration = sim_end - from_time
                    if duration > 0:
                        durations.append(duration)
            else:
                # Both milestones achieved
                duration = to_time - from_time
                durations.append(duration)

        typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
        num_out_of_order = stats.get("out_of_order", 0)
        return durations, num_censored, typical_sim_end, num_out_of_order

    def read_multiple_transition_durations(
        self,
        pairs: List[Tuple[str, str]],
        filter_milestone: Optional[str] = None,
        filter_by_year: Optional[float] = None,
    ) -> Tuple[
        List[str],
        List[List[float]],
        List[List[float]],
        List[int],
        List[int],
        List[int],
        Optional[float],
        Optional[float]
    ]:
        """Read transition durations for multiple milestone pairs efficiently.

        For each pair (A, B), computes finite durations B.time - A.time.

        Args:
            pairs: List of (from_milestone, to_milestone) tuples
            filter_milestone: Optional milestone to filter by
            filter_by_year: Optional year threshold for filtering

        Returns:
            labels: ["A to B", ...] - simplified display names
            durations_per_pair: finite durations (both achieved in order)
            durations_with_censored_per_pair: includes censored transitions
            num_b_not_achieved_per_pair: count where A achieved but B not
            num_b_before_a_per_pair: count where both achieved but B before A
            total_a_achieved_per_pair: rollouts where A was achieved
            typical_max_duration: typical max possible duration
            simulation_cutoff: simulation end time from metadata.json
        """
        from .helpers import simplify_milestone_name

        labels: List[str] = [
            f"{simplify_milestone_name(a)} to {simplify_milestone_name(b)}"
            for a, b in pairs
        ]
        durations_per_pair: List[List[float]] = [[] for _ in pairs]
        durations_with_censored_per_pair: List[List[float]] = [[] for _ in pairs]
        num_b_not_achieved: List[int] = [0 for _ in pairs]
        num_b_before_a: List[int] = [0 for _ in pairs]
        total_a_achieved: List[int] = [0 for _ in pairs]
        max_durations: List[float] = []

        # Read simulation cutoff from metadata.json (FIX: use rollouts_file)
        simulation_cutoff: Optional[float] = None
        metadata_path = self.rollouts_file.parent / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as mf:
                    metadata = json.load(mf)
                    time_range = metadata.get("time_range")
                    if time_range and isinstance(time_range, list):
                        if len(time_range) >= 2:
                            simulation_cutoff = float(time_range[1])
            except Exception:
                pass

        for rollout in self.iter_normalized_rollouts():
            # Filter by milestone-by-year if requested
            if filter_milestone is not None and filter_by_year is not None:
                filter_data = self._resolve_milestone(rollout, filter_milestone)
                if filter_data is None:
                    continue
                if filter_data['time'] > filter_by_year:
                    continue

            # Process all pairs for this rollout
            for idx, (a, b) in enumerate(pairs):
                # Resolve both milestones
                ta_data = self._resolve_milestone(rollout, a)
                if ta_data is None:
                    continue  # A not achieved, skip

                ta = ta_data['time']
                total_a_achieved[idx] += 1

                # Track max possible duration
                if rollout.simulation_end > ta:
                    max_durations.append(rollout.simulation_end - ta)

                tb_data = self._resolve_milestone(rollout, b)
                if tb_data is None:
                    # B not achieved (censored)
                    num_b_not_achieved[idx] += 1
                    if rollout.simulation_end > ta:
                        censored_dur = rollout.simulation_end - ta
                        durations_with_censored_per_pair[idx].append(censored_dur)
                    continue

                tb = tb_data['time']
                dur = tb - ta

                if dur <= 0.0:
                    # B before or at same time as A
                    num_b_before_a[idx] += 1
                    continue

                # Both achieved in correct order
                durations_per_pair[idx].append(dur)
                durations_with_censored_per_pair[idx].append(dur)

        typical_max_duration = (
            float(np.median(max_durations)) if max_durations else None
        )
        return (
            labels,
            durations_per_pair,
            durations_with_censored_per_pair,
            num_b_not_achieved,
            num_b_before_a,
            total_a_achieved,
            typical_max_duration,
            simulation_cutoff
        )

    def read_metric_trajectories(
        self,
        metric_name: str,
        include_aa_times: bool = True,
        include_mse: bool = True,
        filter_milestone: Optional[str] = None,
        filter_year: Optional[float] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]], List[Optional[float]]]:
        """Read time series trajectories for any metric field.

        NOTE: This method requires trajectory arrays and will automatically
        read from the full JSONL file (bypassing cache) when needed.

        Args:
            metric_name: Name of metric in results dict (e.g., 'horizon_lengths')
            include_aa_times: Include aa_time values in return
            include_mse: Include metr_mse values in return
            filter_milestone: Optional milestone to filter by
            filter_year: Optional year threshold for milestone filter

        Returns:
            times: Common time array in decimal years
            trajectories: List of metric value arrays (one per rollout)
            aa_times: List of aa_time values (empty list if not included)
            mse_values: List of METR MSE values (empty list if not included)
        """
        trajectories: List[np.ndarray] = []
        aa_times: List[Optional[float]] = []
        mse_values: List[Optional[float]] = []
        common_times: Optional[np.ndarray] = None

        # Trajectories require full JSONL data (not in cache)
        # Use JSONL iterator if cache is active
        rollouts_iter = (
            self._iter_normalized_rollouts_from_jsonl()
            if self._cache_data is not None
            else self.iter_normalized_rollouts()
        )

        for rollout in rollouts_iter:
            # Filter by milestone achievement if requested
            if filter_milestone is not None and filter_year is not None:
                milestone_data = self._resolve_milestone(rollout, filter_milestone)
                if milestone_data is None:
                    continue
                milestone_year = milestone_data['time']
                if abs(milestone_year - filter_year) > 0.5:
                    continue

            metric_values = rollout.results.get(metric_name)
            if metric_values is None:
                continue

            try:
                metric_arr = np.asarray(metric_values, dtype=float)
            except Exception:
                continue

            if metric_arr.ndim != 1 or rollout.times.size != metric_arr.size:
                continue

            if common_times is None:
                common_times = rollout.times

            trajectories.append(metric_arr)

            if include_aa_times:
                aa_times.append(rollout.aa_time)

            if include_mse:
                mse_val = rollout.results.get("metr_mse")
                try:
                    mse_values.append(
                        float(mse_val)
                        if mse_val is not None and np.isfinite(float(mse_val))
                        else None
                    )
                except Exception:
                    mse_values.append(None)

        if common_times is None or len(trajectories) == 0:
            raise ValueError(f"No '{metric_name}' trajectories found in rollouts file")

        return common_times, trajectories, aa_times, mse_values

    def read_metric_trajectories_with_milestone(
        self,
        metric_name: str,
        milestone_name: str,
        include_aa_times: bool = True,
        include_mse: bool = True,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Read time series trajectories with milestone timing info.

        Like read_metric_trajectories but also returns milestone arrival times.

        Args:
            metric_name: Name of metric in results dict (e.g., 'horizon_lengths')
            milestone_name: Name of milestone (e.g., 'AC')
            include_aa_times: Include aa_time values in return
            include_mse: Include metr_mse values in return

        Returns:
            times: Common time array in decimal years
            trajectories: List of metric value arrays (one per rollout)
            aa_times: List of aa_time values (empty list if not included)
            mse_values: List of METR MSE values (empty list if not included)
            milestone_times: List of milestone arrival times (None if not achieved)
        """
        trajectories: List[np.ndarray] = []
        aa_times: List[Optional[float]] = []
        mse_values: List[Optional[float]] = []
        milestone_times: List[Optional[float]] = []
        common_times: Optional[np.ndarray] = None

        # Trajectories require full JSONL data (not in cache)
        rollouts_iter = (
            self._iter_normalized_rollouts_from_jsonl()
            if self._cache_data is not None
            else self.iter_normalized_rollouts()
        )

        for rollout in rollouts_iter:
            metric_values = rollout.results.get(metric_name)
            if metric_values is None:
                continue

            try:
                metric_arr = np.asarray(metric_values, dtype=float)
            except Exception:
                continue

            if metric_arr.ndim != 1 or rollout.times.size != metric_arr.size:
                continue

            if common_times is None:
                common_times = rollout.times

            trajectories.append(metric_arr)

            if include_aa_times:
                aa_times.append(rollout.aa_time)

            if include_mse:
                mse_val = rollout.results.get("metr_mse")
                try:
                    mse_values.append(
                        float(mse_val)
                        if mse_val is not None and np.isfinite(float(mse_val))
                        else None
                    )
                except Exception:
                    mse_values.append(None)

            # Get milestone time
            milestone_data = self._resolve_milestone(rollout, milestone_name)
            if milestone_data is not None:
                milestone_times.append(milestone_data['time'])
            else:
                milestone_times.append(None)

        if common_times is None or len(trajectories) == 0:
            raise ValueError(f"No '{metric_name}' trajectories found in rollouts file")

        return common_times, trajectories, aa_times, mse_values, milestone_times

    def read_aa_times(self) -> Tuple[List[float], int, Optional[float]]:
        """Read SC/AA arrival times (aa_time field).

        Returns:
            aa_times: List of SC arrival times
            num_no_sc: Count of rollouts where SC was not achieved
            typical_sim_end: Typical simulation end time (median)
        """
        aa_times: List[float] = []
        num_no_sc: int = 0
        sim_end_times: List[float] = []

        for rollout in self.iter_normalized_rollouts():
            sim_end_times.append(rollout.simulation_end)

            if rollout.aa_time is not None:
                aa_times.append(rollout.aa_time)
            else:
                num_no_sc += 1

        typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
        return aa_times, num_no_sc, typical_sim_end

    def read_milestone_scatter_data(
        self,
        milestone1: str,
        milestone2: str
    ) -> Tuple[List[float], List[float]]:
        """Read scatter plot data for two milestones.

        Args:
            milestone1: First milestone name
            milestone2: Second milestone name

        Returns:
            times1: Arrival times for milestone1
            times2: Arrival times for milestone2
        """
        times1: List[float] = []
        times2: List[float] = []

        for rec in self.iter_rollouts():
            results = rec["results"]
            milestones = results.get("milestones", {})

            # Get milestone1 time
            info1 = milestones.get(milestone1)
            if not isinstance(info1, dict):
                continue
            t1 = info1.get("time")
            try:
                x1 = float(t1) if t1 is not None else np.nan
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x1):
                continue

            # Get milestone2 time
            info2 = milestones.get(milestone2)
            if not isinstance(info2, dict):
                continue
            t2 = info2.get("time")
            try:
                x2 = float(t2) if t2 is not None else np.nan
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x2):
                continue

            times1.append(x1)
            times2.append(x2)

        return times1, times2

    def count_rollouts(self) -> int:
        """Count total number of valid rollouts in file.

        Returns:
            Number of rollouts
        """
        count = 0
        for _ in self.iter_rollouts():
            count += 1
        return count

    def read_parameter_outcome_data(
        self,
        parameter_names: List[str],
        outcome_specs: List[Tuple[str, str, str]]
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Read parameter values and outcome metrics for continuous plots.

        Efficiently extracts parameter values and computes outcome metrics
        in a single pass through the rollouts.

        Args:
            parameter_names: List of parameter names to extract (from parameters,
                            time_series_parameters, or results)
            outcome_specs: List of (metric_name, from_milestone, to_milestone) tuples.
                          For transition metrics like "p_1yr_takeoff", use
                          ("p_1yr_takeoff_AC", "AC", "TED-AI").
                          For direct metrics like "aa_time", use ("aa_time", "", "").

        Returns:
            Tuple of:
              - param_data: Dict mapping parameter name to list of values
              - outcome_data: Dict mapping metric name to list of values (0/1 for
                             transition metrics, raw values for direct metrics)
        """
        param_data: Dict[str, List[float]] = {name: [] for name in parameter_names}
        outcome_data: Dict[str, List[float]] = {spec[0]: [] for spec in outcome_specs}

        for rec in self.iter_rollouts():
            params = rec.get("parameters", {})
            ts_params = rec.get("time_series_parameters", {})
            results = rec.get("results", {})
            all_params = {**params, **ts_params, **results}

            # Extract parameter values
            param_vals = {}
            skip_record = False
            for param_name in parameter_names:
                val = all_params.get(param_name)
                if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                    skip_record = True
                    break
                param_vals[param_name] = float(val)

            if skip_record:
                continue

            # Record parameter values
            for param_name in parameter_names:
                param_data[param_name].append(param_vals[param_name])

            # Extract outcome metrics
            milestones = results.get("milestones", {})
            for metric_name, from_ms, to_ms in outcome_specs:
                if from_ms and to_ms:
                    # Transition metric (e.g., p_1yr_takeoff)
                    from_data = milestones.get(from_ms)
                    to_data = milestones.get(to_ms)

                    if from_data and to_data:
                        from_time = from_data.get("time")
                        to_time = to_data.get("time")
                        if from_time is not None and to_time is not None:
                            duration = float(to_time) - float(from_time)
                            outcome_data[metric_name].append(1.0 if duration <= 1.0 else 0.0)
                        else:
                            outcome_data[metric_name].append(np.nan)
                    else:
                        outcome_data[metric_name].append(np.nan)
                else:
                    # Direct metric from results
                    val = all_params.get(metric_name)
                    if val is not None and isinstance(val, (int, float)):
                        outcome_data[metric_name].append(float(val))
                    else:
                        outcome_data[metric_name].append(np.nan)

        return param_data, outcome_data

    def read_multiple_milestone_times(
        self,
        milestone_names: List[str]
    ) -> Dict[str, List[float]]:
        """Read arrival times for multiple milestones efficiently.

        Args:
            milestone_names: List of milestone names to extract

        Returns:
            Dictionary mapping milestone name to list of arrival times
        """
        times_map, _, _, _ = self.read_milestone_times_batch(milestone_names)
        return times_map

    def read_milestone_times_batch(
        self,
        milestone_names: Sequence[str]
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, int],
        Optional[float],
        int
    ]:
        """Read arrival times for multiple milestones in a single pass.

        Args:
            milestone_names: List of milestone names to extract

        Returns:
            Tuple of:
              - dict mapping milestone to arrival times
              - dict mapping milestone to number of rollouts that missed it
              - typical simulation end time across processed rollouts
              - total number of processed rollouts
        """
        unique_names = list(dict.fromkeys(milestone_names))
        times_map: Dict[str, List[float]] = {name: [] for name in unique_names}
        not_achieved: Dict[str, int] = {name: 0 for name in unique_names}
        sim_end_times: List[float] = []
        total_rollouts = 0

        if not unique_names:
            for rollout in self.iter_normalized_rollouts():
                total_rollouts += 1
                sim_end_times.append(rollout.simulation_end)
            typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
            return times_map, not_achieved, typical_sim_end, total_rollouts

        for rollout in self.iter_normalized_rollouts():
            total_rollouts += 1
            sim_end_times.append(rollout.simulation_end)
            for milestone_name in unique_names:
                result = self._resolve_milestone(
                    rollout,
                    milestone_name,
                )
                if result is None:
                    not_achieved[milestone_name] += 1
                    continue
                times_map[milestone_name].append(result['time'])

        typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
        return times_map, not_achieved, typical_sim_end, total_rollouts

    def read_metric_at_milestone(
        self,
        metric_name: str,
        milestone_key: str = "aa_time",
        clip_min: float = 0.001,
        clip_max: Optional[float] = None
    ) -> List[float]:
        """Read metric value at milestone time for each rollout.

        Interpolates metric trajectory to find value at milestone time.

        Args:
            metric_name: Name of metric trajectory (e.g., "horizon_lengths", "automation_fraction")
            milestone_key: Key for milestone time (default "aa_time" for AC milestone)
            clip_min: Minimum value for clipping (default 0.001)
            clip_max: Maximum value for clipping (default None for no upper clip)

        Returns:
            List of metric values at milestone time
        """
        values: List[float] = []

        for rec in self.iter_rollouts():
            results = rec["results"]

            times = results.get("times")
            metric = results.get(metric_name)
            milestone_time = results.get(milestone_key)

            if times is None or metric is None or milestone_time is None:
                continue

            try:
                times_arr = np.asarray(times, dtype=float)
                metric_arr = np.asarray(metric, dtype=float)
                ms_t = float(milestone_time)
            except (TypeError, ValueError):
                continue

            if not (np.isfinite(ms_t) and times_arr.ndim == 1 and
                    metric_arr.ndim == 1 and times_arr.size == metric_arr.size):
                continue

            # Handle non-finite values
            if np.any(~np.isfinite(metric_arr)):
                valid = np.isfinite(metric_arr)
                if valid.sum() < 2:
                    continue
                times_arr = times_arr[valid]
                metric_arr = metric_arr[valid]

            # Clip metric values
            if clip_max is not None:
                metric_arr = np.clip(metric_arr, clip_min, clip_max)
            elif clip_min is not None:
                metric_arr = np.maximum(metric_arr, clip_min)

            # Interpolate metric at milestone time
            if ms_t <= times_arr.min():
                val = metric_arr[0]
            elif ms_t >= times_arr.max():
                val = metric_arr[-1]
            else:
                val = float(np.interp(ms_t, times_arr, metric_arr))

            if np.isfinite(val):
                values.append(val)

        return values

    def _iter_takeoff_metric_data(
        self,
        filter_milestone: Optional[str] = None,
        filter_year: Optional[int] = None,
    ) -> Iterator[Tuple[float, float]]:
        """Internal iterator yielding (m, beta) pairs for takeoff metric computations.

        Centralizes the extraction of slope_times_log_f (m) and beta_software from
        rollouts, with optional filtering by milestone achievement year.

        Args:
            filter_milestone: If provided, only include rollouts where this milestone
                            was achieved in the specified year
            filter_year: Year to filter by (requires filter_milestone)

        Yields:
            Tuples of (m, beta) where:
              - m = slope_times_log_f
              - beta = beta_software
        """
        for rec in self.iter_rollouts():
            results = rec.get("results")
            if not isinstance(results, dict):
                continue

            # Apply milestone/year filter if specified
            if filter_milestone is not None and filter_year is not None:
                milestones = results.get("milestones", {})
                ms_info = milestones.get(filter_milestone)
                if not isinstance(ms_info, dict):
                    continue
                ms_time = ms_info.get("time")
                try:
                    ms_time_float = float(ms_time) if ms_time is not None else np.nan
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(ms_time_float):
                    continue
                # Check if milestone was achieved in the target year
                if not (float(filter_year) <= ms_time_float < float(filter_year + 1)):
                    continue

            # Extract the two underlying values
            beta_software = results.get("beta_software")
            slope_times_log_f = results.get("slope_times_log_f")

            if slope_times_log_f is None or beta_software is None:
                continue

            try:
                m = float(slope_times_log_f)
                beta = float(beta_software)
                if np.isfinite(m) and np.isfinite(beta):
                    yield (m, beta)
            except (TypeError, ValueError):
                continue

    def read_m_over_beta(
        self,
        filter_milestone: Optional[str] = None,
        filter_year: Optional[int] = None,
    ) -> Tuple[List[float], int]:
        """Read m/beta (slope_times_log_f / beta_software) values from rollouts.

        The m/beta ratio is a key metric for understanding takeoff dynamics.
        When m/beta > 1, growth is superexponential.

        Args:
            filter_milestone: If provided, only include rollouts where this milestone
                            was achieved in filter_year (e.g., "AI2027-SC")
            filter_year: Year to filter by (requires filter_milestone)

        Returns:
            m_over_beta_values: list of m/beta ratio values
            num_skipped: count of rollouts where the value could not be computed
        """
        values: List[float] = []
        num_skipped: int = 0
        num_processed: int = 0

        for m, beta in self._iter_takeoff_metric_data(filter_milestone, filter_year):
            num_processed += 1
            if beta > 0:
                m_over_beta = m / beta
                if np.isfinite(m_over_beta):
                    values.append(m_over_beta)
                else:
                    num_skipped += 1
            else:
                num_skipped += 1

        return values, num_skipped

    def read_takeoff_exponent(
        self,
        filter_milestone: Optional[str] = None,
        filter_year: Optional[int] = None,
    ) -> Tuple[List[float], int]:
        """Read takeoff exponent 2^(1 - beta/m) values from rollouts.

        The takeoff exponent represents the asymptotic exponent in progress growth.
        When m/beta > 1, the exponent > 1 indicating superexponential takeoff.

        Args:
            filter_milestone: If provided, only include rollouts where this milestone
                            was achieved in filter_year
            filter_year: Year to filter by (requires filter_milestone)

        Returns:
            takeoff_exponent_values: list of 2^(1 - beta/m) values
            num_skipped: count of rollouts where the value could not be computed
        """
        values: List[float] = []
        num_skipped: int = 0

        for m, beta in self._iter_takeoff_metric_data(filter_milestone, filter_year):
            if m > 0:
                exponent = 1 - beta / m
                takeoff_exp = 2 ** exponent
                if np.isfinite(takeoff_exp) and takeoff_exp > 0:
                    values.append(takeoff_exp)
                else:
                    num_skipped += 1
            else:
                num_skipped += 1

        return values, num_skipped

    def read_uplift_ratio(
        self,
        filter_milestone: Optional[str] = None,
        filter_year: Optional[int] = None,
    ) -> Tuple[List[float], int]:
        """Read uplift ratio 2^(beta/m - 1) values from rollouts.

        The uplift ratio represents the ratio of successive uplift doublings
        during software intelligence explosion. When beta/m < 1, this value < 1,
        indicating superexponential growth.

        Args:
            filter_milestone: If provided, only include rollouts where this milestone
                            was achieved in filter_year
            filter_year: Year to filter by (requires filter_milestone)

        Returns:
            uplift_ratio_values: list of 2^(beta/m - 1) values
            num_skipped: count of rollouts where the value could not be computed
        """
        values: List[float] = []
        num_skipped: int = 0

        for m, beta in self._iter_takeoff_metric_data(filter_milestone, filter_year):
            if m > 0:
                exponent = beta / m - 1
                uplift_ratio = 2 ** exponent
                if np.isfinite(uplift_ratio) and uplift_ratio > 0:
                    values.append(uplift_ratio)
                else:
                    num_skipped += 1
            else:
                num_skipped += 1

        return values, num_skipped

    def read_takeoff_metrics_conditional(
        self,
        filter_milestone: str,
        filter_year: int,
    ) -> Tuple[List[float], List[float], List[float], int, int, int]:
        """Read all takeoff metrics conditional on milestone achievement year.

        Convenience method that returns all three metrics in one pass, along with
        statistics about the filtering.

        Args:
            filter_milestone: Milestone to filter by (e.g., "AI2027-SC")
            filter_year: Year the milestone must be achieved in

        Returns:
            m_over_beta_values: list of m/beta ratio values
            takeoff_exponent_values: list of 2^(1 - beta/m) values
            uplift_ratio_values: list of 2^(beta/m - 1) values
            num_skipped: count of rollouts where metrics could not be computed
            total_in_year: total rollouts where milestone was in target year
            total_rollouts: total number of rollouts processed
        """
        m_over_beta_values: List[float] = []
        takeoff_exponent_values: List[float] = []
        uplift_ratio_values: List[float] = []
        num_skipped: int = 0
        total_in_year: int = 0
        total_rollouts: int = 0

        for rec in self.iter_rollouts():
            total_rollouts += 1
            results = rec.get("results")
            if not isinstance(results, dict):
                continue

            # Check milestone achievement
            milestones = results.get("milestones", {})
            ms_info = milestones.get(filter_milestone)
            if not isinstance(ms_info, dict):
                continue
            ms_time = ms_info.get("time")
            try:
                ms_time_float = float(ms_time) if ms_time is not None else np.nan
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ms_time_float):
                continue
            if not (float(filter_year) <= ms_time_float < float(filter_year + 1)):
                continue

            total_in_year += 1

            # Extract underlying values
            beta_software = results.get("beta_software")
            slope_times_log_f = results.get("slope_times_log_f")

            if slope_times_log_f is None or beta_software is None:
                num_skipped += 1
                continue

            try:
                m = float(slope_times_log_f)
                beta = float(beta_software)

                if not (np.isfinite(m) and np.isfinite(beta) and m > 0 and beta > 0):
                    num_skipped += 1
                    continue

                # Compute all three metrics
                m_over_beta = m / beta
                takeoff_exp = 2 ** (1 - beta / m)
                uplift_ratio = 2 ** (beta / m - 1)

                if (np.isfinite(m_over_beta) and np.isfinite(takeoff_exp) and
                        np.isfinite(uplift_ratio) and takeoff_exp > 0 and uplift_ratio > 0):
                    m_over_beta_values.append(m_over_beta)
                    takeoff_exponent_values.append(takeoff_exp)
                    uplift_ratio_values.append(uplift_ratio)
                else:
                    num_skipped += 1
            except (TypeError, ValueError):
                num_skipped += 1
                continue

        return (
            m_over_beta_values,
            takeoff_exponent_values,
            uplift_ratio_values,
            num_skipped,
            total_in_year,
            total_rollouts,
        )

    def export_cache(
        self,
        output_path: Optional[Path] = None
    ) -> Path:
        """Export a lightweight cache excluding only trajectory arrays.

        Creates a compact JSON file containing all rollout data EXCEPT large
        trajectory arrays. This includes:
        - sample_id, parameters, time_series_parameters
        - milestones dict, aa_time, simulation_end
        - All scalar values from results dict
        - Error information for failed rollouts

        This is typically 100-1000x smaller and much faster to load than the
        full rollouts.jsonl for scripts that only need milestone and scalar data.

        Args:
            output_path: Optional path for output file. If None, creates
                        'rollouts.cache.json' in same directory as rollouts file

        Returns:
            Path to the created cache file

        Example:
            # Create cache
            reader = RolloutsReader(rollouts_file)
            cache_file = reader.export_cache()

            # Later, RolloutsReader automatically uses cache
            reader = RolloutsReader(rollouts_file)  # Uses cache transparently
            times, _, _ = reader.read_milestone_times("AC")
        """
        if output_path is None:
            output_path = self.rollouts_file.parent / "rollouts.cache.json"

        # List of known trajectory fields to exclude (large arrays)
        TRAJECTORY_FIELDS = {
            'times', 'progress', 'research_stock', 'automation_fraction',
            'ai_research_taste', 'ai_research_taste_sd', 'ai_research_taste_quantile',
            'aggregate_research_taste', 'progress_rates', 'etsp_rates', 'research_efforts',
            'coding_labors', 'serial_coding_labors', 'coding_labors_with_present_resources',
            'software_progress_rates', 'software_efficiency', 'human_only_progress_rates',
            'ai_labor_contributions', 'human_labor_contributions',
            'ai_coding_labor_multipliers', 'ai_coding_labor_mult_ref_present_day',
            'effective_compute', 'horizon_lengths', 'metr_mse_trajectory',
            'compute_investment', 'total_research_labor'
        }

        cache_records = []

        # Process all rollouts (including errors)
        for rec in self.iter_all_records():
            cache_rec = {
                'sample_id': rec.get('sample_id'),
                'parameters': rec.get('parameters'),
                'time_series_parameters': rec.get('time_series_parameters'),
            }

            results = rec.get('results')

            if isinstance(results, dict):
                # Keep all non-trajectory fields from results
                filtered_results = {}
                for key, value in results.items():
                    if key not in TRAJECTORY_FIELDS:
                        # Keep scalars, dicts (like milestones), and small data
                        if not isinstance(value, list):
                            filtered_results[key] = value
                        elif len(value) < 10:  # Keep small lists
                            filtered_results[key] = value

                # Always include simulation_end derived from times
                times_list = results.get('times')
                if times_list and len(times_list) > 0:
                    try:
                        filtered_results['simulation_end'] = float(times_list[-1])
                    except (TypeError, ValueError, IndexError):
                        pass

                cache_rec['results'] = filtered_results
            else:
                # Error case - keep error info
                cache_rec['error'] = rec.get('error')
                cache_rec['results'] = {}

            cache_records.append(cache_rec)

        # Write as compact JSON
        cache_data = {
            'version': 1,
            'source': str(self.rollouts_file.name),
            'rollouts': cache_records
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, separators=(',', ':'))

        return output_path

    def list_milestone_names(self) -> List[str]:
        """List all milestone names found in the rollouts file.

        Returns:
            Sorted list of unique milestone names
        """
        names: set = set()
        for rec in self.iter_rollouts():
            results = rec.get("results")
            if not isinstance(results, dict):
                continue
            milestones = results.get("milestones")
            if isinstance(milestones, dict):
                for k in milestones.keys():
                    names.add(str(k))
        return sorted(names)

    def read_x_years_in_1_year(self) -> List[float]:
        """Read progress trajectories and compute 'X years in 1 year' metric.

        For each rollout that achieved AC, computes the ratio of max OOMs
        crossed in any 1-year window to the OOMs crossed in the present year.

        Returns:
            List of X values (one per rollout that achieved AC with valid data)
        """
        x_values: List[float] = []

        # Must read from JSONL for trajectory data
        rollouts_iter = (
            self._iter_rollouts_from_jsonl()
            if self._cache_data is not None
            else self.iter_rollouts()
        )

        for rec in rollouts_iter:
            results = rec.get("results")
            if not isinstance(results, dict):
                continue

            # Check if AC was achieved
            aa_time = results.get("aa_time")
            try:
                aa_t = float(aa_time) if aa_time is not None else np.nan
            except (TypeError, ValueError):
                aa_t = np.nan
            if not np.isfinite(aa_t):
                continue

            times = results.get("times")
            progress = results.get("progress")
            if times is None or progress is None:
                continue

            try:
                times_arr = np.asarray(times, dtype=float)
                progress_arr = np.asarray(progress, dtype=float)
            except Exception:
                continue

            if (times_arr.ndim != 1 or progress_arr.ndim != 1 or
                    times_arr.size != progress_arr.size or times_arr.size < 2):
                continue

            # Compute X years in 1 year metric
            x_val = self._compute_x_years_in_1_year(times_arr, progress_arr, 0)
            if np.isfinite(x_val) and x_val > 0:
                x_values.append(x_val)

        return x_values

    @staticmethod
    def _compute_x_years_in_1_year(
        times: np.ndarray,
        progress: np.ndarray,
        current_time_idx: int = 0,
    ) -> float:
        """Compute maximum 'X years in 1 year' metric for a trajectory.

        Finds the 1-year window with the maximum OOMs crossed and compares
        it to the OOMs crossed in the present year.

        Args:
            times: array of time points (decimal years)
            progress: array of progress values (in OOMs or log10 scale)
            current_time_idx: index representing the "present" time

        Returns:
            X value such that "X years in 1 year" happened
        """
        if len(times) < 2 or len(progress) < 2:
            return np.nan

        max_ooms_per_year = 0.0

        for i in range(len(times)):
            target_time = times[i] + 1.0
            end_idx = None
            for j in range(i + 1, len(times)):
                if times[j] >= target_time:
                    end_idx = j
                    break

            if end_idx is None:
                if i < len(times) - 1:
                    end_idx = len(times) - 1
                else:
                    continue

            ooms_in_window = progress[end_idx] - progress[i]
            max_ooms_per_year = max(max_ooms_per_year, float(ooms_in_window))

        if current_time_idx >= len(times) - 1:
            return np.nan

        target_time = times[current_time_idx] + 1.0
        current_end_idx = None
        for j in range(current_time_idx + 1, len(times)):
            if times[j] >= target_time:
                current_end_idx = j
                break

        if current_end_idx is None:
            current_end_idx = len(times) - 1

        current_ooms = progress[current_end_idx] - progress[current_time_idx]

        if current_ooms <= 0:
            return np.nan

        return float(max_ooms_per_year / current_ooms)
