#!/usr/bin/env python3
"""
Generate per-AC-anchor-mode breakout distribution CSVs for milestone PDFs.

For simulation runs whose sampling config draws ``ac_anchor_mode`` from a
mixture (e.g. uplift-solved / time-horizon / revenue), this script splits the
milestone arrival-time distributions by the sampled anchor mode and emits
long-format CSVs alongside the combined overlay files:

- milestone_pdfs_mode_breakout_distributions.csv
    ac_anchor_mode,time_decimal_year,<milestone columns...>
- milestone_pdfs_mode_breakout_cdf.csv
    ac_anchor_mode,time_decimal_year,<milestone columns...>
- milestone_pdfs_mode_breakout_statistics.csv
    ac_anchor_mode,milestone_name,num_achieved,num_not_achieved,
    total_rollouts,num_failed_rollouts,num_sampled,sampled_weight_pct,
    achievement_rate_pct,mode,p10,p50,p90

Denominator / plateau semantics (mirrors the combined overlay convention):
- Failed rollouts (results == null) are EXCLUDED from each mode's denominator,
  exactly as the combined export excludes them. ``total_rollouts`` is the
  number of successful rollouts sampled with that mode.
- Each mode's empirical CDF therefore plateaus at that mode's OWN achievement
  rate: cdf_mode(t) = (# mode rollouts achieving milestone by t) / total_rollouts.
- Each mode's KDE PDF is scaled by that mode's own achievement rate (in %),
  matching the combined file's scaling, so the frontend can renormalize the
  same way it does for the combined curves.
- ``num_failed_rollouts`` and ``num_sampled`` (successful + failed) are
  reported so the UI can be explicit about what was dropped.
- ``sampled_weight_pct`` is the mode's share of ALL sampled rollouts
  (including failed ones), i.e. the realized mixture weight.

Design decision (user, 2026-08-07): failed/censored rollouts stay EXCLUDED
from the per-mode denominator; they are NOT booked as never-achieving. This
matches the combined-topline convention, and the Aug 7 censoring-impact
analysis showed the censored uplift-solved mass is fast-takeoff — counting it
as never-achieving would push the displayed distribution slow in the same
direction the censoring already biases.

If the run is not a mixture (fewer than 2 distinct modes with >= 2 achieved
samples for any milestone), no files are written and the script exits 0.

Usage:
    python scripts/export/generate_mode_breakout_distributions.py \
        --rollouts outputs/20260807/rollouts.jsonl --out-dir outputs/20260807 --monthly
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export.generate_overlay_distributions import (
    OVERLAY_ALLOWED_MILESTONES,
    compute_empirical_cdf,
    prepare_gamma_pdf,
)
from plotting_utils.helpers import load_present_day


def _iter_raw_records(rollouts_file: Path):
    """Yield every JSON record from the raw rollouts.jsonl."""
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_cache_records(rollouts_file: Path) -> Optional[list]:
    """Return the sibling rollouts.cache.json's records when they are usable.

    The cache (created by batch_rollout post-processing) keeps everything this
    script consumes — sampled parameters, milestone times, simulation_end,
    failure information — at ~2% of the raw file's size, so reading it turns a
    multi-GB parse into seconds. Returns None (=> read the raw JSONL) when the
    cache is absent, unreadable, or from an older layout that lacks sampled
    parameters, since the breakout is meaningless without ac_anchor_mode.
    """
    cache_path = rollouts_file.parent / "rollouts.cache.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Cache unreadable ({e}); falling back to raw rollouts.jsonl")
        return None
    records = cache.get("rollouts") if isinstance(cache, dict) else None
    if not isinstance(records, list) or not records:
        return None
    if not any(isinstance(r, dict) and r.get("parameters") for r in records[:50]):
        print("  Cache lacks sampled parameters; falling back to raw rollouts.jsonl")
        return None
    return records


def read_mode_grouped_milestone_times(
    rollouts_file: Path,
    use_cache: bool = True,
) -> Tuple[
    Dict[str, Dict[str, List[float]]],  # mode -> milestone -> achieved times
    Dict[str, Dict[str, int]],  # mode -> milestone -> num not achieved
    Dict[str, int],  # mode -> num successful rollouts
    Dict[str, int],  # mode -> num failed rollouts (results == null)
    Optional[float],  # typical simulation end (median across successful)
]:
    """Read milestone times grouped by ac_anchor_mode.

    Prefers the sibling rollouts.cache.json (which carries sampled parameters,
    milestone times and simulation_end) and falls back to streaming the raw
    JSONL when no usable cache exists or use_cache is False. The two paths are
    semantically identical; note the cache may represent a failed rollout's
    results as {} rather than null, so "failed" is `not results` on both paths
    (a successful rollout always has a populated results dict).
    """
    records = _load_cache_records(rollouts_file) if use_cache else None
    if records is not None:
        print(f"  Reading {len(records)} records from rollouts.cache.json")
        record_iter = iter(records)
    else:
        record_iter = _iter_raw_records(rollouts_file)

    times_by_mode: Dict[str, Dict[str, List[float]]] = {}
    not_achieved_by_mode: Dict[str, Dict[str, int]] = {}
    num_success: Dict[str, int] = {}
    num_failed: Dict[str, int] = {}
    sim_ends: List[float] = []

    for rec in record_iter:
            params = rec.get("parameters") or {}
            mode = params.get("ac_anchor_mode")
            if not isinstance(mode, str) or not mode:
                mode = "unknown"

            results = rec.get("results")
            if not isinstance(results, dict) or not results:
                # Failed rollout: excluded from denominators (matches the
                # combined overlay convention), but counted for reporting.
                num_failed[mode] = num_failed.get(mode, 0) + 1
                continue

            num_success[mode] = num_success.get(mode, 0) + 1

            # The cache stores the scalar results['simulation_end']; the raw
            # records carry the full times array instead.
            sim_end_val = results.get("simulation_end")
            if sim_end_val is None:
                times_list = results.get("times")
                if isinstance(times_list, list) and times_list:
                    sim_end_val = times_list[-1]
            if sim_end_val is not None:
                try:
                    sim_ends.append(float(sim_end_val))
                except (TypeError, ValueError):
                    pass

            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                milestones = {}

            mode_times = times_by_mode.setdefault(mode, {})
            mode_not_achieved = not_achieved_by_mode.setdefault(mode, {})

            for milestone in OVERLAY_ALLOWED_MILESTONES:
                info = milestones.get(milestone)
                time_val = info.get("time") if isinstance(info, dict) else None
                try:
                    time_float = float(time_val) if time_val is not None else math.nan
                except (TypeError, ValueError):
                    time_float = math.nan

                if math.isfinite(time_float):
                    mode_times.setdefault(milestone, []).append(time_float)
                else:
                    mode_not_achieved[milestone] = mode_not_achieved.get(milestone, 0) + 1

    typical_sim_end = float(np.median(sim_ends)) if sim_ends else None
    return times_by_mode, not_achieved_by_mode, num_success, num_failed, typical_sim_end


def generate_mode_breakout_distributions(
    rollouts_file: Path,
    out_dir: Path,
    monthly_sampling: bool = False,
    use_cache: bool = True,
) -> bool:
    """Generate per-mode breakout CSVs. Returns True on success (including the
    benign 'not a mixture' case where nothing is written)."""
    if not rollouts_file.exists():
        print(f"Error: rollouts.jsonl not found: {rollouts_file}")
        return False

    present_day = load_present_day(rollouts_file.parent)
    print(f"Using present_day: {present_day:.4f}")

    (
        times_by_mode,
        not_achieved_by_mode,
        num_success,
        num_failed,
        typical_sim_end,
    ) = read_mode_grouped_milestone_times(rollouts_file, use_cache=use_cache)

    all_modes = sorted(set(num_success) | set(num_failed))
    total_sampled = sum(num_success.values()) + sum(num_failed.values())
    print(f"Total sampled rollouts: {total_sampled}")
    for mode in all_modes:
        ok = num_success.get(mode, 0)
        bad = num_failed.get(mode, 0)
        print(f"  {mode}: {ok} successful, {bad} failed ({(ok + bad) / total_sampled * 100:.1f}% of sampled)")

    if len(all_modes) < 2:
        print("Run is not an ac_anchor_mode mixture (<2 distinct modes); nothing to do.")
        return True

    boundary = float(present_day)

    # Fit per-(mode, milestone) KDEs
    # kde_data[(mode, milestone)] = (xs, pdf_values)
    kde_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    stats_rows: List[Dict] = []

    for mode in all_modes:
        mode_sampled = num_success.get(mode, 0) + num_failed.get(mode, 0)
        for milestone in OVERLAY_ALLOWED_MILESTONES:
            times = times_by_mode.get(mode, {}).get(milestone, [])
            n_not_achieved = not_achieved_by_mode.get(mode, {}).get(milestone, 0)
            total_runs = len(times) + n_not_achieved  # successful rollouts in mode

            if len(times) < 2:
                if total_runs > 0:
                    print(f"  {mode}/{milestone}: insufficient data ({len(times)} achieved), skipping")
                continue

            raw_data = np.asarray(times, dtype=float)
            data = raw_data[raw_data >= boundary]
            dropped = raw_data.size - data.size
            if dropped:
                print(f"  {mode}/{milestone}: dropped {dropped} samples before present_day")
            if data.size < 2:
                print(f"  {mode}/{milestone}: insufficient data after filtering, skipping")
                continue

            prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0

            # Percentiles include not-achieved rollouts pinned at sim end,
            # matching the combined overlay statistics convention.
            if n_not_achieved > 0 and typical_sim_end is not None:
                combined_data = np.concatenate([raw_data, np.full(n_not_achieved, typical_sim_end)])
            else:
                combined_data = raw_data
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            try:
                xs, pdf_values, _xs_plot, _pdf_plot, mode_year = prepare_gamma_pdf(
                    data,
                    boundary,
                    monthly_sampling,
                    prob_achieved * 100.0,
                    plot_max_year=2050,
                )
            except Exception as e:  # noqa: BLE001 - mirror combined script's behavior
                print(f"  {mode}/{milestone}: error generating KDE: {e}")
                continue

            kde_data[(mode, milestone)] = (xs, pdf_values)
            stats_rows.append(
                {
                    "ac_anchor_mode": mode,
                    "milestone_name": milestone,
                    "num_achieved": len(times),
                    "num_not_achieved": n_not_achieved,
                    "total_rollouts": total_runs,
                    "num_failed_rollouts": num_failed.get(mode, 0),
                    "num_sampled": mode_sampled,
                    "sampled_weight_pct": mode_sampled / total_sampled * 100.0 if total_sampled else 0.0,
                    "achievement_rate_pct": prob_achieved * 100.0,
                    "mode": mode_year,
                    "p10": q10,
                    "p50": q50,
                    "p90": q90,
                }
            )

    modes_with_data = sorted({mode for mode, _ in kde_data})
    if len(modes_with_data) < 2:
        print("Fewer than 2 modes with sufficient data; nothing to do.")
        return True

    milestones_with_data = [
        m for m in OVERLAY_ALLOWED_MILESTONES if any((mode, m) in kde_data for mode in modes_with_data)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Common time grid per mode: use that mode's first milestone grid, the same
    # convention the combined export uses (first milestone's xs array).
    def mode_grid(mode: str) -> np.ndarray:
        for m in milestones_with_data:
            if (mode, m) in kde_data:
                return kde_data[(mode, m)][0]
        return np.array([])

    # Save distributions CSV (long format: one block of rows per mode)
    dist_path = out_dir / "milestone_pdfs_mode_breakout_distributions.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ac_anchor_mode", "time_decimal_year"] + milestones_with_data)
        for mode in modes_with_data:
            grid = mode_grid(mode)
            for t in grid:
                row = [mode, f"{t:.6f}"]
                for milestone in milestones_with_data:
                    entry = kde_data.get((mode, milestone))
                    if entry is None:
                        row.append("")
                        continue
                    xs, pdf_vals = entry
                    pdf_val = float(np.interp(t, xs, pdf_vals, left=0.0, right=0.0))
                    row.append(f"{pdf_val:.10f}")
                writer.writerow(row)
    print(f"\nSaved mode-breakout distributions to: {dist_path}")

    # Save CDF CSV (empirical, plateaus at each mode's own achievement rate)
    cdf_path = out_dir / "milestone_pdfs_mode_breakout_cdf.csv"
    with cdf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ac_anchor_mode", "time_decimal_year"] + milestones_with_data)
        for mode in modes_with_data:
            grid = mode_grid(mode)
            # Precompute sorted times + totals per milestone for this mode
            per_milestone: Dict[str, Tuple[np.ndarray, int]] = {}
            for milestone in milestones_with_data:
                times = times_by_mode.get(mode, {}).get(milestone, [])
                n_not_achieved = not_achieved_by_mode.get(mode, {}).get(milestone, 0)
                per_milestone[milestone] = (
                    np.asarray(times, dtype=float),
                    len(times) + n_not_achieved,
                )
            for t in grid:
                row = [mode, f"{t:.6f}"]
                for milestone in milestones_with_data:
                    times_array, total_runs = per_milestone[milestone]
                    if total_runs == 0:
                        row.append("")
                        continue
                    cdf_val = compute_empirical_cdf(times_array, np.array([t]), total_runs)[0]
                    row.append(f"{cdf_val:.10f}")
                writer.writerow(row)
    print(f"Saved mode-breakout CDF to: {cdf_path}")

    # Save statistics CSV
    stats_path = out_dir / "milestone_pdfs_mode_breakout_statistics.csv"
    fieldnames = [
        "ac_anchor_mode",
        "milestone_name",
        "num_achieved",
        "num_not_achieved",
        "total_rollouts",
        "num_failed_rollouts",
        "num_sampled",
        "sampled_weight_pct",
        "achievement_rate_pct",
        "mode",
        "p10",
        "p50",
        "p90",
    ]
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats_rows:
            writer.writerow(
                {
                    **{k: stat[k] for k in ("ac_anchor_mode", "milestone_name", "num_achieved", "num_not_achieved", "total_rollouts", "num_failed_rollouts", "num_sampled")},
                    "sampled_weight_pct": f"{stat['sampled_weight_pct']:.2f}",
                    "achievement_rate_pct": f"{stat['achievement_rate_pct']:.2f}",
                    "mode": f"{stat['mode']:.4f}",
                    "p10": f"{stat['p10']:.4f}",
                    "p50": f"{stat['p50']:.4f}",
                    "p90": f"{stat['p90']:.4f}",
                }
            )
    print(f"Saved mode-breakout statistics to: {stats_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-AC-anchor-mode breakout distribution CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts.jsonl file")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for CSV files")
    parser.add_argument("--monthly", action="store_true", help="Use monthly sampling for smaller CSV files")
    parser.add_argument("--no-cache", action="store_true",
                        help="Read the raw rollouts.jsonl even when rollouts.cache.json exists (verification)")
    args = parser.parse_args()

    rollouts_file = Path(args.rollouts)
    out_dir = Path(args.out_dir)

    if not rollouts_file.exists():
        print(f"Error: Rollouts file not found: {rollouts_file}")
        sys.exit(1)

    success = generate_mode_breakout_distributions(
        rollouts_file,
        out_dir,
        monthly_sampling=args.monthly,
        use_cache=not args.no_cache,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
