#!/usr/bin/env python3
"""
Pause-reduction experiment (Tier A: full re-integration).

For each sampled parameter draw we compute the statistic N over a grid of pause
times. N measures the EXTRA slowdown from cutting compute+labor by a factor R on
top of merely freezing their growth:

    N(theta, t_pause) = (ASI_time[freeze + reduce-by-R after t_pause] - t_pause)
                        / (ASI_time[freeze only after t_pause]        - t_pause)

Both continuations share the same trajectory up to t_pause and differ only in
what happens to the four input channels (human labor, experiment / inference /
training compute) after it. "Freeze only" holds them flat (the denominator);
"freeze + reduce" additionally divides each by R (R=10 by default), which for
training_compute (log10 FLOP/yr) means subtracting log10(R). The transform is
progress_model.pause_transform.freeze_and_reduce_time_series, which is
golden-tested against the frontend applyPause.

Pause grid: per sample, t_pause = AC_time + frac * (ASI_baseline - AC_time) for
each frac in --grid-fracs, i.e. fractions of the way from AC to (baseline) ASI.
All such pauses are strictly before baseline ASI for frac < 1.

Censoring: under a freeze (and especially a reduce) ASI may never be reached
within the integration horizon. Such (sample, grid point) pairs have N = null
and are flagged censored; we report the censored fraction rather than dropping
them silently.

This is Tier A: every (sample, grid point, variant) re-integrates the full
trajectory from the start. It reuses ProgressModel unchanged and is the
correctness reference. Conditioning on (AC time, AC->ASI gap) is done downstream
as weighted post-processing over the per-sample records this script emits.

RUNS REMOTELY: execute on the aifm-runpod box with python3.13 (see project memory).
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.random import SeedSequence
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as cfg  # noqa: E402
from progress_model import ProgressModel, Parameters, load_time_series_data  # noqa: E402
from progress_model.pause_transform import freeze_and_reduce_time_series  # noqa: E402
from scripts.pipelines.batch_rollout import (  # noqa: E402
    _load_config,
    _sample_parameter_dict,
    _calibrate_gaussian_copula_latent_corr,
    _suppress_noise,
    _to_jsonable,
)

DEFAULT_CONFIG = "plan_a_covert_proj/config/model_uncertain_thomas.yaml"
# Pause grid. Two parameterizations, each point gets a label and is grouped by it
# downstream:
#   - gap fractions: t_pause = AC + frac*(ASI_baseline - AC). frac in [0,1) sweeps
#     from AC toward ASI; negative frac pauses before AC (in units of the gap).
#   - years before AC: t_pause = AC - years. Clearer for pauses well before AC,
#     since the AC->ASI gap is often short (~months).
DEFAULT_GRID_FRACS = [0.0, 0.25, 0.5, 0.75]
DEFAULT_YEARS_BEFORE_AC = [0.5, 1.0, 2.0]
DEFAULT_REDUCTION = 10.0


def _milestone_time(results, name):
    """Return the achievement time of a milestone, or None if not reached."""
    ms = (results or {}).get("milestones", {}) or {}
    m = ms.get(name)
    if not m:
        return None
    t = m.get("time")
    try:
        t = float(t)
    except (TypeError, ValueError):
        return None
    return t if np.isfinite(t) else None


def _run_to_milestones(params_dict, data, time_range, initial_progress):
    """Build a fresh model on the given input series and return its results dict.

    A fresh Parameters is constructed each call so estimation state never leaks
    between variants (matches batch_rollout's per-rollout construction)."""
    params_obj = Parameters(**params_dict)
    model = ProgressModel(params_obj, data)
    model.compute_progress_trajectory(time_range, initial_progress)
    return model.results


def _pause_specs_to_points(grid_specs, ac_time, asi_base, gap):
    """Resolve labeled grid specs into (label, kind, value, pause_time) tuples."""
    out = []
    for kind, value in grid_specs:
        if kind == "gapfrac":
            pause = ac_time + value * gap
            label = f"gapfrac={value:g}"
        elif kind == "yrbeforeac":
            pause = ac_time - value
            label = f"yrbeforeac={value:g}"
        else:
            continue
        out.append((label, kind, value, pause))
    return out


def _compute_sample(args):
    (sample_id, seed_seq, all_param_dists, correlation_matrix, model_param_names,
     input_data_path, time_range, initial_progress, grid_specs, reduction) = args

    rng = np.random.default_rng(seed_seq)
    with _suppress_noise():
        try:
            sampled_all = _sample_parameter_dict(all_param_dists, rng, correlation_matrix)
            # This experiment varies only model parameters. Time-series-generation
            # params (slowdown_year etc.) are a no-op in the current model, so we
            # drop them rather than thread them through.
            model_params = {k: v for k, v in sampled_all.items() if k in model_param_names}
            model_params.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
            model_params.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)

            base = load_time_series_data(input_data_path)
            base_results = _run_to_milestones(model_params, base, time_range, initial_progress)
            ac_time = _milestone_time(base_results, "AC")
            asi_base = _milestone_time(base_results, "ASI")

            record = {
                "sample_id": sample_id,
                "parameters": _to_jsonable(model_params),
                "ac_time": ac_time,
                "asi_baseline_time": asi_base,
                "gap_baseline": (asi_base - ac_time) if (ac_time is not None and asi_base is not None) else None,
                "baseline_reached": ac_time is not None and asi_base is not None,
                "points": [],
                "success": True,
            }

            if not record["baseline_reached"]:
                # Cannot define the AC->ASI grid; emit the record so it can still be
                # counted (and conditioned out) downstream.
                return record

            gap = asi_base - ac_time
            # Floor: don't pause before the present day / data start (freezing
            # historical inputs is meaningless). present_day comes from the model.
            present_day = base_results.get("present_day")
            floor = max(float(time_range[0]), float(present_day) if present_day else float(time_range[0]))

            for label, kind, value, pause in _pause_specs_to_points(grid_specs, ac_time, asi_base, gap):
                # Validity: pause must sit in (floor, baseline ASI).
                if pause < floor or pause >= asi_base:
                    record["points"].append({
                        "label": label, "kind": kind, "value": value, "pause_time": pause,
                        "valid": False, "reason": "pause_before_floor" if pause < floor else "pause_after_asi",
                        "asi_frozen_time": None, "asi_reduced_time": None,
                        "delay_frozen": None, "delay_reduced": None, "N": None, "censored": None,
                    })
                    continue

                frozen = freeze_and_reduce_time_series(base, pause)
                reduced = freeze_and_reduce_time_series(
                    base, pause,
                    training_reduction=reduction,
                    experiment_reduction=reduction,
                    inference_reduction=reduction,
                    labor_reduction=reduction,
                )
                asi_frozen = _milestone_time(
                    _run_to_milestones(model_params, frozen, time_range, initial_progress), "ASI")
                asi_reduced = _milestone_time(
                    _run_to_milestones(model_params, reduced, time_range, initial_progress), "ASI")

                # Denominator must be positive (ASI strictly after pause).
                denom = (asi_frozen - pause) if asi_frozen is not None else None
                censored = (asi_frozen is None) or (asi_reduced is None) or (denom is None) or (denom <= 0)
                n_val = None if censored else (asi_reduced - pause) / denom

                record["points"].append({
                    "label": label, "kind": kind, "value": value, "pause_time": pause,
                    "valid": True, "reason": None,
                    "asi_frozen_time": asi_frozen,
                    "asi_reduced_time": asi_reduced,
                    "delay_frozen": (asi_frozen - pause) if asi_frozen is not None else None,
                    "delay_reduced": (asi_reduced - pause) if asi_reduced is not None else None,
                    "N": n_val,
                    "censored": censored,
                })
            return record

        except Exception as e:  # noqa: BLE001
            import traceback
            return {
                "sample_id": sample_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


def _build_args():
    p = argparse.ArgumentParser(description="Pause-reduction experiment (Tier A)")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Sampling config YAML")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reduction", type=float, default=DEFAULT_REDUCTION,
                   help="Reduction factor R applied to all four channels (numerator)")
    p.add_argument("--grid-fracs", default=",".join(str(f) for f in DEFAULT_GRID_FRACS),
                   help="Comma-separated fractions of the AC->ASI gap (negative = before AC)")
    p.add_argument("--years-before-ac", default=",".join(str(y) for y in DEFAULT_YEARS_BEFORE_AC),
                   help="Comma-separated absolute years-before-AC pause offsets")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "pause_reduction"))
    p.add_argument("--input-data", default=None, help="Override input CSV (else from config)")
    return p.parse_args()


def main():
    import os
    args = _build_args()
    fracs = [float(x) for x in args.grid_fracs.split(",") if x.strip() != ""]
    years_before = [float(x) for x in args.years_before_ac.split(",") if x.strip() != ""]
    grid_specs = [("gapfrac", f) for f in fracs] + [("yrbeforeac", y) for y in years_before]
    # Stable display order: earliest pause first (most-before-AC -> toward ASI).
    grid_labels = ([f"yrbeforeac={y:g}" for y in sorted(years_before, reverse=True)]
                   + [f"gapfrac={f:g}" for f in sorted(fracs)])

    user_cfg = _load_config(args.config) or {}
    param_dists = user_cfg.get("parameters", {})
    ts_param_dists = user_cfg.get("time_series_parameters", {})
    correlation_matrix = user_cfg.get("correlation_matrix")

    input_data_path = str(Path(args.input_data or user_cfg.get("input_data", "inputs/input_data.csv")))
    if not Path(input_data_path).is_absolute():
        input_data_path = str((REPO_ROOT / input_data_path).resolve())

    initial_progress = float(user_cfg.get("initial_progress", 0.0))
    data = load_time_series_data(input_data_path)
    cfg_tr = user_cfg.get("time_range")
    if cfg_tr is not None:
        time_range = [float(cfg_tr[0]), float(cfg_tr[1])]
    else:
        time_range = [float(np.min(data.time)), float(np.max(data.time))]

    all_param_dists = {}
    all_param_dists.update(param_dists)
    all_param_dists.update(ts_param_dists)
    model_param_names = set(param_dists.keys())

    master_ss = SeedSequence(int(args.seed))

    # Calibrate the Gaussian-copula latent correlation once (mirrors batch_rollout).
    if correlation_matrix is not None:
        correlated_params = correlation_matrix.get("parameters", [])
        target_corr = correlation_matrix.get("correlation_matrix")
        if correlated_params and target_corr is not None:
            interpretation = str(correlation_matrix.get("correlation_type",
                                 correlation_matrix.get("interpretation", "spearman")))
            calib_samples = int(correlation_matrix.get("calibration_samples", 20000))
            calib_rng = np.random.default_rng(master_ss.spawn(1)[0])
            try:
                latent = _calibrate_gaussian_copula_latent_corr(
                    param_dists, correlated_params, np.array(target_corr, dtype=float),
                    calib_rng, interpretation=interpretation, calibration_samples=calib_samples,
                )
                correlation_matrix["latent_correlation_matrix"] = latent.tolist()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: correlation calibration failed ({e}); using provided matrix.")

    num_workers = args.num_workers or max(1, (os.cpu_count() or 2))
    sample_seeds = master_ss.spawn(int(args.num_samples))
    work = [
        (i, sample_seeds[i], all_param_dists, correlation_matrix, model_param_names,
         input_data_path, time_range, initial_progress, grid_specs, float(args.reduction))
        for i in range(int(args.num_samples))
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "pause_reduction_samples.jsonl"

    print(f"Config: {args.config}  | samples: {args.num_samples}  | workers: {num_workers}")
    print(f"Reduction R={args.reduction}")
    print(f"grid (earliest->latest pause): {grid_labels}")
    print(f"time_range={time_range}  input={input_data_path}")
    print(f"Output: {out_path}")

    results = []
    n_done = 0
    with out_path.open("w", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=num_workers) as ex:
        for rec in ex.map(_compute_sample, work, chunksize=max(1, len(work) // (num_workers * 4))):
            f.write(json.dumps(rec) + "\n")
            results.append(rec)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(work)} done")

    _summarize(results, grid_labels, run_dir)


def _summarize(results, grid_labels, run_dir):
    ok = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    baseline_reached = [r for r in ok if r.get("baseline_reached")]
    print("\n===== SUMMARY =====")
    print(f"samples: {len(results)}  | errored: {len(failed)}  | "
          f"baseline reached AC+ASI: {len(baseline_reached)}/{len(ok)}")

    summary = {"n_samples": len(results), "n_errored": len(failed),
               "n_baseline_reached": len(baseline_reached), "by_label": {}}
    for label in grid_labels:
        ns, n_cens, n_invalid, n_pts = [], 0, 0, 0
        for r in baseline_reached:
            for pt in r.get("points", []):
                if pt.get("label") != label:
                    continue
                n_pts += 1
                if not pt.get("valid", True):
                    n_invalid += 1
                elif pt["censored"] or pt["N"] is None:
                    n_cens += 1
                else:
                    ns.append(pt["N"])
        if ns:
            arr = np.array(ns, dtype=float)
            stats = {
                "n": n_pts, "n_censored": n_cens, "n_invalid": n_invalid,
                "median_N": float(np.median(arr)),
                "mean_N": float(np.mean(arr)),
                "p10_N": float(np.percentile(arr, 10)),
                "p90_N": float(np.percentile(arr, 90)),
            }
        else:
            stats = {"n": n_pts, "n_censored": n_cens, "n_invalid": n_invalid, "median_N": None}
        summary["by_label"][label] = stats
        print(f"  {label}: n={n_pts} valid_reached={len(ns)} censored={n_cens} "
              f"invalid={n_invalid} median_N={stats.get('median_N')}")

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
