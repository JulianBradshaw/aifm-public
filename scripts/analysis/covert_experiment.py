#!/usr/bin/env python3
"""Covert-project Monte Carlo experiment — the covert-race analogue of the
pause-reduction (N) experiment.

For each sampled parameter draw we sweep the pause time over a grid and, at each
pause, simulate a covert project racing a paused legal project to TED-AI:

    legal project  must reach TED-AI + alignment_tax_ooms   (it pays the tax)
    covert project must reach TED-AI                          (it does not)

and record whether/when the covert wins. The covert physics is
``progress_model.covert_sim`` (a faithful, game-free port of the monorepo's
covert engine). This script reuses the batch_rollout sampling machinery
(distributions, ci80, Gaussian-copula correlations) exactly as
``pause_reduction_experiment.py`` does — it only supplies its own per-sample
computation.

Parameter categories in the sampling config:
    parameters:           model Parameters fields (vary the trajectory)
    covert_parameters:    covert knobs — SAMPLED per draw (alignment_tax_ooms,
                          max_hw_rate, legal_scaling_duration, total_h100e,
                          frac_stolen, dormancy_years)

The legal "target year" is pause + legal_scaling_duration, so it stays well-posed as
the pause grid sweeps.
"""

from __future__ import annotations

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
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert  # noqa: E402
from scripts.pipelines.batch_rollout import (  # noqa: E402
    _load_config,
    _sample_parameter_dict,
    _calibrate_gaussian_copula_latent_corr,
    _suppress_noise,
    _to_jsonable,
)

DEFAULT_CONFIG = "plan_a_covert_proj/config/covert_scenario_branch.yaml"
# Pause grid relative to the AC->TED-AI gap (the covert race endpoint), plus
# absolute years-before-AC. Mirrors pause_reduction_experiment's grid.
DEFAULT_GRID_FRACS = [0.0, 0.5]
DEFAULT_YEARS_BEFORE_AC = [0.0, 1.0, 2.0]
DEFAULT_DT = 0.05

# The covert knobs sampled from the config's covert_parameters: section.
COVERT_PARAM_NAMES = (
    "alignment_tax_progress_years",   # alignment tax, a stock in 2025-progress-years
    "max_hw_rate",
    "legal_scaling_duration",
    "total_h100e",
    "frac_stolen",
    "dormancy_years",
    "setup_period_years",             # training-R&D pause length (the "six-month pause")
    "setup_sw_gain_progress_months",  # windfall during the pause, a stock in 2025-progress-months
)


def _pause_specs_to_points(grid_specs, ac_time, gap):
    """Resolve labeled grid specs into (label, kind, value, pause_time) tuples.

    gapfrac: pause = AC + frac*(TED-AI - AC); yrbeforeac: pause = AC - years.
    """
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
     covert_param_names, input_data_path, time_range, dt, grid_specs) = args

    with _suppress_noise():
        rng = np.random.default_rng(seed_seq)
        try:
            sampled = _sample_parameter_dict(all_param_dists, rng, correlation_matrix)
            model_params = {k: v for k, v in sampled.items() if k in model_param_names}
            model_params.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
            model_params.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)
            covert = {k: float(sampled[k]) for k in covert_param_names if k in sampled}

            data = load_time_series_data(input_data_path)
            params = Parameters(**model_params)
            # run_default DEEP-COPIES params and mutates only the copy; the prepared form
            # (rescaled r_software, converted slopes) is ref['prepared_params']. Use that
            # for the covert sim — `params` here stays RAW. (simulate_covert also normalises
            # to ref['prepared_params'] internally, but pass it explicitly for clarity.)
            ref = run_default(params, data, time_range)
            params = ref["prepared_params"]

            ac_time, ted_time, ted_e = ref["ac_time"], ref["ted_ai_time"], ref["ted_ai_e"]
            record = {
                "sample_id": sample_id,
                "success": True,
                "parameters": _to_jsonable(model_params),
                "covert_parameters": covert,
                "ac_time": ac_time,
                "ted_ai_time": ted_time,
                "ted_ai_e": ted_e,
                "ac_to_ted_gap": ref["ac_to_ted_gap"],
                "baseline_reached": ac_time is not None and ted_time is not None and ted_e is not None,
                "points": [],
            }
            if not record["baseline_reached"]:
                return record

            gap = ted_time - ac_time
            floor = max(float(cfg.TRAINING_COMPUTE_REFERENCE_YEAR), float(time_range[0]))
            for label, kind, value, pause in _pause_specs_to_points(grid_specs, ac_time, gap):
                if pause < floor or pause >= ted_time:
                    record["points"].append({
                        "label": label, "kind": kind, "value": value, "pause_time": pause,
                        "valid": False,
                        "reason": "pause_before_floor" if pause < floor else "pause_after_ted",
                    })
                    continue
                # legal_scaling_duration is the POST-pause window: target = end-of-pause + duration
                target_year = pause + covert.get("setup_period_years", 0.0) + covert["legal_scaling_duration"]
                rec = simulate_covert(
                    params, data, pause,
                    alignment_tax_progress_years=covert["alignment_tax_progress_years"],
                    target_year=target_year,
                    max_hw_rate=covert["max_hw_rate"],
                    compute_mode="absolute",
                    total_h100e=covert["total_h100e"],
                    frac_stolen=covert["frac_stolen"],
                    dormancy_years=covert.get("dormancy_years", 0.0),
                    setup_period_years=covert.get("setup_period_years", 0.0),
                    setup_sw_gain_progress_months=covert.get("setup_sw_gain_progress_months", 0.0),
                    dt=dt, t_end=float(time_range[1]), ref=ref,
                )
                rec.update({"label": label, "kind": kind, "value": value})
                record["points"].append(rec)
            return record

        except Exception as e:  # noqa: BLE001
            import traceback
            return {
                "sample_id": sample_id, "success": False,
                "error": str(e), "traceback": traceback.format_exc(),
            }


def _build_args():
    p = argparse.ArgumentParser(description="Covert-project Monte Carlo experiment")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Sampling config YAML")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-fracs", default=",".join(str(f) for f in DEFAULT_GRID_FRACS),
                   help="Comma-separated fractions of the AC->TED-AI gap")
    p.add_argument("--years-before-ac", default=",".join(str(y) for y in DEFAULT_YEARS_BEFORE_AC),
                   help="Comma-separated absolute years-before-AC pause offsets")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="Covert integrator step (yr)")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "covert_experiment"))
    p.add_argument("--input-data", default=None, help="Override input CSV (else from config)")
    return p.parse_args()


def main():
    import os
    args = _build_args()
    fracs = [float(x) for x in args.grid_fracs.split(",") if x.strip() != ""]
    years_before = [float(x) for x in args.years_before_ac.split(",") if x.strip() != ""]
    grid_specs = [("gapfrac", f) for f in fracs] + [("yrbeforeac", y) for y in years_before]
    grid_labels = ([f"yrbeforeac={y:g}" for y in sorted(years_before, reverse=True)]
                   + [f"gapfrac={f:g}" for f in sorted(fracs)])

    user_cfg = _load_config(args.config) or {}
    param_dists = user_cfg.get("parameters", {}) or {}
    covert_dists = user_cfg.get("covert_parameters", {}) or {}
    ts_param_dists = user_cfg.get("time_series_parameters", {}) or {}
    correlation_matrix = user_cfg.get("correlation_matrix")

    input_data_path = str(Path(args.input_data or user_cfg.get("input_data", "inputs/input_data.csv")))
    if not Path(input_data_path).is_absolute():
        input_data_path = str((REPO_ROOT / input_data_path).resolve())

    data = load_time_series_data(input_data_path)
    cfg_tr = user_cfg.get("time_range")
    time_range = ([float(cfg_tr[0]), float(cfg_tr[1])] if cfg_tr is not None
                  else [float(np.min(data.time)), float(np.max(data.time))])

    all_param_dists = {}
    all_param_dists.update(param_dists)
    all_param_dists.update(ts_param_dists)
    all_param_dists.update(covert_dists)
    model_param_names = set(param_dists.keys())
    covert_param_names = set(covert_dists.keys())

    master_ss = SeedSequence(int(args.seed))

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
                    all_param_dists, correlated_params, np.array(target_corr, dtype=float),
                    calib_rng, interpretation=interpretation, calibration_samples=calib_samples,
                )
                correlation_matrix["latent_correlation_matrix"] = latent.tolist()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: correlation calibration failed ({e}); using provided matrix.")

    num_workers = args.num_workers or max(1, (os.cpu_count() or 2))
    sample_seeds = master_ss.spawn(int(args.num_samples))
    work = [
        (i, sample_seeds[i], all_param_dists, correlation_matrix, model_param_names,
         covert_param_names, input_data_path, time_range, float(args.dt), grid_specs)
        for i in range(int(args.num_samples))
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "covert_experiment_samples.jsonl"

    print(f"Config: {args.config}  | samples: {args.num_samples}  | workers: {num_workers}")
    print(f"covert params sampled: {sorted(covert_param_names)}")
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
    reached = [r for r in ok if r.get("baseline_reached")]
    print("\n===== SUMMARY =====")
    print(f"samples: {len(results)}  | errored: {len(failed)}  | "
          f"baseline reached AC+TED-AI: {len(reached)}/{len(ok)}")

    summary = {"n_samples": len(results), "n_errored": len(failed),
               "n_baseline_reached": len(reached), "by_label": {}}
    for label in grid_labels:
        wins, leads, n_valid, n_invalid = [], [], 0, 0
        for r in reached:
            for pt in r.get("points", []):
                if pt.get("label") != label:
                    continue
                if not pt.get("valid", True):
                    n_invalid += 1
                    continue
                n_valid += 1
                wins.append(1.0 if pt.get("covert_wins") else 0.0)
                if pt.get("lead_years") is not None:
                    leads.append(pt["lead_years"])
        stats = {"n_valid": n_valid, "n_invalid": n_invalid}
        if wins:
            stats["covert_win_frac"] = float(np.mean(wins))
        if leads:
            arr = np.array(leads, dtype=float)
            stats.update({
                "median_lead_years": float(np.median(arr)),
                "p10_lead_years": float(np.percentile(arr, 10)),
                "p90_lead_years": float(np.percentile(arr, 90)),
            })
        summary["by_label"][label] = stats
        if wins:
            print(f"  {label:16s}  covert_win_frac={stats.get('covert_win_frac', float('nan')):.3f}  "
                  f"median_lead={stats.get('median_lead_years', float('nan')):+.2f}y  (n={n_valid})")

    (run_dir / "covert_experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {run_dir / 'covert_experiment_summary.json'}")


if __name__ == "__main__":
    main()
