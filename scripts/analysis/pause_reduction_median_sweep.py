#!/usr/bin/env python3
"""
Pause-timing sweep at FIXED (median) parameters.

Fixes every model parameter at the sampling-config marginal median (thomas:
AC ~ 2030.0, ASI ~ 2031.0, gap ~ 1.0yr) and varies ONLY the pause time, tracing
N(t_pause) as a single deterministic curve.

    N(t_pause) = (ASI[freeze + reduce-by-R] - t_pause) / (ASI[freeze only] - t_pause)

Sweeps t_pause over a fine grid from a floor (present day) up to just before the
baseline ASI. Marks where freeze-only or freeze+reduce fail to reach ASI within
the horizon (N undefined / censored).

Cheap: one parameter set, ~2 trajectories per grid point.
Runs remotely on aifm-runpod (python3.13).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as cfg  # noqa: E402
from progress_model import ProgressModel, Parameters, load_time_series_data  # noqa: E402
from progress_model.pause_transform import freeze_and_reduce_time_series  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict, _to_jsonable  # noqa: E402


def _milestone_time(results, name):
    m = (results or {}).get("milestones", {}).get(name)
    if not m:
        return None
    t = m.get("time")
    try:
        t = float(t)
    except (TypeError, ValueError):
        return None
    return t if np.isfinite(t) else None


def _run(params, data, time_range, ip):
    model = ProgressModel(Parameters(**params), data)
    model.compute_progress_trajectory(time_range, ip)
    return model.results


def main():
    ap = argparse.ArgumentParser(description="Pause-timing sweep at fixed median params")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--pre-ac-years", type=float, default=2.0,
                    help="Start the sweep this many years before AC (clamped to present-day floor)")
    ap.add_argument("--start-year", type=float, default=None,
                    help="Explicit sweep start year (overrides --pre-ac-years; clamped to present-day floor)")
    ap.add_argument("--max-frac", type=float, default=0.9,
                    help="Stop the sweep at AC + max_frac*gap (excludes the degenerate pause-at-ASI point)")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "median_sweep"))
    args = ap.parse_args()

    ucfg = _load_config(args.config) or {}
    pdists = ucfg.get("parameters", {})
    med = _get_median_parameter_dict(pdists)
    med.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    med.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)

    input_path = str(ucfg.get("input_data", "inputs/input_data.csv"))
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)
    tr = ucfg.get("time_range", [2017.0, 2200.0])
    time_range = [float(tr[0]), float(tr[1])]
    ip = float(ucfg.get("initial_progress", 0.0))

    base_res = _run(med, base, time_range, ip)
    ac = _milestone_time(base_res, "AC")
    asi = _milestone_time(base_res, "ASI")
    present_day = base_res.get("present_day")
    if ac is None or asi is None:
        print("Median trajectory did not reach AC/ASI."); sys.exit(1)
    gap = asi - ac
    floor = max(time_range[0], float(present_day) if present_day else time_range[0])
    start_req = args.start_year if args.start_year is not None else (ac - args.pre_ac_years)
    start = max(floor, start_req)
    end = ac + args.max_frac * gap  # stop short of ASI (avoid degenerate pause-at-ASI)
    pauses = np.linspace(start, end, args.n_points)

    print(f"Median params: AC={ac:.4f}, ASI={asi:.4f}, gap={gap:.4f}yr  "
          f"(include_gap={med.get('include_gap')}, gap_years={med.get('gap_years'):.3f})")
    print(f"Sweeping {args.n_points} pause times in [{start:.3f}, {end:.3f}]  R={args.reduction}")

    rows = []
    for p in pauses:
        frozen = freeze_and_reduce_time_series(base, float(p))
        reduced = freeze_and_reduce_time_series(
            base, float(p), training_reduction=args.reduction, experiment_reduction=args.reduction,
            inference_reduction=args.reduction, labor_reduction=args.reduction)
        asi_f = _milestone_time(_run(med, frozen, time_range, ip), "ASI")
        asi_r = _milestone_time(_run(med, reduced, time_range, ip), "ASI")
        denom = (asi_f - p) if asi_f is not None else None
        censored = (asi_f is None) or (asi_r is None) or (denom is None) or (denom <= 0)
        n = None if censored else (asi_r - p) / denom
        rows.append({"pause_time": float(p),
                     "pause_minus_ac": float(p - ac),
                     "asi_frozen": asi_f, "asi_reduced": asi_r,
                     "delay_frozen": (asi_f - p) if asi_f is not None else None,
                     "delay_reduced": (asi_r - p) if asi_r is not None else None,
                     "N": n, "censored": censored})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(rows, ac, asi, gap, args.reduction, out_dir)
    summary = {"config": args.config, "ac_time": ac, "asi_time": asi, "gap": gap,
               "reduction": args.reduction, "median_params": med, "rows": rows}
    with (out_dir / "median_sweep.json").open("w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
    defined = [r for r in rows if r["N"] is not None]
    n_cens = sum(1 for r in rows if r["censored"])
    print(f"N defined at {len(defined)}/{len(rows)} pause times (censored: {n_cens}).")
    if defined:
        ns = [r["N"] for r in defined]
        print(f"N range: {min(ns):.3f} .. {max(ns):.3f}; at AC N={_n_at(rows, ac):.3f}")
    print(f"Wrote median_sweep.json + plot to {out_dir}")


def _n_at(rows, t):
    best = min((r for r in rows if r["N"] is not None), key=lambda r: abs(r["pause_time"] - t), default=None)
    return best["N"] if best else float("nan")


def _plot(rows, ac, asi, gap, R, out_dir):
    p = np.array([r["pause_time"] for r in rows])
    N = np.array([r["N"] if r["N"] is not None else np.nan for r in rows])
    df = np.array([r["delay_frozen"] if r["delay_frozen"] is not None else np.nan for r in rows])
    dr = np.array([r["delay_reduced"] if r["delay_reduced"] is not None else np.nan for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(p, N, "-o", ms=3, color="C0")
    for x, lbl, c in [(ac, "AC", "C2"), (asi, "ASI", "C3")]:
        ax1.axvline(x, color=c, ls="--", lw=1.2, label=lbl)
    ax1.set_xlabel("pause time (year)")
    ax1.set_ylabel("N  =  reduce-delay / freeze-delay")
    ax1.set_title(f"N vs pause time at MEDIAN params (R={R:g})\nAC={ac:.2f}, ASI={asi:.2f}, gap={gap:.2f}yr")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(p, df, "-", color="0.5", label="freeze-only delay (ASI−pause)")
    ax2.plot(p, dr, "-", color="C0", label=f"freeze+÷{R:g} delay (ASI−pause)")
    for x, c in [(ac, "C2"), (asi, "C3")]:
        ax2.axvline(x, color=c, ls="--", lw=1.2)
    ax2.set_xlabel("pause time (year)")
    ax2.set_ylabel("years from pause to ASI")
    ax2.set_yscale("log")
    ax2.set_title("Time-to-ASI after pause: freeze-only vs freeze+reduce")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "median_sweep.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
