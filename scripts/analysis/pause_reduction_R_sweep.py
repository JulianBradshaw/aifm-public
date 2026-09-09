#!/usr/bin/env python3
"""
Reduction-factor (R) sweep at fixed pause + median params.

Fixes params at the config medians and the pause time, varies only the reduction
factor R (applied equally to all four channels), and traces the slowdown

    N(R) = (ASI[freeze + ÷R] - pause) / (ASI[freeze only] - pause)

Tests whether the slowdown scales as a constant-elasticity power law in R, i.e.
N(R) = N_perOOM ^ log10(R)  (the "3x at 10x -> 9x at 100x -> 27x at 1000x"
expectation). Under that hypothesis the "implied per-OOM N",

    implied_N(R) = N(R) ^ (1 / log10 R),

is constant across R. We plot N(R) and implied_N(R) and the power-law prediction
anchored at R=10, so deviations from constant elasticity are visible.

Runs on aifm-runpod (python3.13).
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


def _ms_time(results, name):
    m = (results or {}).get("milestones", {}).get(name)
    if not m or m.get("time") is None:
        return None
    t = float(m["time"])
    return t if np.isfinite(t) else None


def _run(params, data, tr, ip):
    model = ProgressModel(Parameters(**params), data)
    model.compute_progress_trajectory(tr, ip)
    return model.results


def main():
    ap = argparse.ArgumentParser(description="Reduction-factor sweep -> implied N")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--pause", type=float, default=2029.0)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--r-max", type=float, default=1e4)
    ap.add_argument("--n-r", type=int, default=24)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "R_sweep"))
    args = ap.parse_args()

    ucfg = _load_config(args.config) or {}
    med = _get_median_parameter_dict(ucfg.get("parameters", {}))
    med.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    med.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)
    input_path = str(ucfg.get("input_data", "inputs/input_data.csv"))
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)
    tr = ucfg.get("time_range", [2017.0, 2200.0]); tr = [float(tr[0]), float(tr[1])]
    ip = float(ucfg.get("initial_progress", 0.0))
    pause = args.pause

    base_res = _run(med, base, tr, ip)
    ac = _ms_time(base_res, "AC"); asi0 = _ms_time(base_res, "ASI")
    # Freeze-only (denominator) — R-independent.
    frozen = freeze_and_reduce_time_series(base, pause)
    asi_freeze = _ms_time(_run(med, frozen, tr, ip), "ASI")
    if asi_freeze is None or (asi_freeze - pause) <= 0:
        print("Freeze-only did not reach ASI after pause; abort."); sys.exit(1)
    freeze_delay = asi_freeze - pause
    print(f"Median params: AC={ac:.3f}, ASI(unpaused)={asi0:.3f}. Pause={pause}.")
    print(f"Freeze-only ASI={asi_freeze:.3f}, freeze-delay={freeze_delay:.4f}yr (denominator).")

    Rs = np.geomspace(args.r_min, args.r_max, args.n_r)
    rows = []
    print(f"\n{'R':>10}  {'ASI_red':>9}  {'reduce_delay':>12}  {'N':>8}  {'implied_N':>9}  {'elasticity_k':>12}")
    for R in Rs:
        reduced = freeze_and_reduce_time_series(
            base, pause, training_reduction=R, experiment_reduction=R,
            inference_reduction=R, labor_reduction=R)
        asi_r = _ms_time(_run(med, reduced, tr, ip), "ASI")
        if asi_r is None or (asi_r - pause) <= 0:
            rows.append({"R": float(R), "asi_reduced": asi_r, "N": None,
                         "implied_N": None, "elasticity": None, "censored": True})
            print(f"{R:>10.1f}  {'--':>9}  {'--':>12}  {'--':>8}  {'--':>9}  {'(censored)':>12}")
            continue
        N = (asi_r - pause) / freeze_delay
        log10R = np.log10(R)
        implied_N = N ** (1.0 / log10R) if log10R > 0 else float("nan")
        k = np.log10(N) / log10R if log10R > 0 else float("nan")  # elasticity: N = R^k
        rows.append({"R": float(R), "asi_reduced": asi_r, "reduce_delay": asi_r - pause,
                     "N": float(N), "implied_N": float(implied_N), "elasticity": float(k),
                     "censored": False})
        print(f"{R:>10.1f}  {asi_r:>9.3f}  {asi_r - pause:>12.4f}  {N:>8.3f}  {implied_N:>9.4f}  {k:>12.4f}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": args.config, "pause": pause, "ac": ac, "asi_unpaused": asi0,
               "freeze_delay": freeze_delay, "rows": rows, "median_params": med}
    with (out_dir / "R_sweep.json").open("w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
    _plot(rows, pause, freeze_delay, out_dir)
    print(f"\nWrote R_sweep.json + plot to {out_dir}")


def _plot(rows, pause, freeze_delay, out_dir):
    good = [r for r in rows if not r["censored"] and r["N"] is not None]
    if not good:
        print("No uncensored points to plot."); return
    R = np.array([r["R"] for r in good])
    N = np.array([r["N"] for r in good])
    impN = np.array([r["implied_N"] for r in good])

    # Constant-elasticity prediction anchored at R=10 (or nearest available).
    i10 = int(np.argmin(np.abs(np.log10(R) - 1.0)))
    N10 = N[i10]
    N_pred = N10 ** (np.log10(R))  # = N10^{log10 R}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.loglog(R, N, "-o", color="C0", label="actual N(R)")
    ax1.loglog(R, N_pred, "--", color="C3",
               label=f"constant-elasticity (N=R^k anchored @10x, N10={N10:.2f})")
    ax1.set_xlabel("reduction factor R (all four channels)")
    ax1.set_ylabel("N = reduce-delay / freeze-delay")
    ax1.set_title(f"Slowdown N vs reduction R  (pause={pause}, median params)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogx(R, impN, "-o", color="C0")
    ax2.axhline(N10, color="C3", ls="--", lw=1.0, label=f"value at 10x = {N10:.2f}")
    ax2.set_xlabel("reduction factor R")
    ax2.set_ylabel("implied per-OOM N = N(R)^(1/log10 R)")
    ax2.set_title("Implied per-OOM N\n(flat ⇒ constant elasticity / power-law scaling)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "R_sweep.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
