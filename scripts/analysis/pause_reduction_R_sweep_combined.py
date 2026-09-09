#!/usr/bin/env python3
"""
Combined reduction-factor (R) sweep across several pause times, one figure.

Top row: slowdown N(R) vs R (log-log) with the constant-elasticity power law
anchored at 10x. Bottom row: implied per-OOM N = N(R)^(1/log10 R) (flat ⇒
constant elasticity). One column per pause time.

Reuses the single-pause sweep logic. Runs on aifm-runpod (python3.13).
"""

import argparse
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
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402


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


def sweep_one(med, base, tr, ip, pause, Rs):
    """Return (R_arr, N_arr, impN_arr, freeze_delay) for uncensored R values."""
    frozen = freeze_and_reduce_time_series(base, pause)
    asi_freeze = _ms_time(_run(med, frozen, tr, ip), "ASI")
    if asi_freeze is None or (asi_freeze - pause) <= 0:
        return None
    fd = asi_freeze - pause
    Rg, Ng, impg = [], [], []
    for R in Rs:
        reduced = freeze_and_reduce_time_series(
            base, pause, training_reduction=R, experiment_reduction=R,
            inference_reduction=R, labor_reduction=R)
        asi_r = _ms_time(_run(med, reduced, tr, ip), "ASI")
        if asi_r is None or (asi_r - pause) <= 0:
            continue
        N = (asi_r - pause) / fd
        l10 = np.log10(R)
        Rg.append(R); Ng.append(N)
        impg.append(N ** (1.0 / l10) if l10 > 0 else np.nan)
    return np.array(Rg), np.array(Ng), np.array(impg), fd


def main():
    ap = argparse.ArgumentParser(description="Combined R sweep across pauses")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--pauses", default="2027.0,2029.0,2030.5")
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--r-max", type=float, default=1e4)
    ap.add_argument("--n-r", type=int, default=24)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "R_sweep_combined"))
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
    pauses = [float(x) for x in args.pauses.split(",") if x.strip()]
    Rs = np.geomspace(args.r_min, args.r_max, args.n_r)

    base_res = _run(med, base, tr, ip)
    ac = _ms_time(base_res, "AC"); asi0 = _ms_time(base_res, "ASI")

    results = {}
    for p in pauses:
        results[p] = sweep_one(med, base, tr, ip, p, Rs)
        print(f"pause {p}: {0 if results[p] is None else len(results[p][0])} uncensored R values")

    ncol = len(pauses)
    fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol, 8.4), squeeze=False)
    # shared y-limits per row for comparability
    all_N = np.concatenate([results[p][1] for p in pauses if results[p] is not None])
    all_imp = np.concatenate([results[p][2] for p in pauses if results[p] is not None])
    nlo, nhi = all_N.min() * 0.8, all_N.max() * 1.2
    ilo, ihi = all_imp.min() - 0.05, all_imp.max() + 0.05

    for c, p in enumerate(pauses):
        res = results[p]
        ax_top, ax_bot = axes[0][c], axes[1][c]
        if res is None:
            ax_top.set_title(f"pause={p}: freeze-only never reaches ASI")
            continue
        R, N, imp, fd = res
        i10 = int(np.argmin(np.abs(np.log10(R) - 1.0)))
        N10 = N[i10]
        N_pred = N10 ** (np.log10(R))
        k = np.log10(N10)  # since N10 at ~10x => k ~ log10(N10)
        ax_top.loglog(R, N, "-o", color="C0", ms=4, label="actual N(R)")
        ax_top.loglog(R, N_pred, "--", color="C3", lw=1.2, label="power law @10x")
        ax_top.set_title(f"pause={p:.2f}\nfreeze-delay={fd:.2f}yr, N(10x)={N10:.2f}, k≈{k:.3f}", fontsize=10)
        ax_top.set_ylim(nlo, nhi)
        ax_top.set_xlabel("reduction factor R")
        if c == 0:
            ax_top.set_ylabel("N = reduce-delay / freeze-delay")
            ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3, which="both")

        ax_bot.semilogx(R, imp, "-o", color="C0", ms=4)
        ax_bot.axhline(N10, color="C3", ls="--", lw=1.0)
        ax_bot.set_ylim(ilo, ihi)
        ax_bot.set_xlabel("reduction factor R")
        if c == 0:
            ax_bot.set_ylabel("implied per-OOM N = N^(1/log10 R)")
        ax_bot.set_title("flat ⇒ constant elasticity", fontsize=9)
        ax_bot.grid(alpha=0.3, which="both")

    fig.suptitle(f"Reduction-factor sweep at median params (AC={ac:.2f}, ASI={asi0:.2f}) — "
                 f"top: N(R) vs power law; bottom: implied per-OOM N", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "R_sweep_combined.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
