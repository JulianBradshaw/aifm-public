#!/usr/bin/env python3
"""
Ablation: reduce ALL channels by R vs reduce COMPUTE-only by R (labor just paused).

Like pause_reduction_R_sweep_combined, but each panel overlays two reduction modes:
  * "all"     : training + experiment + inference compute AND human labor all ÷R
  * "compute" : training + experiment + inference compute ÷R, human labor ÷1
                (i.e. labor merely frozen/paused at its full pause-instant value)

The freeze-only denominator (pure pause, no reduction) is identical for both
modes, so differences in N isolate the marginal slowdown contributed by cutting
labor on top of cutting compute.

Top row: N(R) for both modes. Bottom row: implied per-OOM N = N^(1/log10 R).
One column per pause time. Median params. Runs on aifm-runpod (python3.13).
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

MODES = {
    "all":     {"color": "C0", "label": "all 4 channels ÷R"},
    "compute": {"color": "C1", "label": "compute ÷R, labor paused (×1)"},
}


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


def sweep_mode(med, base, tr, ip, pause, Rs, freeze_delay, reduce_labor):
    Rg, Ng, impg = [], [], []
    for R in Rs:
        labor_R = R if reduce_labor else 1.0
        reduced = freeze_and_reduce_time_series(
            base, pause, training_reduction=R, experiment_reduction=R,
            inference_reduction=R, labor_reduction=labor_R)
        asi_r = _ms_time(_run(med, reduced, tr, ip), "ASI")
        if asi_r is None or (asi_r - pause) <= 0:
            continue
        N = (asi_r - pause) / freeze_delay
        l10 = np.log10(R)
        Rg.append(R); Ng.append(N); impg.append(N ** (1.0 / l10) if l10 > 0 else np.nan)
    return np.array(Rg), np.array(Ng), np.array(impg)


def main():
    ap = argparse.ArgumentParser(description="R-sweep ablation: all channels vs compute-only")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--pauses", default="2027.0,2029.0,2030.5")
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--r-max", type=float, default=1e4)
    ap.add_argument("--n-r", type=int, default=24)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "R_ablation"))
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

    data = {}
    for p in pauses:
        frozen = freeze_and_reduce_time_series(base, p)
        asi_freeze = _ms_time(_run(med, frozen, tr, ip), "ASI")
        fd = asi_freeze - p if (asi_freeze and asi_freeze > p) else None
        if fd is None:
            data[p] = None; continue
        data[p] = {
            "fd": fd,
            "all": sweep_mode(med, base, tr, ip, p, Rs, fd, reduce_labor=True),
            "compute": sweep_mode(med, base, tr, ip, p, Rs, fd, reduce_labor=False),
        }
        na = len(data[p]["all"][0]); nc = len(data[p]["compute"][0])
        print(f"pause {p}: freeze-delay={fd:.3f}yr | all: {na} uncensored R | compute-only: {nc} uncensored R")

    ncol = len(pauses)
    fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol, 8.4), squeeze=False)
    # shared limits
    allN = np.concatenate([data[p][m][1] for p in pauses if data[p] for m in ("all", "compute") if len(data[p][m][1])])
    allI = np.concatenate([data[p][m][2] for p in pauses if data[p] for m in ("all", "compute") if len(data[p][m][2])])
    nlo, nhi = allN.min() * 0.8, allN.max() * 1.2
    ilo, ihi = np.nanmin(allI) - 0.1, np.nanmax(allI) + 0.1

    for c, p in enumerate(pauses):
        ax_top, ax_bot = axes[0][c], axes[1][c]
        if data[p] is None:
            ax_top.set_title(f"pause={p}: freeze never reaches ASI"); continue
        for m in ("all", "compute"):
            R, N, imp = data[p][m]
            col = MODES[m]["color"]
            ax_top.loglog(R, N, "-o", color=col, ms=3.5, label=MODES[m]["label"])
            ax_bot.semilogx(R, imp, "-o", color=col, ms=3.5, label=MODES[m]["label"])
        # N(10x) for each mode
        def n10(m):
            R, N, _ = data[p][m]
            return N[int(np.argmin(np.abs(np.log10(R) - 1.0)))] if len(N) else float("nan")
        ax_top.set_title(f"pause={p:.2f}  (freeze-delay={data[p]['fd']:.2f}yr)\n"
                         f"N(10x): all={n10('all'):.2f}, compute-only={n10('compute'):.2f}", fontsize=9)
        ax_top.set_ylim(nlo, nhi); ax_bot.set_ylim(ilo, ihi)
        ax_top.set_xlabel("reduction factor R"); ax_bot.set_xlabel("reduction factor R")
        ax_top.grid(alpha=0.3, which="both"); ax_bot.grid(alpha=0.3, which="both")
        if c == 0:
            ax_top.set_ylabel("N = reduce-delay / freeze-delay")
            ax_bot.set_ylabel("implied per-OOM N")
            ax_top.legend(fontsize=8); ax_bot.legend(fontsize=8)
        ax_bot.set_title("flat ⇒ constant elasticity", fontsize=9)

    fig.suptitle(f"R-sweep ablation at median params (AC={ac:.2f}, ASI={asi0:.2f}) — "
                 f"reduce ALL 4 channels vs COMPUTE-only (labor paused)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "R_ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
