#!/usr/bin/env python3
"""
Broken-out trajectories for a few illustrative pause times at MEDIAN params.

For each pause time, runs and overlays three progress trajectories (progress =
cumulative effective compute, OOMs):
  * unpaused  — the baseline inputs (no pause)
  * paused    — inputs frozen at the pause (freeze-only, R=1)  [denominator]
  * pause+10x — inputs frozen AND cut by R at the pause          [numerator]

AC and ASI are fixed progress levels (same params -> same milestone->progress
mapping), drawn as horizontal lines; each trajectory's ASI crossing time is
marked, so the freeze-delay and reduce-delay (and thus N) are readable as
horizontal distances from the pause.

Runs on aifm-runpod (python3.13).
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
    t, prog, _ = model.compute_progress_trajectory(tr, ip)
    return model.results, np.asarray(t, dtype=float), np.asarray(prog, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Broken-out trajectories at median params")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--pauses", default="2027.0,2029.0,2030.5")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "trajectory_breakouts"))
    args = ap.parse_args()

    ucfg = _load_config(args.config) or {}
    med = _get_median_parameter_dict(ucfg.get("parameters", {}))
    med.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    med.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)

    input_path = str(ucfg.get("input_data", "inputs/input_data.csv"))
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)
    tr = ucfg.get("time_range", [2017.0, 2200.0])
    tr = [float(tr[0]), float(tr[1])]
    ip = float(ucfg.get("initial_progress", 0.0))
    pauses = [float(x) for x in args.pauses.split(",") if x.strip()]

    base_res, bt, bp = _run(med, base, tr, ip)
    ac = _ms_time(base_res, "AC"); asi = _ms_time(base_res, "ASI")
    ac_level = float(np.interp(ac, bt, bp))
    asi_level = float(np.interp(asi, bt, bp))
    print(f"Median params: AC={ac:.3f} (progress {ac_level:.3f}), ASI={asi:.3f} (progress {asi_level:.3f})")

    n = len(pauses)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.3 * n), squeeze=False)
    for j, pause in enumerate(pauses):
        ax = axes[j][0]
        frozen = freeze_and_reduce_time_series(base, pause)
        reduced = freeze_and_reduce_time_series(
            base, pause, training_reduction=args.reduction, experiment_reduction=args.reduction,
            inference_reduction=args.reduction, labor_reduction=args.reduction)
        fres, ft, fp = _run(med, frozen, tr, ip)
        rres, rt, rp = _run(med, reduced, tr, ip)
        asi_unp = asi
        asi_fz = _ms_time(fres, "ASI")
        asi_rd = _ms_time(rres, "ASI")

        # plot window: from a bit before the pause to just past the reduced ASI
        x0 = pause - 1.0
        x1 = (asi_rd if asi_rd else tr[1]) + 0.05 * ((asi_rd if asi_rd else tr[1]) - pause)
        for t, p, c, lbl in [(bt, bp, "0.5", "unpaused"),
                             (ft, fp, "C2", "paused (freeze)"),
                             (rt, rp, "C0", f"pause + ÷{args.reduction:g}")]:
            sel = (t >= x0) & (t <= x1)
            ax.plot(t[sel], p[sel], "-", color=c, lw=1.8, label=lbl)
        # milestone levels
        ax.axhline(ac_level, color="0.7", ls=":", lw=1.0)
        ax.axhline(asi_level, color="C3", ls="--", lw=1.0)
        ax.text(x0, ac_level, " AC", color="0.4", va="bottom", fontsize=8)
        ax.text(x0, asi_level, " ASI", color="C3", va="bottom", fontsize=8)
        ax.axvline(pause, color="k", ls=":", lw=1.0)
        # ASI crossings
        for at, c in [(asi_unp, "0.5"), (asi_fz, "C2"), (asi_rd, "C0")]:
            if at is not None:
                ax.plot([at], [asi_level], "o", color=c, ms=7, zorder=5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(top=asi_level + 0.08 * (asi_level - ac_level), bottom=float(np.interp(x0, bt, bp)))
        ax.set_xlabel("year")
        ax.set_ylabel("progress (OOMs eff. compute)")
        df = (asi_fz - pause) if asi_fz else float("nan")
        dr = (asi_rd - pause) if asi_rd else float("nan")
        nval = dr / df if (df and np.isfinite(df) and df > 0) else float("nan")
        ax.set_title(f"pause = {pause:.2f}\nASI: unp={asi_unp:.2f}, freeze={asi_fz:.2f}, ÷{args.reduction:g}={asi_rd:.2f}\n"
                     f"freeze-delay={df:.2f}yr, ÷{args.reduction:g}-delay={dr:.2f}yr, N={nval:.2f}", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Progress trajectories at median params (AC={ac:.2f}, ASI={asi:.2f}, gap={asi-ac:.2f}yr) — "
                 f"unpaused vs freeze vs freeze+÷{args.reduction:g}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trajectory_breakouts.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
