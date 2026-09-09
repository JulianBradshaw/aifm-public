#!/usr/bin/env python3
"""
Multi-metric broken-out trajectories at MEDIAN params.

Grid of panels: rows = metrics, columns = pause times. Each panel overlays the
three variants (unpaused, freeze-only, pause+reduce) for that metric. Pause
instant is a dotted black line; each variant's ASI crossing is a thin colored
line. For `progress` and `ai_research_taste_sd` the ASI threshold level is drawn.

--metrics takes any comma-separated list of the time-series keys in
model.results (run with --list to print them).

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

# Derived metrics (not stored in results) computed from stored arrays + the
# variant's own input series:
#   etsp                = TC_abs(t) + r_software*log10(RS/RS0)  (= EC, the
#                         "effective training system performance" exponent; the
#                         log10 of the INSTANTANEOUS effective-compute rate)
#   progress_rate_true  = 10^(etsp - progress)/ln10  (the exact ODE RHS, i.e. the
#                         TRUE time-derivative of `progress`=logE). As of the rate-naming
#                         reform this equals the STORED results['progress_rates'], so this
#                         derived metric is now a cross-check of the stored value.
# Stored in results (post-reform):
#   progress_rates      = d(progress)/dt (true derivative of logE)
#   etsp_rates          = d(ETSP)/dt = software_progress_rate + TSP growth rate
DERIVED = {"etsp", "progress_rate_true", "lag", "lag_qs"}

# label + preferred y-scale for common metrics
METRIC_META = {
    "progress": ("progress = log10 cumulative eff. compute (OOMs)", "linear"),
    "etsp": ("ETSP = log10 instantaneous eff. compute rate (OOMs)", "linear"),
    "lag": ("lag δ = ETSP − logE (OOMs)", "linear"),
    "lag_qs": ("quasi-static lag = log10(ln10·dETSP/dt) (OOMs)", "linear"),
    "progress_rate_true": ("d(progress)/dt  [derived cross-check, OOMs/yr]", "log"),
    "progress_rates": ("d(progress)/dt  [stored true rate, OOMs/yr]", "log"),
    "etsp_rates": ("d(ETSP)/dt = sw + TSP  [OOMs/yr]", "log"),
    "effective_compute": ("effective compute", "log"),
    "automation_fraction": ("automation fraction", "linear"),
    "ai_research_taste_sd": ("AI research taste (SD)", "linear"),
    "horizon_lengths": ("time horizon (min)", "log"),
    "software_progress_rates": ("software progress rate", "log"),
    "software_efficiency": ("software efficiency", "log"),
    "coding_labors": ("coding labor", "log"),
    "research_stock": ("research stock", "log"),
    "research_efforts": ("research effort", "log"),
    "experiment_capacity": ("experiment capacity", "log"),
    "ai_coding_labor_mult_ref_present_day": ("AI coding-labor mult (ref present)", "log"),
    "ai_sw_progress_mult_ref_present_day": ("AI sw-progress mult (ref present)", "log"),
}
DEFAULT_METRICS = ["progress_rates", "automation_fraction", "ai_research_taste_sd",
                   "horizon_lengths", "software_progress_rates", "effective_compute"]


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


def _metric_series(metric, res, series, r_soft):
    """Return a metric array, computing derived ones (etsp, progress_rate_true)
    from stored arrays + the variant's own input series."""
    t = np.asarray(res["times"], dtype=float)
    if metric == "etsp":
        tc_abs = np.interp(t, np.asarray(series.time, dtype=float),
                           np.asarray(series.training_compute, dtype=float))
        rs = np.asarray(res["research_stock"], dtype=float)
        rs0 = rs[0] if rs.size else 1.0
        se = (r_soft or 0.0) * np.log10(np.maximum(rs, 1e-30) / max(rs0, 1e-30))
        return tc_abs + se
    if metric == "progress_rate_true":
        etsp = _metric_series("etsp", res, series, r_soft)
        prog = np.asarray(res["progress"], dtype=float)
        return np.power(10.0, etsp - prog) / np.log(10.0)
    if metric == "lag":  # δ = ETSP − logE
        return _metric_series("etsp", res, series, r_soft) - np.asarray(res["progress"], dtype=float)
    if metric == "lag_qs":  # quasi-static prediction log10(ln10 · dETSP/dt)
        etsp = _metric_series("etsp", res, series, r_soft)
        g = np.gradient(etsp, t)
        return np.log10(np.maximum(np.log(10.0) * g, 1e-12))
    return np.asarray(res[metric], dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Multi-metric broken-out trajectories at median params")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--pauses", default="2027.0,2029.0,2030.5")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--list", action="store_true", help="List available time-series metrics and exit")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "metric_breakouts"))
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

    base_res = _run(med, base, tr, ip)
    if args.list:
        n = len(base_res["times"])
        keys = [k for k, v in base_res.items()
                if isinstance(v, (list, np.ndarray)) and len(v) == n]
        print("Available time-series metrics:")
        for k in sorted(keys):
            print("  ", k)
        return

    pauses = [float(x) for x in args.pauses.split(",") if x.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ac = _ms_time(base_res, "AC"); asi = _ms_time(base_res, "ASI")
    bt = np.asarray(base_res["times"], dtype=float)
    r_soft = base_res.get("r_software")

    # Per-pause: variant results.
    VARIANTS = [("unpaused", "0.5"), ("paused (freeze)", "C2"), (f"pause+÷{args.reduction:g}", "C0")]
    per_pause = []
    for pause in pauses:
        frozen = freeze_and_reduce_time_series(base, pause)
        reduced = freeze_and_reduce_time_series(
            base, pause, training_reduction=args.reduction, experiment_reduction=args.reduction,
            inference_reduction=args.reduction, labor_reduction=args.reduction)
        fres = _run(med, frozen, tr, ip)
        rres = _run(med, reduced, tr, ip)
        asis = [asi, _ms_time(fres, "ASI"), _ms_time(rres, "ASI")]
        per_pause.append({"pause": pause, "res": [base_res, fres, rres],
                          "series": [base, frozen, reduced], "asis": asis})

    nrow, ncol = len(metrics), len(pauses)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.9 * nrow), squeeze=False)
    for r, metric in enumerate(metrics):
        label, yscale = METRIC_META.get(metric, (metric, "linear"))
        # ASI threshold level for metrics that have one
        thr = None
        if metric in ("progress", "ai_research_taste_sd", "etsp") and asi is not None:
            thr = float(np.interp(asi, bt, _metric_series(metric, base_res, base, r_soft)))
        for c, pp in enumerate(per_pause):
            ax = axes[r][c]
            pause = pp["pause"]
            x1 = (pp["asis"][2] if pp["asis"][2] else tr[1])
            x1 = x1 + 0.05 * (x1 - pause)
            x0 = pause - 0.5
            win_vals = []  # finite in-window values across variants, for robust y-limits
            for (vlabel, color), res, series, at in zip(VARIANTS, pp["res"], pp["series"], pp["asis"]):
                t = np.asarray(res["times"], dtype=float)
                y = _metric_series(metric, res, series, r_soft)
                sel = (t >= x0) & (t <= x1)
                ax.plot(t[sel], y[sel], "-", color=color, lw=1.6,
                        label=vlabel if (r == 0 and c == 0) else None)
                yv = y[sel][np.isfinite(y[sel])]
                if yv.size:
                    win_vals.append(yv)
                if at is not None:
                    ax.axvline(at, color=color, ls="-", lw=0.7, alpha=0.5)
            ax.axvline(pause, color="k", ls=":", lw=1.0)
            if thr is not None:
                ax.axhline(thr, color="C3", ls="--", lw=0.9)
            ax.set_xlim(x0, x1)
            # Robust y-limits: ignore the one-sample spike at the sharp-step
            # discontinuity (a numerical-derivative artifact in rate metrics).
            if win_vals:
                allv = np.concatenate(win_vals)
                pos = allv[allv > 0]
                use_log = (yscale == "log") and pos.size and pos.min() > 0
                if use_log:
                    lo, hi = np.percentile(pos, 1), np.percentile(pos, 99)
                    if lo > 0 and hi > lo:
                        ax.set_yscale("log")
                        ax.set_ylim(lo / 1.5, hi * 1.5)
                else:
                    lo, hi = np.percentile(allv, 1), np.percentile(allv, 99)
                    if hi > lo:
                        pad = 0.08 * (hi - lo)
                        ax.set_ylim(lo - pad, hi + pad)
            if r == 0:
                ax.set_title(f"pause = {pause:.2f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == nrow - 1:
                ax.set_xlabel("year")
            ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Metric trajectories at median params (AC={ac:.2f}, ASI={asi:.2f}) — "
                 f"unpaused / freeze / freeze+÷{args.reduction:g}  "
                 f"(thin vlines = each variant's ASI crossing)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metric_breakouts.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"Wrote {out}  ({nrow} metrics x {ncol} pauses)")


if __name__ == "__main__":
    main()
