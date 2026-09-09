#!/usr/bin/env python3
"""
Decompose the freeze-vs-reduce ETSP gap into hardware (training compute) and
software (research-stock efficiency) contributions, as a function of progress.

ETSP(p) = TC(p) + SE(p),  SE = r_sw * log10(RS/RS0).
At each progress level p in [p_pause, p_ASI] the reduce variant is slower than
the freeze variant by the pointwise factor

    10^{ΔETSP(p)} = 10^{ΔTC(p)} * 10^{ΔSE(p)},

ΔTC = TC_frz - TC_red  (hardware: the direct training-compute cut, ~log10 R),
ΔSE = SE_frz - SE_red  (software: the research-stock / software-efficiency gap;
                        this is where the experiment+inference compute and labor
                        cuts show up, via slower RS accumulation).

N is the freeze-time-weighted average of 10^{ΔETSP(p)} over the interval.

Top row: ΔETSP and its ΔTC / ΔSE components (OOMs) vs progress.
Bottom row: the multiplicative slowdown factors 10^{Δ·} vs progress (log y).
Columns = pause times. Median params. Runs on aifm-runpod (python3.13).
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


def _ms_time(res, n):
    m = (res or {}).get("milestones", {}).get(n)
    if not m or m.get("time") is None:
        return None
    t = float(m["time"]); return t if np.isfinite(t) else None


def _run(params, data, tr, ip):
    m = ProgressModel(Parameters(**params), data)
    m.compute_progress_trajectory(tr, ip)
    return m.results


def _components(res, series, r_soft):
    """Return progress p, TC(p-array), SE(p-array), ETSP along this trajectory."""
    t = np.asarray(res["times"], dtype=float)
    p = np.asarray(res["progress"], dtype=float)
    RS = np.asarray(res["research_stock"], dtype=float)
    rs0 = RS[0]
    tc = np.interp(t, np.asarray(series.time, dtype=float),
                   np.asarray(series.log10_training_H100e, dtype=float)) + np.log10(cfg.FLOP_PER_H100E_YEAR)
    se = (r_soft or 0.0) * np.log10(np.maximum(RS, 1e-30) / max(rs0, 1e-30))
    return p, tc, se, tc + se


def main():
    ap = argparse.ArgumentParser(description="ETSP gap decomposition (hardware vs software) vs progress")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--pauses", default="2027.0,2029.0,2030.5")
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--p-min", type=float, default=None,
                    help="Lower bound (OOMs) for the x-axis (default: the pause level). "
                         "Set e.g. 26 to show the pre-pause region (where the gap is 0). "
                         "Interpreted on whichever scale --absolute selects.")
    ap.add_argument("--absolute", action="store_true",
                    help="Put the x-axis on the absolute, rebased effective-compute scale "
                         "(OOMs of FLOP; present day ≈ 26.5) instead of the raw progress coordinate.")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "etsp_decomp"))
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
    R = args.reduction
    pauses = [float(x) for x in args.pauses.split(",") if x.strip()]

    base_res = _run(med, base, tr, ip)
    r_soft = base_res.get("r_software")
    ac = _ms_time(base_res, "AC"); asi0 = _ms_time(base_res, "ASI")
    # Additive shift from raw progress (logE) to the rebased absolute effective-compute
    # scale (OOMs FLOP; present-day reference). Constant across variants (they coincide
    # pre-pause), so it just relabels the x-axis.
    if args.absolute:
        ec = np.asarray(base_res["effective_compute"], dtype=float)
        pr0 = np.asarray(base_res["progress"], dtype=float)
        shift = float(np.median(ec - pr0))
        xlabel = "effective compute (OOMs FLOP, absolute; present≈26.5)"
    else:
        shift = 0.0
        xlabel = "progress (OOMs eff. compute)"

    ncol = len(pauses)
    fig, axes = plt.subplots(2, ncol, figsize=(5.4 * ncol, 8.6), squeeze=False)
    for c, pause in enumerate(pauses):
        frz = freeze_and_reduce_time_series(base, pause)
        red = freeze_and_reduce_time_series(
            base, pause, training_reduction=R, experiment_reduction=R,
            inference_reduction=R, labor_reduction=R)
        rf = _run(med, frz, tr, ip); rr = _run(med, red, tr, ip)
        asi_f = _ms_time(rf, "ASI"); asi_r = _ms_time(rr, "ASI")
        pf, tcf, sef, etf = _components(rf, frz, r_soft)
        pr, tcr, ser, etr = _components(rr, red, r_soft)

        p_pause = float(np.interp(pause, np.asarray(rf["times"], dtype=float), pf))
        p_asi = float(np.interp(asi_f, np.asarray(rf["times"], dtype=float), pf))
        p_ac = float(np.interp(ac, np.asarray(base_res["times"], dtype=float),
                               np.asarray(base_res["progress"], dtype=float)))
        # p_min is given on the active axis scale; convert to progress for the grid.
        p_lo = (args.p_min - shift) if args.p_min is not None else p_pause
        P = np.linspace(p_lo, p_asi, 600)
        X = P + shift  # x-axis coordinate (progress, or absolute effective compute)

        def on_grid(p_arr, v_arr):
            return np.interp(P, p_arr, v_arr)
        # Absolute levels on a common progress grid.
        tcf_g, tcr_g = on_grid(pf, tcf), on_grid(pr, tcr)   # training compute (hardware)
        sef_g, ser_g = on_grid(pf, sef), on_grid(pr, ser)   # software efficiency
        etf_g, etr_g = tcf_g + sef_g, tcr_g + ser_g          # ETSP = TC + SE
        # Reduce-run perspective gap (for the multiplicative bottom row).
        dTC = tcr_g - tcf_g
        dSE = ser_g - sef_g
        dET = dTC + dSE

        N = (asi_r - pause) / (asi_f - pause)

        # ---- Top: ABSOLUTE ETSP levels, decomposed TC (hardware) + SE (software) ----
        ax = axes[0][c]
        ax.plot(X, etf_g, color="k", lw=2.0, label="ETSP, freeze")
        ax.plot(X, etr_g, color="k", lw=2.0, ls="--", label="ETSP, reduce")
        ax.plot(X, tcf_g, color="C1", lw=1.4, label="TC hardware, freeze")
        ax.plot(X, tcr_g, color="C1", lw=1.4, ls="--", label="TC hardware, reduce")
        ax.fill_between(X, tcf_g, etf_g, color="C2", alpha=0.12)
        ax.fill_between(X, tcr_g, etr_g, color="C2", alpha=0.22, label="SE software (gap TC→ETSP)")
        for xlvl, lbl, col in [(p_pause + shift, "pause", "k"), (p_ac + shift, "AC", "0.5"), (p_asi + shift, "ASI", "C3")]:
            if X[0] <= xlvl <= X[-1]:
                ax.axvline(xlvl, color=col, ls=":", lw=1.0)
                ax.text(xlvl, ax.get_ylim()[1], f" {lbl}", color=col, va="top", ha="left", fontsize=7, rotation=90)
        ax.set_title(f"pause={pause:.2f}   N={N:.2f}\nabsolute ETSP = TC + SE (OOMs FLOP/yr)", fontsize=10)
        ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel("ETSP / TC / SE  (absolute OOMs, FLOP/yr)")
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax2 = axes[1][c]
        ax2.plot(X, 10 ** dET, color="k", lw=2.0, label="net (reduced ÷ freeze rate)")
        ax2.plot(X, 10 ** dTC, color="C1", lw=1.6, label=f"training compute (×1/R = {1/R:g})")
        ax2.plot(X, 10 ** dSE, color="C2", lw=1.6, label="software efficiency (SE)")
        ax2.axhline(1.0, color="0.7", lw=0.8)
        for xlvl, col in [(p_pause + shift, "k"), (p_ac + shift, "0.5"), (p_asi + shift, "C3")]:
            if X[0] <= xlvl <= X[-1]:
                ax2.axvline(xlvl, color=col, ls="--", lw=1.0)
        ax2.set_yscale("log")
        ax2.set_xlabel(xlabel)
        if c == 0:
            ax2.set_ylabel("progress rate: reduced ÷ freeze\n↓ = reduced run is slower")
            ax2.legend(fontsize=8)
        ax2.set_title(f"reduced runs at this fraction of freeze speed → N≈1/avg≈{N:.1f}", fontsize=9)
        ax2.grid(alpha=0.3, which="both")

    fig.suptitle(f"Effect of the reduction on the reduced run's ETSP vs progress (median params, AC={ac:.2f}, ASI={asi0:.2f}, R={R:g})\n"
                 f"hardware (training compute, ↓) vs software (efficiency SE = r·log10(RS/RS0), ↑ partial recovery); "
                 f"top = OOMs (add), bottom = rate factors (multiply)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "etsp_decomp.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
