#!/usr/bin/env python3
"""Covert analog of pause_reduction_etsp_decomp.py: decompose the covert project's
ETSP penalty (vs a frozen-frontier baseline) into hardware (training compute) and
software (research-stock efficiency) contributions, as a function of progress —
and show how *software theft* changes that software band, and hence the implied
slowdown ("N").

ETSP = TC + SE,  TC = log10(H100e · FLOP/H100e-yr),  SE = r_sw·log10(RS/RS0).

Three runs per pause, all starting from the frontier's state at the pause:
  • FREEZE  — frontier frozen at the pause (no reduction, no theft); the baseline.
  • COVERT  — the current covert sim (small datacenter + S-theft of the legal project).
  • COVERT⁰ — the same covert with frac_stolen = 0 (no theft), to isolate the theft.

Top row: absolute ETSP = TC + SE (OOMs FLOP/yr) vs effective compute.
Bottom row: slowdown factor 10^ΔETSP = (covert ÷ freeze rate), split into hardware
(10^ΔTC, ≈ const) and software (10^ΔSE, the recovery — boosted by theft until the
legal project sits and the theft dries up).
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
from plan_a_covert_proj.sim import run_default, simulate_covert, _tc_ooms  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402

_LOG10_FPH = np.log10(cfg.FLOP_PER_H100E_YEAR)


def _ms_time(res, n):
    m = (res or {}).get("milestones", {}).get(n)
    if not m or m.get("time") is None:
        return None
    t = float(m["time"]); return t if np.isfinite(t) else None


def freeze_components(med, base, pause, tr, ip, r_soft, offset):
    """Frozen-frontier baseline: (eff_compute_rebased, TC, SE, ETSP) vs its trajectory."""
    frz = freeze_and_reduce_time_series(base, pause)   # no reduction => freeze
    m = ProgressModel(Parameters(**med), frz)
    m.compute_progress_trajectory(tr, ip)
    r = m.results
    t = np.asarray(r["times"], float)
    ec = np.asarray(r["progress"], float) + offset      # rebased effective compute
    RS = np.asarray(r["research_stock"], float)
    tc = np.interp(t, frz.time, np.log10(np.maximum(frz.training_H100e, 1e-300))) + _LOG10_FPH
    se = r_soft * np.log10(np.maximum(RS, 1e-30) / RS[0])
    ted = _ms_time(r, "TED-AI")
    return ec, tc, se, tc + se, ted


def covert_components(rec, offset):
    """Covert run components from simulate_covert(return_trajectories=True)."""
    ec = np.asarray(rec["covert_e"], float)             # already rebased
    tc = np.asarray(rec["covert_tsp"], float)
    se = np.asarray(rec["covert_sw"], float)
    et = np.asarray(rec["covert_etsp"], float)
    return ec, tc, se, et, rec["covert_ttt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--covert-config", default="plan_a_covert_proj/config/covert_scenario_branch.yaml")
    ap.add_argument("--pauses", default="2028.0,2029.17,2029.9")
    ap.add_argument("--t-end", type=float, default=2120.0,
                    help="Integration horizon (override the covert config's, so the covert "
                         "has time to climb well past TED-AI).")
    ap.add_argument("--x-max", type=float, default=None,
                    help="Upper effective-compute (OOMs) for the x-axis (default: ASI + 2).")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "covert_etsp_decomp.png"))
    args = ap.parse_args()

    mcfg = _load_config(args.model_config) or {}
    med = _get_median_parameter_dict(mcfg.get("parameters", {}) or {})
    med.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    ccfg = _load_config(args.covert_config) or {}
    cov = _get_median_parameter_dict(ccfg.get("covert_parameters", {}) or {})

    input_path = str(mcfg.get("input_data", "inputs/input_data.csv"))
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)
    tr = ccfg.get("time_range") or [2017.0, 2080.0]
    tr = [float(tr[0]), float(args.t_end)]   # extend the horizon for this plot
    ip = float(mcfg.get("initial_progress", 0.0))
    pauses = [float(x) for x in args.pauses.split(",") if x.strip()]

    params = Parameters(**med)
    ref = run_default(params, base, tr)
    r_soft = ref["r_software"]; offset = ref["rebase_offset"]
    ted_e = ref["ted_ai_e"]; asi_e = ref["milestones"].get("ASI", {}).get("e")

    def covert_run(pause, frac_stolen):
        return simulate_covert(
            params, base, pause,
            alignment_tax_ooms=cov["alignment_tax_ooms"],
            target_year=pause + cov["legal_scaling_duration"], max_hw_rate=cov["max_hw_rate"],
            compute_mode="absolute", total_h100e=cov["total_h100e"], frac_stolen=frac_stolen,
            dormancy_years=cov.get("dormancy_years", 0.0),
            setup_period_years=cov.get("setup_period_years", 0.0),
            setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0),
            t_end=tr[1], ref=ref, return_trajectories=True)

    ncol = len(pauses)
    fig, axes = plt.subplots(2, ncol, figsize=(5.6 * ncol, 8.8), squeeze=False)

    for c, pause in enumerate(pauses):
        ecf, tcf, sef, etf, ted_f = freeze_components(med, base, pause, tr, ip, r_soft, offset)
        rec = covert_run(pause, cov["frac_stolen"]); rec0 = covert_run(pause, 0.0)
        ecc, tcc, sec, etc, ttt = covert_components(rec, offset)
        ec0, tc0, se0, et0, ttt0 = covert_components(rec0, offset)

        # common effective-compute grid: pause level → well past ASI, but never
        # beyond what each run actually reaches (so we don't extrapolate flat tails).
        x_lo = float(np.interp(pause, ref["times"], ref["progress"])) + offset
        if args.x_max is not None:
            x_hi = float(args.x_max)
        else:
            x_hi = (asi_e + 2.0) if asi_e else ((ted_e + 5.0) if ted_e else x_lo + 8.0)
        x_hi = min(x_hi, float(ecf[-1]), float(ecc[-1]), float(ec0[-1]))
        X = np.linspace(x_lo, x_hi, 600)

        def og(ec, v):  # monotone interp onto X (ec is increasing)
            return np.interp(X, ec, v)
        tcf_g, sef_g = og(ecf, tcf), og(ecf, sef); etf_g = tcf_g + sef_g
        tcc_g, sec_g = og(ecc, tcc), og(ecc, sec); etc_g = tcc_g + sec_g
        tc0_g, se0_g = og(ec0, tc0), og(ec0, se0); et0_g = tc0_g + se0_g

        # slowdown factors (covert ÷ freeze)
        dTC, dSE = tcc_g - tcf_g, sec_g - sef_g
        dSE0 = se0_g - sef_g

        # effective N (time to TED-AI vs freeze)
        N = ((ttt - pause) / (ted_f - pause)) if (ttt and ted_f and ted_f > pause) else np.nan
        N0 = ((ttt0 - pause) / (ted_f - pause)) if (ttt0 and ted_f and ted_f > pause) else np.nan
        # effective-compute level on the COVERT's axis where the legal sits and the
        # covert's theft dries up = the covert's E at target_year (NOT the legal's
        # target E — the covert is far behind the legal at that moment).
        target_year = pause + cov["legal_scaling_duration"]
        legE = float(np.interp(target_year, np.asarray(rec["covert_t"], float),
                               np.asarray(rec["covert_e"], float)))

        # ---- top: absolute ETSP = TC + SE ----
        ax = axes[0][c]
        ax.plot(X, etf_g, color="k", lw=2.0, label="ETSP freeze (frozen frontier)")
        ax.plot(X, etc_g, color="#a23b3b", lw=2.0, label="ETSP covert (theft)")
        ax.plot(X, et0_g, color="#a23b3b", lw=1.3, ls="--", label="ETSP covert (no theft)")
        ax.plot(X, tcf_g, color="#3a6cb0", lw=1.2, label="TC freeze (hardware)")
        ax.plot(X, tcc_g, color="#d08770", lw=1.2, label="TC covert (hardware, frozen)")
        ax.fill_between(X, tcc_g, etc_g, color="#a23b3b", alpha=0.10)  # covert SE band
        for xl, lbl, col in [(x_lo, "pause", "k"), (ted_e, "TED-AI", "#777"),
                             (asi_e, "ASI", "#777")]:
            if xl and X[0] <= xl <= X[-1]:
                ax.axvline(xl, color=col, ls=":", lw=1.0)
                ax.text(xl, ax.get_ylim()[1], f" {lbl}", color=col, va="top", fontsize=7, rotation=90)
        ax.set_title(f"pause={pause:.2f}   N_theft={N:.1f}  vs  N_no-theft={N0:.1f}\n"
                     f"absolute ETSP = TC + SE (OOMs FLOP/yr)", fontsize=9)
        ax.set_xlabel("effective compute (rebased OOMs)")
        if c == 0:
            ax.set_ylabel("ETSP / TC / SE  (OOMs FLOP/yr)")
            ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

        # ---- bottom: slowdown factors (covert ÷ freeze) ----
        ax2 = axes[1][c]
        ax2.plot(X, 10 ** (dTC + dSE), color="#a23b3b", lw=2.0, label="net covert÷freeze (theft)")
        ax2.plot(X, 10 ** (dTC + dSE0), color="#a23b3b", lw=1.3, ls="--", label="net (no theft)")
        ax2.plot(X, 10 ** dTC, color="#3a6cb0", lw=1.5, label="hardware (TC)")
        ax2.plot(X, 10 ** dSE, color="#2e8b57", lw=1.5, label="software w/ theft (SE)")
        ax2.plot(X, 10 ** dSE0, color="#2e8b57", lw=1.3, ls="--", label="software no theft")
        ax2.axhline(1.0, color="0.7", lw=0.8)
        if legE and X[0] <= legE <= X[-1]:
            ax2.axvline(legE, color="#3a6cb0", ls="-.", lw=1.0)
            ax2.text(legE, ax2.get_ylim()[1], " legal sits\n(theft ends)", color="#3a6cb0",
                     va="top", fontsize=7)
        for xl in (ted_e, asi_e):
            if xl and X[0] <= xl <= X[-1]:
                ax2.axvline(xl, color="#777", ls=":", lw=1.0)
        ax2.set_yscale("log")
        ax2.set_xlabel("effective compute (rebased OOMs)")
        if c == 0:
            ax2.set_ylabel("rate: covert ÷ freeze\n↓ = covert slower")
            ax2.legend(fontsize=7, loc="lower left")
        ax2.set_title("covert runs at this fraction of frozen-frontier speed", fontsize=9)
        ax2.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Covert ETSP decomposition vs a frozen frontier — how software THEFT changes the slowdown (N)\n"
        f"median params, AC={ref['ac_time']:.2f}, TED-AI_e={ted_e:.1f}; covert={cov['total_h100e']:.0e} H100e, "
        f"frac_stolen={cov['frac_stolen']:.2f}; hardware (↓) vs software (↑ recovery, theft-boosted until legal sits)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
