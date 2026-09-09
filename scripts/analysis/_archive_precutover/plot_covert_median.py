#!/usr/bin/env python3
"""Plot the covert project's capability (effective-compute) trajectory at the
median scenario: model parameters from sampling_config_thomas.yaml (the
daniel-04-02-26 medians) and covert knobs from sampling_config_covert.yaml.

Effective compute (rebased OOMs) is the model's capability axis — the milestone
lines (AC / SAR / TED-AI / ASI) are fixed capability thresholds, so a trajectory
crossing a line = that project reaching that capability.

    python scripts/analysis/plot_covert_median.py [--years-before-ac N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as cfg  # noqa: E402
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert  # noqa: E402

_LOG10_FPH = np.log10(cfg.FLOP_PER_H100E_YEAR)
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402
from scripts.plotting_utils.style import (  # noqa: E402
    apply_ai2030_style, MILESTONE_COLORS,
    AI2030_NAVY, AI2030_RED, AI2030_WARMGREY, AI2030_STROKE, AI2030_MUTED,
)

apply_ai2030_style()

# (milestone key in results['milestones'], display label)
MS_ORDER = [
    ("AC", "AC"),
    ("SAR-level-experiment-selection-skill", "SAR"),
    ("TED-AI", "TED-AI"),
    ("ASI", "ASI"),
]

# Project colors, from the ai2030-graphs series palette: legal consortium = navy
# ("Plan A"), covert = deep red ("Default World"), no-deal = warm grey.
C_DEFAULT = AI2030_WARMGREY  # no-deal baseline
C_LEGAL = AI2030_NAVY        # legal consortium
C_COVERT = AI2030_RED        # covert project
C_PAUSE = AI2030_STROKE      # pause / setup guide lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="config/sampling_config_thomas.yaml")
    ap.add_argument("--covert-config", default="config/sampling_config_covert.yaml")
    ap.add_argument("--years-before-ac", type=float, default=0.0,
                    help="Pause this many years before AC (0 = at AC). Ignored if --pause-year set.")
    ap.add_argument("--pause-year", type=float, default=None,
                    help="Absolute pause year (e.g. 2029.17 for ~March 2029). Overrides --years-before-ac.")
    ap.add_argument("--total-h100e", type=float, default=None,
                    help="Override covert datacenter size (total H100e). Default: covert config median.")
    ap.add_argument("--legal-mode", choices=["backsolve", "capped"], default="backsolve",
                    help="Legal model: 'backsolve' (exogenous SW) or 'capped' (effort <= AIFM ceiling).")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="Override a model parameter (repeatable), e.g. --set ai_research_taste_slope=1.3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(REPO_ROOT / "outputs" / f"covert_median_{args.legal_mode}.png")

    # --- model params (medians) + covert knobs (fixed medians) ---
    mcfg = _load_config(args.model_config) or {}
    model_params = _get_median_parameter_dict(mcfg.get("parameters", {}) or {})
    model_params.setdefault("horizon_extrapolation_type", "decaying doubling time")
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            model_params[k] = float(v)
        except ValueError:
            model_params[k] = v
        print(f"override {k} = {model_params[k]}")

    ccfg = _load_config(args.covert_config) or {}
    cov = _get_median_parameter_dict(ccfg.get("covert_parameters", {}) or {})
    if args.total_h100e is not None:
        cov["total_h100e"] = args.total_h100e

    input_data = mcfg.get("input_data", "inputs/input_data.csv")
    if not Path(input_data).is_absolute():
        input_data = str((REPO_ROOT / input_data).resolve())
    tr = ccfg.get("time_range") or mcfg.get("time_range") or [2017.0, 2080.0]
    time_range = [float(tr[0]), float(tr[1])]
    t_end = time_range[1]

    data = load_time_series_data(input_data)
    params = Parameters(**model_params)
    ref = run_default(params, data, time_range)

    ac_time = ref["ac_time"]
    t_pause = float(args.pause_year) if args.pause_year is not None else ac_time - float(args.years_before_ac)

    rec = simulate_covert(
        params, data, t_pause,
        alignment_tax_ooms=cov["alignment_tax_ooms"],
        target_year=t_pause + cov["target_lead_years"],
        max_hw_rate=cov["max_hw_rate"],
        total_h100e=cov["total_h100e"],
        frac_stolen=cov["frac_stolen"],
        dormancy_years=cov.get("dormancy_years", 0.0),
        setup_period_years=cov.get("setup_period_years", 0.0),
        setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0),
        t_end=t_end, ref=ref, return_trajectories=True,
        legal_mode=args.legal_mode,
        total_h100e_frac=cov.get("total_h100e_frac"),
    )

    cov_t = np.array(rec["covert_t"]); cov_e = np.array(rec["covert_e"])
    leg_t = np.array(rec["legal_t"]); leg_e = np.array(rec["legal_e"])
    ted = rec["ted_ai_e"]; legal_target = rec["legal_target_e"]
    cttt, lttt = rec["covert_ttt"], rec["legal_ttt"]

    # ETSP decomposition (rates): ETSP = TSP (hardware) + SW (software efficiency)
    leg_tsp = np.array(rec["legal_tsp"]); leg_sw = np.array(rec["legal_sw"]); leg_etsp = leg_tsp + leg_sw
    cov_tsp = np.array(rec["covert_tsp"]); cov_sw = np.array(rec["covert_sw"]); cov_etsp = np.array(rec["covert_etsp"])

    # default (shared, pre-pause) trajectory for context, rebased
    def_t = ref["times"]; def_e = ref["progress"] + ref["rebase_offset"]
    # default ETSP decomposition (same raw frame as legal/covert): TSP from the input
    # training H100e, SW from the default research stock.
    def_tsp = np.interp(def_t, data.time, np.log10(np.maximum(data.training_H100e, 1e-300))) + _LOG10_FPH
    def_sw = ref["r_software"] * np.log10(np.maximum(ref["RS"], 1e-300) / ref["RS_0"])
    def_etsp = def_tsp + def_sw

    # --- plot window: focus on the race, not the post-TED-AI explosion ---
    setup = cov.get("setup_period_years", 0.0); dorm = cov.get("dormancy_years", 0.0)
    asi_e = ref["milestones"].get("ASI", {}).get("e")
    e_top = (asi_e + 1.0) if asi_e is not None else (ted + 6.0)
    e_bot = float(np.interp(t_pause, def_t, def_e)) - 1.0
    x_hi = min(t_end, max([v for v in (cttt, lttt, ref["ted_ai_time"]) if v is not None] + [t_pause]) + 4.0)
    x_lo = t_pause - 2.0

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})

    # milestone capability thresholds (AI-2030 palette: colored dashed lines + labels)
    for key, label in MS_ORDER:
        m = ref["milestones"].get(key)
        if not m or not (e_bot <= m["e"] <= e_top):
            continue
        mc = MILESTONE_COLORS.get(label, "#9ca3af")
        ax.axhline(m["e"], color=mc, lw=1.0, ls="--", alpha=0.45, zorder=1)
        ax.text(x_lo + 0.1, m["e"] + 0.05, label, fontsize=8, color=mc,
                fontweight="bold", va="bottom")

    # legal alignment-tax bar (TED-AI + tax)
    ax.axhline(legal_target, color=C_LEGAL, lw=1.0, ls="--", alpha=0.6, zorder=1)
    ax.text(x_hi, legal_target + 0.05, f"TED-AI + tax ({cov['alignment_tax_ooms']:g})",
            fontsize=8, color=C_LEGAL, ha="right", va="bottom")

    # pause-window shading
    if setup > 0:
        ax.axvspan(t_pause, t_pause + setup, color=C_LEGAL, alpha=0.07, zorder=0)
    if dorm > 0:
        ax.axvspan(t_pause, t_pause + dorm, color=C_COVERT, alpha=0.06, zorder=0)

    # trajectories
    ax.plot(def_t, def_e, color=C_DEFAULT, lw=1.2, alpha=0.7, label="default (no deal)")
    ax.plot(leg_t, leg_e, color=C_LEGAL, lw=2.4, label="legal consortium")
    ax.plot(cov_t, cov_e, color=C_COVERT, lw=2.4, label="covert project")

    # pause + crossings
    ax.axvline(t_pause, color=C_PAUSE, lw=1.0, ls="-", alpha=0.5)
    ax.text(t_pause, e_top, "  pause", fontsize=8, color=C_PAUSE, va="top")
    for t, e, c in [(cttt, ted, C_COVERT), (lttt, legal_target, C_LEGAL)]:
        if t is not None and x_lo <= t <= x_hi:
            ax.plot([t], [e], "o", color=c, ms=6, mec="white", mew=1.0, zorder=5)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(e_bot, e_top)
    ax.set_ylabel("Effective compute  —  AI capability")
    win = ("covert WINS" if rec["covert_wins"]
           else "covert loses") + (f", lead {rec['lead_years']:+.1f} yr"
                                    if rec["lead_years"] is not None else "")
    fig.suptitle("Covert project — median scenario", fontsize=20, y=0.96)
    h100e = cov["total_h100e"]; _frac = cov.get("total_h100e_frac")
    h100e_used = rec.get("total_h100e_used")
    size_str = (f"{_frac*100:.2f}% of frontier R&D compute at pause ({h100e_used/1e3:.0f}k H100e here)"
                if _frac is not None and h100e_used is not None
                else (f"{h100e/1e6:g}M" if h100e >= 1e6 else f"{h100e/1e3:g}k") + " H100e datacenter")
    gain = cov.get("setup_sw_gain_ooms", 0)
    ax.set_title(
        f"Covert: {size_str}, steals {cov['frac_stolen']*100:.0f}% of the "
        f"consortium's algorithmic progress, {dorm:g}-yr construction time\n"
        f"Consortium: {cov['alignment_tax_ooms']:g}-OOM alignment tax, {setup:g}-yr training pause "
        f"(+{gain:g} OOM windfall) · pause {t_pause:.1f}",
        fontsize=10, color=AI2030_MUTED)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(False)

    # --- bottom panel: ETSP decomposition (rates) = TSP (hardware) + SW (software) ---
    if setup > 0:
        ax2.axvspan(t_pause, t_pause + setup, color=C_LEGAL, alpha=0.07, zorder=0)
    if dorm > 0:
        ax2.axvspan(t_pause, t_pause + dorm, color=C_COVERT, alpha=0.06, zorder=0)
    ax2.axvline(t_pause, color=C_PAUSE, lw=1.0, alpha=0.5)

    # default (no-deal) total for context — the shared pre-pause trajectory (left side).
    ax2.plot(def_t, def_etsp, color=C_DEFAULT, lw=1.2, alpha=0.7, label="no deal — total")

    # software-efficiency contribution = shaded band between training compute and the total.
    ax2.fill_between(leg_t, leg_tsp, leg_etsp, color=C_LEGAL, alpha=0.10, zorder=1)
    ax2.fill_between(cov_t, cov_tsp, cov_etsp, color=C_COVERT, alpha=0.10, zorder=1)
    ax2.plot(leg_t, leg_etsp, color=C_LEGAL, lw=2.2, label="legal — total")
    ax2.plot(cov_t, cov_etsp, color=C_COVERT, lw=2.2, label="covert — total")
    ax2.plot(leg_t, leg_tsp, color=C_LEGAL, lw=1.2, ls=":", label="legal — training compute")
    ax2.plot(cov_t, cov_tsp, color=C_COVERT, lw=1.2, ls=":", label="covert — training compute (frozen)")

    # clip y to the race window (trim the post-takeoff explosion tail)
    def _inwin(tt, arr):
        m = (tt >= x_lo) & (tt <= x_hi)
        return arr[m]
    allv = np.concatenate([_inwin(leg_t, leg_etsp), _inwin(cov_t, cov_etsp),
                           _inwin(leg_t, leg_tsp), _inwin(cov_t, cov_tsp)])
    ax2.set_ylim(float(np.nanmin(allv)) - 1.0, float(np.nanpercentile(allv, 85)) + 3.0)

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xlabel("year")
    ax2.set_ylabel("Effective Training System Performance\n(OOMs of FLOP/yr)")
    ax2.set_title(
        "Effective Training System Performance\n"
        "= training compute (hardware, dotted) + software efficiency (algorithmic, shaded)",
        fontsize=11, color=AI2030_MUTED)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"AC={ac_time:.2f}  TED-AI={ref['ted_ai_time']:.2f}  pause={t_pause:.2f}")
    print(f"covert_ttt={cttt}  legal_ttt={lttt}  -> {win}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
