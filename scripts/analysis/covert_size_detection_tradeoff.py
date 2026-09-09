#!/usr/bin/env python3
"""Covert success vs total compute, and the size-vs-detection tradeoff.

Bigger covert compute => reaches TED-AI faster (more own R&D), BUT at a fixed per-site
cluster size it means more sites => higher detection hazard. So expected success
  P(reach TED-AI before detection) = [reaches] * P(survive detection until covert_ttt)
can be NON-monotonic in total compute: too small => too slow; too big => caught.

Controlled sweep: model fixed at the median (model_uncertain_thomas medians), median
covert knobs, plan-a.csv. The covert's arrival depends only on total compute (the per-site
cluster size affects only detection, which is analytic), so we run the sim once per size
and evaluate survival for several cluster sizes cheaply. Two legal regimes: full tuned
software (the covert leans on theft) vs no tuned software (the covert must self-power on
its own compute — where size matters most).

    python scripts/analysis/covert_size_detection_tradeoff.py <out.png>
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from progress_model import Parameters, load_time_series_data            # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert         # noqa: E402
import plan_a_covert_proj.detection as D                                # noqa: E402
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402
try:
    from scripts.plotting_utils.style import (apply_ai2030_style, AI2030_NAVY, AI2030_RED,
                                              AI2030_OLIVE, AI2030_MUTED, AI2030_GRID, AI2030_AMBER)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID, AI2030_AMBER = (
        "#1f4e9c", "#a23b3b", "#6b8e23", "#777", "#ccc", "#c8862b")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

MODEL = "plan_a_covert_proj/config/model_uncertain_thomas.yaml"
COVERT = "plan_a_covert_proj/config/covert_uncertain.yaml"   # has detection grids + silver bullet
SITE_MWS = [100.0, 300.0, 1000.0]      # per-site cluster sizes to hold fixed
N_SIZES = 36
TOTAL_H100E_GRID = np.logspace(np.log10(5e4), np.log10(2e7), N_SIZES)


def main(out_path):
    mc = _load_config(MODEL)
    med = _get_median_parameter_dict(mc.get("parameters", {}) or {})
    med.setdefault("horizon_extrapolation_type", "decaying doubling time")
    tr = [float((mc.get("time_range") or [2017, 2100])[0]), 2100.0]
    data = load_time_series_data(str(ROOT / (mc.get("input_data") or "inputs/input_data.csv")))

    cc = _load_config(COVERT)
    cov = _get_median_parameter_dict(cc.get("covert_parameters", {}))
    fac = float(cov["facility_w_per_h100e"])
    fits, r = D.fit_detection_model(cc.get("detection_grids"))
    # NOTE: the silver bullet is now a per-rollout ceiling-logistic anchored on the legal
    # used->TED-AI crossing, which this standalone size-sweep does not simulate. It is a
    # size-INDEPENDENT multiplicative factor on survival, so omitting it leaves the
    # size↔detection tradeoff shape intact (only absolute survival shifts up). The full
    # legal-coupled silver bullet lives in the pipeline (mc.py + detection.build_silver_bullet).

    p = Parameters(**med); ref = run_default(p, data, tr); tp = ref["ac_time"] - 0.8

    def surv_to_goal(total_h100e, site_mw, t_risk):
        size_mw = total_h100e * fac / 1e6
        n, eff = D.deploy(size_mw, site_mw)
        return float(D.survival(D.interp_frailty(eff, fits), n, t_risk, r))

    regimes = {  # legal software strategy
        "legal full software": dict(legal_strategy="target_year", max_sw_rate=None),
        "legal NO software":   dict(legal_strategy="target_year_capped_sw", max_sw_rate=0.0),
    }
    res = {}
    for rlab, rkw in regimes.items():
        ttt, trisk = [], []
        for h in TOTAL_H100E_GRID:
            rec = simulate_covert(
                p, data, tp, alignment_tax_ooms=cov["alignment_tax_ooms"],
                target_year=tp + cov["legal_scaling_duration"], max_hw_rate=cov["max_hw_rate"],
                compute_mode="absolute", total_h100e=float(h), frac_stolen=cov["frac_stolen"],
                dormancy_years=cov.get("dormancy_years", 0.0), setup_period_years=cov.get("setup_period_years", 0.0),
                setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0), t_end=tr[1], ref=ref,
                legal_mode="capped", **rkw)
            c = rec["covert_ttt"]
            ttt.append(c); trisk.append((c - tp) if c is not None else np.nan)
        res[rlab] = (np.array(ttt, dtype=object), np.array(trisk, float))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))
    # Panel A: expected success vs total compute, per cluster size, both regimes
    styles = {"legal full software": "-", "legal NO software": "--"}
    cols = {100.0: AI2030_OLIVE, 300.0: AI2030_NAVY, 1000.0: AI2030_RED}
    for rlab, (ttt, trisk) in res.items():
        for site in SITE_MWS:
            succ = np.array([0.0 if not np.isfinite(t) else surv_to_goal(h, site, t)
                             for h, t in zip(TOTAL_H100E_GRID, trisk)])
            axA.plot(TOTAL_H100E_GRID / 1e6, succ, styles[rlab], color=cols[site], lw=2,
                     label=f"{rlab}, site={site:g} MW")
            k = int(np.argmax(succ))
            axA.plot(TOTAL_H100E_GRID[k] / 1e6, succ[k], "o", color=cols[site], ms=5)
    axA.set_xscale("log")
    axA.set_xlabel("total covert compute (M H100e)")
    axA.set_ylabel("P(reach TED-AI before detection)")
    axA.set_title("(a) covert success vs size — dots mark the optimum", fontsize=10)
    axA.legend(fontsize=7.2, loc="best"); axA.grid(True, alpha=.15)

    # Panel B: decompose the tradeoff for one regime + cluster size (no-SW, 300 MW)
    ttt, trisk = res["legal NO software"]
    reach = np.array([1.0 if np.isfinite(t) else 0.0 for t in trisk])
    surv = np.array([0.0 if not np.isfinite(t) else surv_to_goal(h, 300.0, t)
                     for h, t in zip(TOTAL_H100E_GRID, trisk)])
    axB.plot(TOTAL_H100E_GRID / 1e6, reach, color=AI2030_OLIVE, lw=2, label="reaches TED-AI (speed) ↑")
    axB.plot(TOTAL_H100E_GRID / 1e6, surv, color=AI2030_AMBER, lw=2, label="survives detection to goal ↓")
    axB.plot(TOTAL_H100E_GRID / 1e6, reach * surv, color="#222", lw=2.6, label="success = reach × survive")
    axB.set_xscale("log"); axB.set_ylim(0, 1.02)
    axB.set_xlabel("total covert compute (M H100e)")
    axB.set_ylabel("probability")
    axB.set_title("(b) the two opposing effects (legal NO software, site = 300 MW)", fontsize=10)
    axB.legend(fontsize=8, loc="center left"); axB.grid(True, alpha=.15)

    fig.suptitle("Covert success vs total compute: speed gain vs detection risk  ·  median scenario, plan-a.csv",
                 fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print("wrote", out_path)
    # quick console readout of the optima
    for rlab, (ttt, trisk) in res.items():
        for site in SITE_MWS:
            succ = np.array([0.0 if not np.isfinite(t) else surv_to_goal(h, site, t)
                             for h, t in zip(TOTAL_H100E_GRID, trisk)])
            k = int(np.argmax(succ))
            print(f"  {rlab:20s} site={site:6g}MW: peak success {succ[k]:.2f} at "
                  f"{TOTAL_H100E_GRID[k]/1e6:.2f}M H100e (max-size success {succ[-1]:.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "covert_size_detection_tradeoff.png")
