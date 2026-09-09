#!/usr/bin/env python3
"""Physics check: given the SAME amount of compute, does any LEGAL trajectory
extract more effective compute than its own DEFAULT trajectory did?

The legal project does NOT freeze compute: after a short setup it ramps training
compute (TSP) at max_hw_rate = 0.5 OOM/yr, and runs an exogenous back-solved
software rate plus a one-time "radical transparency" software windfall. Because
the default training-compute ramp decelerates (~0.5 -> ~0.2 OOM/yr) while the
legal ramp is a flat 0.5, the legal project can have MORE compute than the
default at a given capability -- so comparing rates at matched capability would
wrongly flag that as a violation.

The compute-matched test: dE/dt = 10^(TSP + SW), so at equal compute (TSP) the
rate ratio is 10^(SW_legal - SW_default). We therefore compare software
efficiency SW as a function of TSP. A positive (SW_legal - SW_default) at matched
TSP means the legal project is squeezing more effective compute out of the same
FLOP than the unpaused frontier ever did -- i.e. faster than the default would
allow given the same compute. The transparency windfall (pooled algorithms) is
the mechanism that can produce this.

    python scripts/analysis/legal_physics_check.py --json outputs/covert_mc_500.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert  # noqa: E402
from progress_model.progress_rate import progress_rate_at_time  # noqa: E402
from scripts.pipelines.batch_rollout import _suppress_noise  # noqa: E402
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID = (
        "#1f4e9c", "#a23b3b", "#6b8e23", "#777", "#ccc")

_W = {}
REL_TOL = 1.05   # call effort > 1.05x the AIFM ceiling a real violation


def _init(input_path, cov, yba, tr, legal_mode):
    _W.update(dict(data=load_time_series_data(input_path), cov=cov, yba=yba, tr=tr,
                   legal_mode=legal_mode))


def _check_one(rec, example=False):
    """Reconstruct one rollout. Compare the legal project's back-derived research
    effort to the AIFM production-function ceiling for its capability + frontier
    resources (via progress_rate_at_time, which returns dRS = research_effort).
    Returns scalar stats (plus example arrays if requested)."""
    cov, tr, data = _W["cov"], _W["tr"], _W["data"]
    try:
        params = Parameters(**rec["model_params"])
        with _suppress_noise():
            ref = run_default(params, data, tr)
            if ref["ac_time"] is None or ref["ted_ai_e"] is None:
                return None
            tp = ref["ac_time"] - _W["yba"]
            setup = cov.get("setup_period_years", 0.0)
            r = simulate_covert(
                params, data, tp,
                alignment_tax_ooms=cov["alignment_tax_ooms"],
                target_year=tp + cov["target_lead_years"], max_hw_rate=cov["max_hw_rate"],
                total_h100e=cov["total_h100e"], frac_stolen=cov["frac_stolen"],
                dormancy_years=cov.get("dormancy_years", 0.0),
                setup_period_years=setup, setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0),
                t_end=tr[1], ref=ref, return_trajectories=True,
                legal_mode=_W.get("legal_mode", "backsolve"))

            leg_t = np.asarray(r["legal_t"], float)
            leg_e = np.asarray(r["legal_e"], float)        # rebased OOMs
            leg_sw = np.asarray(r["legal_sw"], float)      # raw software efficiency (OOMs)
            if leg_t.size < 4:
                return None
            off = ref["rebase_offset"]; RS_0 = ref["RS_0"]; r_sw = params.r_software

            # legal's back-derived research effort: re = d/dt [RS_0 * 10^(SW/r_sw)]
            rs_legal = RS_0 * 10.0 ** (leg_sw / r_sw)
            re_legal = np.gradient(rs_legal, leg_t)

            # The legal only does R&D until it sits at TED-AI+tax (re -> 0 after).
            # Restrict to the active window and subsample to keep the oracle cheap.
            active = np.where(re_legal > 1e-12 * max(rs_legal.max(), 1e-30))[0]
            if active.size < 3:
                return {"index": rec["index"], "setup_max_ratio": 0.0, "post_max_ratio": 0.0,
                        "post_frac_time_violating": 0.0, "ac_to_ted_gap": rec.get("ac_to_ted_gap"),
                        "covert_wins": rec["covert_wins"], "sw_rate_solved": rec.get("sw_rate_solved")}
            a0, a1 = int(active[0]), int(active[-1])
            idx = np.unique(np.linspace(a0, a1, min(300, a1 - a0 + 1)).astype(int))
            leg_t = leg_t[idx]; re_legal = re_legal[idx]; rs_legal = rs_legal[idx]
            logE_raw = leg_e[idx] - off

            # AIFM ceiling: research_effort the model allows at the legal's capability
            # (raw logE) with frontier resources at each time t.
            ceil = np.empty_like(leg_t)
            for i, tt in enumerate(leg_t):
                ceil[i] = progress_rate_at_time(
                    float(tt), [float(logE_raw[i]), float(max(rs_legal[i], 1e-30))],
                    data, params, RS_0=RS_0)[1]
    except Exception:
        return None

    in_setup = leg_t <= (tp + setup + 1e-9)
    post = (~in_setup) & (re_legal > 0)
    ceil_safe = np.maximum(ceil, 1e-30)
    ratio = re_legal / ceil_safe

    def _stat(mask):
        if not np.any(mask):
            return 0.0, 0.0
        rr = ratio[mask]
        return float(np.nanmax(rr)), float(np.mean(rr > REL_TOL))

    setup_max, _ = _stat(in_setup & (re_legal > 0))
    post_max, post_frac_viol = _stat(post)
    out = {
        "index": rec["index"],
        "setup_max_ratio": setup_max,          # windfall period (intended free boost)
        "post_max_ratio": post_max,            # sustained policy ramp (the real test)
        "post_frac_time_violating": post_frac_viol,
        "ac_to_ted_gap": rec.get("ac_to_ted_gap"),
        "covert_wins": rec["covert_wins"],
        "sw_rate_solved": rec.get("sw_rate_solved"),
    }
    if example:
        out["ex"] = {"t": leg_t.tolist(), "re_legal": re_legal.tolist(),
                     "ceil": ceil.tolist(), "tp": tp, "setup": setup}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "legal_physics_check.png"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "outputs" / "legal_physics_check_summary.json"))
    args = ap.parse_args()

    d = json.load(open(args.json))
    recs = d["records"]
    cov = d["config"]["covert_knobs_fixed"]
    yba = d["config"].get("years_before_ac", 0.0)
    tr = [float(d["config"]["time_range"][0]), float(d["config"]["time_range"][1])]
    input_path = d["config"]["input_data"]
    if not Path(input_path).exists():
        input_path = str((REPO_ROOT / "inputs" / "input_data.csv").resolve())
    legal_mode = d["config"].get("legal_mode", "backsolve")
    print(f"legal_mode = {legal_mode}")

    init_args = (input_path, cov, yba, tr, legal_mode)
    workers = max(1, min(args.workers, len(recs)))
    if workers == 1:
        _init(*init_args)
        results = [_check_one(r) for r in recs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init, initargs=init_args) as pool:
            results = pool.map(_check_one, recs, chunksize=4)
    res = [r for r in results if r is not None]

    setup_r = np.array([r["setup_max_ratio"] for r in res])
    post_r = np.array([r["post_max_ratio"] for r in res])
    post_viol = np.array([r["post_frac_time_violating"] for r in res])
    n_post_flag = int(np.sum(post_r > REL_TOL))
    n_setup_flag = int(np.sum(setup_r > REL_TOL))

    print(f"checked {len(res)}/{len(recs)} rollouts  (effort / AIFM ceiling; >1 = over budget)")
    print(f"SETUP/windfall period over ceiling (>{REL_TOL}x): {n_setup_flag} "
          f"({100*n_setup_flag/max(len(res),1):.1f}%)  "
          f"setup_max_ratio p50 {np.nanpercentile(setup_r,50):.2f} p99 {np.nanpercentile(setup_r,99):.2f} max {np.nanmax(setup_r):.2f}")
    print(f"SUSTAINED (post-setup policy) over ceiling: {n_post_flag} "
          f"({100*n_post_flag/max(len(res),1):.1f}%)  "
          f"post_max_ratio p50 {np.nanpercentile(post_r,50):.2f} p90 {np.nanpercentile(post_r,90):.2f} "
          f"p99 {np.nanpercentile(post_r,99):.2f} max {np.nanmax(post_r):.2f}")
    for thr in (2.0, 10.0):
        print(f"  sustained peak ratio > {thr:g}x: {int(np.sum(post_r>thr))} "
              f"({100*np.mean(post_r>thr):.1f}%)")
    if n_post_flag:
        fv = np.array([r["ac_to_ted_gap"] for r in res
                       if r["post_max_ratio"] > REL_TOL and r["ac_to_ted_gap"] is not None])
        ok = np.array([r["ac_to_ted_gap"] for r in res
                       if r["post_max_ratio"] <= REL_TOL and r["ac_to_ted_gap"] is not None])
        if fv.size and ok.size:
            print(f"  sustained-violation AC->TED gap median {np.median(fv):.2f}y "
                  f"vs ok {np.median(ok):.2f}y")

    # ---- figure (log-ratio; the tail blows up where the ceiling momentarily ~0) ----
    fig, (axh, axs) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    lo_s = np.log10(np.clip(setup_r, 1e-3, 1e8))
    lo_p = np.log10(np.clip(post_r, 1e-3, 1e8))
    bins = np.linspace(-1.5, 4.0, 45)
    axh.hist(np.clip(lo_s, bins[0], bins[-1]), bins=bins, color=AI2030_OLIVE, alpha=0.55,
             label=f"setup windfall (peak)  [{n_setup_flag}/{len(res)} > ceiling]")
    axh.hist(np.clip(lo_p, bins[0], bins[-1]), bins=bins, color=AI2030_NAVY, alpha=0.7,
             label=f"sustained policy (peak)  [{n_post_flag}/{len(res)} > ceiling]")
    axh.axvline(0.0, color=AI2030_RED, lw=1.6)
    axh.text(0.05, axh.get_ylim()[1] * 0.92, "ceiling (1x)", color=AI2030_RED, fontsize=8)
    axh.set_xlabel("log₁₀  (legal research effort) / (AIFM resource+capability ceiling)")
    axh.set_ylabel("rollouts"); axh.legend(fontsize=8.5, loc="upper right")
    axh.set_title("(a) peak effort vs production-function ceiling", fontsize=10)

    gap = np.array([r["ac_to_ted_gap"] if r["ac_to_ted_gap"] is not None else np.nan for r in res])
    win = np.array([r["covert_wins"] for r in res])
    axs.scatter(gap[~win], lo_p[~win], s=12, color=AI2030_RED, alpha=0.5, label="covert loses")
    axs.scatter(gap[win], lo_p[win], s=12, color=AI2030_OLIVE, alpha=0.7, label="covert wins")
    axs.axhline(0.0, color=AI2030_RED, lw=1.2)
    axs.set_xlabel("default AC→TED-AI gap (years)  —  takeoff slowness")
    axs.set_ylabel("log₁₀ sustained peak effort / ceiling")
    axs.set_title("(b) violation grows with slower takeoff", fontsize=10)
    axs.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Legal project: is back-derived research effort within the AIFM resource+capability ceiling?",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)

    summary = {
        "n_checked": len(res), "rel_tol": REL_TOL,
        "n_setup_over_ceiling": n_setup_flag, "n_sustained_over_ceiling": n_post_flag,
        "frac_sustained_over_ceiling": n_post_flag / max(len(res), 1),
        "setup_max_ratio_p50": float(np.nanpercentile(setup_r, 50)),
        "setup_max_ratio_max": float(np.nanmax(setup_r)),
        "post_max_ratio_p50": float(np.nanpercentile(post_r, 50)),
        "post_max_ratio_p90": float(np.nanpercentile(post_r, 90)),
        "post_max_ratio_max": float(np.nanmax(post_r)),
        "sustained_flagged_indices": sorted(r["index"] for r in res if r["post_max_ratio"] > REL_TOL),
    }
    json.dump(summary, open(args.out_json, "w"), indent=2)
    print(f"wrote {out}\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
