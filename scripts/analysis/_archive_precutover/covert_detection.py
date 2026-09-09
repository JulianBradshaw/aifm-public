#!/usr/bin/env python3
"""Winnow the covert Monte-Carlo trajectories by a detection model.

Ports the monorepo's covert-project detection model
(`apps/covert-detection-model/src/lib/model.ts`, the
`detection-elicitation-full.xlsx` elicitation) and applies it to the covert MC
trajectories saved by `covert_mc_trajectories.py`.

Detection model (per the workbook fit)
--------------------------------------
A covert program of total power ``compute_mw`` is split across ``n`` near-identical
sites of per-site size ``s`` (MW), ``n = compute/s`` (with an ``n >= 1`` floor: a
single site then holds all the compute and its signature is computed at its actual
power). For a site size ``s`` with gamma-frailty ``(pi, thetaC, eta, k)`` the
probability the program is STILL undetected at horizon ``t`` (years it has run) is

    S(n,t) = [pi + (1-pi)(1 + thetaC*n*H(t))^(-1/thetaC)] * exp(-n*r*t),
    H(t) = (t/eta)^k

with an optional constant per-site operational-fuckup rate ``r`` (per year).
Frailty params are log-interpolated in log10(size) between the three elicited
sizes (10/100/1000 MW). Detection P = 1 - S.

Winnowing
---------
The covert is "at risk" from the pause (project initiated) until it reaches its
goal (TED-AI). For each MC trajectory we (a) weight it by S(t_risk) = P(still
undetected when it would reach TED-AI), the rigorous expected-value view, and
(b) Monte-Carlo sample a detection time and drop trajectories detected before
they reach the goal, for the thinned-trajectory plot.

    python scripts/analysis/covert_detection.py [--compute-mw 300] [--site-mw 300] [--r 0]
"""

from __future__ import annotations

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

from scripts.plotting_utils.style import (  # noqa: E402
    apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED,
    AI2030_GRID, AI2030_AMBER, AI2030_TEXT,
)
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402

apply_ai2030_style()

# Workbook Solver fits (apps/covert-detection-model/src/lib/defaults.ts SEED_FITS),
# the documented reference frailty per elicited site size.
SEED_FITS = {
    10: dict(pi=0.0, thetaC=5.661, eta=23.3332, k=0.7392),
    100: dict(pi=0.0, thetaC=4.4864, eta=7.5827, k=0.9706),
    1000: dict(pi=0.000889, thetaC=2.6099, eta=4.0326, k=0.9891),
}


def interp_frailty(size_mw: float) -> dict:
    """Log10(size)-linear interpolation of the four frailty params, clamped
    outside the elicited [10, 1000] MW range (mirrors model.ts interpFrailty)."""
    sizes = sorted(SEED_FITS)
    x = np.log10(max(size_mw, 1e-9))
    if x <= np.log10(sizes[0]):
        return dict(SEED_FITS[sizes[0]])
    if x >= np.log10(sizes[-1]):
        return dict(SEED_FITS[sizes[-1]])
    for i in range(len(sizes) - 1):
        if np.log10(sizes[i + 1]) >= x:
            a, b = sizes[i], sizes[i + 1]
            break
    xa, xb = np.log10(a), np.log10(b)
    f = (x - xa) / (xb - xa)
    pa, pb = SEED_FITS[a], SEED_FITS[b]
    return {key: pa[key] + f * (pb[key] - pa[key]) for key in pa}


def deploy(compute_mw: float, site_mw: float, cap_sites: bool = True):
    """Effective site count and signature size (n>=1 floor)."""
    n = compute_mw / site_mw
    eff = site_mw
    if cap_sites and n < 1:
        n, eff = 1.0, compute_mw
    return n, eff


def survival(p: dict, n: float, t, r: float = 0.0):
    """S(n,t): gamma-frailty capability term * optional Poisson fuckup factor."""
    t = np.asarray(t, dtype=float)
    H = np.power(np.maximum(t, 0.0) / p["eta"], p["k"])
    x = n * H
    if p["thetaC"] < 1e-9:
        inner = np.exp(-x)
    else:
        inner = np.power(1.0 + p["thetaC"] * x, -1.0 / p["thetaC"])
    cap = p["pi"] + (1.0 - p["pi"]) * inner
    fuckup = np.exp(-n * r * t) if r > 0 else 1.0
    return np.clip(cap * fuckup, 0.0, 1.0)


def quantile_bands(series, grid, qs=(10, 25, 50, 75, 90)):
    """Cross-trajectory percentiles on a common time grid. Computed over the FULL
    set passed in (we deliberately do NOT restrict to detection survivors — that
    would condition the band on having evaded detection up to each time)."""
    M = np.full((len(series), len(grid)), np.nan)
    for i, (t, y) in enumerate(series):
        t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        t, y = t[ok], y[ok]
        inside = (grid >= t.min()) & (grid <= t.max())
        M[i, inside] = np.interp(grid[inside], t, y)
    valid = np.isfinite(M).sum(axis=0) >= max(5, int(0.25 * len(series)))
    out = {}
    for q in qs:
        row = np.full(grid.shape, np.nan)
        if valid.any():
            row[valid] = np.nanpercentile(M[:, valid], q, axis=0)
        out[q] = row
    return out, valid


def median_parameter_trajectory(cfg: dict):
    """Recompute the single deterministic median-parameter covert run (the one in
    covert_median.png) and return (t, pct_to_asi) using the same pause convention
    as the MC dataset. Mirrors plot_covert_median.py / covert_mc_trajectories.py."""
    # Resolve config/input LOCALLY (the JSON may store remote-box paths from where
    # the MC was generated). Mirrors plot_covert_median.py's local loading.
    mc_path = Path(cfg.get("model_config", "config/sampling_config_thomas.yaml"))
    if not mc_path.is_absolute():
        mc_path = REPO_ROOT / mc_path
    mcfg = _load_config(str(mc_path)) or {}
    mp = _get_median_parameter_dict(mcfg.get("parameters", {}) or {})
    mp.setdefault("horizon_extrapolation_type", "decaying doubling time")
    cov = cfg["covert_knobs_fixed"]
    input_path = REPO_ROOT / mcfg.get("input_data", "inputs/input_data.csv")
    data = load_time_series_data(str(input_path))
    tr = cfg["time_range"]
    params = Parameters(**mp)
    ref = run_default(params, data, tr)
    tp = ref["ac_time"] - cfg.get("years_before_ac", 0.0)
    rec = simulate_covert(
        params, data, tp, alignment_tax_ooms=cov["alignment_tax_ooms"],
        target_year=tp + cov["target_lead_years"], max_hw_rate=cov["max_hw_rate"],
        total_h100e=cov["total_h100e"], frac_stolen=cov["frac_stolen"],
        dormancy_years=cov.get("dormancy_years", 0.0),
        setup_period_years=cov.get("setup_period_years", 0.0),
        setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0),
        t_end=tr[1], ref=ref, return_trajectories=True,
        legal_mode=cfg.get("legal_mode", "backsolve"),
        total_h100e_frac=cov.get("total_h100e_frac"))
    cov_t = np.array(rec["covert_t"], dtype=float)
    cov_e = np.array(rec["covert_e"], dtype=float)
    e_pause = float(np.interp(tp, ref["times"], ref["effective_compute"]))
    asi_e = ref["milestones"].get("ASI", {}).get("e")
    pct = 100.0 * (cov_e - e_pause) / (asi_e - e_pause)
    return cov_t, pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_trajectories.json"))
    ap.add_argument("--compute-mw", type=float, default=300.0,
                    help="Total covert power (MW). Default 300.")
    ap.add_argument("--site-mw", type=float, default=300.0,
                    help="Per-site power (MW). Default = whole project in one site.")
    ap.add_argument("--r", type=float, default=0.0,
                    help="Extra constant per-site operational-fuckup detection rate (per year).")
    ap.add_argument("--no-cap-sites", action="store_true",
                    help="Disable the n>=1 floor (allow fractional sites; workbook raw n).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Spaghetti-line opacity. Default: adaptive (~20/N).")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "covert_detection_winnowed.png"))
    args = ap.parse_args()

    n_sites, eff_size = deploy(args.compute_mw, args.site_mw, not args.no_cap_sites)
    p = interp_frailty(eff_size)

    d = json.load(open(args.json))
    recs = d["records"]
    n_tot = len(recs)

    # Per-trajectory race timing.
    pause = np.array([r["pause_year"] for r in recs], dtype=float)
    cttt = np.array([r["covert_ttt"] if r["covert_ttt"] is not None else np.nan for r in recs])
    wins = np.array([bool(r["covert_wins"]) for r in recs])
    reaches = np.isfinite(cttt)
    # Years the covert runs before reaching TED-AI (its goal). Trajectories that
    # never reach it are at risk for the full window but already lose the race.
    t_risk_goal = np.where(reaches, cttt - pause, np.nan)

    # (a) Survival weighting: P(undetected when it would reach TED-AI).
    s_at_goal = np.full(n_tot, np.nan)
    s_at_goal[reaches] = survival(p, n_sites, t_risk_goal[reaches], args.r)

    base_win = wins.mean()
    # Expected P(covert wins AND evades detection until it gets there).
    win_and_undetected = np.where(wins & reaches, s_at_goal, 0.0)
    win_det_adj = np.nansum(win_and_undetected) / n_tot

    # (b) MC thinning for the plot: draw a detection time per trajectory and drop
    # those detected before reaching their endpoint. Invert S(t) by root scan.
    rng = np.random.default_rng(args.seed)
    tgrid = np.linspace(0.0, 60.0, 6000)
    sgrid = survival(p, n_sites, tgrid, args.r)  # monotone decreasing in t
    floor = float(sgrid[-1])                     # asymptotic survival (never-detected mass)
    u = rng.random(n_tot)
    detected_ever = u > floor                    # else never detected within horizon
    # detection time = first t with S(t) <= u  (interp on the monotone curve)
    t_det = np.full(n_tot, np.inf)
    suse = u[detected_ever]
    # np.interp needs increasing x; S is decreasing so flip.
    t_det[detected_ever] = np.interp(suse, sgrid[::-1], tgrid[::-1])
    # caught before reaching TED-AI?
    caught_before_goal = reaches & (t_det < t_risk_goal)
    survivors = ~caught_before_goal
    win_after_thin = (wins & survivors).sum() / n_tot

    # ---- console summary -------------------------------------------------------
    print(f"detection model: compute={args.compute_mw:g} MW, site={eff_size:g} MW, "
          f"n_sites={n_sites:.2f}, r={args.r:g}/yr")
    print(f"  interp frailty @ {eff_size:g} MW: pi={p['pi']:.4f} thetaC={p['thetaC']:.3f} "
          f"eta={p['eta']:.3f} k={p['k']:.3f}")
    for yr in (1, 3, 5, 10):
        print(f"  P(detected by {yr:2d}y) = {1 - survival(p, n_sites, yr, args.r):.3f}")
    print(f"\n  trajectories: {n_tot}  | covert reaches TED-AI: {reaches.sum()}  "
          f"| covert wins race: {wins.sum()} ({base_win*100:.0f}%)")
    print(f"  median covert time-to-TED-AI (years at risk): {np.nanmedian(t_risk_goal):.1f}")
    print(f"  median P(undetected at win) among winners: "
          f"{np.nanmedian(s_at_goal[wins & reaches]):.3f}")
    print(f"\n  covert win rate  BEFORE detection : {base_win*100:.1f}%")
    print(f"  covert win rate  AFTER  detection : {win_det_adj*100:.1f}%  "
          f"(survival-weighted expectation)")
    print(f"  covert win rate  AFTER  detection : {win_after_thin*100:.1f}%  "
          f"(one MC thinning draw, seed={args.seed})")
    print(f"  => detection removes {(1 - win_det_adj/base_win)*100:.0f}% of covert wins")

    # ---- plot: detection curve + thinned trajectories --------------------------
    traj_key = "pct_to_asi"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.6),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    # left: detection probability vs years running
    axL.plot(tgrid, 1 - sgrid, color=AI2030_AMBER, lw=2.6)
    for yr in (3, 10):
        dpy = 1 - float(survival(p, n_sites, yr, args.r))
        axL.plot([yr], [dpy], "o", color=AI2030_AMBER, ms=6, mec="white", mew=1.0)
        axL.annotate(f"{dpy*100:.0f}% by {yr}y", (yr, dpy), textcoords="offset points",
                     xytext=(8, -4), fontsize=9, color=AI2030_MUTED)
    axL.set_xlim(0, 30); axL.set_ylim(0, 1)
    axL.set_xlabel("years the covert project has been running")
    axL.set_ylabel("probability detected")
    axL.set_title(f"Detection of a {args.compute_mw:g} MW covert project", fontsize=12,
                  color=AI2030_MUTED)
    axL.grid(False)

    # right: pct-to-ASI trajectories, survivors solid, detected faded
    ac_med = float(np.median([r["ac_time"] for r in recs]))
    ted_ts = [r.get("ted_ai_time") for r in recs if r.get("ted_ai_time") is not None]
    x_lo, x_hi = ac_med - 3, float(np.nanmedian(ted_ts)) + 20
    axR.axhline(0.0, color=AI2030_GRID, lw=0.9)
    axR.axhline(100.0, color=AI2030_GRID, lw=0.9)
    axR.text(x_lo + 0.1, 1.0, "pause level (0%)", fontsize=8, color=AI2030_MUTED)
    axR.text(x_lo + 0.1, 97.0, "ASI (100%)", fontsize=8, color=AI2030_MUTED)
    spag_a = args.alpha if args.alpha is not None else float(min(0.55, 20.0 / max(n_tot, 1)))
    for i, r in enumerate(recs):
        t = np.array(r["covert_t"], dtype=float)
        y = np.array(r[f"covert_{traj_key}"], dtype=float)
        if caught_before_goal[i]:
            # truncate the drawn trajectory at the detection time, faint
            tcut = pause[i] + t_det[i]
            m = t <= tcut
            axR.plot(t[m], y[m], color=AI2030_GRID, lw=0.6, alpha=spag_a * 0.7, zorder=1)
            if m.any():
                axR.plot([t[m][-1]], [y[m][-1]], "x", color=AI2030_RED, ms=4,
                         mew=1.0, alpha=min(1.0, spag_a * 2.5 + 0.15), zorder=2)
        else:
            col = AI2030_OLIVE if wins[i] else AI2030_RED
            axR.plot(t, y, color=col, lw=0.8, alpha=min(1.0, spag_a * 1.4), zorder=3)
    # Winnowed quantile bands over the full ensemble. We do NOT condition on
    # survival (which would bias toward survivors); instead we keep every
    # trajectory but FREEZE detected ones at the capability level where they were
    # caught (a detected project is exposed and stops progressing). The median
    # then reflects detection without conditioning.
    grid = np.linspace(x_lo, x_hi, 400)
    frozen = []
    for i, r in enumerate(recs):
        t = np.array(r["covert_t"], dtype=float)
        y = np.array(r[f"covert_{traj_key}"], dtype=float)
        tstop = pause[i] + t_det[i]  # +inf when never detected
        if np.isfinite(tstop) and t.size and tstop < t[-1]:
            m = t <= tstop
            ys = float(np.interp(tstop, t, y))
            frozen.append((np.concatenate([t[m], [tstop, x_hi]]),
                           np.concatenate([y[m], [ys, ys]])))
        else:
            frozen.append((t, y))

    b, valid = quantile_bands(frozen, grid)  # winnowed (caught frozen at detection)
    if valid.any():
        g = grid[valid]
        axR.fill_between(g, b[10][valid], b[90][valid], color=AI2030_RED, alpha=0.12, zorder=4, lw=0)
        axR.fill_between(g, b[25][valid], b[75][valid], color=AI2030_RED, alpha=0.20, zorder=5, lw=0)
        axR.plot(g, b[50][valid], color=AI2030_RED, lw=2.6, zorder=7)

    # Overlay the single deterministic median-PARAMETER run (the one in covert_median.png).
    mt, mp_pct = median_parameter_trajectory(d["config"])
    axR.plot(mt, mp_pct, color=AI2030_TEXT, lw=2.2, zorder=8)

    axR.plot([], [], color=AI2030_RED, lw=2.6, label="covert median — winnowed (band 25-75, 10-90)")
    axR.plot([], [], color=AI2030_TEXT, lw=2.2, label="median-parameter run (no detection)")
    axR.plot([], [], color=AI2030_OLIVE, lw=2, label=f"survives + wins ({(wins & survivors).sum()})")
    axR.plot([], [], color=AI2030_RED, lw=2, label=f"survives, loses ({(~wins & survivors).sum()})")
    axR.plot([], [], color=AI2030_GRID, lw=2, alpha=0.5,
             label=f"detected, removed ({caught_before_goal.sum()})  ✕ = caught")
    axR.set_xlim(x_lo, x_hi); axR.set_ylim(-5, 110)
    axR.set_xlabel("year"); axR.set_ylabel("percent of the way from pause to ASI")
    axR.set_title(f"Covert trajectories winnowed by detection  ·  "
                  f"win rate {base_win*100:.0f}% → {win_det_adj*100:.0f}%",
                  fontsize=12, color=AI2030_MUTED)
    axR.legend(loc="lower right", fontsize=9)
    axR.grid(False)

    fig.suptitle(
        f"Covert detection winnowing — {args.compute_mw:g} MW project"
        + (f", +{args.r*100:g}%/yr fuckup rate" if args.r > 0 else ""),
        fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
