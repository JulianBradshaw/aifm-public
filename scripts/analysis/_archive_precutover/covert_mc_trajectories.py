#!/usr/bin/env python3
"""Monte Carlo of covert trajectories (N-distribution form): sample MODEL parameters
from the sampling config (same machinery as the pause-reduction MC), keep the covert
knobs FIXED at the covert-config medians, and plot N random covert capability
trajectories. The pause is held a fixed offset before each sample's AC.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as cfg  # noqa: E402
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert  # noqa: E402
from scripts.pipelines.batch_rollout import (  # noqa: E402
    _load_config, _sample_parameter_dict, _get_median_parameter_dict, _suppress_noise,
    _calibrate_gaussian_copula_latent_corr)
from scripts.plotting_utils.style import (  # noqa: E402
    apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID,
)

apply_ai2030_style()

TRAJ_KEY = {"uplift": "uplift", "pct_asi": "pct_to_asi", "effective_compute": "e"}
TED_KEY = {"uplift": "ted_ai_uplift", "pct_asi": "ted_ai_pct_to_asi", "effective_compute": "ted_ai_e"}


# --------------------------------------------------------------------------- #
# Parallel worker: one whole sample. State identical across samples (time
# series + config) is loaded once per process in the initializer. Results are
# deterministic per seed regardless of worker count.
# --------------------------------------------------------------------------- #
_W = {}


def _init_worker(input_path, all_dists, model_names, corr, cov, tr, years_before_ac,
                 hor_type, taste_type, legal_mode):
    _W.update(dict(
        data=load_time_series_data(input_path), all_dists=all_dists,
        model_names=set(model_names), corr=corr, cov=cov, tr=tr,
        yba=years_before_ac, hor=hor_type, taste=taste_type, legal_mode=legal_mode))


def _e_to_uplift(ref, E):
    ec = np.asarray(ref["effective_compute"], dtype=float)
    up = np.asarray(ref["uplift"], dtype=float)
    order = np.argsort(ec)
    return np.interp(np.asarray(E, dtype=float), ec[order], up[order])


def _run_sample(task):
    """Run one sample; return a full record dict (all metric series) or None."""
    idx, ss = task
    data, cov, tr = _W["data"], _W["cov"], _W["tr"]
    rng = np.random.default_rng(ss)
    with _suppress_noise():
        try:
            sm = _sample_parameter_dict(_W["all_dists"], rng, _W["corr"])
            mp = {k: v for k, v in sm.items() if k in _W["model_names"]}
            mp.setdefault("horizon_extrapolation_type", _W["hor"])
            mp.setdefault("taste_schedule_type", _W["taste"])
            params = Parameters(**mp)
            ref = run_default(params, data, tr)
            if ref["ac_time"] is None or ref["ted_ai_e"] is None:
                return None
            tp = ref["ac_time"] - _W["yba"]
            rec = simulate_covert(
                params, data, tp,
                alignment_tax_ooms=cov["alignment_tax_ooms"],
                target_year=tp + cov["target_lead_years"], max_hw_rate=cov["max_hw_rate"],
                total_h100e=cov["total_h100e"], frac_stolen=cov["frac_stolen"],
                dormancy_years=cov.get("dormancy_years", 0.0),
                setup_period_years=cov.get("setup_period_years", 0.0),
                setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0),
                t_end=tr[1], ref=ref, return_trajectories=True,
                legal_mode=_W.get("legal_mode", "backsolve"),
                total_h100e_frac=cov.get("total_h100e_frac"))
        except Exception:
            return None

    cov_e = np.asarray(rec["covert_e"], dtype=float)
    leg_e = np.asarray(rec["legal_e"], dtype=float)
    cov_u = _e_to_uplift(ref, cov_e)
    leg_u = _e_to_uplift(ref, leg_e)
    ted_u = float(_e_to_uplift(ref, ref["ted_ai_e"]))

    e_pause = float(np.interp(tp, ref["times"], ref["effective_compute"]))
    asi_e = ref["milestones"].get("ASI", {}).get("e")
    asi_e = float(asi_e) if asi_e is not None else None
    span = (asi_e - e_pause) if asi_e is not None else None

    def _pct(E):
        if span is None or span <= 0:
            return np.full(np.shape(E), np.nan)
        return 100.0 * (np.asarray(E, dtype=float) - e_pause) / span
    cov_pct = _pct(cov_e); leg_pct = _pct(leg_e)
    ted_pct = (100.0 * (ref["ted_ai_e"] - e_pause) / span) if span and span > 0 else np.nan

    return {
        "index": idx,
        "seed_entropy": list(ss.entropy) if hasattr(ss.entropy, "__iter__") else ss.entropy,
        "model_params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in mp.items()},
        "ac_time": float(ref["ac_time"]),
        "ted_ai_time": (float(ref["ted_ai_time"]) if ref["ted_ai_time"] is not None else None),
        "ted_ai_e": float(ref["ted_ai_e"]),
        "ac_to_ted_gap": (float(ref["ac_to_ted_gap"]) if ref.get("ac_to_ted_gap") is not None else None),
        # full milestone map {label: {e (rebased OOMs), time (yr)}} so downstream slices
        # (e.g. AC->ASI calendar gap) need no run_default re-derivation.
        "milestones": {k: {"e": float(m["e"]), "time": float(m["time"])}
                       for k, m in (ref.get("milestones") or {}).items()},
        "pause_year": float(tp),
        "covert_wins": bool(rec["covert_wins"]),
        "covert_ttt": (float(rec["covert_ttt"]) if rec["covert_ttt"] is not None else None),
        "legal_ttt": (float(rec["legal_ttt"]) if rec["legal_ttt"] is not None else None),
        "lead_years": (float(rec["lead_years"]) if rec["lead_years"] is not None else None),
        "sw_rate_solved": (float(rec["sw_rate_solved"]) if rec.get("sw_rate_solved") is not None else None),
        "legal_mode": rec.get("legal_mode"),
        "legal_theta_solved": (float(rec["legal_theta_solved"]) if rec.get("legal_theta_solved") is not None else None),
        "total_h100e_used": (float(rec["total_h100e_used"]) if rec.get("total_h100e_used") is not None else None),
        "total_h100e_frac": rec.get("total_h100e_frac"),
        "legal_target_e": float(rec["legal_target_e"]),
        "ted_ai_uplift": ted_u,
        "e_pause": e_pause,
        "asi_e": asi_e,
        "ted_ai_pct_to_asi": (float(ted_pct) if np.isfinite(ted_pct) else None),
        "covert_t": [float(x) for x in rec["covert_t"]],
        "covert_e": [float(x) for x in cov_e],
        "covert_uplift": [float(x) for x in cov_u],
        "covert_pct_to_asi": [float(x) for x in cov_pct],
        "legal_t": [float(x) for x in rec["legal_t"]],
        "legal_e": [float(x) for x in leg_e],
        "legal_uplift": [float(x) for x in leg_u],
        "legal_pct_to_asi": [float(x) for x in leg_pct],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="config/sampling_config_thomas.yaml")
    ap.add_argument("--covert-config", default="config/sampling_config_covert.yaml")
    ap.add_argument("-n", "--num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallel worker processes (default: cores-1).")
    ap.add_argument("--years-before-ac", type=float, default=0.8,
                    help="Pause this many years before each sample's AC (deal timing).")
    ap.add_argument("--legal-mode", choices=["backsolve", "capped"], default="backsolve",
                    help="Legal project model: 'backsolve' (exogenous SW rate, default) or "
                         "'capped' (effort bounded by the AIFM physics ceiling).")
    ap.add_argument("--y", choices=["uplift", "pct_asi", "effective_compute"], default="pct_asi",
                    help="Y-axis metric. pct_asi = %% of the way (in effective-compute OOMs) "
                         "from the pause level to ASI.")
    ap.add_argument("--bands", action=argparse.BooleanOptionalAction, default=True,
                    help="Overlay median + IQR (25-75) + 10-90 quantile bands.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-json", default=str(REPO_ROOT / "outputs" / "covert_mc_trajectories.json"))
    args = ap.parse_args()
    if args.out is None:
        args.out = str(REPO_ROOT / "outputs" / f"covert_mc_{args.y}.png")

    mcfg = _load_config(args.model_config) or {}
    pdists = mcfg.get("parameters", {}) or {}
    tsd = mcfg.get("time_series_parameters", {}) or {}
    corr = mcfg.get("correlation_matrix")
    all_dists = {**pdists, **tsd}
    model_names = set(pdists) | set(tsd)
    tr = [float((mcfg.get("time_range") or [2017, 2100])[0]), 2100.0]
    input_path = mcfg.get("input_data", "inputs/input_data.csv")
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())

    # FIXED covert knobs (the user's covert config medians)
    cov = _get_median_parameter_dict(_load_config(args.covert_config).get("covert_parameters", {}))

    master_ss = SeedSequence(args.seed)
    # Calibrate the Gaussian-copula latent correlation once (mirrors batch_rollout).
    if corr is not None:
        cpar = corr.get("parameters", []); tcorr = corr.get("correlation_matrix")
        if cpar and tcorr is not None:
            interp = str(corr.get("correlation_type", corr.get("interpretation", "spearman")))
            csamp = int(corr.get("calibration_samples", 20000))
            try:
                latent = _calibrate_gaussian_copula_latent_corr(
                    all_dists, cpar, np.array(tcorr, dtype=float),
                    np.random.default_rng(master_ss.spawn(1)[0]),
                    interpretation=interp, calibration_samples=csamp)
                corr["latent_correlation_matrix"] = latent.tolist()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: correlation calibration failed ({e}); dropping correlations.")
                corr = None

    def quantile_bands(series, grid, qs=(10, 25, 50, 75, 90)):
        # interpolate each (t, y) trajectory onto a common grid (NaN outside its
        # own time span), then take cross-sample percentiles where enough remain.
        M = np.full((len(series), len(grid)), np.nan)
        for i, (t, y) in enumerate(series):
            t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
            ok = np.isfinite(t) & np.isfinite(y)
            if ok.sum() < 2:
                continue
            t, y = t[ok], y[ok]
            inside = (grid >= t.min()) & (grid <= t.max())
            M[i, inside] = np.interp(grid[inside], t, y)
        cnt = np.isfinite(M).sum(axis=0)
        valid = cnt >= max(5, int(0.25 * len(series)))
        out = {}
        for q in qs:
            row = np.full(grid.shape, np.nan)
            if valid.any():
                row[valid] = np.nanpercentile(M[:, valid], q, axis=0)
            out[q] = row
        return out, valid

    seeds = master_ss.spawn(args.num)
    tasks = list(enumerate(seeds))
    workers = max(1, min(args.workers, args.num))
    t0 = time.perf_counter()
    print(f"running {args.num} samples on {workers} worker(s) ...", flush=True)

    init_args = (input_path, all_dists, model_names, corr, cov, tr,
                 args.years_before_ac, cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE,
                 cfg.DEFAULT_TASTE_SCHEDULE_TYPE, args.legal_mode)
    records = []

    def _progress(done):
        if done % 25 == 0 or done == len(tasks):
            print(f"  {done}/{len(tasks)} done, {len(records)} ok "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    if workers == 1:
        _init_worker(*init_args)
        for done, task in enumerate(tasks, 1):
            r = _run_sample(task)
            if r is not None:
                records.append(r)
            _progress(done)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            for done, r in enumerate(pool.imap_unordered(_run_sample, tasks, chunksize=1), 1):
                if r is not None:
                    records.append(r)
                _progress(done)
    records.sort(key=lambda r: r["index"])
    n_ok = len(records)

    # rebuild the plotting series from records, selected by --y
    tk, tdk = TRAJ_KEY[args.y], TED_KEY[args.y]
    covs = [(np.array(r["covert_t"]), np.array(r[f"covert_{tk}"])) for r in records]
    legs = [(np.array(r["legal_t"]), np.array(r[f"legal_{tk}"])) for r in records]
    wins = np.array([bool(r["covert_wins"]) for r in records], dtype=bool)
    ac_times = [r["ac_time"] for r in records]
    ted_ys = [(r.get(tdk) if r.get(tdk) is not None else np.nan) for r in records]
    ted_ts = [r.get("ted_ai_time") for r in records]

    ted_y_med = float(np.nanmedian(ted_ys)) if ted_ys else float("nan")
    ac_med = float(np.median(ac_times)) if ac_times else float("nan")
    x_lo = ac_med - 3
    x_hi = float(np.nanmedian([t for t in ted_ts if t is not None])) + 20

    fig, ax = plt.subplots(figsize=(11, 7))
    if args.y == "uplift":
        ax.set_yscale("log")
        ylabel = "AI software R&D uplift  (× present-day, ref present day)"
        y_lo, y_hi = 1.0, ted_y_med * 1e3
        ted_txt_y = ted_y_med * 1.3
    elif args.y == "pct_asi":
        ylabel = "percent of the way from pause to ASI"
        y_lo, y_hi = -5.0, 110.0
        ted_txt_y = ted_y_med + 2.0
    else:  # effective_compute
        ylabel = "Effective compute  —  AI capability"
        y_lo, y_hi = ted_y_med - 7.0, ted_y_med + 4.0
        ted_txt_y = ted_y_med + 0.1

    # 0% (pause) and 100% (ASI) guides for the pct view
    if args.y == "pct_asi":
        ax.axhline(0.0, color=AI2030_GRID, lw=0.9, ls="-", zorder=0)
        ax.text(x_lo + 0.1, 1.0, "pause level (0%)", fontsize=8, color=AI2030_MUTED)
        ax.axhline(100.0, color=AI2030_GRID, lw=0.9, ls="-", zorder=0)
        ax.text(x_lo + 0.1, 97.0, "ASI (100%)", fontsize=8, color=AI2030_MUTED)
    # median TED-AI reference
    ax.axhline(ted_y_med, color=AI2030_GRID, ls=":", lw=1.0)
    ax.text(x_lo + 0.1, ted_txt_y, "TED-AI (median)", fontsize=8, color=AI2030_MUTED)
    spag_a = 0.10 if args.bands else 0.18
    for (lt, lu) in legs:
        ax.plot(lt, lu, color=AI2030_NAVY, lw=0.6, alpha=spag_a, zorder=1)
    for (ct, cu), w in zip(covs, wins):
        ax.plot(ct, cu, color=(AI2030_OLIVE if w else AI2030_RED), lw=0.7,
                alpha=spag_a + 0.06, zorder=2)

    # quantile bands (median + IQR + 10-90) on a common grid
    if args.bands:
        grid = np.linspace(x_lo, x_hi, 400)
        for series, color, name in [(legs, AI2030_NAVY, "legal"), (covs, AI2030_RED, "covert")]:
            b, valid = quantile_bands(series, grid)
            if not valid.any():
                continue
            g = grid[valid]
            ax.fill_between(g, b[10][valid], b[90][valid], color=color, alpha=0.12, zorder=3, lw=0)
            ax.fill_between(g, b[25][valid], b[75][valid], color=color, alpha=0.20, zorder=4, lw=0)
            ax.plot(g, b[50][valid], color=color, lw=2.6, zorder=5)

    # legend proxies
    ax.plot([], [], color=AI2030_NAVY, lw=2.6, label="legal: median (band: 25-75, 10-90)")
    ax.plot([], [], color=AI2030_RED, lw=2.6, label="covert: median (band: 25-75, 10-90)")
    ax.plot([], [], color=AI2030_RED, lw=2, label=f"covert loses ({int((~wins).sum())})")
    ax.plot([], [], color=AI2030_OLIVE, lw=2, label=f"covert wins ({int(wins.sum())})")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("year"); ax.set_ylabel(ylabel)
    h100e = cov["total_h100e"]; _frac = cov.get("total_h100e_frac")
    size_str = (f"{_frac*100:.2f}% of frontier R&D compute at pause"
                if _frac is not None
                else (f"{h100e/1e6:g}M" if h100e >= 1e6 else f"{h100e/1e3:g}k") + " H100e datacenter")
    fig.suptitle(
        f"Covert project — Monte-Carlo trajectories  ·  covert wins {wins.mean()*100:.0f}%",
        fontsize=18, y=0.965)
    ax.set_title(
        f"{n_ok} runs, model params sampled · covert knobs fixed: {size_str}, "
        f"steals {cov['frac_stolen']*100:.0f}% of consortium progress,\n"
        f"{cov.get('dormancy_years',0):g}-yr construction time, "
        f"{cov['alignment_tax_ooms']:g}-OOM alignment tax, pause {args.years_before_ac:g}y before AC",
        fontsize=10, color=AI2030_MUTED)
    ax.legend(loc="lower right", fontsize=9); ax.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)

    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "model_config": args.model_config, "covert_config": args.covert_config,
            "num": args.num, "seed": args.seed, "years_before_ac": args.years_before_ac,
            "legal_mode": args.legal_mode,
            "workers": workers, "time_range": tr, "input_data": input_path,
            "covert_knobs_fixed": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                   for k, v in cov.items()},
        },
        "summary": {
            "n_ok": n_ok, "n_requested": args.num,
            "covert_win_frac": float(wins.mean()) if len(wins) else None,
            "y_metric": args.y, "ted_ai_y_median": ted_y_med, "ac_time_median": ac_med,
        },
        "records": records,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    win_pct = wins.mean() * 100 if len(wins) else float("nan")
    print(f"ran {n_ok}/{args.num} in {time.perf_counter()-t0:.0f}s on {workers} workers; "
          f"covert wins {win_pct:.0f}%; wrote {out}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
