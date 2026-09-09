#!/usr/bin/env python3
"""Re-plot the covert Monte-Carlo trajectories from a saved JSON (no rollouts).

The rollouts are deterministic and every series is already stored in the JSON
written by covert_mc_trajectories.py, so changing the *plot* (metric, quantile
bands, window) needs no re-simulation -- just reads the file and redraws.

    python scripts/analysis/plot_covert_mc_from_json.py --y pct_asi [--no-bands]
"""

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
    apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID,
)

apply_ai2030_style()

# metric -> (record trajectory key, record TED-AI scalar key, axis label, scale)
METRICS = {
    "pct_asi": ("pct_to_asi", "ted_ai_pct_to_asi",
                "percent of the way from pause to ASI", "linear"),
    "uplift": ("uplift", "ted_ai_uplift",
               "AI software R&D uplift  (× present-day)", "log"),
    "effective_compute": ("e", "ted_ai_e",
                          "Effective compute  —  AI capability", "linear"),
}


def quantile_bands(series, grid, n_total, qs=(10, 25, 50, 75, 90)):
    M = np.full((len(series), len(grid)), np.nan)
    for i, (t, y) in enumerate(series):
        t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        t, y = t[ok], y[ok]
        inside = (grid >= t.min()) & (grid <= t.max())
        M[i, inside] = np.interp(grid[inside], t, y)
    valid = np.isfinite(M).sum(axis=0) >= max(5, int(0.25 * n_total))
    out = {}
    for q in qs:
        row = np.full(grid.shape, np.nan)
        if valid.any():
            row[valid] = np.nanpercentile(M[:, valid], q, axis=0)
        out[q] = row
    return out, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_trajectories.json"))
    ap.add_argument("--y", choices=list(METRICS), default="pct_asi")
    ap.add_argument("--bands", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Spaghetti-line opacity. Default: adaptive (~25/N, lower for more lines).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(REPO_ROOT / "outputs" / f"covert_mc_{args.y}.png")

    d = json.load(open(args.json))
    recs = d["records"]
    cov_knobs = d["config"]["covert_knobs_fixed"]
    yba = d["config"].get("years_before_ac", 0.0)
    traj_key, ted_key, ylabel, yscale = METRICS[args.y]

    covs, legs, wins, ac_times, ted_ys, ted_ts = [], [], [], [], [], []
    for r in recs:
        covs.append((np.array(r["covert_t"]), np.array(r[f"covert_{traj_key}"])))
        legs.append((np.array(r["legal_t"]), np.array(r[f"legal_{traj_key}"])))
        wins.append(bool(r["covert_wins"]))
        ac_times.append(r["ac_time"])
        ted_ys.append(r.get(ted_key))
        ted_ts.append(r.get("ted_ai_time"))
    wins = np.array(wins, dtype=bool)
    n = len(recs)

    ted_y_med = float(np.nanmedian([v for v in ted_ys if v is not None]))
    ac_med = float(np.median(ac_times))
    x_lo = ac_med - 3
    x_hi = float(np.nanmedian([t for t in ted_ts if t is not None])) + 20

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_yscale(yscale)
    if args.y == "uplift":
        y_lo, y_hi, ted_txt = 1.0, ted_y_med * 1e3, ted_y_med * 1.3
    elif args.y == "pct_asi":
        y_lo, y_hi, ted_txt = -5.0, 110.0, ted_y_med + 2.0
        ax.axhline(0.0, color=AI2030_GRID, lw=0.9, zorder=0)
        ax.text(x_lo + 0.1, 1.0, "pause level (0%)", fontsize=8, color=AI2030_MUTED)
        ax.axhline(100.0, color=AI2030_GRID, lw=0.9, zorder=0)
        ax.text(x_lo + 0.1, 97.0, "ASI (100%)", fontsize=8, color=AI2030_MUTED)
    else:
        y_lo, y_hi, ted_txt = ted_y_med - 7.0, ted_y_med + 4.0, ted_y_med + 0.1

    ax.axhline(ted_y_med, color=AI2030_GRID, ls=":", lw=1.0)
    ax.text(x_lo + 0.1, ted_txt, "TED-AI (median)", fontsize=8, color=AI2030_MUTED)

    base = 0.10 if args.bands else 0.18
    spag_a = args.alpha if args.alpha is not None else float(min(base, 25.0 / max(n, 1)))
    cov_a = min(1.0, spag_a * 1.4)  # covert a touch more visible than legal
    for (lt, lu) in legs:
        ax.plot(lt, lu, color=AI2030_NAVY, lw=0.6, alpha=spag_a, zorder=1)
    for (ct, cu), w in zip(covs, wins):
        ax.plot(ct, cu, color=(AI2030_OLIVE if w else AI2030_RED), lw=0.7, alpha=cov_a, zorder=2)

    if args.bands:
        grid = np.linspace(x_lo, x_hi, 400)
        for series, color in [(legs, AI2030_NAVY), (covs, AI2030_RED)]:
            b, valid = quantile_bands(series, grid, n)
            if not valid.any():
                continue
            g = grid[valid]
            ax.fill_between(g, b[10][valid], b[90][valid], color=color, alpha=0.12, zorder=3, lw=0)
            ax.fill_between(g, b[25][valid], b[75][valid], color=color, alpha=0.20, zorder=4, lw=0)
            ax.plot(g, b[50][valid], color=color, lw=2.6, zorder=5)

    ax.plot([], [], color=AI2030_NAVY, lw=2.6, label="legal: median (band 25-75, 10-90)")
    ax.plot([], [], color=AI2030_RED, lw=2.6, label="covert: median (band 25-75, 10-90)")
    ax.plot([], [], color=AI2030_RED, lw=2, label=f"covert loses ({int((~wins).sum())})")
    ax.plot([], [], color=AI2030_OLIVE, lw=2, label=f"covert wins ({int(wins.sum())})")

    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("year"); ax.set_ylabel(ylabel)
    h100e = cov_knobs["total_h100e"]; _frac = cov_knobs.get("total_h100e_frac")
    size_str = (f"{_frac*100:.2f}% of frontier R&D compute at pause"
                if _frac is not None
                else (f"{h100e/1e6:g}M" if h100e >= 1e6 else f"{h100e/1e3:g}k") + " H100e datacenter")
    fig.suptitle(
        f"Covert project — Monte-Carlo trajectories  ·  covert wins {wins.mean()*100:.0f}%",
        fontsize=18, y=0.965)
    ax.set_title(
        f"{n} runs, model params sampled · covert knobs fixed: {size_str}, "
        f"steals {cov_knobs['frac_stolen']*100:.0f}% of consortium progress,\n"
        f"{cov_knobs.get('dormancy_years',0):g}-yr construction time, "
        f"{cov_knobs['alignment_tax_ooms']:g}-OOM alignment tax, pause {yba:g}y before AC",
        fontsize=10, color=AI2030_MUTED)
    ax.legend(loc="lower right", fontsize=8.5); ax.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"replotted {n} trajectories ({args.y}, bands={args.bands}) -> {out}")


if __name__ == "__main__":
    main()
