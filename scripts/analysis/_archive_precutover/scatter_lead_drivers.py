#!/usr/bin/env python3
"""Scatter covert LEAD (= legal reaches TED-AI+tax  -  covert reaches TED-AI, years)
against two candidate drivers:
  (a) OOMs of effective compute between AC and TED-AI  (= ted_ai_e - ac_e),
  (b) software_progress_rate_at_reference_year.

ac_e is re-derived per rollout from its stored model_params (run_default). Lead uses
a symlog y-axis (linear within +-1 yr of the tie, log-compressed tails); runs where
the covert never reaches TED-AI are drawn on a floor strip ("inf behind"). Spearman
rho (with never-reach ranked worst) annotates each panel.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default  # noqa: E402
from scripts.pipelines.batch_rollout import _suppress_noise  # noqa: E402
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID = (
        "#1f4e9c", "#a23b3b", "#6b8e23", "#777", "#ccc")

_W = {}
def _init(input_path, tr):
    _W["data"] = load_time_series_data(input_path); _W["tr"] = tr
def _ac_ted(mp):
    try:
        with _suppress_noise():
            ref = run_default(Parameters(**mp), _W["data"], _W["tr"])
        return [ref.get("ac_e"), ref.get("ted_ai_e")]
    except Exception:
        return [None, None]


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500_capped.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "scatter_lead_drivers.png"))
    args = ap.parse_args()
    d = json.load(open(args.json))
    recs = d["records"]; tr = [float(d["config"]["time_range"][0]), float(d["config"]["time_range"][1])]
    input_path = d["config"]["input_data"]
    if not Path(input_path).exists():
        input_path = str((REPO_ROOT / "inputs" / "input_data.csv").resolve())

    mps = [r["model_params"] for r in recs]
    workers = max(1, min(args.workers, len(recs)))
    if workers == 1:
        _init(input_path, tr); ac_ted = [_ac_ted(mp) for mp in mps]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(input_path, tr)) as pool:
            ac_ted = pool.map(_ac_ted, mps, chunksize=4)

    oom = np.array([(t - a) if (a is not None and t is not None) else np.nan for a, t in ac_ted])
    swr = np.array([r["model_params"].get("software_progress_rate_at_reference_year", np.nan)
                    for r in recs], float)
    win = np.array([r["covert_wins"] for r in recs], bool)
    lead = np.array([r["lead_years"] if r["lead_years"] is not None else np.nan for r in recs], float)
    never = ~np.isfinite(lead)        # covert never reached TED-AI

    # lead proxy for Spearman: never-reach ranked below the worst finite lead
    finite_min = np.nanmin(lead) if np.isfinite(lead).any() else 0.0
    lead_rankable = np.where(never, finite_min - 1.0, lead)

    floor = finite_min - max(3.0, 0.15 * abs(finite_min))   # y for the never-reach strip
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    xmax_oom = float(np.nanpercentile(oom[oom < 1e6], 99) + 2)     # ignore numeric-blowup outliers
    xmax_swr = float(np.nanpercentile(swr, 99) + 0.5)
    panels = [("(a)  OOMs of effective compute between AC and TED-AI", oom, "AC→TED-AI distance (OOMs)", xmax_oom),
              ("(b)  software progress rate at reference year", swr, "software_progress_rate_at_reference_year", xmax_swr)]
    for ax, (title, x, xlab, xmax) in zip(axes, panels):
        rho = spearman(x, lead_rankable)        # rank-based: robust to the outlier
        x = np.where(x > xmax, np.nan, x)        # drop blow-up/outlier points from this panel
        n_out = int(np.sum(np.isfinite(oom if "(a)" in title else swr) & ~np.isfinite(x)))
        ax.set_xlim(left=0 if "(a)" in title else None, right=xmax)
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.9)
        # finite-lead points
        fin = np.isfinite(lead)
        ax.scatter(x[fin & ~win], lead[fin & ~win], s=16, color=AI2030_RED, alpha=0.5, label="covert loses")
        ax.scatter(x[fin & win], lead[fin & win], s=18, color=AI2030_OLIVE, alpha=0.75, label="covert wins")
        # never-reach on a floor strip (jittered for visibility)
        if never.any():
            xj = x[never]
            yj = floor * (1 + 0.04 * (np.argsort(np.argsort(xj)) / max(len(xj), 1) - 0.5))
            ax.scatter(xj, yj, s=12, marker="v", color="#c98", alpha=0.6,
                       label=f"covert never reaches TED-AI ({int(never.sum())})")
            ax.axhline(floor * 1.12, color=AI2030_GRID, ls=":", lw=1.0)
        ax.axhline(0.0, color=AI2030_NAVY, lw=1.3)
        # binned-median trend over x (finite leads)
        xx = x[fin]; yy = lead[fin]
        ok = np.isfinite(xx)
        if ok.sum() > 20:
            qs = np.quantile(xx[ok], np.linspace(0, 1, 9))
            cx, cy = [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                sel = ok & (xx >= lo) & (xx <= hi)
                if sel.sum() >= 3:
                    cx.append(np.median(xx[sel])); cy.append(np.median(yy[sel]))
            ax.plot(cx, cy, "-", color=AI2030_MUTED, lw=2.0, marker="o", ms=4, label="binned median")
        ax.set_xlabel(xlab); ax.set_ylabel("covert lead (years; symlog)  —  + = covert first")
        ax.set_title(f"{title}\nSpearman ρ(driver, lead) = {rho:+.2f}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("What drives the covert lead? (capped legal, 500-run)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)

    # ---- rank-rank version (pure Spearman view) ----
    def rankpct(v):
        r = np.full(len(v), np.nan); m = np.isfinite(v)
        if m.sum() > 1:
            r[m] = 100.0 * np.argsort(np.argsort(v[m])) / (m.sum() - 1)
        return r
    figr, axesr = plt.subplots(1, 2, figsize=(13, 6.0))
    rpanels = [("(a)  AC→TED-AI distance", np.where(oom < 1e6, oom, np.nan), "rank: AC→TED-AI OOMs"),
               ("(b)  software progress rate", swr, "rank: software_progress_rate_at_ref_year")]
    for ax, (title, x, xlab) in zip(axesr, rpanels):
        m = np.isfinite(x)
        rx = rankpct(np.where(m, x, np.nan))
        ry = rankpct(np.where(m, lead_rankable, np.nan))
        rho = float(np.corrcoef(rx[m], ry[m])[0, 1])
        ax.plot([0, 100], [0, 100], color=AI2030_GRID, lw=1.0, ls="--", zorder=0)
        ax.scatter(rx[m & ~win], ry[m & ~win], s=15, color=AI2030_RED, alpha=0.5, label="covert loses")
        ax.scatter(rx[m & win], ry[m & win], s=17, color=AI2030_OLIVE, alpha=0.75, label="covert wins")
        if never.any():
            ax.scatter(rx[m & never], ry[m & never], s=13, marker="v", color="#c98", alpha=0.7,
                       label=f"covert never reaches TED-AI ({int((m & never).sum())})")
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
        ax.set_xlabel(xlab + "  (percentile)"); ax.set_ylabel("rank: covert lead  (percentile)")
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    figr.suptitle("Rank–rank: covert lead vs its drivers (capped legal, 500-run)", fontsize=13, y=0.99)
    figr.tight_layout(rect=[0, 0, 1, 0.95])
    rout = out.with_name("scatter_lead_drivers_rank.png")
    figr.savefig(rout, dpi=140); plt.close(figr)
    print(f"wrote {rout}")
    print(f"n={len(recs)} never_reach={int(never.sum())}  "
          f"rho_oom={spearman(oom,lead_rankable):+.2f}  rho_swr={spearman(swr,lead_rankable):+.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
