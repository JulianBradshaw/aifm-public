#!/usr/bin/env python3
"""Figure registry for the covert pipeline. Every figure is a PURE FUNCTION of the
dataset dict (records carry trajectories + milestones + uplift/pct + ceiling ratios),
so nothing here re-runs the model. Add an analysis = append one entry to FIGURES.

Each builder: fn(ds: dict, outdir: Path, **opts) -> str (the written filename).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as _pe  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID,
        AI2030_AMBER, AI2030_TEXT, MILESTONE_COLORS)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID, AI2030_AMBER, AI2030_TEXT = (
        "#1f4e9c", "#a23b3b", "#6b8e23", "#777", "#ccc", "#c8862b", "#222")
    MILESTONE_COLORS = {"AC": AI2030_OLIVE, "SAR": "#4d77a8", "TED-AI": AI2030_AMBER,
                        "STRAT-AI": "#412941", "ASI": "#a04444"}

from plan_a_covert_proj import detection  # noqa: E402  (pure detection model; never re-runs physics)
from plan_a_covert_proj.pipeline.covert_within_vs_scaling_duration import (  # noqa: E402
    render as covert_within_vs_scaling_duration)

TRAJ = {"pct_asi": "pct_to_asi", "uplift": "uplift", "effective_compute": "e"}
TED = {"pct_asi": "ted_ai_pct_to_asi", "uplift": "ted_ai_uplift", "effective_compute": "ted_ai_e"}


# --------------------------------------------------------------------------- #
# OPTIONAL importance weights (post-hoc AC-time conditioning).
#
# By default NOTHING here is active: a normal run never attaches weights, so every
# helper below collapses to plain numpy. `ac_reweight.py --emit` is the ONLY caller
# that sets ds["_weights"] (one finite weight per record, aligned to ds["records"]),
# producing a SIBLING run dir whose distributional figures are reweighted toward a
# target AC year. The "off" path deliberately passes weights=None everywhere — the
# numpy/matplotlib default — so default-run figures are byte-identical to before this
# feature existed (verified by md5, not assumed).
#
# Reweighting only touches RECORD-DISTRIBUTIONAL content (histogram heights, empirical
# CDFs, weighted medians/percentiles/means/fractions across rollouts). It does NOT and
# CANNOT touch: (a) median-scenario figures (oom_*, median_2panel, implied_compute_*,
# legal_implied_compute) which are a single median-parameter trajectory, not a record
# distribution; (b) per-rollout scatter clouds / Spearman driver diagnostics
# (lead_drivers, characterize_wins panel-a, success_drivers, shapley) where a weight is
# not a distributional summary. Those stay structurally unweighted by design.
# --------------------------------------------------------------------------- #
def _wts(ds):
    """Per-record importance weights aligned to ds['records'], or None if unweighted."""
    w = ds.get("_weights")
    return None if w is None else np.asarray(w, float)


def _wsub(w, mask):
    """Weights restricted to a boolean/index mask (for figures that filter records),
    or None when unweighted so the masked path also collapses to numpy defaults."""
    return None if w is None else np.asarray(w, float)[mask]


def _wmean(x, w):
    """Weighted mean of x; plain np.mean when w is None. Used for fractions (P(win) etc.)."""
    x = np.asarray(x, float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, float)
    s = float(np.sum(w))
    return float(np.sum(w * x) / s) if s > 0 else float("nan")


def _wquantile(x, w, q):
    """Weighted quantile(s); q scalar or array in [0,1]. Plain np.quantile when w is None.
    Hazen-style weighted plotting positions, identical convention to ac_reweight._wquantile."""
    x = np.asarray(x, float)
    if w is None:
        return np.quantile(x, q)
    w = np.asarray(w, float)
    i = np.argsort(x, kind="mergesort")
    x, w = x[i], w[i]
    s = float(np.sum(w))
    if s <= 0:
        return np.quantile(x, q)
    cw = (np.cumsum(w) - 0.5 * w) / s
    return np.interp(q, cw, x)


def _wpct(x, w, p):
    """Weighted percentile(s); p scalar or array in [0,100]. Plain np.percentile when w is None."""
    if w is None:
        return np.percentile(np.asarray(x, float), p)
    return _wquantile(x, w, np.asarray(p, float) / 100.0)


def _wcumcount(x, w, grid):
    """(Weighted) cumulative count of x <= each grid point, for empirical CDFs. When w is
    None this returns float counts numerically identical to np.searchsorted(sort(x), grid,
    side='right'), so CDF figures stay byte-identical. Divide by the chosen denominator
    (n records, or total weight) in the caller to get the CDF height."""
    x = np.asarray(x, float)
    grid = np.asarray(grid, float)
    if x.size == 0:
        return np.zeros_like(grid, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ww = np.ones(xs.size) if w is None else np.asarray(w, float)[order]
    csum = np.cumsum(ww)
    idx = np.searchsorted(xs, grid, side="right")
    return np.where(idx > 0, csum[np.clip(idx - 1, 0, max(xs.size - 1, 0))], 0.0)


def _wtot(R, w):
    """CDF denominator: number of records (unweighted) or total weight (weighted)."""
    return float(len(R)) if w is None else float(np.sum(np.asarray(w, float)))


def _rw_note(ds):
    """Short banner describing the AC-time reweighting + its effective-sample-size cost,
    so a reweighted figure never *looks* better-resolved than its ESS. '' when unweighted."""
    m = ds.get("_reweight_meta")
    if not m:
        return ""
    return (f"AC≈{m['target']:g} reweighted · ESS {m['ess']:.0f}/{m['n']} "
            f"({100*m['ess']/m['n']:.0f}% kept, σ={m['sigma']:g} yr)")


def _size_str(cov):
    if cov.get("compute_mode") == "bottom_up_estimate":
        return "bottom-up compute estimate"
    f = cov.get("total_h100e_frac"); h = cov.get("total_h100e", 0)
    if f is not None:
        return f"{f*100:.2f}% of frontier R&D compute at pause"
    return (f"{h/1e6:g}M" if h >= 1e6 else f"{h/1e3:g}k") + " H100e datacenter"


def _spaghetti_caption(ds):
    c = ds["config"]; k = c["covert_knobs_fixed"]
    return (f"{ds['summary']['n_ok']} runs · {c.get('legal_mode')} legal · covert = {_size_str(k)}, "
            f"steals {k['frac_stolen']*100:.0f}%, {k.get('dormancy_years',0):g}y dormant + "
            f"{k.get('construction_years',0):g}y build, "
            f"{k.get('alignment_tax_progress_years', 0):g}-progress-yr tax, pause {c.get('years_before_ac',0):g}y pre-AC")


def _cap_rw(ds):
    """Shared spaghetti caption, plus the reweight/ESS banner when (and only when) the figure
    it's used on is actually reweighted. Use this in WEIGHTED figures; leave _spaghetti_caption
    bare in figures whose content is NOT reweighted, so the banner never over-claims."""
    note = _rw_note(ds)
    return _spaghetti_caption(ds) + ("  ·  " + note if note else "")


def _freeze_at(t, v, cut):
    """Hold a (t, v) trajectory flat at its value once calendar time exceeds ``cut``
    (the detection time). Returns the trajectory unchanged when ``cut`` is None/inf or
    falls after the trajectory's last sample."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    if cut is None or not np.isfinite(cut) or t.size == 0 or cut >= t[-1]:
        return t, v
    vc = float(np.interp(cut, t, v))
    keep = t < cut
    t2 = np.concatenate([t[keep], [cut, t[-1]]])
    v2 = np.concatenate([v[keep], [vc, vc]])
    return t2, v2


def _quantile_bands(series, grid, n_total, qs=(10, 25, 50, 75, 90), weights=None):
    M = np.full((len(series), len(grid)), np.nan)
    for i, (t, y) in enumerate(series):
        t = np.asarray(t, float); y = np.asarray(y, float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        t, y = t[ok], y[ok]
        ins = (grid >= t.min()) & (grid <= t.max())
        M[i, ins] = np.interp(grid[ins], t, y)
    # Resolution gate stays a RAW rollout count (how many trajectories cover the column),
    # independent of weighting — a column thinly covered is still thinly resolved.
    valid = np.isfinite(M).sum(0) >= max(5, int(0.25 * n_total))
    out = {}
    if weights is None:
        for q in qs:
            row = np.full(grid.shape, np.nan)
            if valid.any():
                row[valid] = np.nanpercentile(M[:, valid], q, axis=0)
            out[q] = row
        return out, valid
    # weighted per-column quantiles (importance reweighting): each column's finite rows are
    # quantiled with their per-rollout weights.
    wv = np.asarray(weights, float)
    for q in qs:
        out[q] = np.full(grid.shape, np.nan)
    qfrac = np.asarray(qs, float) / 100.0
    for c in np.where(valid)[0]:
        col = M[:, c]; fin = np.isfinite(col)
        qq = _wquantile(col[fin], wv[fin], qfrac)
        for k, q in enumerate(qs):
            out[q][c] = qq[k]
    return out, valid


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def spaghetti(ds, outdir, y="pct_asi", fname=None, subtitle=None, detect=False, simple=False):
    """``simple=True`` renders the identical graph in the median_uplift presentation style:
    one declarative in-axes title, no config subtitle, plain-number ticks, bigger fonts,
    minimal legend. The plotted data (thin lines, bands, medians, detection X's) is untouched."""
    R = ds["records"]; tk, tdk = TRAJ[y], TED[y]
    wts = _wts(ds)
    win = np.array([r["covert_wins"] for r in R], bool)
    # Covert trajectories. With detect=True, each covert is frozen at its detection time
    # (pause_year + t_det); undetected coverts (t_det None) run on unchanged. det_pts holds
    # the (time, value) where each detected covert is caught, marked with an X. The thin lines
    # are drawn only up to detection (covs_line, truncated), but the QUANTILE BANDS use the
    # frozen full-length trajectories (covs) so the post-detection plateau still counts.
    covs = []; covs_line = []; det_pts = []
    for r in R:
        ct = np.array(r["covert_t"]); cu = np.array(r[f"covert_{tk}"])
        if detect:
            td = r.get("t_det")
            cut = (float(r["pause_year"]) + float(td)) if td is not None else None
            if cut is not None and np.isfinite(cut) and ct.size and cut < ct[-1]:
                vc = float(np.interp(cut, ct, cu))
                det_pts.append((cut, vc))
                keep = ct < cut
                covs_line.append((np.concatenate([ct[keep], [cut]]),
                                  np.concatenate([cu[keep], [vc]])))   # line stops at the X
                covs.append(_freeze_at(ct, cu, cut))                   # band keeps the plateau
                continue
        covs.append((ct, cu)); covs_line.append((ct, cu))
    # Legal = the USED (deployed-capability) glidepath, not the discovered (TED-AI + tax)
    # track. Fall back to discovered for datasets built before used trajectories were stored.
    legs = [(np.array(r.get("legal_used_t", r["legal_t"])),
             np.array(r.get(f"legal_used_{tk}", r[f"legal_{tk}"]))) for r in R]
    ac = np.array([r["ac_time"] for r in R], float)
    tedt = np.array([r["ted_ai_time"] for r in R if r["ted_ai_time"] is not None], float)
    ted_y = [r.get(tdk) for r in R]
    ted_y_med = float(np.nanmedian([v for v in ted_y if v is not None]))
    x_lo, x_hi = float(np.median(ac)) - 3, float(np.median(tedt)) + 22
    n = len(R)
    fig, ax = plt.subplots(figsize=(11, 7))
    if y == "uplift":
        ax.set_yscale("log"); ylab = "AI software R&D uplift (× present-day)"; y_lo, y_hi = 1.0, ted_y_med * 1e3
    elif y == "pct_asi":
        start_lab = "pre-deal AI" if simple else "pause"
        ylab = f"percent of the way from {start_lab} to ASI"; y_lo, y_hi = -5, 110
        ax.axhline(0, color=AI2030_GRID, lw=.9); ax.axhline(100, color=AI2030_GRID, lw=.9)
        ax.text(x_lo + .1, 1, f"{start_lab} (0%)", fontsize=(12 if simple else 8), color=AI2030_MUTED)
        ax.text(x_lo + .1, 97, "ASI (100%)", fontsize=(12 if simple else 8), color=AI2030_MUTED)
    else:
        ylab = "effective compute (rebased OOMs)"; y_lo, y_hi = ted_y_med - 7, ted_y_med + 4
    ax.axhline(ted_y_med, color=AI2030_GRID, ls=":", lw=1)
    ax.text(x_lo + .1, ted_y_med + (2 if y != "uplift" else 0) + (ted_y_med * .3 if y == "uplift" else 0),
            "TED-AI (median)", fontsize=(12 if simple else 8), color=AI2030_MUTED)
    a = min(.10, 25.0 / max(n, 1))
    for lt, lu in legs:
        ax.plot(lt, lu, color=AI2030_NAVY, lw=.6, alpha=a, zorder=1)
    for (ct, cu), w in zip(covs_line, win):
        ax.plot(ct, cu, color=(AI2030_OLIVE if w else AI2030_RED), lw=.7, alpha=a * 1.4, zorder=2)
    grid = np.linspace(x_lo, x_hi, 400)
    for series, color in [(legs, AI2030_NAVY), (covs, AI2030_RED)]:
        b, v = _quantile_bands(series, grid, n, weights=wts)
        if v.any():
            g = grid[v]
            ax.fill_between(g, b[10][v], b[90][v], color=color, alpha=.12, lw=0, zorder=3)
            ax.fill_between(g, b[25][v], b[75][v], color=color, alpha=.20, lw=0, zorder=4)
            ax.plot(g, b[50][v], color=color, lw=2.6, zorder=5)
    if det_pts:
        dx = np.array([p[0] for p in det_pts]); dy = np.array([p[1] for p in det_pts])
        ms = max(.12, 30.0 / max(len(det_pts), 1))
        ax.scatter(dx, dy, marker="x", s=26, color="#111", linewidths=.9,
                   alpha=min(.8, ms), zorder=6)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    if simple:
        from matplotlib.patches import Patch
        handles = [plt.Line2D([], [], color=AI2030_NAVY, lw=2.6, label="legal project (median)"),
                   plt.Line2D([], [], color=AI2030_RED, lw=2.6, label="covert project (median)"),
                   Patch(facecolor="#888", alpha=.32, label="25–75% of runs"),
                   Patch(facecolor="#888", alpha=.14, label="10–90% of runs")]
        if det_pts:
            handles.append(plt.Line2D([], [], marker="x", ls="none", ms=7, color="#111",
                                      markeredgewidth=1.0, label="detection"))
        ax.set_xlabel("year", fontsize=14); ax.set_ylabel(ylab, fontsize=14)
        ax.tick_params(labelsize=13)
        if y == "uplift":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.legend(handles=handles, loc="lower right", fontsize=12); ax.grid(False)
        ax.set_title("US forecast of covert project AI capabilities\n"
                     + ("(accounting for detection)" if detect else "(not accounting for detection)"),
                     fontsize=20, pad=12)
        fig.tight_layout()
    else:
        ax.plot([], [], color=AI2030_NAVY, lw=2.6, label="legal — used/deployed: median (band 25-75, 10-90)")
        ax.plot([], [], color=AI2030_RED, lw=2.6, label="covert: median (band 25-75, 10-90)")
        ax.plot([], [], color=AI2030_RED, lw=2, label=f"covert loses ({int((~win).sum())})")
        ax.plot([], [], color=AI2030_OLIVE, lw=2, label=f"covert wins ({int(win.sum())})")
        if det_pts:
            ax.scatter([], [], marker="x", s=30, color="#111", linewidths=1.0,
                       label=f"detection ({len(det_pts)})")
        ax.set_xlabel("year"); ax.set_ylabel(ylab); ax.legend(loc="lower right", fontsize=8.5)
        head = "Covert project — Monte-Carlo trajectories"
        if detect:
            n_det = sum(1 for r in R if r.get("t_det") is not None)
            head += f"  ·  frozen at detection ({n_det}/{len(R)} caught)"
        else:
            head += f"  ·  covert wins {_wmean(win, wts)*100:.0f}%"
        fig.suptitle(head, fontsize=17, y=.965)
        ax.set_title(subtitle or _cap_rw(ds), fontsize=9.5, color=AI2030_MUTED)
        fig.tight_layout(rect=[0, 0, 1, .94])
    default = f"spaghetti_{y}{'_detected' if detect else ''}{'_simple' if simple else ''}.png"
    out = outdir / (fname or default); fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def spaghetti_legal_tsp(ds, outdir):
    """Spaghetti of the legal project's hardware (training-system performance, log10 FLOP/yr)
    across rollouts — the discovered-track TSP. Hardware follows the real training_H100e
    series post-pause, so the lines mostly differ by WHEN each rollout's pause falls (AC time
    and years_before_ac vary) and how far each discovered track runs."""
    w = _wts(ds)
    mask = np.array([bool(r.get("legal_tsp")) for r in ds["records"]])
    R = [r for r in ds["records"] if r.get("legal_tsp")]
    if not R:
        return None
    sw = _wsub(w, mask)
    legs = [(np.asarray(r["legal_t"], float), np.asarray(r["legal_tsp"], float)) for r in R]
    ac = np.array([r["ac_time"] for r in R], float)
    tedt = np.array([r["ted_ai_time"] for r in R if r["ted_ai_time"] is not None], float)
    x_lo, x_hi = float(np.median(ac)) - 3, float(np.median(tedt)) + 22
    allv = np.concatenate([v for _, v in legs])
    y_lo, y_hi = float(np.percentile(allv, .5)) - .3, float(np.percentile(allv, 99.5)) + .3
    n = len(R)
    fig, ax = plt.subplots(figsize=(11, 7))
    a = min(.10, 25.0 / max(n, 1))
    for lt, lv in legs:
        ax.plot(lt, lv, color=AI2030_NAVY, lw=.6, alpha=a, zorder=1)
    grid = np.linspace(x_lo, x_hi, 400)
    b, v = _quantile_bands(legs, grid, n, weights=sw)
    if v.any():
        g = grid[v]
        ax.fill_between(g, b[10][v], b[90][v], color=AI2030_NAVY, alpha=.12, lw=0, zorder=3)
        ax.fill_between(g, b[25][v], b[75][v], color=AI2030_NAVY, alpha=.20, lw=0, zorder=4)
        ax.plot(g, b[50][v], color=AI2030_NAVY, lw=2.6, zorder=5)
    ax.plot([], [], color=AI2030_NAVY, lw=2.6, label="legal TSP: median (band 25-75, 10-90)")
    pm = float(_wquantile([r["pause_year"] for r in R], sw, 0.5))
    ax.axvline(pm, color=AI2030_GRID, ls="--", lw=1.1, zorder=2)
    ax.text(pm + .1, y_lo + .15, "median pause", fontsize=8, color=AI2030_MUTED)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("year"); ax.set_ylabel("training-system performance (log10 FLOP/yr)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Legal project hardware (TSP) — Monte-Carlo trajectories", fontsize=17, y=.965)
    ax.set_title(_cap_rw(ds), fontsize=9.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = outdir / "spaghetti_legal_tsp.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def spaghetti_deal_breakdown(ds, outdir, y="pct_asi"):
    """Spaghetti of the consortium's DEPLOYED capability when the deal can collapse:
    each rollout samples a breakdown time from the deal_breakdown hazard curve (None =>
    the deal holds for the whole horizon). Navy = the consortium's used/deployed glidepath
    while the deal holds; olive = the leading company continuing alone after the collapse
    (the same splice the /covert explorer draws); black dot = the breakdown moment.
    Rollouts whose deal never broke down stay navy end-to-end. The median + quantile
    bands are of the SPLICED composite (consortium, then leading co), i.e. "what frontier
    capability is actually deployed at t given the deal may fall apart".

    Returns None (figure skipped) when the dataset was built without a deal_breakdown
    block."""
    bd_cfg = (ds.get("config") or {}).get("deal_breakdown")
    if not bd_cfg:
        return None
    R = ds["records"]; tk, tdk = TRAJ[y], TED[y]
    wts = _wts(ds)
    comps = []                       # spliced composite per rollout (bands + medians)
    pre_lines, post_lines, bd_pts = [], [], []
    n_broke = 0
    for r in R:
        ut = np.asarray(r.get("legal_used_t", r["legal_t"]), float)
        uy = np.asarray(r.get(f"legal_used_{tk}", r[f"legal_{tk}"]), float)
        bd = r.get("deal_breakdown_t")
        lt = r.get("leading_t")
        if bd is not None and lt:
            lt = np.asarray(lt, float); ly = np.asarray(r[f"leading_{tk}"], float)
            yb = float(np.interp(bd, ut, uy))
            keep = ut < bd
            pre_lines.append((np.concatenate([ut[keep], [bd]]),
                              np.concatenate([uy[keep], [yb]])))
            post_lines.append((lt, ly))
            bd_pts.append((float(bd), yb))
            comps.append((np.concatenate([ut[keep], [bd], lt]),
                          np.concatenate([uy[keep], [yb], ly])))
            n_broke += 1
        else:
            # deal held (never sampled, or sampled beyond the horizon): plain used track
            pre_lines.append((ut, uy)); comps.append((ut, uy))
    ac = np.array([r["ac_time"] for r in R], float)
    tedt = np.array([r["ted_ai_time"] for r in R if r["ted_ai_time"] is not None], float)
    ted_y = [r.get(tdk) for r in R]
    ted_y_med = float(np.nanmedian([v for v in ted_y if v is not None]))
    x_lo, x_hi = float(np.median(ac)) - 3, float(np.median(tedt)) + 22
    n = len(R)
    fig, ax = plt.subplots(figsize=(11, 7))
    if y == "uplift":
        ax.set_yscale("log"); ylab = "AI software R&D uplift (× present-day)"
        y_lo, y_hi = 1.0, ted_y_med * 1e3
    else:
        ylab = "percent of the way from pause to ASI"; y_lo, y_hi = -5, 110
        ax.axhline(0, color=AI2030_GRID, lw=.9); ax.axhline(100, color=AI2030_GRID, lw=.9)
        ax.text(x_lo + .1, 1, "pause (0%)", fontsize=8, color=AI2030_MUTED)
        ax.text(x_lo + .1, 97, "ASI (100%)", fontsize=8, color=AI2030_MUTED)
    ax.axhline(ted_y_med, color=AI2030_GRID, ls=":", lw=1)
    ax.text(x_lo + .1, ted_y_med + (ted_y_med * .3 if y == "uplift" else 2),
            "TED-AI (median)", fontsize=8, color=AI2030_MUTED)
    a = min(.10, 25.0 / max(n, 1))
    for lt, lv in pre_lines:
        ax.plot(lt, lv, color=AI2030_NAVY, lw=.6, alpha=a, zorder=1)
    for lt, lv in post_lines:
        ax.plot(lt, lv, color=AI2030_OLIVE, lw=.7, alpha=a * 1.4, zorder=2)
    grid = np.linspace(x_lo, x_hi, 400)
    b, v = _quantile_bands(comps, grid, n, weights=wts)
    if v.any():
        g = grid[v]
        ax.fill_between(g, b[10][v], b[90][v], color=AI2030_NAVY, alpha=.12, lw=0, zorder=3)
        ax.fill_between(g, b[25][v], b[75][v], color=AI2030_NAVY, alpha=.20, lw=0, zorder=4)
        ax.plot(g, b[50][v], color=AI2030_NAVY, lw=2.6, zorder=5)
    if bd_pts:
        bx = np.array([p[0] for p in bd_pts]); by = np.array([p[1] for p in bd_pts])
        inside = (bx >= x_lo) & (bx <= x_hi)
        ax.scatter(bx[inside], by[inside], s=14, color="#111",
                   alpha=min(.8, max(.15, 30.0 / max(len(bd_pts), 1))), zorder=6)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.plot([], [], color=AI2030_NAVY, lw=2.6,
            label="deployed frontier: median (band 25-75, 10-90)")
    ax.plot([], [], color=AI2030_NAVY, lw=2, label="consortium — deal holding")
    ax.plot([], [], color=AI2030_OLIVE, lw=2, label=f"leading co. after breakdown ({n_broke})")
    if bd_pts:
        ax.scatter([], [], s=16, color="#111", label="deal breaks down")
    ax.set_xlabel("year"); ax.set_ylabel(ylab); ax.legend(loc="lower right", fontsize=8.5)
    p_never = bd_cfg.get("p_never")
    frac = ds["summary"].get("deal_breakdown_frac")
    head = "Deal breakdown — Monte-Carlo trajectories"
    if frac is not None:
        head += f"  ·  broke down in {frac*100:.0f}% of runs"
        if p_never is not None:
            head += f" (never: ~{p_never*100:.0f}%)"
    fig.suptitle(head, fontsize=15, y=.965)
    ax.set_title(_cap_rw(ds), fontsize=9.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = outdir / f"spaghetti_deal_breakdown_{y}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def _unconstrained_caps(ds, tgrid, tk, include_covert=True):
    """Per-rollout unconstrained-project capability by calendar time: the running max
    over the covert track — CENSORED AT DETECTION (freeze-at-catch: a covert caught at
    pause + t_det holds its at-catch level, and contributes nothing if caught before its
    track starts) — and the post-breakdown leading track (legal, no detection). -inf
    where no unconstrained project has any capability yet; levels persist once reached.
    ``include_covert=False`` gives the no-covert-world counterfactual (leading track
    only) for the P(covert exists) mixture. Returns (caps[n_records, len(tgrid)],
    n_caught)."""
    R = ds["records"]
    W = len(tgrid)

    def _by_time(r, tkey, vkey, cut=None):
        t = np.asarray(r.get(tkey) or [], float); v = np.asarray(r.get(vkey) or [], float)
        ok = np.isfinite(t) & np.isfinite(v)
        if ok.sum() < 2:
            return np.full(W, -np.inf)
        t, v = t[ok], v[ok]
        if cut is not None and t.size and cut < t[-1]:
            if cut <= t[0]:
                return np.full(W, -np.inf)
            vc = float(np.interp(cut, t, v))
            keep = t < cut
            t = np.concatenate([t[keep], [cut]]); v = np.concatenate([v[keep], [vc]])
        v = np.maximum.accumulate(v)
        return np.interp(tgrid, t, v, left=-np.inf, right=v[-1])

    n_caught = 0
    caps = np.full((len(R), W), -np.inf)
    for i, r in enumerate(R):
        td = r.get("t_det")
        cut = (float(r["pause_year"]) + float(td)) if td is not None else None
        n_caught += cut is not None
        cap = (_by_time(r, "covert_t", f"covert_{tk}", cut=cut) if include_covert
               else np.full(W, -np.inf))
        if r.get("leading_t"):
            cap = np.maximum(cap, _by_time(r, "leading_t", f"leading_{tk}"))
        caps[i] = cap
    return caps, n_caught


def _exists_mixture(ds, tgrid, tk, caps, ww, exists_prior):
    """Bayesian update re: P(a covert project exists), at the epistemic vantage
    ``pause + deal_breakdown.min_survival_years`` (the same "N years in" clock the
    survival conditioning uses; N = 0 when unconditioned).

    Prior: with probability p0 a covert project exists (the dataset's rollouts describe
    that branch); with 1 - p0 there is none (only the post-breakdown leading company can
    be unconstrained). Evidence at the vantage: NO covert has been detected. Covert-world
    rollouts whose detection draw fired by then (t_det <= N, years after pause) are ruled
    out — each surviving rollout's indicator is the Monte-Carlo estimate of its
    no-detection likelihood — and the no-covert branch has likelihood 1.

    Returns (caps_stacked, weights_stacked, posterior_P_exists): the posterior-weighted
    world sample — covert-world rows at p0 * ww * 1[survived], no-covert rows (leading
    track only) at (1 - p0) * ww — for the same exceedance code the unmixed field uses."""
    R = ds["records"]
    p0 = float(exists_prior)
    n_min = float(((ds.get("config") or {}).get("deal_breakdown") or {})
                  .get("min_survival_years") or 0.0)
    surv = np.array([1.0 if (r.get("t_det") is None or float(r["t_det"]) > n_min) else 0.0
                     for r in R])
    caps_lead, _ = _unconstrained_caps(ds, tgrid, tk, include_covert=False)
    w_cov = p0 * ww * surv
    w_no = (1.0 - p0) * ww
    tot = float(w_cov.sum() + w_no.sum())
    posterior = float(w_cov.sum() / tot) if tot > 0 else float("nan")
    return np.vstack([caps, caps_lead]), np.concatenate([w_cov, w_no]), posterior


def _exceedance_field(caps, ww, ycenters):
    """P[y, t] = (weighted) fraction of world-sample rows with cap(t) >= y."""
    n, W = caps.shape
    tot = float(np.sum(ww))
    ycenters = np.asarray(ycenters, float)
    P = np.empty((len(ycenters), W))
    for j in range(W):
        col = caps[:, j]
        order = np.argsort(col, kind="mergesort")
        cw = np.cumsum(np.asarray(ww, float)[order])
        idx = np.searchsorted(col[order], ycenters, side="left")
        below = np.where(idx > 0, cw[np.clip(idx - 1, 0, n - 1)], 0.0)
        P[:, j] = (tot - below) / tot
    return P


def capability_cdf2d_csv(ds, outdir, y="pct_asi", levels=None, exists_prior=None):
    """CSV companion of capability_cdf2d: the same detection-aware exceedance field
    P(an unconstrained project has reached level >= L by year T), tabulated on an exact
    monthly (1/12-yr) calendar grid. One row per month; one `p_ge_<level>` column per
    capability level. Default levels: pct-to-ASI 0..100 in steps of 5 (100 = ASI), or a
    1-2-5 ladder from 1x to 100,000x for y="uplift". Weighted-run aware (reweighted
    datasets tabulate the weighted fractions)."""
    R = ds["records"]
    if not R:
        return None
    tk = TRAJ[y]
    w = _wts(ds)
    ww = np.ones(len(R)) if w is None else np.asarray(w, float)
    tot = float(ww.sum())
    if tot <= 0:
        return None
    if levels is None:
        if y == "uplift":
            # Dense through the linear-plot range (1x steps to 100x), then coarsening;
            # a 1-2-5 ladder alone renders as fake "shelves" on a linear axis.
            levels = (list(range(1, 101))
                      + list(range(105, 201, 5))
                      + list(range(220, 1001, 20))
                      + [m * 10 ** e for e in (3, 4) for m in (1, 2, 5)] + [10 ** 5])
        else:
            levels = list(range(0, 101, 5))
    ac = np.array([r["ac_time"] for r in R], float)
    tedt = np.array([r["ted_ai_time"] for r in R if r["ted_ai_time"] is not None], float)
    x_lo, x_hi = float(np.median(ac)) - 3, float(np.median(tedt)) + 22
    # exact month boundaries (integer months, so the years are float-exact)
    months = np.arange(int(np.ceil(x_lo * 12)), int(np.floor(x_hi * 12)) + 1)
    tgrid = months / 12.0
    caps, n_caught = _unconstrained_caps(ds, tgrid, tk)
    # Optional Bayesian update re: P(covert exists) — yaml `covert_exists_prior`, or an
    # explicit argument. None => the covert exists with certainty (unmixed field).
    if exists_prior is None:
        exists_prior = (ds.get("config") or {}).get("covert_exists_prior")
    if exists_prior is not None:
        caps, ww, _post = _exists_mixture(ds, tgrid, tk, caps, ww, exists_prior)
    # P[level, month] = weighted fraction of posterior worlds at or above the level by then
    P = _exceedance_field(caps, ww, levels)
    rows = ["year," + ",".join(f"p_ge_{g:g}" for g in levels)]
    for j, t in enumerate(tgrid):
        rows.append(f"{t:.6f}," + ",".join(f"{P[k, j]:.4f}" for k in range(len(levels))))
    out = outdir / f"capability_cdf2d_{y}.csv"
    out.write_text("\n".join(rows) + "\n")
    return out.name


def capability_cdf2d(ds, outdir, y="pct_asi", floor=0.10, exists_prior=None, linear_max=None,
                     x_max=None):
    """2-D empirical CDF, rendered as dithering: P(an UNCONSTRAINED project — the covert
    project or, after a deal breakdown, the leading company — has reached capability
    level Y by year T). Per rollout the relevant capability is the running max over the
    covert track and (when the deal broke down) the post-breakdown leading track; the
    consortium doesn't count (it's the constrained/deal-bound actor). The covert track
    is CENSORED AT DETECTION (the same freeze-at-catch convention as the detected
    spaghetti): a covert caught at pause + t_det holds its at-catch capability and
    progresses no further; the leading company is legal and faces no detection. Solid
    black = every rollout has somebody at that level by then; blank = fewer than
    ``floor`` (default 10%) of rollouts do; in between, stipple density is linear in
    the probability. The dither pattern is a fixed-seed random threshold, so the figure
    is a pure, reproducible function of the dataset."""
    R = ds["records"]; tk, tdk = TRAJ[y], TED[y]
    w = _wts(ds)
    ww = np.ones(len(R)) if w is None else np.asarray(w, float)
    tot = float(ww.sum())
    if not R or tot <= 0:
        return None
    ac = np.array([r["ac_time"] for r in R], float)
    tedt = np.array([r["ted_ai_time"] for r in R if r["ted_ai_time"] is not None], float)
    ted_y = [r.get(tdk) for r in R]
    ted_y_med = float(np.nanmedian([v for v in ted_y if v is not None]))
    x_lo, x_hi = float(np.median(ac)) - 3, float(np.median(tedt)) + 22
    if x_max is not None:                # x_max: crop the year axis (default: ted median + 22)
        x_hi = min(x_hi, float(x_max))
    W, H = 560, 320                      # dither cells (x, y): ~2px/cell at 11x7in, 140dpi
    tgrid = np.linspace(x_lo, x_hi, W)
    if y == "uplift":
        # linear_max: linear y-axis from 1x to that value (default: log to ~ted*1e3)
        if linear_max is not None:
            y_lo, y_hi = 1.0, float(linear_max)
            yedges = np.linspace(y_lo, y_hi, H + 1)
        else:
            y_lo, y_hi = 1.0, ted_y_med * 1e3
            yedges = np.geomspace(y_lo, y_hi, H + 1)
        ylab = "AI software R&D uplift (× present-day)"
    else:
        y_lo, y_hi = 0.0, 110.0
        yedges = np.linspace(y_lo, y_hi, H + 1)
        ylab = "percent of the way from pause to ASI"
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    caps, n_caught = _unconstrained_caps(ds, tgrid, tk)
    # Optional Bayesian update re: P(covert exists) — yaml `covert_exists_prior`, or an
    # explicit argument. None => the covert exists with certainty (unmixed field).
    if exists_prior is None:
        exists_prior = (ds.get("config") or {}).get("covert_exists_prior")
    posterior = None
    if exists_prior is not None:
        caps, ww, posterior = _exists_mixture(ds, tgrid, tk, caps, ww, exists_prior)
    # Exceedance field P[y, t] = (weighted) fraction of posterior worlds with cap(t) >= y.
    P = _exceedance_field(caps, ww, ycenters)

    # Random-threshold dithering: dot probability 0 below the floor, then linear to
    # solid at P = 1 (P >= 1 - 1/n renders fully solid so "all rollouts" reads black).
    prob = np.clip((P - floor) / (1.0 - floor), 0.0, 1.0)
    prob[P >= 1.0 - 1e-12] = 1.0
    rng = np.random.default_rng(0)                     # fixed seed: reproducible pattern
    dots = rng.random((H, W)) < prob

    fig, ax = plt.subplots(figsize=(11, 7))
    from matplotlib.colors import ListedColormap
    ax.pcolormesh(np.linspace(x_lo, x_hi, W + 1), yedges, dots,
                  cmap=ListedColormap([(0, 0, 0, 0), AI2030_TEXT]), vmin=0, vmax=1,
                  shading="flat", rasterized=True, zorder=1)
    if y == "uplift":
        if linear_max is None:
            ax.set_yscale("log")
    else:
        ax.axhline(100, color=AI2030_GRID, lw=.9, zorder=2)
        ax.text(x_lo + .1, 101.5, "ASI (100%)", fontsize=8, color=AI2030_MUTED)
    if ted_y_med <= y_hi:                  # off-scale on a clipped linear axis
        ax.axhline(ted_y_med, color=AI2030_RED, ls=":", lw=1.2, zorder=3)
        ax.text(x_lo + .1, ted_y_med * (1.05 if y == "uplift" else 1) + (0 if y == "uplift" else 1.5),
                "TED-AI (median)", fontsize=8, color=AI2030_RED)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo if y != "uplift" else 1.0, y_hi)
    ax.set_xlabel("year"); ax.set_ylabel(ylab)
    n_lead = sum(1 for r in R if r.get("leading_t"))
    who = (f"covert (frozen at detection; {n_caught}/{len(R)} caught)"
           + (f" or post-breakdown leading co. ({n_lead} runs)" if n_lead else ""))
    caption = f"actor: {who}  ·  solid = 100% of runs, blank < {floor*100:.0f}%\n"
    if posterior is not None:
        n_min = float(((ds.get("config") or {}).get("deal_breakdown") or {})
                      .get("min_survival_years") or 0.0)
        caption += (f"Bayesian update — P(covert exists): prior {float(exists_prior)*100:.0f}% → "
                    f"{posterior*100:.0f}% given none detected by pause+{n_min:g}y\n")
    caption += _cap_rw(ds)
    fig.suptitle("P(an unconstrained project has reached this capability by this year)",
                 fontsize=15, y=.97)
    ax.set_title(caption, fontsize=8.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .92 if posterior is not None else .93])
    lin = f"_lin{linear_max:g}" if linear_max is not None else ""
    lin += f"_end{x_max:g}" if x_max is not None else ""
    out = outdir / f"capability_cdf2d_{y}{lin}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def lead_hist(ds, outdir, clip_lo=-30, clip_hi=5):
    R = ds["records"]; w = _wts(ds)
    lmask = np.array([r["legal_ttt"] is not None and r["covert_ttt"] is not None for r in R])
    lead = np.array([r["legal_ttt"] - r["covert_ttt"] for r in R
                     if r["legal_ttt"] is not None and r["covert_ttt"] is not None])
    lw = _wsub(w, lmask)
    never = sum(1 for r in R if r["covert_ttt"] is None)
    win_frac = _wmean([r["covert_wins"] for r in R], w)
    under = int(np.sum(lead < clip_lo))
    edges = np.arange(clip_lo, clip_hi + 1, 1.0); ctr = .5 * (edges[:-1] + edges[1:])
    cnt, _ = np.histogram(np.clip(lead, clip_lo, clip_hi), bins=edges, weights=lw)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.bar(ctr, cnt, width=.92, color=[AI2030_OLIVE if c >= 0 else AI2030_RED for c in ctr], alpha=.85)
    ax.axvline(0, color=AI2030_NAVY, lw=1.6)
    med = float(_wquantile(lead, lw, 0.5)); ax.axvline(np.clip(med, clip_lo, clip_hi), color=AI2030_MUTED, ls="--", lw=1.4)
    ax.text(np.clip(med, clip_lo, clip_hi), ax.get_ylim()[1]*.9, f"  median {med:+.1f}y", color=AI2030_MUTED, fontsize=9)
    notes = []
    if under:
        notes.append(f"{under} runs lead < {clip_lo:g}y (binned at edge)")
    if never:
        notes.append(f"{never} runs: covert NEVER reaches TED-AI")
    if notes:
        ax.text(.02, .80, "\n".join(notes), transform=ax.transAxes, fontsize=8.5, color=AI2030_RED, va="top",
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec=AI2030_GRID, alpha=.9))
    ax.set_xlim(clip_lo - 1, clip_hi)
    ax.set_xlabel("lead = legal reaches TED-AI+tax − covert reaches TED-AI (years)")
    ax.set_ylabel("Monte-Carlo runs")
    fig.suptitle(f"Covert lead-time distribution  ·  covert wins {win_frac*100:.0f}%", fontsize=15, y=.97)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = outdir / "lead_hist.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def covert_slowdown_hist(ds, outdir):
    """Distribution of the per-rollout covert SLOWDOWN FACTOR: time from the pause (where
    the covert project splits off the un-paused default trajectory) to TED-AI, as a ratio
    to the default world's — (covert_ttt − pause_year) / (ted_ai_time − pause_year).
    Computed per-rollout by simulate_covert (`covert_slowdown`); log-spaced bins with
    p10 / median / p90 marked. Runs whose covert never reaches TED-AI have no factor and
    are excluded (counted in the note)."""
    R = ds["records"]; w = _wts(ds)
    if not R or "covert_slowdown" not in R[0]:
        raise RuntimeError("dataset predates covert_slowdown — rebuild with --force")
    mask = np.array([r["covert_slowdown"] is not None for r in R])
    sf = np.array([r["covert_slowdown"] for r in R if r["covert_slowdown"] is not None], float)
    if sf.size == 0:
        return None
    sw = _wsub(w, mask)
    never = int(np.sum(~mask))
    lo, med, hi = _wpct(sf, sw, 10), _wquantile(sf, sw, 0.5), _wpct(sf, sw, 90)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bins = np.logspace(np.log10(min(sf.min(), 1.0) * .9), np.log10(sf.max() * 1.1), 34)
    ax.hist(sf, bins=bins, color=AI2030_RED, alpha=.8, zorder=2, weights=sw)
    ax.axvline(1.0, color=AI2030_NAVY, lw=1.6, zorder=3)
    ax.text(1.0, ax.get_ylim()[1] * .60, "no slowdown ", rotation=90, ha="right", va="top",
            fontsize=8.5, color=AI2030_NAVY)
    ax.axvline(med, color=AI2030_MUTED, ls="--", lw=1.8, zorder=3, label=f"median {med:.2f}×")
    ax.axvline(lo, color=AI2030_MUTED, ls="--", lw=1.4, alpha=.7, zorder=3, label=f"p10 {lo:.2f}×")
    ax.axvline(hi, color=AI2030_MUTED, ls=":", lw=1.4, zorder=3, label=f"p90 {hi:.2f}×")
    if never:
        ax.plot([], [], " ", label=f"{never} runs: covert NEVER reaches TED-AI (excluded)")
    ax.legend(loc="upper left", fontsize=9, framealpha=.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("slowdown = (covert TED-AI − pause) / (default TED-AI − pause)")
    ax.set_ylabel("Monte-Carlo runs")
    ax.grid(True, which="both", axis="x", alpha=.12); ax.grid(True, axis="y", alpha=.15)
    fig.suptitle("Covert slowdown vs the un-paused default world", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "covert_slowdown_hist.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def lead_drivers(ds, outdir):
    R = ds["records"]
    def ms(r, n, f):
        m = (r.get("milestones") or {}).get(n); return (m[f] if m else None)
    oom = np.array([(ms(r, "TED-AI", "e") - ms(r, "AC", "e"))
                    if (ms(r, "TED-AI", "e") is not None and ms(r, "AC", "e") is not None) else np.nan
                    for r in R], float)
    swr = np.array([r["model_params"].get("software_progress_rate_at_reference_year", np.nan) for r in R], float)
    lead = np.array([r["lead_years"] if r["lead_years"] is not None else np.nan for r in R], float)
    never = ~np.isfinite(lead); fmin = np.nanmin(lead) if np.isfinite(lead).any() else 0.0
    rankable = np.where(never, fmin - 1, lead)
    win = np.array([r["covert_wins"] for r in R], bool)
    floor = fmin - max(3, .15 * abs(fmin))
    def rho(x):
        m = np.isfinite(x) & (np.abs(x) < 1e6) & np.isfinite(rankable)
        rx, ry = np.argsort(np.argsort(x[m])), np.argsort(np.argsort(rankable[m]))
        return float(np.corrcoef(rx, ry)[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (x, lab, xmax) in zip(axes, [(oom, "AC→TED-AI distance (OOMs)", float(np.nanpercentile(oom[oom < 1e6], 99)+2)),
                                         (swr, "software_progress_rate_at_ref_year", float(np.nanpercentile(swr, 99)+.5))]):
        ax.set_yscale("symlog", linthresh=1.0)
        xd = np.where(x > xmax, np.nan, x); fin = np.isfinite(lead)
        ax.scatter(xd[fin & ~win], lead[fin & ~win], s=14, color=AI2030_RED, alpha=.5, label="covert loses")
        ax.scatter(xd[fin & win], lead[fin & win], s=16, color=AI2030_OLIVE, alpha=.75, label="covert wins")
        if never.any():
            ax.scatter(xd[never], np.full(int(never.sum()), floor), s=12, marker="v", color="#c98", alpha=.6,
                       label=f"never reaches TED-AI ({int(never.sum())})")
        ax.axhline(0, color=AI2030_NAVY, lw=1.2); ax.set_xlim(left=0, right=xmax)
        ax.set_xlabel(lab); ax.set_ylabel("covert lead (yr; symlog) — + = covert first")
        ax.set_title(f"Spearman ρ = {rho(x):+.2f}", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("What drives the covert lead?", fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "lead_drivers.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def physics_check(ds, outdir):
    R = ds["records"]; w = _wts(ds)
    sr = np.array([r.get("legal_setup_ceiling_ratio") or 0.0 for r in R], float)
    pr = np.array([r.get("legal_sustained_ceiling_ratio") or 0.0 for r in R], float)
    n_post = int(np.sum(pr > 1.05)); n_setup = int(np.sum(sr > 1.05))
    los = np.log10(np.clip(sr, 1e-3, 1e8)); lop = np.log10(np.clip(pr, 1e-3, 1e8))
    bins = np.linspace(-1.5, 4, 45)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.hist(np.clip(los, bins[0], bins[-1]), bins=bins, color=AI2030_OLIVE, alpha=.55, weights=w,
            label=f"setup windfall (peak)  [{n_setup}/{len(R)} > ceiling]")
    ax.hist(np.clip(lop, bins[0], bins[-1]), bins=bins, color=AI2030_NAVY, alpha=.7, weights=w,
            label=f"sustained policy (peak)  [{n_post}/{len(R)} > ceiling]")
    ax.axvline(0, color=AI2030_RED, lw=1.6); ax.text(.05, ax.get_ylim()[1]*.92, "ceiling (1x)", color=AI2030_RED, fontsize=8)
    ax.set_xlabel("log₁₀ (legal research effort) / (AIFM resource+capability ceiling)")
    ax.set_ylabel("rollouts"); ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(f"Legal effort vs production-function ceiling — {ds['config'].get('legal_mode')} legal"
                 + (("  ·  " + _rw_note(ds)) if _rw_note(ds) else ""), fontsize=10)
    fig.tight_layout()
    out = outdir / "legal_physics_check.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def _ms(r, n, f):
    m = (r.get("milestones") or {}).get(n)
    return (m[f] if m else None)


def _sci10(v):
    """log10-value -> '1.4e28'-style scientific string (rounds the mantissa safely)."""
    return f"{10**v:.1e}".replace("e+0", "e").replace("e+", "e")


def dist_conditional(ds, outdir):
    """Unconditional vs win-conditional distributions of the takeoff variables."""
    R = ds["records"]; w = _wts(ds); win = np.array([r["covert_wins"] for r in R], bool)
    oom = np.array([(_ms(r, "TED-AI", "e") - _ms(r, "AC", "e"))
                    if (_ms(r, "TED-AI", "e") is not None and _ms(r, "AC", "e") is not None) else np.nan
                    for r in R], float)
    asited = np.array([(r["asi_e"] - r["ted_ai_e"]) if (r.get("asi_e") and r.get("ted_ai_e")) else np.nan
                       for r in R], float)
    swr = np.array([r["model_params"].get("software_progress_rate_at_reference_year", np.nan) for r in R], float)
    gap = np.array([r["ac_to_ted_gap"] if r.get("ac_to_ted_gap") is not None else np.nan for r in R], float)
    panels = [("AC→TED-AI distance (OOMs eff. compute)", oom, False),
              ("software_progress_rate_at_reference_year", swr, False),
              ("AC→TED-AI gap (calendar years)", gap, True),
              ("ASI−TED-AI distance (OOMs eff. compute)", asited, False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (lab, v, logx) in zip(axes.ravel(), panels):
        good = np.isfinite(v) & (np.abs(v) < 1e6); va, vw = v[good], v[good & win]
        wa = _wsub(w, good); ww = _wsub(w, good & win)
        if va.size == 0:
            continue
        lo, hi = np.percentile(va, 1), np.percentile(va, 99)
        if logx and (va > 0).any():
            bins = np.logspace(np.log10(max(np.percentile(va[va > 0], 1), 1e-3)), np.log10(hi), 34); ax.set_xscale("log")
        else:
            bins = np.linspace(lo, hi, 34)
        ax.hist(np.clip(va, bins[0], bins[-1]), bins=bins, density=True, color=AI2030_MUTED, alpha=.45,
                weights=wa, label=f"all runs (n={good.sum()})")
        ax.hist(np.clip(vw, bins[0], bins[-1]), bins=bins, density=True, color=AI2030_OLIVE, alpha=.55,
                weights=ww, label=f"covert wins (n={(good & win).sum()})")
        ax.axvline(_wquantile(va, wa, 0.5), color=AI2030_MUTED, ls="--", lw=1.3)
        if vw.size:
            ax.axvline(_wquantile(vw, ww, 0.5), color="#46600f", ls="--", lw=1.6)
        ax.set_xlabel(lab); ax.set_ylabel("density")
        ax.set_title(f"median: all {_wquantile(va, wa, 0.5):.2f} · wins {_wquantile(vw, ww, 0.5):.2f}"
                     if vw.size else "—", fontsize=9.5)
        ax.legend(fontsize=8.5)
    fig.suptitle("Takeoff variables: unconditional vs conditional on covert win"
                 + (("  ·  " + _rw_note(ds)) if _rw_note(ds) else ""), fontsize=13, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    out = outdir / "dist_conditional.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def characterize_wins(ds, outdir):
    """Are wins a degenerate corner? regime scatter + the key parameter's position."""
    R = ds["records"]; w = _wts(ds); win = np.array([r["covert_wins"] for r in R], bool)
    jump = np.array([(_ms(r, "TED-AI", "e") - _ms(r, "AC", "e"))
                     if (_ms(r, "TED-AI", "e") is not None and _ms(r, "AC", "e") is not None) else np.nan
                     for r in R], float)
    gap = np.array([r["ac_to_ted_gap"] if r.get("ac_to_ted_gap") is not None else np.nan for r in R], float)
    swr = np.array([r["model_params"].get("software_progress_rate_at_reference_year", np.nan) for r in R], float)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    axa.set_yscale("log"); XM = 30.0; n_out = int(np.sum(jump > XM)); jd = np.clip(jump, None, XM)
    axa.scatter(jd[~win], gap[~win], s=14, color=AI2030_RED, alpha=.45, label="covert loses")
    axa.scatter(jd[win], gap[win], s=16, color=AI2030_OLIVE, alpha=.7, label="covert wins")
    axa.set_xlim(0, XM)
    if n_out:
        axa.text(.98, .02, f"{n_out} jump>{XM:g} clipped", transform=axa.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color=AI2030_MUTED)
    axa.set_xlabel("AC→TED-AI jump (OOMs eff. compute)"); axa.set_ylabel("AC→TED-AI gap (calendar yr, log)")
    axa.set_title("(a) win regime: big jump + fast takeoff", fontsize=10); axa.legend(fontsize=8.5, loc="upper left")
    fin = np.isfinite(swr); av = swr[fin]; aw = _wsub(w, fin)
    bins = np.linspace(np.percentile(av, 1), np.percentile(av, 99), 40)
    axb.hist(swr[~win], bins=bins, color=AI2030_RED, alpha=.5, density=True, weights=_wsub(w, ~win), label="loses")
    axb.hist(swr[win], bins=bins, color=AI2030_OLIVE, alpha=.6, density=True, weights=_wsub(w, win), label="wins")
    for q, ls in [(10, ":"), (50, "--"), (90, ":")]:
        axb.axvline(_wpct(av, aw, q), color=AI2030_MUTED, ls=ls, lw=1)
    axb.set_xlabel("software_progress_rate_at_ref_year\n(dotted/dashed = full-dist p10/p50/p90)")
    axb.set_ylabel("density"); axb.set_title("(b) wins are upper-mid, not tail", fontsize=10); axb.legend(fontsize=8.5)
    fig.suptitle("Are covert-win worlds degenerate? — regime + parameter position"
                 + (("  ·  " + _rw_note(ds)) if _rw_note(ds) else ""), fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "characterize_wins.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def target_times(ds, outdir):
    """When each project reaches its target; legal slip vs the intended target_year."""
    R = ds["records"]; w = _wts(ds); lead_yrs = ds["config"]["covert_knobs_fixed"]["legal_scaling_duration"]
    legal_m = np.array([r["legal_ttt"] is not None for r in R])
    covert_m = np.array([r["covert_ttt"] is not None for r in R])
    legal = np.array([r["legal_ttt"] for r in R if r["legal_ttt"] is not None])
    covert = np.array([r["covert_ttt"] for r in R if r["covert_ttt"] is not None])
    lw = _wsub(w, legal_m); cw = _wsub(w, covert_m)
    n_cov_never = sum(1 for r in R if r["covert_ttt"] is None)
    slip = np.array([r["legal_ttt"] - (r["pause_year"] + lead_yrs) for r in R if r["legal_ttt"] is not None])
    slipped = int(np.sum(slip > 0.05))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    lo = min(legal.min(), covert.min() if covert.size else legal.min())
    hi = np.percentile(np.concatenate([legal, covert]) if covert.size else legal, 97)
    bins = np.linspace(lo - .5, hi + 1, 45)
    axa.hist(legal, bins=bins, color=AI2030_NAVY, alpha=.7, weights=lw, label=f"legal → TED-AI+tax ({len(legal)})")
    if covert.size:
        axa.hist(covert, bins=bins, color=AI2030_RED, alpha=.6, weights=cw, label=f"covert → TED-AI ({len(covert)})")
    axa.set_xlabel("calendar year reaching target"); axa.set_ylabel("rollouts")
    axa.set_title(f"(a) target-arrival year · covert never reaches: {n_cov_never}", fontsize=10); axa.legend(fontsize=8.5)
    smax = max(.5, np.percentile(slip, 98)); sb = np.linspace(-.5, smax + .5, 45)
    axb.hist(np.clip(slip, sb[0], sb[-1]), bins=sb, color=AI2030_OLIVE, alpha=.8, weights=lw)
    axb.axvline(0, color=AI2030_RED, lw=1.6); axb.text(.03, axb.get_ylim()[1]*.92, "on time", color=AI2030_RED, fontsize=8)
    axb.set_xlabel("legal SLIP = arrival − intended target_year (yr)"); axb.set_ylabel("rollouts")
    axb.set_title(f"(b) deal-timeline slip · {slipped}/{len(slip)} slip late", fontsize=10)
    fig.suptitle(f"Target-arrival distribution · {ds['config'].get('legal_mode')} legal"
                 + (("  ·  " + _rw_note(ds)) if _rw_note(ds) else ""), fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "target_times.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def spaghetti_acasi(ds, outdir, lo=0.5, hi=2.0, detect=False):
    """Spaghetti conditioned on the default-trajectory AC→ASI calendar gap (ASI time
    is in the stored milestones, so this is a pure-JSON filter — no re-derivation)."""
    w = _wts(ds)
    keep = []
    for i, r in enumerate(ds["records"]):
        at, ai = r.get("ac_time"), _ms(r, "ASI", "time")
        if at is not None and ai is not None and lo <= (ai - at) <= hi:
            keep.append(i)
    if not keep:
        return None
    R = [ds["records"][i] for i in keep]
    sub = dict(ds); sub["records"] = R
    if w is not None:
        sub["_weights"] = w[keep]   # keep weights aligned to the filtered records
    sub["summary"] = dict(ds["summary"], n_ok=len(R),
                          covert_win_frac=float(_wmean([x["covert_wins"] for x in R], _wsub(w, keep))))
    suffix = "_detected" if detect else ""
    return spaghetti(sub, outdir, "pct_asi", detect=detect,
                     fname=f"spaghetti_acASIgap_{lo:g}_{hi:g}{suffix}.png",
                     subtitle=f"conditioned on default AC→ASI gap ∈ [{lo:g}, {hi:g}] yr  ·  "
                              f"{len(R)}/{len(ds['records'])} runs  ·  " + _cap_rw(ds))


def _median_param_pct(ds):
    """(t, pct_to_asi) for the deterministic median-parameter covert run, read from
    the dataset's median_scenario (no model re-run). None if unavailable."""
    ms = ds.get("median_scenario") or {}
    cov = ms.get("covert") or {}
    asi_e = ((ms.get("milestones") or {}).get("ASI") or {}).get("e")
    e_pause = ms.get("e_pause")
    if not cov.get("t") or asi_e is None or e_pause is None or asi_e <= e_pause:
        return None
    t = np.asarray(cov["t"], float); e = np.asarray(cov["e"], float)
    return t, 100.0 * (e - e_pause) / (asi_e - e_pause)


def detection_winnow(ds, outdir):
    """Winnow the covert trajectories by their per-rollout detection draw (baked into
    each record by the MC pipeline). Pure reader: the only model touched is the cheap
    detection frailty (for display curves), never the physics. Detected-before-goal
    rollouts are truncated at their sampled capture time and marked with a red ✕;
    survivors are solid (olive = wins, red = loses). The red band is the ensemble
    median with detected runs FROZEN at the capability level where they were caught."""
    R = ds["records"]
    if not R:
        return None
    if "covert_size_mw" not in R[0]:
        raise RuntimeError("dataset predates detection integration — rebuild with "
                           "`python -m plan_a_covert_proj.pipeline.run --force`")
    tk = TRAJ["pct_asi"]
    N = len(R)
    win = np.array([r_["covert_wins"] for r_ in R], bool)
    detected = np.array([bool(r_["detected_before_goal"]) for r_ in R], bool)
    survivors = ~detected
    t_det = [r_.get("t_det") for r_ in R]                                   # yr after pause, or None
    s_goal = np.array([(r_.get("survival_at_goal") or 0.0) for r_ in R], float)  # P(undetected @ goal)

    base = float(win.mean())
    after_ev = float(np.sum(win * s_goal) / N)        # survival-weighted expectation
    after_thin = float((win & survivors).sum() / N)   # this MC draw

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})

    # --- left: detection probability vs years run, at the median covert size --------
    # Frailty fit (from the elicited grids) is stored in the dataset; fall back to the
    # module default for pre-fit datasets.
    cfg = ds.get("config") or {}
    fits = detection.normalize_fits(cfg.get("detection_frailty_fit"))
    # Silver bullet is now a per-rollout ceiling-logistic anchored on the legal used->TED-AI
    # crossing. For this single display curve use REPRESENTATIVE anchors: the median across
    # rollouts that reach TED-AI of t0 = midpoint(pause-end, used->TED) and t1 = used->TED + Δ.
    # None => inert.
    sb = None
    _sbc = cfg.get("detection_silver_bullet")
    if _sbc:
        _t0s, _t1s = [], []
        for r_ in R:
            ut, py = r_.get("legal_used_ttt"), r_.get("pause_year")
            sp = r_.get("covert_params", {}).get("setup_period_years")
            if ut is None or py is None or sp is None:
                continue
            t_used = ut - py
            _t0s.append(0.5 * (sp + t_used))                 # time-midpoint pause-end..used->TED
            _t1s.append(t_used + _sbc["delta_years"])
        if _t0s:
            sb = detection.build_silver_bullet(
                _sbc["p_mid_pause_to_ted"], float(np.median(_t0s)),
                _sbc["p_used_ted_plus_d"], float(np.median(_t1s)), _sbc["ceiling"])
    r_med = float(np.median([r_.get("detection_r", 0.0) for r_ in R]))
    sizes = np.array([r_["covert_size_mw"] for r_ in R if r_.get("covert_size_mw") is not None], float)
    # Per-site power the covert deploys at (sampled knob); median sets the display curve's
    # site size (n = total / site). None for pre-site datasets -> single-facility curve.
    site_vals = [r_.get("covert_params", {}).get("optimal_site_mw") for r_ in R]
    site_vals = [s for s in site_vals if s is not None]
    site_med = float(np.median(site_vals)) if site_vals else None
    tg = np.linspace(0, 15, 200)
    smed = None
    # Dormancy shifts the detection clock (no facility to detect before construction
    # starts); the display curve uses the median sampled dormancy.
    dorm_vals = [r_.get("covert_params", {}).get("dormancy_years") for r_ in R]
    dorm_med = float(np.median([v for v in dorm_vals if v is not None] or [0.0]))
    if sizes.size:
        smed = float(np.median(sizes)); s10, s90 = np.percentile(sizes, [10, 90])
        dp = lambda sz, t: detection.detection_probability(sz, t, r_med, site_mw=site_med, fits=fits, silver_bullet=sb, dormancy_years=dorm_med)  # noqa: E731
        axL.fill_between(tg, dp(s10, tg), dp(s90, tg), color=AI2030_AMBER, alpha=.18, lw=0)
        axL.plot(tg, dp(smed, tg), color=AI2030_AMBER, lw=2.4)
        for yr in (3, 5, 10):
            axL.plot(yr, float(dp(smed, yr)), "o", color=AI2030_MUTED, ms=4)
        axL.set_title(f"Detection vs years run · median {smed:.0f} MW (band p10–p90 size)", fontsize=10)
    axL.set_xlabel("years the covert project has run"); axL.set_ylabel("probability detected")
    axL.set_xlim(0, 15); axL.set_ylim(0, 1)

    # --- right: winnowed trajectories ----------------------------------------------
    ac = np.array([r_["ac_time"] for r_ in R], float)
    tedt = np.array([r_["ted_ai_time"] for r_ in R if r_["ted_ai_time"]], float)
    x_lo = float(np.median(ac)) - 3
    x_hi = (float(np.median(tedt)) + 22) if tedt.size else (float(np.median(ac)) + 25)
    axR.axhline(100, color=AI2030_GRID, lw=.9); axR.axhline(0, color=AI2030_GRID, lw=.9)
    spag_a = float(min(0.55, 20.0 / max(N, 1)))
    for i, r_ in enumerate(R):
        t = np.asarray(r_["covert_t"], float); y = np.asarray(r_[f"covert_{tk}"], float)
        td = t_det[i]
        if detected[i] and td is not None:                       # caught before the goal
            m = t <= (r_["pause_year"] + td)
            if m.any():
                axR.plot(t[m], y[m], color=AI2030_GRID, lw=.6, alpha=spag_a * 0.7, zorder=1)
                axR.plot([t[m][-1]], [y[m][-1]], "x", color=AI2030_RED, ms=4, mew=1.0,
                         alpha=min(1.0, spag_a * 2.5 + 0.15), zorder=2)
        else:                                                    # survives to the goal (or never reaches)
            axR.plot(t, y, color=(AI2030_OLIVE if win[i] else AI2030_RED),
                     lw=.8, alpha=min(1.0, spag_a * 1.4), zorder=3)

    # winnowed median band: keep every run but FREEZE caught ones at the capture level
    # (a detected project is exposed and stops progressing) — no survival conditioning.
    grid = np.linspace(x_lo, x_hi, 400)
    frozen = []
    for i, r_ in enumerate(R):
        t = np.asarray(r_["covert_t"], float); y = np.asarray(r_[f"covert_{tk}"], float)
        td = t_det[i]
        if detected[i] and td is not None and t.size:
            tstop = r_["pause_year"] + td
            if tstop < t[-1]:
                m = t <= tstop; ys = float(np.interp(tstop, t, y))
                frozen.append((np.concatenate([t[m], [tstop, x_hi]]), np.concatenate([y[m], [ys, ys]])))
                continue
        frozen.append((t, y))
    b, valid = _quantile_bands(frozen, grid, N)
    if valid.any():
        g = grid[valid]
        axR.fill_between(g, b[10][valid], b[90][valid], color=AI2030_RED, alpha=.12, zorder=4, lw=0)
        axR.fill_between(g, b[25][valid], b[75][valid], color=AI2030_RED, alpha=.20, zorder=5, lw=0)
        axR.plot(g, b[50][valid], color=AI2030_RED, lw=2.4, zorder=7)

    mp = _median_param_pct(ds)
    if mp is not None:
        axR.plot(mp[0], mp[1], color=AI2030_TEXT, lw=2.0, zorder=8)

    # legend (empty handles)
    axR.plot([], [], color=AI2030_RED, lw=2.4, label="covert median — winnowed (band 25-75, 10-90)")
    if mp is not None:
        axR.plot([], [], color=AI2030_TEXT, lw=2.0, label="median-parameter run (no detection)")
    axR.plot([], [], color=AI2030_OLIVE, lw=2, label=f"survives + wins ({int((win & survivors).sum())})")
    axR.plot([], [], color=AI2030_RED, lw=2, label=f"survives, loses ({int((~win & survivors).sum())})")
    axR.plot([], [], color=AI2030_GRID, lw=2, alpha=.6,
             label=f"detected, removed ({int(detected.sum())})  x = caught")
    axR.legend(loc="upper left", fontsize=7.5, framealpha=.9)
    axR.set_xlim(x_lo, x_hi); axR.set_ylim(-5, 110)
    axR.set_xlabel("year"); axR.set_ylabel("percent of the way from pause to ASI")
    axR.set_title(f"winnowed by detection · win rate {base*100:.0f}% → {after_ev*100:.0f}% "
                  f"(expected) · {after_thin*100:.0f}% (this draw)", fontsize=10)
    sz = f"median {smed:.0f} MW" if smed is not None else "per-rollout sized"
    fig.suptitle(f"Covert detection winnowing — {sz} projects", fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "detection_winnow.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def silver_bullet_curve(ds, outdir, doc=False):
    """Illustrate the per-rollout silver-bullet detection sigmoid and how it composes
    with facility frailty. Left: P(silver-bullet detection by t) — the family of
    per-rollout ceiling-logistics (median + p10–p90 band, the spread coming from each
    rollout's legal used->TED-AI time), with the two fixed anchor probabilities plotted at
    the median anchor times and the ceiling. Right: survival decomposition (silver-bullet
    vs frailty vs their independent product) at the median covert size. Returns None when
    no silver bullet is configured or no rollout reaches TED-AI.

    ``doc=True`` renders the document-embed variant: compact geometry, fonts sized to
    stay legible at text width (~6.5 in), shortened labels, higher dpi."""
    F = (dict(w=11.4, h=4.5, tick=11, label=12.5, title=13, sup=14.5, ann=9.5,
              leg=10, lw=2.6, lwm=3.0, ms=8, dpi=200) if doc else
         dict(w=13.5, h=5.4, tick=9, label=10, title=9.5, sup=13, ann=8,
              leg=8, lw=2.0, lwm=2.4, ms=6, dpi=140))
    R = ds["records"]; cfg = ds.get("config") or {}
    sbc = cfg.get("detection_silver_bullet")
    if not sbc:
        return None
    p0, p1, C, delta = (sbc["p_mid_pause_to_ted"], sbc["p_used_ted_plus_d"],
                        sbc["ceiling"], sbc["delta_years"])

    # Per-rollout anchor TIMES (years after pause): t0 = midpoint(pause-end, used->TED),
    # t1 = used->TED + Δ. Rollouts whose legal never reaches TED-AI have no silver bullet.
    t0s, t1s = [], []
    n_inert = 0
    for r_ in R:
        ut, py = r_.get("legal_used_ttt"), r_.get("pause_year")
        sp = r_.get("covert_params", {}).get("setup_period_years")
        if ut is None or py is None or sp is None:
            n_inert += 1
            continue
        t_used = ut - py
        t0s.append(0.5 * (sp + t_used)); t1s.append(t_used + delta)
    if not t0s:
        return None
    t0s, t1s = np.array(t0s, float), np.array(t1s, float)

    tg = np.linspace(0.0, float(max(15.0, np.percentile(t1s, 90) + 5.0)), 300)
    curves = np.array([1.0 - detection.g_no_silver_bullet(
        detection.build_silver_bullet(p0, a, p1, b, C), tg) for a, b in zip(t0s, t1s)])
    q10, q50, q90 = np.percentile(curves, [10, 50, 90], axis=0)
    t0m, t1m = float(np.median(t0s)), float(np.median(t1s))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(F["w"], F["h"]))

    # --- left: silver-bullet detection CDF (per-rollout family) ---------------------
    axL.fill_between(tg, q10, q90, color=AI2030_AMBER, alpha=.18, lw=0, label="per-rollout p10–p90")
    axL.plot(tg, q50, color=AI2030_NAVY, lw=F["lwm"], label="median")
    axL.axhline(C, color=AI2030_RED, ls="--", lw=1.2)
    axL.text(tg[-1], C, f" ceiling {C:g}", color=AI2030_RED, fontsize=F["ann"],
             va="bottom", ha="right")
    axL.plot([t0m, t1m], [p0, p1], "o", color=AI2030_RED, ms=F["ms"], zorder=5)
    axL.annotate(f"midpoint · P={p0:g}", (t0m, p0), textcoords="offset points",
                 xytext=(6, 10), fontsize=F["ann"], color=AI2030_MUTED)
    axL.annotate(f"used→TED + Δ={delta:g}y · P={p1:g}", (t1m, p1), textcoords="offset points",
                 xytext=((-8, -16) if doc else (6, -14)), ha=("right" if doc else "left"),
                 fontsize=F["ann"], color=AI2030_MUTED)
    axL.tick_params(labelsize=F["tick"])
    axL.set_xlabel("years after pause start", fontsize=F["label"])
    axL.set_ylabel("P(silver-bullet detection by t)", fontsize=F["label"])
    axL.set_xlim(0, tg[-1]); axL.set_ylim(0, 1)
    ttl = ("Silver-bullet emergence CDF" if doc else
           "Silver-bullet detection CDF (anchored on legal used→TED-AI)")
    if n_inert and not doc:
        ttl += f"\n{n_inert}/{len(R)} rollouts never reach TED-AI → no silver bullet (excluded)"
    axL.set_title(ttl, fontsize=F["title"]); axL.legend(fontsize=F["leg"], loc="upper left")

    # --- right: survival decomposition at the median covert size --------------------
    fits = detection.normalize_fits(cfg.get("detection_frailty_fit"))
    r_med = float(np.median([r_.get("detection_r", 0.0) for r_ in R]))
    sizes = [r_["covert_size_mw"] for r_ in R if r_.get("covert_size_mw") is not None]
    sites = [r_.get("covert_params", {}).get("optimal_site_mw") for r_ in R
             if r_.get("covert_params", {}).get("optimal_site_mw") is not None]
    if sizes:
        smed = float(np.median(sizes)); site_med = float(np.median(sites)) if sites else smed
        _dv = [r_.get("covert_params", {}).get("dormancy_years") for r_ in R]
        dorm_med = float(np.median([v for v in _dv if v is not None] or [0.0]))
        n, eff = detection.deploy(smed, site_med)
        p_frail = detection.interp_frailty(eff, fits)
        g = detection.g_no_silver_bullet(detection.build_silver_bullet(p0, t0m, p1, t1m, C), tg)
        # frailty clock starts at the end of dormancy; combined survival is 1 during it
        sf = detection.survival(p_frail, n, np.maximum(tg - dorm_med, 0.0), r_med)
        comb = detection.combined_survival(p_frail, n, tg, r_med,
                                           silver_bullet=detection.build_silver_bullet(p0, t0m, p1, t1m, C),
                                           dormancy_years=dorm_med)
        axR.plot(tg, g, color=AI2030_NAVY, lw=F["lw"],
                 label="no silver bullet yet" if doc else "silver-bullet survival  1−P_SB")
        axR.plot(tg, sf, color=AI2030_OLIVE, lw=F["lw"],
                 label="facility frailty" if doc else
                 "facility-frailty survival (clock starts post-dormancy)")
        axR.plot(tg, comb, color=AI2030_RED, lw=F["lwm"],
                 label="combined" if doc else "combined (1 during dormancy)")
        if dorm_med > 0:
            axR.axvspan(0, dorm_med, color=AI2030_GRID, alpha=.25, lw=0)
            axR.text(dorm_med / 2, 0.04, "dormant", ha="center", fontsize=F["ann"],
                     color=AI2030_MUTED)
        axR.set_xlim(0, tg[-1]); axR.set_ylim(0, 1)
        axR.tick_params(labelsize=F["tick"])
        axR.set_xlabel("years after pause start", fontsize=F["label"])
        axR.set_ylabel("P(undetected by t)", fontsize=F["label"])
        axR.set_title(f"Survival · {n:.0f} sites × {eff:.0f} MW" if doc else
                      f"Survival decomposition · median deployment {n:.1f} × {eff:.0f} MW sites "
                      f"({smed:.0f} MW total) · median dormancy {dorm_med:g}y",
                      fontsize=F["title"])
        axR.legend(fontsize=F["leg"], loc="lower left")
    else:
        axR.set_visible(False)

    fig.suptitle("Silver-bullet detection — emergence CDF and how it composes with facility frailty"
                 if doc else "Detection silver bullet", fontsize=F["sup"], y=.99)
    fig.tight_layout(rect=[0, 0, 1, .93 if doc else .95])
    name = "silver_bullet_curve_doc.png" if doc else "silver_bullet_curve.png"
    out = outdir / name; fig.savefig(out, dpi=F["dpi"]); plt.close(fig)
    return out.name


def detection_elicitation_fit(ds, outdir, doc=False):
    """Document the facility-detection model against its interpretable inputs: per
    elicited site size, the fitted gamma-frailty P(detected by t) curves for each
    elicited site count, with the elicited 3y/10y grid points overlaid. Facility
    channel only — no silver bullet or dormancy shift, matching the conditions the
    elicitation was stated under (GPUs acquired undetected). The curves DO include the
    dataset's jointly-fit shared slip-up hazard r (detection_r_fit; 0 for legacy
    datasets) since it is part of the fitted facility model. The curves past 10y show
    what the fit extrapolates beyond the elicited horizons. Returns None when the
    dataset predates detection integration.

    ``doc=True`` renders the document-embed variant: same content, compact geometry
    with fonts sized to stay legible when the image is placed at text width in a doc
    (~6.5 in), a one-line caption, and higher dpi."""
    cfg = ds.get("config") or {}
    grids = detection.normalize_grids(cfg.get("detection_grids"))
    fit_blk = cfg.get("detection_frailty_fit")
    if not grids or not fit_blk:
        return None
    fits = detection.normalize_fits(fit_blk)
    # Shared per-site slip-up hazard from the joint fit (fit_detection_model); 0.0 for
    # datasets built under the legacy per-size r=0 fit.
    r_fit = float(cfg.get("detection_r_fit") or 0.0)
    grids = sorted(grids, key=lambda g: g["size_mw"])

    # font/geometry sets: screen (run browser, full-size) vs doc (embedded at ~6.5in)
    F = (dict(w=3.75, h=3.9, tick=11, label=12.5, title=13, sup=14.5, ann=9, nlab=10.5,
              leg=10, cap=9.5, lw=2.4, ms=7, dpi=200) if doc else
         dict(w=4.6, h=4.6, tick=9, label=10, title=10, sup=13, ann=7.5, nlab=8,
              leg=8, cap=8, lw=2.0, ms=6, dpi=140))

    import matplotlib.colors as mcolors
    base = np.array(mcolors.to_rgb(AI2030_NAVY))
    def _shade(f):  # site count is ordered -> one hue, light (few sites) to dark (many)
        return tuple(1.0 - (1.0 - base) * (0.45 + 0.55 * f))

    tg = np.linspace(0.0, 30.0, 400)
    fig, axes = plt.subplots(1, len(grids), figsize=(F["w"] * len(grids), F["h"]), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, grids):
        size = g["size_mw"]
        p = detection.interp_frailty(size, fits)   # exact fitted params at an elicited size
        rows = sorted(g["rows"], key=lambda r_: r_[0])
        resid = []
        for i, (n, y3, y10) in enumerate(rows):
            col = _shade(i / max(len(rows) - 1, 1))
            det_t = 1.0 - detection.survival(p, n, tg, r_fit)
            ax.plot(tg, det_t, color=col, lw=F["lw"])
            ax.annotate(f"n={n:g}", (tg[-1], det_t[-1]), textcoords="offset points",
                        xytext=(3, -3), fontsize=F["nlab"], color=col, va="center")
            for t_e, y_e in ((3.0, y3), (10.0, y10)):
                if y_e is None:
                    continue
                ax.plot([t_e], [y_e], "o", color=col, ms=F["ms"], mec="white", mew=0.8, zorder=5)
                resid.append(abs((1.0 - float(detection.survival(p, n, t_e, r_fit))) - y_e))
        for t_e in (3.0, 10.0):
            ax.axvline(t_e, color=AI2030_GRID, ls=":", lw=1.0, zorder=0)
        mae = float(np.mean(resid)) if resid else float("nan")
        fit_line = (f"θ={p['thetaC']:.2g}  η={p['eta']:.3g}y  k={p['k']:.2g}" if doc else
                    f"fit: π={p['pi']:.2g}  θ={p['thetaC']:.2g}  η={p['eta']:.3g}y  k={p['k']:.2g}"
                    + (f"  r={r_fit:.2g}/yr/site" if r_fit else ""))
        ax.text(0.98, 0.04,
                fit_line + "\n" + f"mean |fit − elicited| = {mae * 100:.1f} pp",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=F["ann"],
                color=AI2030_MUTED)
        ax.set_xlim(0, tg[-1]); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=F["tick"])
        if not doc:
            ax.set_xlabel("years the facilities have existed", fontsize=F["label"])
        ax.set_title(f"{size:g} MW per site", fontsize=F["title"])
    if doc:   # one shared x-label — per-panel labels visually merge at doc width
        fig.supxlabel("years the facilities have existed", fontsize=F["label"], y=0.055)
    axes[0].set_ylabel("P(program detected by t)", fontsize=F["label"])
    axes[0].plot([], [], "o", color=_shade(1.0), ms=F["ms"], mec="white", mew=0.8,
                 label="elicited (3y / 10y)" if doc else "elicited P(detect by 3y / 10y)")
    axes[0].legend(fontsize=F["leg"], loc="upper left")
    fig.suptitle("Facility-detection model vs its elicitation — one curve per number of sites n"
                 if doc else
                 "Facility-detection model vs its elicitation — gamma-frailty fit per site size "
                 "(facility channel only)", fontsize=F["sup"], y=0.99)
    if doc:
        cap = ("Facility channel only (no silver bullet / dormancy). Fit includes the shared "
               f"per-site slip-up hazard r = {r_fit:.2g}/yr." if r_fit else
               "Facility channel only (no silver bullet / dormancy / operational hazard).")
        rect = [0, 0.05, 1, 0.94]
        fig.text(0.5, 0.012, cap, ha="center", fontsize=F["cap"], color=AI2030_MUTED)
    else:
        rect = [0, 0.09, 1, 0.95]
        fig.text(0.5, 0.02,
                 "One curve per number of near-identical sites n. The model is FIT to the dotted grid points (the\n"
                 "interpretable inputs): per-size frailty (π, θ, η, k), interpolated in log10(site MW) and clamped\n"
                 "outside, "
                 + (f"plus ONE shared per-site slip-up hazard r = {r_fit:.2g}/yr (included in the curves). "
                    if r_fit else "with no per-site operational hazard (r = 0). ")
                 + "Excludes the silver bullet and the dormancy shift.",
                 ha="center", fontsize=F["cap"], color=AI2030_MUTED)
    fig.tight_layout(rect=rect)
    name = "detection_elicitation_fit_doc.png" if doc else "detection_elicitation_fit.png"
    out = outdir / name; fig.savefig(out, dpi=F["dpi"]); plt.close(fig)
    return out.name


_MS_ORDER = [("AC", "AC"), ("SAR-level-experiment-selection-skill", "SAR"),
             ("TED-AI", "TED-AI"), ("STRAT-AI", "STRAT-AI"), ("ASI", "ASI")]


def median_2panel(ds, outdir):
    """Median-of-distributions scenario: capability E(t) + ETSP decomposition.
    Reads ds['median_scenario'] (computed once at build time at the median params)."""
    ms = ds.get("median_scenario")
    if not ms:
        return None
    d, lg, cv = ms["default"], ms["legal"], ms["covert"]
    dt, de = np.array(d["t"]), np.array(d["e"]); lt, le = np.array(lg["t"]), np.array(lg["e"])
    ct, ce = np.array(cv["t"]), np.array(cv["e"])
    tp = ms["pause_year"]; setup = ms["knobs"].get("setup_period_years", 0)
    # covert inactive window = dormancy (lying low) + construction (build); E frozen for both
    dorm = ((ms["knobs"].get("dormancy_years", 0) or 0)
            + (ms["knobs"].get("construction_years", 0) or 0))
    cttt, lttt = ms["covert_ttt"], ms["legal_ttt"]; tgt = ms["legal_target_e"]; ted = ms["ted_ai_e"]
    asi_e = (ms["milestones"].get("ASI", {}) or {}).get("e")
    e_top = (asi_e + 1) if asi_e else ted + 6; e_bot = float(np.interp(tp, dt, de)) - 1
    x_hi = min(dt[-1], max([v for v in (cttt, lttt) if v] + [tp]) + 4); x_lo = tp - 2
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for key, lab in _MS_ORDER:
        m = ms["milestones"].get(key)
        if m and e_bot <= m["e"] <= e_top:
            ax.axhline(m["e"], color=AI2030_GRID, lw=.9, ls=":"); ax.text(x_lo + .1, m["e"] + .05, lab, fontsize=8, color=AI2030_MUTED)
    ax.axhline(tgt, color=AI2030_NAVY, lw=.9, ls="--", alpha=.6)
    ax.text(x_hi, tgt + .05, f"TED-AI + tax ({ms['knobs'].get('alignment_tax_ooms',0):g})", fontsize=8,
            color=AI2030_NAVY, ha="right", va="bottom")
    if setup > 0:
        ax.axvspan(tp, tp + setup, color=AI2030_NAVY, alpha=.07)
    if dorm > 0:
        ax.axvspan(tp, tp + dorm, color=AI2030_RED, alpha=.06)
    ax.plot(dt, de, color=AI2030_MUTED, lw=1.2, alpha=.6, label="default (no deal)")
    ax.plot(lt, le, color=AI2030_NAVY, lw=2.4, label="legal — discovered (reaches TED-AI + tax)")
    if "used_e" in lg:  # deployed capability: discovered minus the alignment tax (reaches bare TED-AI)
        ax.plot(np.array(lg["used_t"]), np.array(lg["used_e"]), color=AI2030_NAVY, lw=1.5, ls="--",
                label="legal — used (deployed capability)")
    ax.plot(ct, ce, color=AI2030_RED, lw=2.4, label="covert project")
    ax.axvline(tp, color="#444", lw=1, alpha=.5); ax.text(tp, e_top, "  pause", fontsize=8, color="#444", va="top")
    for t, e, c in [(cttt, ted, AI2030_RED), (lttt, tgt, AI2030_NAVY)]:
        if t is not None and x_lo <= t <= x_hi:
            ax.plot([t], [e], "o", color=c, ms=6, zorder=5)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(e_bot, e_top)
    ax.set_ylabel("effective compute (rebased OOMs) — capability")
    win = ("covert WINS" if ms["covert_wins"] else "covert loses") + (
        f", lead {ms['lead_years']:+.1f}y" if ms["lead_years"] is not None else "")
    ax.set_title(f"Covert project — median scenario  ·  {win}", fontsize=11)
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=.15)
    # ETSP = TSP + SW
    for proj, color, t, e in [(d, AI2030_MUTED, dt, de), (lg, AI2030_NAVY, lt, le), (cv, AI2030_RED, ct, ce)]:
        tsp = np.array(proj["tsp"]); sw = np.array(proj["sw"]); etsp = tsp + sw
        ax2.fill_between(t, tsp, etsp, color=color, alpha=.10)
        ax2.plot(t, etsp, color=color, lw=2.0); ax2.plot(t, tsp, color=color, lw=1.1, ls=":")
    def _in(t, a):
        m = (t >= x_lo) & (t <= x_hi); return a[m]
    allv = np.concatenate([_in(lt, np.array(lg["tsp"]) + np.array(lg["sw"])),
                           _in(ct, np.array(cv["tsp"]) + np.array(cv["sw"]))])
    ax2.set_ylim(float(np.nanmin(allv)) - 1, float(np.nanpercentile(allv, 85)) + 3)
    ax2.set_xlim(x_lo, x_hi); ax2.set_xlabel("year"); ax2.set_ylabel("ETSP rate (OOMs FLOP/yr)")
    ax2.set_title("ETSP = TSP (hardware, dotted) + SW (software, shaded)", fontsize=9); ax2.grid(True, alpha=.15)
    fig.tight_layout()
    out = outdir / "median_2panel.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def median_uplift(ds, outdir, sit_years=None, out_name="median_uplift.png"):
    """Median-of-distributions scenario, single panel: AI software R&D uplift
    (ref-present-day, log scale) for the default (no deal), legal used (deployed) and
    covert trajectories. Milestone levels marked at their uplift (STRAT-AI omitted).
    Requires the uplift series in median_scenario (datasets built after the dense
    uplift-map fix); returns None on older datasets.

    ``sit_years``: if set, the legal used track sits at TED-AI for that many years after
    arrival and then goes FULL SPEED — it continues along the default (un-paused) run's
    trajectory time-shifted to start at its capability level, i.e. it resumes at the
    unconstrained model's pace for a project at bare TED-AI. (Approximation: uses the
    default run's compute base at that capability level, not the legal's larger 2040
    hardware, so if anything it understates the resumed speed.)"""
    ms = ds.get("median_scenario")
    if not ms or "uplift" not in (ms.get("default") or {}):
        return None
    d, lg, cv = ms["default"], ms["legal"], ms["covert"]
    dt, du = np.array(d["t"]), np.array(d["uplift"])
    ut, uu = np.array(lg["used_t"]), np.array(lg["used_uplift"])
    ct, cu = np.array(cv["t"]), np.array(cv["uplift"])
    tp = ms["pause_year"]; setup = ms["knobs"].get("setup_period_years", 0)
    cttt, lttt = ms["covert_ttt"], ms["legal_ttt"]
    used_label = "legal — used (deployed capability)"
    if sit_years is not None and lttt is not None:
        # Splice the full-speed continuation onto the used track: flat until
        # resume_t = arrival + sit_years, then the default run's curve time-shifted so its
        # bare-TED-AI point lands at resume_t.
        de = np.array(d["e"])
        t_star = float(np.interp(ms["ted_ai_e"], de, dt))   # default reaches bare TED-AI
        resume_t = float(lttt) + float(sit_years)
        # anchor the resume point at the EXACT TED-AI uplift (dense-map milestone value) —
        # interpolating the default's (E, uplift) samples here would put a chord artifact
        # (the sparse-grid scalloping) into the sit segment.
        ted_u = (ms["milestones"].get("TED-AI") or {}).get("uplift") or float(np.interp(lttt, ut, uu))
        keep = ut <= resume_t
        cont = dt > t_star
        ut = np.concatenate([ut[keep], [resume_t], resume_t + (dt[cont] - t_star)])
        uu = np.concatenate([uu[keep], [ted_u], du[cont]])
        used_label = f"legal — used (full speed after {sit_years:g}y at TED-AI)"
    x_hi = min(dt[-1], max([v for v in (cttt, lttt) if v] + [tp]) + 4); x_lo = tp - 2

    fig, ax = plt.subplots(figsize=(11, 6.2))
    def _in(t, a):
        m = (t >= x_lo) & (t <= x_hi); return a[m]
    vis = np.concatenate([_in(dt, du), _in(ut, uu), _in(ct, cu)])
    vis = vis[np.isfinite(vis) & (vis > 0)]
    y_lo = max(float(np.min(vis)) / 1.5, 1.0) if vis.size else 1.0
    # top: shortly above the ASI uplift level (tracks past it just run off the top)
    asi_u = (ms["milestones"].get("ASI") or {}).get("uplift")
    y_hi = asi_u * 5 if asi_u else (float(np.max(vis)) * 3 if vis.size else 1e6)
    for key, lab in _MS_ORDER:
        if key == "STRAT-AI":
            continue
        u = (ms["milestones"].get(key) or {}).get("uplift")
        if u and y_lo <= u <= y_hi:
            ax.axhline(u, color=AI2030_GRID, lw=.9, ls=":")
            ax.text(x_lo + .1, u * 1.05, lab, fontsize=12, color=AI2030_MUTED)
    if setup > 0:
        ax.axvspan(tp, tp + setup, color=AI2030_NAVY, alpha=.07)
    # after this year the lines turn dashed + faded (far-future portion)
    fade_t = 2040.0
    def _plot_split(t, v, color, lw, alpha, label):
        t = np.asarray(t, float); v = np.asarray(v, float)
        pre, post = t <= fade_t, t >= fade_t
        if pre.any():
            tb = np.append(t[pre], fade_t) if post.any() else t[pre]
            vb = np.append(v[pre], np.interp(fade_t, t, v)) if post.any() else v[pre]
            ax.plot(tb, vb, color=color, lw=lw, alpha=alpha, label=label)
        if post.any():
            ta = np.append(fade_t, t[post]) if pre.any() else t[post]
            va = np.append(np.interp(fade_t, t, v), v[post]) if pre.any() else v[post]
            ax.plot(ta, va, color=color, lw=lw, alpha=alpha * .4, ls="--",
                    label=None if pre.any() else label)
    _plot_split(dt, du, AI2030_MUTED, 1.2, .6, "default (no deal)")
    _plot_split(ut, uu, AI2030_NAVY, 2.4, 1.0, used_label)
    _plot_split(ct, cu, AI2030_RED, 2.4, 1.0, "covert project")
    ax.axvline(tp, color="#444", lw=1, alpha=.5)
    ax.text(tp, y_hi, "  pause", fontsize=12, color="#444", va="top")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("year", fontsize=14)
    ax.set_ylabel("AI software R&D uplift (× present-day)", fontsize=14)
    ax.tick_params(labelsize=13)
    # plain numbers, not 10^k / offset notation; whole calendar years on x
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    if cttt is not None:
        _MONTHS = ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"]
        month = _MONTHS[min(int((cttt % 1) * 12), 11)]
        ax.set_title(f"Covert project would reach TED-AI in {month} {int(cttt)}", fontsize=16)
    else:
        ax.set_title("Covert project would never reach TED-AI", fontsize=16)
    ax.legend(loc="lower right", fontsize=12); ax.grid(False)
    fig.tight_layout()
    out = outdir / out_name; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


_HW_COLOR = "#4f6d8c"      # hardware (training-compute scaling)
_SW_COLOR = "#b5651d"      # software — own R&D
_STOLEN_COLOR = "#7d97bf"  # software — stolen = f × the (navy) legal project's software gain
_WINDFALL_COLOR = "#e3b23c"  # software — one-time radical-transparency sharing (the windfall)
_MIN_SW_COLOR = "#6a9b8f"  # software — mandatory minimum floor (>= frac × TSP growth during scaling)
_EXT_COLOR = "#8155a6"     # software — post-scaling externalities (discovered after the legal arrives)


def _bartot(ax, xs, tot, y_hi):
    for x, tt in zip(xs, tot):
        if tt > .04 * y_hi:
            ax.text(x, tt + .015 * y_hi, f"{tt:.1f}", ha="center", va="bottom", fontsize=7, color=AI2030_MUTED)


def _seg_readout(ax, xs, segs, y_hi, thr=0.07):
    """Per-bar readout: label each stacked segment's OOMs (white text with a dark stroke,
    legible on any segment colour) and the total above the bar. `segs` is the list of
    component arrays in stacking order."""
    stroke = [_pe.withStroke(linewidth=1.6, foreground="#2b2b2b")]
    bottoms = np.zeros(len(xs))
    for vals in segs:
        vals = np.asarray(vals, float)
        for x, b, v in zip(xs, bottoms, vals):
            if v > thr * y_hi:
                ax.text(x, b + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=6.3,
                        color="white", zorder=6, path_effects=stroke)
        bottoms = bottoms + vals
    _bartot(ax, xs, bottoms, y_hi)


def _windfall_per_year(yrs, tp, setup, windfall):
    """OOMs of the one-time radical-transparency sharing (setup_sw_gain_ooms, ramped
    uniformly over [tp, tp+setup]) that land in each calendar year [Y, Y+1]."""
    if setup <= 0 or windfall <= 0:
        return np.zeros(len(yrs))
    return np.array([windfall * max(0.0, min(Y + 1, tp + setup) - max(Y, tp)) / setup for Y in yrs])


def _legal_floor_cum(ms, grid, used=False):
    """Cumulative mandatory-minimum SW guarantee (OOMs since the pause) interpolated onto
    ``grid``, read from the SIM-emitted series (legal sw_floor / used_sw_floor — accrued
    only while that track is scaling, frozen at arrival). The sim owns the phase knowledge;
    figures only consume it. Zeros when the dataset predates the field — rebuild
    (run.py --force) to get the mandatory-minimum split."""
    key, tkey = ("used_sw_floor", "used_t") if used else ("sw_floor", "t")
    fl = (ms.get("legal") or {}).get(key)
    if not fl:
        return np.zeros(len(grid))
    t = np.asarray(ms["legal"][tkey], float)
    return np.maximum(np.interp(grid, t, np.asarray(fl, float)), 0.0)


def oom_per_year(ds, outdir):
    """OOMs of hardware (ΔTSP) and software (ΔSW) added in each calendar year, for the
    legal consortium and the covert project, under the median-of-distributions scenario.

    The covert's software is split into OWN R&D vs STOLEN. The model defines the covert's
    research stock as a product, R_covert = R_own · (R_legal/R_legal_at_pause)^frac_stolen,
    so because software efficiency is logarithmic in R it decomposes with NO cross-term:
        SW_covert(t) − SW_covert(pause) = [SW_own gain] + frac_stolen · [SW_legal gain].
    i.e. the covert gets its own-research software plus a fraction `frac_stolen` of the
    legal project's software progress since the pause. (The ODE's `boost`·E_own term is
    just the product rule's own-stock term scaled by the legal factor — not a separate
    channel; we fold it into OWN. The stored sw_boost component is lumped with sw_indig.)
    Pure-JSON: ds['median_scenario'] stores tsp/sw and the sw_* components as OOM levels;
    per-year additions are the level diffed onto a calendar-year grid. The covert
    datacenter is frozen, so its hardware contribution is ~0."""
    ms = ds.get("median_scenario")
    if not ms:
        return None
    tp = ms["pause_year"]
    cttt, lttt = ms["covert_ttt"], ms["legal_ttt"]
    # AC line per panel: when THAT project's own effective compute crosses the AC level
    # (not the default-trajectory AC time). E is non-decreasing, so invert e(t) at AC_e.
    ac_e = (ms["milestones"].get("AC") or {}).get("e")

    def _cross(t, e, lvl):
        t, e = np.asarray(t, float), np.asarray(e, float)
        if lvl is None or not (e.min() <= lvl <= e.max()):
            return None
        return float(np.interp(lvl, e, t))
    ac_legal = _cross(ms["legal"]["t"], ms["legal"]["e"], ac_e)
    ac_covert = _cross(ms["covert"]["t"], ms["covert"]["e"], ac_e)
    # covert inactive window = dormancy (lying low) + construction (build); E frozen for both
    dorm = ((ms["knobs"].get("dormancy_years", 0) or 0)
            + (ms["knobs"].get("construction_years", 0) or 0))
    setup = ms["knobs"].get("setup_period_years", 0) or 0
    f = ms["knobs"].get("frac_stolen", 0) or 0
    cov = ms["covert"]
    has_split = all(k in cov for k in ("sw_stolen", "sw_boost", "sw_indig"))

    def grid(t_stop, tmax):
        hi = t_stop if (t_stop is not None and t_stop <= tmax) else tmax
        return np.arange(int(np.floor(tp)), int(np.ceil(hi)) + 1, dtype=float)

    def dlev(t, lvl, yrs):
        return np.diff(np.interp(yrs, np.asarray(t, float), np.asarray(lvl, float)))

    # one extra year of tail so the final (partial) year of progress is visible; when post-
    # scaling externalities are active, extend the legal panel to the covert's TED-AI so the
    # post-arrival discovered-software stream (which the covert steals) is visible.
    ext_rate = ms["knobs"].get("post_scaling_externalities_rate", 0) or 0
    leg_stop = (lttt + 1) if lttt else None
    if ext_rate > 0 and lttt and cttt and cttt > lttt:
        leg_stop = cttt + 1
    lt = ms["legal"]["t"]; lyrs = grid(leg_stop, np.asarray(lt, float)[-1])
    lyr, lhw, lsw = lyrs[:-1], dlev(lt, ms["legal"]["tsp"], lyrs), dlev(lt, ms["legal"]["sw"], lyrs)
    # post-scaling externalities = discovered SW the legal adds AFTER reaching TED-AI+tax
    if ext_rate > 0 and lttt:
        _lsw = np.asarray(ms["legal"]["sw"], float)
        _sw_arr = float(np.interp(lttt, lt, _lsw))
        l_ext = np.diff(np.maximum(np.interp(np.maximum(lyrs, lttt), lt, _lsw) - _sw_arr, 0.0))
    else:
        l_ext = np.zeros(len(lyr))
    ct = cov["t"]; cyrs = grid((cttt + 1) if cttt else None, np.asarray(ct, float)[-1])
    cyr, chw = cyrs[:-1], dlev(ct, cov["tsp"], cyrs)
    if has_split:
        c_steal = dlev(ct, cov["sw_stolen"], cyrs)                       # = f · legal SW gain
        c_own = dlev(ct, cov["sw_boost"], cyrs) + dlev(ct, cov["sw_indig"], cyrs)  # own R&D (boost folded in)
        csw_tot = c_steal + c_own
    else:
        csw_tot = dlev(ct, cov["sw"], cyrs); c_steal = c_own = None

    # bars straddle the one-year interval they represent: a bar for [Y, Y+1] is drawn
    # centred at the interval midpoint Y+0.5 on the continuous (decimal-year) time axis.
    lmid, cmid = lyr + .5, cyr + .5
    # one-time starting compute deficit: at the pause the covert's frozen datacenter is
    # only `frac` of the frontier's compute, so its TSP begins log10(frac) OOM BELOW what
    # the legal/default project had at that moment (legal hasn't ramped yet at the pause).
    cov_tsp0, leg_tsp0 = float(np.asarray(cov["tsp"], float)[0]), float(np.asarray(ms["legal"]["tsp"], float)[0])
    hw_deficit = cov_tsp0 - leg_tsp0  # negative
    frac_compute = ms["knobs"].get("total_h100e_frac")
    f_label = f"{frac_compute*100:.2f}%" if frac_compute else f"{10**hw_deficit*100:.2f}%"
    x_lo, x_hi = min(lyr.min(), cyr.min()) - .3, max(lyr.max(), cyr.max()) + 1.3
    y_hi = max(1e-6, np.nanmax([(lhw + lsw).max() if lhw.size else 0,
                                (chw + csw_tot).max() if chw.size else 0])) * 1.18
    y_lo = min(0.0, hw_deficit) * 1.12

    # Split each year's software into the one-time radical-transparency sharing (the
    # windfall, which lands in the pause-touching years) and the rest. The legal gets the
    # windfall directly; the covert gets frac_stolen of it (siphoned during the build).
    windfall = ms["knobs"].get("setup_sw_gain_ooms", 0) or 0
    wf_l = np.minimum(_windfall_per_year(lyr, tp, setup, windfall), lsw)
    lsw_rest = np.maximum(lsw - wf_l - l_ext, 0.0)      # pre-arrival tuned R&D (windfall & externalities split out)
    # mandatory-minimum SW floor, split out of the tuned R&D: the sim-emitted cumulative
    # guarantee (capped at the realized tuned SW) differenced back to per-year — mirrors
    # _legal_oom_tax_segments and the frontend.
    l_mand = np.diff(np.concatenate([[0.0],
             np.minimum(_legal_floor_cum(ms, lyrs[1:]), np.cumsum(lsw_rest))]))
    lsw_rest = np.maximum(lsw_rest - l_mand, 0.0)
    wf_c = _windfall_per_year(cyr, tp, setup, windfall)
    c_steal_ext = None
    if has_split:
        c_steal_wf = np.minimum(f * wf_c, c_steal)      # stolen share of the one-time sharing
        c_steal_on = c_steal - c_steal_wf               # stolen ongoing software
        # the covert keeps siphoning AFTER the legal arrives — that ongoing theft is the legal's
        # post-scaling externalities (all the legal produces post-arrival). Break it out.
        if ext_rate > 0 and lttt:
            _cstl = np.asarray(cov["sw_stolen"], float)
            _stl_arr = float(np.interp(lttt, ct, _cstl))
            c_steal_ext = np.clip(np.diff(np.maximum(np.interp(np.maximum(cyrs, lttt), ct, _cstl) - _stl_arr, 0.0)),
                                  0.0, c_steal_on)
            c_steal_on = c_steal_on - c_steal_ext

    fig, (axL, axC) = plt.subplots(2, 1, figsize=(11.5, 8.8), sharex=True)
    # legal: hardware + one-time-sharing software + tuned software
    axL.bar(lmid, lhw, width=.96, color=_HW_COLOR, label="hardware (ΔTSP, OOMs FLOP/yr)", zorder=3)
    axL.bar(lmid, wf_l, width=.96, bottom=lhw, color=_WINDFALL_COLOR,
            label="software — one-time sharing (windfall)", zorder=3)
    _l_base = lhw + wf_l
    _l_segs = [lhw, wf_l]
    if np.any(l_mand > 1e-9):
        axL.bar(lmid, l_mand, width=.96, bottom=_l_base, color=_MIN_SW_COLOR,
                label="software — mandatory minimum (≥ frac × TSP growth)", zorder=3)
        _l_base = _l_base + l_mand; _l_segs.append(l_mand)
    axL.bar(lmid, lsw_rest, width=.96, bottom=_l_base, color=_SW_COLOR,
            label="software — tuned R&D (ΔSW)", zorder=3)
    _l_base = _l_base + lsw_rest; _l_segs.append(lsw_rest)
    if np.any(l_ext > 1e-9):
        axL.bar(lmid, l_ext, width=.96, bottom=_l_base, color=_EXT_COLOR,
                label="software — post-scaling externalities", zorder=3)
        _l_segs.append(l_ext)
    _seg_readout(axL, lmid, _l_segs, y_hi)
    # covert: hardware + stolen one-time sharing + stolen ongoing + own R&D
    axC.bar(cmid, chw, width=.96, color=_HW_COLOR, label="hardware (ΔTSP) — frozen ≈ 0", zorder=3)
    if has_split:
        axC.bar(cmid, c_steal_wf, width=.96, bottom=chw, color=_WINDFALL_COLOR,
                label=f"software — stolen one-time sharing ({f:g} × windfall)", zorder=3)
        axC.bar(cmid, c_steal_on, width=.96, bottom=chw + c_steal_wf, color=_STOLEN_COLOR,
                label=f"software — stolen ongoing ({f:g} × legal's tuned SW)", zorder=3)
        _c_base = chw + c_steal_wf + c_steal_on
        _c_segs = [chw, c_steal_wf, c_steal_on]
        if c_steal_ext is not None and np.any(c_steal_ext > 1e-9):
            axC.bar(cmid, c_steal_ext, width=.96, bottom=_c_base, color=_EXT_COLOR,
                    label=f"software — stolen post-scaling externalities ({f:g} ×)", zorder=3)
            _c_base = _c_base + c_steal_ext; _c_segs.append(c_steal_ext)
        axC.bar(cmid, c_own, width=.96, bottom=_c_base, color=_SW_COLOR,
                label="software — own R&D", zorder=3)
        _c_segs.append(c_own)
        _seg_readout(axC, cmid, _c_segs, y_hi)
    else:
        axC.bar(cmid, csw_tot, width=.96, bottom=chw, color=_SW_COLOR, label="software (ΔSW, OOMs)", zorder=3)
        _seg_readout(axC, cmid, [chw, csw_tot], y_hi)
    # one-time starting hardware deficit (covert begins below the frontier's compute)
    if hw_deficit < -1e-6 and cmid.size:
        axC.bar(cmid[0], hw_deficit, width=.96, facecolor=_HW_COLOR, alpha=.45, hatch="////",
                edgecolor=_HW_COLOR, lw=0, zorder=3,
                label=f"starting compute deficit ({f_label} of frontier)")
        axC.text(cmid[0] + .55, hw_deficit, f"covert starts {hw_deficit:.1f} OOM\nbelow frontier compute",
                 fontsize=7.3, color=AI2030_MUTED, va="bottom", ha="left")

    for ax, name, arr, arr_lab, leg_loc, ac_x in [
            (axL, "Legal consortium", lttt, "TED-AI + tax", "upper right", ac_legal),
            (axC, "Covert project", cttt, "TED-AI", "upper left", ac_covert)]:
        ax.axvline(tp, color="#444", lw=1, alpha=.5); ax.text(tp, y_hi, " pause", fontsize=8, color="#444", va="top")
        if ac_x is not None and x_lo <= ac_x <= x_hi:
            ax.axvline(ac_x, color=AI2030_OLIVE, lw=1.1, ls=(0, (4, 2)), alpha=.85)
            ax.text(ac_x, y_hi * .9, " AC", fontsize=8, color=AI2030_OLIVE, va="top")
        if dorm > 0 and name.startswith("Covert"):
            ax.axvspan(tp, tp + dorm, color=AI2030_RED, alpha=.06, zorder=0)
            ax.text(tp + dorm / 2, y_hi * .42, f"{dorm:g}y dormant+build\n(theft only)", ha="center", va="center",
                    fontsize=7.5, color=AI2030_MUTED)
        if setup > 0 and name.startswith("Legal"):
            ax.axvspan(tp, tp + setup, color=AI2030_NAVY, alpha=.07, zorder=0)
        if arr is not None and x_lo <= arr <= x_hi:
            ax.axvline(arr, color=AI2030_GRID, ls="--", lw=1.2)
            ax.text(arr, y_hi * .96, f"{arr_lab} ({arr:.1f}) ", fontsize=8, color=AI2030_MUTED, va="top", ha="right")
        ax.set_ylim(y_lo, y_hi); ax.set_xlim(x_lo, x_hi)
        if y_lo < 0:
            ax.axhline(0, color=AI2030_MUTED, lw=.9, zorder=2)
        ax.set_ylabel("OOMs added that year"); ax.set_title(name, fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=.15); ax.legend(loc=leg_loc, fontsize=8.2)
    axC.set_xlabel("calendar year")
    fig.suptitle("Hardware vs software OOMs added per year — median scenario", fontsize=14, y=.987)
    fig.text(.5, .947, _spaghetti_caption(ds), ha="center", fontsize=8.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .937])
    out = outdir / "oom_per_year.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def oom_cumulative(ds, outdir):
    """Cumulative companion to oom_per_year: running totals of hardware and software
    OOMs added SINCE THE PAUSE (the integral of the per-year bars), as stacked areas
    over time. Covert software is split stolen (f × legal's gain) vs own R&D; the
    covert's one-time starting compute deficit is shown as a band below zero.
    Pure-JSON: the covert sw_* series are already cumulative-since-pause; legal hardware/
    software are levels, shown relative to their pause value."""
    ms = ds.get("median_scenario")
    if not ms:
        return None
    tp = ms["pause_year"]
    cttt, lttt = ms["covert_ttt"], ms["legal_ttt"]
    ac_e = (ms["milestones"].get("AC") or {}).get("e")
    # covert inactive window = dormancy (lying low) + construction (build); E frozen for both
    dorm = ((ms["knobs"].get("dormancy_years", 0) or 0)
            + (ms["knobs"].get("construction_years", 0) or 0))
    setup = ms["knobs"].get("setup_period_years", 0) or 0
    f = ms["knobs"].get("frac_stolen", 0) or 0
    cov = ms["covert"]
    has_split = all(k in cov for k in ("sw_stolen", "sw_boost", "sw_indig"))

    def _cross(t, e, lvl):
        t, e = np.asarray(t, float), np.asarray(e, float)
        if lvl is None or not (e.min() <= lvl <= e.max()):
            return None
        return float(np.interp(lvl, e, t))
    ac_legal = _cross(ms["legal"]["t"], ms["legal"]["e"], ac_e)
    ac_covert = _cross(cov["t"], cov["e"], ac_e)

    # legal cumulative (level minus pause value); covert sw_* already cumulative-since-pause
    lt = np.asarray(ms["legal"]["t"], float)
    l_hw = np.asarray(ms["legal"]["tsp"], float) - ms["legal"]["tsp"][0]
    l_sw = np.asarray(ms["legal"]["sw"], float) - ms["legal"]["sw"][0]
    # mandatory-minimum SW floor (sim-emitted cumulative guarantee), split out of total SW
    l_mand = np.minimum(_legal_floor_cum(ms, lt), l_sw)
    l_sw_rest = np.maximum(l_sw - l_mand, 0.0)
    ct = np.asarray(cov["t"], float)
    c_hw = np.asarray(cov["tsp"], float) - cov["tsp"][0]
    if has_split:
        c_steal = np.asarray(cov["sw_stolen"], float)
        c_own = np.asarray(cov["sw_boost"], float) + np.asarray(cov["sw_indig"], float)
    else:
        c_steal, c_own = None, np.asarray(cov["sw"], float) - cov["sw"][0]
    hw_deficit = float(cov["tsp"][0]) - float(ms["legal"]["tsp"][0])
    frac_compute = ms["knobs"].get("total_h100e_frac")
    f_label = f"{frac_compute*100:.2f}%" if frac_compute else f"{10**hw_deficit*100:.2f}%"

    x_lo, x_hi = tp - .5, max((cttt or ct[-1]), (lttt or lt[-1])) + 1.5
    # Draw BOTH tracks across the full shared window rather than truncating each at its own
    # arrival + 1y, so the legal's post-arrival externalities (software keeps accruing at the
    # externalities rate; capped-legal hardware keeps following the real series) stay visible
    # until the covert's end.
    lmask = lt <= x_hi
    cmask = ct <= x_hi
    c_top = c_hw + (c_steal if c_steal is not None else 0) + c_own
    y_hi = max(1e-6, max((l_hw + l_sw)[lmask].max() if lmask.any() else 0,
                         (hw_deficit + c_top)[cmask].max() if cmask.any() else 0)) * 1.10
    y_lo = min(0.0, hw_deficit) * 1.12

    fig, (axL, axC) = plt.subplots(2, 1, figsize=(11.5, 8.8), sharex=True)
    # Both panels share the baseline 0 = the frontier's compute (ETSP) level at the pause.
    # The legal starts there and accumulates up; the covert starts BELOW it by its frozen
    # compute deficit, so its cumulation baselines at hw_deficit (≈ −1.9) and climbs.
    if np.any(l_mand > 1e-9):
        axL.stackplot(lt[lmask], l_hw[lmask], l_mand[lmask], l_sw_rest[lmask],
                      colors=[_HW_COLOR, _MIN_SW_COLOR, _SW_COLOR],
                      labels=["hardware (cumulative TSP gain)",
                              "software — mandatory minimum (≥ frac × TSP growth)",
                              "software — tuned (cumulative SW gain)"], alpha=.9)
    else:
        axL.stackplot(lt[lmask], l_hw[lmask], l_sw[lmask], colors=[_HW_COLOR, _SW_COLOR],
                      labels=["hardware (cumulative TSP gain)", "software (cumulative SW gain)"], alpha=.9)
    cx = ct[cmask]; y0 = np.full(cx.shape, hw_deficit)
    cov_layers = [(c_hw[cmask], _HW_COLOR, "hardware (TSP gain) — frozen ≈ 0")]
    if has_split:
        cov_layers += [(c_steal[cmask], _STOLEN_COLOR, f"software — stolen ({f:g} × legal's gain)"),
                       (c_own[cmask], _SW_COLOR, "software — own R&D")]
    else:
        cov_layers += [(c_own[cmask], _SW_COLOR, "software (cumulative SW gain)")]
    for vals, color, lab in cov_layers:
        y1 = y0 + vals
        axC.fill_between(cx, y0, y1, color=color, alpha=.9, label=lab); y0 = y1
    if hw_deficit < -1e-6:
        axC.axhline(hw_deficit, color=_HW_COLOR, ls=":", lw=1, alpha=.7)
        axC.text(x_lo + .2, hw_deficit, f" covert starts {hw_deficit:.1f} OOM below frontier ({f_label} of its compute)",
                 fontsize=7.3, color=AI2030_MUTED, va="bottom")
    axC.text(x_hi, 0, "frontier compute at pause ", fontsize=7, color=AI2030_MUTED, ha="right", va="bottom")

    for ax, name, arr, arr_lab, ac_x in [(axL, "Legal consortium", lttt, "TED-AI + tax", ac_legal),
                                         (axC, "Covert project", cttt, "TED-AI", ac_covert)]:
        ax.axvline(tp, color="#444", lw=1, alpha=.5); ax.text(tp, y_hi, " pause", fontsize=8, color="#444", va="top")
        if ac_x is not None and x_lo <= ac_x <= x_hi:
            ax.axvline(ac_x, color=AI2030_OLIVE, lw=1.1, ls=(0, (4, 2)), alpha=.85)
            ax.text(ac_x, y_hi * .9, " AC", fontsize=8, color=AI2030_OLIVE, va="top")
        if dorm > 0 and name.startswith("Covert"):
            ax.axvspan(tp, tp + dorm, color=AI2030_RED, alpha=.06, zorder=0)
        if setup > 0 and name.startswith("Legal"):
            ax.axvspan(tp, tp + setup, color=AI2030_NAVY, alpha=.07, zorder=0)
        if arr is not None and x_lo <= arr <= x_hi:
            ax.axvline(arr, color=AI2030_GRID, ls="--", lw=1.2)
            ax.text(arr, y_hi * .96, f"{arr_lab} ({arr:.1f}) ", fontsize=8, color=AI2030_MUTED, va="top", ha="right")
        ax.set_ylim(y_lo, y_hi); ax.set_xlim(x_lo, x_hi)
        if y_lo < 0:
            ax.axhline(0, color=AI2030_MUTED, lw=.9)
        ax.set_ylabel("cumulative OOMs vs frontier at pause"); ax.set_title(name, fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=.15); ax.legend(loc="upper left", fontsize=8.2)
    axC.set_xlabel("calendar year")
    fig.suptitle("Cumulative hardware + software OOMs vs frontier at pause — median scenario", fontsize=14, y=.987)
    fig.text(.5, .947, _spaghetti_caption(ds), ha="center", fontsize=8.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .937])
    out = outdir / "oom_cumulative.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def explorer(ds, outdir):
    """Interactive HTML explorer (self-contained). Reuses the dataset's pstats and
    downsampled trajectories; injects a JSON payload into explorer_template.html."""
    import json
    R = ds["records"]; PS = ds["pstats"]; cfg = ds["config"]; k = cfg["covert_knobs_fixed"]
    lead_yrs = k["legal_scaling_duration"]
    # sorted distributions for exact percentile of each varying param
    dists = {p: np.sort(np.array([r["model_params"][p] for r in R if p in r["model_params"]], float))
             for p in PS}
    def pct(p, v):
        d = dists[p]; return round(100.0 * float(np.searchsorted(d, v, "right")) / len(d), 1)
    def dn(a, m=60):
        a = np.asarray(a, float); i = np.unique(np.linspace(0, a.size - 1, min(m, a.size)).astype(int)) if a.size > 1 else [0]
        return [round(float(a[j]), 2) for j in i]
    recs = []
    for r in R:
        unu = sorted(((abs(pct(p, r["model_params"][p]) - 50), p, round(float(r["model_params"][p]), 4), pct(p, r["model_params"][p]))
                      for p in PS if p in r["model_params"]), reverse=True)[:5]
        slip = (r["legal_ttt"] - (r["pause_year"] + lead_yrs)) if r["legal_ttt"] else None
        recs.append({
            "id": r["index"], "win": r["covert_wins"], "ac": round(r["ac_time"], 2),
            "ted": (round(r["ted_ai_time"], 2) if r["ted_ai_time"] else None),
            "gap": (round(r["ac_to_ted_gap"], 2) if r.get("ac_to_ted_gap") is not None else None),
            "covt": (round(r["covert_ttt"], 2) if r["covert_ttt"] else None),
            "legt": (round(r["legal_ttt"], 2) if r["legal_ttt"] else None),
            "lead": (round(r["lead_years"], 2) if r["lead_years"] is not None else None),
            "slip": (round(slip, 2) if slip is not None else None),
            "tedp": (round(r["ted_ai_pct_to_asi"], 1) if r.get("ted_ai_pct_to_asi") is not None else None),
            "jump": round(r["ted_ai_e"] - r["e_pause"], 2),
            "hw": (round(r["legal_hw_rate_solved"], 3) if r.get("legal_hw_rate_solved") is not None else None),
            "sw": (round(r["sw_rate_solved"], 3) if r.get("sw_rate_solved") is not None else None),
            "ct": dn(r["covert_t"]), "cp": dn(r["covert_pct_to_asi"]),
            "lt": dn(r["legal_t"]), "lp": dn(r["legal_pct_to_asi"]),
            "unusual": [{"k": p, "v": v, "pct": pp} for _, p, v, pp in unu],
            "params": {p: round(float(r["model_params"][p]), 4) for p in PS if p in r["model_params"]},
        })
    ac = np.array([r["ac_time"] for r in R]); tedt = np.array([r["ted_ai_time"] for r in R if r["ted_ai_time"]])
    tp = np.array([r["ted_ai_pct_to_asi"] for r in R if r.get("ted_ai_pct_to_asi") is not None])
    payload = {
        "meta": {"n": len(recs), "legal_mode": cfg.get("legal_mode"),
                 "win_frac": round(float(np.mean([r["covert_wins"] for r in R])), 3),
                 "x_lo": round(float(np.median(ac)) - 3, 1), "x_hi": round(float(np.median(tedt)) + 22, 1),
                 "ted_pct_med": round(float(np.median(tp)), 1) if tp.size else 60.0, "knobs": k},
        "pstats": PS, "records": recs,
    }
    tpl = (Path(__file__).parent / "explorer_template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    out = outdir / "explorer.html"; out.write_text(html)
    return out.name


def detection_pct_at_catch(ds, outdir, sit_years=5.0):
    """Distribution of how far the covert project has progressed (% of the way from
    pause to ASI) at the moment it is detected.

    Detection time per rollout is the sampled frailty detection time, FLOORED by a
    backstop: once the legal project reaches its target (TED-AI+tax) and has sat for
    ``sit_years`` (now an aligned ASI), it finds the covert for sure. So

        t_detect = min(pause_year + t_det,  legal_ttt + sit_years)

    using the per-rollout sampled ``t_det`` (None => frailty never fires, only the
    backstop) and ``legal_ttt`` (None => legal never reaches its target, only frailty).
    The covert's %-to-ASI is read off its recorded trajectory at ``t_detect``. Values
    >= 100% mean the covert reached ASI BEFORE detection (it got away with it)."""
    R = ds["records"]
    if not R:
        return None
    if "t_det" not in R[0]:
        raise RuntimeError("dataset predates detection integration — rebuild with "
                           "`python -m plan_a_covert_proj.pipeline.run --force`")
    w = _wts(ds)
    tk = TRAJ["pct_asi"]
    pct, pw, never, via_backstop, via_backstop_w, no_asi = [], [], 0, 0, 0.0, 0
    for i, r in enumerate(R):
        cands = []
        td = r.get("t_det")
        if td is not None:
            cands.append(r["pause_year"] + td)                 # frailty-sampled catch
        backstop = (r["legal_ttt"] + sit_years) if r.get("legal_ttt") is not None else None
        if backstop is not None:
            cands.append(backstop)
        if not cands:
            never += 1; continue                               # neither channel ever fires
        t_detect = min(cands)
        ct = np.asarray(r["covert_t"], float); cp = np.asarray(r[f"covert_{tk}"], float)
        if ct.size == 0:
            never += 1; continue
        v = float(np.interp(t_detect, ct, cp))                 # interp clamps outside range
        if not np.isfinite(v):                                 # %-to-ASI undefined (no ASI milestone)
            no_asi += 1; continue
        wi = 1.0 if w is None else float(w[i])
        if backstop is not None and t_detect == backstop and (td is None or backstop <= r["pause_year"] + td):
            via_backstop += 1; via_backstop_w += wi
        pct.append(v); pw.append(wi)
    pct = np.array(pct); pw = None if w is None else np.array(pw)
    n = pct.size
    n_w = float(n) if w is None else float(np.sum(pw))         # weighted denominator for fractions

    fig, ax = plt.subplots(figsize=(11, 6.2))
    if n:
        n_esc = int((pct >= 100.0).sum())                      # reached ASI before being caught (raw count)
        esc_frac = float(_wmean(pct >= 100.0, pw))             # escaped mass (weighted)
        cdf100 = float(_wmean(pct < 100.0, pw))                # CDF value at x=100 (the rest escaped)
        order = np.argsort(pct, kind="mergesort"); xs = pct[order]
        ww = np.ones(n) if pw is None else pw[order]
        ys = np.cumsum(ww) / np.sum(ww)                        # (weighted) empirical CDF
        ax.step(np.concatenate([[0.0], xs]), np.concatenate([[0.0], ys]), where="post",
                color=AI2030_NAVY, lw=2.4, zorder=4)
        # escaped mass piles at >=100: draw it as a vertical jump at x=100 up to 1.0
        if n_esc:
            ax.plot([100, 100], [cdf100, 1.0], color=AI2030_RED, lw=2.6, zorder=5)
            ax.fill_between([0, 100], [cdf100, cdf100], [1, 1], color=AI2030_RED, alpha=.07, zorder=1)
            ax.annotate(f"reached ASI before detection: {n_esc} ({esc_frac*100:.0f}%)",
                        xy=(99, (cdf100 + 1) / 2), ha="right", va="center", color=AI2030_RED, fontsize=8.5)
        med = float(_wquantile(pct, pw, 0.5))                  # over ALL detected runs (weighted)
        if med <= 100:                                         # median crosshair on the CDF
            ax.plot([med, med], [0, .5], color=AI2030_MUTED, ls="--", lw=1.2, zorder=3)
            ax.plot([0, med], [.5, .5], color=AI2030_MUTED, ls="--", lw=1.2, zorder=3)
            ax.text(med + 1.5, .5, f"median {med:.0f}%", color=AI2030_MUTED, fontsize=9, va="center")
        stats = [f"detected projects: {n}",
                 f"reached ASI before detection: {n_esc} ({esc_frac*100:.0f}%)",
                 f"caught via sit-backstop: {via_backstop} ({via_backstop_w/n_w*100:.0f}%)"]
        if never:
            stats.append(f"never detected (no backstop): {never}")
        if no_asi:
            stats.append(f"excluded (no ASI milestone): {no_asi}")
        ax.text(.985, .035, "\n".join(stats), transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
                color=AI2030_MUTED, bbox=dict(boxstyle="round,pad=.4", fc="white", ec=AI2030_GRID, alpha=.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
    ax.set_xlabel("covert progress when detected (% of the way from pause to ASI)")
    ax.set_ylabel("cumulative fraction of detected covert projects (≤ x)")
    fig.suptitle(f"How far does the covert get before detection?  ·  sit_years = {sit_years:g}",
                 fontsize=15, y=.97)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "detection_pct_at_catch.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def covert_ted_cdf(ds, outdir, fname=None, subtitle=None, relative=False, logit=False, marks=None,
                   legal_field="legal_used_ttt"):
    """CDF of WHEN the covert project reaches TED-AI, under two detection regimes, normalized
    to ALL rollouts (so each curve plateaus at its reach-fraction):
      1. no detection                — every run that ever reaches TED-AI (covert_ttt).
      2. freeze at frailty detection — reaches only if it gets to TED-AI before the sampled
                                       frailty detection fires (pause_year + t_det); otherwise
                                       its capabilities freeze and it never arrives.
    relative=True puts the x-axis in YEARS SINCE THE LEGAL'S ARRIVAL (covert_ttt −
    legal_field; − = covert first), with a zero line at the legal's arrival. legal_field
    picks which legal crossing anchors the zero: legal_used_ttt (default — used/deployed
    capability reaching bare TED-AI), legal_discovered_ttt (discovered capability
    reaching TED-AI + alignment tax), or pause_year (the deal start — x becomes years of
    covert effort since the pause began)."""
    R = ds["records"]
    if not R or "t_det" not in R[0]:
        raise RuntimeError("dataset predates detection integration — rebuild with --force")
    discovered = legal_field == "legal_discovered_ttt"
    pause = legal_field == "pause_year"
    if relative and not any(r.get(legal_field) is not None for r in R):
        print(f"[covert_ted_cdf] no record carries {legal_field} — dataset predates it; skipping")
        return None
    w = _wts(ds)
    n = _wtot(R, w)   # CDF denominator: rollout count, or total weight when reweighted

    def frailty_cal(r):
        return (r["pause_year"] + r["t_det"]) if r.get("t_det") is not None else np.inf
    def xval(r):  # plotted quantity: relative-to-legal or absolute calendar year
        return (r["covert_ttt"] - r[legal_field]) if relative else r["covert_ttt"]
    reach_idx = [i for i, r in enumerate(R) if r["covert_ttt"] is not None
                 and (not relative or r.get(legal_field) is not None)]
    reach = [R[i] for i in reach_idx]
    nodet = np.array([xval(r) for r in reach], float)
    nodet_w = None if w is None else w[reach_idx]
    fmask = np.array([r["covert_ttt"] <= frailty_cal(r) for r in reach], bool) if reach else np.zeros(0, bool)
    freeze = nodet[fmask]
    freeze_w = None if w is None else nodet_w[fmask]
    if nodet.size == 0:
        return None
    x_lo = float(np.percentile(nodet, 1)) - (1.0 if relative else 0.5)   # viewport: unweighted
    x_hi = float(np.percentile(nodet, 99)) + 1.0
    if pause:
        x_lo = 0.0   # anchor the axis at the deal start so the zero line is in view
    grid = np.linspace(x_lo, x_hi, 600)
    def cdf(vals, wts):
        return _wcumcount(vals, wts, grid) / n

    fig, ax = plt.subplots(figsize=(11, 6.4))
    y_top = 0.997 if logit else 1.0
    if relative:
        ax.axvline(0, color=AI2030_NAVY, lw=1.6, zorder=2)
        ax.text(0, y_top, (" deal begins (pause starts)" if pause
                           else "legal discovered reaches TED-AI + tax " if discovered
                           else "legal reaches TED-AI "), rotation=90,
                ha="left" if pause else "right", va="top",
                fontsize=8.5, color=AI2030_NAVY)
    curves = [(nodet, nodet_w, AI2030_RED, "no detection"),
              (freeze, freeze_w, AI2030_AMBER, "freeze at detection")]
    for vals, wts, col, lab in curves:
        y = cdf(vals, wts)
        if logit:
            y = np.clip(y, 1e-3, 1 - 1e-3)   # keep off the 0/1 asymptotes
        frac = (vals.size / n) if w is None else ((float(np.sum(wts)) / n) if vals.size else 0.0)
        ax.plot(grid, y, color=col, lw=2.6, zorder=3,
                label=f"{lab} — reaches TED-AI {vals.size}/{len(R)} ({100*frac:.0f}%)")
        ax.axhline(float(frac), color=col, ls=":", lw=1, alpha=.5)
    # markers + value labels at chosen x positions (both curves)
    for mxk in (marks or []):
        if not (x_lo <= mxk <= x_hi):
            continue
        if mxk != 0:
            ax.axvline(mxk, color=AI2030_GRID, ls="--", lw=.9, alpha=.7, zorder=1)
        for vals, wts, col, _ in curves:
            v = float(_wcumcount(vals, wts, np.array([float(mxk)]))[0] / n)
            vp = min(max(v, 1e-3), 1 - 1e-3) if logit else v
            ax.plot([mxk], [vp], "o", color=col, ms=6, zorder=6)
            dy = 10 if col == AI2030_RED else -18
            ax.annotate(f"{v*100:.1f}%", (mxk, vp), textcoords="offset points",
                        xytext=(7, dy), fontsize=8.5, color=col, fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=ax.get_facecolor(),
                                  edgecolor="none", alpha=.85))
    ax.set_xlim(x_lo, x_hi)
    if logit:
        ax.set_yscale("logit"); ax.set_ylim(3e-3, y_top)
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel(("years after the deal begins (pause start) that the covert reaches TED-AI"
                   if pause else
                   "years the covert reaches TED-AI relative to the legal's discovered "
                   "TED-AI + tax crossing  (− = covert first)" if discovered else
                   "years the covert reaches TED-AI relative to the legal  (− = covert first)")
                  if relative else "calendar year the covert project reaches TED-AI")
    ax.set_ylabel("fraction of covert projects reached (CDF over all runs)")
    ax.legend(loc="center right", fontsize=9); ax.grid(True, alpha=.15)
    head = (("When does the covert reach TED-AI, after the deal begins? — detection CDF" if pause
             else "When does the covert reach TED-AI, relative to the legal? — detection CDF")
            if relative else "When does the covert project reach TED-AI? — detection CDF")
    fig.suptitle(head, fontsize=15, y=.97)
    ax.set_title(((subtitle + "  ·  ") if subtitle else "") + _cap_rw(ds),
                 fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / (fname or (f"covert_ted_cdf{'_relative' if relative else ''}"
                              f"{'_logit' if logit else ''}"
                              f"{'_discovered' if (relative and discovered) else ''}"
                              f"{'_pause' if (relative and pause) else ''}.png"))
    fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def covert_ted_cdf_acasi(ds, outdir, lo=0.5, hi=2.0):
    """covert_ted_cdf conditioned on the default-trajectory AC→ASI calendar gap ∈ [lo, hi]
    years (the fast-takeoff slice). Pure-JSON filter on the stored milestones; the CDF is
    then re-normalized over just the conditioned subset."""
    keep = [i for i, r in enumerate(ds["records"])
            if r.get("ac_time") is not None and _ms(r, "ASI", "time") is not None
            and lo <= (_ms(r, "ASI", "time") - r["ac_time"]) <= hi]
    if not keep:
        return None
    R = [ds["records"][i] for i in keep]
    sub = dict(ds); sub["records"] = R
    w = _wts(ds)
    if w is not None:
        sub["_weights"] = w[keep]   # keep weights aligned to the filtered records
    sub["summary"] = dict(ds["summary"], n_ok=len(R))
    return covert_ted_cdf(sub, outdir, fname=f"covert_ted_cdf_acASIgap_{lo:g}_{hi:g}.png",
                          subtitle=f"conditioned on default AC→ASI gap ∈ [{lo:g}, {hi:g}] yr "
                                   f"({len(R)}/{len(ds['records'])} runs)")


def default_tc_at_milestones(ds, outdir):
    """CDF, per capability milestone, of the DEFAULT (un-paused) project's cumulative
    training compute (log10 FLOP — the frontier run's integrated hardware-FLOP stock,
    NOT the TSP rate) at the moment each rollout's default trajectory reaches the
    milestone, normalized to ALL rollouts (each curve plateaus at its reach-fraction).

    The default project's TSP series is the shared input hardware series — identical
    across rollouts — so TC(t) is ONE deterministic curve, integrated here from the stored median-scenario
    default (t, tsp) with the same log-mean segment rule metrics_computation.py uses for
    `training_compute`. All spread comes from WHEN each rollout crosses the milestone."""
    d = (ds.get("median_scenario") or {}).get("default") or {}
    t = np.asarray(d.get("t") or [], float)
    tsp = np.asarray(d.get("tsp") or [], float)
    if t.size < 2:
        print("[default_tc_at_milestones] no median-scenario default tsp series; skipping")
        return None
    rate = np.power(10.0, tsp)
    dt = np.diff(t)
    g0 = np.log(10.0) * (tsp[1] - tsp[0]) / dt[0]
    T0 = rate[0] / g0 if g0 > 0 else rate[0]
    la, lb = np.log(rate[:-1]), np.log(rate[1:])
    logmean = np.where(np.abs(lb - la) < 1e-9, np.sqrt(rate[:-1] * rate[1:]),
                       (rate[1:] - rate[:-1]) / (lb - la))
    tc = np.log10(T0 + np.concatenate([[0.0], np.cumsum(dt * logmean)]))

    R = ds["records"]
    w = _wts(ds)
    n = _wtot(R, w)
    MS = [("AC", "AC"), ("SAR-level-experiment-selection-skill", "SAR"),
          ("TED-AI", "TED-AI"), ("ASI", "ASI")]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    all_vals = []
    for key, lab in MS:
        idx = [i for i, r in enumerate(R) if _ms(r, key, "time") is not None]
        if not idx:
            continue
        vals = np.interp(np.array([_ms(R[i], key, "time") for i in idx], float), t, tc)
        wts = None if w is None else w[idx]
        all_vals.append(vals)
        frac = (len(idx) / n) if w is None else (float(np.sum(wts)) / n)
        col = MILESTONE_COLORS.get(lab, AI2030_TEXT)
        grid = np.sort(vals)
        # weighted quartiles among reachers -> median in the legend, dots on the curve
        o = np.argsort(vals)
        cw = np.cumsum(np.ones(len(idx)) if wts is None else wts[o])
        q25, med, q75 = (float(np.interp(q * cw[-1], cw, vals[o])) for q in (.25, .5, .75))
        ax.plot(grid, _wcumcount(vals, wts, grid) / n, color=col, lw=2.6, zorder=3,
                label=f"{lab} — median {_sci10(med)}, IQR {_sci10(q25)}–{_sci10(q75)} "
                      f"· reached {100*frac:.0f}%")
        ax.plot([med], [.5 * frac], "o", color=col, ms=6, zorder=6)
        for q, y in ((q25, .25 * frac), (q75, .75 * frac)):
            ax.plot([q], [y], "o", color=col, ms=3.5, zorder=6, alpha=.7)
    if not all_vals:
        plt.close(fig)
        return None
    av = np.concatenate(all_vals)
    x_lo = float(np.percentile(av, .5)) - .3
    x_hi = float(np.percentile(av, 97)) + .3
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
    ax.set_xlabel("default project's cumulative training compute when the milestone is reached (FLOP)")
    ax.set_ylabel("fraction of rollouts (CDF over all runs)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=.15)
    fig.suptitle("Training compute at each milestone — default (un-paused) project", fontsize=15, y=.97)
    ax.set_title("cumulative frontier-run FLOP stock (not TSP); dots = median / 25/75% "
                 "among reachers\n" + _cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "default_tc_at_milestones.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def covert_tc_at_milestones(ds, outdir):
    """CDF, per capability milestone, of the COVERT project's cumulative training compute
    (log10 FLOP) when it reaches the milestone — the covert twin of default_tc_at_milestones.
    No detection: arrivals use the raw covert capability trajectory (covert_ttt convention).

    The covert's hardware is a FLAT stock (sim.make_covert_data: the training-mix share of
    total_h100e_used at the pause, held constant), and its capability is frozen through
    dormancy + construction — so cumulative training FLOP is simply rate × (t − online),
    with rate reconstructed per rollout from total_h100e_used + the shared input profile
    (exactly the make_covert_data split) and online = pause + dormancy + construction.
    Milestone crossings interpolate the stored covert_e trajectory against each rollout's
    milestone E thresholds (downsampled grid => crossings are ~0.1y coarse; TED-AI here
    matches the exact covert_ttt to that resolution)."""
    from plan_a_covert_proj.pipeline.tables import _resolve_input_csv   # lazy: avoids cycle
    from plan_a_covert_proj.sim import _LOG10_FPH, _log_interp
    from progress_model import load_time_series_data
    fp = _resolve_input_csv(ds)
    if fp is None:
        print("[covert_tc_at_milestones] input-data CSV not found; skipping")
        return None
    prof = load_time_series_data(str(fp))
    R = ds["records"]
    w = _wts(ds)
    n = _wtot(R, w)
    MS = [("AC", "AC"), ("SAR-level-experiment-selection-skill", "SAR"),
          ("TED-AI", "TED-AI"), ("ASI", "ASI")]

    def _covert_tc_rate(r):
        """log10(FLOP/yr) of the covert's training slice — mirrors make_covert_data."""
        h = r.get("total_h100e_used")
        tp = r.get("pause_year")
        if not h or tp is None:
            return None
        inf = _log_interp(tp, prof.time, prof.inference_H100e)
        exp_ = _log_interp(tp, prof.time, prof.experiment_H100e)
        tc = _log_interp(tp, prof.time, prof.training_H100e)
        tot = inf + exp_ + tc
        if tot <= 0 or tc <= 0:
            return None
        return np.log10(tc * float(h) / tot) + _LOG10_FPH

    def _crossing(r, level):
        t = np.asarray(r["covert_t"], float)
        e = np.asarray(r["covert_e"], float)
        above = np.where(e >= level)[0]
        if not len(above):
            return None
        i = int(above[0])
        if i == 0:
            return float(t[0])
        return float(t[i-1] + (level - e[i-1]) / (e[i] - e[i-1]) * (t[i] - t[i-1]))

    fig, ax = plt.subplots(figsize=(11, 6.4))
    all_vals, n_pre = [], 0
    for key, lab in MS:
        idx, vals = [], []
        for i, r in enumerate(R):
            e_m = _ms(r, key, "e")
            rate = _covert_tc_rate(r)
            if e_m is None or rate is None or not r.get("covert_t"):
                continue
            tx = _crossing(r, float(e_m))
            if tx is None:
                continue
            cp = r.get("covert_params") or {}
            t_on = (r["pause_year"] + float(cp.get("dormancy_years") or 0.0)
                    + float(cp.get("construction_years") or 0.0))
            if tx <= t_on:      # crossed on stolen software alone, before the facility ran
                n_pre += 1
                continue
            idx.append(i)
            vals.append(rate + np.log10(tx - t_on))
        if not idx:
            continue
        vals = np.asarray(vals, float)
        wts = None if w is None else w[idx]
        all_vals.append(vals)
        frac = (len(idx) / n) if w is None else (float(np.sum(wts)) / n)
        col = MILESTONE_COLORS.get(lab, AI2030_TEXT)
        grid = np.sort(vals)
        o = np.argsort(vals)
        cw = np.cumsum(np.ones(len(idx)) if wts is None else wts[o])
        q25, med, q75 = (float(np.interp(q * cw[-1], cw, vals[o])) for q in (.25, .5, .75))
        ax.plot(grid, _wcumcount(vals, wts, grid) / n, color=col, lw=2.6, zorder=3,
                label=f"{lab} — median {_sci10(med)}, IQR {_sci10(q25)}–{_sci10(q75)} "
                      f"· reached {100*frac:.0f}%")
        ax.plot([med], [.5 * frac], "o", color=col, ms=6, zorder=6)
        for q, y in ((q25, .25 * frac), (q75, .75 * frac)):
            ax.plot([q], [y], "o", color=col, ms=3.5, zorder=6, alpha=.7)
    if not all_vals:
        plt.close(fig)
        return None
    av = np.concatenate(all_vals)
    x_lo = float(np.percentile(av, .5)) - .3
    x_hi = float(np.percentile(av, 97)) + .3
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
    ax.set_xlabel("covert project's cumulative training compute when it reaches the milestone (FLOP)")
    ax.set_ylabel("fraction of rollouts (CDF over all runs)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=.15)
    fig.suptitle("Training compute at each milestone — covert project (no detection)",
                 fontsize=15, y=.97)
    ax.set_title("covert facility FLOP burned since coming online (flat stock; not TSP); "
                 "dots = median / 25/75% among reachers"
                 + (f"; {n_pre} crossings pre-online dropped" if n_pre else "")
                 + "\n" + _cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "covert_tc_at_milestones.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def leading_rd_at_pause(ds, outdir):
    """Distribution (over rollouts) of the LEADING COMPANY's R&D compute fleet at the moment
    the pause starts: training + internal-inference + experiment H100e, log-interpolated from
    the run's input-data CSV at each rollout's pause_year — the same supplement definition the
    compute-acquisition table uses (tables._leading_lab_rd). The fleet curve is shared input
    data, so all spread comes from WHEN each rollout's pause lands."""
    # Lazy import: tables.py imports helpers from this module, so a top-level import would cycle.
    from plan_a_covert_proj.pipeline.tables import _leading_lab_rd, _fmt_h100e
    R = ds["records"]
    w = _wts(ds)
    pauses = np.array([r["pause_year"] for r in R], float)
    arr, _med, fp = _leading_lab_rd(ds, pauses, w)
    if fp is None:
        print("[leading_rd_at_pause] input-data CSV not found — only the fallback constant "
              "anchor is available; skipping")
        return None
    logs = np.log10(arr)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    bins = np.linspace(float(np.min(logs)) - .02, float(np.max(logs)) + .02, 45)
    ax.hist(logs, bins=bins, weights=w, color=AI2030_NAVY, alpha=.85, edgecolor="none")
    o = np.argsort(logs)
    cw = np.cumsum(np.ones(len(R)) if w is None else w[o])
    y_top = ax.get_ylim()[1]
    for q, lab_dy in ((.10, .84), (.50, .95), (.90, .84)):
        v = float(np.interp(q * cw[-1], cw, logs[o]))
        ax.axvline(v, color=AI2030_RED, ls="--" if q != .5 else "-", lw=1.8 if q == .5 else 1.2)
        ax.annotate(f"{'p10' if q == .10 else 'median' if q == .5 else 'p90'}\n"
                    f"{_fmt_h100e(10**v)} H100e",
                    (v, y_top * lab_dy), ha="center", fontsize=9, color=AI2030_RED,
                    fontweight="bold" if q == .5 else "normal",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=ax.get_facecolor(),
                              edgecolor="none", alpha=.85))
    # ticks at round 1/2/5 H100e counts (the axis variable is log10(H100e))
    ticks = [np.log10(m * 10.0**k) for k in range(5, 10) for m in (1, 2, 5)]
    ax.xaxis.set_major_locator(mticker.FixedLocator(
        [v for v in ticks if bins[0] <= v <= bins[-1]]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_h100e(10**v)))
    ax.set_xlabel("leading company's R&D compute at pause start "
                  "(H100e — training + internal inference + experiment)")
    ax.set_ylabel("rollouts" if w is None else "total weight")
    ax.grid(True, alpha=.15)
    fig.suptitle("Leading company's R&D compute when the pause begins", fontsize=15, y=.97)
    ax.set_title(f"fleet curve from {Path(fp).name} (shared input); spread = pause-timing "
                 "spread\n" + _cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "leading_rd_at_pause.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def default_ted_time_dist(ds, outdir):
    """Distribution (over rollouts) of the calendar time from the PAUSE START until the
    DEFAULT (un-paused) project's trajectory reaches TED-AI (ted_ai_time − pause_year;
    run_default's crossing — no pause, no covert, no detection — measured from when the
    deal/pause begins)."""
    R = ds["records"]
    w = _wts(ds)
    idx = [i for i, r in enumerate(R)
           if r.get("ted_ai_time") is not None and r.get("pause_year") is not None]
    if not idx:
        return None
    vals = np.array([R[i]["ted_ai_time"] - R[i]["pause_year"] for i in idx], float)
    wts = None if w is None else w[idx]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    # log-spaced bins: the distribution is heavily right-skewed (fast worlds cross within
    # ~2y of the pause, slow worlds take decades)
    x_lo = max(float(np.percentile(vals, .5)) * .9, .05)   # outliers clip into the edge bins
    x_hi = float(np.percentile(vals, 99.5)) * 1.1
    bins = np.geomspace(x_lo, x_hi, 55)
    n_clip = int(np.sum(vals > x_hi))
    ax.hist(np.clip(vals, x_lo, x_hi), bins=bins, weights=wts, color=AI2030_NAVY, alpha=.85,
            edgecolor="none")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    o = np.argsort(vals)
    cw = np.cumsum(np.ones(len(idx)) if wts is None else wts[o])
    y_top = ax.get_ylim()[1]
    for q, lab_dy in ((.10, .84), (.50, .95), (.90, .84)):
        v = float(np.interp(q * cw[-1], cw, vals[o]))
        ax.axvline(v, color=AI2030_RED, ls="--" if q != .5 else "-", lw=1.8 if q == .5 else 1.2)
        ax.annotate(f"{'p10' if q == .10 else 'median' if q == .5 else 'p90'}\n{v:.1f}y",
                    (v, y_top * lab_dy), ha="center", fontsize=9, color=AI2030_RED,
                    fontweight="bold" if q == .5 else "normal",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=ax.get_facecolor(),
                              edgecolor="none", alpha=.85))
    ax.set_xlabel("years from the pause start until the default project reaches TED-AI")
    ax.set_ylabel("rollouts" if w is None else "total weight")
    ax.grid(True, alpha=.15)
    fig.suptitle("Default (un-paused) project: time from pause start to TED-AI",
                 fontsize=15, y=.97)
    ax.set_title((f"{len(idx)}/{len(R)} rollouts reach TED-AI"
                  + (f"; {n_clip} beyond {x_hi:.0f}y clipped into the last bin" if n_clip else "")
                  + "\n" + _cap_rw(ds)), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "default_ted_time_dist.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def ted_cdf_compare(ds, outdir):
    """CDF (over ALL rollouts) of WHEN each capability milestone is reached, measured in
    YEARS AFTER THE PAUSE ENDS (pause_year + setup_period_years):
      - covert → TED-AI                  (covert_ttt; no detection)
      - legal USED → TED-AI              (legal_used_ttt; the deployed capability)
      - legal DISCOVERED → TED-AI + tax  (legal_discovered_ttt; full research output)
    Each curve is normalized to all runs, so it plateaus at its reach-fraction. The two
    legal curves usually coincide (both land at target_year); they separate only when the
    steeper discovered glidepath clamps and arrives later than the used one."""
    R = ds["records"]
    w = _wts(ds)
    n = _wtot(R, w)   # CDF denominator: rollout count, or total weight when reweighted
    if not R:
        return None

    def pause_end(r):
        sp = (r.get("covert_params") or {}).get("setup_period_years")
        py = r.get("pause_year")
        return (py + sp) if (py is not None and sp is not None) else None

    def offsets(key, fallback=None):
        xs, ws = [], []
        for i, r in enumerate(R):
            pe = pause_end(r)
            if pe is None:
                continue
            t = r.get(key)
            if t is None and fallback is not None:
                t = r.get(fallback)   # older datasets: legal_ttt ≈ discovered→TED+tax
            if t is not None and np.isfinite(t):
                xs.append(float(t) - pe)
                ws.append(1.0 if w is None else float(w[i]))
        x = np.array(xs, float); ww = np.array(ws, float)
        order = np.argsort(x, kind="mergesort")
        return x[order], (None if w is None else ww[order])

    series = [
        ("covert → TED-AI (no detection)", offsets("covert_ttt"), AI2030_RED),
        ("legal used → TED-AI", offsets("legal_used_ttt"), AI2030_NAVY),
        ("legal discovered → TED-AI + tax", offsets("legal_discovered_ttt", "legal_ttt"), AI2030_OLIVE),
    ]
    allx = (np.concatenate([a for _, (a, _), _ in series if a.size])
            if any(a.size for _, (a, _), _ in series) else np.array([]))
    if not allx.size:
        return None
    x_lo = min(0.0, float(allx.min()))
    x_hi = float(np.percentile(allx, 99)) + 1.0
    grid = np.linspace(x_lo, x_hi, 600)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for lab, (a, aw), col in series:
        if not a.size:
            continue
        frac = (a.size / n) if w is None else (float(np.sum(aw)) / n)
        ax.plot(grid, _wcumcount(a, aw, grid) / n, color=col, lw=2.6, zorder=3,
                label=f"{lab} — {a.size}/{len(R)} ({100 * frac:.0f}%)")
        ax.axhline(frac, color=col, ls=":", lw=1, alpha=.5)
    ax.axvline(0, color=AI2030_MUTED, lw=1, alpha=.6)
    ax.text(0, 1.005, "pause ends", fontsize=8, color=AI2030_MUTED, ha="center", va="bottom")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(0, 1)
    ax.set_xlabel("years after the pause ends")
    ax.set_ylabel("fraction reached (CDF over all runs)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=.15)
    fig.suptitle("When is TED-AI reached? — covert vs legal (used / discovered+tax)", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "ted_cdf_compare.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def covert_success_drivers(ds, outdir, sit_years=5.0, top=18):
    """Which sampled parameters most move the binary outcome: does the covert project
    reach TED-AI BEFORE it is caught — by either the sampled frailty detection
    (pause_year + t_det) OR the legal handoff backstop (legal_ttt + sit_years, an aligned
    ASI that sweeps it up)? Two complementary, signed views over every numeric parameter
    that VARIES (model + covert knobs), on the existing rollouts:
      (a) SHAP (TreeSHAP on a gradient-boosted classifier): mean|SHAP| = importance,
          interaction- and nonlinearity-aware; bar color = SHAP direction.
      (b) standardized logistic-regression coefficients (±90% bootstrap CI): the PARTIAL,
          signed linear effect of each param adjusting for all others.
    Same row = same parameter (both ordered by SHAP importance). Agreement between the
    panels = a robust driver; disagreement flags an interaction or a correlation artifact
    (the model params share a calibrated correlation matrix, so read clusters, not exact ranks)."""
    R = ds["records"]
    if not R or "t_det" not in R[0]:
        raise RuntimeError("dataset predates detection integration — rebuild with --force")

    def success(r):
        if r["covert_ttt"] is None:
            return 0
        fr = (r["pause_year"] + r["t_det"]) if r.get("t_det") is not None else np.inf
        bs = (r["legal_ttt"] + sit_years) if r.get("legal_ttt") is not None else np.inf
        return int(r["covert_ttt"] <= min(fr, bs))
    y = np.array([success(r) for r in R], int)
    base = float(y.mean())

    # Drop the INACTIVE compute-sizing knob: with compute_mode fixed, only one of
    # {total_h100e, total_h100e_frac} is ever used; the other is definitionally
    # irrelevant and would otherwise show spurious importance. (Keep both if the mode
    # itself varies across runs.)
    cmodes = {(r.get("covert_params") or {}).get("compute_mode") for r in R}
    drop = set()
    if cmodes == {"absolute"}:
        drop.add("total_h100e_frac")
    elif cmodes == {"frontier_relative"}:
        drop.add("total_h100e")

    # numeric, varying features from both param dicts (held-fixed/inactive knobs skipped)
    names, cols, seen = [], [], set()
    for src in ("model_params", "covert_params"):
        keys = set()
        for r in R:
            keys |= {k for k, v in (r.get(src) or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for k in sorted(keys):
            if k in drop:
                continue
            arr = np.array([(r.get(src) or {}).get(k, np.nan) for r in R], float)
            if np.isfinite(arr).sum() < 0.9 * len(R):
                continue
            if np.nanstd(arr) <= 1e-12 * (abs(np.nanmean(arr)) + 1e-12):
                continue  # held fixed -> not a driver
            arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
            lab = k if k not in seen else f"{k} [{src.split('_')[0]}]"
            seen.add(k); names.append(lab); cols.append(arr)
    if not cols or y.std() == 0:
        return None
    X = np.column_stack(cols)

    # (a) SHAP importance from a gradient-boosted classifier (TreeSHAP: exact, signed,
    # interaction-aware). mean|SHAP| = importance; direction = sign of corr(feature, its SHAP).
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    sv = np.asarray(shap.TreeExplainer(gb).shap_values(X))
    if sv.ndim == 3:                      # (n, d, classes) -> positive class
        sv = sv[..., -1]
    shap_imp = np.abs(sv).mean(0)
    shap_dir = np.array([np.sign(np.corrcoef(X[:, j], sv[:, j])[0, 1]) if sv[:, j].std() > 0 else 0.0
                         for j in range(X.shape[1])])

    # (b) standardized logistic coefficients with a bootstrap CI (partial, signed, linear).
    from sklearn.linear_model import LogisticRegression
    Xs = (X - X.mean(0)) / X.std(0)
    rng = np.random.default_rng(0); B, nrow = 400, len(y)
    boot = np.empty((B, X.shape[1]))
    for b in range(B):
        idx = rng.integers(0, nrow, nrow)
        if y[idx].sum() in (0, nrow):     # degenerate resample (essentially never)
            boot[b] = np.nan; continue
        boot[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[idx], y[idx]).coef_[0]
    co_med, co_lo, co_hi = np.nanpercentile(boot, [50, 5, 95], axis=0)

    order = np.argsort(shap_imp)[::-1][:top][::-1]  # ascending so the largest sits at the top
    yp = np.arange(len(order)); labs = [names[i] for i in order]
    col_a = [AI2030_OLIVE if shap_dir[i] >= 0 else AI2030_RED for i in order]
    col_b = [AI2030_OLIVE if co_med[i] >= 0 else AI2030_RED for i in order]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, max(4.5, 0.46 * len(order) + 1.8)), sharey=True)
    axA.barh(yp, shap_imp[order], color=col_a, alpha=.9, zorder=3)
    axA.set_yticks(yp); axA.set_yticklabels(labs, fontsize=8.5)
    axA.set_xlabel("mean |SHAP|  (log-odds contribution)")
    axA.set_title("(a) SHAP importance — interaction-aware", fontsize=10)
    axA.grid(True, axis="x", alpha=.15)
    cm = co_med[order]
    axB.barh(yp, cm, xerr=[cm - co_lo[order], co_hi[order] - cm], color=col_b, alpha=.9, zorder=3,
             error_kw=dict(ecolor=AI2030_MUTED, lw=1, capsize=2))
    axB.axvline(0, color=AI2030_MUTED, lw=.8)
    axB.set_xlabel("standardized logistic coef  (±90% bootstrap CI)")
    axB.set_title("(b) logistic — partial, signed (linear)", fontsize=10)
    axB.grid(True, axis="x", alpha=.15)
    axA.plot([], [], color=AI2030_OLIVE, lw=6, label="↑ parameter → more likely to beat detection/handoff")
    axA.plot([], [], color=AI2030_RED, lw=6, label="↑ parameter → less likely")
    axA.legend(loc="lower right", fontsize=8.2)
    fig.suptitle("What drives the covert reaching TED-AI before detection / handoff?", fontsize=14, y=.99)
    fig.text(.5, .945, f"SHAP (gradient boosting) + standardized logistic · success {int(y.sum())}/{len(R)} "
             f"({base*100:.0f}%) · sit_years={sit_years:g}  ·  " + _spaghetti_caption(ds),
             ha="center", fontsize=8.3, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .935])
    out = outdir / "covert_success_drivers.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def covert_shapley_effects(ds, outdir, sit_years=5.0, top_k=10, n_perm=240, k=15):
    """Variance-based SHAPLEY EFFECTS (Owen 2014) for the same binary outcome — the
    correlation-honest 'next rung' beyond SHAP/permutation. Each input's Shapley effect
    is its average marginal contribution to the output variance over all orderings of the
    inputs; unlike Sobol indices these stay meaningful (and sum to the explained variance)
    when inputs are CORRELATED, fairly splitting shared credit among correlated knobs.

    Estimated from the EXISTING rollouts (no re-run) via the random-permutation estimator
    with a leave-one-out kNN estimate of E[Y | X_S] (Broto-Bachoc-Depecker given-data
    approach). Restricted to the top_k features by |SHAP| (the rest are ~0 under both prior
    methods) so the kNN conditional-variance stays well-conditioned. Bars = each feature's
    share of the explained outcome variance; color = direction (Spearman sign)."""
    R = ds["records"]
    if not R or "t_det" not in R[0]:
        raise RuntimeError("dataset predates detection integration — rebuild with --force")

    def success(r):
        if r["covert_ttt"] is None:
            return 0
        fr = (r["pause_year"] + r["t_det"]) if r.get("t_det") is not None else np.inf
        bs = (r["legal_ttt"] + sit_years) if r.get("legal_ttt") is not None else np.inf
        return int(r["covert_ttt"] <= min(fr, bs))
    y = np.array([success(r) for r in R], float)
    if y.std() == 0:
        return None
    cmodes = {(r.get("covert_params") or {}).get("compute_mode") for r in R}
    drop = ({"total_h100e_frac"} if cmodes == {"absolute"}
            else {"total_h100e"} if cmodes == {"frontier_relative"} else set())
    names, cols, seen = [], [], set()
    for src in ("model_params", "covert_params"):
        keys = set()
        for r in R:
            keys |= {kk for kk, v in (r.get(src) or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for kk in sorted(keys):
            if kk in drop:
                continue
            arr = np.array([(r.get(src) or {}).get(kk, np.nan) for r in R], float)
            if np.isfinite(arr).sum() < 0.9 * len(R) or np.nanstd(arr) <= 1e-12 * (abs(np.nanmean(arr)) + 1e-12):
                continue
            arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
            names.append(kk if kk not in seen else f"{kk} [{src.split('_')[0]}]"); seen.add(kk); cols.append(arr)
    if not cols:
        return None
    X = np.column_stack(cols); names = np.array(names)
    Xs = (X - X.mean(0)) / X.std(0)

    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import NearestNeighbors
    sv = np.asarray(shap.TreeExplainer(GradientBoostingClassifier(random_state=0).fit(X, y)).shap_values(X))
    if sv.ndim == 3:
        sv = sv[..., -1]
    top = np.argsort(np.abs(sv).mean(0))[::-1][:min(top_k, X.shape[1])]
    Z, sub_names, d = Xs[:, top], names[top], len(top)

    def sgn(j):
        rx, ry = np.argsort(np.argsort(Z[:, j])), np.argsort(np.argsort(y))
        c = np.corrcoef(rx, ry)[0, 1]
        return np.sign(c) if np.isfinite(c) else 0.0
    signs = np.array([sgn(j) for j in range(d)])

    memo = {}
    def v(S):                       # Var(E[Y | X_S]) via leave-one-out kNN conditional mean
        key = frozenset(S)
        if key in memo:
            return memo[key]
        if not S:
            memo[key] = 0.0; return 0.0
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(y))).fit(Z[:, list(S)])
        idx = nn.kneighbors(Z[:, list(S)], return_distance=False)[:, 1:]   # drop self -> leave-one-out
        memo[key] = float(np.var(y[idx].mean(1))); return memo[key]
    rng = np.random.default_rng(0)
    phi = np.zeros(d)
    for _ in range(n_perm):
        S, prev = [], 0.0
        for j in rng.permutation(d):
            S.append(j); cur = v(S); phi[j] += cur - prev; prev = cur
    phi = np.clip(phi / n_perm, 0, None)
    frac = phi / phi.sum() if phi.sum() > 0 else phi
    explained = v(list(range(d))) / float(np.var(y))

    order = np.argsort(frac)[::-1][::-1]   # ascending for barh (largest on top)
    yp = np.arange(d)
    colors = [AI2030_OLIVE if signs[i] >= 0 else AI2030_RED for i in order]
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.5 * d + 1.8)))
    ax.barh(yp, 100 * frac[order], color=colors, alpha=.9, zorder=3)
    for i, yy in zip(order, yp):
        ax.text(100 * frac[i] + 0.5, yy, f"{100*frac[i]:.0f}%", va="center", fontsize=8, color=AI2030_MUTED)
    ax.set_yticks(yp); ax.set_yticklabels([sub_names[i] for i in order], fontsize=8.5)
    ax.set_xlabel("Shapley effect — share of explained outcome variance (%)")
    ax.plot([], [], color=AI2030_OLIVE, lw=6, label="↑ parameter → more likely to beat detection/handoff")
    ax.plot([], [], color=AI2030_RED, lw=6, label="↑ parameter → less likely")
    ax.legend(loc="lower right", fontsize=8.3); ax.grid(True, axis="x", alpha=.15)
    fig.suptitle("Variance-based Shapley effects — correlation-honest attribution", fontsize=13.5, y=.985)
    ax.set_title(f"top {d} features by |SHAP| · they explain ≈{explained*100:.0f}% of outcome variance "
                 f"· success {int(y.sum())}/{len(R)} · sit_years={sit_years:g}\n" + _spaghetti_caption(ds),
                 fontsize=8.3, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = outdir / "covert_shapley_effects.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def site_deployment(ds, outdir, doc=False):
    """How the covert deploys its total compute: the distribution of chosen per-site
    power (the sampled optimal_site_mw) and the resulting number of sites
    (n = total_covert_MW / site, floored at 1). For a fixed-site scenario the left
    panel is a single bar; n still varies with the (compute-driven) total power.

    ``doc=True`` renders the document-embed variant: compact geometry, fonts sized to
    stay legible at text width (~6.5 in), higher dpi."""
    F = (dict(w=11.4, h=4.5, tick=11, label=12.5, title=13, sup=14.5, ann=9.5,
              lw=2.8, dpi=200) if doc else
         dict(w=13.0, h=5.4, tick=9, label=10, title=11, sup=15, ann=8,
              lw=2.4, dpi=140))
    R = ds["records"]; w = _wts(ds)
    if not R:
        return None
    site_m = np.array([r_.get("covert_params", {}).get("optimal_site_mw") is not None for r_ in R])
    nsite_m = np.array([r_.get("n_sites") is not None for r_ in R])
    site = np.array([r_.get("covert_params", {}).get("optimal_site_mw") for r_ in R
                     if r_.get("covert_params", {}).get("optimal_site_mw") is not None], float)
    nsite = np.array([r_["n_sites"] for r_ in R if r_.get("n_sites") is not None], float)
    site_w = _wsub(w, site_m); nsite_w = _wsub(w, nsite_m)
    if site.size == 0 and nsite.size == 0:
        raise RuntimeError("dataset has no site-deployment fields — rebuild with "
                           "`python -m plan_a_covert_proj.pipeline.run --force`")
    # elicited grid size range (frailty support), for reference markers on the left panel
    grids = (ds.get("config") or {}).get("detection_grids") or []
    gsizes = sorted(float(g["size_mw"]) for g in grids) if grids else []

    def _cdf(ax, v, vw, color):
        order = np.argsort(v, kind="mergesort"); xs = v[order]
        ww = np.ones(xs.size) if vw is None else vw[order]
        ys = np.cumsum(ww) / np.sum(ww)
        ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0.0], ys]), where="post",
                color=color, lw=F["lw"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of runs ≤ x" if doc else "cumulative fraction of runs (≤ x)",
                      fontsize=F["label"])
        ax.tick_params(labelsize=F["tick"])
        return float(_wquantile(v, vw, 0.5))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(F["w"], F["h"]))

    # --- left: per-site power chosen (empirical CDF, log-x) ------------------------
    if site.size:
        med = _cdf(axL, site, site_w, AI2030_NAVY)
        axL.set_xscale("log")
        axL.plot([med, med], [0, .5], color=AI2030_MUTED, ls="--", lw=1.2)
        axL.plot([axL.get_xlim()[0], med], [.5, .5], color=AI2030_MUTED, ls="--", lw=1.2)
        axL.text(med, .52, f" median {med:.0f} MW", color=AI2030_MUTED, fontsize=F["ann"],
                 va="bottom")
        for g in gsizes:                                   # elicited grid anchors (frailty support)
            axL.axvline(g, color=AI2030_GRID, lw=.8, ls=":", zorder=0)
        if gsizes:
            axL.text(.015, .985, f"elicited grid: {', '.join(f'{g:g}' for g in gsizes)} MW (dotted)",
                     transform=axL.transAxes, fontsize=F["ann"], color=AI2030_MUTED, va="top")
    axL.set_xlabel("per-site power chosen (MW, log scale)", fontsize=F["label"])
    axL.set_title("Site size", fontsize=F["title"])

    # --- right: number of sites (n = total / site, floored at 1; empirical CDF) ----
    if nsite.size:
        med = _cdf(axR, nsite, nsite_w, AI2030_OLIVE)
        axR.set_xlim(left=1.0)
        axR.plot([med, med], [0, .5], color=AI2030_MUTED, ls="--", lw=1.2)
        axR.plot([1.0, med], [.5, .5], color=AI2030_MUTED, ls="--", lw=1.2)
        axR.text(med, .52, f" median {med:.2f}", color=AI2030_MUTED, fontsize=F["ann"],
                 va="bottom")
        frac1 = float(_wmean(nsite <= 1.0 + 1e-9, nsite_w))
        axR.text(.985, .04, f"{frac1*100:.0f}% at the n=1 floor (single site)\n"
                 f"more sites ⇒ more detection exposure", transform=axR.transAxes,
                 ha="right", va="bottom", fontsize=F["ann"], color=AI2030_MUTED)
    axR.set_xlabel("number of sites n" if doc else
                   "number of sites  n = total covert MW / per-site MW  (floored at 1)",
                   fontsize=F["label"])
    axR.set_title("Site count", fontsize=F["title"])

    fig.suptitle("Covert deployment — per-site power and site count" if doc else
                 "Covert deployment — site size and site count", fontsize=F["sup"], y=.98)
    if doc:
        fig.text(.5, .015, "n = total covert MW / per-site MW, floored at 1. "
                 "Dotted verticals: the elicited detection-grid sizes.",
                 ha="center", fontsize=F["ann"], color=AI2030_MUTED)
        rect = [0, 0.04, 1, .92]
    else:
        fig.text(.5, .93, _cap_rw(ds), ha="center", fontsize=8.5, color=AI2030_MUTED)
        rect = [0, 0, 1, .91]
    fig.tight_layout(rect=rect)
    name = "site_deployment_doc.png" if doc else "site_deployment.png"
    out = outdir / name; fig.savefig(out, dpi=F["dpi"]); plt.close(fig)
    return out.name


def legal_implied_compute(ds, outdir):
    """Implied R&D compute for the median-parameter legal run: the inference & experiment
    compute that reproduces the legal's THROTTLED research-effort glidepath (a common
    factor f on both, human labor held fixed), vs the actual frontier series. Top: compute
    (H100e, log-y), actual vs implied; bottom: the reduction factor f(t). NaN during the
    setup pause (effort = exogenous windfall); 0 once the legal sits (no R&D compute)."""
    ms = ds.get("median_scenario") or {}
    lg = ms.get("legal") or {}
    if "compute_factor" not in lg:
        raise RuntimeError("median_scenario lacks implied-compute fields — rebuild with "
                           "`python -m plan_a_covert_proj.pipeline.run --force`")
    t = np.asarray(lg["t"], float); ut = np.asarray(lg["used_t"], float)
    f = np.asarray(lg["compute_factor"], float)
    uf = np.asarray(lg["used_compute_factor"], float)
    # total R&D compute = inference + experiment (summed; not broken out)
    a_tot = np.asarray(lg["actual_inf"], float) + np.asarray(lg["actual_exp"], float)
    i_tot = np.asarray(lg["implied_inf"], float) + np.asarray(lg["implied_exp"], float)
    u_tot = np.asarray(lg["used_implied_inf"], float) + np.asarray(lg["used_implied_exp"], float)
    i_tot[~(f > 0)] = np.nan; u_tot[~(uf > 0)] = np.nan        # plot implied only where f>0
    pause, lttt = ms.get("pause_year"), ms.get("legal_ttt")
    x_lo = (pause - 0.5) if pause is not None else float(t.min())
    x_hi = (lttt + 1.5) if lttt else float(t.max())

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    axT.plot(t, a_tot, color=AI2030_GRID, lw=1.6, ls=":", label="actual frontier R&D (inf + exp)")
    axT.plot(t, i_tot, color=AI2030_NAVY, lw=2.6, label="implied — discovered (full research)")
    axT.plot(ut, u_tot, color=AI2030_OLIVE, lw=2.6, label="implied — used (deployed capability only)")
    axT.set_yscale("log"); axT.set_ylabel("R&D compute, inf + exp (H100e)")
    axT.legend(fontsize=8.5, loc="upper left"); axT.grid(True, alpha=.15)
    axT.set_title("Implied R&D compute (inference + experiment) — median-parameter legal run", fontsize=11)

    axB.plot(t, np.where(f > 0, f, np.nan), color=AI2030_NAVY, lw=2.2, label="discovered")
    axB.plot(ut, np.where(uf > 0, uf, np.nan), color=AI2030_OLIVE, lw=2.2, label="used")
    axB.axhline(1.0, color=AI2030_GRID, lw=1, ls="--")
    axB.text(x_lo, 1.0, " f = 1 (full frontier compute)", fontsize=7.5, color=AI2030_MUTED, va="bottom")
    if np.any(f > 0) or np.any(uf > 0):
        axB.set_yscale("log")
    axB.set_ylabel("reduction factor f"); axB.set_xlabel("year"); axB.grid(True, alpha=.15)
    axB.legend(fontsize=8, loc="upper right")

    for ax in (axT, axB):
        if pause is not None:
            ax.axvline(pause, color="#444", lw=1, alpha=.5)
            ax.text(pause, ax.get_ylim()[1], " pause", fontsize=8, color="#444", va="top")
        if lttt:
            ax.axvline(lttt, color=AI2030_GRID, ls="--", lw=1.2)
        ax.set_xlim(x_lo, x_hi)
    axB.text(.985, .04, "f<1 ⇒ the throttled glidepath effort needs less than the\nfrontier's inference+experiment compute",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=8, color=AI2030_MUTED)
    fig.suptitle("Compute implied by the legal research-effort glidepath", fontsize=15, y=.98)
    fig.text(.5, .93, _spaghetti_caption(ds), ha="center", fontsize=8.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .91])
    out = outdir / "legal_implied_compute.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def _legal_compute_series(ms):
    """(years, per-year H100e-yr, cumulative H100e-yr), each a triple
    [actual frontier, discovered-implied, used-implied], for the legal's R&D compute =
    inference + experiment integrated over time. implied = the throttled amount that
    yields the legal's effort (NaN in setup / 0 once it sits -> treated as 0). The
    discovered series is on the discovered grid; the used-implied on the used grid."""
    from scipy.integrate import cumulative_trapezoid
    lg = ms["legal"]

    def cum_on(t_key, inf_key, exp_key):
        t = np.asarray(lg[t_key], float)
        tot = np.nan_to_num(np.asarray(lg[inf_key], float)) + np.nan_to_num(np.asarray(lg[exp_key], float))
        return t, np.concatenate([[0.0], cumulative_trapezoid(tot, t)])
    t_d, cum_a = cum_on("t", "actual_inf", "actual_exp")          # actual frontier (discovered grid)
    _,   cum_i = cum_on("t", "implied_inf", "implied_exp")        # discovered-implied
    t_u, cum_u = cum_on("used_t", "used_implied_inf", "used_implied_exp")  # used-implied
    tp = ms["pause_year"]; lttt = ms["legal_ttt"]
    hi = (lttt + 1) if lttt else float(t_d[-1])
    yrs = np.arange(int(np.floor(tp)), int(np.ceil(hi)) + 1, dtype=float)
    A = np.interp(yrs, t_d, cum_a); I = np.interp(yrs, t_d, cum_i); U = np.interp(yrs, t_u, cum_u)
    return yrs, (np.diff(A), np.diff(I), np.diff(U)), (A, I, U)


def implied_compute_per_year(ds, outdir):
    """R&D compute (inference + experiment, H100e-years) the legal consumes each year:
    the IMPLIED amount that reproduces its throttled glidepath effort vs the ACTUAL
    frontier series. Median-parameter run; log-y (the two span many OOMs)."""
    ms = ds.get("median_scenario") or {}
    if not (ms.get("legal") or {}).get("implied_inf"):
        raise RuntimeError("median_scenario lacks implied-compute fields — rebuild (run.py --force)")
    yrs, (py_a, py_i, py_u), _ = _legal_compute_series(ms)
    mid = yrs[:-1] + 0.5
    pa = np.where(py_a > 0, py_a, np.nan)
    pi = np.where(py_i > 1, py_i, np.nan)            # implied ~0 (legal needs ~no compute) -> no bar
    pu = np.where(py_u > 1, py_u, np.nan)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(mid, pa, width=.96, color=AI2030_GRID, alpha=.7, label="actual frontier R&D compute")
    ax.bar(mid, pi, width=.62, color=AI2030_NAVY, label="implied — discovered (full research)")
    ax.bar(mid, pu, width=.34, color=AI2030_OLIVE, label="implied — used (deployed capability only)")
    ax.set_yscale("log")
    fin = np.concatenate([v[np.isfinite(v)] for v in (pa, pi, pu)])
    if fin.size:
        ax.set_ylim(max(1.0, float(fin.min()) / 3), float(fin.max()) * 5)
    ax.set_xlabel("year"); ax.set_ylabel("R&D compute used that year (H100e-years)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, axis="y", alpha=.15)
    if ms.get("legal_ttt"):
        ax.axvline(ms["legal_ttt"], color=AI2030_GRID, ls="--", lw=1.2)
    fig.suptitle("Legal R&D compute used per year — implied vs actual frontier", fontsize=14, y=.97)
    ax.set_title(_spaghetti_caption(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "implied_compute_per_year.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def implied_compute_cumulative(ds, outdir):
    """Cumulative R&D compute (inference + experiment, H100e-years) the legal consumes
    since the pause: IMPLIED (throttled) vs ACTUAL frontier. Median-parameter run."""
    ms = ds.get("median_scenario") or {}
    if not (ms.get("legal") or {}).get("implied_inf"):
        raise RuntimeError("median_scenario lacks implied-compute fields — rebuild (run.py --force)")
    yrs, _, (cum_a, cum_i, cum_u) = _legal_compute_series(ms)
    pos = np.concatenate([v[v > 0] for v in (cum_i, cum_u)])
    floor = max(1.0, float(pos.min()) / 5) if pos.size else 1.0
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(yrs, floor, np.maximum(cum_a, floor), color=AI2030_GRID, alpha=.5,
                    label="actual frontier (cumulative)")
    ax.fill_between(yrs, floor, np.maximum(cum_i, floor), color=AI2030_NAVY, alpha=.55,
                    label="implied — discovered (full research)")
    ax.fill_between(yrs, floor, np.maximum(cum_u, floor), color=AI2030_OLIVE, alpha=.65,
                    label="implied — used (deployed capability only)")
    fr_i = (cum_i[-1] / cum_a[-1]) if cum_a[-1] > 0 else float("nan")
    fr_u = (cum_u[-1] / cum_a[-1]) if cum_a[-1] > 0 else float("nan")
    ax.set_yscale("log"); ax.set_ylim(floor, float(cum_a.max()) * 5)
    ax.set_xlabel("year"); ax.set_ylabel("cumulative R&D compute since pause (H100e-years)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, axis="y", alpha=.15)
    if np.isfinite(fr_i):
        ax.text(.985, .92, f"by the end, of frontier R&D compute:\ndiscovered = {fr_i*100:.2f}%   used = {fr_u*100:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=AI2030_MUTED)
    if ms.get("legal_ttt"):
        ax.axvline(ms["legal_ttt"], color=AI2030_GRID, ls="--", lw=1.2)
    fig.suptitle("Legal cumulative R&D compute — implied vs actual frontier", fontsize=14, y=.97)
    ax.set_title(_spaghetti_caption(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "implied_compute_cumulative.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def _legal_oom_tax_segments(ms):
    """Per-year legal OOM segments: returns (yrs, hardware, windfall, mandatory_min,
    used_tuned, tax, externalities). discovered SW = windfall + mandatory_min + used_tuned +
    tax + post-scaling externalities; used (deployed) SW = windfall + mandatory_min +
    used_tuned; tax = discovered − used during scaling; the externalities are the discovered SW
    the legal keeps adding AFTER it reaches TED-AI+tax (broken out so they don't masquerade as
    alignment tax). mandatory_min is the guaranteed SW floor (>= min_sw_frac_of_tsp × TSP
    growth WHILE scaling — the sim emits it as used_sw_floor, frozen at arrival), split out
    of the tuned/used software. When externalities are active the year grid is extended to
    the covert's TED-AI so the full post-arrival stream is visible."""
    lg = ms["legal"]; tp = ms["pause_year"]; lttt = ms["legal_ttt"]
    setup = ms["knobs"].get("setup_period_years", 0) or 0
    windfall = ms["knobs"].get("setup_sw_gain_ooms", 0) or 0
    ext_rate = ms["knobs"].get("post_scaling_externalities_rate", 0) or 0
    cttt = ms.get("covert_ttt")
    lt = np.asarray(lg["t"], float); sw = np.asarray(lg["sw"], float)
    hi = (lttt + 1) if lttt else lt[-1]
    if ext_rate > 0 and lttt and cttt and cttt > lttt:   # extend to covert TED-AI to show externalities
        hi = min(cttt + 1, lt[-1])
    yrs = np.arange(int(np.floor(tp)), int(np.ceil(hi)) + 1, dtype=float)
    d = lambda lvl: np.diff(np.interp(yrs, lt, np.asarray(lvl, float)))      # discovered grid
    ut = np.asarray(lg["used_t"], float)                                     # used track has its OWN
    du = lambda lvl: np.diff(np.interp(yrs, ut, np.asarray(lvl, float)))     # grid (differs once the
    hw = d(lg["tsp"]); dsw = d(lg["sw"]); usw = du(lg["used_sw"])            # discovered track is extended)
    # post-scaling externalities = discovered SW added AFTER the legal reaches TED-AI+tax
    if ext_rate > 0 and lttt:
        sw_arr = float(np.interp(lttt, lt, sw))
        ext_cum = np.maximum(np.interp(np.maximum(yrs, lttt), lt, sw) - sw_arr, 0.0)
        ext = np.diff(ext_cum)
    else:
        ext = np.zeros(len(yrs) - 1)
    wf = np.minimum(_windfall_per_year(yrs[:-1], tp, setup, windfall), np.maximum(usw, 0))
    used_tuned = np.maximum(usw - wf, 0.0)
    tax = np.maximum(dsw - usw - ext, 0.0)               # alignment tax excludes the externalities
    hw = np.maximum(hw, 0.0)
    # Mandatory-minimum SW floor: the SIM emits the cumulative guarantee (used_sw_floor —
    # accrued only while scaling, frozen at arrival); clamp it to the realized tuned/used SW
    # and difference back to per-year so it splits cleanly out of used_tuned. (Matches the
    # frontend's api/covert_formatters.py legal_breakdown split.)
    ut_cum = np.cumsum(used_tuned)
    floor_cum = np.minimum(_legal_floor_cum(ms, yrs[1:], used=True), ut_cum)
    mand = np.diff(np.concatenate([[0.0], floor_cum]))    # per-year, >= 0, same length
    used_tuned = np.maximum(used_tuned - mand, 0.0)
    return yrs[:-1], hw, wf, mand, used_tuned, tax, ext


def oom_per_year_tax(ds, outdir):
    """Legal OOMs of progress added per year (median run), software split into the
    radical-transparency windfall, USED (deployed-to-capability) software, and the
    alignment TAX (discovered − used). Hardware (ΔTSP) at the base."""
    ms = ds.get("median_scenario") or {}
    if not (ms.get("legal") or {}).get("used_sw"):
        raise RuntimeError("median_scenario lacks used/discovered fields — rebuild (run.py --force)")
    yr, hw, wf, mand, used_tuned, tax, ext = _legal_oom_tax_segments(ms)
    mid = yr + .5
    fig, ax = plt.subplots(figsize=(11, 6))
    b = np.zeros(len(mid))
    segs = [(hw, _HW_COLOR, "hardware (ΔTSP)"),
            (wf, _WINDFALL_COLOR, "software — transparency windfall")]
    if np.any(mand > 1e-9):
        segs.append((mand, _MIN_SW_COLOR, "software — mandatory minimum (≥ frac × TSP growth)"))
    segs += [(used_tuned, _SW_COLOR, "software — used (deployed to capability)"),
             (tax, AI2030_RED, "software — alignment tax (discovered − used)")]
    if np.any(ext > 1e-9):
        segs.append((ext, _EXT_COLOR, "software — post-scaling externalities (discovered after TED-AI+tax)"))
    for vals, c, lab in segs:
        ax.bar(mid, vals, width=.96, bottom=b, color=c, label=lab, zorder=3); b = b + vals
    ax.set_xlabel("year"); ax.set_ylabel("OOMs added that year")
    ax.legend(loc="upper right", fontsize=8.5); ax.grid(True, axis="y", alpha=.15)
    if ms.get("legal_ttt"):
        ax.axvline(ms["legal_ttt"], color=AI2030_GRID, ls="--", lw=1.2)
    fig.suptitle("Legal progress per year — used capability vs alignment tax", fontsize=14, y=.97)
    ax.set_title(_spaghetti_caption(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "oom_per_year_tax.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def oom_cumulative_tax(ds, outdir):
    """Cumulative legal OOMs since the pause (median run), software split windfall / used
    (deployed) / alignment tax — the cumulative companion to oom_per_year_tax."""
    ms = ds.get("median_scenario") or {}
    if not (ms.get("legal") or {}).get("used_sw"):
        raise RuntimeError("median_scenario lacks used/discovered fields — rebuild (run.py --force)")
    yr, hw, wf, mand, used_tuned, tax, ext = _legal_oom_tax_segments(ms)
    x = np.concatenate([yr, [yr[-1] + 1]])               # right edge of the last interval
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
           (("hw", hw), ("wf", wf), ("mand", mand), ("used", used_tuned), ("tax", tax), ("ext", ext))}
    fig, ax = plt.subplots(figsize=(11, 6))
    base = np.zeros(len(x))
    layers = [("hw", _HW_COLOR, "hardware (cumulative ΔTSP)"),
              ("wf", _WINDFALL_COLOR, "software — transparency windfall")]
    if np.any(mand > 1e-9):
        layers.append(("mand", _MIN_SW_COLOR, "software — mandatory minimum (≥ frac × TSP growth)"))
    layers += [("used", _SW_COLOR, "software — used (deployed to capability)"),
               ("tax", AI2030_RED, "software — alignment tax (discovered − used)")]
    if np.any(ext > 1e-9):
        layers.append(("ext", _EXT_COLOR, "software — post-scaling externalities (discovered after TED-AI+tax)"))
    for key, c, lab in layers:
        top = base + cum[key]
        ax.fill_between(x, base, top, color=c, alpha=.9, label=lab); base = top
    ax.set_xlabel("year"); ax.set_ylabel("cumulative OOMs added since pause")
    ax.legend(loc="upper left", fontsize=8.5); ax.grid(True, axis="y", alpha=.15)
    if ms.get("legal_ttt"):
        ax.axvline(ms["legal_ttt"], color=AI2030_GRID, ls="--", lw=1.2)
    fig.suptitle("Legal cumulative progress — used capability vs alignment tax", fontsize=14, y=.97)
    ax.set_title(_spaghetti_caption(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "oom_cumulative_tax.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def bottom_up_compute(ds, outdir):
    """Bottom-up covert-compute estimate (covert-AI-projects supplement §2.3 Method #1).
    LEFT: the diverted fraction of WORLD AI-relevant compute, decomposed by acquisition source
    (smuggled-chip diversion / false decommissioning / concealed domestic production) plus the
    total — median bar with the 10-90% spread across rollouts. RIGHT: the resulting absolute
    covert H100e (each rollout's fraction x world compute at its pause), as a CDF with p10 /
    median / p90 marked. Returns None (skipped) unless this run used compute_mode=bottom_up_estimate."""
    R = ds.get("records") or []; w = _wts(ds)
    kmask = np.array([r.get("bottom_up_world_frac") is not None for r in R])
    keep = [r for r in R if r.get("bottom_up_world_frac") is not None]
    if not keep:
        return None  # not a bottom_up_estimate run
    kw = _wsub(w, kmask)
    col = lambda k: np.array([float(r.get(k) or 0.0) for r in keep], float)
    tot, dec, dom, smu = col("bottom_up_world_frac"), col("bottom_up_decommission_frac"), \
        col("bottom_up_domestic_frac"), col("bottom_up_smuggled_frac")
    h = col("total_h100e_used")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # --- Panel A: diverted fraction of world compute, by source + total ---
    rows = [("Total diverted", tot, AI2030_RED),
            ("Smuggled flow", smu, AI2030_NAVY),
            ("False decommissioning", dec, AI2030_OLIVE),
            ("Concealed domestic", dom, AI2030_AMBER)]
    ys = np.arange(len(rows))[::-1]
    xmax = 0.0
    for y, (lab, a, c) in zip(ys, rows):
        med, lo, hi = _wquantile(a, kw, 0.5) * 100, _wpct(a, kw, 10) * 100, _wpct(a, kw, 90) * 100
        axL.barh(y, med, color=c, alpha=.85, height=.62, zorder=3)
        axL.errorbar(med, y, xerr=[[max(med - lo, 0)], [max(hi - med, 0)]], fmt="none",
                     ecolor=AI2030_TEXT, capsize=4, lw=1.3, zorder=4)
        axL.text(hi + .04, y, f"med {med:.2f}%  [10-90: {lo:.2f}, {hi:.2f}]",
                 va="center", fontsize=8.5, color=AI2030_MUTED)
        xmax = max(xmax, hi)
    axL.set_yticks(ys); axL.set_yticklabels([r[0] for r in rows])
    axL.axvline(0.74, color=AI2030_MUTED, ls="--", lw=1.1, label="supplement Method-1 (0.74%)")
    axL.axvline(0.44, color=AI2030_MUTED, ls=":", lw=1.1, label="supplement ATC (0.44%)")
    axL.legend(loc="lower right", fontsize=8, framealpha=.9)
    axL.set_xlim(0, xmax * 1.45 + .1)
    axL.set_ylim(-.6, len(rows) - .4)
    axL.set_xlabel("diverted compute (% of world AI-relevant compute at the declaration)")
    axL.set_title("Where the covert compute comes from", fontsize=10, color=AI2030_MUTED)
    axL.grid(True, axis="x", alpha=.15)

    # --- Panel B: resulting absolute covert H100e (CDF, log-x) ---
    order = np.argsort(h, kind="mergesort"); hs = h[order]
    ww = np.ones(hs.size) if kw is None else kw[order]
    cdf = np.cumsum(ww) / np.sum(ww) * 100
    axR.plot(hs, cdf, color=AI2030_RED, lw=2.4, zorder=3)
    med, lo, hi = _wquantile(h, kw, 0.5), _wpct(h, kw, 10), _wpct(h, kw, 90)
    for x, c, lab in [(lo, AI2030_MUTED, f"p10 {lo/1e6:.2f}M"), (med, AI2030_NAVY, f"median {med/1e6:.2f}M"),
                      (hi, AI2030_MUTED, f"p90 {hi/1e6:.2f}M")]:
        axR.axvline(x, color=c, ls="--", lw=1.3, zorder=2)
        axR.text(x, 4, "  " + lab, rotation=90, va="bottom", ha="left", fontsize=8, color=c)
    axR.set_xscale("log"); axR.set_ylim(0, 100)
    axR.set_xlabel("covert project compute (H100e)")
    axR.set_ylabel("cumulative % of rollouts")
    axR.set_title("Resulting covert compute (per-source over the diversion window)",
                  fontsize=10, color=AI2030_MUTED)
    axR.grid(True, which="both", alpha=.15)

    fig.suptitle("Bottom-up covert-compute estimate"
                 + (("  ·  " + _rw_note(ds)) if _rw_note(ds) else ""), fontsize=14, y=.98)
    fig.text(.5, .005, f"{len(keep)} runs · covert-AI-projects supplement §2.3 Method #1 "
             f"(diversion params sampled per rollout; decommissioning off the start-of-window world "
             f"stock, smuggling + domestic off the window flow)",
             ha="center", fontsize=8.5, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, .03, 1, .95])
    out = outdir / "bottom_up_compute.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def bottom_up_power_calc(ds, outdir):
    """Doc figure for the covert-AI-projects supplement §3.3 ("How much datacenter capacity
    does the diverted compute require?"): the power calculation behind each chip source's
    facility W/H100e. Draws the US power-efficiency curve — IT W/H100e anchored at 2026.0
    (DGX GB200 SuperPOD) and improving power_efficiency_per_year x/yr — its facility version
    (x peak_pue) and the Chinese-domestic version (x domestic_efficiency_gap), then places
    every rollout's RECORDED per-source W/H100e at its proxy year: smuggled/domestic at the
    diversion-window midpoint + offset, decommissioned at the window start - age. A worked
    example at the median-pause window calls sim.bottom_up_facility_w_per_h100e (the
    production function, not a reimplementation). Curve parameters come from the run's
    bottom-up yaml, resolved via the dataset's covert_config — a HASHED run input, not a
    model output (any drift between that yaml and the run is visible as dots off the
    curves). Returns None (skipped) unless this run used compute_mode=bottom_up_estimate."""
    R = [r for r in (ds.get("records") or []) if r.get("bottom_up_smuggled_w") is not None]
    if not R:
        return None  # not a bottom_up_estimate run (or dataset predates per-source w fields)
    cfg = ds.get("config") or {}
    try:  # lazy: batch_rollout/sim are heavy imports and only this figure needs them
        from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict
        from plan_a_covert_proj.sim import bottom_up_facility_w_per_h100e
        cov_p = Path(cfg.get("covert_config") or "")
        cov_p = cov_p if cov_p.is_absolute() else REPO_ROOT / cov_p
        rel = (_load_config(str(cov_p)) or {}).get("bottom_up_compute_config")
        bu_p = Path(rel); bu_p = bu_p if bu_p.is_absolute() else REPO_ROOT / rel
        bu = _get_median_parameter_dict(
            (_load_config(str(bu_p)) or {}).get("bottom_up_compute_parameters") or {})
    except Exception as e:  # noqa: BLE001
        print(f"[bottom_up_power_calc] cannot resolve the run's bottom-up yaml ({e}); skipping")
        return None
    w26, rate, pue = bu["it_w_per_h100e_2026"], bu["power_efficiency_per_year"], bu["peak_pue"]
    gap, off, age = (bu["domestic_efficiency_gap"], bu["smuggled_proxy_offset_years"],
                     bu["decommissioned_age_years"])

    col = lambda k: np.array([float(r[k]) for r in R], float)
    pause, gapyr, dur = col("pause_year"), col("bottom_up_declaration_to_pause_years"), \
        col("bottom_up_diversion_duration")
    decl = pause - gapyr; start = decl - dur; mid = 0.5 * (start + decl)
    smu_x, dec_x = mid + off, start - age
    smu_w, dec_w, dom_w = col("bottom_up_smuggled_w"), col("bottom_up_decommission_w"), \
        col("bottom_up_domestic_w")

    # worked example at the median-pause window, via the production function
    w = _wts(ds)
    p_m = _wquantile(pause, w, 0.5)
    decl_m = p_m - float(np.median(gapyr)); start_m = decl_m - float(np.median(dur))
    mid_m = 0.5 * (start_m + decl_m)
    ex = bottom_up_facility_w_per_h100e(bu, start_m, decl_m)

    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    xlo = min(2025.8, dec_x.min() - 0.2); xhi = smu_x.max() + 0.5
    xs = np.linspace(xlo, xhi, 400)
    us_it = w26 / rate ** (xs - 2026.0)
    ax.plot(xs, us_it, color=AI2030_NAVY, lw=1.3, ls="--", zorder=2,
            label="US-designed — IT power")
    ax.plot(xs, us_it * pue, color=AI2030_NAVY, lw=2.4, zorder=2,
            label=f"US-designed — facility (IT × {pue:g} peak PUE)")
    ax.plot(xs, us_it * pue * gap, color=AI2030_AMBER, lw=2.4, zorder=2,
            label=f"Chinese domestic — facility (× {gap:g} efficiency gap)")

    # per-rollout recorded values at their proxy years (spread = pause-timing variation)
    ax.scatter(dec_x, dec_w, s=12, color=AI2030_OLIVE, alpha=.4, lw=0, zorder=2.5,
               label="falsely decommissioned (per rollout)")
    ax.scatter(smu_x, smu_w, s=12, color=AI2030_NAVY, alpha=.4, lw=0, zorder=2.5,
               label="smuggled (per rollout)")
    ax.scatter(smu_x, dom_w, s=12, color=AI2030_AMBER, alpha=.4, lw=0, zorder=2.5,
               label="concealed domestic (per rollout)")

    # 2026 anchor: the SuperPOD IT number the whole curve hangs off
    ax.scatter([2026.0], [w26], s=55, color=AI2030_NAVY, zorder=4, marker="D")
    ax.annotate(f"DGX GB200 SuperPOD anchor\n{w26:g} W IT / H100e @ 2026.0",
                (2026.0, w26), xytext=(10, -46), textcoords="offset points",
                fontsize=8.5, color=AI2030_NAVY,
                arrowprops=dict(arrowstyle="-", color=AI2030_NAVY, lw=.8))
    ax.text(xlo + .12, w26 * pue * .52, f"efficiency improves ×{rate:g}/yr\n→ W/H100e falls",
            fontsize=8.5, color=AI2030_MUTED, style="italic")

    # explicit y-range so annotation anchors are predictable on the log axis
    ylo = w26 / rate ** (xhi - 2026.0) * 0.72
    yhi = w26 * pue * gap / rate ** (xlo - 2026.0) * 1.10
    ax.set_ylim(ylo, yhi)

    # median-pause worked example: diversion window + the three evaluated chip sources
    ax.axvspan(start_m, decl_m, color=AI2030_GRID, alpha=.35, zorder=1)
    ax.text(mid_m, ylo * 1.10, f"diversion window [{start_m:.1f}, {decl_m:.1f}]"
            f"\n(median pause {p_m:.1f})", ha="center", va="bottom", fontsize=8,
            color=AI2030_MUTED)
    for x_e, y_e, c, lab, dxy in (
            (start_m - age, ex["decommission_w"], AI2030_OLIVE,
             f"falsely decommissioned ≈ {ex['decommission_w']:.0f} W\n"
             f"(window start − {age:g} yr: older stock)", (10, 14)),
            (mid_m + off, ex["smuggled_w"], AI2030_NAVY,
             f"smuggled ≈ {ex['smuggled_w']:.0f} W\n(window midpoint + {off:g} yr)", (16, 12)),
            (mid_m + off, ex["domestic_w"], AI2030_AMBER,
             f"concealed domestic ≈ {ex['domestic_w']:.0f} W\n(same vintage, × {gap:g})",
             (10, 8))):
        ax.scatter([x_e], [y_e], s=70, color=c, zorder=4, edgecolor="white", lw=1.2)
        ax.annotate(lab, (x_e, y_e), xytext=dxy, textcoords="offset points",
                    fontsize=8.5, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, lw=.8))

    ax.set_yscale("log")
    ax.set_yticks([200, 300, 500, 1000, 2000, 3000, 5000])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("chip vintage (proxy year the source is evaluated at)")
    ax.set_ylabel("facility W per H100e")
    ax.grid(True, which="major", alpha=.15)
    ax.legend(loc="upper right", fontsize=8, framealpha=.9)
    fig.suptitle("W per H100e by chip source — the §3.3 power calculation", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.text(.5, .005,
             f"w_US(y) = {w26:g} / {rate:g}^(y − 2026) W IT · × {pue:g} peak PUE → facility; "
             f"smuggled/domestic @ window midpoint + {off:g} yr, decommissioned @ window start "
             f"− {age:g} yr, domestic × {gap:g}.\n"
             f"Curves from {bu_p.name} (hashed run input); dots = recorded per-rollout values "
             f"({len(R)} runs; the spread is pause-timing variation).",
             ha="center", fontsize=8, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, .055, 1, .95])
    out = outdir / "bottom_up_power_calc.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def deployed_power(ds, outdir):
    """Distribution of the covert project's total deployed power (GW) across rollouts —
    facility power = covert H100e x facility_w_per_h100e, the heat/power signature the
    detection model keys on. Histogram (log-spaced bins) with p10 / median / p90 marked.
    Green overlay: the detection-aware "close race" slice — each rollout weighted by
    1{covert reaches TED-AI within 5yr of the legal} x survival_at_goal (the same
    estimator covert_within_vs_scaling_duration uses)."""
    R = ds.get("records") or []; w = _wts(ds)
    gmask = np.array([r.get("covert_size_mw") is not None and r["covert_size_mw"] > 0 for r in R])
    gw = np.array([r["covert_size_mw"] / 1000.0 for r in R
                   if r.get("covert_size_mw") is not None and r["covert_size_mw"] > 0], float)
    if gw.size == 0:
        return None
    gwt = _wsub(w, gmask)
    lead = np.array([r["lead_years"] if r.get("lead_years") is not None else np.nan
                     for r in R], float)
    s_goal = np.array([(r.get("survival_at_goal") or 0.0) for r in R], float)
    within5 = (lead >= -5.0)                  # NaN (covert never reaches TED) -> False
    base_w = np.ones(gw.size) if gwt is None else np.asarray(gwt, float)
    gwt_und5 = base_w * within5[gmask] * s_goal[gmask]
    lo, med, hi = _wpct(gw, gwt, 10), _wquantile(gw, gwt, 0.5), _wpct(gw, gwt, 90)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bins = np.logspace(np.log10(gw.min() * .9), np.log10(gw.max() * 1.1), 34)
    ax.hist(gw, bins=bins, color=AI2030_NAVY, alpha=.78, zorder=2, weights=gwt)
    if gwt_und5.sum() > 0:
        ax.hist(gw, bins=bins, color=AI2030_OLIVE, alpha=.85, zorder=2.5, weights=gwt_und5,
                label="reaches TED-AI undetected within 5yr of legal\n"
                      f"(survival-weighted; {gwt_und5.sum() / base_w.sum():.1%} of rollouts)")
    ax.axvline(med, color=AI2030_RED, ls="--", lw=1.8, zorder=3, label=f"median {med:.2f} GW")
    ax.axvline(lo, color=AI2030_MUTED, ls="--", lw=1.4, zorder=3, label=f"p10 {lo:.2f} GW")
    ax.axvline(hi, color=AI2030_MUTED, ls=":", lw=1.4, zorder=3, label=f"p90 {hi:.2f} GW")
    ax.legend(loc="upper right", fontsize=9, framealpha=.9)
    ax.set_xscale("log")
    ax.set_xlabel("covert project deployed power (GW)")
    ax.set_ylabel("number of rollouts")
    ax.grid(True, which="both", axis="x", alpha=.12); ax.grid(True, axis="y", alpha=.15)
    fig.suptitle("Covert project deployed power", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "deployed_power.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def deployed_power_cdf(ds, outdir):
    """CDF companion of deployed_power: the full deployed-power distribution vs the
    detection-aware close-race subset (weight = 1{covert reaches TED-AI within 5yr of
    legal} x survival_at_goal). Each curve is normalized to 1 — the subset curve is the
    CONDITIONAL size distribution of surviving close-race projects; its share of the
    total rollout mass is quoted in the legend."""
    R = ds.get("records") or []; w = _wts(ds)
    gmask = np.array([r.get("covert_size_mw") is not None and r["covert_size_mw"] > 0 for r in R])
    gw = np.array([r["covert_size_mw"] / 1000.0 for r in R
                   if r.get("covert_size_mw") is not None and r["covert_size_mw"] > 0], float)
    if gw.size == 0:
        return None
    gwt = _wsub(w, gmask)
    base_w = np.ones(gw.size) if gwt is None else np.asarray(gwt, float)
    lead = np.array([r["lead_years"] if r.get("lead_years") is not None else np.nan
                     for r in R], float)
    s_goal = np.array([(r.get("survival_at_goal") or 0.0) for r in R], float)
    order = np.argsort(gw); x = gw[order]

    def _cdf(wts):
        c = np.cumsum(np.asarray(wts, float)[order])
        return c / c[-1] if c[-1] > 0 else None

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, _cdf(base_w), color=AI2030_NAVY, lw=2.2, zorder=3, label="all rollouts")
    for N, colour, word in ((5.0, AI2030_OLIVE, "within 5yr of legal"),
                            (0.0, AI2030_AMBER, "BEFORE legal")):
        sub_w = base_w * (lead >= -N)[gmask] * s_goal[gmask]
        cs = _cdf(sub_w)
        if cs is not None:
            ax.plot(x, cs, color=colour, lw=2.2, zorder=3,
                    label=f"reaches TED-AI undetected {word}\n"
                          f"(survival-weighted; {sub_w.sum() / base_w.sum():.1%} of rollout mass)")
    for q in (0.1, 0.5, 0.9):
        ax.axhline(q, color=AI2030_GRID, lw=.8, zorder=1)
    med_all = _wquantile(gw, gwt, 0.5)
    ax.axvline(med_all, color=AI2030_RED, ls="--", lw=1.4, zorder=2,
               label=f"median (all) {med_all:.2f} GW")
    ax.legend(loc="upper left", fontsize=9, framealpha=.9)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("covert project deployed power (GW)")
    ax.set_ylabel("cumulative fraction of rollouts")
    ax.grid(True, which="both", axis="x", alpha=.12)
    fig.suptitle("Covert project deployed power — CDF", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "deployed_power_cdf.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def pause_cdf(ds, outdir):
    """CDF of the pause-START year across rollouts (pause_year = AC time - years_before_ac, the
    moment training freezes / the deal lands). Pause timing is the dominant driver of covert
    outcomes, so this shows how it's distributed. p10 / median / p90 marked."""
    R = ds.get("records") or []; w = _wts(ds)
    pmask = np.array([r.get("pause_year") is not None for r in R])
    py = np.array([r["pause_year"] for r in R if r.get("pause_year") is not None], float)
    if py.size == 0:
        return None
    pw = _wsub(w, pmask)
    order = np.argsort(py, kind="mergesort"); xs = py[order]
    ww = np.ones(xs.size) if pw is None else pw[order]
    cdf = np.cumsum(ww) / np.sum(ww) * 100
    med, lo, hi = _wquantile(py, pw, 0.5), _wpct(py, pw, 10), _wpct(py, pw, 90)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0.0], cdf]), where="post",
            color=AI2030_NAVY, lw=2.4, zorder=3)
    ax.axvline(med, color=AI2030_RED, ls="--", lw=1.6, label=f"median {med:.2f}", zorder=2)
    ax.axvline(lo, color=AI2030_MUTED, ls="--", lw=1.3, label=f"p10 {lo:.2f}", zorder=2)
    ax.axvline(hi, color=AI2030_MUTED, ls=":", lw=1.3, label=f"p90 {hi:.2f}", zorder=2)
    ax.legend(loc="lower right", fontsize=9, framealpha=.9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("pause-start year"); ax.set_ylabel("cumulative % of rollouts")
    ax.grid(True, alpha=.15)
    fig.suptitle("Pause-start time (CDF)", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "pause_cdf.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def legal_sw_rate_cdf(ds, outdir):
    """CDF of the consortium DISCOVERED software rate over the scaling window (OOM/yr).
    Primary curve: sw_rate_incl_floor — the realized slope of the sw path, i.e. the tuned
    rate PLUS the mandatory min_sw_frac_of_tsp floor wherever it binds (and any effort
    clamping). Muted curve: sw_rate_solved — the solver's tuned rate ON TOP of that floor;
    its spike at 0 means hardware + windfall + mandatory floor alone reach TED-AI+tax (no
    EXTRA tuned software), not zero software. A pile-up at the tuned max = worlds that can't
    reach even at the cap. Datasets predating sw_rate_incl_floor get the solved curve only."""
    R = ds["records"]; w = _wts(ds)

    def _cdf(key):
        mask = np.array([r.get(key) is not None for r in R])
        vals = np.array([r[key] for r in R if r.get(key) is not None], float)
        if vals.size == 0:
            return None
        vw = _wsub(w, mask)
        order = np.argsort(vals, kind="mergesort"); v = vals[order]
        ww = np.ones(vals.size) if vw is None else vw[order]
        return v, np.cumsum(ww) / np.sum(ww), vals, vw

    solved = _cdf("sw_rate_solved")
    if solved is None:
        return None
    sv, scdf, svals, svw = solved
    frac0 = float(_wmean(svals == 0.0, svw))
    smax = float(sv.max())
    frac_cap = float(_wmean(svals >= smax - 1e-9, svw))
    realized = _cdf("sw_rate_incl_floor")

    fig, ax = plt.subplots(figsize=(10, 6))
    if realized is not None:
        rv, rcdf, rvals, rvw = realized
        vmax = max(smax, float(rv.max()))
        med = float(_wquantile(rvals, rvw, 0.5))
        ax.plot(rv, rcdf, color=AI2030_NAVY, lw=2.4, drawstyle="steps-post", zorder=3,
                label="realized rate (incl. mandatory SW floor)")
        ax.plot(sv, scdf, color=AI2030_MUTED, lw=1.6, drawstyle="steps-post", zorder=2,
                label="tuned rate on top of the floor (solver)")
        ax.plot([0], [frac0], "o", color=AI2030_RED, ms=6, zorder=4)
        ax.annotate(f"{frac0*100:.0f}% need zero TUNED software\n"
                    f"(hardware + windfall + mandatory floor reach TED-AI+tax)",
                    (0, frac0), textcoords="offset points", xytext=(12, -6), fontsize=8.5,
                    color=AI2030_RED, va="top")
        ax.legend(loc="lower right", fontsize=9, framealpha=.9)
        med_label = f" median {med:.2f} (incl. floor)"
    else:
        vmax = smax
        med = float(_wquantile(svals, svw, 0.5))
        ax.plot(sv, scdf, color=AI2030_NAVY, lw=2.4, drawstyle="steps-post", zorder=3)
        ax.plot([0], [frac0], "o", color=AI2030_RED, ms=7, zorder=4)
        ax.annotate(f"{frac0*100:.0f}% need zero tuned software\n"
                    f"(hardware + windfall + mandatory floor reach TED-AI+tax)",
                    (0, frac0), textcoords="offset points", xytext=(12, -6), fontsize=8.5,
                    color=AI2030_RED, va="top")
        med_label = f" median {med:.2f}"
    ax.axvline(med, color=AI2030_GRID, ls="--", lw=1.2)
    ax.text(med, 0.02, med_label, fontsize=8.5, color=AI2030_MUTED)
    if frac_cap > 0.02:
        ax.text(smax, 0.5, f"  {frac_cap*100:.0f}% at the tuned cap ({smax:g})",
                fontsize=8.5, color=AI2030_MUTED, ha="right", rotation=90, va="center")
    ax.set_xlim(left=-0.02 * vmax, right=vmax * 1.02)
    ax.set_ylim(0, 1)
    ax.set_xlabel("consortium discovered software rate (OOM/yr)")
    ax.set_ylabel("CDF over rollouts")
    ax.grid(True, alpha=.15)
    fig.suptitle("Consortium discovered software rate — CDF", fontsize=14, y=.98)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "legal_sw_rate_cdf.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


def covert_undetected_reached_rel(ds, outdir):
    """P over time that a covert has REACHED TED-AI and is still UNDETECTED, with time measured
    relative to when the legal USED capability reaches TED-AI (x=0). Per rollout, at T = legal_used
    →TED-AI + x:  reached = covert_ttt <= T ;  undetected = detection (pause_year + t_det) has not
    fired by T (or never fires). P(x) = fraction of ALL rollouts in the reached-&-undetected state.
    Rises as coverts arrive, falls as detection catches reached coverts; the plateau is the
    reached-and-never-detected fraction. Faint line = reached, ignoring detection (the gap is
    detection's effect)."""
    R = ds["records"]
    if not R or "t_det" not in R[0]:
        raise RuntimeError("dataset predates detection integration — rebuild with --force")
    n = len(R)
    rr = [r for r in R if r.get("legal_used_ttt") is not None]
    if not rr:
        return None
    lu = np.array([r["legal_used_ttt"] for r in rr], float)
    rel_reach = np.array([(r["covert_ttt"] - lu[i]) if r.get("covert_ttt") is not None else np.inf
                          for i, r in enumerate(rr)], float)
    rel_det = np.array([((r["pause_year"] + r["t_det"]) - lu[i]) if r.get("t_det") is not None else np.inf
                        for i, r in enumerate(rr)], float)
    fin = rel_reach[np.isfinite(rel_reach)]
    if fin.size == 0:
        return None
    x_lo = float(np.percentile(fin, 1)) - 1.0
    x_hi = float(np.percentile(fin, 99)) + 2.0
    grid = np.linspace(x_lo, x_hi, 600)
    reached = rel_reach[:, None] <= grid[None, :]
    undet = rel_det[:, None] > grid[None, :]
    P = (reached & undet).sum(0) / n          # reached AND still undetected
    P_reached = reached.sum(0) / n            # reached, ignoring detection
    peak_i = int(np.argmax(P)); peak_x, peak_y = grid[peak_i], P[peak_i]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.axvline(0, color=AI2030_NAVY, lw=1.6, zorder=2)
    ax.text(0, 1.0, "legal reaches TED-AI ", rotation=90, ha="right", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color=AI2030_NAVY)
    ax.fill_between(grid, 0, P, color=AI2030_RED, alpha=.12, zorder=2)
    ax.plot(grid, P_reached, color=AI2030_MUTED, lw=1.8, ls="--", zorder=3,
            label="covert reached TED-AI (ignoring detection)")
    ax.plot(grid, P, color=AI2030_RED, lw=2.8, zorder=4,
            label="covert reached TED-AI AND undetected")
    ax.plot([peak_x], [peak_y], "o", color=AI2030_RED, ms=6, zorder=5)
    ax.annotate(f"peak {peak_y*100:.1f}% at {peak_x:+.1f} yr", (peak_x, peak_y),
                textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=AI2030_RED)
    ax.axhline(float(P[-1]), color=AI2030_AMBER, ls=":", lw=1, alpha=.6)
    ax.text(x_hi, float(P[-1]), f" plateau {P[-1]*100:.1f}% (never detected)", ha="right",
            va="bottom", fontsize=8, color=AI2030_MUTED)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(0, max(P_reached.max(), 1e-3) * 1.05)
    ax.set_xlabel("years relative to when the legal (used) reaches TED-AI  (− = covert first)")
    ax.set_ylabel("fraction of covert projects (over all runs)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=.15)
    fig.suptitle("Covert reached TED-AI & undetected — over legal-relative time", fontsize=15, y=.97)
    ax.set_title(_cap_rw(ds), fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = outdir / "covert_undetected_reached_rel.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out.name


FIGURES = [
    ("median_2panel", median_2panel),
    ("median_uplift", median_uplift),
    ("median_uplift_fullspeed",
     lambda ds, o: median_uplift(ds, o, sit_years=5.0, out_name="median_uplift_fullspeed.png")),
    ("oom_per_year", oom_per_year),
    ("oom_cumulative", oom_cumulative),
    ("oom_per_year_tax", oom_per_year_tax),
    ("oom_cumulative_tax", oom_cumulative_tax),
    ("legal_sw_rate_cdf", legal_sw_rate_cdf),
    ("legal_implied_compute", legal_implied_compute),
    ("implied_compute_per_year", implied_compute_per_year),
    ("implied_compute_cumulative", implied_compute_cumulative),
    ("bottom_up_compute", bottom_up_compute),
    ("bottom_up_power_calc", bottom_up_power_calc),
    ("deployed_power", deployed_power),
    ("deployed_power_cdf", deployed_power_cdf),
    ("pause_cdf", pause_cdf),
    ("covert_ted_cdf", covert_ted_cdf),
    ("covert_ted_cdf_relative", lambda ds, o: covert_ted_cdf(ds, o, relative=True, marks=(0, 5, 10))),
    ("covert_ted_cdf_relative_pause",
     lambda ds, o: covert_ted_cdf(ds, o, relative=True, marks=(5, 10, 20, 30),
                                  legal_field="pause_year")),
    ("covert_undetected_reached_rel", covert_undetected_reached_rel),
    ("covert_ted_cdf_relative_logit",
     lambda ds, o: covert_ted_cdf(ds, o, relative=True, logit=True, marks=(0, 5, 10))),
    ("covert_ted_cdf_relative_discovered",
     lambda ds, o: covert_ted_cdf(ds, o, relative=True, marks=(0, 5, 10),
                                  legal_field="legal_discovered_ttt")),
    ("covert_ted_cdf_relative_logit_discovered",
     lambda ds, o: covert_ted_cdf(ds, o, relative=True, logit=True, marks=(0, 5, 10),
                                  legal_field="legal_discovered_ttt")),
    ("covert_ted_cdf_acasi", covert_ted_cdf_acasi),
    ("default_tc_at_milestones", default_tc_at_milestones),
    ("covert_tc_at_milestones", covert_tc_at_milestones),
    ("leading_rd_at_pause", leading_rd_at_pause),
    ("default_ted_time_dist", default_ted_time_dist),
    ("ted_cdf_compare", ted_cdf_compare),
    ("covert_success_drivers", covert_success_drivers),
    ("covert_shapley_effects", covert_shapley_effects),
    ("explorer", explorer),
    ("spaghetti_pct", lambda ds, o: spaghetti(ds, o, "pct_asi")),
    ("spaghetti_uplift", lambda ds, o: spaghetti(ds, o, "uplift")),
    ("spaghetti_acasi", spaghetti_acasi),
    ("spaghetti_pct_detected", lambda ds, o: spaghetti(ds, o, "pct_asi", detect=True)),
    ("spaghetti_uplift_detected", lambda ds, o: spaghetti(ds, o, "uplift", detect=True)),
    ("spaghetti_uplift_simple", lambda ds, o: spaghetti(ds, o, "uplift", simple=True)),
    ("spaghetti_uplift_detected_simple", lambda ds, o: spaghetti(ds, o, "uplift", detect=True, simple=True)),
    ("spaghetti_pct_simple", lambda ds, o: spaghetti(ds, o, "pct_asi", simple=True)),
    ("spaghetti_pct_detected_simple", lambda ds, o: spaghetti(ds, o, "pct_asi", detect=True, simple=True)),
    ("spaghetti_acasi_detected", lambda ds, o: spaghetti_acasi(ds, o, detect=True)),
    ("spaghetti_legal_tsp", spaghetti_legal_tsp),
    ("spaghetti_deal_breakdown_pct", lambda ds, o: spaghetti_deal_breakdown(ds, o, "pct_asi")),
    ("spaghetti_deal_breakdown_uplift", lambda ds, o: spaghetti_deal_breakdown(ds, o, "uplift")),
    ("capability_cdf2d_pct", lambda ds, o: capability_cdf2d(ds, o, "pct_asi")),
    ("capability_cdf2d_uplift", lambda ds, o: capability_cdf2d(ds, o, "uplift")),
    ("capability_cdf2d_uplift_lin100", lambda ds, o: capability_cdf2d(ds, o, "uplift", linear_max=100)),
    ("capability_cdf2d_csv", capability_cdf2d_csv),
    ("capability_cdf2d_csv_uplift", lambda ds, o: capability_cdf2d_csv(ds, o, "uplift")),
    ("lead_hist", lead_hist),
    ("covert_slowdown_hist", covert_slowdown_hist),
    ("lead_drivers", lead_drivers),
    ("physics_check", physics_check),
    ("dist_conditional", dist_conditional),
    ("characterize_wins", characterize_wins),
    ("target_times", target_times),
    ("detection_winnow", detection_winnow),
    ("silver_bullet_curve", silver_bullet_curve),
    ("silver_bullet_curve_doc", lambda ds, o: silver_bullet_curve(ds, o, doc=True)),
    ("detection_elicitation_fit", detection_elicitation_fit),
    ("detection_elicitation_fit_doc", lambda ds, o: detection_elicitation_fit(ds, o, doc=True)),
    ("detection_pct_at_catch", detection_pct_at_catch),
    ("site_deployment", site_deployment),
    ("site_deployment_doc", lambda ds, o: site_deployment(ds, o, doc=True)),
    ("covert_within_vs_scaling_duration", covert_within_vs_scaling_duration),
]
