#!/usr/bin/env python3
"""Effect of any predictor on any binary outcome (or continuous outcome) across rollouts.

Generalizes the one-off tax-effect plot: pick an X predictor and a Y outcome, the script
quantile-bins the rollouts by X and plots P(Y) (or mean Y) per bin with CIs plus a fitted
trend (logistic for binary, linear for continuous).

  python -m plan_a_covert_proj.pipeline.param_effect <run> --x <X> --y <Y> [--year 2040] [--bins 8]
  python -m plan_a_covert_proj.pipeline.param_effect <run> --list

X (predictor) can be:
  * any covert_params key (e.g. alignment_tax_progress_years, frac_stolen, dormancy_years)
  * any model_params key  (e.g. software_progress_rate_at_reference_year, gap_years)
  * any numeric top-level record field (e.g. the sampled bottom_up_* diversion knobs)
  * a derived name from X_DERIVED (e.g. oom_ac_to_ted, takeoff_ac_to_asi)
Derived names win over param lookups on a clash.

Y (outcome) is a name in Y_EVENTS (binary, plotted as a probability) or Y_CONT (continuous,
plotted as a mean). Year-dependent events read --year. Any lead is available dynamically as
ted_within_<d>y_of_used[_undetected] (e.g. ted_within_0y_of_used_undetected). A comma-
separated --y list of binary events overlays their curves on shared bins with each curve's
peak bin starred — how the optimal X shifts with the metric.

Observational caveat: derived predictors (and most params in a multi-param sweep) co-vary
with other inputs, so a slope here is a MARGINAL association, not an isolated causal effect.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


def os_access(p):
    return os.access(p, os.R_OK)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_MUTED, AI2030_GRID)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_MUTED, AI2030_GRID = "#1f4e9c", "#a23b3b", "#777", "#ccc"


# --------------------------------------------------------------------------- #
# predictors (X): record -> float (nan to drop the rollout from the plot)
# --------------------------------------------------------------------------- #
def _ms(r, name, field):
    m = (r.get("milestones") or {}).get(name)
    return float(m[field]) if m and m.get(field) is not None else np.nan


# frac_of_frontier_at_pause needs the run's input series (frontier compute is a function
# of the pause year, not a record field); main() installs this after loading the dataset.
_FRONTIER_FN = None


def _frac_of_frontier(r):
    if _FRONTIER_FN is None or not r.get("total_h100e_used") or r.get("pause_year") is None:
        return np.nan
    return float(r["total_h100e_used"]) / _FRONTIER_FN(float(r["pause_year"]))


def _install_frontier_fn(ds):
    """pause_year -> frontier (inference+experiment+training) H100e, log-interpolated from
    the run's input series — the same quantity sim.py's frontier_relative mode uses."""
    global _FRONTIER_FN
    path = (ds.get("config") or {}).get("input_data")
    if not path:
        return
    from progress_model import load_time_series_data
    from plan_a_covert_proj.sim import _log_interp
    # remote builds record the remote abs path; fall back to the repo copy by basename
    cands = [Path(path), REPO_ROOT / path, REPO_ROOT / Path(path).name]
    fp = next((c for c in cands if c.is_file() and os_access(c)), None)
    if fp is None:
        raise FileNotFoundError(f"input series not found locally: {path}")
    data = load_time_series_data(str(fp))
    cols = [np.asarray(getattr(data, c), float)
            for c in ("inference_H100e", "experiment_H100e", "training_H100e")]
    t = np.asarray(data.time, float)
    _FRONTIER_FN = lambda yr: float(sum(_log_interp(yr, t, c) for c in cols))


X_DERIVED = {
    "frac_of_frontier_at_pause": ("covert compute / pre-deal frontier compute at pause",
                                  _frac_of_frontier),
    "oom_ac_to_ted":   ("OOMs of effective compute, AC → TED-AI",
                        lambda r: _ms(r, "TED-AI", "e") - _ms(r, "AC", "e")),
    "oom_ac_to_asi":   ("OOMs of effective compute, AC → ASI",
                        lambda r: _ms(r, "ASI", "e") - _ms(r, "AC", "e")),
    "oom_ted_to_asi":  ("OOMs of effective compute, TED-AI → ASI",
                        lambda r: _ms(r, "ASI", "e") - _ms(r, "TED-AI", "e")),
    "takeoff_ac_to_asi": ("default takeoff length: AC → ASI (calendar yr)",
                          lambda r: _ms(r, "ASI", "time") - _ms(r, "AC", "time")),
    "takeoff_ac_to_ted": ("AC → TED-AI gap (calendar yr)",
                          lambda r: _ms(r, "TED-AI", "time") - _ms(r, "AC", "time")),
    "ac_time":  ("AC calendar year", lambda r: r.get("ac_time", np.nan)),
    "ted_ai_time": ("TED-AI calendar year", lambda r: r.get("ted_ai_time") or np.nan),
    "lead_years": ("legal−covert lead (yr)",
                   lambda r: r["lead_years"] if r.get("lead_years") is not None else np.nan),
}


def get_x(r, key):
    if key in X_DERIVED:
        try:
            return float(X_DERIVED[key][1](r))
        except Exception:  # noqa: BLE001
            return np.nan
    for bag in ("covert_params", "model_params"):
        d = r.get(bag) or {}
        if key in d and isinstance(d[key], (int, float)):
            return float(d[key])
    v = r.get(key)   # top-level record fields (e.g. the sampled bottom_up_* knobs)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return np.nan


def x_label(key):
    if key in X_DERIVED:
        return X_DERIVED[key][0]
    return key


# --------------------------------------------------------------------------- #
# binary outcomes (Y): record, year -> bool   (probability plots)
# --------------------------------------------------------------------------- #
def _ttt(r):
    return r.get("covert_ttt")


Y_EVENTS = {
    "covert_wins": ("P(covert wins)", lambda r, yr: bool(r.get("covert_wins"))),
    "reaches_ted": ("P(covert reaches TED-AI)", lambda r, yr: _ttt(r) is not None),
    "reaches_ted_by": ("P(covert reaches TED-AI by {yr:.0f})",
                       lambda r, yr: _ttt(r) is not None and _ttt(r) <= yr),
    "detected_before_goal": ("P(covert detected before reaching TED-AI)",
                             lambda r, yr: bool(r.get("detected_before_goal"))),
    "undetected_reaches_ted": ("P(covert reaches TED-AI, undetected)",
                               lambda r, yr: _ttt(r) is not None and not r.get("detected_before_goal")),
    "undetected_ted_by": ("P(covert reaches TED-AI by {yr:.0f}, undetected)",
                          lambda r, yr: _ttt(r) is not None and _ttt(r) <= yr
                          and not r.get("detected_before_goal")),
    "undetected_by_year": ("P(covert still undetected at {yr:.0f})",
                           lambda r, yr: (r.get("t_det") is None)
                           or (r["pause_year"] + r["t_det"] > yr)),
    "win_undetected": ("P(covert wins, undetected before TED-AI)",
                       lambda r, yr: bool(r.get("covert_wins")) and not r.get("detected_before_goal")),
    # Within-5y-of-the-consortium arrivals: covert reaches TED-AI no later than 5 years
    # after the legal USED (deployed) capability does. Rollouts where the legal never
    # reaches TED-AI count only if the covert reaches it at all (no deadline to beat).
    "ted_within_5y_of_used": ("P(covert reaches TED-AI ≤ legal used→TED-AI + 5y)",
                              lambda r, yr: _ttt(r) is not None
                              and (r.get("legal_used_ttt") is None
                                   or _ttt(r) <= r["legal_used_ttt"] + 5.0)),
    "ted_within_5y_of_used_undetected": (
        "P(covert reaches TED-AI ≤ legal used→TED-AI + 5y, undetected)",
        lambda r, yr: _ttt(r) is not None
        and (r.get("legal_used_ttt") is None or _ttt(r) <= r["legal_used_ttt"] + 5.0)
        and not r.get("detected_before_goal")),
}

# Dynamic Y names: ted_within_<d>y_of_used[_undetected] for any lead d (e.g.
# ted_within_0y_of_used_undetected = covert reaches TED-AI no later than the legal used
# arrival AND is undetected at its own arrival). Same no-legal-arrival convention as the
# static 5y events above.
_TED_WITHIN_RE = re.compile(r"^ted_within_(\d+(?:\.\d+)?)y_of_used(_undetected)?$")


def resolve_y_event(name):
    """(label, fn) for a binary Y — Y_EVENTS entries plus the dynamic ted_within_* names;
    None if the name is not a binary event (caller then tries Y_CONT)."""
    if name in Y_EVENTS:
        return Y_EVENTS[name]
    m = _TED_WITHIN_RE.match(name)
    if not m:
        return None
    d, und = float(m.group(1)), bool(m.group(2))

    def fn(r, yr, _d=d, _u=und):
        if _ttt(r) is None:
            return False
        if r.get("legal_used_ttt") is not None and _ttt(r) > r["legal_used_ttt"] + _d:
            return False
        return not r.get("detected_before_goal") if _u else True

    return (f"P(covert reaches TED-AI ≤ legal used→TED-AI + {d:g}y"
            + (", undetected)" if und else ")"), fn)


# --------------------------------------------------------------------------- #
# continuous outcomes (Y): record -> float or None (mean plots)
# --------------------------------------------------------------------------- #
Y_CONT = {
    "covert_ttt": ("covert TED-AI calendar year", lambda r: r.get("covert_ttt")),
    "lead_years": ("legal−covert lead (yr)", lambda r: r.get("lead_years")),
    "survival_at_goal": ("covert survival prob. at TED-AI", lambda r: r.get("survival_at_goal")),
    "covert_ttt_after_pause": ("covert years pause→TED-AI",
                               lambda r: (r["covert_ttt"] - r["pause_year"])
                               if r.get("covert_ttt") is not None else None),
    "lead_used": ("covert lead = legal used→TED-AI − covert→TED-AI (yr; + = covert first)",
                  lambda r: (r["legal_used_ttt"] - r["covert_ttt"])
                  if (r.get("legal_used_ttt") is not None and r.get("covert_ttt") is not None)
                  else None),
}


# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.0):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def logistic_fit(x, y, iters=300):
    xs = (x - x.mean()) / (x.std() + 1e-12)
    X = np.column_stack([np.ones_like(xs), xs]); b = np.zeros(2)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(X @ b)))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = X @ b + (y - mu) / W
        b_new = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(2), X.T @ (W * z))
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new; break
        b = b_new
    # map back to raw-x slope for reporting
    slope_raw = b[1] / (x.std() + 1e-12)
    return b, xs, slope_raw


def plot_effect(ds, run, x_key, y_key, n_bins, year, outdir, logx=False):
    R = ds["records"]
    ev = resolve_y_event(y_key)
    is_binary = ev is not None
    if not is_binary and y_key not in Y_CONT:
        raise SystemExit(f"unknown --y '{y_key}'. See --list.")
    ylabel_t, yfn = ev if is_binary else Y_CONT[y_key]
    ylabel = ylabel_t.format(yr=year) if is_binary else ylabel_t

    x = np.array([get_x(r, x_key) for r in R], float)
    if is_binary:
        yv = np.array([bool(yfn(r, year)) for r in R], float)
        keep = np.isfinite(x)
    else:
        raw = [yfn(r) for r in R]
        yv = np.array([v if v is not None else np.nan for v in raw], float)
        keep = np.isfinite(x) & np.isfinite(yv)
    if logx:
        n_drop = int(np.sum(keep & (x <= 0)))
        if n_drop:
            print(f"--logx: dropping {n_drop} rollouts with {x_key} <= 0")
        keep = keep & (x > 0)
    x, yv = x[keep], yv[keep]
    if x.size < n_bins * 3:
        raise SystemExit(f"too few usable rollouts ({x.size}) for {n_bins} bins.")
    if np.unique(x).size < 2:
        raise SystemExit(f"predictor '{x_key}' is constant in this run — nothing to plot.")

    order = np.argsort(x)
    bx, by, blo, bhi, bn = [], [], [], [], []
    for ch in np.array_split(order, n_bins):
        n = len(ch); bn.append(n)
        bx.append(float(np.exp(np.log(x[ch]).mean())) if logx else float(x[ch].mean()))
        if is_binary:
            p, lo, hi = wilson(int(yv[ch].sum()), n)
            by.append(p); blo.append(lo); bhi.append(hi)
        else:
            m = float(yv[ch].mean()); se = float(yv[ch].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            by.append(m); blo.append(m - se); bhi.append(m + se)
    bx, by, blo, bhi = map(np.array, (bx, by, blo, bhi))
    sc = 100.0 if is_binary else 1.0

    stat = "proportion" if is_binary else "mean"
    band_lab = "68% Wilson CI" if is_binary else "±1 SEM (mean ± std error)"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(bx, blo * sc, bhi * sc, color=AI2030_NAVY, alpha=.12, lw=0,
                    label=band_lab)
    ax.plot(bx, by * sc, "o-", color=AI2030_NAVY, lw=1.6, ms=6, zorder=4,
            label=f"bin {stat.upper()}  (equal-count, n≈{int(np.median(bn))} each)")
    # fit in the displayed space: log10(x) when logx, else raw x. unit = "decade" on a log axis.
    xfit = np.log10(x) if logx else x
    gx = (np.logspace(np.log10(x.min()), np.log10(x.max()), 200) if logx
          else np.linspace(x.min(), x.max(), 200))
    gxfit = np.log10(gx) if logx else gx
    unit = "decade" if logx else "unit"
    if is_binary:
        b, _, slope_raw = logistic_fit(xfit, yv)
        gy = 1.0 / (1.0 + np.exp(-(b[0] + b[1] * (gxfit - xfit.mean()) / (xfit.std() + 1e-12))))
        fit_lab = f"logistic fit (slope {slope_raw:+.3g}/{unit})"
    else:
        m, c = np.polyfit(xfit, yv, 1); gy = m * gxfit + c
        fit_lab = f"linear fit (slope {m:+.3g}/{unit})"
    ax.plot(gx, gy * sc, "--", color=AI2030_RED, lw=2, label=fit_lab)
    for xx, yy, nn in zip(bx, by, bn):
        ax.annotate(f"n={nn}", (xx, yy * sc), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=AI2030_MUTED)
    base = float(np.nanmean(yv)) * sc
    ax.axhline(base, color=AI2030_GRID, ls=":", lw=1)
    ax.text(x.max(), base, f" overall {stat} {base:.1f}{'%' if is_binary else ''}", ha="right",
            va="bottom", fontsize=8, color=AI2030_MUTED)
    if is_binary:
        ax.set_ylim(bottom=0)
    elif (np.nanmin(yv) < 0 < np.nanmax(yv)):   # signed outcome (e.g. lead): mark the zero line
        ax.axhline(0, color=AI2030_NAVY, lw=1.2, zorder=1)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x_label(x_key) + ("  [log scale]" if logx else ""))
    ax.set_ylabel((ylabel + "  [%]") if is_binary else ("mean  " + ylabel))
    ax.legend(loc="best", fontsize=8.5)
    fig.suptitle(f"{x_label(x_key)}  vs  {ylabel}", fontsize=13, y=.97)
    ax.set_title(f"{x.size} rollouts · run {run} · marginal (observational) relationship",
                 fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"effect_{x_key}__{y_key}{'_logx' if logx else ''}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)
    print(f"overall {'P' if is_binary else 'mean'} = {base:.3g}{'%' if is_binary else ''}  "
          f"·  {fit_lab}")
    return out


def overlay_effect(ds, run, x_key, y_keys, n_bins, year, outdir, logx=False):
    """Several BINARY outcomes vs one predictor, on SHARED equal-count bins, one curve
    each with its peak bin starred — shows how the optimum X shifts with the metric
    (e.g. --y ted_within_0y_of_used_undetected,ted_within_5y_of_used_undetected,...)."""
    import matplotlib.colors as mcolors
    R = ds["records"]
    evs = []
    for k in y_keys:
        ev = resolve_y_event(k)
        if ev is None:
            raise SystemExit(f"overlay needs binary Y events; unknown/continuous: {k!r}")
        evs.append((k, ev[0].format(yr=year), ev[1]))

    x = np.array([get_x(r, x_key) for r in R], float)
    keep = np.isfinite(x)
    if logx:
        n_drop = int(np.sum(keep & (x <= 0)))
        if n_drop:
            print(f"--logx: dropping {n_drop} rollouts with {x_key} <= 0")
        keep &= x > 0
    kept = [r for r, k in zip(R, keep) if k]
    x = x[keep]
    if x.size < n_bins * 3:
        raise SystemExit(f"too few usable rollouts ({x.size}) for {n_bins} bins.")
    order = np.argsort(x)
    chunks = np.array_split(order, n_bins)
    bx = np.array([float(np.exp(np.log(x[ch]).mean())) if logx else float(x[ch].mean())
                   for ch in chunks])
    bn = [len(ch) for ch in chunks]

    # short legend labels when every key is a ted_within_* variant: just the lead
    ms = [_TED_WITHIN_RE.match(k) for k, _, _ in evs]
    all_lead = all(ms)
    ramp = mcolors.LinearSegmentedColormap.from_list("nvr", [AI2030_NAVY, AI2030_RED])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (k, lab, fn) in enumerate(evs):
        yv = np.array([bool(fn(r, year)) for r in kept], float)
        p, lo, hi = np.transpose([wilson(int(yv[ch].sum()), len(ch)) for ch in chunks])
        col = ramp(i / max(len(evs) - 1, 1))
        short = f"≤ legal + {float(ms[i].group(1)):g}y" if all_lead else lab
        ax.fill_between(bx, lo * 100, hi * 100, color=col, alpha=.10, lw=0)
        ax.plot(bx, p * 100, "o-", color=col, lw=1.6, ms=4.5, zorder=4, label=short)
        j = int(np.argmax(p))
        ax.plot([bx[j]], [p[j] * 100], "*", color=col, ms=15, mec="white", mew=.6, zorder=5)
        ax.annotate(f"{bx[j]:.2g}", (bx[j], p[j] * 100), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7.5, color=col)
    title = ("P(covert reaches TED-AI ≤ legal used→TED-AI + Δ, undetected)"
             if all_lead and all(m.group(2) for m in ms)
             else ("P(covert reaches TED-AI ≤ legal used→TED-AI + Δ)"
                   if all_lead and not any(m.group(2) for m in ms) else "P(event)"))
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(x_label(x_key) + ("  [log scale]" if logx else ""))
    ax.set_ylabel(title + "  [%]")
    ax.legend(loc="best", fontsize=8.5, title="star = peak bin (x annotated)",
              title_fontsize=8)
    st = f"{x_label(x_key)}  vs  {title}"
    fig.suptitle(st if len(st) <= 80 else f"{x_label(x_key)}  vs\n{title}",
                 fontsize=13, y=.97)
    ax.set_title(f"{x.size} rollouts · run {run} · shared equal-count bins "
                 f"(n≈{int(np.median(bn))} each) · marginal (observational)",
                 fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    outdir.mkdir(parents=True, exist_ok=True)
    if all_lead:
        sfx = ("_undetected" if all(m.group(2) for m in ms)
               else "" if not any(m.group(2) for m in ms) else "_mixed")
        yname = "ted_within_" + "-".join(f"{float(m.group(1)):g}" for m in ms) + "y_of_used" + sfx
    else:
        yname = f"overlay_{evs[0][0]}_plus{len(evs) - 1}"
    out = outdir / f"effect_{x_key}__{yname}{'_logx' if logx else ''}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)
    return out


def heatmap_effect(ds, run, x_key, y_key, z_key, n_bins, year, outdir,
                   logx=False, logy=False, min_n=5, cmap=None, center=None, quantile_axes=False,
                   agg="mean"):
    """Joint effect of two predictors on an outcome: bin both (quantile edges, log-aware) and
    color each cell by the outcome. Binary outcome (Y_EVENTS) -> P(event) %; continuous
    outcome (Y_CONT) -> mean value (e.g. covert_ttt = when the covert reaches TED-AI, ignoring
    detection). Cells with n below min_n are blanked; each cell is annotated value + n."""
    R = ds["records"]
    zev = resolve_y_event(z_key)
    is_binary = zev is not None
    if not is_binary and z_key not in Y_CONT:
        raise SystemExit(f"unknown --y '{z_key}'. See --list.")
    if is_binary:
        zlab = zev[0].format(yr=year); zfn = zev[1]
        zv = np.array([float(zfn(r, year)) for r in R], float)   # 0/1, always defined
    else:
        zlab = Y_CONT[z_key][0]; zfn = Y_CONT[z_key][1]
        zv = np.array([(v if v is not None else np.nan) for v in (zfn(r) for r in R)], float)
    x = np.array([get_x(r, x_key) for r in R], float)
    yy = np.array([get_x(r, y_key) for r in R], float)
    keep = np.isfinite(x) & np.isfinite(yy)
    if logx:
        keep &= x > 0
    if logy:
        keep &= yy > 0
    x, yy, zv = x[keep], yy[keep], zv[keep]
    if x.size < n_bins * n_bins:
        print(f"warning: only {x.size} rollouts for a {n_bins}x{n_bins} grid — cells will be sparse")

    def edges(v):
        return np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    ex, ey = edges(x), edges(yy)
    nx, ny = len(ex) - 1, len(ey) - 1
    ix = np.clip(np.digitize(x, ex[1:-1]), 0, nx - 1)
    iy = np.clip(np.digitize(yy, ey[1:-1]), 0, ny - 1)
    Z = np.full((ny, nx), np.nan); N = np.zeros((ny, nx), int)
    for cx in range(nx):
        for cy in range(ny):
            m = (ix == cx) & (iy == cy)
            zc = zv[m]; zc = zc[np.isfinite(zc)]   # for continuous, drop never-reach (NaN)
            N[cy, cx] = zc.size
            if zc.size >= min_n:
                if is_binary:
                    Z[cy, cx] = zc.mean() * 100.0
                else:
                    Z[cy, cx] = float(np.median(zc) if agg == "median" else zc.mean())

    import matplotlib.colors as mcolors
    vmin = 0.0 if is_binary else float(np.nanmin(Z)) if np.isfinite(Z).any() else 0.0
    vmax = (float(np.nanmax(Z)) if np.isfinite(Z).any() else 1.0)
    # center=c -> diverging map split at c (e.g. lead split at 0: covert-ahead red, behind blue)
    if center is not None and vmin < center < vmax:
        name = cmap or "RdBu_r"
        norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    else:
        name = cmap or ("YlOrRd" if is_binary else "RdYlBu")   # binary: high=red; continuous: low=red
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(name).copy(); cm.set_bad("#e8e8e8")
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    # quantile_axes: equal-size cells indexed by quantile rank (axes show the bin-edge values),
    # vs value-scaled cells whose widths reflect the (log) value ranges.
    mx = np.arange(nx + 1) if quantile_axes else ex
    my = np.arange(ny + 1) if quantile_axes else ey
    pm = ax.pcolormesh(mx, my, np.ma.masked_invalid(Z), cmap=cm, norm=norm,
                       edgecolors="white", lw=.5)
    fig.colorbar(pm, ax=ax).set_label(zlab + ("  [%]" if is_binary else f"  ({agg})"))
    cen = lambda e, i, log: (np.sqrt(e[i] * e[i + 1]) if log else .5 * (e[i] + e[i + 1]))
    fmt = (lambda v: f"{v:.0f}%") if is_binary else (lambda v: f"{v:.1f}")
    for cx in range(nx):
        for cy in range(ny):
            xc = (cx + 0.5) if quantile_axes else cen(ex, cx, logx)
            yc = (cy + 0.5) if quantile_axes else cen(ey, cy, logy)
            if N[cy, cx] >= min_n:
                r_, g_, b_, _ = cm(norm(Z[cy, cx]))
                txt = "white" if (0.299 * r_ + 0.587 * g_ + 0.114 * b_) < 0.5 else "#222"
                ax.text(xc, yc, f"{fmt(Z[cy, cx])}\nn={N[cy, cx]}", ha="center", va="center",
                        fontsize=7.5, color=txt)
            elif N[cy, cx] > 0:
                ax.text(xc, yc, f"n={N[cy, cx]}", ha="center", va="center", fontsize=6.5,
                        color=AI2030_MUTED)
    if quantile_axes:
        # ticks at quantile ranks: show the percentile (top) and the bin-edge value (below)
        xpct = [100.0 * np.mean(x <= e) for e in ex]
        ypct = [100.0 * np.mean(yy <= e) for e in ey]
        ax.set_xticks(np.arange(nx + 1))
        ax.set_xticklabels([f"p{p:.0f}\n{v:.4g}" for p, v in zip(xpct, ex)], fontsize=7)
        ax.set_yticks(np.arange(ny + 1))
        ax.set_yticklabels([f"p{p:.0f} ({v:.4g})" for p, v in zip(ypct, ey)], fontsize=7)
    else:
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
    sfx_lab = "  [quantile-binned]" if quantile_axes else ("  [log]" if logx else "")
    sfy_lab = "  [quantile-binned]" if quantile_axes else ("  [log]" if logy else "")
    ax.set_xlabel(x_label(x_key) + sfx_lab)
    ax.set_ylabel(x_label(y_key) + sfy_lab)
    if is_binary:
        base = float(np.nanmean(zv)) * 100.0
    else:
        base = float(np.nanmedian(zv) if agg == "median" else np.nanmean(zv))
    n_valid = int(np.isfinite(zv).sum())
    fig.suptitle(f"{zlab}\nby {x_label(x_key)} × {x_label(y_key)}", fontsize=12.5, y=.99)
    over = (f"overall {base:.1f}%" if is_binary
            else f"overall {agg} {base:.1f} (n={n_valid}/{zv.size} reach)")
    ax.set_title(f"{x.size} rollouts · run {run} · {over} · cells with n<{min_n} blank",
                 fontsize=9, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, .95])
    outdir.mkdir(parents=True, exist_ok=True)
    agg_sfx = ("_" + agg) if (not is_binary and agg != "mean") else ""
    sfx = ("_qaxes" if quantile_axes else ("_logx" if logx else "") + ("_logy" if logy else "")) + agg_sfx
    out = outdir / f"heatmap_{x_key}__{y_key}__{z_key}{sfx}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)
    rng = (f"{np.nanmin(Z):.1f}–{np.nanmax(Z):.1f}" if np.isfinite(Z).any() else "n/a")
    print(f"{'overall P' if is_binary else 'overall ' + agg} = {base:.2f}"
          f"{'%' if is_binary else ''} · grid {nx}x{ny} · cell range {rng}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runid under outputs/runs/")
    ap.add_argument("--x", default="alignment_tax_progress_years")
    ap.add_argument("--y", default="undetected_ted_by")
    ap.add_argument("--year", type=float, default=2040.0)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--logx", action="store_true", help="log-scale the X axis (fit in log space)")
    ap.add_argument("--x2", default=None,
                    help="second predictor -> 2D heatmap of P(--y) over --x × --x2")
    ap.add_argument("--logy", action="store_true", help="log-scale the second axis (heatmap)")
    ap.add_argument("--min-n", type=int, default=5, help="blank heatmap cells below this count")
    ap.add_argument("--cmap", default=None, help="heatmap colormap override (matplotlib name)")
    ap.add_argument("--center", type=float, default=None,
                    help="heatmap: diverging colormap split at this value (e.g. 0 for lead)")
    ap.add_argument("--quantile-axes", action="store_true",
                    help="heatmap: equal-size cells indexed by quantile rank (axes show bin edges)")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean",
                    help="heatmap cell aggregation for a continuous outcome (default mean)")
    ap.add_argument("--outdir", default=None, help="default: the run's figures/ dir")
    ap.add_argument("--list", action="store_true", help="list X derived names and Y outcomes")
    a = ap.parse_args()

    if a.list:
        print("X derived predictors:")
        for k, (lab, _) in X_DERIVED.items():
            print(f"  {k:22s} {lab}")
        print("  <any covert_params / model_params key, or numeric top-level record field> (numeric)")
        print("\nY binary events (probability):")
        for k, (lab, _) in Y_EVENTS.items():
            print(f"  {k:24s} {lab}")
        print("  ted_within_<d>y_of_used[_undetected]   any lead d (dynamic)")
        print("\nY continuous outcomes (mean):")
        for k, (lab, _) in Y_CONT.items():
            print(f"  {k:24s} {lab}")
        return

    run_dir = REPO_ROOT / "plan_a_covert_proj/outputs/runs" / a.run
    ds = json.load(open(run_dir / "dataset.json"))
    try:
        _install_frontier_fn(ds)
    except Exception as e:  # noqa: BLE001
        print(f"frac_of_frontier_at_pause unavailable ({e})")
    outdir = Path(a.outdir) if a.outdir else (run_dir / "figures")
    y_keys = [k.strip() for k in a.y.split(",") if k.strip()]
    if a.x2:
        if len(y_keys) > 1:
            raise SystemExit("--x2 heatmaps take a single --y")
        heatmap_effect(ds, a.run, a.x, a.x2, a.y, a.bins, a.year, outdir,
                       logx=a.logx, logy=a.logy, min_n=a.min_n, cmap=a.cmap, center=a.center,
                       quantile_axes=a.quantile_axes, agg=a.agg)
    elif len(y_keys) > 1:
        overlay_effect(ds, a.run, a.x, y_keys, a.bins, a.year, outdir, logx=a.logx)
    else:
        plot_effect(ds, a.run, a.x, y_keys[0], a.bins, a.year, outdir, logx=a.logx)

    # If the figure went into the run's own figures dir, refresh that run's manifest
    # (re-scan the dir) and rebuild the viewer so the new plot shows up in the gallery.
    if outdir.resolve() == (run_dir / "figures").resolve():
        mf = run_dir / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            m["figures"] = sorted(p.name for p in (run_dir / "figures").iterdir() if p.is_file())
            mf.write_text(json.dumps(m, indent=2))
        try:
            from plan_a_covert_proj.pipeline import make_viewer
            vp, n = make_viewer.regenerate()
            if vp is not None:
                print(f"viewer -> {vp.relative_to(REPO_ROOT)} ({n} runs)")
        except Exception as e:  # noqa: BLE001
            print(f"viewer refresh skipped: {e}")


if __name__ == "__main__":
    main()
