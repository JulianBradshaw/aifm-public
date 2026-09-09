#!/usr/bin/env python3
"""
Conditional analysis + CDFs for the pause-reduction experiment.

Pure post-processing over pause_reduction_samples.jsonl (no model integration).
Computes the distribution of N at each pause point, both UNCONDITIONALLY and
CONDITIONAL on (AC time approx t*_AC, AC->ASI gap approx g*) via Gaussian kernel
weighting:

    w_i = exp(-0.5 ((ac_i - t*_AC)/h_ac)^2) * exp(-0.5 ((gap_i - g*)/h_gap)^2)

The bandwidths h_ac, h_gap ARE the conditioning tolerance. Conditioning is
therefore a reweight, not a filter: every sample contributes, and the
conditioning point / bandwidths can be swept without re-running the model.

For each pause label we report (weighted) median / p10 / p90 / mean / std of N
over draws that REACHED ASI under both variants, the (weighted) censored
fraction separately, the effective sample size ESS = (sum w)^2 / sum(w^2), and
the ratio of conditional to unconditional variance ("fraction of variance
remaining" after pinning AC time and gap).

Outputs JSON stats + CDF and summary plots.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_records(path):
    path = Path(path)
    if path.is_dir():
        path = path / "pause_reduction_samples.jsonl"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs, path


def ordered_labels(records):
    """Pause labels ordered earliest->latest pause (yrbeforeac desc, then gapfrac asc)."""
    yrs, fracs = set(), set()
    for r in records:
        for pt in r.get("points", []) or []:
            if pt.get("kind") == "yrbeforeac":
                yrs.add(pt["value"])
            elif pt.get("kind") == "gapfrac":
                fracs.add(pt["value"])
    return ([f"yrbeforeac={y:g}" for y in sorted(yrs, reverse=True)]
            + [f"gapfrac={f:g}" for f in sorted(fracs)])


def collect(records, n_cap=None):
    """Per-label arrays over baseline-reached samples.

    Returns dict[label] -> dict with parallel numpy arrays:
      N (float, nan if censored/invalid), censored (bool), valid (bool),
      excluded (bool: reached but N > n_cap, a near-ASI blow-up outlier),
      ac (float), gap (float).
    """
    by_label = {}
    for r in records:
        if not (r.get("success") and r.get("baseline_reached")):
            continue
        ac = r.get("ac_time")
        gap = r.get("gap_baseline")
        if ac is None or gap is None:
            continue
        for pt in r.get("points", []) or []:
            label = pt.get("label")
            d = by_label.setdefault(label, {"N": [], "censored": [], "valid": [], "excluded": [], "ac": [], "gap": []})
            valid = pt.get("valid", True)
            n = pt.get("N")
            nval = float(n) if (valid and n is not None) else np.nan
            excluded = bool(n_cap is not None and np.isfinite(nval) and nval > n_cap)
            d["valid"].append(bool(valid))
            d["censored"].append(bool(pt.get("censored")) if valid else False)
            d["excluded"].append(excluded)
            d["N"].append(nval)
            d["ac"].append(float(ac))
            d["gap"].append(float(gap))
    for label, d in by_label.items():
        for k in d:
            d[k] = np.array(d[k])
    return by_label


def gaussian_weights(ac, gap, t_ac, g, h_ac, h_gap):
    z = ((ac - t_ac) / h_ac) ** 2 + ((gap - g) / h_gap) ** 2
    return np.exp(-0.5 * z)


def weighted_quantile(values, weights, q):
    if len(values) == 0 or np.sum(weights) <= 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / np.sum(w)  # midpoint rule
    return float(np.interp(q, cw, v))


def weighted_stats(values, weights):
    sw = np.sum(weights)
    if sw <= 0 or len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    mean = float(np.sum(weights * values) / sw)
    var = float(np.sum(weights * (values - mean) ** 2) / sw)
    return {"mean": mean, "std": float(np.sqrt(var)), "var": var}


def label_summary(d, weights):
    """Stats for one label given per-sample weights (same length as arrays).

    N>n_cap outliers (d['excluded']) are dropped from the N distribution but
    counted separately (n_excluded)."""
    valid = d["valid"]
    excluded = d.get("excluded", np.zeros_like(valid))
    reached = valid & ~d["censored"] & ~excluded & np.isfinite(d["N"])
    w_valid = weights * valid
    w_reached = weights * reached
    n_arr = d["N"][reached]
    w_arr = weights[reached]

    ess_reached = (np.sum(w_reached) ** 2 / np.sum(w_reached ** 2)) if np.sum(w_reached ** 2) > 0 else 0.0
    sw_valid = np.sum(w_valid)
    censored_frac = float(np.sum(weights * (valid & d["censored"])) / sw_valid) if sw_valid > 0 else float("nan")

    st = weighted_stats(n_arr, w_arr) if len(n_arr) else {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    return {
        "n_valid": int(np.sum(valid)),
        "n_reached": int(np.sum(reached)),
        "n_excluded": int(np.sum(valid & excluded)),
        "ess_reached": float(ess_reached),
        "censored_frac": censored_frac,
        "median_N": weighted_quantile(n_arr, w_arr, 0.5),
        "p10_N": weighted_quantile(n_arr, w_arr, 0.10),
        "p80_N": weighted_quantile(n_arr, w_arr, 0.80),
        "p90_N": weighted_quantile(n_arr, w_arr, 0.90),
        "mean_N": st["mean"],
        "std_N": st["std"],
        "var_N": st["var"],
        # for CDF plotting
        "_N": n_arr, "_w": w_arr,
    }


def main():
    ap = argparse.ArgumentParser(description="Conditional analysis + CDFs for pause-reduction")
    ap.add_argument("run", help="Run dir or pause_reduction_samples.jsonl path")
    ap.add_argument("--t-ac", default="median", help="Conditioning AC time (year) or 'median'")
    ap.add_argument("--g", default="median", help="Conditioning AC->ASI gap (years) or 'median'")
    ap.add_argument("--h-ac", type=float, default=None, help="AC-time bandwidth (default 0.3*std)")
    ap.add_argument("--h-gap", type=float, default=None, help="gap bandwidth (default 0.3*std)")
    ap.add_argument("--out", default=None, help="Output dir for plots/json (default: run dir)")
    ap.add_argument("--n-cap", type=float, default=50.0,
                    help="Exclude (reached) points with N above this from the distribution/plots "
                         "(near-ASI blow-ups). Set <=0 to disable.")
    ap.add_argument("--xlim", default=None,
                    help="Fixed N-axis range 'lo,hi' for the CDF/PDF plots (crops view; data unchanged).")
    ap.add_argument("--pdf-bins", type=int, default=30, help="Histogram bins for the PDF plot")
    ap.add_argument("--single-pdf", default=None, metavar="LABEL",
                    help="Also write a single-panel density plot for this pause label "
                         "(e.g. 'gapfrac=0')")
    args = ap.parse_args()
    xlim = tuple(float(x) for x in args.xlim.split(",")) if args.xlim else None

    records, src = load_records(args.run)
    out_dir = Path(args.out) if args.out else (src.parent if src.is_file() else src)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cap = args.n_cap if args.n_cap and args.n_cap > 0 else None
    by_label = collect(records, n_cap=n_cap)
    labels = [l for l in ordered_labels(records) if l in by_label]
    if not labels:
        print("No usable labels found.")
        sys.exit(1)

    # Conditioning point + bandwidths from the pooled AC/gap of baseline-reached samples.
    any_d = by_label[labels[0]]
    ac_all, gap_all = any_d["ac"], any_d["gap"]
    t_ac = float(np.median(ac_all)) if args.t_ac == "median" else float(args.t_ac)
    g = float(np.median(gap_all)) if args.g == "median" else float(args.g)
    h_ac = args.h_ac if args.h_ac is not None else max(0.05, 0.3 * float(np.std(ac_all)))
    h_gap = args.h_gap if args.h_gap is not None else max(0.02, 0.3 * float(np.std(gap_all)))

    # 80% central interval of the Gaussian kernel: +/- z * h, z = Phi^-1(0.9).
    Z80 = 1.2815515594908223
    ci_ac, ci_gap = Z80 * h_ac, Z80 * h_gap
    ac_lo, ac_hi = t_ac - ci_ac, t_ac + ci_ac
    gap_lo, gap_hi = g - ci_gap, g + ci_gap
    n_total = len(any_d["ac"])  # baseline-reached samples (all have ac/gap)
    in_box_global = (np.abs(any_d["ac"] - t_ac) <= ci_ac) & (np.abs(any_d["gap"] - g) <= ci_gap)
    n_box = int(np.sum(in_box_global))

    ci_text = (f"AC 80% CI [{ac_lo:.3f}, {ac_hi:.3f}] (±{ci_ac:.3f}yr), "
               f"gap 80% CI [{gap_lo:.3f}, {gap_hi:.3f}] (±{ci_gap:.3f}yr)")

    print(f"Records: {len(records)}  | pause labels: {len(labels)}")
    print(f"Conditioning point: AC={t_ac:.3f}yr (h={h_ac:.3f}), gap={g:.3f}yr (h={h_gap:.3f})")
    print(f"Kernel {ci_text}")
    print(f"Rollouts inside the 80% box (both axes): {n_box}/{n_total} baseline-reached")
    if n_cap:
        print(f"Excluding (reached) points with N > {n_cap:g} as near-ASI blow-up outliers.")
    print(f"{'label':>16}  {'med_N(unc)':>10}  {'med_N(cond)':>11}  {'var_remain':>10}  {'ESS':>6}  "
          f"{'in80box':>7}  {'reached':>7}  {'excl(unc)':>9}  {'cens(unc)':>9}  {'cens(cond)':>10}")

    out = {"source": str(src), "n_records": len(records),
           "conditioning": {"t_ac": t_ac, "g": g, "h_ac": h_ac, "h_gap": h_gap,
                            "ci80_ac": [ac_lo, ac_hi], "ci80_gap": [gap_lo, gap_hi],
                            "n_in_80box": n_box, "n_baseline_reached": n_total},
           "by_label": {}}
    flat_w = np.ones(n_total)
    box_reached = {}

    for label in labels:
        d = by_label[label]
        in_box = (np.abs(d["ac"] - t_ac) <= ci_ac) & (np.abs(d["gap"] - g) <= ci_gap)
        reached = d["valid"] & ~d["censored"] & ~d["excluded"] & np.isfinite(d["N"])
        n_box_reached = int(np.sum(in_box & reached))
        box_reached[label] = n_box_reached

        uncond = label_summary(d, flat_w)
        w = gaussian_weights(d["ac"], d["gap"], t_ac, g, h_ac, h_gap)
        cond = label_summary(d, w)
        var_remain = (cond["var_N"] / uncond["var_N"]) if (uncond["var_N"] and np.isfinite(uncond["var_N"]) and uncond["var_N"] > 0) else float("nan")
        out["by_label"][label] = {
            "unconditional": {k: v for k, v in uncond.items() if not k.startswith("_")},
            "conditional": {k: v for k, v in cond.items() if not k.startswith("_")},
            "var_remaining_ratio": var_remain,
            "n_in_80box": int(np.sum(in_box)),
            "n_in_80box_reached": n_box_reached,
        }
        print(f"{label:>16}  {uncond['median_N']:>10.3f}  {cond['median_N']:>11.3f}  "
              f"{var_remain:>10.3f}  {cond['ess_reached']:>6.1f}  {int(np.sum(in_box)):>7}  "
              f"{n_box_reached:>7}  {uncond['n_excluded']:>9}  {uncond['censored_frac']:>9.3f}  {cond['censored_frac']:>10.3f}")

    with (out_dir / "conditional_summary.json").open("w") as f:
        json.dump(out, f, indent=2)

    _plot_cdfs(by_label, labels, flat_w, t_ac, g, h_ac, h_gap, ci_text, n_box, n_total, box_reached, out_dir, n_cap, xlim)
    _plot_pdfs(by_label, labels, flat_w, t_ac, g, h_ac, h_gap, ci_text, n_box, n_total, box_reached, out_dir, n_cap, xlim, args.pdf_bins)
    if args.single_pdf:
        if args.single_pdf not in by_label:
            sys.exit(f"--single-pdf: unknown label {args.single_pdf!r}; have {labels}")
        _plot_pdf_single(by_label, args.single_pdf, flat_w, t_ac, g, h_ac, h_gap,
                         ci_text, out_dir, xlim, args.pdf_bins)
    _plot_summary(out, labels, ci_text, n_box, n_total, out_dir)
    print(f"\nWrote conditional_summary.json + plots to {out_dir}")


def _ecdf_weighted(values, weights):
    if len(values) == 0:
        return np.array([]), np.array([])
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) / np.sum(w)
    return v, cw


def _plot_cdfs(by_label, labels, flat_w, t_ac, g, h_ac, h_gap, ci_text, n_box, n_total, box_reached, out_dir, n_cap=None, xlim=None):
    n = len(labels)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for i, label in enumerate(labels):
        ax = axes[i // ncol][i % ncol]
        d = by_label[label]
        excluded = d.get("excluded", np.zeros_like(d["valid"]))
        reached = d["valid"] & ~d["censored"] & ~excluded & np.isfinite(d["N"])
        n_excl = int(np.sum(d["valid"] & excluded))
        nu = d["N"][reached]
        vu, cu = _ecdf_weighted(nu, flat_w[reached])
        ax.step(vu, cu, where="post", color="0.5", lw=1.6, label="unconditional")

        w = gaussian_weights(d["ac"], d["gap"], t_ac, g, h_ac, h_gap)[reached]
        vc, cc = _ecdf_weighted(nu, w)
        ax.step(vc, cc, where="post", color="C0", lw=1.8, label="conditional")

        # p10/p50/p80 readouts: guide lines, markers on each curve, corner table
        qs = (0.10, 0.50, 0.80)
        qu = [weighted_quantile(nu, flat_w[reached], q) for q in qs]
        qc = [weighted_quantile(nu, w, q) for q in qs]
        for q, xu, xc in zip(qs, qu, qc):
            ax.axhline(q, color="0.88", lw=0.7, ls=":", zorder=0)
            ax.plot([xu], [q], "o", ms=4, color="0.5", zorder=4)
            ax.plot([xc], [q], "o", ms=4, color="C0", zorder=5)
        table = "\n".join([f"{'':>4} {'unc':>5} {'cond':>5}"]
                          + [f"p{int(q * 100):<3} {xu:>5.2f} {xc:>5.2f}"
                             for q, xu, xc in zip(qs, qu, qc)])
        ax.text(0.98, 0.04, table, transform=ax.transAxes, fontsize=7.5,
                family="monospace", ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85, pad=3))
        ess = (np.sum(w) ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
        ax.set_title(f"{label}\nESS={ess:.0f}, in80box(reached)={box_reached.get(label, 0)}, "
                     f"n={len(nu)}, excl={n_excl}", fontsize=9)
        ax.set_xlabel("N")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    cap_note = f"  |  excl = #(reached, N>{n_cap:g}) outliers dropped" if n_cap else ""
    xnote = f"  |  x cropped to [{xlim[0]:g},{xlim[1]:g}]" if xlim else ""
    fig.suptitle(f"N CDFs by pause point — conditioning kernel {ci_text}\n"
                 f"{n_box}/{n_total} rollouts inside the 80% box (both axes){cap_note}{xnote}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = f"_x{xlim[0]:g}-{xlim[1]:g}" if xlim else ""
    fig.savefig(out_dir / f"pause_reduction_cdfs{suffix}.png", dpi=130)
    fig.savefig(out_dir / f"pause_reduction_cdfs{suffix}.pdf")
    plt.close(fig)


def _plot_pdfs(by_label, labels, flat_w, t_ac, g, h_ac, h_gap, ci_text, n_box, n_total, box_reached, out_dir, n_cap=None, xlim=None, bins=30):
    """Weighted PDF (histogram) per pause point. Each density is normalized by the
    TOTAL reached weight, so the visible area = fraction of reached mass within the
    plotted range (mass beyond the x-range is cropped, not renormalized away)."""
    n = len(labels)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    lo, hi = xlim if xlim else (1.0, 10.0)
    edges = np.linspace(lo, hi, bins + 1)
    bw = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for i, label in enumerate(labels):
        ax = axes[i // ncol][i % ncol]
        d = by_label[label]
        excluded = d.get("excluded", np.zeros_like(d["valid"]))
        reached = d["valid"] & ~d["censored"] & ~excluded & np.isfinite(d["N"])
        nu = d["N"][reached]
        wu = flat_w[reached]
        wc = gaussian_weights(d["ac"], d["gap"], t_ac, g, h_ac, h_gap)[reached]
        cu, _ = np.histogram(nu, bins=edges, weights=wu)
        cc, _ = np.histogram(nu, bins=edges, weights=wc)
        # density normalized by total reached weight -> visible area = fraction in [lo,hi]
        du = cu / (np.sum(wu) * bw) if np.sum(wu) > 0 else cu
        dc = cc / (np.sum(wc) * bw) if np.sum(wc) > 0 else cc
        frac_u = np.sum(wu[(nu >= lo) & (nu <= hi)]) / np.sum(wu) if np.sum(wu) > 0 else 0.0
        frac_c = np.sum(wc[(nu >= lo) & (nu <= hi)]) / np.sum(wc) if np.sum(wc) > 0 else 0.0
        ax.step(centers, du, where="mid", color="0.5", lw=1.6, label="unconditional")
        ax.step(centers, dc, where="mid", color="C0", lw=1.8, label="conditional")

        # p10/p50/p80 readouts (computed on the full N distribution, not the crop)
        qs = (0.10, 0.50, 0.80)
        qu = [weighted_quantile(nu, wu, q) for q in qs]
        qc = [weighted_quantile(nu, wc, q) for q in qs]
        for xu, xc in zip(qu, qc):
            ax.axvline(xu, color="0.5", lw=0.8, ls=":", zorder=0)
            ax.axvline(xc, color="C0", lw=0.8, ls=":", zorder=0)
        table = "\n".join([f"{'':>4} {'unc':>5} {'cond':>5}"]
                          + [f"p{int(q * 100):<3} {xu:>5.2f} {xc:>5.2f}"
                             for q, xu, xc in zip(qs, qu, qc)])
        ax.text(0.98, 0.96, table, transform=ax.transAxes, fontsize=7.5,
                family="monospace", ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85, pad=3))
        ax.set_title(f"{label}\nmass in range: unc={frac_u:.0%}, cond={frac_c:.0%}", fontsize=9)
        ax.set_xlabel("N")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    cap_note = f"  |  excl = #(reached, N>{n_cap:g}) dropped" if n_cap else ""
    fig.suptitle(f"N PDFs by pause point (x∈[{lo:g},{hi:g}]) — conditioning kernel {ci_text}\n"
                 f"{n_box}/{n_total} rollouts in 80% box; area = fraction of reached mass in range{cap_note}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"pause_reduction_pdfs_x{lo:g}-{hi:g}.png", dpi=130)
    plt.close(fig)


def _plot_pdf_single(by_label, label, flat_w, t_ac, g, h_ac, h_gap, ci_text, out_dir,
                     xlim=None, bins=30):
    """One larger density panel (unconditional) for a single pause label, with
    p10/p50/p80 readouts."""
    lo, hi = xlim if xlim else (1.0, 10.0)
    edges = np.linspace(lo, hi, bins + 1)
    bw = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    d = by_label[label]
    excluded = d.get("excluded", np.zeros_like(d["valid"]))
    reached = d["valid"] & ~d["censored"] & ~excluded & np.isfinite(d["N"])
    nu = d["N"][reached]
    wu = flat_w[reached]
    cu, _ = np.histogram(nu, bins=edges, weights=wu)
    du = cu / (np.sum(wu) * bw) if np.sum(wu) > 0 else cu

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.step(centers, du, where="mid", color="C0", lw=2.0)
    qs = (0.10, 0.50, 0.80)
    qu = [weighted_quantile(nu, wu, q) for q in qs]
    for xu in qu:
        ax.axvline(xu, color="C0", lw=0.9, ls=":", zorder=0)
    table = "\n".join(f"p{int(q * 100):<3} {xu:>5.2f}" for q, xu in zip(qs, qu))
    ax.text(0.98, 0.96, table, transform=ax.transAxes, fontsize=10,
            family="monospace", ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85, pad=4))
    ax.set_xlabel("N")
    ax.set_ylabel("density")
    ax.set_xlim(lo, hi)
    ax.grid(alpha=0.3)
    ax.set_title(f"N density at {label}", fontsize=11)
    fig.tight_layout()
    stem = f"pause_reduction_pdf_single_{label.replace('=', '')}"
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _plot_summary(out, labels, ci_text, n_box, n_total, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(labels))
    for kind, color in [("unconditional", "0.5"), ("conditional", "C0")]:
        med = [out["by_label"][l][kind]["median_N"] for l in labels]
        p10 = [out["by_label"][l][kind]["p10_N"] for l in labels]
        p90 = [out["by_label"][l][kind]["p90_N"] for l in labels]
        ax1.plot(x, med, "-o", color=color, label=f"median ({kind})")
        ax1.fill_between(x, p10, p90, color=color, alpha=0.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("N")
    ax1.set_title("N vs pause point (median, p10–p90)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    cu = [out["by_label"][l]["unconditional"]["censored_frac"] for l in labels]
    cc = [out["by_label"][l]["conditional"]["censored_frac"] for l in labels]
    ax2.plot(x, cu, "-o", color="0.5", label="unconditional")
    ax2.plot(x, cc, "-o", color="C3", label="conditional")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("censored fraction (ASI not reached)")
    ax2.set_title("Censored fraction vs pause point")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Conditioning kernel {ci_text} | {n_box}/{n_total} rollouts in 80% box", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "pause_reduction_summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
