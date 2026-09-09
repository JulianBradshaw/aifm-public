#!/usr/bin/env python3
"""
Residual-variance attribution for the pause-reduction experiment.

Question: among scenarios that all reach AC ~ t*_AC and have AC->ASI gap ~ g*
(e.g. "AC in 2030, ASI in 2031"), N still varies a lot. WHICH other parameters
drive that residual variation, and what do high-N vs low-N scenarios look like?

Approach: take the rollouts inside the conditioning 80% box (AC within +/- z*h_ac
of t*_AC AND gap within +/- z*h_gap of g*), at a chosen pause label, restricted
to draws that reached ASI (N defined). Then for each numeric model parameter:
  - Spearman rho of N vs the parameter across the in-slice rollouts (rank, robust
    to nonlinearity/monotone transforms).
Rank parameters by |rho|. Also contrast the high-N tercile vs low-N tercile of
in-slice rollouts (median parameter values) so the "kinds of scenarios" are
legible. Categorical parameters are summarized by N per category.

Outputs JSON + a tornado plot of |rho| and a scatter of N vs the top driver.
Pure post-processing over pause_reduction_samples.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Z80 = 1.2815515594908223


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


def main():
    ap = argparse.ArgumentParser(description="Residual-variance attribution for N within an (AC, gap) slice")
    ap.add_argument("run", help="Run dir or pause_reduction_samples.jsonl")
    ap.add_argument("--t-ac", default="median", help="AC time center (year) or 'median'")
    ap.add_argument("--g", type=float, default=1.0, help="gap center (years)")
    ap.add_argument("--h-ac", type=float, default=0.15, help="AC bandwidth (std). 80%% CI = +/-1.28*h")
    ap.add_argument("--h-gap", type=float, default=0.15, help="gap bandwidth (std)")
    ap.add_argument("--label", default="gapfrac=0", help="Pause label to attribute N at")
    ap.add_argument("--top", type=int, default=12, help="How many top drivers to show")
    ap.add_argument("--n-cap", type=float, default=50.0,
                    help="Exclude in-slice reached points with N above this (near-ASI blow-ups). <=0 disables.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n_cap = args.n_cap if args.n_cap and args.n_cap > 0 else None

    records, src = load_records(args.run)
    out_dir = Path(args.out) if args.out else (src.parent if src.is_file() else src)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather baseline-reached rollouts with the chosen pause label point.
    ac, gap, N, params = [], [], [], []
    for r in records:
        if not (r.get("success") and r.get("baseline_reached")):
            continue
        a, gp = r.get("ac_time"), r.get("gap_baseline")
        if a is None or gp is None:
            continue
        pt = next((p for p in r.get("points", []) if p.get("label") == args.label), None)
        if pt is None or not pt.get("valid", True) or pt.get("censored") or pt.get("N") is None:
            continue
        ac.append(float(a)); gap.append(float(gp)); N.append(float(pt["N"])); params.append(r["parameters"])
    ac = np.array(ac); gap = np.array(gap); N = np.array(N)
    if len(N) == 0:
        print("No reached rollouts at this label."); sys.exit(1)

    t_ac = float(np.median(ac)) if args.t_ac == "median" else float(args.t_ac)
    ci_ac, ci_gap = Z80 * args.h_ac, Z80 * args.h_gap
    in_box = (np.abs(ac - t_ac) <= ci_ac) & (np.abs(gap - args.g) <= ci_gap)
    in_cap = (N <= n_cap) if n_cap else np.ones_like(N, dtype=bool)
    n_excluded_slice = int(np.sum(in_box & ~in_cap))
    idx = np.where(in_box & in_cap)[0]
    n_slice = len(idx)

    print(f"Label {args.label}: {len(N)} reached rollouts total")
    print(f"Slice: AC in [{t_ac-ci_ac:.3f},{t_ac+ci_ac:.3f}], gap in [{args.g-ci_gap:.3f},{args.g+ci_gap:.3f}]")
    if n_cap:
        print(f"Excluded {n_excluded_slice} in-slice point(s) with N > {n_cap:g} (near-ASI blow-ups).")
    print(f"In-slice reached rollouts: {n_slice}")
    if n_slice < 20:
        print("WARNING: <20 rollouts in slice; attribution will be noisy. Use a larger run / wider bandwidth.")
    if n_slice < 4:
        sys.exit(1)

    Ns = N[idx]
    # Collect numeric and categorical params over the slice.
    keys = sorted({k for i in idx for k in params[i].keys()})
    numeric, categorical = {}, {}
    for k in keys:
        vals = [params[i].get(k) for i in idx]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) and len(set(vals)) > 1:
            numeric[k] = np.array(vals, dtype=float)
        elif len(set(map(str, vals))) > 1:
            categorical[k] = [str(v) for v in vals]

    # Spearman rho of N vs each numeric parameter within the slice.
    rows = []
    for k, v in numeric.items():
        rho, p = stats.spearmanr(v, Ns)
        if np.isfinite(rho):
            rows.append((k, float(rho), float(p)))
    rows.sort(key=lambda x: -abs(x[1]))

    print(f"\nTop drivers of N within the slice (Spearman rho of N vs param):")
    print(f"{'parameter':>40}  {'rho':>7}  {'p':>8}")
    for k, rho, p in rows[:args.top]:
        print(f"{k:>40}  {rho:>7.3f}  {p:>8.1e}")

    # High-N vs low-N tercile contrast (parameter medians).
    order = np.argsort(Ns)
    k_ter = max(2, n_slice // 3)
    lo_idx, hi_idx = order[:k_ter], order[-k_ter:]
    contrast = {}
    for k, v in numeric.items():
        contrast[k] = {"low_N_median": float(np.median(v[lo_idx])),
                       "high_N_median": float(np.median(v[hi_idx]))}

    cat_summary = {}
    for k, vals in categorical.items():
        cats = {}
        arr = np.array(vals)
        for c in sorted(set(vals)):
            mask = arr == c
            cats[c] = {"n": int(np.sum(mask)), "median_N": float(np.median(Ns[mask]))}
        cat_summary[k] = cats

    out = {
        "source": str(src), "label": args.label,
        "slice": {"t_ac": t_ac, "g": args.g, "h_ac": args.h_ac, "h_gap": args.h_gap,
                  "ci80_ac": [t_ac - ci_ac, t_ac + ci_ac], "ci80_gap": [args.g - ci_gap, args.g + ci_gap],
                  "n_in_slice": n_slice},
        "N_in_slice": {"median": float(np.median(Ns)), "p10": float(np.percentile(Ns, 10)),
                       "p90": float(np.percentile(Ns, 90)), "min": float(np.min(Ns)), "max": float(np.max(Ns))},
        "spearman_vs_N": [{"param": k, "rho": rho, "p": p} for k, rho, p in rows],
        "tercile_contrast": contrast,
        "categorical_summary": cat_summary,
    }
    with (out_dir / f"attribution_{args.label.replace('=', '')}.json").open("w") as f:
        json.dump(out, f, indent=2)

    _plot(rows[:args.top], numeric, Ns, contrast, args.label, n_slice, out_dir)
    print(f"\nWrote attribution JSON + plot to {out_dir}")


def _plot(top_rows, numeric, Ns, contrast, label, n_slice, out_dir):
    if not top_rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = [k for k, _, _ in top_rows][::-1]
    rhos = [rho for _, rho, _ in top_rows][::-1]
    colors = ["C3" if r > 0 else "C0" for r in rhos]
    ax1.barh(range(len(names)), rhos, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("Spearman rho(N, parameter)")
    ax1.set_title(f"Drivers of N within slice ({label}, n={n_slice})\nred=higher param -> higher N")
    ax1.grid(alpha=0.3, axis="x")

    top_param = top_rows[0][0]
    v = numeric[top_param]
    ax2.scatter(v, Ns, s=18, alpha=0.6, color="C0")
    ax2.set_xlabel(top_param)
    ax2.set_ylabel("N")
    ax2.set_title(f"N vs top driver: {top_param}")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"attribution_{label.replace('=', '')}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
