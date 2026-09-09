#!/usr/bin/env python3
"""
Test how well 10^ece predicts the slowdown N, where ece (experiment_compute_exponent)
is recomputed per-sample in closed form from the stored sampled parameters
(inv_compute_anchor_exp_cap, inf_compute_asymptote, inf_labor_asymptote).

Prediction: N ≈ R^a, a ≈ ece, R=10  =>  N ≈ 10^ece.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from progress_model import config as cfg
from progress_model.ces_functions import (
    compute_rho_from_asymptotes, compute_experiment_compute_exponent_from_anchor,
)

D = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "gapfrac=0"
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(D).parent / "ece_predict"
out.mkdir(parents=True, exist_ok=True)

N, PRED, ICC = [], [], []
for line in open(D):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if not (r.get("success") and r.get("baseline_reached")):
        continue
    p = r["parameters"]
    icc = p.get("inv_compute_anchor_exp_cap"); ica = p.get("inf_compute_asymptote"); ila = p.get("inf_labor_asymptote")
    if not (icc and ica and ila):
        continue
    pt = next((q for q in r["points"] if q.get("label") == label and q.get("valid", True)
               and not q.get("censored") and q.get("N") is not None), None)
    if not pt or pt["N"] >= 50:
        continue
    try:
        rho = compute_rho_from_asymptotes(ila, ica)
        ece = compute_experiment_compute_exponent_from_anchor(ica, ila, (cfg.REFERENCE_COMPUTE_CHANGE, 1.0 / icc), rho)
    except Exception:
        continue
    if not np.isfinite(ece):
        continue
    N.append(pt["N"]); PRED.append(10.0 ** ece); ICC.append(icc)

N = np.array(N); PRED = np.array(PRED); ICC = np.array(ICC)
mask = np.isfinite(N) & np.isfinite(PRED) & (N > 0) & (PRED > 0)
n_bad = int((~mask).sum())
N, PRED, ICC = N[mask], PRED[mask], ICC[mask]
if n_bad:
    print(f"(dropped {n_bad} samples with non-finite/non-positive ece prediction)")
ratio = N / PRED
lr = np.log10(ratio)
pear = np.corrcoef(np.log10(PRED), np.log10(N))[0, 1]
from scipy.stats import spearmanr
sp = spearmanr(PRED, N).correlation
print(f"label={label}  n={len(N)}")
print(f"  Pearson(log10 N, log10 10^ece) = {pear:.4f}   Spearman = {sp:.4f}")
print(f"  median N/pred = {np.median(ratio):.2f}   (geomean {10**np.mean(lr):.2f})")
print(f"  frac within 1.25x: {100*np.mean(np.abs(lr)<np.log10(1.25)):.1f}%   within 1.5x: {100*np.mean(np.abs(lr)<np.log10(1.5)):.1f}%   within 2x: {100*np.mean(np.abs(lr)<np.log10(2)):.1f}%")
# how well does ALSO a constant*pred do (fit a multiplicative factor)?
c = 10 ** np.median(lr)
print(f"  best single multiplier c: N ≈ {c:.2f} * 10^ece  -> then within 1.25x: {100*np.mean(np.abs(lr-np.log10(c))<np.log10(1.25)):.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
ax1.scatter(PRED, N, s=8, alpha=0.3, color="C0")
lim = [min(PRED.min(), N.min())*0.9, max(PRED.max(), N.max())*1.1]
ax1.plot(lim, lim, "k--", lw=1, label="N = 10^ece")
ax1.plot(lim, [c*x for x in lim], "C3-", lw=1, label=f"N = {c:.2f}·10^ece")
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("predicted 10^ece"); ax1.set_ylabel("actual N")
ax1.set_title(f"N vs 10^ece ({label})\nPearson(log)={pear:.3f}, median N/pred={np.median(ratio):.2f}")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")

ax2.scatter(ICC, ratio, s=8, alpha=0.3, color="C0")
ax2.axhline(1, color="k", ls="--", lw=1)
ax2.axhline(c, color="C3", lw=1, label=f"median {c:.2f}")
ax2.set_xscale("log"); ax2.set_xlabel("inv_compute_anchor_exp_cap"); ax2.set_ylabel("N / 10^ece")
ax2.set_title("residual ratio vs inv_cc")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(out / "ece_predict.png", dpi=130)
print("Wrote", out / "ece_predict.png")
