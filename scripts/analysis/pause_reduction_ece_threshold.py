#!/usr/bin/env python3
"""When does ece (experiment_compute_exponent) exceed 1? (=> N > R, super-linear)
Computes ece in closed form per sampled draw and characterizes the ece>1 region."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from progress_model import config as cfg
from progress_model.ces_functions import (
    compute_rho_from_asymptotes, compute_experiment_compute_exponent_from_anchor)

D = sys.argv[1]
ECE, ICC, RATIO, ICA, ILA = [], [], [], [], []
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
    try:
        rho = compute_rho_from_asymptotes(ila, ica)
        ece = compute_experiment_compute_exponent_from_anchor(ica, ila, (cfg.REFERENCE_COMPUTE_CHANGE, 1.0/icc), rho)
    except Exception:
        continue
    if np.isfinite(ece):
        ECE.append(ece); ICC.append(icc); RATIO.append(ica/ila); ICA.append(ica); ILA.append(ila)
ECE = np.array(ECE); ICC = np.array(ICC); RATIO = np.array(RATIO)
print("n=%d" % len(ECE))
print("  fraction ece>1 (=> N>R, super-linear): %.1f%%" % (100*np.mean(ECE > 1)))
print("  median ece=%.3f   median inv_cc=%.2f   median asymptote ratio (ica/ila)=%.2f" % (np.median(ECE), np.median(ICC), np.median(RATIO)))
# crossing in inv_cc by asymptote-ratio bins
print("  inv_cc at which ece crosses 1, by asymptote ratio bin:")
for lo, hi in [(0,0.5),(0.5,1),(1,1.5),(1.5,3),(3,99)]:
    m = (RATIO>=lo)&(RATIO<hi)
    if m.sum() < 30: continue
    sup = m & (ECE>1)
    # approx crossing: min inv_cc with ece>1 in this bin (robust-ish: 10th pct of inv_cc among ece>1)
    cross = np.percentile(ICC[sup],10) if sup.sum()>5 else float("nan")
    print("    ica/ila in [%g,%g): n=%4d  frac ece>1=%4.0f%%  ~inv_cc crossing=%.2f" % (lo,hi,m.sum(),100*np.mean(ECE[m]>1),cross))

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(ICC, ECE, c=np.clip(RATIO,0,3), s=10, alpha=0.5, cmap="coolwarm")
ax.axhline(1.0, color="k", ls="--", lw=1.2, label="ece=1  (N=R, slowdown=10)")
xs = np.logspace(0, 1, 50)
ax.plot(xs, np.log10(xs), "g:", lw=1.5, label="skeleton ece=log10(inv_cc)  [k_=0]")
ax.set_xscale("log"); ax.set_xlabel("inv_compute_anchor_exp_cap"); ax.set_ylabel("ece (experiment_compute_exponent)")
ax.set_title("When does ece exceed 1?  (ece>1 => N>R)\nfrac ece>1 = %.0f%%" % (100*np.mean(ECE>1)))
cb = fig.colorbar(sc, ax=ax); cb.set_label("inf_compute_asymptote / inf_labor_asymptote")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
out = Path(D).parent / "ece_threshold"; out.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out / "ece_threshold.png", dpi=140)
print("Wrote", out / "ece_threshold.png")
