#!/usr/bin/env python3
"""
Scatter of the closed-form slowdown law vs actual N.

  N_pred = R^k,  k = (r_software*ece + 1)/(r_software + 1)

ece = experiment_compute_exponent and r_software are read from the rebuilt model
for each sampled draw; actual N is the measured slowdown at the chosen pause label.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from progress_model import config as cfg
from progress_model import ProgressModel, Parameters, load_time_series_data

D = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "gapfrac=0"
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 400
R = 10.0
out = Path(D).parent / "kformula_scatter"; out.mkdir(parents=True, exist_ok=True)

base = load_time_series_data(str(REPO / "inputs" / "input_data.csv")); tr = [2017.0, 2200.0]; ip = 0.0
recs = [json.loads(l) for l in open(D) if l.strip()]
reached = [r for r in recs if r.get("success") and r.get("baseline_reached")]
step = max(1, len(reached) // nmax)
sample = reached[::step][:nmax]

Nact, Npred, ICC = [], [], []
for r in sample:
    pt = next((q for q in r["points"] if q.get("label") == label and q.get("valid", True)
               and not q.get("censored") and q.get("N") is not None), None)
    if not pt or pt["N"] >= 50:
        continue
    p = dict(r["parameters"])
    p.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    p.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)
    try:
        po = Parameters(**p); m = ProgressModel(po, base); m.compute_progress_trajectory(tr, ip)
        rs = float(m.params.r_software); ece = float(m.params.experiment_compute_exponent)
    except Exception:
        continue
    if not (np.isfinite(rs) and np.isfinite(ece)):
        continue
    k = (rs * ece + 1.0) / (rs + 1.0)
    Nact.append(pt["N"]); Npred.append(R ** k); ICC.append(p.get("inv_compute_anchor_exp_cap"))

Nact = np.array(Nact); Npred = np.array(Npred); ICC = np.array(ICC)
m = np.isfinite(Nact) & np.isfinite(Npred) & (Nact > 0) & (Npred > 0)
Nact, Npred, ICC = Nact[m], Npred[m], ICC[m]
lr = np.log10(Nact / Npred)
pear = np.corrcoef(np.log10(Npred), np.log10(Nact))[0, 1]
sp = spearmanr(Npred, Nact).correlation
print("n=%d  Pearson(log)=%.3f  Spearman=%.3f  median Nact/Npred=%.3f" % (len(Nact), pear, sp, np.median(Nact/Npred)))
print("  within 1.1x: %.1f%%  1.25x: %.1f%%  1.5x: %.1f%%" % (
    100*np.mean(np.abs(lr) < np.log10(1.1)), 100*np.mean(np.abs(lr) < np.log10(1.25)), 100*np.mean(np.abs(lr) < np.log10(1.5))))

fig, ax = plt.subplots(figsize=(7.2, 6.8))
sc = ax.scatter(Npred, Nact, c=np.log10(ICC), s=16, alpha=0.6, cmap="viridis")
lim = [min(Npred.min(), Nact.min()) * 0.9, max(Npred.max(), Nact.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=1.2, label="N = R^k  (identity)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted  N = R^((r·ece+1)/(r+1))")
ax.set_ylabel("actual N  (%s)" % label)
ax.set_title("Closed-form slowdown law vs actual N\nPearson(log)=%.3f, median ratio=%.2f, %.0f%% within 1.25x"
             % (pear, np.median(Nact/Npred), 100*np.mean(np.abs(lr) < np.log10(1.25))))
cb = fig.colorbar(sc, ax=ax); cb.set_label("log10 inv_compute_anchor_exp_cap")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(out / "kformula_scatter.png", dpi=140)
print("Wrote", out / "kformula_scatter.png")
