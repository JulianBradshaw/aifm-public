#!/usr/bin/env python3
"""Distribution of when projects hit their target, from a saved MC JSON.

Legal target = TED-AI + tax (intended at target_year = pause + target_lead).
In 'capped' mode the legal can only go as fast as the AIFM ceiling allows, so its
arrival SLIPS past target_year when full effort isn't enough. We show:
  (a) calendar year each project reaches its target,
  (b) the legal SLIP = legal_arrival - intended target_year.

    python scripts/analysis/plot_target_times.py --json outputs/covert_mc_500_capped.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID = (
        "#1f4e9c", "#a23b3b", "#6b8e23", "#777", "#ccc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500_capped.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = json.load(open(args.json))
    recs = d["records"]
    mode = d["config"].get("legal_mode", "backsolve")
    lead = d["config"]["covert_knobs_fixed"]["target_lead_years"]
    if args.out is None:
        args.out = str(REPO_ROOT / "outputs" / f"target_times_{mode}.png")

    legal = np.array([r["legal_ttt"] for r in recs if r["legal_ttt"] is not None])
    n_legal_never = sum(1 for r in recs if r["legal_ttt"] is None)
    covert = np.array([r["covert_ttt"] for r in recs if r["covert_ttt"] is not None])
    n_cov_never = sum(1 for r in recs if r["covert_ttt"] is None)
    # slip = legal arrival - intended target year (pause + lead)
    slip = np.array([r["legal_ttt"] - (r["pause_year"] + lead)
                     for r in recs if r["legal_ttt"] is not None])
    on_time = int(np.sum(slip <= 0.05))
    slipped = int(np.sum(slip > 0.05))

    print(f"legal_mode={mode}; legal reaches target {len(legal)}, never {n_legal_never}")
    print(f"legal slip vs target_year (= pause+{lead}):  on-time {on_time}  slipped {slipped} "
          f"({100*slipped/max(len(slip),1):.0f}%)")
    for p in (50, 75, 90, 95, 99):
        print(f"  slip p{p}: {np.percentile(slip, p):+.2f} yr")
    print(f"  slip max {slip.max():+.2f} yr; legal arrival p50 {np.percentile(legal,50):.2f}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    # (a) calendar arrival year
    lo = min(legal.min(), covert.min() if covert.size else legal.min())
    hi = np.percentile(np.concatenate([legal, covert]) if covert.size else legal, 97)
    bins = np.linspace(lo - 0.5, hi + 1, 45)
    axa.hist(legal, bins=bins, color=AI2030_NAVY, alpha=0.7, label=f"legal → TED-AI+tax ({len(legal)})")
    if covert.size:
        axa.hist(covert, bins=bins, color=AI2030_RED, alpha=0.6, label=f"covert → TED-AI ({len(covert)})")
    axa.set_xlabel("calendar year reaching target")
    axa.set_ylabel("rollouts")
    axa.set_title(f"(a) target-arrival year  ·  covert never reaches: {n_cov_never}", fontsize=10)
    axa.legend(fontsize=8.5)

    # (b) legal slip
    smax = max(0.5, np.percentile(slip, 98))
    sb = np.linspace(-0.5, smax + 0.5, 45)
    axb.hist(np.clip(slip, sb[0], sb[-1]), bins=sb, color=AI2030_OLIVE, alpha=0.8)
    axb.axvline(0.0, color=AI2030_RED, lw=1.6)
    axb.text(0.03, axb.get_ylim()[1] * 0.92, "on time", color=AI2030_RED, fontsize=8)
    axb.set_xlabel("legal SLIP = arrival − intended target_year  (years)")
    axb.set_ylabel("rollouts")
    axb.set_title(f"(b) deal-timeline slip  ·  {slipped}/{len(slip)} slip late "
                  f"({100*slipped/max(len(slip),1):.0f}%)", fontsize=10)

    fig.suptitle(f"Target-arrival distribution  ·  legal_mode = {mode}", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
