#!/usr/bin/env python3
"""Distribution of the covert-vs-legal lead time, from a saved MC JSON.

    lead = (legal reaches TED-AI + tax)  -  (covert reaches bare TED-AI)
         = legal_ttt - covert_ttt   [years]

lead > 0  => covert gets to TED-AI before the consortium reaches its taxed
target, i.e. the covert project "wins" with that many years to spare.

Covert runs that never reach TED-AI within the horizon have no finite lead;
they are counted separately (they are losses with unbounded lag).

    python scripts/analysis/plot_covert_lead_hist.py [--json PATH] [--clip-lo -30] [--clip-hi 5]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500.json"))
    ap.add_argument("--clip-lo", type=float, default=-30.0,
                    help="Left edge; runs below go into a labeled overflow bin.")
    ap.add_argument("--clip-hi", type=float, default=5.0)
    ap.add_argument("--bin-width", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "covert_lead_hist.png"))
    args = ap.parse_args()

    d = json.load(open(args.json))
    recs = d["records"]
    cov_knobs = d["config"]["covert_knobs_fixed"]
    yba = d["config"].get("years_before_ac", 0.0)
    n = len(recs)

    lead = np.array([r["legal_ttt"] - r["covert_ttt"] for r in recs
                     if r["legal_ttt"] is not None and r["covert_ttt"] is not None])
    n_never = sum(1 for r in recs if r["covert_ttt"] is None)  # covert never hit TED-AI
    win_frac = float(np.mean([bool(r["covert_wins"]) for r in recs]))

    lo, hi, bw = args.clip_lo, args.clip_hi, args.bin_width
    n_underflow = int(np.sum(lead < lo))
    clipped = np.clip(lead, lo, hi)
    edges = np.arange(lo, hi + bw, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(11, 6.4))
    counts, _ = np.histogram(clipped, bins=edges)
    colors = [AI2030_OLIVE if c >= 0 else AI2030_RED for c in centers]
    ax.bar(centers, counts, width=bw * 0.92, color=colors, alpha=0.85, zorder=3)

    ax.axvline(0.0, color=AI2030_NAVY, lw=1.6, zorder=4)
    ax.text(0.15, ax.get_ylim()[1] * 0.96, "tie", color=AI2030_NAVY, fontsize=9, va="top")

    # median marker
    med = float(np.median(lead))
    ax.axvline(np.clip(med, lo, hi), color=AI2030_MUTED, ls="--", lw=1.4, zorder=4)
    ax.text(np.clip(med, lo, hi), ax.get_ylim()[1] * 0.88, f"  median {med:+.1f}y",
            color=AI2030_MUTED, fontsize=9, va="top")

    # annotate the overflow + never-reach tail mass
    notes = []
    if n_underflow:
        notes.append(f"{n_underflow} runs with lead < {lo:g}y (binned at left edge)")
    if n_never:
        notes.append(f"{n_never} runs: covert NEVER reaches TED-AI (∞ behind)")
    if notes:
        ax.text(0.02, 0.80, "\n".join(notes), transform=ax.transAxes, fontsize=8.5,
                color=AI2030_RED, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=AI2030_GRID, alpha=0.9))

    # legend proxies
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=AI2030_OLIVE, label=f"covert wins (lead > 0):  {win_frac*100:.0f}%"),
        Patch(facecolor=AI2030_RED, label="covert loses (lead < 0 or never)"),
    ]

    ax.set_xlim(lo - bw, hi)
    ax.set_xlabel("lead time  =  legal reaches TED-AI+tax  −  covert reaches TED-AI   (years)")
    ax.set_ylabel("number of Monte-Carlo runs")
    h100e = cov_knobs["total_h100e"]; _frac = cov_knobs.get("total_h100e_frac")
    size_str = (f"{_frac*100:.2f}% of frontier R&D compute"
                if _frac is not None
                else (f"{h100e/1e6:g}M" if h100e >= 1e6 else f"{h100e/1e3:g}k") + " H100e")
    fig.suptitle("Covert lead-time distribution  ·  positive = covert reaches TED-AI first",
                 fontsize=16, y=0.97)
    ax.set_title(
        f"{len(lead)} finite of {n} runs · covert knobs fixed: {size_str}, "
        f"steals {cov_knobs['frac_stolen']*100:.0f}%, {cov_knobs.get('dormancy_years',0):g}-yr build, "
        f"{cov_knobs['alignment_tax_ooms']:g}-OOM tax, pause {yba:g}y pre-AC",
        fontsize=10, color=AI2030_MUTED)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9); ax.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"lead: median {med:+.2f}y, n_finite={len(lead)}, underflow={n_underflow}, "
          f"never={n_never}, win_frac={win_frac:.3f} -> {out}")


if __name__ == "__main__":
    main()
