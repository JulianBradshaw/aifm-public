#!/usr/bin/env python3
"""Characterize the covert-win worlds: are they degenerate parameter corners, or a
coherent (fast, software-heavy takeoff) regime? Reads a saved MC JSON.

Panel (a): regime scatter — AC->TED effective-compute jump (x) vs AC->TED calendar
gap (y, log); color = win/lose; ring = cap-bit (legal slipped).
Panel (b): the one real discriminator, software_progress_rate_at_reference_year,
win vs lose, with the full-distribution range shaded (shows wins are upper-mid, not tail).
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
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "characterize_wins.png"))
    args = ap.parse_args()
    recs = json.load(open(args.json))["records"]
    lead = json.load(open(args.json))["config"]["covert_knobs_fixed"]["target_lead_years"]

    win = np.array([r["covert_wins"] for r in recs], bool)
    jump = np.array([r["ted_ai_e"] - r["e_pause"] for r in recs], float)        # OOMs
    gap = np.array([r["ac_to_ted_gap"] if r["ac_to_ted_gap"] is not None else np.nan
                    for r in recs], float)
    swr = np.array([r["model_params"].get("software_progress_rate_at_reference_year", np.nan)
                    for r in recs], float)
    slip = np.array([(r["legal_ttt"] - (r["pause_year"] + lead)) if r["legal_ttt"] else np.nan
                     for r in recs], float)
    bit = slip > 0.05

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    axa.set_yscale("log")
    XMAX = 30.0
    n_out = int(np.sum(jump > XMAX))
    jd = np.clip(jump, None, XMAX)
    axa.scatter(jd[~win], gap[~win], s=14, color=AI2030_RED, alpha=0.45, label="covert loses")
    axa.scatter(jd[win], gap[win], s=16, color=AI2030_OLIVE, alpha=0.7, label="covert wins")
    axa.scatter(jd[bit], gap[bit], s=70, facecolors="none", edgecolors=AI2030_NAVY, lw=1.1,
                label="cap bit (legal slipped)")
    axa.set_xlim(0, XMAX)
    if n_out:
        axa.text(0.98, 0.02, f"{n_out} slip-world outlier(s) clipped (jump>{XMAX:g})",
                 transform=axa.transAxes, ha="right", va="bottom", fontsize=7.5, color=AI2030_MUTED)
    axa.set_xlabel("AC→TED-AI jump  (OOMs of effective compute)")
    axa.set_ylabel("AC→TED-AI gap  (calendar years, log)")
    axa.set_title("(a) win regime: big jump + fast takeoff (top-right-ish)\n"
                  "cap-bit worlds (rings) sit in the slow regime the covert loses", fontsize=9.5)
    axa.legend(fontsize=8.5, loc="upper left")

    av = swr[np.isfinite(swr)]
    bins = np.linspace(np.percentile(av, 1), np.percentile(av, 99), 40)
    axb.hist(swr[~win], bins=bins, color=AI2030_RED, alpha=0.5, density=True, label="loses")
    axb.hist(swr[win], bins=bins, color=AI2030_OLIVE, alpha=0.6, density=True, label="wins")
    for q, ls in [(10, ":"), (50, "--"), (90, ":")]:
        axb.axvline(np.percentile(av, q), color=AI2030_MUTED, ls=ls, lw=1.0)
    axb.set_xlabel("software_progress_rate_at_reference_year\n(dotted/dashed = full-dist p10/p50/p90)")
    axb.set_ylabel("density")
    axb.set_title("(b) the one real discriminator — wins are upper-mid, not tail", fontsize=9.5)
    axb.legend(fontsize=8.5)

    fig.suptitle("Are covert-win worlds degenerate? — regime + parameter position", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wins={int(win.sum())}  median win jump={np.median(jump[win]):.1f} OOM  "
          f"gap={np.nanmedian(gap[win]):.2f}y  swr={np.nanmedian(swr[win]):.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
