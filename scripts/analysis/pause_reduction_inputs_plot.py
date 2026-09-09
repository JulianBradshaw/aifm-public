#!/usr/bin/env python3
"""
Visual verification of the pause (freeze + reduce) transform on the INPUT series.

Plots the four input channels (L_HUMAN, inference_compute, experiment_compute,
training_compute) for the base series and for the frozen (R=1) and frozen+reduced
(R) variants produced by freeze_and_reduce_time_series — the same transform the
experiment uses (golden-tested against the frontend applyPause).

Surfaces the subtleties to check:
  * the point INSERTED at the (possibly fractional) pause instant carries the
    FULL, un-reduced value; reduction takes effect only strictly after it
    (drawn as a star);
  * interpolation convention at the pause instant: EXPONENTIAL (log-space) for
    L_HUMAN / inference / experiment (so on a log axis the inserted point lands
    on the straight segment between its CSV neighbors), LINEAR for
    training_compute (already log10 FLOP/yr);
  * reduction magnitude: /R in linear space for the first three, -log10(R) for
    training_compute.

Base CSV grid points are drawn as open circles so you can see exactly where the
inserted pause point falls relative to them.

The transform depends only on the pause time, not on model parameters, so a pause
time is all that distinguishes rollouts. Pass explicit --pauses or pull them from
a run's records via --from-run/--label.
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

from progress_model import load_time_series_data  # noqa: E402
from progress_model.pause_transform import freeze_and_reduce_time_series  # noqa: E402

CHANNELS = [
    ("L_HUMAN", "Human labor (FTEs)", "log"),
    ("inference_compute", "Inference compute (H100be)", "log"),
    ("experiment_compute", "Experiment compute (H100be)", "log"),
    ("training_compute", "Training compute (OOMs log10 FLOP/yr)", "linear"),
]


def _pauses_from_run(run, label, n):
    p = Path(run)
    if p.is_dir():
        p = p / "pause_reduction_samples.jsonl"
    pauses = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("success") and r.get("baseline_reached")):
                continue
            pt = next((q for q in r.get("points", []) if q.get("label") == label
                       and q.get("valid", True)), None)
            if pt is not None:
                pauses.append((r["sample_id"], float(pt["pause_time"])))
            if len(pauses) >= n:
                break
    return pauses


def main():
    ap = argparse.ArgumentParser(description="Plot input series under pause/freeze/reduce")
    ap.add_argument("--input-data", default="inputs/input_data.csv")
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--pauses", default=None, help="Comma-separated explicit pause years")
    ap.add_argument("--from-run", default=None, help="Run dir/jsonl to pull pause times from")
    ap.add_argument("--label", default="gapfrac=0")
    ap.add_argument("--n", type=int, default=3, help="How many pause times from the run")
    ap.add_argument("--window", type=float, default=4.0, help="x half-window (years) around pause")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "input_checks"))
    args = ap.parse_args()

    input_path = args.input_data
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)

    if args.pauses:
        pauses = [(None, float(x)) for x in args.pauses.split(",") if x.strip()]
    elif args.from_run:
        pauses = _pauses_from_run(args.from_run, args.label, args.n)
    else:
        # Default: a fractional pause and an on-grid pause near 2030.
        pauses = [(None, 2029.6), (None, 2030.0)]
    if not pauses:
        print("No pause times."); sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ncol = len(pauses)
    fig, axes = plt.subplots(4, ncol, figsize=(5.2 * ncol, 12), squeeze=False)
    bt = np.asarray(base.time)

    for j, (sid, pause) in enumerate(pauses):
        frozen = freeze_and_reduce_time_series(base, pause)
        reduced = freeze_and_reduce_time_series(
            base, pause, training_reduction=args.reduction, experiment_reduction=args.reduction,
            inference_reduction=args.reduction, labor_reduction=args.reduction)
        x0, x1 = pause - args.window, pause + args.window
        for i, (ch, ylabel, yscale) in enumerate(CHANNELS):
            ax = axes[i][j]
            b = np.asarray(getattr(base, ch))
            fz = np.asarray(getattr(frozen, ch))
            rd = np.asarray(getattr(reduced, ch))
            ft = np.asarray(frozen.time)
            rt = np.asarray(reduced.time)
            # base grid points (open circles) + line
            ax.plot(bt, b, "-o", color="0.6", ms=4, mfc="none", lw=1.0, label="base (CSV grid)")
            ax.plot(ft, fz, "--", color="C2", lw=1.4, label="frozen (R=1)")
            ax.plot(rt, rd, "-", color="C0", lw=1.6, label=f"frozen+÷{args.reduction:g}")
            # the inserted pause-instant point (full value) — first reduced.time == pause
            ins = np.argmin(np.abs(rt - pause))
            ax.plot([rt[ins]], [rd[ins]], "*", color="C3", ms=14, zorder=5,
                    label="inserted pause pt (full value)")
            ax.axvline(pause, color="k", ls=":", lw=1.0)
            ax.set_xlim(x0, x1)
            ax.set_yscale(yscale)
            # y-limits from base within window for readability
            wsel = (bt >= x0) & (bt <= x1)
            if np.any(wsel):
                lo, hi = np.min(b[wsel]), np.max(b[wsel])
                if yscale == "log" and lo > 0:
                    ax.set_ylim(lo / (args.reduction * 2), hi * 2)
                else:
                    pad = 0.5
                    ax.set_ylim(lo - pad, hi + pad)
            if i == 0:
                title = f"pause={pause:.3f}" + (f"  (sample {sid})" if sid is not None else "")
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == 3:
                ax.set_xlabel("year")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)

    fig.suptitle("Input series under pause: base vs frozen (R=1) vs frozen+reduce — "
                 "log-interp (labor/inf/exp), linear-interp (training); inserted point = full value",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = out_dir / "input_transform_check.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Pauses plotted: {[round(p,3) for _, p in pauses]}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
