#!/usr/bin/env python3
"""Unconditional vs win-conditional distributions for the key takeoff variables.
Shows how the worlds where the covert wins differ from the full ensemble.

    python scripts/analysis/dist_conditional.py --json outputs/covert_mc_500_capped.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default  # noqa: E402
from scripts.pipelines.batch_rollout import _suppress_noise  # noqa: E402
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_RED, AI2030_OLIVE, AI2030_MUTED, AI2030_GRID = (
        "#1f4e9c", "#a23b3b", "#6b8e23", "#777", "#ccc")

_W = {}
def _init(p, tr): _W["data"] = load_time_series_data(p); _W["tr"] = tr
def _ac_ted(mp):
    try:
        with _suppress_noise():
            ref = run_default(Parameters(**mp), _W["data"], _W["tr"])
        return [ref.get("ac_e"), ref.get("ted_ai_e"), ref.get("asi_e") if "asi_e" in ref else None]
    except Exception:
        return [None, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500_capped.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "dist_conditional.png"))
    args = ap.parse_args()
    d = json.load(open(args.json)); recs = d["records"]
    tr = [float(d["config"]["time_range"][0]), float(d["config"]["time_range"][1])]
    input_path = d["config"]["input_data"]
    if not Path(input_path).exists():
        input_path = str((REPO_ROOT / "inputs" / "input_data.csv").resolve())

    mps = [r["model_params"] for r in recs]
    workers = max(1, min(args.workers, len(recs)))
    if workers == 1:
        _init(input_path, tr); at = [_ac_ted(mp) for mp in mps]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(input_path, tr)) as pool:
            at = pool.map(_ac_ted, mps, chunksize=4)
    ac = np.array([a for a, t, s in at], float)
    ted = np.array([t for a, t, s in at], float)
    asi = np.array([r.get("asi_e", np.nan) for r in recs], float)  # asi_e is stored per record
    win = np.array([r["covert_wins"] for r in recs], bool)

    oom = ted - ac
    asited = asi - ted
    swr = np.array([r["model_params"].get("software_progress_rate_at_reference_year", np.nan) for r in recs], float)
    gap = np.array([r["ac_to_ted_gap"] if r.get("ac_to_ted_gap") is not None else np.nan for r in recs], float)

    panels = [
        ("AC→TED-AI distance (OOMs eff. compute)", oom, False),
        ("software_progress_rate_at_reference_year", swr, False),
        ("AC→TED-AI gap (calendar years)", gap, True),
        ("ASI−TED-AI distance (OOMs eff. compute)", asited, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (lab, v, logx) in zip(axes.ravel(), panels):
        good = np.isfinite(v) & (np.abs(v) < 1e6)
        va, vw = v[good], v[good & win]
        lo, hi = np.percentile(va, 1), np.percentile(va, 99)
        if logx:
            lo = max(lo, np.percentile(va[va > 0], 1));
            bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 34); ax.set_xscale("log")
        else:
            bins = np.linspace(lo, hi, 34)
        ax.hist(np.clip(va, bins[0], bins[-1]), bins=bins, density=True, color=AI2030_MUTED,
                alpha=0.45, label=f"all runs (n={good.sum()})")
        ax.hist(np.clip(vw, bins[0], bins[-1]), bins=bins, density=True, color=AI2030_OLIVE,
                alpha=0.55, label=f"covert wins (n={(good & win).sum()})")
        ax.axvline(np.median(va), color=AI2030_MUTED, ls="--", lw=1.3)
        ax.axvline(np.median(vw), color="#46600f", ls="--", lw=1.6)
        ax.set_xlabel(lab); ax.set_ylabel("density")
        ax.set_title(f"median: all {np.median(va):.2f}  ·  wins {np.median(vw):.2f}", fontsize=9.5)
        ax.legend(fontsize=8.5)
    fig.suptitle("Takeoff variables: unconditional vs conditional on covert win  (capped legal, 500-run)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}  (wins={int(win.sum())}/{len(recs)})")


if __name__ == "__main__":
    main()
