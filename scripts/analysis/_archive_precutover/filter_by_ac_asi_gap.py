#!/usr/bin/env python3
"""Write a subset of an MC json, keeping only rollouts whose DEFAULT-trajectory
AC->ASI calendar-time gap is in [lo, hi] years. The ASI time isn't stored in the
records, so it's re-derived per rollout from the stored model_params (run_default).

    python scripts/analysis/filter_by_ac_asi_gap.py --json outputs/covert_mc_500_fracR.json \
        --lo 0.5 --hi 2.0 --out outputs/covert_mc_500_fracR_acasi_0p5_2.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from progress_model import Parameters, load_time_series_data  # noqa: E402
from plan_a_covert_proj.sim import run_default  # noqa: E402
from scripts.pipelines.batch_rollout import _suppress_noise  # noqa: E402

_W = {}
def _init(p, tr): _W["data"] = load_time_series_data(p); _W["tr"] = tr
def _gap(mp):
    try:
        with _suppress_noise():
            ref = run_default(Parameters(**mp), _W["data"], _W["tr"])
        ac = ref.get("ac_time"); asi = (ref.get("milestones", {}).get("ASI", {}) or {}).get("time")
        return (asi - ac) if (ac is not None and asi is not None) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500_fracR.json"))
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.json)); recs = d["records"]
    tr = [float(d["config"]["time_range"][0]), float(d["config"]["time_range"][1])]
    input_path = d["config"]["input_data"]
    if not Path(input_path).exists():
        input_path = str((REPO_ROOT / "inputs" / "input_data.csv").resolve())

    mps = [r["model_params"] for r in recs]
    workers = max(1, min(args.workers, len(recs)))
    if workers == 1:
        _init(input_path, tr); gaps = [_gap(mp) for mp in mps]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(input_path, tr)) as pool:
            gaps = pool.map(_gap, mps, chunksize=4)

    keep = []
    for r, g in zip(recs, gaps):
        if g is not None and args.lo <= g <= args.hi:
            r = dict(r); r["ac_to_asi_gap"] = round(float(g), 3); keep.append(r)
    d["records"] = keep
    wins = np.array([r["covert_wins"] for r in keep], bool)
    d.setdefault("summary", {})
    d["summary"]["n_ok"] = len(keep)
    d["summary"]["covert_win_frac"] = round(float(wins.mean()), 3) if len(keep) else None
    d.setdefault("config", {})["filter"] = f"default AC->ASI gap in [{args.lo}, {args.hi}] yr"
    with open(args.out, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    n_valid = sum(1 for g in gaps if g is not None)
    print(f"AC->ASI gap in [{args.lo},{args.hi}]y: kept {len(keep)}/{n_valid} (of {len(recs)}); "
          f"covert wins {wins.mean()*100:.0f}% (was {100*np.mean([r['covert_wins'] for r in recs]):.0f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
