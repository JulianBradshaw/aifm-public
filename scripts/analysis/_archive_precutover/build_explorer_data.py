#!/usr/bin/env python3
"""Build a slim, self-contained JSON for the HTML trajectory explorer:
downsampled pct-to-ASI trajectories + per-parameter percentile positions so each
trajectory can show its "most unusual parameters".

    python scripts/analysis/build_explorer_data.py --json outputs/covert_mc_500_capped.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def downsample(t, y, lo, hi, n=50):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = (t >= lo) & (t <= hi) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size == 0:
        return [], []
    if t.size > n:
        idx = np.unique(np.linspace(0, t.size - 1, n).astype(int))
        t, y = t[idx], y[idx]
    return [round(float(x), 2) for x in t], [round(float(v), 1) for v in y]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "outputs" / "covert_mc_500_capped.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "explorer_data.json"))
    args = ap.parse_args()
    d = json.load(open(args.json))
    recs = d["records"]
    cov_knobs = d["config"]["covert_knobs_fixed"]
    lead_yrs = cov_knobs["target_lead_years"]
    mode = d["config"].get("legal_mode", "backsolve")

    ac = np.array([r["ac_time"] for r in recs], float)
    tedt = np.array([r["ted_ai_time"] for r in recs if r["ted_ai_time"] is not None], float)
    x_lo = float(np.percentile(ac, 5) - 3)
    x_hi = float(np.percentile(tedt, 50) + 22)
    ted_pct_med = float(np.nanmedian([r["ted_ai_pct_to_asi"] for r in recs
                                      if r.get("ted_ai_pct_to_asi") is not None]))

    # varying model params + their sorted distributions (for percentile lookups)
    pkeys = sorted({k for r in recs for k, v in r["model_params"].items()
                    if isinstance(v, (int, float))})
    dists = {}
    for k in pkeys:
        v = np.array([r["model_params"].get(k, np.nan) for r in recs], float)
        v = v[np.isfinite(v)]
        if v.size and v.std() > 1e-9 * (abs(v.mean()) + 1e-12):
            dists[k] = np.sort(v)
    pstats = {k: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
              for k, v in dists.items()}

    def pctile(k, val):
        v = dists[k]
        return round(100.0 * float(np.searchsorted(v, val, side="right")) / len(v), 1)

    out_recs = []
    for r in recs:
        ct, cp = downsample(r["covert_t"], r["covert_pct_to_asi"], x_lo, x_hi)
        lt, lp = downsample(r["legal_t"], r["legal_pct_to_asi"], x_lo, x_hi)
        # unusualness = distance of this run's param percentile from the median (50)
        unu = []
        for k in dists:
            val = r["model_params"].get(k)
            if val is None:
                continue
            p = pctile(k, val)
            unu.append((abs(p - 50.0), k, round(float(val), 4), p))
        unu.sort(reverse=True)
        slip = (r["legal_ttt"] - (r["pause_year"] + lead_yrs)) if r["legal_ttt"] else None
        out_recs.append({
            "id": r["index"], "win": bool(r["covert_wins"]),
            "ac": round(r["ac_time"], 2),
            "ted": (round(r["ted_ai_time"], 2) if r["ted_ai_time"] is not None else None),
            "gap": (round(r["ac_to_ted_gap"], 2) if r.get("ac_to_ted_gap") is not None else None),
            "covt": (round(r["covert_ttt"], 2) if r["covert_ttt"] is not None else None),
            "legt": (round(r["legal_ttt"], 2) if r["legal_ttt"] is not None else None),
            "lead": (round(r["lead_years"], 2) if r["lead_years"] is not None else None),
            "slip": (round(slip, 2) if slip is not None else None),
            "tedp": (round(r["ted_ai_pct_to_asi"], 1) if r.get("ted_ai_pct_to_asi") is not None else None),
            "jump": round(r["ted_ai_e"] - r["e_pause"], 2),
            "ct": ct, "cp": cp, "lt": lt, "lp": lp,
            "unusual": [{"k": k, "v": v, "pct": p} for _, k, v, p in unu[:5]],
            "params": {k: round(float(r["model_params"][k]), 4) for k in dists
                       if r["model_params"].get(k) is not None},
        })

    payload = {
        "meta": {
            "n": len(out_recs), "legal_mode": mode,
            "win_frac": round(float(np.mean([r["win"] for r in out_recs])), 3),
            "x_lo": round(x_lo, 1), "x_hi": round(x_hi, 1), "ted_pct_med": round(ted_pct_med, 1),
            "knobs": {k: cov_knobs[k] for k in cov_knobs},
        },
        "pstats": pstats,
        "records": out_recs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    sz = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}  ({len(out_recs)} records, {len(dists)} varying params, {sz:.2f} MB)")


if __name__ == "__main__":
    main()
