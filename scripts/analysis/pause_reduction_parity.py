#!/usr/bin/env python3
"""
Production-parity diagnostic for the pause-reduction experiment.

Verifies that feeding a sample's parameters + its frozen / frozen+reduced input
series through the PRODUCTION website API (api/index.py's /api/compute, the
custom_time_series "input time series editor" round-trip) yields the same N as
the local experiment pipeline (freeze_and_reduce + ProgressModel).

The input series sent to the API is built with the exact same
freeze_and_reduce_time_series used by the experiment (itself golden-tested
against the frontend applyPause), so this closes the loop end-to-end:
   sampled params  ->  applyPause-equivalent series  ->  production /api/compute
should reproduce the experiment's ASI times and N.

Start the production API first (on the remote, repo root):
   python3.13 -c "from api.index import app; app.run(host='127.0.0.1', port=5400, threaded=True)" &

Then:
   python3.13 scripts/analysis/pause_reduction_parity.py <run_dir_or_jsonl> \
       --input-data inputs/input_data.csv --url http://127.0.0.1:5400
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import load_time_series_data  # noqa: E402
from progress_model.pause_transform import freeze_and_reduce_time_series  # noqa: E402


def _ts_to_dict(ts):
    return {
        "time": list(map(float, ts.time)),
        "L_HUMAN": list(map(float, ts.L_HUMAN)),
        "inference_compute": list(map(float, ts.inference_compute)),
        "experiment_compute": list(map(float, ts.experiment_compute)),
        "training_compute": list(map(float, ts.training_compute)),
    }


def _post_compute(url, params, ts, time_range, initial_progress, timeout=120):
    payload = {
        "parameters": params,
        "custom_time_series": _ts_to_dict(ts),
        "time_range": list(time_range),
        "initial_progress": float(initial_progress),
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/compute",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    ms = (body.get("milestones") or {}).get("ASI") or {}
    t = ms.get("time")
    try:
        t = float(t)
        return t if np.isfinite(t) else None
    except (TypeError, ValueError):
        return None


def load_records(path):
    path = Path(path)
    if path.is_dir():
        path = path / "pause_reduction_samples.jsonl"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser(description="Production-parity diagnostic")
    ap.add_argument("run", help="Run dir or pause_reduction_samples.jsonl")
    ap.add_argument("--input-data", default="inputs/input_data.csv", help="Base input CSV the run used")
    ap.add_argument("--time-range", default="2017,2200")
    ap.add_argument("--initial-progress", type=float, default=0.0)
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--label", default="gapfrac=0", help="Which pause point to check")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--url", default="http://127.0.0.1:5400")
    ap.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for N parity")
    args = ap.parse_args()

    input_path = args.input_data
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)
    time_range = [float(x) for x in args.time_range.split(",")]

    records = load_records(args.run)
    checked = 0
    failures = 0
    print(f"Parity check: label={args.label}  url={args.url}  input={input_path}")
    print(f"{'sample':>6}  {'pause':>9}  {'N_local':>9}  {'N_prod':>9}  {'relΔN':>9}  "
          f"{'asiF_loc':>9}  {'asiF_prod':>9}  {'asiR_loc':>9}  {'asiR_prod':>9}  status")

    for r in records:
        if checked >= args.n_samples:
            break
        if not (r.get("success") and r.get("baseline_reached")):
            continue
        pt = next((p for p in r.get("points", []) if p.get("label") == args.label
                   and p.get("valid", True) and not p.get("censored") and p.get("N") is not None), None)
        if pt is None:
            continue

        params = r["parameters"]
        pause = pt["pause_time"]
        frozen = freeze_and_reduce_time_series(base, pause)
        reduced = freeze_and_reduce_time_series(
            base, pause,
            training_reduction=args.reduction, experiment_reduction=args.reduction,
            inference_reduction=args.reduction, labor_reduction=args.reduction,
        )
        try:
            asi_f = _post_compute(args.url, params, frozen, time_range, args.initial_progress)
            asi_r = _post_compute(args.url, params, reduced, time_range, args.initial_progress)
        except Exception as e:  # noqa: BLE001
            print(f"{r['sample_id']:>6}  POST failed: {e}")
            failures += 1
            checked += 1
            continue

        n_prod = None
        if asi_f is not None and asi_r is not None and (asi_f - pause) > 0:
            n_prod = (asi_r - pause) / (asi_f - pause)

        n_local = pt["N"]
        rel = abs(n_prod - n_local) / abs(n_local) if (n_prod is not None and n_local) else float("nan")
        ok = (n_prod is not None) and np.isfinite(rel) and rel <= args.rtol
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"{r['sample_id']:>6}  {pause:>9.3f}  {n_local:>9.4f}  "
              f"{(n_prod if n_prod is not None else float('nan')):>9.4f}  {rel:>9.2e}  "
              f"{pt['asi_frozen_time']:>9.3f}  {(asi_f if asi_f else float('nan')):>9.3f}  "
              f"{pt['asi_reduced_time']:>9.3f}  {(asi_r if asi_r else float('nan')):>9.3f}  {status}")
        checked += 1

    print(f"\nChecked {checked} samples at label {args.label}: "
          f"{checked - failures} OK, {failures} mismatched (rtol={args.rtol}).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
