#!/usr/bin/env python3
"""Bake the joint detection fit into config/detection_fit_baked.json.

  python -m plan_a_covert_proj.pipeline.bake_detection_fit \
        [--covert-config plan_a_covert_proj/config/covert_scenario_branch.yaml]

Runs detection.fit_detection_model (~2 minutes) on the resolved config's
``detection_grids`` and writes the result, keyed by a fingerprint of the grids, where
detection_baked.load_baked_fit finds it. Re-run this whenever the elicited grids change;
tests/test_detection_baked_fit.py fails if the baked file goes stale. The explorer API
(/api/plan-a) refuses nothing without it — it just falls back to the live fit, which is
far too slow for a Vercel lambda cold start (the plan-a 504 of 2026-07)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plan_a_covert_proj import detection_baked  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config  # noqa: E402

DEFAULT_CONFIG = ROOT / "plan_a_covert_proj" / "config" / "covert_scenario_branch.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--covert-config", default=str(DEFAULT_CONFIG),
                    help="covert config yaml whose resolved detection_grids to fit")
    args = ap.parse_args()

    cfg = _load_config(args.covert_config) or {}
    grids = cfg.get("detection_grids")
    if not grids:
        raise SystemExit(f"no detection_grids in resolved config {args.covert_config}")

    print(f"fitting detection model to {len(grids)} size grids "
          f"(this takes ~2 minutes) ...")
    t0 = time.time()
    fits, r = detection_baked.bake_fit(grids)
    print(f"done in {time.time() - t0:.0f}s; shared per-site hazard r = {r:.4g}/yr")
    for size, p in sorted(fits.items()):
        print(f"  {size:g} MW: thetaC={p['thetaC']:.4g} eta={p['eta']:.4g} "
              f"k={p['k']:.4g} pi={p['pi']:g}")
    print(f"wrote {detection_baked.BAKED_FIT_PATH}")


if __name__ == "__main__":
    main()
