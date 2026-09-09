"""Baked detection fit (config/detection_fit_baked.json) — freshness and quality.

The explorer API (/api/plan-a) loads the baked joint detection fit instead of running
detection.fit_detection_model (~2 minutes) in the request path; a Vercel lambda cold
start cannot afford the live fit (60s maxDuration -> 504). These tests pin the contract
that makes that safe:

  1. the baked file is FRESH — fitted from exactly the detection_grids the explorer
     serves today (edit the grids without re-running pipeline/bake_detection_fit.py and
     this fails);
  2. staleness is detected — perturbed grids do NOT load the baked fit;
  3. the baked params still pass the same per-size SSE gate the live fit enforces,
     evaluated with the production survival model.
"""

import copy

import pytest

from plan_a_covert_proj import detection, detection_baked
from scripts.pipelines.batch_rollout import _load_config

CONFIG = "plan_a_covert_proj/config/covert_scenario_branch.yaml"


@pytest.fixture(scope="module")
def config_grids():
    grids = (_load_config(CONFIG) or {}).get("detection_grids")
    assert grids, f"no detection_grids in resolved {CONFIG}"
    return grids


@pytest.fixture(scope="module")
def baked(config_grids):
    fit = detection_baked.load_baked_fit(config_grids)
    assert fit is not None, (
        "config/detection_fit_baked.json is missing or was fitted from DIFFERENT "
        f"detection_grids than {CONFIG} now resolves to. Re-run "
        "python -m plan_a_covert_proj.pipeline.bake_detection_fit"
    )
    return fit


def test_baked_fit_is_fresh_for_config_grids(baked):
    fits, r = baked
    assert r > 0
    assert set(fits) == {10.0, 100.0, 1000.0}


def test_perturbed_grids_do_not_load_baked_fit(config_grids):
    perturbed = copy.deepcopy(config_grids)
    perturbed[0]["rows"][0][1] += 0.01
    assert detection_baked.load_baked_fit(perturbed) is None


def test_grids_fingerprint_stable_across_json_roundtrip(config_grids):
    # The client echoes the grids back through JSON (ints may become floats, tuples
    # become lists); the fingerprint must not change or the API would re-fit live.
    roundtripped = [
        {"size_mw": float(g["size_mw"]),
         "rows": [[float(r[0]),
                   None if r[1] is None else float(r[1]),
                   None if r[2] is None else float(r[2])] for r in g["rows"]]}
        for g in config_grids
    ]
    assert (detection_baked.grids_fingerprint(roundtripped)
            == detection_baked.grids_fingerprint(config_grids))


def test_baked_params_pass_the_live_fit_sse_gate(config_grids, baked):
    fits, r = baked
    for g in detection.normalize_grids(config_grids):
        size = g["size_mw"]
        pts = detection._grid_points(g["rows"])
        sse = sum(((1.0 - float(detection.survival(fits[size], n, t, r))) - d) ** 2
                  for n, t, d in pts)
        assert sse <= detection._FIT_MAX_SSE, (
            f"baked fit at {size:g} MW fails the SSE gate ({sse:.4g}); re-bake"
        )
