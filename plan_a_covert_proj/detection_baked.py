#!/usr/bin/env python3
"""Baked (precomputed) detection fit — the I/O companion to ``detection.py``.

``detection.fit_detection_model`` is deterministic given the elicited
``detection_grids`` but EXPENSIVE (a joint multi-start Nelder-Mead, ~2 minutes): far too
slow for the explorer API's request path, where a Vercel lambda cold start would re-run
it inside a 60s budget (the plan-a 504 of 2026-07). Since the grids live in config and
almost never change, the fit is baked once (``pipeline/bake_detection_fit.py``) into
``config/detection_fit_baked.json`` keyed by :func:`grids_fingerprint`, and consumers
load it with :func:`load_baked_fit`.

The fingerprint check makes staleness impossible to hit silently: grids that don't
match the baked file return ``None`` and the caller decides whether to pay for a live
fit. This module deliberately lives OUTSIDE ``detection.py``: detection.py stays pure
(no file I/O) and is content-hashed into pipeline run IDs (run.py ``_code_inputs``), so
keeping the cache layer here avoids invalidating every cached dataset."""
from __future__ import annotations

import hashlib
import json
import os

from plan_a_covert_proj import detection

BAKED_FIT_PATH = os.path.join(os.path.dirname(__file__), "config",
                              "detection_fit_baked.json")


def grids_fingerprint(grids) -> str:
    """Canonical sha256 of the NORMALIZED grids — the identity of the fit inputs.
    Hashing the normalized form (floats, size-sorted) makes the key stable across the
    YAML -> JSON -> client -> request round-trip (ints vs floats, list vs tuple), so a
    fit baked from the config matches the grids echoed back by the explorer UI."""
    norm = sorted(detection.normalize_grids(grids), key=lambda g: g["size_mw"])
    blob = json.dumps([[g["size_mw"], [list(r) for r in g["rows"]]] for g in norm])
    return hashlib.sha256(blob.encode()).hexdigest()


def load_baked_fit(grids, path: str | None = None):
    """``(fits, r)`` from the baked JSON if it exists and was fitted from exactly these
    ``grids`` (by :func:`grids_fingerprint`); ``None`` on a missing file or a
    fingerprint mismatch — never a silently-stale fit."""
    path = path or BAKED_FIT_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        baked = json.load(f)
    if baked.get("grids_fingerprint") != grids_fingerprint(grids):
        return None
    return detection.normalize_fits(baked["fits"]), float(baked["r"])


def bake_fit(grids, path: str | None = None):
    """Run the joint fit on ``grids`` and write the baked JSON (fingerprint + per-size
    frailty table + shared hazard r). Returns ``(fits, r)``."""
    path = path or BAKED_FIT_PATH
    fits, r = detection.fit_detection_model(grids)
    payload = {
        "_comment": "Precomputed detection.fit_detection_model output. Regenerate with "
                    "pipeline/bake_detection_fit.py whenever detection_grids change "
                    "(tests/test_detection_baked_fit.py enforces freshness).",
        "grids_fingerprint": grids_fingerprint(grids),
        "fits": {repr(size): params for size, params in sorted(fits.items())},
        "r": r,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return fits, r
