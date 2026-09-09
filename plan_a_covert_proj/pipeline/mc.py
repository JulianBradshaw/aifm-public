#!/usr/bin/env python3
"""Build the covert experiment's canonical dataset: N Monte-Carlo rollouts that
sample the AI-futures model parameters (model.yaml) AND the covert knobs
(covert.yaml), each per draw via the same distribution machinery (fixed dists just
return their value, so an all-fixed covert.yaml reproduces the old held-fixed
behavior). Each record stores EVERYTHING the figures need — trajectories, the full
milestone map, uplift/pct series, the sampled model_params/covert_params, and the
legal effort-vs-ceiling ratios — so no downstream figure ever re-runs the model.

    python -m plan_a_covert_proj.pipeline.mc --num 500 --out plan_a_covert_proj/outputs/runs/<id>/dataset.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from numpy.random import SeedSequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as mcfg_mod  # noqa: E402
from progress_model import Parameters, load_time_series_data, TimeSeriesData  # noqa: E402
from progress_model.progress_rate import progress_rate_at_time  # noqa: E402
from plan_a_covert_proj.sim import (run_default, simulate_covert, _tc_ooms, _make_legal_data,  # noqa: E402
                                     bottom_up_covert_fraction, _BOTTOM_UP_KEYS)
from plan_a_covert_proj import detection  # noqa: E402
from plan_a_covert_proj import deal_hazard  # noqa: E402
from scripts.pipelines.batch_rollout import (  # noqa: E402
    _load_config, _sample_parameter_dict, _get_median_parameter_dict, _suppress_noise,
    _calibrate_gaussian_copula_latent_corr)

# --- covert.yaml validation -------------------------------------------------
# The sampler is lenient: a misspelled `dist`/`value` key silently samples to None,
# and an unknown knob is silently sampled-and-ignored. These tables turn those
# (and missing required knobs) into loud load-time errors instead.
# total_h100e / total_h100e_frac are conditionally required per compute_mode and
# validated inside simulate_covert, so they're known but not always-required here.
_KNOWN_COVERT_KNOBS = {
    "compute_mode", "total_h100e", "total_h100e_frac", "facility_w_per_h100e",
    "dormancy_years", "construction_years", "frac_stolen", "max_hw_rate", "legal_scaling_duration",
    "legal_strategy", "max_sw_rate",
    "alignment_tax_progress_years", "setup_period_years", "setup_sw_gain_progress_months",
    "optimal_site_mw",
    # scenario-timing / physics knobs (formerly CLI-only --years-before-ac / --legal-mode)
    "years_before_ac", "legal_mode",
    # post-arrival software the legal keeps discovering as an alignment-capabilities
    # externality (OOM/yr), which the covert steals like the normal discovered software.
    "post_scaling_externalities_rate",
    # mandatory minimum software during scaling: SW grows at >= this fraction of TSP growth.
    "min_sw_frac_of_tsp",
}
_REQUIRED_COVERT_KNOBS = {
    "compute_mode", "frac_stolen", "max_hw_rate", "legal_scaling_duration", "legal_strategy",
    "alignment_tax_progress_years", "dormancy_years", "setup_period_years",
    "setup_sw_gain_progress_months",
    "facility_w_per_h100e", "optimal_site_mw",
    "years_before_ac", "legal_mode", "post_scaling_externalities_rate",
}
_LEGAL_MODES = {"capped", "backsolve"}
_COMPUTE_MODES = {"frontier_relative", "absolute", "bottom_up_estimate"}
_KNOWN_DISTS = {"fixed", "uniform", "normal", "lognormal", "shifted_lognormal", "beta", "choice",
                "cdf_points"}


def _validate_bottom_up_config(bu_dists: dict) -> None:
    """Validate the bottom-up compute config (covert_compute_*.yaml's
    bottom_up_compute_parameters): every required diversion knob present, and each spec a
    well-formed dist (mirrors the covert-config checks so the sampler can't silently swallow
    a typo). Required knobs are sim._BOTTOM_UP_KEYS."""
    errors = []
    if not bu_dists:
        raise ValueError("bottom_up_compute_config has no 'bottom_up_compute_parameters'")
    missing = set(_BOTTOM_UP_KEYS) - set(bu_dists)
    if missing:
        errors.append(f"missing bottom_up_compute_parameters: {sorted(missing)}")
    unknown = set(bu_dists) - set(_BOTTOM_UP_KEYS)
    if unknown:
        errors.append(f"unknown bottom_up_compute_parameters (typo?): {sorted(unknown)}; "
                      f"known: {sorted(_BOTTOM_UP_KEYS)}")
    for name, spec in bu_dists.items():
        if not isinstance(spec, dict):
            errors.append(f"{name!r}: spec must be a mapping (dist + params), got {type(spec).__name__}")
            continue
        kind = spec.get("dist")
        if kind is None:
            errors.append(f"{name!r}: missing 'dist'")
        elif kind not in _KNOWN_DISTS:
            errors.append(f"{name!r}: unknown dist {kind!r} (known: {sorted(_KNOWN_DISTS)})")
        elif kind == "fixed" and spec.get("value") is None:
            errors.append(f"{name!r}: dist=fixed needs a non-null 'value'")
    if errors:
        raise ValueError("Invalid bottom_up_compute_parameters:\n  - " + "\n  - ".join(errors))


def _parse_silver_bullet(block):
    """Parse the detection_silver_bullet config (fixed sigmoid spec) into
    {p0, p1, delta, C}, or None when absent (silver bullet globally disabled). p0 is the
    value at the TIME-MIDPOINT between end of pause and legal used->TED-AI; p1 at
    used->TED-AI + delta. The per-rollout ceiling-logistic is built later in _run_sample,
    since both anchor TIMES come from each rollout's legal used->TED-AI crossing. Halts on
    missing keys or a bad fixed-value ordering (time-ordering t1>t0 holds by construction)."""
    if not block:
        return None
    try:
        cfg = {
            "p0": float(block["p_mid_pause_to_ted"]),
            "p1": float(block["p_used_ted_plus_d"]),
            "delta": float(block["delta_years"]),
            "C": float(block["ceiling"]),
        }
    except (KeyError, TypeError) as e:
        raise ValueError(
            "detection_silver_bullet must have numeric keys p_mid_pause_to_ted, "
            f"p_used_ted_plus_d, delta_years, ceiling; got {block!r}") from e
    if not (0.0 < cfg["p0"] < cfg["p1"] < cfg["C"] <= 1.0):
        raise ValueError(
            "detection_silver_bullet requires 0 < p_mid_pause_to_ted < p_used_ted_plus_d < "
            f"ceiling <= 1; got {cfg}")
    if cfg["delta"] < 0:
        raise ValueError(f"detection_silver_bullet delta_years must be >= 0, got {cfg['delta']}")
    return cfg


_DEAL_BD_PARAM_DISTS = {"breakdown_compute_reduction", "post_breakdown_compute_growth_rate"}
# Optional (defaulted when absent so older configs keep working). breakdown_compute_mode is
# categorical — 'frontier_at_pause' (default: the leading company restarts from a share of
# the pre-deal frontier; deal-era compute is effectively destroyed) or 'world_at_breakdown'
# (compute survives the dissolution; the share is of the world's compute at the collapse).
_DEAL_BD_OPTIONAL_DISTS = {"breakdown_compute_mode"}
_DEAL_BD_MODES = ("frontier_at_pause", "world_at_breakdown")


def _parse_deal_breakdown(block) -> dict | None:
    """Parse/validate the covert yaml's optional `deal_breakdown:` block into
    {hazard, hazard_raw, dists}. None when absent or `enabled: false`.

    The block holds (a) the hazard spec that samples WHEN the deal breaks down
    (deal_hazard.parse_hazard_spec — the clock starts at the DEAL START, i.e. the
    pause), and (b) the sim's per-rollout breakdown knobs as dist specs, sampled per
    draw exactly like covert_parameters (the same knobs the frontend's Deal Collapse
    tab exposes): breakdown_compute_reduction, post_breakdown_compute_growth_rate, and
    optionally breakdown_compute_mode (fixed categorical; defaults to 'frontier_at_pause').
    Fails loudly on unknown/missing keys and malformed dists (mirrors
    _validate_covert_config so a typo can't silently sample None)."""
    if not block:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"deal_breakdown must be a mapping, got {type(block).__name__}")
    if block.get("enabled") is False:
        return None
    errors = []
    known = _DEAL_BD_PARAM_DISTS | _DEAL_BD_OPTIONAL_DISTS | {"enabled", "hazard", "min_survival_years"}
    unknown = set(block) - known
    if unknown:
        errors.append(f"unknown deal_breakdown key(s) (typo?): {sorted(unknown)}; "
                      f"known: {sorted(known)}")
    missing = _DEAL_BD_PARAM_DISTS - set(block)
    if missing:
        errors.append(f"missing required deal_breakdown parameter dist(s): {sorted(missing)}")
    for name in (_DEAL_BD_PARAM_DISTS | _DEAL_BD_OPTIONAL_DISTS) & set(block):
        spec = block[name]
        if not isinstance(spec, dict):
            errors.append(f"{name!r}: spec must be a mapping (dist + params), got {type(spec).__name__}")
            continue
        kind = spec.get("dist")
        if kind is None:
            errors.append(f"{name!r}: missing 'dist' (sampler would default to fixed -> None)")
        elif kind not in _KNOWN_DISTS:
            errors.append(f"{name!r}: unknown dist {kind!r} (known: {sorted(_KNOWN_DISTS)})")
        elif kind == "fixed" and spec.get("value") is None:
            errors.append(f"{name!r}: dist=fixed needs a non-null 'value' (typo'd value key?)")
    # breakdown_compute_mode is categorical: only a fixed dist with a known value makes sense.
    mode_spec = block.get("breakdown_compute_mode")
    if isinstance(mode_spec, dict) and mode_spec.get("dist") == "fixed" \
            and mode_spec.get("value") is not None and mode_spec["value"] not in _DEAL_BD_MODES:
        errors.append(f"breakdown_compute_mode: value must be one of {list(_DEAL_BD_MODES)}, "
                      f"got {mode_spec['value']!r}")
    elif isinstance(mode_spec, dict) and mode_spec.get("dist") not in (None, "fixed"):
        errors.append(f"breakdown_compute_mode: categorical knob — only dist=fixed is supported, "
                      f"got dist={mode_spec.get('dist')!r}")
    if "hazard" not in block:
        errors.append("missing 'hazard' spec (type: linear_plus_decay, points, decay)")
    if errors:
        raise ValueError("Invalid deal_breakdown block in covert yaml:\n  - " + "\n  - ".join(errors))
    hazard = deal_hazard.parse_hazard_spec(block["hazard"])   # raises on malformed spec
    # "deal survives for at least N years": condition the hazard on no breakdown before
    # then (draws come from the truncated survival curve S(t)/S(N); breakdowns land >= N
    # years after the deal starts and P(never) rises accordingly).
    min_years = float(block.get("min_survival_years", 0.0))
    if not (math.isfinite(min_years) and min_years >= 0.0):
        raise ValueError(f"deal_breakdown min_survival_years must be a finite value >= 0, "
                         f"got {block.get('min_survival_years')!r}")
    return {"hazard": hazard, "hazard_raw": block["hazard"], "min_survival_years": min_years,
            "dists": {k: block[k] for k in
                      (_DEAL_BD_PARAM_DISTS | _DEAL_BD_OPTIONAL_DISTS) & set(block)}}


def _validate_covert_config(covert_dists: dict) -> None:
    """Fail loudly on covert.yaml mistakes the sampler would otherwise swallow:
    unknown/misspelled knobs, missing required knobs, and malformed dist specs
    (a typo'd `dist` or `value` key makes the sampler return None silently)."""
    errors = []
    unknown = set(covert_dists) - _KNOWN_COVERT_KNOBS
    if unknown:
        errors.append(f"unknown covert_parameters key(s) (typo or renamed knob?): "
                      f"{sorted(unknown)}; known: {sorted(_KNOWN_COVERT_KNOBS)}")
    missing = _REQUIRED_COVERT_KNOBS - set(covert_dists)
    if missing:
        errors.append(f"missing required covert_parameters: {sorted(missing)}")
    for name, spec in covert_dists.items():
        if not isinstance(spec, dict):
            errors.append(f"{name!r}: spec must be a mapping (dist + params), got {type(spec).__name__}")
            continue
        kind = spec.get("dist")
        if kind is None:
            errors.append(f"{name!r}: missing 'dist' (sampler would default to fixed -> None)")
        elif kind not in _KNOWN_DISTS:
            errors.append(f"{name!r}: unknown dist {kind!r} (known: {sorted(_KNOWN_DISTS)})")
        elif kind == "fixed" and spec.get("value") is None:
            errors.append(f"{name!r}: dist=fixed needs a non-null 'value' (typo'd value key?)")
    lm = covert_dists.get("legal_mode")
    if isinstance(lm, dict) and lm.get("dist") == "fixed" and lm.get("value") not in _LEGAL_MODES:
        errors.append(f"legal_mode value must be one of {sorted(_LEGAL_MODES)}, got {lm.get('value')!r}")
    cmode = covert_dists.get("compute_mode")
    if isinstance(cmode, dict) and cmode.get("dist") == "fixed" and cmode.get("value") not in _COMPUTE_MODES:
        errors.append(f"compute_mode value must be one of {sorted(_COMPUTE_MODES)}, got {cmode.get('value')!r}")
    if errors:
        raise ValueError("Invalid covert_parameters in covert.yaml:\n  - " + "\n  - ".join(errors))


def _site_size_range(covert_dists: dict, grids) -> tuple:
    """Validate ``optimal_site_mw`` against the detection_grids size range and return
    that range ``(gmin, gmax)``. The per-site power feeds the frailty interpolation,
    which CLAMPS (constant) outside the elicited grid — so a distribution that could
    sample beyond it would silently extrapolate. We require the support to stay inside:
    a fixed value within [gmin, gmax], or a non-fixed dist with min/max within it."""
    sizes = sorted(float(g["size_mw"]) for g in (grids or []))
    if not sizes:
        raise ValueError("covert config has no 'detection_grids' to bound optimal_site_mw")
    gmin, gmax = sizes[0], sizes[-1]
    spec = covert_dists.get("optimal_site_mw")
    if not isinstance(spec, dict):
        raise ValueError("covert config missing required 'optimal_site_mw' (per-site power, MW)")
    rng = f"the detection_grids range [{gmin:g}, {gmax:g}] MW"
    if spec.get("dist") == "fixed":
        v = spec.get("value")
        if v is None or not (gmin <= float(v) <= gmax):
            raise ValueError(f"optimal_site_mw fixed value {v} must lie within {rng}")
    else:
        lo, hi = spec.get("min"), spec.get("max")
        if lo is None or hi is None:
            raise ValueError(f"optimal_site_mw ({spec.get('dist')}) must supply 'min' and 'max' "
                             f"within {rng} (else the frailty interpolation extrapolates)")
        if not (gmin <= float(lo) and float(hi) <= gmax):
            raise ValueError(f"optimal_site_mw bounds [{lo}, {hi}] must lie within {rng}")
    return gmin, gmax


_W = {}


def _init(input_path, all_dists, model_names, corr, covert_dists, covert_corr, det_fits,
          det_r_fit, det_size_range, det_sb_cfg, tr, hor, taste, solve_legal_compute,
          plan_a_path=None, pause_med=None, pause_end_med=None, bottom_up_dists=None,
          deal_bd=None):
    # years_before_ac and legal_mode are now covert_parameters (sampled per draw), so
    # they ride along in covert_dists -> cov rather than as run-wide worker globals.
    # det_fits/det_r_fit are the JOINT detection fit (fit_detection_model): per-size
    # frailty + ONE shared per-site operational hazard r fit to the elicited grids.
    # plan_a_data (+ the median pause anchors) drive the per-rollout Plan-A input series;
    # None => legal/covert use the single (default) series.
    # deal_bd (parsed deal_breakdown block) => each rollout samples a breakdown time from
    # the hazard curve; None => the deal always holds (current behaviour).
    _W.update(dict(data=load_time_series_data(input_path), all_dists=all_dists,
                   model_names=set(model_names), corr=corr, covert_dists=covert_dists,
                   covert_corr=covert_corr, det_fits=det_fits, det_r_fit=float(det_r_fit),
                   det_size_range=det_size_range,
                   det_sb_cfg=det_sb_cfg, tr=tr, hor=hor, taste=taste,
                   solve_legal_compute=solve_legal_compute,
                   plan_a_data=(load_time_series_data(plan_a_path) if plan_a_path else None),
                   pause_med=pause_med, pause_end_med=pause_end_med,
                   bottom_up_dists=bottom_up_dists, deal_bd=deal_bd))


def _ds_idx(n, m=250):
    """Indices to subsample an array of length n to <= m points (endpoints kept).
    Stored trajectories are for plotting only; scalars (ttt, lead) come from the
    full-resolution sim, so this never affects computed outcomes."""
    return np.unique(np.linspace(0, n - 1, min(m, n)).astype(int)) if n > 1 else np.arange(n)


def _ds_idx_time(times, m=250, dense_until=None, dense_frac=0.7):
    """Like _ds_idx but TIME-aware: keep the window up to ``dense_until`` densely sampled
    (``dense_frac`` of the budget) and the tail sparse. Without this, a trajectory that runs
    to t_end (e.g. the discovered legal track once post-scaling externalities extend it to
    2100) gets index-uniform sampling that is coarse over the short, steep race window — which
    then mis-renders in the per-year figures. Endpoints kept."""
    times = np.asarray(times, float); n = times.size
    if n <= m or dense_until is None:
        return _ds_idx(n, m)
    early = np.where(times <= dense_until)[0]
    if early.size == 0 or early[-1] >= n - 1:
        return _ds_idx(n, m)
    me = min(max(int(m * dense_frac), 2), early.size)
    ml = max(m - me, 2)
    ie = np.linspace(early[0], early[-1], me)
    il = np.linspace(early[-1], n - 1, ml)
    return np.unique(np.concatenate([ie, il]).astype(int))


def _e_to_uplift(ref, E):
    # Prefer the dense uniform-E (E, uplift) map (sim._dense_capability_maps) — uplift
    # evaluated exactly at every grid point. The native ODE samples are sparse in E across
    # the steep takeoff, so interpolating them draws chords under the true convex curve
    # (visible as scalloping on the log-scale spaghetti plots). Same preference as the
    # webapp path (api/covert_formatters.py).
    ec, up = ref.get("uplift_map_e"), ref.get("uplift_map_u")
    if ec is None or up is None:
        ec, up = ref["effective_compute"], ref["uplift"]
    ec = np.asarray(ec, float); up = np.asarray(up, float)
    o = np.argsort(ec)
    return np.interp(np.asarray(E, float), ec[o], up[o])


def _loginterp(t, ts, vs):
    """log10-linear interpolation (np.interp, clamped at the ends), scalar or array."""
    return 10.0 ** np.interp(t, ts, np.log10(np.maximum(np.asarray(vs, float), 1e-300)))


_PLAN_A_COLS = ("inference_H100e", "experiment_H100e", "training_H100e")


def build_plan_a_series(default, plan_a, pause, pause_end, pause_med, pause_end_med, t_end):
    """Construct the per-rollout legal/covert INPUT series: the default series up to this
    rollout's pause, then Plan A's compute ramp RELATIVIZED to the pause.

    Per compute column, for k = 0, 1, 2, ... (years after this rollout's pause-end E):
        G(t<=pause) = default(t)                                   # shared counterfactual
        G(E + k)    = [default(pause)/default(pause_med)] * plan_a(pause_end_med + k)
    i.e. Plan A's trajectory, time-shifted so the median pause-end maps to this rollout's
    pause-end, and scaled by how this rollout's default compute at its pause compares to
    the median's. A single (log-linear) segment spans the pause window [pause, E]. L_HUMAN
    follows the default throughout (Plan A is a compute policy, not a labor change).

    `default`/`plan_a` are TimeSeriesData; returns a new TimeSeriesData on the constructed
    (generally non-integer, unevenly-spaced) grid — the model interpolates it, so that's
    fine. For the median rollout (pause==pause_med) the result is Plan A re-anchored exactly.
    """
    tD = np.asarray(default.time, float)
    grid = np.concatenate([tD[tD < pause - 1e-9], [pause],
                           pause_end + np.arange(0.0, np.floor(t_end - pause_end) + 1.0)])
    grid = grid[grid <= t_end + 1e-9]
    cols = {}
    for c in _PLAN_A_COLS:
        vD = np.asarray(getattr(default, c), float); vA = np.asarray(getattr(plan_a, c), float)
        scale = _loginterp(pause, tD, vD) / _loginterp(pause_med, tD, vD)
        vals = np.where(
            grid <= pause + 1e-9,
            _loginterp(grid, tD, vD),                                   # default (incl. pause point)
            scale * _loginterp(pause_end_med + (grid - pause_end), np.asarray(plan_a.time, float), vA),
        )
        cols[c] = vals
    L = _loginterp(grid, tD, np.asarray(default.L_HUMAN, float))
    return TimeSeriesData(grid, L, cols["inference_H100e"], cols["experiment_H100e"],
                          cols["training_H100e"])


def _legal_ceiling_ratios(ref, params, data, leg_t, leg_e, leg_sw, tp, setup, RS_0, r_sw, off,
                          hw_rate, t_end):
    """Fold in the physics check: max(legal research effort / AIFM ceiling) over the
    legal active window, split into the setup-windfall period and the sustained ramp.
    Ceiling = progress_rate_at_time(...)[1], evaluated with the legal's own resources
    (whole compute budget scaled at hw_rate), not the default frontier trajectory.
    hw_rate=None (the actual-training-series strategy) => use the real `data` directly."""
    leg_t = np.asarray(leg_t, float); leg_sw = np.asarray(leg_sw, float)
    legal_data = data if hw_rate is None else _make_legal_data(data, tp, hw_rate, setup, t_end)
    rs = RS_0 * 10.0 ** (leg_sw / r_sw)
    re = np.gradient(rs, leg_t)
    active = np.where(re > 1e-12 * max(rs.max(), 1e-30))[0]
    if active.size < 3:
        return None, None
    idx = np.unique(np.linspace(active[0], active[-1], min(250, active[-1] - active[0] + 1)).astype(int))
    logE_raw = np.asarray(leg_e, float)[idx] - off
    ratios = np.empty(idx.size)
    for j, i in enumerate(idx):
        ceil = max(progress_rate_at_time(float(leg_t[i]), [float(logE_raw[j]), float(max(rs[i], 1e-30))],
                                         legal_data, params, RS_0=RS_0)[1], 1e-30)
        ratios[j] = re[i] / ceil
    in_setup = leg_t[idx] <= (tp + setup + 1e-9)
    smax = float(np.nanmax(ratios[in_setup & (re[idx] > 0)])) if np.any(in_setup & (re[idx] > 0)) else 0.0
    post = (~in_setup) & (re[idx] > 0)
    pmax = float(np.nanmax(ratios[post])) if np.any(post) else 0.0
    return smax, pmax


def _run_sample(task):
    idx, ss = task
    data, tr = _W["data"], _W["tr"]
    rng = np.random.default_rng(ss)
    # Independent per-rollout child streams: covert-knob sampling, the detection draw,
    # bottom-up sizing, and the deal-breakdown draw each get their own, so none perturbs
    # the model-parameter stream (rng) nor each other as knobs are added/removed.
    # (spawn(4)'s first three children are identical to spawn(3)'s, so pre-existing
    # runs' streams are unchanged.)
    child_cov, child_det, child_bu, child_deal = ss.spawn(4)
    with _suppress_noise():
        try:
            sm = _sample_parameter_dict(_W["all_dists"], rng, _W["corr"])
            # Covert knobs are sampled per draw, exactly like the model parameters
            # (fixed dists just return their value). Held as `cov` below.
            cov = _sample_parameter_dict(_W["covert_dists"], np.random.default_rng(child_cov), _W["covert_corr"])
            # bottom_up_estimate sizing: sample the diversion params (own stream) and combine
            # into a diverted fraction of world compute, passed to simulate_covert.
            bu_fracs = None; bu = None; bu_eval_offset = 0.0; bu_duration = None
            if cov["compute_mode"] == "bottom_up_estimate":
                bu = _sample_parameter_dict(_W["bottom_up_dists"], np.random.default_rng(child_bu))
                # natural-base diversion fractions (decommission->stock, smuggle/domestic->flow)
                bu_fracs = bottom_up_covert_fraction(bu)
                # declaration->pause gap + diversion-window length set the window: decommissioning
                # acts on the stock at the window start, smuggling/domestic on the window's flow.
                bu_eval_offset = float(bu.get("declaration_to_pause_years", 0.0))
                bu_duration = (float(bu["diversion_duration"]) if "diversion_duration" in bu else None)
            mp = {k: v for k, v in sm.items() if k in _W["model_names"]}
            mp.setdefault("horizon_extrapolation_type", _W["hor"])
            mp.setdefault("taste_schedule_type", _W["taste"])
            params = Parameters(**mp)
            ref = run_default(params, data, tr)
            if ref["ac_time"] is None or ref["ted_ai_e"] is None:
                return None
            # Use the model-prepared copy (progress_at_aa set, slopes converted) for the
            # covert/legal sim; run_default leaves the caller's params untouched.
            params = ref["prepared_params"]
            tp = ref["ac_time"] - cov["years_before_ac"]
            setup = cov["setup_period_years"]
            # Legal/covert INPUT series: the default series up to THIS rollout's pause, then
            # Plan A's ramp relativized to the pause (build_plan_a_series). run_default above
            # defines AC/pause & milestone targets from the DEFAULT series; the legal & covert
            # race in the Plan-A world. No Plan A configured => single series (current behaviour).
            lc_data = data
            if _W.get("plan_a_data") is not None:
                lc_data = build_plan_a_series(data, _W["plan_a_data"], tp, tp + setup,
                                              _W["pause_med"], _W["pause_end_med"], tr[1])
            # Deal breakdown (optional): sample WHEN the deal collapses from the hazard
            # curve (clock starts at the DEAL START = the pause, tp). tau=None => the deal
            # never breaks down (finite total hazard) — run exactly as a no-breakdown
            # rollout. The sim keys the breakdown off the END of the setup pause, so a tau
            # inside the setup window clamps to the earliest representable breakdown
            # (the moment the pause ends). A tau beyond t_end is passed through — the sim
            # treats it as no breakdown within the horizon (censored).
            deal_bd = _W.get("deal_bd")
            bd_knobs = None; bd_tau = None; bd_u = None; bd_enabled = False
            if deal_bd is not None:
                rng_bd = np.random.default_rng(child_deal)
                bd_knobs = _sample_parameter_dict(deal_bd["dists"], rng_bd)
                bd_u = float(rng_bd.random())
                bd_tau = deal_hazard.sample_breakdown_time(
                    deal_bd["hazard"], bd_u, min_years=deal_bd["min_survival_years"])
                bd_enabled = bd_tau is not None
            rec = simulate_covert(
                params, lc_data, tp, alignment_tax_progress_years=cov["alignment_tax_progress_years"],
                # legal_scaling_duration is the POST-pause window: target = end-of-pause + duration
                target_year=tp + setup + cov["legal_scaling_duration"], max_hw_rate=cov["max_hw_rate"],
                compute_mode=cov["compute_mode"], frac_stolen=cov["frac_stolen"],
                total_h100e=cov.get("total_h100e"), total_h100e_frac=cov.get("total_h100e_frac"),
                bottom_up_fracs=bu_fracs, bottom_up_eval_offset_years=bu_eval_offset,
                bottom_up_diversion_duration=bu_duration,
                bottom_up_power=bu,
                dormancy_years=cov["dormancy_years"],
                construction_years=cov.get("construction_years", 0.0) or 0.0,
                setup_period_years=setup,
                setup_sw_gain_progress_months=cov["setup_sw_gain_progress_months"], t_end=tr[1], ref=ref,
                return_trajectories=True, legal_mode=cov["legal_mode"],
                legal_strategy=cov["legal_strategy"], max_sw_rate=cov.get("max_sw_rate"),
                post_scaling_externalities_rate=cov["post_scaling_externalities_rate"],
                min_sw_frac_of_tsp=cov.get("min_sw_frac_of_tsp", 0.2),
                solve_legal_compute=_W.get("solve_legal_compute", False),
                enable_deal_breakdown=bd_enabled,
                deal_breakdown_years_after_pause_end=(max(float(bd_tau) - setup, 0.0)
                                                      if bd_enabled else None),
                breakdown_compute_reduction=(float(bd_knobs["breakdown_compute_reduction"])
                                             if bd_enabled else None),
                breakdown_compute_mode=(str(bd_knobs.get("breakdown_compute_mode") or "frontier_at_pause")
                                        if bd_enabled else "frontier_at_pause"),
                post_breakdown_compute_growth_rate=(float(bd_knobs["post_breakdown_compute_growth_rate"])
                                                    if bd_enabled else 0.0))
        except Exception:
            return None

    off, RS_0, r_sw = ref["rebase_offset"], ref["RS_0"], params.r_software
    cov_e = np.asarray(rec["covert_e"], float); leg_e = np.asarray(rec["legal_e"], float)
    leg_sw = np.asarray(rec["legal_sw"], float)
    cov_u = _e_to_uplift(ref, cov_e); leg_u = _e_to_uplift(ref, leg_e)
    e_pause = float(np.interp(tp, ref["times"], ref["effective_compute"]))
    asi_e = (ref["milestones"].get("ASI", {}) or {}).get("e")
    span = (float(asi_e) - e_pause) if asi_e is not None else None

    def _pct(E):
        return (100.0 * (np.asarray(E, float) - e_pause) / span) if (span and span > 0) else np.full(np.shape(E), np.nan)
    cov_pct, leg_pct = _pct(cov_e), _pct(leg_e)
    ted_pct = (100.0 * (ref["ted_ai_e"] - e_pause) / span) if (span and span > 0) else np.nan
    setup_ratio, post_ratio = _legal_ceiling_ratios(ref, params, lc_data, rec["legal_t"], leg_e,
                                                     leg_sw, tp, setup, RS_0, r_sw, off,
                                                     rec.get("legal_hw_rate_solved", cov["max_hw_rate"]), tr[1])
    # downsample stored trajectories (plotting only) to keep the dataset lean
    ci = _ds_idx(cov_e.size); li = _ds_idx(leg_e.size)
    cov_t = np.asarray(rec["covert_t"], float); leg_t = np.asarray(rec["legal_t"], float)
    # legal USED (deployed-capability) glidepath — reaches bare TED-AI; on its own grid.
    used_e = np.asarray(rec["legal_used_e"], float); used_t = np.asarray(rec["legal_used_t"], float)
    used_u = _e_to_uplift(ref, used_e); used_pct = _pct(used_e); ui = _ds_idx(used_e.size)

    # Detection: cheap post-stage on the physics result. The covert's facility power
    # (MW) is derived from its realized compute; one detection event is drawn from a
    # child of this rollout's seed (independent of the parameter-sampling stream, so
    # the draw is stable as model knobs are added/removed). t_risk = years the covert
    # runs from the pause until it reaches TED-AI (None when it never does).
    fac = float(cov["facility_w_per_h100e"])
    # Per-site slip-up hazard = the JOINT-FIT shared rate (fit_detection_model). Not a
    # config knob: it is fit to the elicited grids like the frailty parameters.
    det_r = _W["det_r_fit"]
    h100e_used = rec.get("total_h100e_used")
    # bottom_up_estimate sizes the facility per-source (supplement §3.3: decommissioned /
    # domestic chips draw more W/H100e than smuggled US chips, at proxy years tied to the
    # diversion window) — use the sim's mix-weighted MW and record the effective W/H100e.
    # Other compute modes keep the flat facility_w_per_h100e conversion.
    if rec.get("bottom_up_facility_mw") is not None:
        size_mw = float(rec["bottom_up_facility_mw"])
        if rec.get("bottom_up_w_per_h100e") is not None:
            fac = float(rec["bottom_up_w_per_h100e"])
    else:
        size_mw = (float(h100e_used) * fac / 1e6) if h100e_used else None
    cttt = rec["covert_ttt"]
    t_risk_goal = (float(cttt) - float(tp)) if cttt is not None else None
    # Per-site power the covert splits its total compute across (sampled knob). Clamp to
    # the elicited grid range as a hard guard so the frailty interp never extrapolates
    # (config validation already requires the dist's support to lie inside this range).
    gmin, gmax = _W["det_size_range"]
    site_mw = float(np.clip(cov["optimal_site_mw"], gmin, gmax))
    det_u = float(np.random.default_rng(child_det).random())
    # Per-rollout silver-bullet sigmoid, both anchor TIMES in years-after-pause-start:
    #   t0 = time-midpoint between end of pause and legal USED->TED-AI
    #   t1 = legal USED->TED-AI + Δ
    # If the legal used capability never reaches TED-AI, there is no silver bullet (inert).
    sb_cfg = _W["det_sb_cfg"]
    sb_spec = None
    if sb_cfg is not None:
        used_ttt = rec.get("legal_used_ttt")
        if used_ttt is not None and np.isfinite(used_ttt):
            t_pause_end = float(rec["setup_period_years"])      # end of pause
            t_used_ted = float(used_ttt) - float(tp)            # legal used -> TED-AI
            t0 = 0.5 * (t_pause_end + t_used_ted)               # time-midpoint
            t1 = t_used_ted + sb_cfg["delta"]
            sb_spec = detection.build_silver_bullet(
                sb_cfg["p0"], t0, sb_cfg["p1"], t1, sb_cfg["C"])
    if size_mw is not None:
        # The detection clock starts when CONSTRUCTION starts: during the dormancy phase
        # no facility exists, so P(detected) = 0 and the facility-frailty clock is shifted
        # by dormancy_years. The silver-bullet sigmoid stays anchored on the pause clock
        # (it tracks the LEGAL project's progress, not the covert's schedule).
        det = detection.sample_detection(det_u, size_mw, t_risk_goal, r=det_r,
                                         site_mw=site_mw, fits=_W["det_fits"],
                                         silver_bullet=sb_spec,
                                         dormancy_years=float(cov["dormancy_years"]))
    else:
        det = {"t_det": float("inf"), "detected_before_goal": False,
               "survival_at_goal": float("nan"), "n_sites": float("nan")}
    _t_det = det["t_det"]; _sag = det["survival_at_goal"]; _nst = det["n_sites"]
    out = {
        "index": idx,
        "model_params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in mp.items()},
        "covert_params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in cov.items()},
        "milestones": {k: {"e": float(m["e"]), "time": float(m["time"])}
                       for k, m in (ref.get("milestones") or {}).items()},
        "ac_time": float(ref["ac_time"]),
        "ted_ai_time": (float(ref["ted_ai_time"]) if ref["ted_ai_time"] is not None else None),
        "ted_ai_e": float(ref["ted_ai_e"]),
        "asi_e": (float(asi_e) if asi_e is not None else None),
        "ac_to_ted_gap": (float(ref["ac_to_ted_gap"]) if ref.get("ac_to_ted_gap") is not None else None),
        "e_pause": e_pause, "pause_year": float(tp),
        "covert_wins": bool(rec["covert_wins"]),
        "covert_ttt": (float(rec["covert_ttt"]) if rec["covert_ttt"] is not None else None),
        "legal_ttt": (float(rec["legal_ttt"]) if rec["legal_ttt"] is not None else None),
        # legal USED capability -> TED-AI crossing (silver-bullet anchor; None if never reached)
        "legal_used_ttt": (float(rec["legal_used_ttt"]) if rec.get("legal_used_ttt") is not None else None),
        # legal DISCOVERED capability -> TED-AI + tax crossing (None if never reached)
        "legal_discovered_ttt": (float(rec["legal_discovered_ttt"]) if rec.get("legal_discovered_ttt") is not None else None),
        # counterfactual (deal-holds) crossings: when the legal glidepaths WOULD have
        # reached their targets had no breakdown frozen them (== the realized crossings
        # when the deal never breaks down within the horizon)
        "legal_would_ttt": (float(rec["legal_would_ttt"]) if rec.get("legal_would_ttt") is not None else None),
        "legal_used_would_ttt": (float(rec["legal_used_would_ttt"]) if rec.get("legal_used_would_ttt") is not None else None),
        "legal_discovered_would_ttt": (float(rec["legal_discovered_would_ttt"]) if rec.get("legal_discovered_would_ttt") is not None else None),
        "lead_years": (float(rec["lead_years"]) if rec["lead_years"] is not None else None),
        # covert time-to-TED-AI from the pause split / default world's (see simulate_covert)
        "covert_slowdown": (float(rec["covert_slowdown"]) if rec.get("covert_slowdown") is not None else None),
        "legal_mode": rec.get("legal_mode"),
        "legal_strategy": rec.get("legal_strategy"),
        "max_sw_rate": rec.get("max_sw_rate"),
        "sw_rate_solved": (float(rec["sw_rate_solved"]) if rec.get("sw_rate_solved") is not None else None),
        # realized scaling-window SW rate = actual sw-path slope (tuned rate + mandatory
        # min_sw_frac_of_tsp floor + effort clamps), excl. the setup windfall
        "sw_rate_incl_floor": (float(rec["sw_rate_incl_floor"]) if rec.get("sw_rate_incl_floor") is not None else None),
        "used_sw_rate_incl_floor": (float(rec["used_sw_rate_incl_floor"]) if rec.get("used_sw_rate_incl_floor") is not None else None),
        "sw_floor_rate_avg": (float(rec["sw_floor_rate_avg"]) if rec.get("sw_floor_rate_avg") is not None else None),
        "legal_theta_solved": (float(rec["legal_theta_solved"]) if rec.get("legal_theta_solved") is not None else None),
        "legal_hw_rate_solved": (float(rec["legal_hw_rate_solved"]) if rec.get("legal_hw_rate_solved") is not None else None),
        "compute_mode": rec.get("compute_mode"),
        "total_h100e_used": (float(rec["total_h100e_used"]) if rec.get("total_h100e_used") is not None else None),
        "total_h100e_frac": rec.get("total_h100e_frac"),
        # bottom_up_estimate: per-source decomposition (computed in simulate_covert from the natural-
        # base fractions x the window stock/flow). *_frac are fractions of world stock at the
        # declaration (sum to bottom_up_world_frac); *_h100e are the absolute amounts.
        "bottom_up_world_frac": rec.get("bottom_up_world_frac"),
        "bottom_up_world_h100e_at_pause": rec.get("bottom_up_world_h100e_at_pause"),
        "bottom_up_stock_at_start": rec.get("bottom_up_stock_at_start"),
        "bottom_up_diverted_pool": rec.get("bottom_up_diverted_pool"),
        "bottom_up_decommission_frac": rec.get("bottom_up_decommission_frac"),
        "bottom_up_domestic_frac": rec.get("bottom_up_domestic_frac"),
        "bottom_up_smuggled_frac": rec.get("bottom_up_smuggled_frac"),
        "bottom_up_decommission_h100e": rec.get("bottom_up_decommission_h100e"),
        "bottom_up_domestic_h100e": rec.get("bottom_up_domestic_h100e"),
        "bottom_up_smuggled_h100e": rec.get("bottom_up_smuggled_h100e"),
        # per-source facility W/H100e (PUE included) behind covert_size_mw; the top-level
        # facility_w_per_h100e is the mix-weighted effective value in bottom-up runs.
        "bottom_up_smuggled_w": rec.get("bottom_up_smuggled_w"),
        "bottom_up_decommission_w": rec.get("bottom_up_decommission_w"),
        "bottom_up_domestic_w": rec.get("bottom_up_domestic_w"),
        # diversion timing knobs (relative to the pause): the declaration gap (window end) and the
        # diversion-window length.
        "bottom_up_declaration_to_pause_years": (float(bu["declaration_to_pause_years"])
                                                 if bu and "declaration_to_pause_years" in bu else None),
        "bottom_up_diversion_duration": (float(bu["diversion_duration"])
                                         if bu and "diversion_duration" in bu else None),
        "legal_target_e": float(rec["legal_target_e"]),
        "ted_ai_uplift": float(_e_to_uplift(ref, ref["ted_ai_e"])),
        "ted_ai_pct_to_asi": (float(ted_pct) if np.isfinite(ted_pct) else None),
        "legal_setup_ceiling_ratio": setup_ratio,
        "legal_sustained_ceiling_ratio": post_ratio,
        # detection (orthogonal to covert_wins; combine at analysis time as wins & ~detected)
        "facility_w_per_h100e": fac,
        "covert_size_mw": (round(size_mw, 3) if size_mw is not None else None),
        "detection_r": det_r,
        "detection_u": round(det_u, 6),
        "t_det": (round(float(_t_det), 4) if np.isfinite(_t_det) else None),   # yr after pause; None=never
        "detected_before_goal": bool(det["detected_before_goal"]),
        "survival_at_goal": (round(float(_sag), 6) if np.isfinite(_sag) else None),
        "n_sites": (float(_nst) if np.isfinite(_nst) else None),
        "covert_t": [round(float(x), 4) for x in cov_t[ci]],
        "covert_e": [round(float(x), 4) for x in cov_e[ci]],
        "covert_uplift": [float(cov_u[i]) for i in ci],
        "covert_pct_to_asi": [round(float(cov_pct[i]), 3) for i in ci],
        "legal_t": [round(float(x), 4) for x in leg_t[li]],
        "legal_e": [round(float(x), 4) for x in leg_e[li]],
        "legal_uplift": [float(leg_u[i]) for i in li],
        "legal_pct_to_asi": [round(float(leg_pct[i]), 3) for i in li],
        # legal hardware (training-system performance, log10 FLOP/yr) on the discovered grid
        "legal_tsp": [round(float(x), 4) for x in np.asarray(rec["legal_tsp"], float)[li]],
        # legal USED (deployed) glidepath, on its own grid — for spaghetti plots that want
        # deployed capability rather than the discovered (TED-AI + tax) track.
        "legal_used_t": [round(float(x), 4) for x in used_t[ui]],
        "legal_used_e": [round(float(x), 4) for x in used_e[ui]],
        "legal_used_uplift": [float(used_u[i]) for i in ui],
        "legal_used_pct_to_asi": [round(float(used_pct[i]), 3) for i in ui],
    }
    # Deal breakdown (fields present only when the run has a deal_breakdown block).
    # tau = sampled years after DEAL START (None => the deal never breaks down);
    # deal_breakdown_t = the sim's effective absolute breakdown year (tau clamped to the
    # end of the setup pause; may exceed t_end, in which case nothing broke within the
    # horizon and there is no leading track). The leading company's trajectory gets the
    # same uplift/pct mapping as the other tracks.
    if _W.get("deal_bd") is not None:
        out["deal_breakdown_u"] = round(bd_u, 6)
        out["deal_breakdown_tau"] = (round(float(bd_tau), 4) if bd_tau is not None else None)
        out["deal_breakdown_t"] = (float(rec["deal_breakdown_t"])
                                   if rec.get("deal_breakdown_t") is not None else None)
        out["breakdown_compute_reduction"] = (float(bd_knobs["breakdown_compute_reduction"])
                                              if bd_enabled else None)
        out["breakdown_compute_mode"] = (str(bd_knobs.get("breakdown_compute_mode") or "frontier_at_pause")
                                         if bd_enabled else None)
        out["post_breakdown_compute_growth_rate"] = (float(bd_knobs["post_breakdown_compute_growth_rate"])
                                                     if bd_enabled else None)
        out["leading_ttt"] = (float(rec["leading_ttt"]) if rec.get("leading_ttt") is not None else None)
        # counterfactual (deal-holds) legal glidepaths, unfrozen — present only when a
        # breakdown truncated the realized tracks (else identical to legal_*/legal_used_*)
        if rec.get("legal_would_t") is not None:
            for pre in ("legal_would", "legal_used_would"):
                w_t = np.asarray(rec[f"{pre}_t"], float); w_e = np.asarray(rec[f"{pre}_e"], float)
                w_u = _e_to_uplift(ref, w_e); w_pct = _pct(w_e); wi = _ds_idx(w_e.size)
                out[f"{pre}_t"] = [round(float(x), 4) for x in w_t[wi]]
                out[f"{pre}_e"] = [round(float(x), 4) for x in w_e[wi]]
                out[f"{pre}_uplift"] = [float(w_u[i]) for i in wi]
                out[f"{pre}_pct_to_asi"] = [round(float(w_pct[i]), 3) for i in wi]
        if rec.get("leading_t") is not None:
            ld_t = np.asarray(rec["leading_t"], float); ld_e = np.asarray(rec["leading_e"], float)
            ld_u = _e_to_uplift(ref, ld_e); ld_pct = _pct(ld_e); di = _ds_idx(ld_e.size)
            out["leading_t"] = [round(float(x), 4) for x in ld_t[di]]
            out["leading_e"] = [round(float(x), 4) for x in ld_e[di]]
            out["leading_uplift"] = [float(ld_u[i]) for i in di]
            out["leading_pct_to_asi"] = [round(float(ld_pct[i]), 3) for i in di]
    # implied inference+experiment compute that reproduces the legal's throttled research
    # effort (per-rollout; only when --solve-legal-compute is set). Downsampled like legal_t.
    if "legal_implied_inf" in rec:
        for k in ("legal_actual_inf", "legal_actual_exp", "legal_implied_inf",
                  "legal_implied_exp", "legal_compute_factor"):
            out[k] = [round(float(x), 4) for x in np.asarray(rec[k], float)[li]]
        ui = _ds_idx(len(rec["legal_used_implied_inf"]))   # used glidepath has its own grid
        for k in ("legal_used_implied_inf", "legal_used_implied_exp", "legal_used_compute_factor"):
            out[k] = [round(float(x), 4) for x in np.asarray(rec[k], float)[ui]]
    return out


def _median_scenario(model_cfg, covert_dists, data, tr, plan_a_data=None,
                     pause_med=None, pause_end_med=None, bottom_up_dists=None):
    """The single median-of-distributions trajectory, with ETSP decomposition, stored
    in the dataset so the median 2-panel is pure-JSON and shares the run's hash.
    `covert_dists` is the resolved covert_parameters dict (so any CLI override of
    years_before_ac / legal_mode is reflected here too); years_before_ac and legal_mode
    are read from its median like every other covert knob. When `plan_a_data` is given,
    the legal/covert run on the Plan-A series relativized to the (median) pause."""
    med = _get_median_parameter_dict(_load_config(model_cfg).get("parameters", {}) or {})
    med.setdefault("horizon_extrapolation_type", "decaying doubling time")
    cov = _get_median_parameter_dict(covert_dists)
    try:
        p = Parameters(**med); ref = run_default(p, data, tr)
        if ref["ac_time"] is None or ref["ted_ai_e"] is None:
            return None
        p = ref["prepared_params"]   # model-prepared params for the covert/legal sim
        tp = ref["ac_time"] - cov["years_before_ac"]
        # bottom_up_estimate: median diversion params -> natural-base fractions + window timing.
        bu_fracs = None; bu_med = None; bu_eval_offset = 0.0; bu_duration = None
        if cov["compute_mode"] == "bottom_up_estimate" and bottom_up_dists:
            bu_med = _get_median_parameter_dict(bottom_up_dists)
            bu_fracs = bottom_up_covert_fraction(bu_med)
            bu_eval_offset = float(bu_med.get("declaration_to_pause_years", 0.0))
            bu_duration = (float(bu_med["diversion_duration"]) if "diversion_duration" in bu_med else None)
        lc_data = data
        if plan_a_data is not None:
            lc_data = build_plan_a_series(data, plan_a_data, tp, tp + cov["setup_period_years"],
                                          pause_med, pause_end_med, tr[1])
        with _suppress_noise():
            rec = simulate_covert(p, lc_data, tp,
                                  alignment_tax_progress_years=cov["alignment_tax_progress_years"],
                                  # legal_scaling_duration is the POST-pause window
                                  target_year=tp + cov["setup_period_years"] + cov["legal_scaling_duration"],
                                  max_hw_rate=cov["max_hw_rate"],
                                  compute_mode=cov["compute_mode"], frac_stolen=cov["frac_stolen"],
                                  total_h100e=cov.get("total_h100e"), total_h100e_frac=cov.get("total_h100e_frac"),
                                  bottom_up_fracs=bu_fracs, bottom_up_eval_offset_years=bu_eval_offset,
                                  bottom_up_diversion_duration=bu_duration,
                                  bottom_up_power=bu_med,
                                  dormancy_years=cov["dormancy_years"],
                                  construction_years=cov.get("construction_years", 0.0) or 0.0,
                                  setup_period_years=cov["setup_period_years"],
                                  setup_sw_gain_progress_months=cov["setup_sw_gain_progress_months"], t_end=tr[1], ref=ref,
                                  return_trajectories=True, legal_mode=cov["legal_mode"],
                                  legal_strategy=cov["legal_strategy"], max_sw_rate=cov.get("max_sw_rate"),
                                  post_scaling_externalities_rate=cov["post_scaling_externalities_rate"],
                                  min_sw_frac_of_tsp=cov.get("min_sw_frac_of_tsp", 0.2),
                                  solve_legal_compute=True)   # implied inf+exp compute, median run
    except Exception:
        return None
    off = ref["rebase_offset"]
    dt = np.asarray(ref["times"], float); de = np.asarray(ref["effective_compute"], float)
    d_tsp = np.array([_tc_ooms(data, float(t)) for t in dt])
    d_sw = p.r_software * np.log10(np.maximum(np.asarray(ref["RS"], float), 1e-300) / ref["RS_0"])

    # Time-aware downsample: keep the race window (up to ~covert/legal arrival or ASI) dense,
    # so the discovered legal track — extended to t_end by post-scaling externalities — isn't
    # coarse over the steep ramp. Each track downsampled on ITS OWN time grid.
    _asi_t = (ref.get("milestones", {}).get("ASI") or {}).get("time")
    horizon = max(x for x in (rec.get("covert_ttt"), rec.get("legal_ttt"), _asi_t, tp + 6.0)
                  if x is not None) + 2.0
    _idx = {}
    def dn(a, t=None):
        a = np.asarray(a, float)
        if t is None:
            i = _ds_idx(a.size)
        else:
            key = id(t)
            if key not in _idx:
                _idx[key] = _ds_idx_time(np.asarray(t, float), dense_until=horizon)
            i = _idx[key]
        return [round(float(x), 4) for x in a[i]]
    lgt = rec["legal_t"]; ut = rec["legal_used_t"]; cvt = rec["covert_t"]
    return {
        "pause_year": float(tp), "e_pause": float(np.interp(tp, dt, de)),
        "legal_target_e": float(rec["legal_target_e"]), "ted_ai_e": float(ref["ted_ai_e"]),
        "covert_ttt": rec["covert_ttt"], "legal_ttt": rec["legal_ttt"],
        "covert_wins": bool(rec["covert_wins"]), "lead_years": rec["lead_years"],
        "milestones": {k: {"e": float(m["e"]), "time": float(m["time"]),
                           "uplift": float(_e_to_uplift(ref, m["e"]))}
                       for k, m in (ref.get("milestones") or {}).items()},
        # raw knobs (progress-time units) + the per-rollout OOMs the tax/windfall converted to
        "knobs": {**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in cov.items()},
                  "alignment_tax_ooms": float(rec["alignment_tax_ooms"]),
                  "setup_sw_gain_ooms": float(rec["setup_sw_gain_ooms"]),
                  "anchor_progress_rate": float(rec["anchor_progress_rate"])},
        # uplift: ref-present-day AI software R&D uplift, mapped from each track's E(t)
        # through the reference (E, uplift) relationship (dense map via _e_to_uplift) —
        # for the median uplift figure (pure-JSON, no model re-run).
        "default": {"t": dn(dt, dt), "e": dn(de, dt), "tsp": dn(d_tsp, dt), "sw": dn(d_sw, dt),
                    "uplift": dn(_e_to_uplift(ref, de), dt)},
        "legal": {"t": dn(lgt, lgt), "e": dn(rec["legal_e"], lgt),
                  "tsp": dn(rec["legal_tsp"], lgt), "sw": dn(rec["legal_sw"], lgt),
                  # used (deployed-capability) glidepath: reaches bare TED-AI — its OWN grid (ut)
                  "used_t": dn(ut, ut), "used_e": dn(rec["legal_used_e"], ut),
                  "used_sw": dn(rec["legal_used_sw"], ut),
                  # mandatory-minimum SW guarantee (cumulative, frozen at that track's
                  # arrival) — emitted by the sim so figures never re-derive the floor
                  "sw_floor": dn(rec["legal_sw_floor"], lgt),
                  "used_sw_floor": dn(rec["legal_used_sw_floor"], ut),
                  "used_uplift": dn(_e_to_uplift(ref, np.asarray(rec["legal_used_e"], float)), ut),
                  # implied inference+experiment compute (H100e): actual frontier series vs
                  # the amount that reproduces the legal's throttled research effort, + the
                  # common reduction factor f(t). NaN in setup, 0 once the legal sits.
                  "actual_inf": dn(rec["legal_actual_inf"], lgt), "actual_exp": dn(rec["legal_actual_exp"], lgt),
                  "implied_inf": dn(rec["legal_implied_inf"], lgt), "implied_exp": dn(rec["legal_implied_exp"], lgt),
                  "compute_factor": dn(rec["legal_compute_factor"], lgt),
                  # used (deployed-capability) glidepath's implied compute (excludes tax research)
                  "used_implied_inf": dn(rec["legal_used_implied_inf"], ut),
                  "used_implied_exp": dn(rec["legal_used_implied_exp"], ut),
                  "used_compute_factor": dn(rec["legal_used_compute_factor"], ut)},
        "covert": {"t": dn(cvt, cvt), "e": dn(rec["covert_e"], cvt),
                   "uplift": dn(_e_to_uplift(ref, np.asarray(rec["covert_e"], float)), cvt),
                   "tsp": dn(rec["covert_tsp"], cvt), "sw": dn(rec["covert_sw"], cvt),
                   "sw_stolen": dn(rec["covert_sw_stolen"], cvt), "sw_boost": dn(rec["covert_sw_boost"], cvt),
                   "sw_indig": dn(rec["covert_sw_indig"], cvt)},
    }


def build_dataset(model_cfg, covert_cfg, num, seed, years_before_ac, legal_mode, workers,
                  solve_legal_compute=False, min_survival_years=None,
                  covert_exists_prior=None):
    """seed / years_before_ac / legal_mode / min_survival_years / covert_exists_prior
    are CLI OVERRIDES: pass None to take the value from the config (covert yaml's `seed`
    for the run seed; the years_before_ac / legal_mode covert_parameters for the pause
    lead and legal physics mode; deal_breakdown.min_survival_years for the survival
    conditioning; the top-level covert_exists_prior for the P(covert exists) update).
    A non-None value pins that knob to a fixed value, overriding the yaml."""
    mc = _load_config(model_cfg) or {}
    pd_, ts = mc.get("parameters", {}) or {}, mc.get("time_series_parameters", {}) or {}
    corr = mc.get("correlation_matrix")
    all_dists = {**pd_, **ts}; model_names = set(pd_) | set(ts)
    input_path = mc.get("input_data", "inputs/input_data.csv")
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    covert_cfg_loaded = _load_config(covert_cfg) or {}
    covert_dists = covert_cfg_loaded.get("covert_parameters", {}) or {}
    # Horizon: the covert experiment's own time_range caps it (the model config's range is
    # general-purpose, e.g. out to 2200); fall back to the model config, then a default.
    tr_src = covert_cfg_loaded.get("time_range") or mc.get("time_range") or [2017.0, 2100.0]
    tr = [float(tr_src[0]), float(tr_src[1])]
    # Run seed: CLI override, else covert yaml `seed`, else model yaml `seed`, else 42.
    if seed is None:
        seed = int(covert_cfg_loaded.get("seed", mc.get("seed", 42)))
    # years_before_ac / legal_mode are covert_parameters; a non-None CLI value pins them
    # to a fixed value across all draws (overriding whatever the yaml specifies).
    if years_before_ac is not None:
        covert_dists["years_before_ac"] = {"dist": "fixed", "value": float(years_before_ac)}
    if legal_mode is not None:
        covert_dists["legal_mode"] = {"dist": "fixed", "value": legal_mode}
    _validate_covert_config(covert_dists)
    covert_corr = covert_cfg_loaded.get("correlation_matrix")
    # Detection inputs are the elicited probability grids; the model is JOINTLY fit from
    # them (fit once here, reused for every rollout & figure): per-size frailty
    # (pi=0, thetaC, eta, k) plus ONE shared per-site operational hazard r.
    det_grids = covert_cfg_loaded.get("detection_grids")
    det_fits, det_r_fit = detection.fit_detection_model(det_grids)
    det_size_range = _site_size_range(covert_dists, det_grids)   # validates optimal_site_mw vs grids
    det_sb_raw = covert_cfg_loaded.get("detection_silver_bullet")
    det_sb_cfg = _parse_silver_bullet(det_sb_raw)   # fixed sigmoid values; None => disabled
    cov = _get_median_parameter_dict(covert_dists)   # central knobs (median scenario + config snapshot)

    # Bottom-up compute config (only used when compute_mode == bottom_up_estimate). Referenced by
    # `bottom_up_compute_config:` in the covert config (path rel to REPO_ROOT). Holds the diversion
    # parameter distributions (covert-AI-projects supplement §2.3 Method #1), sampled per rollout.
    bottom_up_dists = None
    bu_rel = covert_cfg_loaded.get("bottom_up_compute_config")
    if bu_rel:
        _bp = Path(bu_rel); _bp = _bp if _bp.is_absolute() else (REPO_ROOT / bu_rel)
        bottom_up_dists = (_load_config(str(_bp)) or {}).get("bottom_up_compute_parameters") or {}
        _validate_bottom_up_config(bottom_up_dists)
    if cov.get("compute_mode") == "bottom_up_estimate" and not bottom_up_dists:
        raise ValueError("compute_mode 'bottom_up_estimate' requires a bottom_up_compute_config: "
                         "<path> entry in the covert config pointing at a covert_compute_*.yaml")

    # Deal breakdown (optional `deal_breakdown:` block): hazard curve that samples WHEN
    # the deal collapses per rollout, plus the sim's breakdown knobs as dists. None =>
    # the deal always holds (all pre-existing configs). A CLI min_survival_years pins
    # the survival conditioning, overriding the yaml (sweep driver's knob).
    bd_block = covert_cfg_loaded.get("deal_breakdown")
    if min_survival_years is not None:
        if not bd_block:
            raise ValueError("--min-survival-years given but the covert config has no "
                             "deal_breakdown block to condition")
        bd_block = {**bd_block, "min_survival_years": float(min_survival_years)}
    deal_bd = _parse_deal_breakdown(bd_block)

    # Optional Bayesian update re: P(a covert project exists) — CLI override > yaml
    # top-level `covert_exists_prior` > None (unmixed field). Consumed by the
    # capability_cdf2d figures/CSVs (the sim itself is unconditional); it rides in the
    # config snapshot so figures stay pure functions of the dataset.
    if covert_exists_prior is None:
        covert_exists_prior = covert_cfg_loaded.get("covert_exists_prior")
    if covert_exists_prior is not None:
        covert_exists_prior = float(covert_exists_prior)
        if not (0.0 <= covert_exists_prior <= 1.0):
            raise ValueError(f"covert_exists_prior must be in [0, 1], got {covert_exists_prior}")
    if bottom_up_dists is not None and load_time_series_data(input_path).world_H100e is None:
        raise ValueError(f"bottom_up_compute_config is set but the input series {input_path} has no "
                         "world_H100e column (needed to scale the diverted fraction to H100e)")

    # Plan-A divergence series (optional `plan_a_input_data` in the covert config). When set,
    # the legal/covert run on the DEFAULT series up to each rollout's pause, then Plan A's
    # ramp relativized to that pause (build_plan_a_series). Plan A's SHAPE is taken from the
    # MEDIAN scenario (median takeoff -> AC_med -> pause_med/pause_end_med) and slid+scaled to
    # each rollout. When unset, the legal/covert use the single (default) series, as before.
    plan_a_path = pause_med = pause_end_med = None
    plan_a_rel = covert_cfg_loaded.get("plan_a_input_data")
    if plan_a_rel:
        _p = Path(plan_a_rel)
        plan_a_path = str(_p if _p.is_absolute() else (REPO_ROOT / plan_a_rel).resolve())
        _med = _get_median_parameter_dict(pd_)
        _med.setdefault("horizon_extrapolation_type", mcfg_mod.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
        _med.setdefault("taste_schedule_type", mcfg_mod.DEFAULT_TASTE_SCHEDULE_TYPE)
        _ref_med = run_default(Parameters(**_med), load_time_series_data(input_path), tr)
        if _ref_med["ac_time"] is None:
            raise ValueError("plan_a_input_data set but the median scenario never reaches AC; "
                             "cannot anchor the Plan-A relativization.")
        pause_med = float(_ref_med["ac_time"]) - float(cov["years_before_ac"])
        pause_end_med = pause_med + float(cov["setup_period_years"])

    master = SeedSequence(seed)

    def _calibrate(c, dists, label):
        """Pre-calibrate a Gaussian-copula latent correlation (mirrors model params).
        Returns the (possibly mutated) corr spec, or None if calibration fails."""
        if c is None:
            return None
        cpar, tcorr = c.get("parameters", []), c.get("correlation_matrix")
        if cpar and tcorr is not None:
            try:
                lat = _calibrate_gaussian_copula_latent_corr(
                    dists, cpar, np.array(tcorr, float), np.random.default_rng(master.spawn(1)[0]),
                    interpretation=str(c.get("correlation_type", c.get("interpretation", "spearman"))),
                    calibration_samples=int(c.get("calibration_samples", 20000)))
                c["latent_correlation_matrix"] = lat.tolist()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: {label} corr calibration failed ({e}); dropping"); return None
        return c

    corr = _calibrate(corr, all_dists, "model")
    covert_corr = _calibrate(covert_corr, covert_dists, "covert")

    seeds = master.spawn(num); tasks = list(enumerate(seeds))
    init = (input_path, all_dists, model_names, corr, covert_dists, covert_corr, det_fits,
            det_r_fit, det_size_range, det_sb_cfg, tr,
            mcfg_mod.DEFAULT_HORIZON_EXTRAPOLATION_TYPE, mcfg_mod.DEFAULT_TASTE_SCHEDULE_TYPE,
            solve_legal_compute, plan_a_path, pause_med, pause_end_med, bottom_up_dists,
            deal_bd)
    t0 = time.perf_counter(); recs = []
    nw = max(1, min(workers, num))
    print(f"building dataset: {num} samples on {nw} workers", flush=True)
    if nw == 1:
        _init(*init); recs = [r for r in (_run_sample(t) for t in tasks) if r is not None]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(nw, initializer=_init, initargs=init) as pool:
            for done, r in enumerate(pool.imap_unordered(_run_sample, tasks, chunksize=1), 1):
                if r is not None:
                    recs.append(r)
                if done % 50 == 0 or done == num:
                    print(f"  {done}/{num} ({len(recs)} ok, {time.perf_counter()-t0:.0f}s)", flush=True)
    recs.sort(key=lambda r: r["index"])

    # param percentile stats (for the explorer): only params that actually vary
    def _pstats(field):
        out = {}
        for k in sorted({k for r in recs for k, v in r.get(field, {}).items() if isinstance(v, (int, float))}):
            v = np.array([r.get(field, {}).get(k, np.nan) for r in recs], float); v = v[np.isfinite(v)]
            if v.size and v.std() > 1e-9 * (abs(v.mean()) + 1e-12):
                out[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                          "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
        return out
    pstats = _pstats("model_params")
    covert_pstats = _pstats("covert_params")
    wins = np.array([r["covert_wins"] for r in recs], bool)
    median_scenario = _median_scenario(
        model_cfg, covert_dists, load_time_series_data(input_path), tr,
        (load_time_series_data(plan_a_path) if plan_a_path else None), pause_med, pause_end_med,
        bottom_up_dists)
    # resolved model params: central value of EVERY model param, split by whether it
    # was sampled (varies, in pstats) or held fixed (e.g. median_to_top_taste_multiplier).
    resolved = _get_median_parameter_dict(pd_)
    _flt = lambda v: float(v) if isinstance(v, (int, float, np.floating)) else v
    model_params_sampled = {k: _flt(resolved[k]) for k in resolved if k in pstats}
    model_params_fixed = {k: _flt(resolved[k]) for k in resolved if k not in pstats}
    # covert knobs split the same way: those that vary across draws vs held fixed
    covert_params_sampled = {k: _flt(cov[k]) for k in cov if k in covert_pstats}
    covert_params_fixed = {k: _flt(cov[k]) for k in cov if k not in covert_pstats}
    return {
        "median_scenario": median_scenario,
        "config": {"model_config": str(model_cfg), "covert_config": str(covert_cfg), "num": num,
                   "seed": seed, "years_before_ac": _flt(cov.get("years_before_ac")),
                   "legal_mode": cov.get("legal_mode"),
                   "workers": nw, "time_range": tr, "input_data": input_path,
                   # all covert knobs at their median (captions read specific keys here)
                   "covert_knobs_fixed": {k: _flt(v) for k, v in cov.items()},
                   # detection: the elicited grids (interpretable input) plus the JOINT
                   # fit derived from them — per-size frailty and the shared per-site
                   # hazard r — so figures re-derive curves from the dataset, not by
                   # re-reading config / re-fitting. JSON stringifies the MW keys;
                   # detection.normalize_fits() coerces them back. Per-record
                   # detection_r already INCLUDES detection_r_fit.
                   "detection_grids": det_grids,
                   "detection_frailty_fit": {str(s): v for s, v in det_fits.items()},
                   "detection_r_fit": det_r_fit,
                   "detection_silver_bullet": det_sb_raw,   # fixed sigmoid spec block (or None)
                   # deal breakdown: the raw hazard spec + resolved knob medians + the
                   # analytic never-breaks fraction (None when the block is absent — the
                   # deal always holds). Figures read this instead of re-parsing config.
                   # prior P(a covert project exists) for the epistemic update (or None)
                   "covert_exists_prior": covert_exists_prior,
                   "deal_breakdown": (None if deal_bd is None else {
                       "hazard": deal_bd["hazard_raw"],
                       "min_survival_years": deal_bd["min_survival_years"],
                       "params_median": {k: _flt(v) for k, v in
                                         _get_median_parameter_dict(deal_bd["dists"]).items()},
                       # conditional on surviving min_survival_years (== unconditional at 0)
                       "p_never": deal_hazard.p_never(deal_bd["hazard"],
                                                      deal_bd["min_survival_years"])}),
                   "covert_params_fixed": covert_params_fixed,           # held constant across draws
                   "covert_params_sampled_median": covert_params_sampled,  # varied; median shown
                   "model_params_fixed": model_params_fixed,       # held constant across draws
                   "model_params_sampled_median": model_params_sampled},  # varied; median shown
        "summary": {"n_ok": len(recs), "n_requested": num,
                    "covert_win_frac": float(wins.mean()) if len(wins) else None,
                    # fraction of rollouts whose deal actually broke down within the horizon
                    # (None when the run has no deal_breakdown block)
                    "deal_breakdown_frac": (
                        float(np.mean([r.get("deal_breakdown_t") is not None
                                       and r["deal_breakdown_t"] < tr[1] for r in recs]))
                        if (deal_bd is not None and recs) else None),
                    "build_seconds": round(time.perf_counter() - t0, 1)},
        "pstats": pstats, "covert_pstats": covert_pstats, "records": recs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default=str(REPO_ROOT / "plan_a_covert_proj/config/model_uncertain_thomas.yaml"))
    ap.add_argument("--covert-config", default=str(REPO_ROOT / "plan_a_covert_proj/config/covert_scenario_branch.yaml"))
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None, help="override the covert yaml `seed`")
    ap.add_argument("--years-before-ac", type=float, default=None,
                    help="override the covert yaml years_before_ac knob (pins it fixed)")
    ap.add_argument("--legal-mode", choices=["backsolve", "capped"], default=None,
                    help="override the covert yaml legal_mode knob (pins it fixed)")
    ap.add_argument("--min-survival-years", type=float, default=None,
                    help="override deal_breakdown.min_survival_years (condition on the deal "
                         "surviving at least this long; requires a deal_breakdown block)")
    ap.add_argument("--covert-exists-prior", type=float, default=None,
                    help="override the covert yaml covert_exists_prior (prior P(a covert "
                         "project exists) for the figures' Bayesian update)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    ap.add_argument("--solve-legal-compute", action="store_true",
                    help="also solve the implied inference+experiment compute PER ROLLOUT "
                         "(costly; the median scenario always computes it)")
    a = ap.parse_args()
    ds = build_dataset(a.model_config, a.covert_config, a.num, a.seed, a.years_before_ac,
                       a.legal_mode, a.workers, solve_legal_compute=a.solve_legal_compute,
                       min_survival_years=a.min_survival_years,
                       covert_exists_prior=a.covert_exists_prior)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(ds, f, separators=(",", ":"))
    s = ds["summary"]
    print(f"wrote {a.out}: {s['n_ok']}/{s['n_requested']} ok, covert wins "
          f"{(s['covert_win_frac'] or 0)*100:.0f}%, {s['build_seconds']}s")


if __name__ == "__main__":
    main()
