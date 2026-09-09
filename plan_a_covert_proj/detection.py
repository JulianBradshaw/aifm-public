#!/usr/bin/env python3
"""Covert-project detection model — pure, no plotting, no I/O.

Ported from the pre-cutover ``scripts/analysis/_archive_precutover/covert_detection.py``,
which itself ports the monorepo's detection elicitation
(``apps/covert-detection-model/src/lib/model.ts``, ``detection-elicitation-full.xlsx``).

Model
-----
A covert program of total power ``compute_mw`` is split across ``n`` near-identical
sites of per-site power ``site_mw`` (``n = compute_mw / site_mw``, with an ``n >= 1``
floor: a single site then holds all the compute and its signature is computed at its
actual power). For per-site size ``s`` with gamma-frailty ``(pi, thetaC, eta, k)`` the
probability the program is STILL undetected after running ``t`` years is

    S(n, t) = [pi + (1-pi)(1 + thetaC*n*H(t))^(-1/thetaC)] * exp(-n*r*t),
    H(t)    = (t / eta)^k

with an optional constant per-site operational-mistake hazard ``r`` (per year). Frailty
params are interpolated linearly in log10(size) between the elicited site sizes (default
10 / 100 / 1000 MW), clamped outside that range. Detection probability is ``1 - S``.

The frailty ``(pi, thetaC, eta, k)`` and the hazard ``r`` are NOT inputs — they are FIT
to the elicited detection-probability grids (covert.yaml's ``detection_grids``, the
interpretable inputs) via ``fit_detection_model``: a JOINT fit across all site sizes of
per-size ``(thetaC, eta, k)`` with ``pi`` pinned to 0, plus ONE shared per-site
operational hazard ``r`` (deterministic multi-start, no seed value or placeholder
fallback). The pipeline fits once and threads the result through as ``fits`` + ``r``;
there is no default fits table — missing grids/fits are a hard error.
``fit_frailty_from_grids`` (per-size, r=0, pi free) is the legacy fit kept for
reference against the monorepo model.ts.

This module is the SINGLE SOURCE OF TRUTH for the detection model: the MC pipeline
(``pipeline/mc.py``) draws one detection event per rollout and records it; the figure
layer (``pipeline/figures.py``) reads those draws and may re-derive survival curves for
display. Neither re-runs the (expensive) physics model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# The MODEL INPUTS are the elicited detection probabilities (covert.yaml's
# `detection_grids`): per site size, P(detected | n sites, 3y/10y). The frailty
# params (pi, thetaC, eta, k) are NOT inputs — they are FIT per size to those grids
# (``fit_frailty_from_grids``), because they're uninterpretable on their own. The fit uses
# a deterministic MULTI-START Nelder-Mead (``_FIT_STARTS``): no per-fit seed value, no
# placeholder fallback — the result is determined entirely by the elicited grid, and a
# poor fit halts (``_FIT_MAX_SSE``) rather than returning silently.
_FRAILTY_KEYS = ("pi", "thetaC", "eta", "k")

# Explicit cap on the asymptotic never-detected floor pi (mirrors model.ts): a structural
# bound on the parameterization (at least 1-_PI_MAX chance of eventual detection), NOT a
# fit default. _to_z/_from_z map pi into (0, _PI_MAX) during the fit.
_PI_MAX = 0.95

# Optimizer START points for the multi-start fit — spanning the plausible region, NOT
# defaults/placeholders. The grid determines the fit; these only seed Nelder-Mead, which
# converges to the same minimum from any of them (we keep several for robustness).
_FIT_STARTS = [
    {"pi": pi, "thetaC": thetaC, "eta": eta, "k": k}
    for pi in (0.0, 0.1) for thetaC in (2.0, 5.0)
    for eta in (5.0, 15.0, 25.0) for k in (0.8, 1.0)
]
# Detection-space SSE above which a per-size fit is treated as a FAILURE (halt, don't
# silently return a bad fit). The frailty form cannot perfectly match the hand-elicited
# grids: the current grids' best fits land at SSE ~0.013–0.031 (≈6% avg per-point error),
# which IS the global minimum. This gate (~3x headroom) catches gross failures — a missed
# optimization basin or an unfittable grid — not that inherent imperfection.
_FIT_MAX_SSE = 0.1

def normalize_fits(block) -> dict:
    """Coerce a frailty table (from YAML config or a JSON dataset) into
    ``{size_mw: {pi, thetaC, eta, k}}`` with float keys/values. JSON round-trips turn
    the int size keys into strings, so we float() them. Empty block is a hard error
    (the old SEED_FITS placeholder fallback was removed)."""
    if not block:
        raise ValueError("normalize_fits: empty frailty block; the SEED_FITS placeholder "
                         "was removed — provide frailty fits explicitly.")
    return {float(size): {key: float(vals[key]) for key in _FRAILTY_KEYS}
            for size, vals in block.items()}


def interp_frailty(size_mw: float, fits: dict | None = None) -> dict:
    """Log10(size)-linear interpolation of the four frailty params, clamped outside
    the elicited size range (mirrors model.ts ``interpFrailty``). ``fits`` defaults to
    explicitly-provided fits (the SEED_FITS placeholder default was removed)."""
    if not fits:
        raise ValueError("interp_frailty: no fits provided; the SEED_FITS placeholder "
                         "was removed — pass a frailty-fit table.")
    sizes = sorted(fits)
    x = np.log10(max(float(size_mw), 1e-9))
    if x <= np.log10(sizes[0]):
        return dict(fits[sizes[0]])
    if x >= np.log10(sizes[-1]):
        return dict(fits[sizes[-1]])
    a, b = sizes[0], sizes[-1]
    for i in range(len(sizes) - 1):
        if np.log10(sizes[i + 1]) >= x:
            a, b = sizes[i], sizes[i + 1]
            break
    xa, xb = np.log10(a), np.log10(b)
    f = (x - xa) / (xb - xa)
    pa, pb = fits[a], fits[b]
    return {key: pa[key] + f * (pb[key] - pa[key]) for key in pa}


def deploy(compute_mw: float, site_mw: float, cap_sites: bool = True):
    """Effective site count and per-site signature power (``n >= 1`` floor)."""
    compute_mw = max(float(compute_mw), 1e-9)
    site_mw = max(float(site_mw), 1e-9)
    n = compute_mw / site_mw
    eff = site_mw
    if cap_sites and n < 1.0:
        n, eff = 1.0, compute_mw
    return n, eff


def survival(p: dict, n: float, t, r: float = 0.0):
    """S(n, t): gamma-frailty capability term * optional Poisson operational factor.
    Vectorised over ``t`` (accepts scalar or array)."""
    t = np.asarray(t, dtype=float)
    H = np.power(np.maximum(t, 0.0) / p["eta"], p["k"])
    x = n * H
    if p["thetaC"] < 1e-9:
        inner = np.exp(-x)
    else:
        inner = np.power(1.0 + p["thetaC"] * x, -1.0 / p["thetaC"])
    cap = p["pi"] + (1.0 - p["pi"]) * inner
    fuckup = np.exp(-n * r * t) if r > 0 else 1.0
    return np.clip(cap * fuckup, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Fit per-size frailty to the elicited detection-probability grids
# ---------------------------------------------------------------------------
# Mirrors apps/covert-detection-model/src/lib/model.ts (fitFrailty): minimise SSE in
# detection space between the model's P(detected | n, t) and the elicited Best points,
# over an unconstrained box (pi via logistic in (0, PI_MAX), thetaC/eta/k via exp).

def normalize_grids(grids) -> list:
    """Coerce the config/JSON ``detection_grids`` into ``[{size_mw, rows:[(n,y3,y10)]}]``
    with floats. Accepts rows as ``[n, y3, y10]`` lists (y3/y10 may be null to omit)."""
    out = []
    for g in grids or []:
        rows = [(float(r[0]),
                 None if r[1] is None else float(r[1]),
                 None if r[2] is None else float(r[2])) for r in g["rows"]]
        out.append({"size_mw": float(g["size_mw"]), "rows": rows})
    return out


def _grid_points(rows):
    """(n, t, best-detection) points from a size's rows; t in {3, 10}, null cells skipped."""
    pts = []
    for n, y3, y10 in rows:
        if y3 is not None:
            pts.append((n, 3.0, y3))
        if y10 is not None:
            pts.append((n, 10.0, y10))
    return pts


def _to_z(p):
    pi = min(max(p["pi"], 1e-6), _PI_MAX - 1e-6)
    return [np.log(pi / (_PI_MAX - pi)), np.log(max(p["thetaC"], 1e-6)),
            np.log(max(p["eta"], 1e-6)), np.log(max(p["k"], 1e-6))]


def _from_z(z):
    return {"pi": _PI_MAX / (1.0 + np.exp(-z[0])), "thetaC": float(np.exp(z[1])),
            "eta": float(np.exp(z[2])), "k": float(np.exp(z[3]))}


def _grid_sse(p, pts):
    return float(sum(((1.0 - float(survival(p, n, t))) - d) ** 2 for n, t, d in pts))


def fit_size(rows):
    """Fit one size's (pi, thetaC, eta, k) to its elicited rows by SSE-minimising
    Nelder–Mead from a deterministic multi-start (``_FIT_STARTS``); each start runs two
    passes (second restarted from the first) and the lowest-SSE solution wins. The fit is
    determined by ``rows`` alone — there is no seed argument and no placeholder default.
    Returns ``(params, sse)``. Raises if ``rows`` carry no elicited points."""
    pts = _grid_points(rows)
    if not pts:
        raise ValueError("fit_size: no elicited (n, t, detection) points in rows")
    obj = lambda z: _grid_sse(_from_z(z), pts)  # noqa: E731
    opt = dict(maxiter=4000, xatol=1e-9, fatol=1e-12)
    best = None
    for start in _FIT_STARTS:
        r1 = minimize(obj, _to_z(start), method="Nelder-Mead", options=opt)
        r2 = minimize(obj, r1.x, method="Nelder-Mead", options=opt)
        cand = r2 if r2.fun < r1.fun else r1
        if best is None or cand.fun < best.fun:
            best = cand
    return _from_z(best.x), float(best.fun)


def fit_frailty_from_grids(grids) -> dict:
    """LEGACY per-size fit (r = 0, pi free): ``{size_mw: {pi,thetaC,eta,k}}`` from the
    elicited detection grids by multi-start Nelder-Mead per size. Superseded by
    ``fit_detection_model`` (joint fit with a shared operational hazard) as the pipeline's
    fit; kept as the reference for the monorepo model.ts parameterization and for starts.
    No silent fallback: raises if there are no grids, and raises if any size fails to fit
    within ``_FIT_MAX_SSE`` (a bad fit halts rather than returning silently)."""
    norm = normalize_grids(grids)
    if not norm:
        raise ValueError("fit_frailty_from_grids: no detection_grids provided; cannot fit "
                         "frailty (the SEED_FITS placeholder fallback was removed).")
    fits = {}
    for g in norm:
        size = g["size_mw"]
        params, sse = fit_size(g["rows"])
        if not np.isfinite(sse) or sse > _FIT_MAX_SSE:
            raise ValueError(
                f"fit_frailty_from_grids: poor frailty fit at size {size:g} MW "
                f"(detection-space SSE {sse:.4g} > {_FIT_MAX_SSE:g}); check detection_grids.")
        fits[size] = params
    return fits


# Joint-fit r starts (coarse grid; the objective is smooth in log r and Nelder-Mead
# converges to the same optimum from any of them — several kept for basin robustness).
_R_STARTS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3)


def fit_detection_model(grids) -> tuple[dict, float]:
    """JOINT fit of the detection model across ALL elicited site sizes at once:
    per-size frailty shape ``(thetaC, eta, k)`` with the never-detected floor ``pi``
    PINNED TO 0, plus ONE shared per-site operational slip-up hazard ``r`` (/yr/site)
    common to every size — the ``e^(-n r t)`` factor in ``survival``.

    Rationale (vs the legacy per-size r=0 fit): the elicited grids for small sites
    demand strong non-saturating late detection growth at high site counts (e.g. 10 MW,
    n=100: 55% by 3y but 75% by 10y) that a shared-frailty term structurally cannot
    produce — frailty selects out sloppy programs early, so high-n curves saturate. A
    constant per-site hazard is exactly that missing channel. pi is dropped because the
    grids never express a never-detected floor (it fit to ~1e-15 everywhere), keeping
    the parameter count at 3 per size + 1 shared, one FEWER than the legacy 4 per size.

    Deterministic multi-start (legacy per-size fits + ``_FIT_STARTS`` x ``_R_STARTS``),
    same per-size ``_FIT_MAX_SSE`` gate as the legacy fit (evaluated with the shared r).
    Returns ``(fits, r)``; ``fits`` keeps the ``pi`` key (always 0.0) so the table is
    drop-in compatible with ``interp_frailty``/``survival`` and dataset round-trips."""
    norm = normalize_grids(grids)
    if not norm:
        raise ValueError("fit_detection_model: no detection_grids provided; cannot fit "
                         "the detection model.")
    norm = sorted(norm, key=lambda g: g["size_mw"])
    sizes = [g["size_mw"] for g in norm]
    pts = {g["size_mw"]: _grid_points(g["rows"]) for g in norm}
    if any(not p for p in pts.values()):
        raise ValueError("fit_detection_model: a size grid has no elicited points")

    def _unpack(z):
        per = {s: {"pi": 0.0, "thetaC": float(np.exp(z[3 * i])),
                   "eta": float(np.exp(z[3 * i + 1])), "k": float(np.exp(z[3 * i + 2]))}
               for i, s in enumerate(sizes)}
        return per, float(np.exp(z[-1]))

    def _size_sse(p, r, size):
        return float(sum(((1.0 - float(survival(p, n, t, r))) - d) ** 2
                         for n, t, d in pts[size]))

    def _obj(z):
        per, r = _unpack(z)
        return sum(_size_sse(per[s], r, s) for s in sizes)

    # Starts: the legacy per-size independent fits (good deterministic shape starts),
    # plus the generic _FIT_STARTS grid, each crossed with the r grid.
    legacy = {s: fit_size(g["rows"])[0] for s, g in zip(sizes, norm)}
    shape_starts = [legacy] + [{s: st for s in sizes} for st in _FIT_STARTS]
    opt = dict(maxiter=40000, maxfev=40000, xatol=1e-10, fatol=1e-13)
    best = None
    for shapes in shape_starts:
        for r0 in _R_STARTS:
            z0 = np.array([np.log(max(shapes[s][key], 1e-6))
                           for s in sizes for key in ("thetaC", "eta", "k")]
                          + [np.log(r0)])
            r1 = minimize(_obj, z0, method="Nelder-Mead", options=opt)
            r2 = minimize(_obj, r1.x, method="Nelder-Mead", options=opt)
            cand = r2 if r2.fun < r1.fun else r1
            if best is None or cand.fun < best.fun:
                best = cand
    per, r = _unpack(best.x)
    for s in sizes:
        sse = _size_sse(per[s], r, s)
        if not np.isfinite(sse) or sse > _FIT_MAX_SSE:
            raise ValueError(
                f"fit_detection_model: poor joint fit at size {s:g} MW "
                f"(detection-space SSE {sse:.4g} > {_FIT_MAX_SSE:g}); check detection_grids.")
    return per, r


# ---------------------------------------------------------------------------
# Silver bullet G(t): a program-level detection breakthrough (e.g. a universal lie
# detector / interpretability tool). If one has emerged by year t, the covert is
# exposed regardless of its facility signature. Unconditional survival multiplies:
#   S_uncond(t) = G(t) * S_frailty(t),   G(t) = 1 - P(silver bullet emerged by t).
# ``t`` is years after the covert starts (the pause).
#
# P(emerged by t) is a CEILING-LOGISTIC tied to the LEGAL project's trajectory, since a
# detection breakthrough is an output of the aligned frontier effort, not the calendar:
#   P(t) = C * sigmoid(k * (t - t_mid)),   G(t) = 1 - P(t).
# It is pinned by three fixed probabilities at two trajectory-derived times:
#   (t0, p0) = (end of pause,                          P at end of pause)
#   (t1, p1) = (legal USED capability -> TED-AI, + Δ,  P there)
#   C        = ceiling (asymptote, t -> inf).
# The two anchor TIMES come from the per-rollout legal trajectory; the three probabilities
# and Δ are fixed config. build_silver_bullet() solves (k, t_mid) in closed form.

def build_silver_bullet(p0: float, t0: float, p1: float, t1: float, ceiling: float) -> dict:
    """Solve the ceiling-logistic P(t)=C*sigmoid(k(t-t_mid)) through (t0,p0) and (t1,p1).

    a_i = logit(p_i / C); k = (a1 - a0)/(t1 - t0); t_mid = t0 - a0/k. Returns the spec
    {'C','k','t_mid'} consumed by g_no_silver_bullet. Halts (ValueError) on any ordering
    violation — the anchors must satisfy 0 < p0 < p1 < C <= 1 and t1 > t0."""
    C = float(ceiling)
    p0, p1, t0, t1 = float(p0), float(p1), float(t0), float(t1)
    if not (0.0 < C <= 1.0):
        raise ValueError(f"silver-bullet ceiling must be in (0, 1], got {C}")
    if not (0.0 < p0 < p1 < C):
        raise ValueError(
            f"silver-bullet anchors must satisfy 0 < p0 < p1 < ceiling; "
            f"got p0={p0}, p1={p1}, ceiling={C}")
    if not (t1 > t0):
        raise ValueError(
            f"silver-bullet anchor times must satisfy t1 > t0 (legal used->TED-AI + Δ must "
            f"be after end of pause); got t0={t0}, t1={t1}")
    a0 = np.log((p0 / C) / (1.0 - p0 / C))
    a1 = np.log((p1 / C) / (1.0 - p1 / C))
    k = (a1 - a0) / (t1 - t0)
    t_mid = t0 - a0 / k
    return {"C": C, "k": float(k), "t_mid": float(t_mid)}


def g_no_silver_bullet(spec, t):
    """G(t) = 1 - P(a detection silver bullet has emerged by year t). ``spec`` is a
    {'C','k','t_mid'} dict from build_silver_bullet; None -> 1.0 (inert, e.g. when the
    legal project never reaches TED-AI). ``t`` may be scalar or array."""
    t = np.asarray(t, dtype=float)
    if not spec:
        return np.ones_like(t)
    C, k, t_mid = spec["C"], spec["k"], spec["t_mid"]
    return 1.0 - C / (1.0 + np.exp(-k * (t - t_mid)))


def combined_survival(p: dict, n: float, t, r: float = 0.0, silver_bullet=None,
                      dormancy_years: float = 0.0):
    """Unconditional survival S_uncond(t) for ``t`` years after the program starts.

    ``dormancy_years`` is the initial lying-low phase before construction begins: no
    facility exists, so NOTHING is detectable — S_uncond = 1 for t < dormancy. The
    facility-frailty clock starts at the END of dormancy (its ``t=0`` is the start of
    construction), while the silver-bullet emergence curve G(t) stays on the program
    clock (a breakthrough is an output of the legal project's effort, indifferent to
    the covert's schedule) — a silver bullet that emerged during dormancy therefore
    bites the moment the covert surfaces:

        S_uncond(t) = 1                                   for t <  d
        S_uncond(t) = G(t) * S_frailty(t - d)             for t >= d
    """
    t = np.asarray(t, dtype=float)
    d = max(float(dormancy_years), 0.0)
    s = g_no_silver_bullet(silver_bullet, t) * survival(p, n, np.maximum(t - d, 0.0), r)
    return np.where(t < d, 1.0, s)


def detection_probability(compute_mw: float, t, r: float = 0.0,
                          site_mw: float | None = None, cap_sites: bool = True,
                          fits: dict | None = None, silver_bullet=None,
                          dormancy_years: float = 0.0):
    """P(detected by horizon ``t``) = 1 - S_uncond(t), for a project of ``compute_mw``
    total power. Convenience wrapper for display curves; ``t`` may be scalar or array."""
    n, eff = deploy(compute_mw, site_mw if site_mw else compute_mw, cap_sites)
    return 1.0 - combined_survival(interp_frailty(eff, fits), n, t, r,
                                   silver_bullet=silver_bullet, dormancy_years=dormancy_years)


def sample_detection(u: float, compute_mw: float, t_risk_goal,
                     r: float = 0.0, site_mw: float | None = None,
                     cap_sites: bool = True, t_max: float = 60.0, n_grid: int = 6000,
                     fits: dict | None = None, silver_bullet=None,
                     dormancy_years: float = 0.0) -> dict:
    """Draw one detection event for a single covert project from a uniform ``u`` in
    [0, 1). Inverts the (monotone-decreasing) survival curve: detection time is the
    first ``t`` with ``S(t) <= u``; if ``u <= S(t_max)`` (the asymptotic never-detected
    mass) the project is never detected within the horizon (``t_det = inf``).

    ``t_risk_goal`` = years the project runs until it reaches its goal (TED-AI), used
    only to classify whether detection lands BEFORE the goal. Pass ``None``/NaN when the
    project never reaches the goal (then ``detected_before_goal`` is False and
    ``survival_at_goal`` is NaN).

    ``dormancy_years`` shifts the detection clock (see ``combined_survival``): nothing is
    detectable before construction starts, so S = 1 for ``t < dormancy_years`` and the
    frailty clock runs from the end of dormancy. Any silver-bullet emergence mass
    accumulated during dormancy lands as detection AT the end of dormancy.

    Returns ``{t_det, detected_before_goal, survival_at_goal, n_sites, eff_size_mw}``.
    ``t_det`` is years-after-start (``inf`` if never detected).
    """
    n, eff = deploy(compute_mw, site_mw if site_mw else compute_mw, cap_sites)
    p = interp_frailty(eff, fits)
    d = max(float(dormancy_years), 0.0)

    tgrid = np.linspace(0.0, t_max, n_grid)
    if 0.0 < d < t_max:
        # Insert the end-of-dormancy instant so S(d) = G(d) is represented exactly.
        tgrid = np.unique(np.concatenate([tgrid, [d]]))
    sgrid = combined_survival(p, n, tgrid, r, silver_bullet=silver_bullet, dormancy_years=d)
    floor = float(sgrid[-1])                      # asymptotic survival (never-detected mass)
    if u > floor:
        # S decreasing -> flip to ascending for np.interp; t_det = t where S(t) = u.
        # S = 1 throughout dormancy, so no detection can land before d: a u inside the
        # end-of-dormancy jump (S: 1 -> G(d)) is jump mass realised AT d — the clamp
        # makes that exact rather than smeared across the last pre-d grid cell.
        t_det = float(max(np.interp(u, sgrid[::-1], tgrid[::-1]), d if d > 0 else 0.0))
    else:
        t_det = float("inf")

    reaches = t_risk_goal is not None and np.isfinite(t_risk_goal)
    if reaches:
        tg = float(t_risk_goal)
        survival_at_goal = float(combined_survival(p, n, tg, r, silver_bullet=silver_bullet,
                                                   dormancy_years=d))
        detected_before_goal = bool(t_det < tg)
    else:
        survival_at_goal = float("nan")
        detected_before_goal = False

    return {
        "t_det": t_det,
        "detected_before_goal": detected_before_goal,
        "survival_at_goal": survival_at_goal,
        "n_sites": float(n),
        "eff_size_mw": float(eff),
    }
