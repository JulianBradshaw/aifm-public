"""Deal-breakdown survival curves: sample WHEN (if ever) the deal collapses.

The hazard rate h(t) (per-year breakdown probability density given survival so far,
t = years after the deal starts) is specified as:

    type: linear_plus_decay
    points:                # [years_after_deal_start, hazard], strictly increasing t
      - [0, 0.12]
      - [6, 0.06]
      - [11, 0.03]
      - [16, 0.02]
    decay: 0.95            # per-year factor applied to the last point's hazard

h(t) linearly interpolates between the points; past the last point it decays
exponentially, h(t) = h_last * decay**(t - t_last). Survival is S(t) = exp(-H(t))
with H the cumulative hazard. For decay < 1 the tail integral is FINITE
(h_last / -ln(decay)), so S(inf) > 0: a fraction of samples NEVER break down —
`sample_breakdown_time` returns None for those, which callers must treat as
"the deal holds for the whole simulated period".

Everything here is closed-form (piecewise-quadratic H on the linear segments,
exponential tail), so sampling is an exact inverse-CDF, not a numerical root-find.
"""
from __future__ import annotations

import math

__all__ = [
    "parse_hazard_spec",
    "cumulative_hazard",
    "survival",
    "max_cumulative_hazard",
    "p_never",
    "sample_breakdown_time",
]


def parse_hazard_spec(block: dict) -> dict:
    """Validate a hazard-spec mapping and return {points: [(t, h), ...], decay: float}.

    Fails loudly on anything malformed (unknown type, unsorted/negative knots,
    negative hazards, decay outside (0, 1]) so a yaml typo can't silently sample
    a different survival curve.
    """
    if not isinstance(block, dict):
        raise ValueError(f"hazard spec must be a mapping, got {type(block).__name__}")
    kind = block.get("type")
    if kind != "linear_plus_decay":
        raise ValueError(f"hazard spec type must be 'linear_plus_decay', got {kind!r}")
    raw = block.get("points")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("hazard spec needs a non-empty 'points' list of [t, hazard] pairs")
    points = []
    for i, p in enumerate(raw):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"hazard points[{i}] must be a [t, hazard] pair, got {p!r}")
        t, h = float(p[0]), float(p[1])
        if not (math.isfinite(t) and math.isfinite(h)):
            raise ValueError(f"hazard points[{i}] must be finite, got {p!r}")
        if h < 0:
            raise ValueError(f"hazard points[{i}] hazard must be >= 0, got {h}")
        points.append((t, h))
    if points[0][0] != 0.0:
        raise ValueError(f"hazard points must start at t=0 (years after deal start), "
                         f"got first knot t={points[0][0]}")
    for (t0, _), (t1, _) in zip(points, points[1:]):
        if t1 <= t0:
            raise ValueError(f"hazard point times must be strictly increasing, got {t0} -> {t1}")
    if "decay" not in block:
        raise ValueError("hazard spec needs a 'decay' (per-year factor for the tail)")
    decay = float(block["decay"])
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"hazard decay must be in (0, 1], got {decay}")
    return {"points": points, "decay": decay}


def _knot_cumhaz(spec: dict) -> list:
    """Cumulative hazard at each knot (trapezoid over the linear segments)."""
    pts = spec["points"]
    H = [0.0]
    for (t0, h0), (t1, h1) in zip(pts, pts[1:]):
        H.append(H[-1] + 0.5 * (h0 + h1) * (t1 - t0))
    return H


def cumulative_hazard(spec: dict, t: float) -> float:
    """H(t) = integral of the hazard from 0 to t (t in years after deal start)."""
    if t <= 0:
        return 0.0
    pts, decay = spec["points"], spec["decay"]
    H = _knot_cumhaz(spec)
    for i, ((t0, h0), (t1, h1)) in enumerate(zip(pts, pts[1:])):
        if t <= t1:
            x = t - t0
            b = (h1 - h0) / (t1 - t0)
            return H[i] + h0 * x + 0.5 * b * x * x
    t_last, h_last = pts[-1]
    x = t - t_last
    if h_last == 0.0:
        return H[-1]
    if decay == 1.0:
        return H[-1] + h_last * x
    ln_d = math.log(decay)
    return H[-1] + h_last * (decay ** x - 1.0) / ln_d


def survival(spec: dict, t: float) -> float:
    """S(t) = P(deal still holding at t years after deal start)."""
    return math.exp(-cumulative_hazard(spec, t))


def max_cumulative_hazard(spec: dict) -> float:
    """H(inf). Finite when decay < 1 (or the last hazard is 0), else +inf."""
    pts, decay = spec["points"], spec["decay"]
    H_last = _knot_cumhaz(spec)[-1]
    h_last = pts[-1][1]
    if h_last == 0.0:
        return H_last
    if decay == 1.0:
        return math.inf
    return H_last + h_last / (-math.log(decay))


def p_never(spec: dict, min_years: float = 0.0) -> float:
    """P(the deal never breaks down), optionally CONDITIONED on it having survived the
    first ``min_years``: S(inf)/S(min_years) = exp(-(H(inf) - H(min_years)))."""
    H = max_cumulative_hazard(spec)
    if math.isinf(H):
        return 0.0
    return math.exp(-(H - cumulative_hazard(spec, min_years)))


def sample_breakdown_time(spec: dict, u: float, min_years: float = 0.0) -> float | None:
    """Inverse-CDF sample of the breakdown time (years after deal start).

    ``u`` is a uniform(0,1) draw interpreted as the survival level: the returned t
    solves S(t) = u, i.e. H(t) = -ln(u). Returns None when u < S(inf) — the deal
    never breaks down (possible whenever decay < 1). u -> 1 gives t -> 0.

    ``min_years`` CONDITIONS on the deal surviving at least that long ("deal survives
    for at least N years"): the draw is from the truncated distribution S(t)/S(min_years),
    i.e. H(t) = H(min_years) - ln(u), so the returned time is always >= min_years and
    the never-breaks probability rises to S(inf)/S(min_years).
    """
    if not (0.0 < u < 1.0):
        raise ValueError(f"u must be in (0, 1), got {u}")
    if min_years < 0:
        raise ValueError(f"min_years must be >= 0, got {min_years}")
    target = cumulative_hazard(spec, min_years) - math.log(u)
    if target > max_cumulative_hazard(spec):
        return None
    # If the hazard is 0 across min_years, H is flat there and the inversion returns the
    # EARLIEST time at that cumulative-hazard level — clamp to honor the conditioning.
    return max(_invert_cumhaz(spec, target), min_years)


def _invert_cumhaz(spec: dict, target: float) -> float:
    """Smallest t with H(t) = target. Caller guarantees target <= H(inf)."""
    pts, decay = spec["points"], spec["decay"]
    H = _knot_cumhaz(spec)
    # Linear segments: H(t0 + x) = H[i] + a*x + b/2*x^2 with a = h0 >= 0.
    for i, ((t0, h0), (t1, h1)) in enumerate(zip(pts, pts[1:])):
        if target <= H[i + 1]:
            rem = target - H[i]
            if rem <= 0.0:
                return t0
            a, b = h0, (h1 - h0) / (t1 - t0)
            if b == 0.0:
                # constant hazard on this segment; a > 0 since rem <= H_seg = a*(t1-t0)
                return t0 + rem / a
            disc = a * a + 2.0 * b * rem
            x = (math.sqrt(max(disc, 0.0)) - a) / b
            return t0 + min(max(x, 0.0), t1 - t0)
    # Exponential tail: H(t_last + x) = H_last + h_last * (decay^x - 1)/ln(decay).
    t_last, h_last = pts[-1]
    rem = target - H[-1]
    if rem <= 0.0:
        return t_last
    # h_last > 0 here (h_last == 0 would have made max_cumulative_hazard == H[-1] < target)
    if decay == 1.0:
        return t_last + rem / h_last
    ln_d = math.log(decay)
    return t_last + math.log(1.0 + rem * ln_d / h_last) / ln_d
