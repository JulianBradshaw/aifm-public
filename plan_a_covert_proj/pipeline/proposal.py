#!/usr/bin/env python3
"""Proposal-based importance sampling for AC-time conditioning.

Post-hoc AC reweighting (ac_reweight.py) conditions a run on AC ≈ target with Gaussian-kernel
weights, but a broad prior wastes most rollouts (low ESS). The unbiased fix is to SAMPLE from a
narrowed PROPOSAL distribution and correct each rollout's weight by the density ratio
prior(θ)/proposal(θ): the estimator still targets the same kernel-conditioned PRIOR — no bias —
while most rollouts now land in the high-weight region, so ESS per rollout goes way up.

Two entry points:

    # 1. fit a proposal config from an existing run's kernel-weighted marginals
    python -m plan_a_covert_proj.pipeline.proposal fit <runid> [--target 2030 --sigma 0.15]
        [--widen 1.25] [--out plan_a_covert_proj/config/<model>_proposal_ac2030.yaml]

    # 2. inspect the correction a proposal-run would get (read-only sanity check)
    python -m plan_a_covert_proj.pipeline.proposal check <runid>

The emitted yaml `parent:`s the ORIGINAL model config and overrides only the shifted parameter
dists; a `proposal_of:` key records the prior config. ac_reweight.py detects that key on the
run's model config and multiplies its kernel weights by exp(log prior(θ) − log proposal(θ)).

Density model (must mirror scripts/pipelines/batch_rollout.py EXACTLY):
  - marginals: fixed / uniform / normal / lognormal / shifted_lognormal / beta / choice, with the
    ci80 → (mu, sigma) conversions the sampler uses;
  - GLOBAL bounds clipping (progress_model.config.PARAMETER_BOUNDS, applied unless clip_to_bounds is
    explicitly false) puts atoms at the bounds: at a clipped value the correction uses the
    atom-MASS ratio, not the pdf ratio;
  - the Gaussian copula over `correlation_matrix.parameters` (spearman → closed-form latent
    matrix, deterministic): the joint is c_R(z) · Π f_i(x_i), so when a copula-member marginal
    changes, the correction includes log c_R(z_prior) − log c_R(z_prop) with
    z_i = Φ⁻¹(F_i(x_i)) under each measure. Fixed copula members never map to an x, so the
    density marginalizes to the latent SUB-matrix over non-fixed members (rows/cols taken
    AFTER the Higham projection, matching the sampler's z ~ N(0, R_full) exactly).

The proposal MUST use the same correlation_matrix as the prior (the fit never touches it) and
may only reshape marginals within the same family — both are asserted at correction time.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy.special as sp
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_config as _mcfg  # noqa: E402  (PARAMETER_BOUNDS — the sampler's global clip)
from scripts.pipelines.batch_rollout import (  # noqa: E402
    _load_config, _calibrate_gaussian_copula_latent_corr, _fit_beta_shapes_ci80)

RUNS = ROOT / "plan_a_covert_proj" / "outputs" / "runs"
_Z80 = 1.2815515655446004          # Φ⁻¹(0.9): the sampler's 80%-CI z-score
_EPS = 1e-12
_LOG_EPS = math.log(_EPS)          # out-of-support sentinel returned by log_pdf


# ---------------------------------------------------------------------------
# Canonical marginals (kind + resolved params), mirroring _sample_from_dist_with_quantile
# ---------------------------------------------------------------------------

def canon(spec: dict, name: str = "?") -> dict:
    """Resolve a yaml dist spec to canonical parameters (ci80 → mu/sigma etc.), exactly as the
    sampler does. Returns {'kind', ..., 'glo', 'ghi'} where glo/ghi are the GLOBAL clip bounds
    actually applied to samples (None when not clipped)."""
    kind = spec.get("dist", "fixed")
    # sampler clips to progress_model.config.PARAMETER_BOUNDS unless clip_to_bounds is explicitly false
    gb = _mcfg.PARAMETER_BOUNDS.get(name) if spec.get("clip_to_bounds", True) else None
    glo, ghi = (float(gb[0]), float(gb[1])) if gb is not None else (None, None)
    out = {"kind": kind, "glo": glo, "ghi": ghi}

    def _ci():
        if "ci80" in spec:
            q10, q90 = float(spec["ci80"][0]), float(spec["ci80"][1])
        else:
            q10, q90 = float(spec["ci80_low"]), float(spec["ci80_high"])
        return (q90, q10) if q10 > q90 else (q10, q90)

    if kind == "fixed":
        out["value"] = spec.get("value")
    elif kind == "uniform":
        out["a"], out["b"] = float(spec["min"]), float(spec["max"])
    elif kind == "normal":
        if "ci80" in spec or ("ci80_low" in spec and "ci80_high" in spec):
            q10, q90 = _ci()
            out["mu"], out["sigma"] = 0.5 * (q10 + q90), (q90 - q10) / (2.0 * _Z80)
        else:
            out["mu"] = float(spec["mean"])
            out["sigma"] = float(spec["sd"]) if "sd" in spec else float(spec.get("sigma", 1.0))
    elif kind in ("lognormal", "shifted_lognormal"):
        if "ci80" in spec or ("ci80_low" in spec and "ci80_high" in spec):
            q10, q90 = _ci()
            l10, l90 = math.log(q10), math.log(q90)
            out["mu"], out["sigma"] = 0.5 * (l10 + l90), (l90 - l10) / (2.0 * _Z80)
        else:
            out["mu"], out["sigma"] = float(spec["mu"]), float(spec["sigma"])
        out["shift"] = float(spec["shift"]) if kind == "shifted_lognormal" else 0.0
    elif kind == "beta":
        lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
        if "alpha" in spec and "beta" in spec:
            a, b = float(spec["alpha"]), float(spec["beta"])
        else:
            c0, c1 = (float(x) for x in spec["ci80"])
            u10, u90 = (c0 - lo) / (hi - lo), (c1 - lo) / (hi - lo)
            a, b = _fit_beta_shapes_ci80(round(u10, 9), round(u90, 9))
        out.update(a=a, b=b, lo=lo, hi=hi)
    elif kind == "choice":
        vals = list(spec["values"])
        p = spec.get("p")
        p = [1.0 / len(vals)] * len(vals) if p is None else [float(x) for x in p]
        s = sum(p)
        out["values"], out["p"] = vals, [x / s for x in p]
    else:
        raise ValueError(f"{name}: unsupported dist {kind!r}")
    # sampler-level clip (clip_to_bounds truthy AND spec min/max present) tightens the bounds
    if spec.get("clip_to_bounds") and "min" in spec and "max" in spec \
            and kind in ("normal", "lognormal", "shifted_lognormal"):
        slo, shi = float(spec["min"]), float(spec["max"])
        out["glo"] = slo if out["glo"] is None else max(out["glo"], slo)
        out["ghi"] = shi if out["ghi"] is None else min(out["ghi"], shi)
    return out


def _same(a: dict, b: dict, tol: float = 1e-12) -> bool:
    if a["kind"] != b["kind"]:
        return False
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) and isinstance(vb, float):
            if not (math.isclose(va, vb, rel_tol=1e-9, abs_tol=tol)
                    or (math.isnan(va) and math.isnan(vb))):
                return False
        elif va != vb:
            return False
    return True


def _norm_cdf(z):
    return 0.5 * (1.0 + sp.erf(np.asarray(z, float) / math.sqrt(2.0)))


def cdf(c: dict, x: float) -> float:
    """CDF of the UNCLIPPED canonical marginal at x."""
    k = c["kind"]
    if k == "uniform":
        return float(np.clip((x - c["a"]) / (c["b"] - c["a"]), 0.0, 1.0))
    if k == "normal":
        return float(_norm_cdf((x - c["mu"]) / c["sigma"]))
    if k in ("lognormal", "shifted_lognormal"):
        y = x - c["shift"]
        if y <= 0:
            return 0.0
        return float(_norm_cdf((math.log(y) - c["mu"]) / c["sigma"]))
    if k == "beta":
        u = (x - c["lo"]) / (c["hi"] - c["lo"])
        return float(sp.betainc(c["a"], c["b"], float(np.clip(u, 0.0, 1.0))))
    raise ValueError(f"cdf: unsupported kind {k}")


def log_pdf(c: dict, x) -> float:
    """Log density of the canonical marginal at x, ATOM-AWARE: at a clipped bound the 'density'
    is the clipped-off probability mass (so ratios of two clipped dists stay correct)."""
    k = c["kind"]
    if k == "choice":
        for v, p in zip(c["values"], c["p"]):
            if v == x or (isinstance(v, (int, float)) and isinstance(x, (int, float))
                          and math.isclose(float(v), float(x), rel_tol=1e-9)):
                return math.log(max(p, _EPS))
        return _LOG_EPS
    x = float(x)
    glo, ghi = c.get("glo"), c.get("ghi")
    if glo is not None and x <= glo + 1e-15 and cdf(c, glo) > 1e-15:
        return math.log(max(cdf(c, glo), _EPS))          # lower atom mass
    if ghi is not None and x >= ghi - 1e-15 and 1.0 - cdf(c, ghi) > 1e-15:
        return math.log(max(1.0 - cdf(c, ghi), _EPS))    # upper atom mass
    if k == "uniform":
        return -math.log(c["b"] - c["a"]) if c["a"] <= x <= c["b"] else _LOG_EPS
    if k == "normal":
        z = (x - c["mu"]) / c["sigma"]
        return -0.5 * z * z - math.log(c["sigma"]) - 0.5 * math.log(2 * math.pi)
    if k in ("lognormal", "shifted_lognormal"):
        y = x - c["shift"]
        if y <= 0:
            return _LOG_EPS
        z = (math.log(y) - c["mu"]) / c["sigma"]
        return (-0.5 * z * z - math.log(c["sigma"]) - math.log(y)
                - 0.5 * math.log(2 * math.pi))
    if k == "beta":
        u = (x - c["lo"]) / (c["hi"] - c["lo"])
        if not (0.0 < u < 1.0):
            return _LOG_EPS
        lb = (sp.gammaln(c["a"] + c["b"]) - sp.gammaln(c["a"]) - sp.gammaln(c["b"])
              + (c["a"] - 1) * math.log(u) + (c["b"] - 1) * math.log(1 - u)
              - math.log(c["hi"] - c["lo"]))
        return float(lb)
    raise ValueError(f"log_pdf: unsupported kind {k}")


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------

def latent_submatrix(corr_spec: dict, param_dists: dict):
    """(names, R_sub): the latent Gaussian correlation over the NON-FIXED copula members.
    Recomputes the full latent matrix exactly as mc.py does (spearman closed form + Higham),
    then marginalizes to the non-fixed rows/cols (a MVN marginal is just the sub-matrix)."""
    cpar = list(corr_spec.get("parameters", []))
    tcorr = np.array(corr_spec["correlation_matrix"], float)
    interp = str(corr_spec.get("correlation_type", corr_spec.get("interpretation", "spearman")))
    if interp.lower() not in {"spearman", "rank", "spearman_rho"}:
        raise ValueError("proposal correction only supports spearman copulas (the pearson/NORTA "
                         "latent calibration depends on the marginals, so prior and proposal "
                         "would need different latent matrices)")
    lat = _calibrate_gaussian_copula_latent_corr(
        param_dists, cpar, tcorr, np.random.default_rng(0), interpretation=interp)
    keep = [i for i, nm in enumerate(cpar)
            if canon(param_dists[nm], nm)["kind"] != "fixed"]
    names = [cpar[i] for i in keep]
    return names, lat[np.ix_(keep, keep)]


def copula_logc(R: np.ndarray, z: np.ndarray) -> float:
    """Log Gaussian-copula density at latent z: −½ zᵀ(R⁻¹−I)z − ½ log det R."""
    Rinv = np.linalg.inv(R)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        raise ValueError("latent correlation matrix is not PD")
    return float(-0.5 * z @ (Rinv - np.eye(len(z))) @ z - 0.5 * logdet)


def _z_of(c: dict, x: float) -> float:
    u = float(np.clip(cdf(c, float(x)), 1e-10, 1.0 - 1e-10))
    return float(sp.erfinv(2.0 * u - 1.0) * math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Correction: log prior(θ) − log proposal(θ) per record
# ---------------------------------------------------------------------------

def load_proposal_info(model_config_path: str):
    """If model_config_path is a proposal config (has `proposal_of`), return
    (prior_cfg, prop_cfg, prior_raw_path); else None."""
    p = ROOT / model_config_path
    raw = yaml.safe_load(p.read_text()) or {}
    prior_path = raw.get("proposal_of")
    if not prior_path:
        return None
    prior_cfg = _load_config(str(ROOT / prior_path))
    prop_cfg = _load_config(str(p))
    return prior_cfg, prop_cfg, prior_path


def correction_log_weights(records: list, model_config_path: str, verbose: bool = True):
    """log[prior(θ_i)/proposal(θ_i)] for each record, or None when the run's model config is
    not a proposal. Marginal terms over changed params + the Gaussian-copula term when a
    copula member changed."""
    info = load_proposal_info(model_config_path)
    if info is None:
        return None
    prior_cfg, prop_cfg, prior_path = info
    prior_pd = prior_cfg.get("parameters", {}) or {}
    prop_pd = prop_cfg.get("parameters", {}) or {}

    prior_corr, prop_corr = prior_cfg.get("correlation_matrix"), prop_cfg.get("correlation_matrix")
    same_corr = (json.dumps(prior_corr, sort_keys=True, default=str)
                 == json.dumps(prop_corr, sort_keys=True, default=str))

    common = sorted(set(prior_pd) & set(prop_pd))
    if set(prior_pd) != set(prop_pd):
        raise ValueError(f"prior/proposal parameter sets differ: "
                         f"{sorted(set(prior_pd) ^ set(prop_pd))}")
    cn_prior = {k: canon(prior_pd[k], k) for k in common}
    cn_prop = {k: canon(prop_pd[k], k) for k in common}
    changed = [k for k in common if not _same(cn_prior[k], cn_prop[k])]
    for k in changed:
        if cn_prior[k]["kind"] != cn_prop[k]["kind"]:
            raise ValueError(f"{k}: proposal changes the dist FAMILY "
                             f"({cn_prior[k]['kind']} → {cn_prop[k]['kind']}) — not supported")
        if cn_prior[k]["kind"] == "fixed":
            raise ValueError(f"{k}: cannot change a fixed param (no density)")

    # Copula terms. Each side's copula density enters over ITS OWN non-fixed member set (with
    # its own marginal CDFs). When both copulas AND all their members' marginals are identical
    # the terms cancel — skip them. Otherwise both are evaluated in full.
    def _cop(corr_spec, pd, cn):
        if not corr_spec or not corr_spec.get("parameters"):
            return [], None
        names, R = latent_submatrix(corr_spec, pd)
        return names, R
    p_names, p_R = _cop(prior_corr, prior_pd, cn_prior)
    q_names, q_R = _cop(prop_corr, prop_pd, cn_prop)
    if same_corr and not any(k in p_names for k in changed):
        p_names, q_names = [], []          # identical terms — cancel

    if verbose:
        print(f"proposal correction: {len(changed)} changed marginal(s): {changed}")
        if p_names or q_names:
            print(f"  + copula terms: prior over {p_names or '—'}, proposal over {q_names or '—'}")

    logw = np.zeros(len(records))
    n_atom = 0
    for i, r in enumerate(records):
        mp = r.get("model_params") or {}
        s = 0.0
        for k in changed:
            x = mp.get(k)
            if x is None:
                raise ValueError(f"record {i} missing sampled value for changed param {k!r}")
            lp, lq = log_pdf(cn_prior[k], x), log_pdf(cn_prop[k], x)
            if math.isclose(lp, _LOG_EPS):      # exact out-of-support sentinel, not just "small"
                n_atom += 1
            s += lp - lq
        if p_names:
            zp = np.array([_z_of(cn_prior[k], mp[k]) for k in p_names])
            s += copula_logc(p_R, zp)
        if q_names:
            zq = np.array([_z_of(cn_prop[k], mp[k]) for k in q_names])
            s -= copula_logc(q_R, zq)
        logw[i] = s
    if n_atom and verbose:
        print(f"  WARNING: {n_atom} record-param values had ~zero prior density "
              f"(outside prior support?) — their weights are ~0")
    return logw


# ---------------------------------------------------------------------------
# Fit: proposal config from a run's kernel-weighted marginals
# ---------------------------------------------------------------------------

def _wmoments(x, w):
    m = float(np.sum(w * x) / np.sum(w))
    v = float(np.sum(w * (x - m) ** 2) / np.sum(w))
    return m, math.sqrt(max(v, 1e-300))


def _arrival(r) -> float:
    """Freeze-at-detection covert TED-AI date: +inf when detected before the goal or never
    reaching it (the censoring used by the p10 early-arrival statistic)."""
    if r.get("covert_ttt") is None or r.get("detected_before_goal"):
        return math.inf
    return float(r["covert_ttt"])


def fit_kernel(records, target: float, sigma: float,
               corner_t: float | None = None, corner_tau: float = 2.0):
    """The FIT kernel: AC-Gaussian × (optionally) a smooth corner kernel on the
    freeze-at-detection arrival, exp(−max(A−corner_t, 0)/corner_tau) — 1 for rollouts arriving
    (undetected) by corner_t, decaying for later arrivals, 0 for censored ones. Tilting the
    proposal toward the corner that determines the p10 early-arrival date."""
    ac = np.array([r["ac_time"] if r.get("ac_time") is not None else np.nan for r in records],
                  float)
    k = np.exp(-0.5 * ((ac - target) / sigma) ** 2)
    k = np.where(np.isfinite(k), k, 0.0)
    if corner_t is not None:
        A = np.array([_arrival(r) for r in records])
        pen = np.where(np.isfinite(A), np.exp(-np.maximum(A - corner_t, 0.0) / corner_tau), 0.0)
        k = k * pen
    return k


def balance_weights(logr: np.ndarray, alpha: float) -> np.ndarray:
    """Balance-heuristic p/m factor for a MIXTURE sampling scheme m = α·prior + (1−α)·proposal:
    p/m = 1 / (α + (1−α)·q/p), computable from logr = log(p/q) alone. Apply to the pooled
    records of BOTH runs (an α-share of rollouts sampled from the prior config, the rest from
    the proposal config), multiplied by the AC kernel. Bounded above by 1/α — the mixture's
    insurance against proposal blind spots."""
    return 1.0 / (alpha + (1.0 - alpha) * np.exp(-np.asarray(logr, float)))


def fit_proposal(run: str, target: float, sigma: float, widen: float = 1.3,
                 corr_shrink: float = 0.3, out: str | None = None,
                 runs_root: Path | None = None,
                 corner_t: float | None = None, corner_tau: float = 2.0) -> Path:
    """Fit a proposal config for AC≈target from an existing run.

    Marginals: every continuous sampled param gets the kernel-weighted mean/sd in its family's
    natural space, sd widened by ``widen`` (floored so the proposal never collapses). Emitted
    in ci80 FORM: the config deep-merge cannot clear a parent's ci80 keys, and the sampler
    prefers ci80 over mu/sigma — a mu/sigma override would silently be ignored.

    Joint: the kernel-weighted Spearman correlation matrix over those params (shrunk toward
    independence by ``corr_shrink`` for stability), emitted as the proposal's
    correlation_matrix. This is what captures the AC≈target SHELL — the conditioning barely
    moves individual marginals but correlates the params that jointly determine AC.

    Diagnostics: predicts the proposal run's ESS/rollout from the source run before you spend
    any compute (ESS/N = (E_p k)² / E_p[k²·p/q], all estimable on prior samples).
    """
    runs_root = runs_root or RUNS
    ds = json.load(open(runs_root / run / "dataset.json"))
    R = ds["records"]
    model_cfg_path = ds["config"]["model_config"]
    if yaml.safe_load((ROOT / model_cfg_path).read_text()).get("proposal_of"):
        raise ValueError("source run already uses a proposal config — fit from a prior run")
    prior_cfg = _load_config(str(ROOT / model_cfg_path))
    prior_pd = prior_cfg.get("parameters", {}) or {}

    k = fit_kernel(R, target, sigma, corner_t=corner_t, corner_tau=corner_tau)
    if k.sum() <= 0:
        raise ValueError("all fit-kernel weights are zero — bad target/sigma/corner for this run")
    w = k / k.sum()
    fit_ess = float(k.sum() ** 2 / np.sum(k ** 2))
    print(f"fit kernel ESS {fit_ess:.0f}/{len(R)}"
          + (f" (corner_t={corner_t:g}, tau={corner_tau:g})" if corner_t is not None else ""))

    overrides, notes, fitted_cont = {}, [], []
    for name, spec in sorted(prior_pd.items()):
        c = canon(spec, name)
        kind = c["kind"]
        if kind not in ("normal", "lognormal", "shifted_lognormal"):
            continue                                  # fixed/uniform/beta/choice: keep the prior
        xf = np.array([float(r["model_params"][name]) for r in R], float)
        ok = np.isfinite(xf) & (w > 0)
        if ok.sum() < 30:
            continue
        t = xf[ok] if kind == "normal" else np.log(np.maximum(xf[ok] - c["shift"], 1e-300))
        mu_w, sd_w = _wmoments(t, w[ok] / w[ok].sum())
        sd_p = max(sd_w * widen, 0.35 * c["sigma"])   # floor: never collapse the proposal
        # ci80 form (see docstring). For (shifted_)lognormal ci80 is in ORIGINAL space.
        if kind == "normal":
            ci = [mu_w - _Z80 * sd_p, mu_w + _Z80 * sd_p]
        else:
            ci = [math.exp(mu_w - _Z80 * sd_p), math.exp(mu_w + _Z80 * sd_p)]
        ov = {"dist": kind, "ci80": [round(float(ci[0]), 8), round(float(ci[1]), 8)]}
        if kind == "shifted_lognormal":
            ov["shift"] = c["shift"]
        for b in ("min", "max"):                      # min/max are required spec keys
            if b in spec:
                ov[b] = float(spec[b])
        if "clip_to_bounds" in spec:
            ov["clip_to_bounds"] = spec["clip_to_bounds"]
        overrides[name] = ov
        fitted_cont.append(name)
        notes.append(f"{name} ({kind}): mu {c['mu']:.3f}->{mu_w:.3f}, "
                     f"sigma {c['sigma']:.3f}->{sd_p:.3f}")

    if not overrides:
        raise ValueError("no continuous sampled params to fit a proposal from")

    # Kernel-weighted Spearman correlation over the fitted params (the AC shell), shrunk.
    from scipy.stats import rankdata
    n = len(R)
    Rk = np.array([rankdata([float(r["model_params"][nm]) for r in R])
                   for nm in fitted_cont]).T / (n + 1)
    m = (w[:, None] * Rk).sum(0)
    C = np.einsum("i,ij,ik->jk", w, Rk - m, Rk - m)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    C = (1.0 - corr_shrink) * C
    np.fill_diagonal(C, 1.0)
    corr_block = {
        "parameters": fitted_cont,
        "correlation_matrix": [[round(float(x), 4) for x in row] for row in C],
        "correlation_type": "spearman",
    }

    model_name = Path(model_cfg_path).stem
    tag = f"ac{target:g}_s{sigma:g}".replace(".", "p")
    if corner_t is not None:
        tag += f"_corner{corner_t:g}".replace(".", "p")
    out_path = ((ROOT / out) if (out and not Path(out).is_absolute()) else Path(out)) if out \
        else (ROOT / "plan_a_covert_proj" / "config" / f"{model_name}_proposal_{tag}.yaml")
    doc = {
        "proposal_of": model_cfg_path,
        "proposal_meta": {"source_run": run, "target": float(target), "sigma": float(sigma),
                          "widen": float(widen), "corr_shrink": float(corr_shrink),
                          "corner_t": (float(corner_t) if corner_t is not None else None),
                          "corner_tau": float(corner_tau),
                          "fitted": sorted(overrides)},
        "parent": Path(model_cfg_path).name,          # same config dir as the prior
        "parameters": overrides,
        "correlation_matrix": corr_block,
    }
    header = ("# AUTO-GENERATED proposal config (plan_a_covert_proj/pipeline/proposal.py fit)\n"
              f"# Prior: {model_cfg_path} · fitted from run {run} kernel AC~{target} s={sigma}\n"
              "# Sampling from this config REQUIRES the prior/proposal weight correction\n"
              "# (ac_reweight.py applies it automatically via the proposal_of key below).\n"
              + "".join(f"#   {n}\n" for n in notes))
    out_path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    print(f"proposal -> {out_path.relative_to(ROOT)}")
    for nmsg in notes:
        print(f"  {nmsg}")

    # --- predicted performance of sampling from this proposal (vs the prior baseline) -------
    score_p10(R, str(out_path.relative_to(ROOT)), target, sigma)
    return out_path


def score_p10(records, proposal_path: str, target: float, sigma: float,
              alphas=(1.0, 0.7, 0.5, 0.3, 0.0), q10: float = 0.10):
    """Predict, from PRIOR-sampled records, how precisely a run sampled from the mixture
    m = α·prior + (1−α)·proposal would estimate (a) the AC-kernel-weighted mean (total-mass
    ESS/rollout) and (b) the p10 freeze-at-detection early-arrival date.

    For (b) the figure of merit is the variance of the weighted CDF at the current p10
    estimate t* — the p10 date's sampling interval shrinks by √gain. Everything is estimable
    on prior samples: with w = k·p/m,  E_m[w] = E_p[k]  and  E_m[w²h] = E_p[k²·(p/m)·h].
    α=1 is the all-prior baseline; α=0 is the pure proposal (its row can be optimistic — the
    p/m factor is unbounded there, so watch the max-share column)."""
    k = fit_kernel(records, target, sigma)          # AC kernel only — the ANALYSIS kernel
    if k.sum() <= 0:
        raise ValueError("all AC-kernel weights are zero")
    A = np.array([_arrival(r) for r in records])
    o = np.argsort(A)
    cw = np.cumsum(k[o]) / k.sum()
    i10 = int(np.searchsorted(cw, q10))
    t_star = float(A[o][min(i10, len(A) - 1)])
    if not math.isfinite(t_star):
        raise ValueError(f"the weighted p{q10*100:g} arrival is censored (never) in this run")
    g = (A <= t_star).astype(float)
    F = float(np.sum(k * g) / k.sum())
    logr = correction_log_weights(records, proposal_path, verbose=False)
    print(f"p10 scoring: t* = {t_star:.2f} (weighted CDF there {F:.3f}); "
          f"interval width shrinks by sqrt(gain)")
    print(f"{'alpha':>7}{'ESS/rollout':>13}{'var(F(t*))·N':>14}{'gain':>7}{'max-share':>11}")
    base_var = None
    for a in alphas:
        pm = balance_weights(logr, a)               # p/m on prior samples
        ess = float(np.mean(k) ** 2 / np.mean(k ** 2 * pm))
        contrib = k ** 2 * (g - F) ** 2 * pm
        var_n = float(np.mean(contrib) / np.mean(k) ** 2)
        if base_var is None:
            base_var = var_n                        # first alpha (=1.0) is the baseline
        share = float(np.max(contrib) / max(np.sum(contrib), 1e-300))
        print(f"{a:>7.2f}{ess:>13.4f}{var_n:>14.4f}{base_var/max(var_n,1e-300):>7.2f}"
              f"{share:>11.2f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit a proposal config from a run")
    f.add_argument("run")
    f.add_argument("--target", type=float, default=2030.0)
    f.add_argument("--sigma", type=float, default=0.15)
    f.add_argument("--widen", type=float, default=1.3)
    f.add_argument("--corr-shrink", type=float, default=0.3)
    f.add_argument("--out", default=None)
    f.add_argument("--runs-root", default=None,
                   help="alternate outputs/runs root to read the source run from")
    f.add_argument("--corner-t", type=float, default=None,
                   help="also tilt the fit toward rollouts whose freeze-at-detection covert "
                        "TED-AI arrival is <= this year (pass 'auto' semantics via score first); "
                        "targets the p10 early-arrival statistic")
    f.add_argument("--corner-tau", type=float, default=2.0,
                   help="softness (years) of the corner kernel's decay past corner-t")
    s = sub.add_parser("score", help="predict p10-precision gain of a proposal + mixture "
                                     "from an existing prior run (no compute spent)")
    s.add_argument("run")
    s.add_argument("proposal", help="proposal yaml path (repo-relative)")
    s.add_argument("--target", type=float, default=2030.0)
    s.add_argument("--sigma", type=float, default=0.15)
    s.add_argument("--runs-root", default=None)
    c = sub.add_parser("check", help="print the correction summary for a proposal run")
    c.add_argument("run")
    c.add_argument("--runs-root", default=None)
    a = ap.parse_args()
    root = Path(a.runs_root) if a.runs_root else RUNS
    if a.cmd == "fit":
        fit_proposal(a.run, a.target, a.sigma, widen=a.widen, corr_shrink=a.corr_shrink,
                     out=a.out, runs_root=root,
                     corner_t=a.corner_t, corner_tau=a.corner_tau)
        return
    if a.cmd == "score":
        ds = json.load(open(root / a.run / "dataset.json"))
        score_p10(ds["records"], a.proposal, a.target, a.sigma)
        return
    ds = json.load(open(root / a.run / "dataset.json"))
    logw = correction_log_weights(ds["records"], ds["config"]["model_config"])
    if logw is None:
        print("not a proposal run (model config has no proposal_of) — no correction")
        return
    w = np.exp(logw - logw.max())
    ess = w.sum() ** 2 / np.sum(w ** 2)
    print(f"correction-only ESS {ess:.0f}/{len(w)} · log-ratio range "
          f"[{logw.min():.2f}, {logw.max():.2f}]")


if __name__ == "__main__":
    main()
