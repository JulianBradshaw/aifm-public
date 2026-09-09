#!/usr/bin/env python3
"""OPTIONAL post-hoc AC-time conditioning via importance reweighting.

NOT part of the default pipeline and nothing here runs automatically — invoke on demand to
re-derive the covert results conditioned on AC happening near a target year (default from
config/ac_reweight.yaml), WITHOUT re-running the model and WITHOUT narrowing any parameter
distribution. Full model uncertainty is kept; rollouts are just reweighted toward the target AC.

Three modes:

    # 1. stats table: unweighted vs reweighted headline metrics (read-only)
    python -m plan_a_covert_proj.pipeline.ac_reweight <runid>

    # 2. ESS vs sigma sweep (read-only) — how much conditioning costs you
    python -m plan_a_covert_proj.pipeline.ac_reweight <runid> --sweep

    # 3. emit: regenerate ALL figures with weights into a SIBLING run dir, for the viewer
    python -m plan_a_covert_proj.pipeline.ac_reweight <runid> --emit

Weights:  w_i ∝ exp(-0.5 * ((ac_i - target) / sigma)^2)   (Gaussian kernel on each rollout's AC
time). This is the Nadaraya–Watson conditional estimator: as sigma→0 the weighted distribution
converges to p(outcome | AC = target). Effective sample size  ESS = (Σw)^2 / Σ(w^2)  reports how
many rollouts effectively remain; shrink sigma for a tighter condition at the cost of ESS.

--emit is LOSSLESS: every figure uses all N rollouts with their weights (no rollout dropped or
duplicated), and each reweighted figure is stamped with its ESS so it can never look better
resolved than it is. It writes a SEPARATE run dir (<runid>__acrw<target>_s<sigma>) — it never
touches the source run's figures. Target/sigma/seed/enabled come from config/ac_reweight.yaml;
CLI flags override. --emit is gated by the yaml `enabled` flag (override once with --enable).
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNS = ROOT / "plan_a_covert_proj" / "outputs" / "runs"
CONFIG = ROOT / "plan_a_covert_proj" / "config" / "ac_reweight.yaml"


def _load_cfg():
    """Defaults from config/ac_reweight.yaml (with hard-coded fallbacks if the file is absent)."""
    base = {"enabled": True, "target": 2030.0, "sigma": 0.3, "seed": 0}
    try:
        base.update({k: v for k, v in (yaml.safe_load(CONFIG.read_text()) or {}).items()
                     if k in base})
    except FileNotFoundError:
        pass
    return base


def _wmean(v, w):
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    return float(np.sum(w[m] * v[m]) / np.sum(w[m]))


def _wquantile(v, w, q):
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    v, ww = v[m], w[m]
    i = np.argsort(v)
    v, ww = v[i], ww[i]
    cw = (np.cumsum(ww) - 0.5 * ww) / ww.sum()      # Hazen-style weighted plotting positions
    return float(np.interp(q, cw, v))


def _metrics(R, w):
    ac = np.array([r["ac_time"] for r in R], float)
    win = np.array([1.0 if r["covert_wins"] else 0.0 for r in R])
    surv = np.array([float(r["survival_at_goal"]) if r.get("survival_at_goal") is not None
                     and np.isfinite(r["survival_at_goal"]) else 0.0 for r in R])
    cttt = np.array([(r["covert_ttt"] - r["pause_year"]) if r.get("covert_ttt") is not None
                     else np.nan for r in R])
    lead = np.array([r["lead_years"] if r.get("lead_years") is not None else np.nan for r in R])
    det = np.array([1.0 if r.get("detected_before_goal") else 0.0 for r in R])
    dep = np.array([r.get("total_h100e_used") if r.get("total_h100e_used") else np.nan for r in R])
    acm = _wmean(ac, w)
    return {
        "AC mean (yr)": acm,
        "AC sd (yr)": float(np.sqrt(max(0.0, _wmean((ac - acm) ** 2, w)))),
        "P(covert wins)": _wmean(win, w),
        "E[success]=win*surv": _wmean(win * surv, w),
        "covert yrs pause->TED (med)": _wquantile(cttt, w, 0.5),
        "lead_years (med)": _wquantile(lead, w, 0.5),
        "P(detected<goal)": _wmean(det, w),
        "deployed H100e (med, M)": _wquantile(dep, w, 0.5) / 1e6,
    }


def _ac_times(R):
    """Per-rollout AC time as a float array (non-finite where AC is missing)."""
    return np.array([r["ac_time"] if r.get("ac_time") is not None else np.nan for r in R], float)


def _weights(ac, target, sigma):
    """Gaussian-kernel importance weights; 0 where AC is undefined (can't be near the target)."""
    w = np.exp(-0.5 * ((ac - target) / sigma) ** 2)
    return np.where(np.isfinite(w), w, 0.0)


def _proposal_correction(ds):
    """When the run was SAMPLED from a proposal config (model config has `proposal_of`),
    return the per-record multiplicative correction prior(θ)/proposal(θ) — the factor that
    makes kernel-weighted estimates target the kernel-conditioned PRIOR, unbiased. Returns
    None for ordinary (prior-sampled) runs. See pipeline/proposal.py."""
    model_cfg = (ds.get("config") or {}).get("model_config")
    if not model_cfg:
        return None
    from plan_a_covert_proj.pipeline.proposal import correction_log_weights
    logw = correction_log_weights(ds["records"], model_cfg)
    if logw is None:
        return None
    return np.exp(logw - np.max(logw))      # scale-invariant downstream (self-normalized)


def _ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0


def _dataset_path(run):
    return RUNS / run / "dataset.json"


def _emit(run, target, sigma, seed, cfg):
    """Regenerate every figure with AC-time weights into a sibling run dir, refresh the viewer."""
    from plan_a_covert_proj.pipeline import figures as figmod
    from plan_a_covert_proj.pipeline import make_viewer

    src = RUNS / run
    ds = json.loads(_dataset_path(run).read_text())
    R = ds["records"]
    n = len(R)
    ac = _ac_times(R)
    w = _weights(ac, target, sigma)
    corr = _proposal_correction(ds)
    if corr is not None:
        kernel_ess = _ess(w)
        w = w * corr
        print(f"proposal run: prior/proposal correction applied "
              f"(kernel-only ESS {kernel_ess:.0f} -> corrected ESS {_ess(w):.0f})")
    if w.sum() <= 0:
        sys.exit(f"all weights zero — no rollouts within reach of AC={target} (sigma={sigma}). "
                 f"AC range in this run: [{np.nanmin(ac):.2f}, {np.nanmax(ac):.2f}].")
    ess = _ess(w)
    # Normalize to sum=N so weighted histogram heights stay comparable to the source run's counts
    # (pure rescale; means / quantiles / ESS are scale-invariant, so this changes nothing else).
    w_norm = w * (n / w.sum())

    # Attach weights + ESS banner metadata; figures pick these up via figures._wts / _rw_note.
    ds["_weights"] = w_norm.tolist()
    ds["_reweight_meta"] = {"target": float(target), "sigma": float(sigma),
                            "ess": float(ess), "n": int(n), "source_run": run}

    tag = f"acrw{target:g}_s{sigma:g}".replace(".", "p")
    dst = RUNS / f"{run}__{tag}"
    (dst / "figures").mkdir(parents=True, exist_ok=True)

    built, changed = [], []
    for name, fn in figmod.FIGURES:
        try:
            r = fn(ds, dst / "figures")
            if r is None:
                continue
            built.append(r)
            print(f"  figure {name} -> {r}")
        except Exception as e:  # noqa: BLE001
            print(f"  figure {name} FAILED: {e}")
    # Which figures actually differ from the source run (i.e. genuinely reflect the reweighting)?
    for fname in built:
        a, b = src / "figures" / fname, dst / "figures" / fname
        if a.exists() and b.exists() and a.read_bytes() != b.read_bytes():
            changed.append(fname)

    # Manifest mirrors run.py's shape so the viewer renders it as a sibling run. We copy the
    # source manifest's config/params/snapshot, set runid to THIS dir (so figure paths resolve),
    # update the win-rate to the reweighted value, and record the reweighting provenance.
    src_manifest = json.loads((src / "manifest.json").read_text()) if (src / "manifest.json").exists() else {}
    summary = dict(src_manifest.get("summary") or ds.get("summary") or {})
    summary["covert_win_frac"] = _wmean([1.0 if r["covert_wins"] else 0.0 for r in R], w_norm)
    summary["ac_reweight_ess"] = ess
    manifest = dict(src_manifest)
    manifest.update({
        "runid": f"{run}__{tag}",
        "summary": summary,
        "ac_reweight": {"source_run": run, "target": float(target), "sigma": float(sigma),
                        "seed": int(seed), "ess": float(ess), "n": int(n),
                        "ess_frac": float(ess / n),
                        "proposal_corrected": bool(corr is not None)},
        "figures": sorted(p.name for p in (dst / "figures").iterdir() if p.is_file()),
        "figures_built_now": built,
        "figures_changed_vs_source": changed,
    })
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Small companion file: the weights + provenance, so the run is reproducible/auditable
    # without re-duplicating the 37 MB dataset (figures already bake the weighting in).
    (dst / "reweight.json").write_text(json.dumps({
        "source_run": run, "target": float(target), "sigma": float(sigma), "seed": int(seed),
        "ess": float(ess), "n": int(n), "weights": w_norm.tolist(),
        "proposal_corrected": bool(corr is not None),
    }))

    print(f"\nreweighted run -> {dst.relative_to(ROOT)}")
    print(f"ESS {ess:.0f}/{n} ({100*ess/n:.0f}% kept) · AC mean {_wmean(ac, w):.3f} "
          f"(unweighted {np.nanmean(ac):.3f}) · {len(built)} figures, {len(changed)} changed vs source")
    if changed:
        print("  changed: " + ", ".join(changed))
    # NOTE: deliberately does NOT move outputs/latest — the source run stays the canonical one.
    _vpath, _vn = make_viewer.regenerate()
    if _vpath is not None:
        print(f"viewer -> {_vpath.relative_to(ROOT)} ({_vn} runs)")


def main():
    cfg = _load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runid under plan_a_covert_proj/outputs/runs/")
    ap.add_argument("--target", type=float, default=None, help="target AC year (default: yaml)")
    ap.add_argument("--sigma", type=float, default=None, help="kernel bandwidth in yr (default: yaml)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: yaml)")
    ap.add_argument("--sweep", action="store_true", help="print ESS + headline stats across sigma")
    ap.add_argument("--emit", action="store_true",
                    help="regenerate all figures with weights into a sibling run dir")
    ap.add_argument("--enable", action="store_true",
                    help="run --emit even if the yaml has enabled:false (one-off override)")
    a = ap.parse_args()
    target = a.target if a.target is not None else float(cfg["target"])
    sigma = a.sigma if a.sigma is not None else float(cfg["sigma"])
    seed = a.seed if a.seed is not None else int(cfg["seed"])

    if a.emit:
        if not bool(cfg.get("enabled", True)) and not a.enable:
            sys.exit("ac_reweight is disabled (config/ac_reweight.yaml: enabled:false). "
                     "Set enabled:true, or pass --enable for a one-off. (Read-only --sweep/stats "
                     "still work, and the source run's figures are unaffected either way.)")
        _emit(a.run, target, sigma, seed, cfg)
        return

    ds = json.loads(_dataset_path(a.run).read_text())
    R = ds["records"]
    ac = _ac_times(R)
    corr = _proposal_correction(ds)
    if corr is not None:
        print("proposal run: prior/proposal correction applied to all weights below "
              "(incl. the 'unweighted' column, which shows the corrected-but-unconditioned "
              "prior estimate)")

    def _w_of(s):
        w = np.ones(len(R)) if not np.isfinite(s) else _weights(ac, target, s)
        return w * corr if corr is not None else w

    if a.sweep:
        print(f"run {a.run} · N={len(R)} · target AC={target}")
        print(f"{'sigma':>8}{'ESS':>8}{'%kept':>7}{'AC mean':>10}{'P(win)':>9}{'E[succ]':>9}")
        for s in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, np.inf):
            w = _w_of(s)
            m = _metrics(R, w); e = _ess(w)
            print(f"{('unwt' if not np.isfinite(s) else f'{s:.2f}'):>8}{e:>8.0f}{100*e/len(R):>7.0f}"
                  f"{m['AC mean (yr)']:>10.3f}{m['P(covert wins)']:>9.3f}{m['E[success]=win*surv']:>9.3f}")
        return

    w = _w_of(sigma); e = _ess(w)
    mu, mw = _metrics(R, _w_of(np.inf)), _metrics(R, w)
    print(f"run {a.run} · N={len(R)} · AC-conditioning: target={target}, sigma={sigma}")
    print(f"effective sample size: {e:.0f} / {len(R)}  ({100*e/len(R):.0f}% kept)")
    print(f"\n{'metric':30s}{'unweighted':>14}{f'AC~{target:g}':>14}")
    for k in mu:
        print(f"{k:30s}{mu[k]:>14.4g}{mw[k]:>14.4g}")


if __name__ == "__main__":
    main()
