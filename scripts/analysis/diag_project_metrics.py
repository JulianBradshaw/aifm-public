"""Diagnostic: re-run the canonical AIFM metrics pass (compute_metrics_loop) on each covert
race trajectory (covert, legal_discovered, legal_used) and surface where it diverges from the
covert sim's ad-hoc per-trajectory metrics.

The goal is to SURFACE SUBTLETIES, not to assert pass/fail. Run:

    PYTHONPATH=. python scripts/analysis/diag_project_metrics.py
"""
from __future__ import annotations

import logging
import os

import numpy as np

os.environ.setdefault("PYTHONPATH", ".")
logging.disable(logging.CRITICAL)  # silence the model's INFO profiling spam

from progress_model import config as cfg  # noqa: E402
from progress_model.parameters import Parameters  # noqa: E402
from plan_a_covert_proj import sim  # noqa: E402

_LOG10_FPH = np.log10(cfg.FLOP_PER_H100E_YEAR)


def _cmp(name, got, ref, mask=None):
    """Print element-wise divergence stats between `got` and `ref` over `mask`."""
    got = np.asarray(got, float)
    ref = np.asarray(ref, float)
    n = min(got.size, ref.size)
    got, ref = got[:n], ref[:n]
    m = np.isfinite(got) & np.isfinite(ref)
    if mask is not None:
        m &= np.asarray(mask, bool)[:n]
    if m.sum() == 0:
        print(f"  {name:28s}  (no comparable points)")
        return
    g, r = got[m], ref[m]
    denom = np.maximum(np.abs(r), 1e-12)
    rel = np.abs(g - r) / denom
    absd = np.abs(g - r)
    i = int(np.argmax(rel))
    print(f"  {name:28s}  n={m.sum():4d}  max_rel={rel.max():.3e}  med_rel={np.median(rel):.3e}"
          f"  max_abs={absd.max():.3e}  (@worst got={g[i]:.4g} ref={r[i]:.4g})")


def _series_at(data, attr, ts):
    arr = np.asarray(getattr(data, attr), float)
    return np.array([sim._log_interp(float(t), np.asarray(data.time, float), arr) for t in ts])


def main():
    params = Parameters()
    data = _load_default_series()

    ref = sim.run_default(params, data, [2017.0, 2100.0])
    ctx = ref["metrics_ctx"]
    off = ref["rebase_offset"]

    # Defaults for the covert knobs (mirror the API defaults loader).
    from api.index import app
    d = app.test_client().get("/api/plan-a/defaults").get_json()
    k = d["knobs"]

    t_pause = float(ref["ac_time"]) - float(k["years_before_ac"])
    setup = float(k["setup_period_years"])
    t_end = min(2100.0, float(ref["ac_time"]) + 25.0)
    target_year = t_pause + setup + float(k["legal_scaling_duration"])

    rec = sim.simulate_covert(
        params, data, t_pause,
        alignment_tax_progress_years=float(k["alignment_tax_progress_years"]),
        target_year=target_year, max_hw_rate=float(k["max_hw_rate"]),
        compute_mode=k["compute_mode"], frac_stolen=float(k["frac_stolen"]),
        total_h100e=(float(k["total_h100e"]) if k.get("total_h100e") is not None else None),
        total_h100e_frac=(float(k["total_h100e_frac"]) if k.get("total_h100e_frac") is not None else None),
        dormancy_years=float(k["dormancy_years"]), setup_period_years=setup,
        setup_sw_gain_progress_months=float(k["setup_sw_gain_progress_months"]),
        t_end=t_end, ref=ref, return_trajectories=True,
        legal_mode=k["legal_mode"], legal_strategy=k["legal_strategy"],
        max_sw_rate=(float(k["max_sw_rate"]) if k.get("max_sw_rate") is not None else None),
        solve_legal_compute=True,
    )
    if not rec.get("valid"):
        print("covert run invalid:", rec.get("reason"))
        return

    # ---------------- COVERT ----------------
    print("\n=== COVERT (effective inputs = make_covert_data; theft/boost in RHS) ===")
    cov_t = np.asarray(rec["covert_t"], float)
    cov_logE = np.asarray(rec["covert_logE_raw"], float)
    cov_RS = np.asarray(rec["covert_rs"], float)
    cdata = sim.make_covert_data(t_pause, t_end, float(rec["total_h100e_used"]), data)
    m = sim.project_metrics(cov_t, cov_logE, cov_RS, cdata, ctx)
    _cmp("effective_compute vs logE_raw", m["effective_compute"], cov_logE)
    _cmp("tsp vs covert_tsp", m["training_system_performance"], rec["covert_tsp"])
    _cmp("software_efficiency vs covert_sw", m["software_efficiency"], rec["covert_sw"])
    _cmp("automation vs covert_automation", m["automation_fractions"], rec["covert_automation"])
    # rate metrics: expect divergence (sim 're' is gradient of theft-laden RS)
    _cmp("research_efforts vs covert_re", m["research_efforts"], rec["covert_re"])
    sw_rate_adhoc = np.gradient(np.asarray(rec["covert_sw"], float), cov_t)
    _cmp("software_rate vs d(covert_sw)/dt", m["software_progress_rates"], sw_rate_adhoc)

    # ---------------- LEGAL DISCOVERED ----------------
    print("\n=== LEGAL DISCOVERED (effective inputs reconstructed from implied compute) ===")
    _legal_block(rec, data, ctx, off, prefix="legal", tsp_key="legal_tsp",
                 sw_key="legal_sw", re_key="legal_re", rs_key="legal_rs",
                 t_key="legal_t", logE_key="legal_logE_raw",
                 inf_key="legal_implied_inf", exp_key="legal_implied_exp",
                 fac_key="legal_compute_factor")

    # ---------------- LEGAL USED ----------------
    print("\n=== LEGAL USED (effective inputs reconstructed from implied compute) ===")
    _legal_block(rec, data, ctx, off, prefix="legal_used", tsp_key="legal_used_tsp",
                 sw_key="legal_used_sw", re_key="legal_used_re", rs_key="legal_used_rs",
                 t_key="legal_used_t", logE_key="legal_used_logE_raw",
                 inf_key="legal_used_implied_inf", exp_key="legal_used_implied_exp",
                 fac_key="legal_used_compute_factor")


def _legal_block(rec, data, ctx, off, *, prefix, tsp_key, sw_key, re_key, rs_key, t_key,
                 logE_key, inf_key, exp_key, fac_key):
    t = np.asarray(rec[t_key], float)
    logE = np.asarray(rec[logE_key], float)
    RS = np.asarray(rec[rs_key], float)
    tsp = np.asarray(rec[tsp_key], float)
    re_adhoc = np.asarray(rec[re_key], float)
    inf = np.asarray(rec[inf_key], float)
    exp = np.asarray(rec[exp_key], float)
    fac = np.asarray(rec[fac_key], float)

    # Active region: implied compute solved (factor finite & > 0). Setup=NaN, sit=0.
    active = np.isfinite(fac) & (fac > 0) & np.isfinite(inf) & (inf > 0) & np.isfinite(exp) & (exp > 0)
    if active.sum() < 2:
        print(f"  (no active compute-driven region: {active.sum()} pts)")
        return
    print(f"  active compute-driven pts: {active.sum()}/{t.size}"
          f"  (setup+sit excluded: {(~active).sum()})")

    # Reconstruct the legal's effective input series over the WHOLE trajectory:
    #   training_H100e from TSP (tsp = log10(H100e) + LOG10_FPH); inference/experiment from
    #   the implied compute; human labor held at the default world's L_HUMAN (as the solve did).
    tc_h100e = np.power(10.0, tsp - _LOG10_FPH)
    Lh = _series_at(data, "L_HUMAN", t)
    # Floor non-active inference/experiment to a small positive value so TimeSeriesData's
    # log-interp stays finite; comparisons are masked to the active region anyway.
    inf_f = np.where(active, inf, 1e-6)
    exp_f = np.where(active, exp, 1e-6)
    cdata = sim.TimeSeriesData(t, np.maximum(Lh, 1e-6), inf_f, exp_f, np.maximum(tc_h100e, 1e-6))

    m = sim.project_metrics(t, logE, RS, cdata, ctx)
    _cmp("effective_compute vs logE_raw", m["effective_compute"], logE)
    _cmp("tsp vs " + tsp_key, m["training_system_performance"], tsp, mask=active)
    _cmp("software_efficiency vs " + sw_key, m["software_efficiency"], rec[sw_key], mask=active)
    # research_efforts SHOULD match the back-solved leg_RE (that's what implied compute targets)
    _cmp("research_efforts vs " + re_key, m["research_efforts"], re_adhoc, mask=active)
    sw_rate_adhoc = np.gradient(np.asarray(rec[sw_key], float), t)
    _cmp("software_rate vs d(sw)/dt", m["software_progress_rates"], sw_rate_adhoc, mask=active)


def _load_default_series():
    from api.index import create_default_time_series
    return create_default_time_series()


if __name__ == "__main__":
    main()
