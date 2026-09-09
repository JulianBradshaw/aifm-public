#!/usr/bin/env python3
"""Validate the covert slowdown BOTEC and characterize the finite-theft-window correction.

Slowdown elasticity (vs the 1x freeze):  e = log10(time-to-TED-AI ratio) / log10(R),
where R = frontier-total-H100e / covert-H100e at the pause.

Theft-active BOTEC (steady state):   e_active(frac) = (r·ece·(1-frac) + 1) / (r+1)
No-theft / post-theft asymptote:     e_off          = (r·ece        + 1) / (r+1)
Full-theft floor:                    e_floor        =  1 / (r+1)

Panel (a): e vs frac_stolen — measured points vs the BOTEC curve.
Panel (b): e vs theft-window length (legal_scaling_duration) at fixed frac — the realized
elasticity relaxes from e_active (long window) toward e_off (short window). Overlays
a time-fraction blend  e ≈ e_off − (e_off − e_active)·w,  w = theft_time / race_time.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import ProgressModel, Parameters, load_time_series_data  # noqa: E402
from progress_model.pause_transform import freeze_and_reduce_time_series  # noqa: E402
from plan_a_covert_proj.sim import run_default, simulate_covert, _log_interp  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402

PAUSE = 2029.17
TR = [2017.0, 2400.0]   # long horizon: short theft windows leave the covert very slow


def _ms(res, n):
    m = (res or {}).get("milestones", {}).get(n)
    return float(m["time"]) if m and m.get("time") else None


def main():
    med = _get_median_parameter_dict(_load_config("plan_a_covert_proj/config/model_uncertain_thomas.yaml").get("parameters", {}))
    med.setdefault("horizon_extrapolation_type", "decaying doubling time")
    cov = _get_median_parameter_dict(_load_config("plan_a_covert_proj/config/covert_scenario_branch.yaml").get("covert_parameters", {}))
    base = load_time_series_data("inputs/input_data.csv")
    p = Parameters(**med)
    ref = run_default(p, base, TR)
    # Read the PREPARED params: run_default rescales r_software and experiment_compute_exponent
    # on its internal deep-copy (the raw `p` is never mutated), so reading p.* here would use
    # the pre-calibration seeds. Use ref's prepared values for the BOTEC.
    r, ece = ref["r_software"], ref["prepared_params"].experiment_compute_exponent

    # 1x-freeze baseline + compute reduction R at the pause
    tot = sum(_log_interp(PAUSE, base.time, getattr(base, c))
              for c in ("inference_H100e", "experiment_H100e", "training_H100e"))
    Rlog = np.log10(tot / cov["total_h100e"])
    frz = freeze_and_reduce_time_series(base, PAUSE)
    mf = ProgressModel(Parameters(**med), frz); mf.compute_progress_trajectory(TR, 0.0)
    ted_f = _ms(mf.results, "TED-AI")

    e_off = (r * ece + 1) / (r + 1)
    e_floor = 1.0 / (r + 1)

    def run(frac, lead):
        rec = simulate_covert(
            p, base, PAUSE, alignment_tax_ooms=cov["alignment_tax_ooms"],
            target_year=PAUSE + lead, max_hw_rate=cov["max_hw_rate"],
            compute_mode="absolute", total_h100e=cov["total_h100e"], frac_stolen=frac,
            dormancy_years=cov.get("dormancy_years", 0.0),
            setup_period_years=cov.get("setup_period_years", 0.0),
            setup_sw_gain_ooms=cov.get("setup_sw_gain_ooms", 0.0),
            t_end=TR[1], ref=ref)
        c = rec["covert_ttt"]
        if c is None:
            return None, None
        return c, np.log10((c - PAUSE) / (ted_f - PAUSE)) / Rlog

    # ---- (a) elasticity vs frac_stolen ----
    fracs = np.array([0.0, 0.2, 0.4, 0.6, 0.78, 0.9, 0.97, 1.0])
    e_meas_a = [run(f, cov["legal_scaling_duration"])[1] for f in fracs]
    e_botec_a = (r * ece * (1 - fracs) + 1) / (r + 1)

    # ---- (b) elasticity vs theft-window length (target_lead) at frac=median ----
    fixed_frac = cov["frac_stolen"]
    e_active = (r * ece * (1 - fixed_frac) + 1) / (r + 1)
    leads = np.array([0.5, 1.0, 2.0, 3.5, 5.5, 8.0, 12.0, 20.0])
    e_meas_b, w_b = [], []
    for L in leads:
        c, e = run(fixed_frac, L)
        e_meas_b.append(e if e is not None else np.nan)
        w_b.append((L / (c - PAUSE)) if (c and c > PAUSE) else np.nan)  # theft-time fraction
    e_meas_b = np.array(e_meas_b); w_b = np.array(w_b)
    e_blend = e_off - (e_off - e_active) * w_b   # time-fraction blend prediction

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    axa.plot(fracs, e_botec_a, "-", color="#2e8b57", lw=2.2,
             label=r"BOTEC  $(r\,ece\,(1-f)+1)/(r+1)$")
    axa.plot(fracs, e_meas_a, "o", color="#a23b3b", ms=7, label="measured (covert sim)")
    axa.axhline(e_off, color="#999", ls="--", lw=1.0)
    axa.text(0.02, e_off + 0.005, f"no theft = (r·ece+1)/(r+1) = {e_off:.3f}", fontsize=8, color="#666")
    axa.axhline(e_floor, color="#999", ls=":", lw=1.0)
    axa.text(0.02, e_floor + 0.005, f"full-theft floor = 1/(r+1) = {e_floor:.3f}", fontsize=8, color="#666")
    axa.set_xlabel("frac_stolen"); axa.set_ylabel("slowdown elasticity  log₁₀(N)/log₁₀(R)")
    axa.set_title(f"(a) BOTEC vs measured across theft fraction\n(pause {PAUSE}, R={Rlog:.2f} OOM, r={r}, ece={ece:.3f})", fontsize=10)
    axa.legend(fontsize=9, loc="upper right"); axa.grid(alpha=0.3)

    axb.plot(leads, e_meas_b, "o-", color="#a23b3b", ms=6, lw=1.8, label="measured (frac=0.78)")
    axb.axhline(e_off, color="#999", ls="--", lw=1.1)
    axb.text(leads[-1], e_off + 0.006, f"no-theft ceiling = {e_off:.3f}", fontsize=8, color="#666", ha="right")
    axb.axhline(e_active, color="#3a6cb0", ls="-.", lw=1.1)
    axb.text(leads[-1], e_active + 0.006, f"theft-active asymptote = {e_active:.3f}", fontsize=8, color="#3a6cb0", ha="right")
    axb.axhline(e_floor, color="#999", ls=":", lw=1.0)
    axb.text(leads[-1], e_floor + 0.006, f"full-theft floor = {e_floor:.3f}", fontsize=8, color="#666", ha="right")
    axb.set_xlabel("deal horizon = legal_scaling_duration (legal sits at TED-AI+tax → theft ends)")
    axb.set_ylabel("realized slowdown elasticity")
    axb.set_title("(b) finite-window: shorter deal horizon → legal ramps harder →\nmore stealable software → lower elasticity (more covert speedup)", fontsize=9)
    axb.legend(fontsize=9, loc="lower right"); axb.grid(alpha=0.3)

    fig.suptitle("Covert slowdown elasticity: BOTEC validation + finite-theft-window correction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPO_ROOT / "outputs" / "covert_elasticity_botec.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)

    print(f"e_off={e_off:.3f}  e_active(0.78)={e_active:.3f}  e_floor={e_floor:.3f}")
    print("(a) frac:    ", "  ".join(f"{f:.2f}" for f in fracs))
    print("    measured:", "  ".join(f"{e:.3f}" for e in e_meas_a))
    print("    botec:   ", "  ".join(f"{e:.3f}" for e in e_botec_a))
    print("(b) lead:    ", "  ".join(f"{L:.1f}" for L in leads))
    print("    measured:", "  ".join(f"{e:.3f}" for e in e_meas_b))
    print("    w(theft):", "  ".join(f"{w:.2f}" for w in w_b))
    print("    blend:   ", "  ".join(f"{e:.3f}" for e in e_blend))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
