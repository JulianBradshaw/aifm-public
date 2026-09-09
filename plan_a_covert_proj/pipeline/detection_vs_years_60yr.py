"""Standalone reproduction of the LEFT panel of `detection_winnow.png`
("Detection vs years run"), extended to a 60-year horizon.

Pure reader: touches only the cheap detection frailty model for the display
curve, never the physics. Mirrors figures.detection_winnow's left-axis logic
(median covert size curve + p10-p90 size band + representative silver bullet).

Usage:
    python -m plan_a_covert_proj.pipeline.detection_vs_years_60yr [RUN_ID]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from scripts.plotting_utils.style import (
        apply_ai2030_style, AI2030_AMBER, AI2030_MUTED)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_AMBER, AI2030_MUTED = "#c8862b", "#777"

from plan_a_covert_proj import detection  # noqa: E402

HORIZON_YEARS = 60.0


def _latest_run_with_data() -> Path:
    runs = REPO_ROOT / "plan_a_covert_proj" / "outputs" / "runs"
    cands = [d for d in runs.iterdir() if (d / "dataset.json").exists()]
    if not cands:
        raise SystemExit("no runs with dataset.json found")
    return max(cands, key=lambda d: (d / "dataset.json").stat().st_mtime)


def main(run_id: str | None = None):
    runs = REPO_ROOT / "plan_a_covert_proj" / "outputs" / "runs"
    run_dir = (runs / run_id) if run_id else _latest_run_with_data()
    ds = json.loads((run_dir / "dataset.json").read_text())
    R = ds["records"]
    if not R:
        raise SystemExit("dataset has no records")

    cfg = ds.get("config") or {}
    fits = detection.normalize_fits(cfg.get("detection_frailty_fit"))

    # Representative silver-bullet anchors (median across rollouts reaching TED-AI),
    # exactly as figures.detection_winnow builds its single display curve.
    sb = None
    _sbc = cfg.get("detection_silver_bullet")
    if _sbc:
        _t0s, _t1s = [], []
        for r_ in R:
            ut, py = r_.get("legal_used_ttt"), r_.get("pause_year")
            sp = r_.get("covert_params", {}).get("setup_period_years")
            if ut is None or py is None or sp is None:
                continue
            t_used = ut - py
            _t0s.append(0.5 * (sp + t_used))
            _t1s.append(t_used + _sbc["delta_years"])
        if _t0s:
            sb = detection.build_silver_bullet(
                _sbc["p_mid_pause_to_ted"], float(np.median(_t0s)),
                _sbc["p_used_ted_plus_d"], float(np.median(_t1s)), _sbc["ceiling"])

    r_med = float(np.median([r_.get("detection_r", 0.0) for r_ in R]))
    sizes = np.array([r_["covert_size_mw"] for r_ in R
                      if r_.get("covert_size_mw") is not None], float)
    site_vals = [r_.get("covert_params", {}).get("optimal_site_mw") for r_ in R]
    site_vals = [s for s in site_vals if s is not None]
    site_med = float(np.median(site_vals)) if site_vals else None

    tg = np.linspace(0, HORIZON_YEARS, 400)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    smed = None
    if sizes.size:
        smed = float(np.median(sizes))
        s10, s90 = np.percentile(sizes, [10, 90])
        dp = lambda sz, t: detection.detection_probability(  # noqa: E731
            sz, t, r_med, site_mw=site_med, fits=fits, silver_bullet=sb)
        ax.fill_between(tg, dp(s10, tg), dp(s90, tg), color=AI2030_AMBER, alpha=.18, lw=0)
        ax.plot(tg, dp(smed, tg), color=AI2030_AMBER, lw=2.4)
        for yr in (3, 5, 10, 20, 40, 60):
            ax.plot(yr, float(dp(smed, yr)), "o", color=AI2030_MUTED, ms=4)
        ax.set_title(f"Detection vs years run · median {smed:.0f} MW "
                     f"(band p10–p90 size)", fontsize=11)
    ax.set_xlabel("years the covert project has run")
    ax.set_ylabel("probability detected")
    ax.set_xlim(0, HORIZON_YEARS)
    ax.set_ylim(0, 1)
    fig.text(0.5, 0.945, "(Rough draft figure by Claude)", ha="center",
             fontsize=8, style="italic", color=AI2030_MUTED)
    fig.suptitle(f"Covert detection over a {int(HORIZON_YEARS)}-year horizon "
                 f"— run {run_dir.name}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = run_dir / "figures" / f"detection_vs_years_{int(HORIZON_YEARS)}yr.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
