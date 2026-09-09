"""P(covert reaches TED-AI within N years of the legal project) vs legal
scaling duration, for N = 0, 5, 10 — with and without detection.

The legal project's post-pause scaling duration (``legal_scaling_duration``,
sampled U[2,8] in the stage-1 run) sets the year it targets TED-AI:
``target_year = pause_end + setup + scaling_duration``. Each Monte-Carlo draw
therefore carries its own scaling duration, so we bin the draws by it and, per
bin, report the share of draws in which the covert project reaches TED-AI no
later than N years after the legal project does:

    within N  <=>  covert_ttt <= legal_ttt + N  <=>  lead_years >= -N

(``lead_years = legal_ttt - covert_ttt``; a draw where covert never reaches
TED-AI has lead_years = None and counts as *not* within N for every N.)

Two layers, mirroring figures.characterize_wins:
  * physics only    — P(within N), ignoring whether the covert project is caught.
  * detection-aware — P(within N AND undetected when it reaches TED-AI), using the
    survival-weighted estimator  E[ 1{lead>=-N} * survival_at_goal ]  where
    ``survival_at_goal`` is the per-draw probability of still being undetected at
    the moment the covert project reaches TED-AI.

Pure reader: touches only the stored dataset, never the physics.

Usage:
    python -m plan_a_covert_proj.pipeline.covert_within_vs_scaling_duration [RUN_ID]
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
        apply_ai2030_style, AI2030_NAVY, AI2030_AMBER, AI2030_OLIVE, AI2030_MUTED)
    apply_ai2030_style()
except Exception:  # noqa: BLE001
    AI2030_NAVY, AI2030_AMBER, AI2030_OLIVE, AI2030_MUTED = \
        "#1e3a5f", "#c87a1f", "#3f6634", "#6b7280"

# The N-values (years) and their series colours, strict -> loose.
N_SERIES = [(0, AI2030_NAVY), (5, AI2030_AMBER), (10, AI2030_OLIVE)]
BIN_WIDTH = 1.0          # years of scaling duration per bin (figure default)


def _latest_run_with_data() -> Path:
    runs = REPO_ROOT / "plan_a_covert_proj" / "outputs" / "runs"
    cands = [d for d in runs.iterdir() if (d / "dataset.json").exists()]
    if not cands:
        raise SystemExit("no runs with dataset.json found")
    return max(cands, key=lambda d: (d / "dataset.json").stat().st_mtime)


def bin_table(ds, width=BIN_WIDTH):
    """Per-bin point estimates, shared by the figure and the sheet export so
    both always show identical numbers for a given bin width.

    Returns a list of dicts (one per non-empty scaling-duration bin)::

        {"x": bin_centre, "n": draws,
         "phys": {N: P(within N)},                 # physics-only race
         "det":  {N: E[within N · survival_at_goal]},   # detection-aware
         "det_se": {N: std-error of the det estimate}}  # for the figure band

    for N in the N_SERIES years (0, 5, 10). Returns None if the run lacks the
    fields this view needs.
    """
    R = ds.get("records") or []
    if not R or "survival_at_goal" not in R[0] \
            or "legal_scaling_duration" not in (R[0].get("covert_params") or {}):
        return None
    sd = np.array([r["covert_params"]["legal_scaling_duration"] for r in R], float)
    lead = np.array([r["lead_years"] if r.get("lead_years") is not None else np.nan
                     for r in R], float)
    # P(undetected at the moment covert reaches TED-AI); 0 if it never gets there.
    s_goal = np.array([(r.get("survival_at_goal") or 0.0) for r in R], float)

    lo_edge = np.floor(sd.min() / width) * width
    hi_edge = np.ceil(sd.max() / width) * width
    edges = np.arange(lo_edge, hi_edge + width * 0.5, width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # last bin is closed on the right so the max value lands somewhere
    idx = np.clip(np.digitize(sd, edges[1:-1], right=False), 0, len(centres) - 1)

    Ns = [N for N, _ in N_SERIES]
    rows = []
    for b in range(len(centres)):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        phys, det, det_se = {}, {}, {}
        for N in Ns:
            within = lead[sel] >= -N          # NaN (covert never reaches TED) -> False
            phys[N] = float(within.mean())
            w = within * s_goal[sel]          # survival-weighted within-N event
            det[N] = float(w.mean())
            det_se[N] = float(np.sqrt(w.var(ddof=0) / n)) if n > 1 else 0.0
        rows.append({"x": float(centres[b]), "n": n,
                     "phys": phys, "det": det, "det_se": det_se})
    return rows


def render(ds, outdir):
    """FIGURES-registry entry point: pure function of the dataset.

    ``outdir`` is the run's ``figures/`` dir; returns the PNG filename (or None
    if the run lacks the fields this view needs, so the pipeline can skip it).
    """
    tbl = bin_table(ds, BIN_WIDTH)
    if not tbl:
        return None
    outdir = Path(outdir)
    run_label = outdir.parent.name
    n_draws = len(ds.get("records") or [])
    xs = np.array([r["x"] for r in tbl])

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for N, colour in N_SERIES:
        det = np.array([r["det"][N] for r in tbl])
        se = np.array([r["det_se"][N] for r in tbl])
        det_lo = np.clip(det - se, 0.0, 1.0)
        det_hi = np.clip(det + se, 0.0, 1.0)
        phys = np.array([r["phys"][N] for r in tbl])
        lbl = f"within {N} yr" + (" (covert reaches TED-AI first)" if N == 0 else "")
        # physics-only reference: faint dotted, no markers
        ax.plot(xs, phys, ":", color=colour, lw=1.4, alpha=.55, zorder=2)
        # detection-aware: solid with band + markers (the primary series)
        ax.fill_between(xs, det_lo, det_hi, color=colour, alpha=.12, lw=0, zorder=1)
        ax.plot(xs, det, "-o", color=colour, lw=2.2, ms=6, zorder=3, label=lbl)

    # per-bin sample sizes along the bottom
    for r in tbl:
        ax.annotate(f"n={r['n']}", (r["x"], 0.012), ha="center", va="bottom",
                    fontsize=7, color=AI2030_MUTED)

    # style-key entry distinguishing the two layers
    from matplotlib.lines import Line2D
    layer_key = [
        Line2D([0], [0], color=AI2030_MUTED, lw=2.2, marker="o", ms=6,
               label="detection-aware (× P undetected at TED-AI)"),
        Line2D([0], [0], color=AI2030_MUTED, lw=1.4, ls=":",
               label="physics only (ignores detection)"),
    ]

    ax.set_xlabel("legal project's scaling duration (years)")
    ax.set_ylabel("P(covert reaches TED-AI within N years of legal)")
    ax.set_xlim(xs[0] - BIN_WIDTH / 2, xs[-1] + BIN_WIDTH / 2)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=.25)
    leg1 = ax.legend(title="N =", loc="upper right", framealpha=.9)
    ax.add_artist(leg1)
    ax.legend(handles=layer_key, loc="center right", fontsize=8, framealpha=.9)
    ax.set_title(f"Covert vs legal race to TED-AI, by legal scaling duration "
                 f"— {n_draws} draws", fontsize=11)
    fig.text(0.5, 0.955, "(Rough draft figure by Claude)", ha="center",
             fontsize=8, style="italic", color=AI2030_MUTED)
    fig.suptitle(f"run {run_label}", fontsize=9, y=0.995, color=AI2030_MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = outdir / "covert_within_vs_scaling_duration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def main(run_id: str | None = None):
    runs = REPO_ROOT / "plan_a_covert_proj" / "outputs" / "runs"
    run_dir = (runs / run_id) if run_id else _latest_run_with_data()
    ds = json.loads((run_dir / "dataset.json").read_text())
    if not ds.get("records"):
        raise SystemExit("dataset has no records")
    name = render(ds, run_dir / "figures")
    print(f"wrote {run_dir / 'figures' / name}")
    return run_dir / "figures" / name


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
