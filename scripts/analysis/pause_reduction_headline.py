#!/usr/bin/env python3
"""
Headline chart for the pause-reduction post: rate of AI progress at median
params, freeze-only vs freeze+reduce, each trajectory ending at its own ASI
crossing. Single clean panel intended as the post's lead image.

Runs on aifm-runpod (python3.13).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as cfg  # noqa: E402
from progress_model import ProgressModel, Parameters, load_time_series_data  # noqa: E402
from progress_model.pause_transform import freeze_and_reduce_time_series  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config, _get_median_parameter_dict  # noqa: E402

# palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
GRAY_SERIES = "#898781"


def _ms_time(results, name):
    m = (results or {}).get("milestones", {}).get(name)
    if not m or m.get("time") is None:
        return None
    t = float(m["time"])
    return t if np.isfinite(t) else None


def _run(params, data, tr, ip):
    model = ProgressModel(Parameters(**params), data)
    model.compute_progress_trajectory(tr, ip)
    return model.results


def main():
    ap = argparse.ArgumentParser(description="Headline chart: progress rate, freeze vs freeze+reduce")
    ap.add_argument("--config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml")
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--pause", type=float, default=None,
                    help="Pause year (default: the AC milestone at median params)")
    ap.add_argument("--lead", type=float, default=1.5, help="Years shown before the pause")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pause_reduction" / "headline"))
    args = ap.parse_args()

    ucfg = _load_config(args.config) or {}
    med = _get_median_parameter_dict(ucfg.get("parameters", {}))
    med.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    med.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)
    input_path = str(ucfg.get("input_data", "inputs/input_data.csv"))
    if not Path(input_path).is_absolute():
        input_path = str((REPO_ROOT / input_path).resolve())
    base = load_time_series_data(input_path)
    tr = ucfg.get("time_range", [2017.0, 2200.0]); tr = [float(tr[0]), float(tr[1])]
    ip = float(ucfg.get("initial_progress", 0.0))

    base_res = _run(med, base, tr, ip)
    pause = args.pause if args.pause is not None else _ms_time(base_res, "AC")
    R = args.reduction

    frozen = freeze_and_reduce_time_series(base, pause)
    reduced = freeze_and_reduce_time_series(
        base, pause, training_reduction=R, experiment_reduction=R,
        inference_reduction=R, labor_reduction=R)
    fres = _run(med, frozen, tr, ip)
    rres = _run(med, reduced, tr, ip)
    asi_f = _ms_time(fres, "ASI")
    asi_r = _ms_time(rres, "ASI")
    if asi_f is None or asi_r is None:
        sys.exit("ASI not reached in one of the variants; pick another pause year")
    N = (asi_r - pause) / (asi_f - pause)

    def rate_until(res, t_end):
        t = np.asarray(res["times"], dtype=float)
        y = np.asarray(res["progress_rates"], dtype=float)
        sel = (t >= pause - args.lead) & (t <= t_end)
        return t[sel], y[sel]

    tf, yf = rate_until(fres, asi_f)
    trd, yrd = rate_until(rres, asi_r)
    # blue line only from the pause on; before it the two variants coincide.
    # Prepend the pre-cut rate at the pause instant so the ÷R drop is drawn.
    post = trd >= pause
    trd, yrd = trd[post], yrd[post]
    rate_at_pause = float(np.interp(pause, tf, yf))
    trd = np.concatenate([[pause], trd])
    yrd = np.concatenate([[rate_at_pause], yrd])

    fig, ax = plt.subplots(figsize=(9.6, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(tf, yf, color=GRAY_SERIES, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot(trd, yrd, color=BLUE, lw=2.2, solid_capstyle="round", zorder=4)
    # ASI end dots
    ax.plot([asi_f], [yf[-1]], "o", ms=7, color=GRAY_SERIES, mec=SURFACE, mew=1.5, zorder=5)
    ax.plot([asi_r], [yrd[-1]], "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=5)

    ax.set_yscale("log")
    lo = min(yrd[yrd > 0].min(), yf[yf > 0].min())
    hi = max(yf.max(), yrd.max())
    ax.set_ylim(lo / 1.6, hi * 4.0)
    x0, x1 = pause - args.lead, asi_r + 0.06 * (asi_r - pause)
    ax.set_xlim(x0, x1)

    # pause marker
    ax.axvline(pause, color=BASELINE, ls=(0, (2, 3)), lw=1.1, zorder=2)
    ax.text(pause, ax.get_ylim()[1] * 0.82, "  compute growth pauses",
            color=INK_2, fontsize=10, ha="left", va="top")

    # "÷10" label beside the vertical drop segment
    ax.annotate(f"÷{R:g}", (pause, np.sqrt(rate_at_pause * yrd[1])), xytext=(7, 0),
                textcoords="offset points", color=BLUE, fontsize=11, ha="left",
                va="center", fontweight="bold")

    # direct labels, anchored to points on each line
    i_f = int(np.argmin(np.abs(yf - 3.5 * rate_at_pause)))
    ax.annotate("compute frozen", (tf[i_f], yf[i_f]), xytext=(-10, 0),
                textcoords="offset points", color=INK_2, fontsize=11, ha="right",
                va="center", fontweight="bold")
    i_r = int(np.searchsorted(trd, pause + 0.40 * (asi_r - pause)))
    ax.annotate(f"frozen, then cut {R:g}×", (trd[i_r], yrd[i_r]), xytext=(0, -24),
                textcoords="offset points", color=BLUE, fontsize=11, ha="center",
                fontweight="bold")

    # ASI labels
    ax.annotate(f"ASI\n+{asi_f - pause:.1f} yr", (asi_f, yf[-1]), xytext=(10, 0),
                textcoords="offset points", color=INK_2, fontsize=10, ha="left", va="center")
    ax.annotate(f"ASI\n+{asi_r - pause:.1f} yr", (asi_r, yrd[-1]), xytext=(10, 0),
                textcoords="offset points", color=BLUE, fontsize=10, ha="left", va="center")

    # N annotation: double arrow between the two ASI times near the top
    y_arrow = hi * 1.9
    ax.annotate("", xy=(asi_r, y_arrow), xytext=(asi_f, y_arrow),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.2))
    ax.text(0.5 * (asi_f + asi_r), y_arrow * 1.15,
            f"takeoff takes {N:.1f}× longer", color=INK, fontsize=11.5,
            ha="center", va="bottom", fontweight="bold")

    # chrome
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_ylabel("rate of AI progress  (OOMs of effective compute / year)",
                  color=INK_2, fontsize=10.5)
    ax.set_title(f"After a {R:g}× R&D-compute cut, AI progress drops {R:g}×, "
                 f"then recovers — partly",
                 color=INK, fontsize=14, loc="left", pad=16, fontweight="bold")
    ax.text(0, 1.015, "AI Futures Model, median parameters; compute frozen at the "
            "automated-coder milestone vs frozen and cut", transform=ax.transAxes,
            color=INK_2, fontsize=10, va="bottom")

    fig.tight_layout()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "headline.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}  pause={pause:.2f} asi_frz={asi_f:.2f} asi_red={asi_r:.2f} N={N:.2f}")


if __name__ == "__main__":
    main()
