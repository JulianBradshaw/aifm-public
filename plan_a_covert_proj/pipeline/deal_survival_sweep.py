#!/usr/bin/env python3
"""Sweep deal_breakdown.min_survival_years ("the legal project's epistemic state N years
in") and package the results: one pipeline run per value (hash-cached — rerunning the
sweep reuses finished datasets), then labeled GIFs cycling the capability_cdf2d dither
figures and combined monthly exceedance CSVs across the sweep.

    python -m plan_a_covert_proj.pipeline.deal_survival_sweep --values 1,2,3,4,5,6,7,8,9 \
        --num 100 --model-config plan_a_covert_proj/config/model_scenario_branch.yaml

Outputs land in plan_a_covert_proj/outputs/sweeps/<out-name>/:
    capability_cdf2d_pct_asi.gif / capability_cdf2d_uplift.gif   (one frame per value)
    capability_cdf2d_pct_asi.csv / capability_cdf2d_uplift.csv   (min_survival column)
    manifest.json                                                (value -> runid map)
Each value's full run (all figures, dataset) lives in its normal outputs/runs/<runid>/
dir and the run viewer; this directory is just the cross-run packaging.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COVERT = ROOT / "plan_a_covert_proj"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_one(value, args) -> str:
    """Run the pipeline for one min-survival value (dataset is hash-cached); return runid."""
    cmd = [sys.executable, "-u", "-m", "plan_a_covert_proj.pipeline.run",
           "--num", str(args.num),
           "--model-config", args.model_config,
           "--covert-config", args.covert_config,
           "--min-survival-years", f"{value:g}",
           "--name", f"{args.name_prefix}-min{value:g}-{args.num}",
           "--workers", str(args.workers)]
    if args.covert_exists_prior is not None:
        cmd += ["--covert-exists-prior", f"{args.covert_exists_prior:g}"]
    if args.local:
        cmd.append("--local")
    if args.force:
        cmd.append("--force")
    print(f"--- min_survival_years = {value:g} ---", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    runid = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = re.match(r"runid (\w+)", line)
        if m:
            runid = m.group(1)
    if proc.wait() != 0:
        raise RuntimeError(f"pipeline run failed for min_survival_years={value:g}")
    if runid is None:
        raise RuntimeError(f"could not parse runid from run output (min_survival_years={value:g})")
    return runid


def _gif(values_runids, fig_name, out_path, duration_ms):
    """Cycle each run's dither figure with a 'conditioned on' banner. Frames keep the
    per-run captions; the banner carries the swept value."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    font = ImageFont.truetype(
        str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono-Bold.ttf"), 30)
    frames = []
    for value, runid in values_runids:
        p = COVERT / "outputs" / "runs" / runid / "figures" / fig_name
        img = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(img)
        label = f"conditioned on: deal survived >= {value:g} years"
        x, y = 120, 92
        bbox = d.textbbox((x, y), label, font=font)
        d.rectangle((bbox[0] - 12, bbox[1] - 8, bbox[2] + 12, bbox[3] + 8),
                    fill="#fffef8", outline="#5c1a1a", width=2)
        d.text((x, y), label, font=font, fill="#5c1a1a")
        frames.append(img)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return out_path.name


def _combined_csv(values_runids, csv_name, out_path):
    """Concatenate the per-run monthly exceedance CSVs with a min_survival column."""
    rows = []
    for value, runid in values_runids:
        lines = (COVERT / "outputs" / "runs" / runid / "figures" / csv_name) \
            .read_text().splitlines()
        if not rows:
            rows.append("min_survival_years," + lines[0])
        rows += [f"{value:g},{ln}" for ln in lines[1:]]
    out_path.write_text("\n".join(rows) + "\n")
    return out_path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True,
                    help="comma-separated min_survival_years values, e.g. 1,2,3,4,5,6,7,8,9")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--model-config", default="plan_a_covert_proj/config/model_scenario_branch.yaml")
    ap.add_argument("--covert-config",
                    default="plan_a_covert_proj/config/covert_scenario_branch_deal_breakdown.yaml")
    ap.add_argument("--name-prefix", default="deal-survival-sweep")
    ap.add_argument("--covert-exists-prior", type=float, default=None,
                    help="prior P(a covert project exists); enables the figures' Bayesian "
                         "update at each value's pause+N vantage")
    ap.add_argument("--out-name", default=None,
                    help="sweep output dir name (default derived from values + num)")
    ap.add_argument("--local", action="store_true", help="build datasets locally (default: remote)")
    ap.add_argument("--force", action="store_true", help="rebuild datasets even if cached")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--gif-ms", type=int, default=1200, help="GIF frame duration (ms)")
    args = ap.parse_args()

    values = [float(v) for v in args.values.split(",") if v.strip() != ""]
    if not values:
        sys.exit("--values is empty")
    values_runids = [(v, _run_one(v, args)) for v in values]

    out_name = args.out_name or ("deal_survival_" + "_".join(f"{v:g}" for v in values)
                                 + f"_x{args.num}"
                                 + (f"_prior{args.covert_exists_prior:g}"
                                    if args.covert_exists_prior is not None else ""))
    out_dir = COVERT / "outputs" / "sweeps" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for fig in ("capability_cdf2d_pct_asi.png", "capability_cdf2d_uplift.png",
                "capability_cdf2d_uplift_lin100.png"):
        built.append(_gif(values_runids, fig, out_dir / fig.replace(".png", ".gif"), args.gif_ms))
    for csv in ("capability_cdf2d_pct_asi.csv", "capability_cdf2d_uplift.csv"):
        built.append(_combined_csv(values_runids, csv, out_dir / csv))
    (out_dir / "manifest.json").write_text(json.dumps({
        "values": {f"{v:g}": rid for v, rid in values_runids},
        "num": args.num, "model_config": args.model_config,
        "covert_config": args.covert_config, "files": built}, indent=1))
    print(f"sweep -> {out_dir.relative_to(ROOT)}: {', '.join(built)}")


if __name__ == "__main__":
    main()
