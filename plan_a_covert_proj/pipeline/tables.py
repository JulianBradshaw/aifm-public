#!/usr/bin/env python3
"""Generate the covert-compute-acquisition tables from a run dataset.

Reproduces the quantitative tables in the ai-2030 covert-AI-projects supplement §2 / §2.3
Method #1 (bottom-up) directly from a Monte-Carlo run, so the numbers stay in sync with the
model instead of being hand-maintained. Emits two files into the run directory:

  tables_compute_acquisition.md    GitHub-flavoured markdown (for reading / pasting to a repo)
  tables_compute_acquisition.html  the supplement's <table> block format (paste into the doc)

Three tables are produced, all from the per-rollout bottom-up fields the pipeline stores
(bottom_up_world_frac, bottom_up_{decommission,domestic,smuggled}_frac, total_h100e_used):

  1. Headline — covert compute acquired: fraction of world / H100e / IT cost / vs leading lab
     (median + 80% CI). This is the BOTTOM-UP (Method #1) result, not the supplement's
     all-things-considered headline.
  2. Acquisition routes — per source (smuggled flow / false decommissioning / concealed
     domestic): P(feasible), conditional median, conditional 80% CI.
  3. Aggregation methods — add-medians / feasibility-weighted-sum / Monte-Carlo, mirroring the
     supplement's §2.3 aggregation table.

Run standalone against an existing run:

    python -m plan_a_covert_proj.pipeline.tables <runid>      # writes into that run's dir

Only bottom_up_estimate runs produce output; other compute modes are skipped (returns []).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Reweight-aware weighting helpers — shared with figures.py so tables and figures agree
# (identical importance weights, identical weighted-quantile convention).
from plan_a_covert_proj.pipeline.figures import _wts, _wmean, _wpct, _rw_note

COVERT = Path(__file__).resolve().parents[1]
ROOT = COVERT.parent                       # repo root (inputs/input_data.csv lives under it)
RUNS = COVERT / "outputs" / "runs"

# ---- external anchor (NOT a model output) --------------------------------------------------
# Needed only for the headline table's "IT cost" row; the bottom-up model does not price
# compute. From the ai-2030 covert-AI-projects supplement §2.3 Method #2 — surfaced here
# (named, documented) rather than buried; edit if the supplement's anchor changes.
USD_PER_H100E = 8_000          # $/H100e China would plausibly pay (supplement §2.3 Method #2 new-server anchor)
# Fallback leading-lab R&D compute (H100e) for the "vs leading lab" row, used ONLY if the run's
# input-data CSV can't be located. Normally this is computed per-rollout from the CSV.
LEADING_LAB_RD_FALLBACK = 22e6  # supplement §2 headline anchor

_EPS = 1e-12


def _resolve_input_csv(ds):
    """Locate the run's input-data CSV locally. The dataset stores the path used at build time,
    which for a remote (aifm-runpod) build is an absolute remote path that doesn't exist here —
    so fall back to the repo-root basename (how the pipeline resolves input_data)."""
    p = (ds.get("config") or {}).get("input_data")
    if p:
        pp = Path(p)
        if pp.exists():
            return pp
        cand = ROOT / pp.name
        if cand.exists():
            return cand
    cand = ROOT / "inputs" / "input_data.csv"
    return cand if cand.exists() else None


def _leading_lab_rd(ds, times, w):
    """Per-rollout leading-lab R&D compute (H100e) = training + internal inference + experiment
    H100e (the supplement's definition), log-interpolated from the input CSV at each rollout's
    given time (the compute declaration). Returns (array aligned to `times`, weighted-median,
    csv path). Falls back to a constant anchor array when the CSV can't be found."""
    fp = _resolve_input_csv(ds)
    times = np.asarray(times, float)
    if fp is None:
        arr = np.full(times.shape, LEADING_LAB_RD_FALLBACK)
        return arr, float(LEADING_LAB_RD_FALLBACK), None
    t, rd = [], []
    with open(fp, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row["time"]))
                rd.append(float(row["inference_H100e"]) + float(row["experiment_H100e"])
                          + float(row["training_H100e"]))
            except (KeyError, ValueError):
                continue
    t, rd = np.asarray(t, float), np.asarray(rd, float)
    if t.size == 0:
        arr = np.full(pauses.shape, LEADING_LAB_RD_FALLBACK)
        return arr, float(LEADING_LAB_RD_FALLBACK), None
    arr = 10.0 ** np.interp(times, t, np.log10(rd))    # geometric (log) interpolation
    med = float(np.atleast_1d(_wpct(arr, w, [50]))[0])
    return arr, med, fp


def _fmt_pct(x, dp=2):
    return f"{x * 100:.{dp}f}%"


def _fmt_h100e(x):
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _fmt_usd(x):
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    if x >= 1e6:
        return f"${x / 1e6:.0f}M"
    return f"${x:,.0f}"


def _source_stats(a, w):
    """(P(feasible), conditional [p10,p50,p90], marginal [p10,p50,p90]) for one source's
    per-rollout fraction array. 'Feasible' = the source contributed a positive fraction on
    that rollout (its feasibility Bernoulli fired); conditional stats are taken over exactly
    those rollouts, matching the supplement's conditional-on-feasible medians/ranges."""
    a = np.asarray(a, float)
    p_feas = _wmean((a > _EPS).astype(float), w)
    m = a > _EPS
    if m.any():
        wc = None if w is None else np.asarray(w, float)[m]
        cond = np.atleast_1d(_wpct(a[m], wc, [10, 50, 90]))
    else:
        cond = np.zeros(3)
    marg = np.atleast_1d(_wpct(a, w, [10, 50, 90]))
    return float(p_feas), cond, marg


def _md_table(rows):
    hdr = rows[0]
    out = ["| " + " | ".join(hdr) + " |",
           "| " + " | ".join("---" for _ in hdr) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _html_table(rows, header=True):
    parts = ["<table><tbody>"]
    for i, row in enumerate(rows):
        parts.append("<tr>")
        for cell in row:
            c = f"<strong>{cell}</strong>" if (header and i == 0) else cell
            parts.append(f'<td colspan="1" rowspan="1"><p>{c}</p></td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _build_tables(ds):
    """Return (title, [(heading, note, rows), ...]) or None if this isn't a bottom-up run."""
    R = ds.get("records") or []
    w = _wts(ds)
    mask = np.array([r.get("bottom_up_world_frac") is not None for r in R])
    keep = [r for r in R if r.get("bottom_up_world_frac") is not None]
    if not keep:
        return None
    kw = None if w is None else np.asarray(w, float)[mask]
    col = lambda k: np.array([float(r.get(k) or 0.0) for r in keep], float)

    tot = col("bottom_up_world_frac")
    h100e = col("total_h100e_used")
    # Compute declaration time (books close): pause - declaration_to_pause_years. The covert pool
    # is sized off the world stock at the declaration, so the leading lab's R&D fleet is compared
    # at the same instant (the supplement's "Jan 2029").
    declaration = col("pause_year") - col("bottom_up_declaration_to_pause_years")
    sources = [  # (label, frac-key, chip origin, basis)
        ("Smuggled Western chips", "bottom_up_smuggled_frac", "US-designed", "2028 flow"),
        ("False decommissioning", "bottom_up_decommission_frac", "US-designed", "pre-2028 stock"),
        ("Concealed domestic production", "bottom_up_domestic_frac", "China-designed", "stock + flow"),
    ]
    src_stats = {lab: _source_stats(col(key), kw) for lab, key, _, _ in sources}

    tot_p = np.atleast_1d(_wpct(tot, kw, [10, 50, 90]))       # [p10, p50, p90] of world fraction
    h_p = np.atleast_1d(_wpct(h100e, kw, [10, 50, 90]))       # [p10, p50, p90] of covert H100e

    # ---- Table 1: headline (bottom-up Method #1) ----
    cost = h_p * USD_PER_H100E
    # Leading-lab R&D compute per rollout (from the input CSV at each rollout's declaration); the
    # comparison ratio is per-rollout (leading-lab / covert), so its 80% CI reflects both the
    # covert-size spread and any timing spread in the leading lab's fleet.
    ll_arr, ll_med, _ll_fp = _leading_lab_rd(ds, declaration, kw)
    ratio = np.atleast_1d(_wpct(ll_arr / np.maximum(h100e, 1.0), kw, [10, 50, 90]))
    headline = [
        ["", "Median estimate", "80% CI"],
        ["Fraction of world", _fmt_pct(tot_p[1]), f"[{_fmt_pct(tot_p[0])}, {_fmt_pct(tot_p[2])}]"],
        ["H100e", _fmt_h100e(h_p[1]), f"[{_fmt_h100e(h_p[0])}, {_fmt_h100e(h_p[2])}]"],
        [f"IT equipment cost (at ${USD_PER_H100E:,}/H100e)", _fmt_usd(cost[1]),
         f"[{_fmt_usd(cost[0])}, {_fmt_usd(cost[2])}]"],
        [f"Compared to leading lab R&D compute ({_fmt_h100e(ll_med)} H100e)",
         f"{ratio[1]:.0f}x less", f"[{ratio[0]:.0f}x less, {ratio[2]:.0f}x less]"],
    ]

    # ---- Table 2: acquisition routes (per source) ----
    routes = [["Source", "Chip origin", "Basis", "P(feasible)", "Median", "80% CI"]]
    for lab, _key, origin, basis in sources:
        pf, cond, _marg = src_stats[lab]
        routes.append([lab, origin, basis, _fmt_pct(pf, 0),
                       _fmt_pct(cond[1]), f"[{_fmt_pct(cond[0])}, {_fmt_pct(cond[2])}]"])

    # ---- Table 3: aggregation methods ----
    cond_medians = {lab: src_stats[lab][1][1] for lab, _, _, _ in sources}
    p_feas = {lab: src_stats[lab][0] for lab, _, _, _ in sources}
    add_medians = sum(cond_medians.values())
    weighted = sum(p_feas[lab] * cond_medians[lab] for lab in cond_medians)
    agg = [
        ["Aggregation method", "Result", "Notes"],
        ["Assume all routes feasible, add conditional medians", _fmt_pct(add_medians),
         "Conservative as a central estimate (ignores feasibility)"],
        ["Feasibility-weighted sum of conditional medians", _fmt_pct(weighted),
         "Each route's conditional median × its P(feasible)"],
        ["Monte-Carlo simulation (this run)", _fmt_pct(tot_p[1]),
         f"80% CI [{_fmt_pct(tot_p[0])}, {_fmt_pct(tot_p[2])}]  ·  full joint over all routes"],
    ]

    n = len(keep)
    rw = _rw_note(ds)
    basis_note = ("Fractions are % of world AI-relevant compute at the compute declaration. "
                  "'Feasible' = the route's feasibility draw fired on that rollout; conditional "
                  "stats are over those rollouts. Bottom-up (Method #1) only — NOT the "
                  "all-things-considered headline.")
    prov = f"{n} rollouts" + (f"  ·  {rw}" if rw else "")
    tables = [
        ("Table 1 — Covert compute acquired (bottom-up, Method #1)",
         "Median + 80% CI across rollouts. Leading-lab R&D compute is computed per rollout from "
         "the input-data CSV (training + inference + experiment H100e at the compute declaration). "
         f"The IT-cost row uses an external ${USD_PER_H100E:,}/H100e anchor.",
         headline),
        ("Table 2 — Acquisition routes", basis_note, routes),
        ("Table 3 — Aggregation methods", "", agg),
    ]
    return f"Covert compute-acquisition tables ({prov})", tables


def generate(ds, rundir):
    """Write tables_compute_acquisition.{md,html} into rundir. Returns the list of filenames
    written (empty when the run isn't a bottom_up_estimate run)."""
    built = _build_tables(ds)
    if built is None:
        return []
    title, tables = built
    rundir = Path(rundir)

    md = [f"# {title}", ""]
    for heading, note, rows in tables:
        md.append(f"## {heading}")
        if note:
            md.append("")
            md.append(f"*{note}*")
        md.append("")
        md.append(_md_table(rows))
        md.append("")
    (rundir / "tables_compute_acquisition.md").write_text("\n".join(md))

    html = [f"<h1>{title}</h1>"]
    for heading, note, rows in tables:
        html.append(f"<h2>{heading}</h2>")
        if note:
            html.append(f"<p><em>{note}</em></p>")
        html.append(_html_table(rows))
    (rundir / "tables_compute_acquisition.html").write_text("\n".join(html))

    return ["tables_compute_acquisition.md", "tables_compute_acquisition.html"]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m plan_a_covert_proj.pipeline.tables <runid>")
    runid = sys.argv[1]
    rundir = RUNS / runid
    dataset = rundir / "dataset.json"
    if not dataset.exists():
        sys.exit(f"no dataset at {dataset}")
    ds = json.loads(dataset.read_text())
    names = generate(ds, rundir)
    if not names:
        sys.exit(f"run {runid} is not a bottom_up_estimate run — no compute-acquisition tables")
    print(f"wrote {', '.join(names)} -> {rundir.relative_to(COVERT.parent)}")


if __name__ == "__main__":
    main()
