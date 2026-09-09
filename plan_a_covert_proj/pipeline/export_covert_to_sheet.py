"""Export the covert-vs-legal probabilities to the Stage-1 widget sheet.

Under the "Covert" header of the "Stage 1 widget params" tab of the "Capability
Scaling Strategy" spreadsheet this writes, in order:

  1. the within-N-of-legal table — P(covert reaches TED-AI within N yr of the
     legal project), detection-aware AND physics-only, for N = 0/5/10, by
     half-year legal scaling duration (``bin_table``);
  2. a per-year CDF — P(covert reaches TED-AI <= N yr after the pause / scaling
     start), for N = 0..CDF_TMAX, one row per (scaling yr x mode);
  3. the same per-year CDF for ASI instead of TED-AI.

For 2/3 the reference clock is the pause: every draw's post-deal trajectory
starts at ``pause_year`` (``covert_t[0] == pause_year``), and the covert project
is dormant ``dormancy_years`` before it scales, so the CDF is flat until then.
  * phys[N] = mean( 1{arrival - pause <= N} )                     physics-only
  * det[N]  = mean( 1{arrival - pause <= N} * survival_at_arrival ) detection-aware
TED-AI arrival is the stored ``covert_ttt`` with survival ``survival_at_goal``.
ASI arrival is the covert trajectory (``covert_e``) crossing ``asi_e``, with
survival re-evaluated at that ASI crossing from the run manifest's frailty fits +
silver-bullet cfg (reproduces the stored ``survival_at_goal`` at the TED time to
~1e-6, so it is exact — still a reader, never re-runs the physics).

Surgical by design: it only touches the title/info/header/data rows of each
table (control rows the widget owns between the within-N info row and its data
header — e.g. "Scaling yr offset", "Include" — are located-around and never
overwritten). The two CDF tables are written below the within-N table; stale
rows around/below them are cleared.

Values are literal (a static snapshot of a Monte-Carlo run, not derivable from
other cells); each table's informational row records which run and when it was
built. Writing to the sheet uses the authenticated `gws` CLI.

Usage:
    python -m plan_a_covert_proj.pipeline.export_covert_to_sheet [RUN_ID]
    # RUN_ID defaults to the latest run with a dataset.
"""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plan_a_covert_proj import detection  # noqa: E402
from plan_a_covert_proj.pipeline.covert_within_vs_scaling_duration import (  # noqa: E402
    bin_table, _latest_run_with_data)

SPREADSHEET_ID = "1MVdjISa09Y4qO3jT5eXYFz-s45gQm4NTXcZqEsFSowU"
SHEET_NAME = "Stage 1 widget params"
SHEET_BIN_WIDTH = 0.5                      # finer than the figure's 1-yr bins
COVERT_HEADER = "Covert"                   # section anchor in column A
DATA_HEADER_KEY = "scaling yr"            # first cell of the data header row
TITLE = "P(covert reaches TED-AI within N yr of legal) by legal scaling duration"
HEADER = ["scaling yr", "det ≤0y", "det ≤5y", "det ≤10y",
          "phys ≤0y", "phys ≤5y", "phys ≤10y", "n draws"]
CLEAR_BUFFER = 12                          # rows below the data to wipe of stale export
CDF_TMAX = 20                              # per-year CDF offset columns: 0..CDF_TMAX yr after pause


def _gws(method: str, params: dict, body: dict | None = None):
    cmd = ["gws", "sheets", "spreadsheets", "values", method,
           "--params", json.dumps(params), "--format", "json"]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gws {method} failed:\n{out.stderr or out.stdout}")
    s = out.stdout.strip()
    if not s:
        return {}
    return json.loads(s[s.index("{"):])   # tolerate any preamble before the JSON


def _rng(a1: str) -> str:
    return f"'{SHEET_NAME}'!{a1}"


def _col_a(n_rows: int = 80) -> list[str]:
    res = _gws("get", {"spreadsheetId": SPREADSHEET_ID, "range": _rng(f"A1:A{n_rows}")})
    return [(row[0] if row else "") for row in res.get("values", [])]


def _run_note(run_dir: Path, n_draws: int) -> str:
    mf = run_dir / "manifest.json"
    name = None
    if mf.exists():
        try:
            name = (json.loads(mf.read_text()) or {}).get("name")
        except Exception:  # noqa: BLE001
            name = None
    built = (datetime.fromtimestamp(mf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
             if mf.exists() else "?")
    run_desc = run_dir.name + (f" ({name})" if name else "")
    return (f"static snapshot · stage-1 run {run_desc} · {n_draws} draws · "
            f"run built {built} · det=detection-aware (×P undetected @TED-AI), "
            f"phys=physics-only race")


def _col_letter(n: int) -> str:
    """1-based column index -> A1 letter (1->A, 26->Z, 27->AA)."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _crossing(t, e, level):
    """First time the covert trajectory ``e`` reaches ``level`` (lerp), or None.
    Mirrors sim._crossing_time; compresses out the None points the stored covert
    trajectory carries during the frozen dormancy window."""
    if level is None or not t or not e:
        return None
    tf, ef = [], []
    for ti, ei in zip(t, e):
        if ei is not None:
            tf.append(ti); ef.append(ei)
    if not ef:
        return None
    e = np.asarray(ef, float); t = np.asarray(tf, float)
    ab = np.where(e >= level)[0]
    if len(ab) == 0:
        return None
    i = int(ab[0])
    if i == 0:
        return float(t[0])
    e0, e1, t0, t1 = e[i - 1], e[i], t[i - 1], t[i]
    return float(t1) if e1 == e0 else float(t0 + (level - e0) / (e1 - e0) * (t1 - t0))


def _load_detection(manifest: dict):
    """Rebuild the detection reader inputs from the run manifest (frailty fits +
    silver-bullet cfg + grid size range) — the manifest stores these precisely so
    readers re-derive survival curves without re-fitting or re-running physics."""
    cfg = manifest.get("config") or {}
    fits = detection.normalize_fits(cfg["detection_frailty_fit"])
    sb_raw = cfg.get("detection_silver_bullet")
    sb = None
    if sb_raw:
        sb = {"p0": float(sb_raw["p_mid_pause_to_ted"]),
              "p1": float(sb_raw["p_used_ted_plus_d"]),
              "delta": float(sb_raw["delta_years"]),
              "C": float(sb_raw["ceiling"])}
    sizes = sorted(float(g["size_mw"]) for g in cfg["detection_grids"])
    return fits, sb, sizes[0], sizes[-1]


def _survival_at(r: dict, t_after_pause: float, fits, sb, gmin, gmax) -> float:
    """P(covert still undetected at ``t_after_pause`` years after the pause) for
    draw ``r``, reconstructed exactly from stored knobs (see _record_out in mc.py:
    size_mw = covert_size_mw, site_mw = clip(optimal_site_mw), silver bullet from
    the legal used->TED-AI crossing). 0.0 when the covert size is undefined."""
    if t_after_pause is None or not np.isfinite(t_after_pause):
        return 0.0
    cp = r["covert_params"]
    size_mw = r.get("covert_size_mw")
    if size_mw is None:
        h = r.get("total_h100e_used")
        size_mw = (float(h) * float(cp["facility_w_per_h100e"]) / 1e6) if h else None
    if not size_mw:
        return 0.0
    site_mw = float(np.clip(cp["optimal_site_mw"], gmin, gmax))
    # the record's detection_r is the hazard used in the draw (the joint-fit shared
    # slip-up rate); legacy datasets (pre-joint-fit) had 0
    det_r = float(r.get("detection_r", 0.0))
    n, eff = detection.deploy(size_mw, site_mw, True)
    p = detection.interp_frailty(eff, fits)
    spec = None
    if sb is not None:
        ut = r.get("legal_used_ttt")
        if ut is not None and np.isfinite(ut):
            t_pause_end = float(cp["setup_period_years"])
            t_used_ted = float(ut) - float(r["pause_year"])
            t0 = 0.5 * (t_pause_end + t_used_ted)
            t1 = t_used_ted + sb["delta"]
            spec = detection.build_silver_bullet(sb["p0"], t0, sb["p1"], t1, sb["C"])
    return float(detection.g_no_silver_bullet(spec, t_after_pause)
                 * detection.survival(p, n, t_after_pause, det_r))


def _bins(sd, width):
    """Half-year legal-scaling-duration bins, identical to bin_table's."""
    lo = np.floor(sd.min() / width) * width
    hi = np.ceil(sd.max() / width) * width
    edges = np.arange(lo, hi + width * 0.5, width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(sd, edges[1:-1], right=False), 0, len(centres) - 1)
    return centres, idx


def cdf_table(ds, manifest, milestone, width=SHEET_BIN_WIDTH, tmax=CDF_TMAX):
    """Per-year CDF of P(covert reaches ``milestone`` <= N yr after the pause),
    one row per (scaling-duration bin x mode). ``milestone`` in {"ted", "asi"}.

    Returns (header, rows) where each bin contributes a "det" row then a "phys"
    row: [scaling_yr, mode, v0, v1, ..., v_tmax, n]. A draw that never reaches the
    milestone (arrival None/NaN) counts as not-reached for every N. det weights
    each reached draw by the survival at THAT milestone's arrival.
    """
    R = ds.get("records") or []
    sd = np.array([r["covert_params"]["legal_scaling_duration"] for r in R], float)
    pause = np.array([r["pause_year"] for r in R], float)

    if milestone == "ted":
        arrival = np.array([r["covert_ttt"] if r.get("covert_ttt") is not None else np.nan
                            for r in R], float)
        weight = np.array([(r.get("survival_at_goal") or 0.0) for r in R], float)
    elif milestone == "asi":
        fits, sb, gmin, gmax = _load_detection(manifest)
        # Anchor ASI arrival to the official covert_ttt (fine-grid TED crossing) plus the
        # covert trajectory's TED->ASI gap, both crossings taken on the SAME stored
        # (downsampled) trajectory so the gap is >= 0. The stored covert_e is coarser than
        # the grid covert_ttt came from, so an absolute ASI crossing could otherwise land
        # microscopically before covert_ttt; this guarantees covert reaches ASI no earlier
        # than TED-AI in every draw.
        def _asi_arrival(r):
            ttt = r.get("covert_ttt")
            c_asi = _crossing(r.get("covert_t"), r.get("covert_e"), r.get("asi_e"))
            if ttt is None or c_asi is None:
                return np.nan                       # never reaches TED-AI, or never reaches ASI
            c_ted = _crossing(r.get("covert_t"), r.get("covert_e"), r.get("ted_ai_e"))
            gap = max(0.0, c_asi - c_ted) if c_ted is not None else 0.0
            return float(ttt) + gap
        arrival = np.array([_asi_arrival(r) for r in R], float)
        weight = np.array([_survival_at(r, a - p, fits, sb, gmin, gmax)
                           for r, a, p in zip(R, arrival, pause)], float)
    else:
        raise ValueError(f"milestone must be 'ted' or 'asi', got {milestone!r}")

    elapsed = arrival - pause                       # years after the pause
    centres, idx = _bins(sd, width)
    Ts = list(range(0, tmax + 1))
    header = ["scaling yr", "mode"] + [f"≤{t}y" for t in Ts] + ["n draws"]
    rows = []
    for b in range(len(centres)):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        el, w = elapsed[sel], weight[sel]
        reached = {t: (el <= t) for t in Ts}        # NaN <= t -> False
        phys = [round(float(reached[t].mean()), 3) for t in Ts]
        det = [round(float((reached[t] * w).mean()), 3) for t in Ts]
        x = round(float(centres[b]), 2)
        rows.append([x, "det"] + det + [n])
        rows.append([x, "phys"] + phys + [n])
    return header, rows


def _cdf_note(run_dir: Path, n_draws: int, milestone: str) -> str:
    base = _run_note(run_dir, n_draws)
    goal = "TED-AI" if milestone == "ted" else "ASI"
    if milestone == "ted":
        extra = ("t=0 is the pause (scaling start; covert_t[0]=pause_year) · "
                 "det=×P undetected @TED-AI (survival_at_goal)")
    else:
        extra = ("t=0 is the pause (scaling start) · covert ASI arrival = covert_e "
                 "crossing asi_e · det=×P undetected @ASI (survival re-evaluated at "
                 "the ASI crossing)")
    # replace the within-N tail of the shared note with the CDF-specific tail
    head = base.split(" · det=")[0]
    return f"{head} · {extra} · phys=physics-only race · [{goal} per-year CDF]"


def main(run_id: str | None = None):
    runs = REPO_ROOT / "plan_a_covert_proj" / "outputs" / "runs"
    run_dir = (runs / run_id) if run_id else _latest_run_with_data()
    ds = json.loads((run_dir / "dataset.json").read_text())
    tbl = bin_table(ds, SHEET_BIN_WIDTH)
    if not tbl:
        raise SystemExit(f"run {run_dir.name} lacks the fields this export needs")
    n_draws = len(ds.get("records") or [])
    note = _run_note(run_dir, n_draws)

    # Locate the section. Title/note sit directly under "Covert"; the data header
    # is found by name so any widget control rows between the note and the header
    # (e.g. "Scaling yr offset", "Include") are preserved untouched.
    col = _col_a()
    try:
        covert_row = col.index(COVERT_HEADER) + 1          # 1-based
    except ValueError:
        raise SystemExit(f'"{COVERT_HEADER}" not found in column A of "{SHEET_NAME}"')
    title_row, note_row = covert_row + 1, covert_row + 2
    header_row = next((i + 1 for i, c in enumerate(col)
                       if i + 1 > note_row and c == DATA_HEADER_KEY), note_row + 1)

    data = [[round(r["x"], 2)] + [round(r["det"][N], 3) for N in (0, 5, 10)]
            + [round(r["phys"][N], 3) for N in (0, 5, 10)] + [r["n"]] for r in tbl]
    first_data, last_data = header_row + 1, header_row + len(data)

    # Two per-year CDF tables below the within-N table (TED-AI then ASI), with a
    # blank separator row before each. ASI survival is rebuilt from the manifest.
    mf_path = run_dir / "manifest.json"
    if not mf_path.exists():
        raise SystemExit(f"run {run_dir.name} has no manifest.json "
                         f"(needed for the covert ASI survival curve)")
    manifest = json.loads(mf_path.read_text())
    ted_header, ted_rows = cdf_table(ds, manifest, "ted")
    asi_header, asi_rows = cdf_table(ds, manifest, "asi")
    ted_title = ("P(covert reaches TED-AI ≤ N yr after the pause / scaling start) — "
                 "per-year CDF, one row per scaling yr × mode")
    asi_title = ("P(covert reaches ASI ≤ N yr after the pause / scaling start) — "
                 "per-year CDF, one row per scaling yr × mode")

    ncdf = len(ted_header)                          # 2 + (CDF_TMAX+1) + 1 columns
    last_col = _col_letter(ncdf)
    pad = lambda row: row + [""] * (ncdf - len(row))
    cdf_section = (
        [pad([ted_title]), pad([_cdf_note(run_dir, n_draws, "ted")]), ted_header] + ted_rows
        + [[""] * ncdf]                             # separator between the two CDF tables
        + [pad([asi_title]), pad([_cdf_note(run_dir, n_draws, "asi")]), asi_header] + asi_rows)
    sep_row = last_data + 1                         # blank row under the within-N table
    cdf_start = last_data + 2
    cdf_end = cdf_start + len(cdf_section) - 1

    _gws("batchUpdate", {"spreadsheetId": SPREADSHEET_ID},
         {"valueInputOption": "USER_ENTERED", "data": [
             {"range": _rng(f"A{title_row}"), "values": [[TITLE]]},
             {"range": _rng(f"A{note_row}"), "values": [[note]]},
             {"range": _rng(f"A{header_row}:H{header_row}"), "values": [HEADER]},
             {"range": _rng(f"A{first_data}:H{last_data}"), "values": data},
             {"range": _rng(f"A{cdf_start}:{last_col}{cdf_end}"), "values": cdf_section},
         ]})
    # Clear the separator row and any stale rows a previous, longer export left below.
    _gws("batchClear", {"spreadsheetId": SPREADSHEET_ID},
         {"ranges": [_rng(f"A{sep_row}:{last_col}{sep_row}"),
                     _rng(f"A{cdf_end + 1}:{last_col}{cdf_end + CLEAR_BUFFER}")]})

    preserved = [c for c in col[note_row:header_row - 1] if c]
    ted_first = cdf_start + 3                        # title, note, header, then data
    ted_last = ted_first + len(ted_rows) - 1
    asi_first = ted_last + 5                         # separator, title, note, header, then data
    print(f"wrote within-N table {len(data)} rows to '{SHEET_NAME}' "
          f"A{first_data}:H{last_data} from run {run_dir.name}")
    print(f"  informational row A{note_row}: {note}")
    print(f"  preserved control rows between note and header: {preserved or '(none)'}")
    print(f"  TED-AI per-year CDF: {len(ted_rows)} rows A{ted_first}:{last_col}{ted_last}")
    print(f"  ASI per-year CDF:    {len(asi_rows)} rows A{asi_first}:{last_col}{cdf_end}")
    print(f"  cleared separator A{sep_row} and stale "
          f"A{cdf_end + 1}:{last_col}{cdf_end + CLEAR_BUFFER}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
