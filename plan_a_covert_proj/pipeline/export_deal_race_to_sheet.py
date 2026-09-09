"""Three-way covert-vs-deal race breakdown -> 'Stage 1 widget v2' tab of the
"Capability Scaling Strategy" spreadsheet.

For a deal_breakdown run, at each point in time every draw is in exactly one of three
mutually-exclusive states (a competing-risks split that sums to 100%):
  1. deal dissolved            (the deal collapsed before the covert secretly reached the milestone)
  2. covert reaches <milestone> before the deal dissolved (the covert "wins")
  3. neither yet

The milestone is TED-AI or ASI. For each milestone we write four tables =
{calendar year, years-after-legal-<milestone>} x {detection-aware, no-detection}, each
bucketed by covert legal_scaling_duration; the ASI block sits below the TED-AI block, with
a self-contained legend at the top.

Per draw: D = deal_breakdown_t (inf if the deal never breaks), T = covert arrival at the
milestone (inf if never), w = P(covert undetected AT the milestone; no-detection uses
w=1). At year Y:
  b2 = w * 1{T<=Y and T<D}                      covert reaches the milestone, undetected, first
  b1 = 1{D<=Y} * (1 - w*1{T<D})                 deal dissolved first (a detected covert can't win)
  b3 = 1 - b1 - b2                              neither
Detection's leftover (1-w) mass follows the deal's own outcome; the split is MECE and
reduces exactly to the physics race when w=1.

Milestone specifics (reusing export_covert_to_sheet's machinery):
  * TED-AI: covert arrival = covert_ttt; detection weight = survival_at_goal; legal arrival
    = the legal USED/deployed track (bare TED-AI).
  * ASI: covert arrival = covert_ttt + the covert trajectory's TED-AI->ASI gap; detection
    weight = survival re-evaluated at that ASI arrival; legal arrival = the legal
    DISCOVERED/research track (the deployed track is held below ASI by the alignment tax,
    so it never reaches ASI).

RELATIVE AXIS: anchored PER WORLD to when the legal reaches the milestone. Because a deal
that collapses early freezes the legal short of the milestone, the anchor is taken from the
identical-seed NO-breakdown twin run (ANCHOR_RUN) so every world has a well-defined "legal
reaches <milestone>" clock and deal dissolution is counted from the START of the deal
(early collapses land at negative offsets). The twin's draws are index-aligned (deal
breakdown is an independent RNG substream); this is asserted at load time.

Values are literal (a static snapshot of a Monte-Carlo run). Writing uses the gws CLI.

Usage:
    python -m plan_a_covert_proj.pipeline.export_deal_race_to_sheet [--write] [RACE_RUN] [ANCHOR_RUN]
"""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from plan_a_covert_proj.pipeline import export_covert_to_sheet as X  # noqa: E402  (_crossing/_load_detection/_survival_at)

RUNS = REPO_ROOT / "plan_a_covert_proj" / "outputs" / "runs"

RACE_RUN = "37246ad0fd20"        # the deal_breakdown run (Eli Final, 6000 draws)
# Identical-seed NO-breakdown twin -> counterfactual legal milestone arrival. None = use the
# race run's own in-simulation counterfactual (legal_*_would_ttt / legal_would_* tracks,
# recorded by sim.py before breakdown truncation) — available on runs built after would-ttt.
ANCHOR_RUN = None
SPREADSHEET_ID = "1MVdjISa09Y4qO3jT5eXYFz-s45gQm4NTXcZqEsFSowU"
SHEET_GID = 527363693            # the 'Stage 1 widget v2' tab; its TITLE is resolved from
                                 # this gid at write time (the tab has been renamed before)

CAL_YEARS = list(range(2028, 2056))
REL_OFFS = list(range(-6, 21))                   # legal-milestone-relative offsets (yr)
SCALE_EDGES = [1, 2, 3, 4, 5, 6, 7, 8]           # integer legal_scaling_duration bins
MILESTONES = [("TED-AI", "ted"), ("ASI", "asi")]


def _records(run):
    return json.loads((RUNS / run / "dataset.json").read_text())["records"]


def _covert_asi_arrival(r):
    """Covert ASI arrival = covert_ttt + the covert trajectory's TED-AI->ASI gap (>=0), so
    ASI never lands before TED-AI (mirrors export_covert_to_sheet.cdf_table)."""
    ttt = r.get("covert_ttt")
    c_asi = X._crossing(r.get("covert_t"), r.get("covert_e"), r.get("asi_e"))
    if ttt is None or c_asi is None:
        return np.inf
    c_ted = X._crossing(r.get("covert_t"), r.get("covert_e"), r.get("ted_ai_e"))
    gap = max(0.0, c_asi - c_ted) if c_ted is not None else 0.0
    return float(ttt) + gap


def load():
    """Shared D/SD plus, per milestone, (T covert arrival, W detection weight, Lcf
    counterfactual legal arrival) — the last from the index-aligned no-breakdown twin."""
    R = _records(RACE_RUN)
    D = np.array([r["deal_breakdown_t"] if r.get("deal_breakdown_t") is not None else np.inf for r in R], float)
    SD = np.array([r["covert_params"]["legal_scaling_duration"] for r in R], float)
    P = np.array([r["pause_year"] for r in R], float)

    if ANCHOR_RUN is None:
        if not R or R[0].get("legal_used_would_ttt") is None:
            raise SystemExit(f"run {RACE_RUN} has no in-run counterfactual (legal_used_would_ttt); "
                             "pass an identical-seed no-breakdown ANCHOR_RUN instead")
        A = R  # counterfactual arrivals recorded in the race run itself
    else:
        A = _records(ANCHOR_RUN)
        if len(A) != len(R):
            raise SystemExit(f"anchor run {ANCHOR_RUN} has {len(A)} records != race run {len(R)}; not aligned")
        if not (np.allclose(P, [r["pause_year"] for r in A])
                and np.allclose(SD, [r["covert_params"]["legal_scaling_duration"] for r in A])):
            raise SystemExit(f"anchor run {ANCHOR_RUN} is not index-aligned with {RACE_RUN} "
                             "(pause_year / legal_scaling_duration differ) — refusing to mix draws")

    # TED-AI milestone: stored fields. In-run mode reads the counterfactual would-ttt; twin
    # mode reads the twin's realized (never-frozen) arrival — the same quantity.
    T_ted = np.array([r["covert_ttt"] if r.get("covert_ttt") is not None else np.inf for r in R], float)
    W_ted = np.array([(r.get("survival_at_goal") or 0.0) for r in R], float)
    ted_key = "legal_used_would_ttt" if ANCHOR_RUN is None else "legal_used_ttt"
    Lcf_ted = np.array([r[ted_key] if r.get(ted_key) is not None else np.nan for r in A], float)

    # ASI milestone: covert arrival = ttt + TED->ASI gap; weight re-evaluated at ASI from the
    # run manifest's detection knobs; legal arrival = DISCOVERED track crossing asi_e (the used
    # track never reaches ASI).
    manifest = json.loads((RUNS / RACE_RUN / "manifest.json").read_text())
    fits, sb, gmin, gmax = X._load_detection(manifest)
    T_asi = np.array([_covert_asi_arrival(r) for r in R], float)
    W_asi = np.array([X._survival_at(r, a - p, fits, sb, gmin, gmax)
                      for r, a, p in zip(R, T_asi, P)], float)
    # In-run mode: the unfrozen would-tracks are stored only when a breakdown actually
    # truncated the realized ones — fall back to legal_t/legal_e (identical otherwise).
    Lcf_asi = np.array([X._crossing(r.get("legal_would_t") or r.get("legal_t"),
                                    r.get("legal_would_e") or r.get("legal_e"),
                                    r.get("asi_e")) or np.nan
                        for r in A], float)

    per = {"ted": (T_ted, W_ted, Lcf_ted), "asi": (T_asi, W_asi, Lcf_asi)}
    return R, D, SD, per


def buckets(Y, D, T, W, detection):
    """Y: per-draw absolute year (array). Returns (b1,b2,b3) means over the given draws."""
    w = W if detection else np.ones_like(W)
    tltd = (T < D)
    b2 = np.where((T <= Y) & tltd, w, 0.0)
    b1 = np.where(D <= Y, 1.0 - w * tltd, 0.0)
    b3 = 1.0 - b1 - b2
    return b1.mean(), b2.mean(), b3.mean()


def scale_bins(SD):
    labels = ["All"] + [f"{SCALE_EDGES[i]}–{SCALE_EDGES[i+1]}y" for i in range(len(SCALE_EDGES) - 1)]
    idx = np.clip(np.digitize(SD, SCALE_EDGES[1:-1], right=False), 0, len(SCALE_EDGES) - 2)
    masks = [np.ones(len(SD), bool)] + [idx == b for b in range(len(SCALE_EDGES) - 1)]
    return labels, masks


def build_table(axis, detection, D, T, W, SD, Lcf, mlabel):
    """axis in {'cal','rel'}. Returns list of rows (each a list of cells)."""
    labels, masks = scale_bins(SD)
    xs = CAL_YEARS if axis == "cal" else REL_OFFS
    xhead = "calendar year" if axis == "cal" else f"yrs after legal {mlabel}"
    base = np.isfinite(Lcf) if axis == "rel" else np.ones(len(SD), bool)

    bhdr = [""]
    for lab, m in zip(labels, masks):
        bhdr += [f"{lab} (n={int((m & base).sum())})", "", ""]
    mhdr = [xhead] + ["deal dissolved", f"covert→{mlabel}", "neither"] * len(labels)

    rows = [bhdr, mhdr]
    for x in xs:
        row = [x]
        for m in masks:
            sel = m & base
            if sel.sum() == 0:
                row += ["", "", ""]; continue
            Y = (np.full(int(sel.sum()), float(x)) if axis == "cal" else Lcf[sel] + x)
            b1, b2, b3 = buckets(Y, D[sel], T[sel], W[sel], detection)
            row += [round(b1 * 100, 1), round(b2 * 100, 1), round(b3 * 100, 1)]
        rows.append(row)
    return rows


def buckets5(Y, D, T, W, Ldiv, detection):
    """Five-way split: the 'deal dissolved' and 'covert win' outcomes are each further
    split by whether the event landed BEFORE or AFTER the legal reached TED-AI (Ldiv).
    Returns (dslv_pre, dslv_post, covert_pre, covert_post, neither) means. MECE: dslv_pre+
    dslv_post = the 3-way 'deal dissolved'; covert_pre+covert_post = the 3-way 'covert win';
    neither is unchanged."""
    w = W if detection else np.ones_like(W)
    tltd = (T < D)
    notwin = 1.0 - w * tltd                       # covert did NOT win before D
    dslv_pre = np.where((D <= Y) & (D < Ldiv), notwin, 0.0)
    dslv_post = np.where((D <= Y) & (D >= Ldiv), notwin, 0.0)
    cov_pre = np.where((T <= Y) & tltd & (T < Ldiv), w, 0.0)
    cov_post = np.where((T <= Y) & tltd & (T >= Ldiv), w, 0.0)
    neither = 1.0 - dslv_pre - dslv_post - cov_pre - cov_post
    return (dslv_pre.mean(), dslv_post.mean(), cov_pre.mean(), cov_post.mean(), neither.mean())


SUB5 = ["deal dslv pre-TED-AI", "deal dslv post-TED-AI", "covert pre-TED-AI",
        "covert post-TED-AI", "neither"]


def build_table5(axis, detection, D, T, W, SD, Lanchor, Ldiv, mlabel):
    """Like build_table but with the 5-way pre/post-legal-TED-AI split (Ldiv = legal TED-AI
    arrival). Lanchor is the relative-axis anchor (legal milestone arrival)."""
    labels, masks = scale_bins(SD)
    xs = CAL_YEARS if axis == "cal" else REL_OFFS
    xhead = "calendar year" if axis == "cal" else f"yrs after legal {mlabel}"
    base = (np.isfinite(Lanchor) & np.isfinite(Ldiv)) if axis == "rel" else np.isfinite(Ldiv)

    bhdr = [""]
    for lab, m in zip(labels, masks):
        bhdr += [f"{lab} (n={int((m & base).sum())})"] + [""] * 4
    mhdr = [xhead] + SUB5 * len(labels)

    rows = [bhdr, mhdr]
    for x in xs:
        row = [x]
        for m in masks:
            sel = m & base
            if sel.sum() == 0:
                row += [""] * 5; continue
            Y = (np.full(int(sel.sum()), float(x)) if axis == "cal" else Lanchor[sel] + x)
            vals = buckets5(Y, D[sel], T[sel], W[sel], Ldiv[sel], detection)
            row += [round(v * 100, 1) for v in vals]
        rows.append(row)
    return rows


def legend(n, built, name):
    """Self-contained reader's guide, one line per row (single cell so it overflows)."""
    B = []
    return [
        ["DEAL-RACE BREAKDOWN — over time, what covert-vs-legal-deal state is each simulated world in?"],
        [f"Static snapshot · run {RACE_RUN} ({name}) · {n} Monte-Carlo draws · built {built}. "
         f"Values = % of draws; each 'deal dissolved / covert→milestone / neither' triple sums to 100."],
        B,
        ["WHAT'S MODELED: a legal consortium pauses frontier AI then scales toward TED-AI under a deal; a covert "
         "project secretly diverts compute and races to the milestone; the deal can collapse at a random time "
         "(hazard curve). Each draw = one simulated world."],
        B,
        ["MILESTONES — two blocks of tables, one below the other:"],
        ["• TED-AI: covert reaches TED-AI = covert_ttt; legal reaches TED-AI = its USED/deployed track (bare TED-AI)."],
        ["• ASI: covert reaches ASI = covert_ttt + the covert trajectory's TED-AI→ASI gap (detection re-evaluated at "
         "ASI); legal reaches ASI = its DISCOVERED/research track — the deployed track is held below ASI by the "
         "alignment tax, so it never gets there."],
        B,
        ["THE 3 OUTCOMES (each triple of columns) — mutually exclusive at every point in time:"],
        ["• deal dissolved — the deal collapsed before the covert secretly reached the milestone (absorbing: a world stays here once it happens)."],
        ["• covert→milestone — the covert project reached the milestone while the deal was still holding (the covert 'wins' the race)."],
        ["• neither — as of this time, neither of the above has happened yet."],
        B,
        ["INDEPENDENT VARIABLE (the rows) — differs by table:"],
        ["• 'CALENDAR YEAR' tables: each row is an actual year. Read: 'by year Y, X% of worlds have dissolved / seen a covert win / neither.'"],
        [f"• 'YEARS AFTER LEGAL REACHES …' tables: each row is an offset measured PER WORLD from the year that "
         f"world's LEGAL project reaches the milestone (0 = that moment, +2 = two years later, −1 = one year before). "
         f"ALL worlds are included and deal dissolution is counted from the START of the deal, so collapses before the "
         f"legal gets there show up at NEGATIVE offsets. For worlds where the collapse would freeze the legal short of "
         f"the milestone, the anchor is the legal's would-be arrival absent breakdown "
         + (f"(from the identical-seed no-breakdown twin run {ANCHOR_RUN})." if ANCHOR_RUN
            else "(the run's own in-simulation deal-holds counterfactual).")],
        B,
        ["COLUMN GROUPS = scaling-yr bucket (the covert config's legal_scaling_duration):"],
        ["• Meaning: the number of years, from the END of the setup pause, the legal project takes to scale up to its "
         "TED-AI target (target_year = pause + setup + legal_scaling_duration). Drawn uniform 1–8 yr across worlds."],
        ["• '1–2y' = worlds where that duration is 1–2 yr (legal scales FAST); '7–8y' = 7–8 yr (legal scales SLOWLY). "
         "'All' = every world. '(n=…)' = number of draws in the bucket."],
        B,
        ["DETECTION REGIMES (each axis is shown both ways):"],
        ["• no-detection (physics race): pure timing — does the covert reach the milestone before the deal collapses, ignoring whether it is caught."],
        ["• detection-aware: the covert counts as a win only if undetected, weighted by P(undetected @milestone). "
         "Detected-covert worlds fall back to the deal's own outcome (dissolved if it collapses, else neither)."],
    ]


# (title, axis, detection) for each milestone block
SPECS = [
    ("CALENDAR YEAR — covert reaches {M} — detection-aware", "cal", True),
    ("CALENDAR YEAR — covert reaches {M} — no-detection (physics race)", "cal", False),
    ("YEARS AFTER LEGAL REACHES {M} — detection-aware", "rel", True),
    ("YEARS AFTER LEGAL REACHES {M} — no-detection (physics race)", "rel", False),
]


SPLIT_BANNER = ("════════════  SPLIT VERSIONS — 'deal dissolved' and 'covert win' each split by "
                "PRE vs POST the legal reaching TED-AI  ════════════")
SPLIT_NOTE = ("Same tables as above, but the 'deal dissolved' and 'covert→milestone' outcomes are each split "
              "in two by whether the event happened BEFORE or AFTER the legal project reached (bare/used-track) "
              "TED-AI — its counterfactual arrival from the no-breakdown twin. Five columns per scaling bucket: "
              "deal dslv pre-TED-AI / deal dslv post-TED-AI / covert pre-TED-AI / covert post-TED-AI / neither, "
              "summing to 100. (pre+post recover the 3-way buckets above.) NOTE: the pre/post divider is always "
              "the legal's TED-AI arrival, even in the ASI block.")


def assemble():
    R, D, SD, per = load()
    n = len(D)
    name = json.loads((RUNS / RACE_RUN / "manifest.json").read_text()).get("name") or RACE_RUN
    built = datetime.fromtimestamp((RUNS / RACE_RUN / "manifest.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    W_COLS = 1 + 5 * len(SCALE_EDGES)            # widest section: 1 label + 8 buckets*5 = 41
    def pad(r): return r + [""] * (W_COLS - len(r))
    Ldiv = per["ted"][2]                          # legal TED-AI arrival = pre/post divider (both blocks)

    out = list(legend(n, built, name))
    # 1) the 3-way tables (deal dissolved / covert / neither)
    for mlabel, mkey in MILESTONES:
        T, W, Lcf = per[mkey]
        out.append([]); out.append([f"════════  MILESTONE: {mlabel}  ════════"])
        for title, axis, det in SPECS:
            out.append([])
            out.append([title.format(M=mlabel)])
            for r in build_table(axis, det, D, T, W, SD, Lcf, mlabel):
                out.append(pad(r))
    # 2) the 5-way split versions (pre/post legal TED-AI)
    out.append([]); out.append([]); out.append([SPLIT_BANNER]); out.append([SPLIT_NOTE])
    for mlabel, mkey in MILESTONES:
        T, W, Lcf = per[mkey]
        out.append([]); out.append([f"════════  MILESTONE: {mlabel}  ·  split pre/post legal TED-AI  ════════"])
        for title, axis, det in SPECS:
            out.append([])
            out.append([title.format(M=mlabel) + "  ·  split pre/post legal TED-AI"])
            for r in build_table5(axis, det, D, T, W, SD, Lcf, Ldiv, mlabel):
                out.append(pad(r))
    return out, D, SD, per


def _preview(D, SD, per):
    for mlabel, mkey in MILESTONES:
        T, W, Lcf = per[mkey]
        for title, axis, det in (SPECS[0], SPECS[2]):
            tbl = build_table(axis, det, D, T, W, SD, Lcf, mlabel)
            print(f"--- [{mlabel}] {title.format(M=mlabel)} (All bucket) ---")
            for row in tbl[2:]:
                keep = (row[0] in (2031, 2035, 2040, 2050) if axis == "cal"
                        else row[0] in (-6, 0, 2, 8, 20))
                if keep:
                    print(f"  {str(row[0]):>6} | dissolved {row[1]:5}  covert {row[2]:5}  neither {row[3]:5}"
                          f"  (Σ{round(row[1]+row[2]+row[3],1)})")
            print()
    # split-version preview (TED-AI, calendar, detection-aware, All bucket)
    Ldiv = per["ted"][2]
    for mlabel, mkey in MILESTONES:
        T, W, Lcf = per[mkey]
        tbl = build_table5("cal", True, D, T, W, SD, Lcf, Ldiv, mlabel)
        print(f"--- SPLIT [{mlabel}] CALENDAR · detection-aware (All bucket): "
              f"dslv_pre / dslv_post / cov_pre / cov_post / neither ---")
        for row in tbl[2:]:
            if row[0] in (2031, 2035, 2040, 2050):
                v = row[1:6]
                print(f"  {row[0]} | {v[0]:5} {v[1]:5} {v[2]:5} {v[3]:5} {v[4]:5}  (Σ{round(sum(v),1)})")
        print()


def main(write):
    out, D, SD, per = assemble()
    print(f"assembled {len(out)} rows for sheet gid {SHEET_GID}\n")
    _preview(D, SD, per)
    if not write:
        print("DRY RUN (no sheet write). Re-run with --write to push.")
        return

    def gws(resource, method, body):
        cmd = ["gws", "sheets", "spreadsheets"] + resource + [method,
               "--params", json.dumps({"spreadsheetId": SPREADSHEET_ID}),
               "--json", json.dumps(body), "--format", "json"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"gws {' '.join(resource)} {method} failed:\n{r.stderr or r.stdout}")
        return r.stdout

    # resolve the tab's current title from its gid (the tab has been renamed before)
    meta = subprocess.run(
        ["gws", "sheets", "spreadsheets", "get",
         "--params", json.dumps({"spreadsheetId": SPREADSHEET_ID, "fields": "sheets.properties"}),
         "--format", "json"],
        capture_output=True, text=True)
    if meta.returncode != 0:
        raise SystemExit(f"gws spreadsheets get failed:\n{meta.stderr or meta.stdout}")
    props = [s["properties"] for s in json.loads(meta.stdout)["sheets"]]
    sheet = next((p["title"] for p in props if p["sheetId"] == SHEET_GID), None)
    if sheet is None:
        raise SystemExit(f"no tab with gid {SHEET_GID} in spreadsheet {SPREADSHEET_ID}")

    # ensure the grid is wide enough for the 41-col split tables (idempotent)
    gws([], "batchUpdate", {"requests": [{"updateSheetProperties": {
        "properties": {"sheetId": SHEET_GID, "gridProperties": {"columnCount": 45}},
        "fields": "gridProperties.columnCount"}}]})
    gws(["values"], "batchClear", {"ranges": [f"'{sheet}'!A1:AS900"]})
    gws(["values"], "batchUpdate", {"valueInputOption": "USER_ENTERED",
                                    "data": [{"range": f"'{sheet}'!A1", "values": out}]})
    print(f"cleared A1:AS900 and wrote {len(out)} rows to '{sheet}' A1 (run {RACE_RUN})")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--write"]
    if len(argv) >= 1:
        RACE_RUN = argv[0]
    if len(argv) >= 2:
        ANCHOR_RUN = argv[1]
    main("--write" in sys.argv)
