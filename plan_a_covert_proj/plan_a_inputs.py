#!/usr/bin/env python3
"""Verify + relativize the Plan-A compute series against the baseline.

The covert experiment runs on TWO compute time series:
  * BASELINE  = inputs/input_data.csv  (the default / "non-Plan-A" world; sets AC & the pause)
  * PLAN A    = inputs/plan-a.csv    (the post-pause divergence: a harder compute ramp)

Two invariants make that split well-posed, and both are easy to break with the kind of
"random numerical adjustments" colleagues make to these CSVs:

  1. The two series are IDENTICAL at every annual grid point up to the fork year (the last
     year before they diverge). That shared region is the counterfactual the pause time is
     read from. (Sub-grid INTERPOLATED values just before the fork differ slightly because
     the fork-year endpoint differs across the series — that's expected and fine; we only
     check grid points.)
  2. The fork happens AFTER the median pause (so the pause lands in the shared region).

This script asserts those, and then RELATIVIZES the divergence: it expresses Plan A as a
multiplicative ramp over the baseline keyed by *years since the fork*, rather than pinned
to calendar 2030. That lets the pipeline re-anchor the ramp to each rollout's own pause
time (`build_plan_a(baseline, fork_year=<pause>)`) so the divergence shifts with the pause.

Run it as a check (exits non-zero on any failure):
    python -m plan_a_covert_proj.plan_a_inputs            # verify + report + round-trip
    python -m plan_a_covert_proj.plan_a_inputs --write-ramp  # also dump plan_a_ramp.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = ROOT / "inputs" / "input_data.csv"          # default / non-Plan-A world
PLAN_A_CSV = ROOT / "inputs" / "plan-a.csv"     # Plan-A divergence

COMPUTE_COLS = ("inference_H100e", "experiment_H100e", "training_H100e")
ALL_COLS = ("time", "L_HUMAN") + COMPUTE_COLS

# The series MUST be bit-identical at every grid point with time <= this year. It has to
# sit at/after the last grid point before the median pause and at/before the fork, so the
# pause is read from the shared region. Median pause is ~2029.3 and the fork is 2030.0, so
# 2029.0 (the last grid point before both) is the right line. Revisit if the pause moves
# before 2029.0 or the fork moves to/under it.
IDENTICAL_THROUGH_YEAR = 2029.0
# The fork (first year the series differ) must be no earlier than this — i.e. the shared
# counterfactual region must extend at least through IDENTICAL_THROUGH_YEAR.
MIN_FORK_YEAR = 2030.0
# Relative tolerance for "identical" (these CSVs are hand-rounded decimals, so exact).
RTOL = 1e-9


def load_series(path: Path) -> dict:
    """Load a compute CSV into {col: np.ndarray}. Validates the header."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = set(ALL_COLS) - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return {c: np.array([float(r[c]) for r in rows], dtype=float) for c in ALL_COLS}


def _first_divergence_year(base: dict, plan: dict) -> float | None:
    """Earliest grid time at which any compute column differs (None if never)."""
    t = base["time"]
    diff = np.zeros(len(t), dtype=bool)
    for c in COMPUTE_COLS:
        diff |= ~np.isclose(base[c], plan[c], rtol=RTOL, atol=0.0)
    idx = np.where(diff)[0]
    return float(t[idx[0]]) if idx.size else None


def verify(base: dict, plan: dict) -> dict:
    """Assert the baseline<->Plan-A invariants. Raises AssertionError on violation;
    returns a small report dict on success."""
    errs: list[str] = []

    if not np.array_equal(base["time"], plan["time"]):
        errs.append("time grids differ between baseline and plan-a")
        # can't meaningfully continue column checks on a mismatched grid
        raise AssertionError("plan_a_inputs invariant failure:\n  - " + "\n  - ".join(errs))

    t = base["time"]
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        errs.append(f"time grid is not uniform (dt values {sorted(set(np.round(dt,6)))})")

    if not np.allclose(base["L_HUMAN"], plan["L_HUMAN"], rtol=RTOL):
        errs.append("L_HUMAN differs between the series (must be shared)")

    # (1) identical at every grid point up to IDENTICAL_THROUGH_YEAR
    pre = t <= IDENTICAL_THROUGH_YEAR + 1e-9
    for c in COMPUTE_COLS:
        bad = pre & ~np.isclose(base[c], plan[c], rtol=RTOL, atol=0.0)
        if bad.any():
            yrs = ", ".join(f"{x:.1f}" for x in t[bad][:6])
            errs.append(f"{c}: series differ at/before {IDENTICAL_THROUGH_YEAR:.0f} "
                        f"(years: {yrs}{'...' if bad.sum() > 6 else ''}) — the pre-pause "
                        f"region must be shared")

    # (2) the fork must be no earlier than MIN_FORK_YEAR
    fork = _first_divergence_year(base, plan)
    if fork is None:
        errs.append("series never diverge — plan-a.csv has no Plan-A ramp at all")
    elif fork < MIN_FORK_YEAR - 1e-9:
        errs.append(f"series diverge at {fork:.1f}, before MIN_FORK_YEAR {MIN_FORK_YEAR:.0f} "
                    f"— Plan A is encroaching on the shared pre-pause region")

    if errs:
        raise AssertionError("plan_a_inputs invariant failure:\n  - " + "\n  - ".join(errs))

    # fork_year = last shared grid year (one step before the first divergence)
    i_fork = int(np.argmin(np.abs(t - fork))) - 1
    return {"fork_year": float(t[i_fork]), "first_divergence_year": fork,
            "n_points": len(t), "t_min": float(t[0]), "t_max": float(t[-1])}


def relativize(base: dict, plan: dict, fork_year: float) -> dict:
    """Express Plan A as a multiplicative ramp over the baseline, keyed by years since the
    fork. ramp[col][k] = plan/baseline at (fork_year + k). By construction ramp[*][0] == 1
    (the fork year is shared). This is the pause-anchorable representation: to apply the
    ramp at a different fork (e.g. a rollout's pause), shift the offset axis."""
    t = base["time"]
    i0 = int(np.argmin(np.abs(t - fork_year)))
    offsets = (t[i0:] - fork_year).tolist()           # 0, 1, 2, ... years since fork
    ramp = {}
    for c in COMPUTE_COLS:
        ratio = plan[c][i0:] / base[c][i0:]
        ramp[c] = ratio.tolist()
    return {"fork_year": float(fork_year), "offsets_years": offsets, "ratio": ramp}


def build_plan_a(base: dict, ramp: dict, fork_year: float) -> dict:
    """Reconstruct a Plan-A series by applying `ramp` to `base` starting at `fork_year`.
    Before the fork the ratio is 1 (shared with baseline); after, it follows the ramp
    (log-interpolated in the multiplier, clamped to the last value past the ramp's end).
    Pass fork_year=<pause time> to slide the whole divergence to the pause."""
    t = base["time"]
    off_axis = np.asarray(ramp["offsets_years"], float)
    out = {"time": t.copy(), "L_HUMAN": base["L_HUMAN"].copy()}
    for c in COMPUTE_COLS:
        log_ratio = np.log10(np.asarray(ramp["ratio"][c], float))
        rel = (t - fork_year)
        # ratio=1 before the fork; interp (in log space) within the ramp; hold the last
        # value beyond it. np.interp already clamps to endpoints, so just floor below 0.
        lr = np.interp(rel, off_axis, log_ratio)
        lr = np.where(rel < 0.0, 0.0, lr)
        out[c] = base[c] * 10.0 ** lr
    return out


def _report(base: dict, plan: dict, info: dict) -> None:
    """Print the per-interval relative-growth table (the thing colleagues will eyeball)."""
    t = base["time"]
    fork = info["fork_year"]
    print(f"  fork year (last shared grid point): {fork:.0f}   "
          f"first divergence: {info['first_divergence_year']:.0f}")
    print(f"  {'interval':>13} | " + " | ".join(f"{c.split('_')[0]:>22}" for c in COMPUTE_COLS))
    print(f"  {'':>13} | " + " | ".join(f"{'base→planA OOM/yr (ratio)':>22}" for _ in COMPUTE_COLS))
    for i in range(len(t) - 1):
        if not (fork - 1e-9 <= t[i] <= fork + 8):    # show the fork + next ~8 yr
            continue
        cells = []
        for c in COMPUTE_COLS:
            gb = np.log10(base[c][i + 1] / base[c][i]) / (t[i + 1] - t[i])
            gp = np.log10(plan[c][i + 1] / plan[c][i]) / (t[i + 1] - t[i])
            cells.append(f"{gb:>5.2f}→{gp:>5.2f} ({plan[c][i+1]/base[c][i+1]:>5.2f}x)")
        print(f"  {t[i]:>6.0f}->{t[i+1]:<5.0f} | " + " | ".join(f"{x:>22}" for x in cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=str(BASELINE_CSV))
    ap.add_argument("--plan-a", default=str(PLAN_A_CSV))
    ap.add_argument("--write-ramp", action="store_true",
                    help="dump the relativized ramp to plan_a_covert_proj/plan_a_ramp.json")
    args = ap.parse_args()

    base = load_series(Path(args.baseline))
    plan = load_series(Path(args.plan_a))

    print(f"baseline: {args.baseline}\nplan-a:   {args.plan_a}\n")
    try:
        info = verify(base, plan)
    except AssertionError as e:
        print(f"VERIFY FAILED\n{e}", file=sys.stderr)
        return 1
    print("VERIFY OK — series share every grid point through "
          f"{IDENTICAL_THROUGH_YEAR:.0f}; Plan A forks at {info['first_divergence_year']:.0f}.\n")

    _report(base, plan, info)

    ramp = relativize(base, plan, info["fork_year"])

    # round-trip: applying the ramp at the original fork must reproduce plan-a exactly.
    rebuilt = build_plan_a(base, ramp, info["fork_year"])
    ok = all(np.allclose(rebuilt[c], plan[c], rtol=1e-9, atol=0.0) for c in COMPUTE_COLS)
    print(f"\nround-trip (relativize -> build_plan_a at the original fork == plan-a.csv): "
          f"{'OK' if ok else 'FAILED'}")
    if not ok:
        return 1

    if args.write_ramp:
        out = ROOT / "plan_a_covert_proj" / "plan_a_ramp.json"
        out.write_text(json.dumps(ramp, indent=2))
        print(f"wrote relativized ramp -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
