#!/usr/bin/env python3
"""Stream a batch-rollout run directory's rollouts.jsonl and write a compact summary.json.

STDLIB ONLY, SINGLE FILE, NO REPO IMPORTS — this script is meant to be copied to (or run
directly on) the remote compute box with a bare `python3.13`, where the repo/venv may not
be importable.

rollouts.jsonl is ~100MB+ per run. It is NEVER loaded whole: the file is streamed one line
at a time and only scalar reductions are kept. The output summary.json is a few KB, which is
what the viewer (scripts/pipelines/make_mc_viewer.py) embeds.

Record schema (verified against real runs — note that scripts/analysis/rollout_summary.py
mis-parses this format, do not use it as a reference):

    {"sample_id": int,
     "parameters": {...sampled scalar params..., "ac_anchor_mode": "revenue"|"uplift-solved"|...},
     "time_series_parameters": {...},
     "results": {...} | null}

  * results is null (or not a dict) -> the rollout FAILED
  * results.aa_time is the AC (automated coder) arrival time in decimal years; it may be
    null / non-finite on a rollout that succeeded but never reached AC in the time range.

CONVENTION: quantiles are CENSORED at each rollout's simulation end (results.times[-1]) for
rollouts that ran but never reached the milestone, and EXCLUDE failed rollouts entirely. This
is what the production forecast pipeline does, so summary.json, rollout_summary.txt, the
viewer and the /forecast page all state the same statistic. A p90 at the horizon therefore
means ">=10% never get there", not "arrives then". n_failed / n_no_ac / n_with_ac are reported
separately so the censored and excluded mass stays visible.

Usage:
    python3 summarize_rollouts.py <run_dir> [more run dirs...] [--force] [--config-name NAME]
    python3 summarize_rollouts.py <root> --glob        # every <root>/*/*/ run dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

# ---------------------------------------------------------------------------

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
PARAM_QUANTILES = (0.10, 0.50, 0.90)

# Headline sampled parameters worth quantiling per anchor mode. Ones absent from a run's
# parameter set are silently skipped (configs differ run to run).
HEADLINE_PARAMS = (
    "ac_time_horizon_minutes",
    "swe_multiplier_at_present_day",
    "serial_uplift_at_reference_date",
    "excess_uplift_doubling_time",
    "automation_logistic_asymptote",
    "present_revenue_growth_rate",
    "revenue_required_for_ac",
)

DEFAULT_MODE = "time-horizon"


def _finite(x):
    """Return x as a float if it is a finite real number, else None."""
    if isinstance(x, bool) or x is None:
        return None
    if not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def quantile(sorted_vals, q):
    """Linear-interpolation quantile (numpy's default 'linear' method)."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def quantiles(vals, qs=QUANTILES):
    """{'p50': ..., ...} for the finite values in vals; {} if there are none."""
    s = sorted(vals)
    if not s:
        return {}
    return {"p%g" % (100 * q): quantile(s, q) for q in qs}


def _stats(vals):
    s = sorted(vals)
    if not s:
        return {"n": 0}
    out = {"n": len(s), "min": s[0], "max": s[-1], "mean": sum(s) / len(s)}
    out.update(quantiles(s))
    return out


# ---------------------------------------------------------------------------


def _sim_end(res):
    """The rollout's simulation end — its last simulated time point."""
    times = res.get("times")
    if isinstance(times, list) and times:
        return _finite(times[-1])
    return _finite(res.get("simulation_end"))


class ModeAcc:
    """Streaming accumulator for one ac_anchor_mode.

    CENSORING: a rollout that ran but never reached the milestone is NOT dropped — its
    simulation end is substituted before quantiling, which is what the production forecast
    pipeline does (scripts/export/generate_overlay_distributions.py, and the /forecast page
    that renders it). Keeping one convention across the repo means a p90 near the horizon
    reads as ">=10% never get there" everywhere, instead of one artifact saying 2035 and
    another saying 2049 for the same statistic.

    Rollouts that FAILED outright are excluded from the quantiles entirely (again matching
    production, which never sees them) — n_failed and n_no_ac are reported separately so the
    excluded mass stays visible rather than being buried in a percentile.
    """

    def __init__(self):
        self.n = 0
        self.n_failed = 0
        self.n_no_ac = 0
        self.n_no_sc = 0
        self.aa_times = []
        self.sc_times = []
        self.aa_censor = []      # sim ends substituted for no-arrival rollouts
        self.sc_censor = []
        self.aa_censor_unknown = 0   # no-arrival rollouts with no recoverable sim end
        self.sc_censor_unknown = 0
        self.sim_ends = []
        self.params = {}         # name -> list of floats

    def add(self, rec):
        self.n += 1
        res = rec.get("results")
        if not isinstance(res, dict):
            self.n_failed += 1
        else:
            end = _sim_end(res)
            if end is not None:
                self.sim_ends.append(end)
            aa = _finite(res.get("aa_time"))
            if aa is None:
                self.n_no_ac += 1
                if end is None:
                    self.aa_censor_unknown += 1
                else:
                    self.aa_censor.append(end)
            else:
                self.aa_times.append(aa)
            sc = _finite(res.get("ai2027_sc_time"))
            if sc is None:
                self.n_no_sc += 1
                if end is None:
                    self.sc_censor_unknown += 1
                else:
                    self.sc_censor.append(end)
            else:
                self.sc_times.append(sc)
        params = rec.get("parameters") or {}
        for name in HEADLINE_PARAMS:
            v = _finite(params.get(name))
            if v is not None:
                self.params.setdefault(name, []).append(v)

    def _censored(self, arrivals, censor, unknown, fallback_end):
        vals = list(arrivals) + list(censor)
        if unknown and fallback_end is not None:
            vals += [fallback_end] * unknown
        return vals

    def to_dict(self, fallback_end=None):
        n_ok = self.n - self.n_failed
        ac_vals = self._censored(self.aa_times, self.aa_censor,
                                 self.aa_censor_unknown, fallback_end)
        sc_vals = self._censored(self.sc_times, self.sc_censor,
                                 self.sc_censor_unknown, fallback_end)
        return {
            "n": self.n,
            "n_failed": self.n_failed,
            "n_ok": n_ok,
            "n_no_ac": self.n_no_ac,
            "n_with_ac": len(self.aa_times),
            "n_censored": len(ac_vals) - len(self.aa_times),
            "failed_pct": (100.0 * self.n_failed / self.n) if self.n else None,
            "achievement_rate_pct": (100.0 * len(self.aa_times) / len(ac_vals)) if ac_vals else None,
            "censoring": "sim_end",
            "ac_time": quantiles(ac_vals),
            "ai2027_sc_time": quantiles(sc_vals),
            "params": {
                k: quantiles(v, PARAM_QUANTILES)
                for k, v in sorted(self.params.items())
            },
        }


# ---------------------------------------------------------------------------


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _provenance(run_dir, metadata):
    """Version markers for the run — the MC analogue of the covert manifest's git_sha.

    batch_rollout.py snapshots progress_model/config.py into the run dir, so hashing that snapshot
    gives a stable "which model code produced this" fingerprint even though no git sha is
    recorded. A real git sha is used when one happens to be available.
    """
    out = {}
    for name in ("model_config_snapshot.py", "model_config_snapshot.json",
                 "input_distributions.yaml"):
        fp = os.path.join(run_dir, name)
        if not os.path.exists(fp):
            continue
        h = hashlib.sha256()
        try:
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            continue
        out[name.split(".")[0] + "_sha"] = h.hexdigest()[:12]
    if isinstance(metadata, dict):
        for key in ("git_sha", "git_commit", "commit"):
            if metadata.get(key):
                out["git_sha"] = str(metadata[key])[:12]
                break
    if "git_sha" not in out:
        fp = os.path.join(run_dir, "git_sha.txt")
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    out["git_sha"] = f.read().strip()[:12] or None
            except OSError:
                pass
    return {k: v for k, v in out.items() if v}


def _discover_config_name(run_dir, metadata):
    """Best-effort: the sampling config's basename is not recorded by batch_rollout.py,
    so look in metadata, then for a stray yaml, then a config_name.txt sidecar."""
    if isinstance(metadata, dict):
        for key in ("config", "config_path", "sampling_config", "config_name"):
            v = metadata.get(key)
            if v:
                return os.path.basename(str(v)).replace(".yaml", "").replace(".yml", "")
    sidecar = os.path.join(run_dir, "config_name.txt")
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as f:
                return f.read().strip() or None
        except Exception:
            pass
    try:
        strays = [
            f for f in sorted(os.listdir(run_dir))
            if f.endswith((".yaml", ".yml")) and f != "input_distributions.yaml"
        ]
    except OSError:
        strays = []
    if strays:
        return strays[0].rsplit(".", 1)[0]
    return None


def summarize_run(run_dir, config_name=None):
    """Stream <run_dir>/rollouts.jsonl and return the summary dict."""
    run_dir = os.path.abspath(run_dir)
    rollouts = os.path.join(run_dir, "rollouts.jsonl")
    if not os.path.exists(rollouts):
        raise FileNotFoundError(rollouts)

    modes = {}
    n_total = 0
    n_bad_lines = 0
    all_aa, all_sc = [], []

    with open(rollouts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                n_bad_lines += 1
                continue
            if not isinstance(rec, dict):
                n_bad_lines += 1
                continue
            n_total += 1
            params = rec.get("parameters") or {}
            mode = params.get("ac_anchor_mode") or DEFAULT_MODE
            mode = str(mode)
            acc = modes.get(mode)
            if acc is None:
                acc = modes[mode] = ModeAcc()
            acc.add(rec)

    metadata = _read_json(os.path.join(run_dir, "metadata.json")) or {}
    n_failed = sum(m.n_failed for m in modes.values())

    # One fallback simulation end for the whole run, used only for a no-arrival rollout whose
    # own end could not be recovered (no times array, no simulation_end).
    all_ends = sorted(e for m in modes.values() for e in m.sim_ends)
    fallback_end = quantile(all_ends, 0.5) if all_ends else None

    mode_dicts = {k: modes[k].to_dict(fallback_end) for k in sorted(modes)}

    # Run-wide series, censored the same way (concatenating the per-mode censored sets).
    all_aa = [v for m in modes.values()
              for v in m._censored(m.aa_times, m.aa_censor, m.aa_censor_unknown, fallback_end)]
    all_sc = [v for m in modes.values()
              for v in m._censored(m.sc_times, m.sc_censor, m.sc_censor_unknown, fallback_end)]
    n_with_ac = sum(len(m.aa_times) for m in modes.values())

    figures, data_files = _scan_files(run_dir)

    summary = {
        "schema_version": 2,
        "run_dir": os.path.basename(run_dir),
        "run_name": os.path.basename(os.path.dirname(run_dir)),
        "rollouts_bytes": os.path.getsize(rollouts),
        "n_total": n_total,
        "n_failed": n_failed,
        "n_ok": n_total - n_failed,
        "n_bad_lines": n_bad_lines,
        "failed_pct": (100.0 * n_failed / n_total) if n_total else None,
        # Quantiles are CENSORED at each rollout's simulation end, matching the production
        # forecast pipeline. Failed rollouts are excluded from the denominator (also matching
        # production); n_failed above is the honest count of what that leaves out.
        "censoring": "sim_end",
        "sim_end": fallback_end,
        "n_with_ac": n_with_ac,
        "n_censored": len(all_aa) - n_with_ac,
        "achievement_rate_pct": (100.0 * n_with_ac / len(all_aa)) if all_aa else None,
        "ac_time": quantiles(all_aa),
        "ac_time_stats": _stats(all_aa),
        "ai2027_sc_time": quantiles(all_sc),
        "modes": mode_dicts,
        "metadata": {
            k: metadata.get(k)
            for k in ("created_at", "num_samples", "seed", "time_range",
                      "input_data_path", "per_sample_timeout")
            if k in metadata
        },
        "config_name": config_name or _discover_config_name(run_dir, metadata),
        "provenance": _provenance(run_dir, metadata),
        "figures": figures,
        "data_files": data_files,
    }
    return summary


# Non-figure artefacts worth linking from the viewer.
_DATA_EXTS = (".csv", ".txt", ".yaml", ".yml", ".json")
_DATA_SKIP = {"summary.json", "metadata.json", "rollouts.cache.json"}


def _scan_files(run_dir):
    """Relative paths of every png (grouped later by the viewer) and small data file.

    Big jsonl payloads (rollouts.jsonl, median_trajectory_rollout.jsonl, samples.jsonl) and
    the model_config snapshots are deliberately excluded — the viewer links files, and these
    are either huge or noise.
    """
    figures, data = [], []
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        rel_dir = os.path.relpath(dirpath, run_dir)
        rel_dir = "" if rel_dir == "." else rel_dir
        for fn in sorted(filenames):
            rel = os.path.join(rel_dir, fn) if rel_dir else fn
            if fn.lower().endswith(".png"):
                figures.append(rel)
            elif fn.lower().endswith(_DATA_EXTS) and fn not in _DATA_SKIP:
                if fn.startswith("model_config_snapshot"):
                    continue
                try:
                    size = os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    size = 0
                if size <= 2_000_000:
                    data.append(rel)
    return figures, data


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run dir(s), or a root with --glob")
    ap.add_argument("--glob", action="store_true",
                    help="treat each path as an outputs root and summarize every <root>/*/*/")
    ap.add_argument("--force", action="store_true", help="rewrite an existing summary.json")
    ap.add_argument("--config-name", default=None,
                    help="record this sampling-config basename in the summary")
    args = ap.parse_args(argv)

    run_dirs = []
    if args.glob:
        for root in args.paths:
            for name in sorted(os.listdir(root)):
                d = os.path.join(root, name)
                if not os.path.isdir(d):
                    continue
                for stamp in sorted(os.listdir(d)):
                    sd = os.path.join(d, stamp)
                    if os.path.isdir(sd) and os.path.exists(os.path.join(sd, "rollouts.jsonl")):
                        run_dirs.append(sd)
    else:
        run_dirs = list(args.paths)

    if not run_dirs:
        sys.exit("no run directories found")

    rc = 0
    for run_dir in run_dirs:
        out = os.path.join(run_dir, "summary.json")
        if os.path.exists(out) and not args.force:
            print("skip (exists): %s" % out)
            continue
        try:
            summary = summarize_run(run_dir, config_name=args.config_name)
        except Exception as e:
            print("FAILED %s: %s" % (run_dir, e), file=sys.stderr)
            rc = 1
            continue
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=1, sort_keys=False)
        os.replace(tmp, out)
        print("wrote %s (%d bytes, n=%d, failed=%d, modes=%s)"
              % (out, os.path.getsize(out), summary["n_total"], summary["n_failed"],
                 ",".join(summary["modes"])))
    return rc


if __name__ == "__main__":
    sys.exit(main())
