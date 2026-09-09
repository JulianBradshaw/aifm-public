#!/usr/bin/env python3
"""
Generate a summary of rollout success/failure statistics.

Analyzes a run's rollouts.jsonl and writes rollout_summary.txt into the run directory.
batch_rollout.py / post_rollout.py invoke this at the end of every run.

Record schema (see scripts/pipelines/summarize_rollouts.py for the canonical description):

    {"sample_id", "parameters", "time_series_parameters", "results": {...}|null,
     "error"?: str, "traceback"?: str}

  * results is null (and "error" carries the reason) -> the rollout FAILED
  * results.aa_time null/non-finite -> the rollout ran but never reached AC ("no AC")

The per-mode counts and quantiles come from summarize_rollouts.ModeAcc, the same accumulator
that builds summary.json for the viewer, so this report and summary.json cannot disagree.
(summarize_rollouts stays stdlib-only and remote-runnable; the dependency is one-way.)

Usage:
    python scripts/analysis/rollout_summary.py <run_dir>
    python scripts/analysis/rollout_summary.py <run_dir> --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))

# Import RolloutsReader for unified rollout parsing
from plotting_utils.rollouts_reader import RolloutsReader

# Shared statistics core — identical semantics to the viewer's summary.json.
from summarize_rollouts import DEFAULT_MODE, ModeAcc, _finite, quantile, quantiles

# Error-message patterns -> human-readable category, checked in order.
ERROR_PATTERNS = (
    ("timeout", "Timeout"),
    ("timed out", "Timeout"),
    ("could not find a logistic schedule", "Uplift Schedule Unsolvable"),
    ("validation", "Validation Error"),
    ("infeasible", "Infeasible Parameters"),
)


def categorize_error(error_msg: str) -> str:
    low = (error_msg or "").lower()
    for needle, label in ERROR_PATTERNS:
        if needle in low:
            return label
    return "Other Error"


def analyze_rollouts(rollouts_path: Path) -> Dict[str, Any]:
    """Analyze a rollouts.jsonl file and return statistics."""
    total = 0
    successful = 0
    failed = 0
    timeouts = 0
    error_types: Dict[str, int] = {}
    error_details: List[Dict[str, Any]] = []
    modes: Dict[str, ModeAcc] = {}
    all_aa: List[float] = []

    reader = RolloutsReader(rollouts_path)

    for record in reader.iter_all_records():
        if not isinstance(record, dict):
            continue
        total += 1

        # build_cache() rewrites a failed rollout's results from null to {}, so when the
        # reader falls back to cached records a failure would otherwise look successful.
        # Normalize before counting so the totals and the per-mode table always agree.
        if record.get("error") and not record.get("results"):
            record = dict(record, results=None)

        params = record.get("parameters") or {}
        mode = str(params.get("ac_anchor_mode") or DEFAULT_MODE)
        modes.setdefault(mode, ModeAcc()).add(record)

        results = record.get("results")
        if isinstance(results, dict):
            successful += 1
            aa = _finite(results.get("aa_time"))
            if aa is not None:
                all_aa.append(aa)
            continue

        # Failed: batch_rollout records the reason in "error" alongside results=null.
        failed += 1
        error_msg = record.get("error") or "Results field is None or invalid"
        error_type = categorize_error(error_msg) if record.get("error") else "Missing Results"
        if error_type == "Timeout":
            timeouts += 1
        error_types[error_type] = error_types.get(error_type, 0) + 1
        error_details.append({
            "sample_id": record.get("sample_id", total - 1),
            "ac_anchor_mode": mode,
            "error_type": error_type,
            "error_message": str(error_msg)[:200],
        })

    # Censor at each rollout's simulation end (see summarize_rollouts.ModeAcc) so this report
    # states the same statistic as summary.json, the viewer and the /forecast page.
    all_ends = sorted(e for m in modes.values() for e in m.sim_ends)
    fallback_end = quantile(all_ends, 0.5) if all_ends else None
    mode_stats = {k: modes[k].to_dict(fallback_end) for k in sorted(modes)}
    no_ac = sum(m["n_no_ac"] for m in mode_stats.values())
    censored_aa = [v for m in modes.values()
                   for v in m._censored(m.aa_times, m.aa_censor,
                                        m.aa_censor_unknown, fallback_end)]

    return {
        "total_rollouts": total,
        "successful_rollouts": successful,
        "failed_rollouts": failed,
        "timeout_rollouts": timeouts,
        "no_ac_rollouts": no_ac,
        "reached_ac_rollouts": len(all_aa),
        "censoring": "sim_end",
        "sim_end": fallback_end,
        "success_rate": successful / total if total else 0.0,
        "failure_rate": failed / total if total else 0.0,
        "ac_time": quantiles(censored_aa),
        "modes": mode_stats,
        "error_types": error_types,
        "error_details": error_details,
    }


def fmt_end(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def _fmt_q(q: Dict[str, float], key: str) -> str:
    v = q.get(key)
    return f"{v:8.2f}" if v is not None else "       -"


def format_summary(stats: Dict[str, Any]) -> str:
    total = stats["total_rollouts"]

    def pct(n):
        return n / total if total else 0.0

    lines = [
        "=" * 72,
        "ROLLOUT SUMMARY",
        "=" * 72,
        "",
        f"Total Rollouts:      {total:5d}",
        f"Successful:          {stats['successful_rollouts']:5d} ({stats['success_rate']:7.2%})",
        f"Failed:              {stats['failed_rollouts']:5d} ({stats['failure_rate']:7.2%})",
        f"Reached AC:          {stats['reached_ac_rollouts']:5d} ({pct(stats['reached_ac_rollouts']):7.2%})",
        f"Ran but no AC:       {stats['no_ac_rollouts']:5d} ({pct(stats['no_ac_rollouts']):7.2%})",
        "",
    ]

    modes = stats.get("modes") or {}
    if modes:
        lines += [
            "BY ac_anchor_mode:",
            "-" * 72,
            f"  {'mode':22s} {'n':>5s} {'ok':>5s} {'fail':>5s} {'noAC':>5s} "
            f"{'AC p10':>8s} {'AC p50':>8s} {'AC p90':>8s}",
        ]
        for name, m in modes.items():
            q = m.get("ac_time") or {}
            lines.append(
                f"  {name:22s} {m['n']:5d} {m['n_ok']:5d} {m['n_failed']:5d} {m['n_no_ac']:5d} "
                f"{_fmt_q(q, 'p10')} {_fmt_q(q, 'p50')} {_fmt_q(q, 'p90')}"
            )
        lines.append("")

    qs = ("p5", "p10", "p25", "p50", "p75", "p90", "p95")
    rows = [("all", stats.get("ac_time") or {},
             stats["reached_ac_rollouts"] + stats["no_ac_rollouts"])]
    rows += [(name, m.get("ac_time") or {}, m["n_with_ac"] + m["n_censored"])
             for name, m in modes.items()]
    rows = [r for r in rows if r[1]]
    if rows:
        lines += [
            "AC ARRIVAL QUANTILES (censored at simulation end; failed rollouts excluded):",
            "-" * 72,
            "  n counts rollouts that ran to completion. Those that never reach AC are",
            f"  substituted at the simulation end ({fmt_end(stats.get('sim_end'))}), so a",
            "  p90 there means '>=10% never get there', not 'arrives then'.",
            f"  {'series':22s} {'n':>5s} " + " ".join(f"{q:>8s}" for q in qs),
        ]
        for name, q, n in rows:
            lines.append(f"  {name:22s} {n:5d} " + " ".join(_fmt_q(q, k) for k in qs))
        lines.append("")

    if stats["error_types"]:
        lines += ["FAILURE BREAKDOWN:", "-" * 72]
        for error_type, count in sorted(stats["error_types"].items(), key=lambda x: -x[1]):
            lines.append(f"  {error_type:34s} {count:5d} ({pct(count):7.2%})")
        lines.append("")

    if stats["error_details"]:
        shown = min(10, len(stats["error_details"]))
        lines += [f"FIRST {shown} ERRORS:", "-" * 72]
        for i, error in enumerate(stats["error_details"][:10], start=1):
            lines.append(f"{i}. Sample ID {error['sample_id']} "
                         f"[{error.get('ac_anchor_mode', '?')}]: {error['error_type']}")
            lines.append(f"   {error['error_message']}")
            lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate rollout summary statistics")
    parser.add_argument("run_dir", type=str, help="Path to run directory containing rollouts.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: run_dir/rollout_summary.txt)")
    parser.add_argument("--json", action="store_true", help="Also output JSON format summary")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rollouts_path = run_dir / "rollouts.jsonl"

    if not rollouts_path.exists():
        print(f"Error: rollouts.jsonl not found at {rollouts_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {rollouts_path}...")
    stats = analyze_rollouts(rollouts_path)

    summary_text = format_summary(stats)
    print(summary_text)

    output_path = Path(args.output) if args.output else run_dir / "rollout_summary.txt"
    output_path.write_text(summary_text)
    print(f"\nSummary written to {output_path}")

    if args.json:
        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"JSON summary written to {json_path}")


if __name__ == "__main__":
    main()
