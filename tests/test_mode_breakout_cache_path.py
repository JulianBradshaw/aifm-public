"""
The mode-breakout generator's cache fast-path must be semantically identical
to the raw rollouts.jsonl path.

The generator prefers the sibling rollouts.cache.json (~2% of the raw size;
carries sampled parameters, milestone times, simulation_end) and falls back
to streaming the raw file. These tests build a small synthetic run, derive
the cache from it with the production exporter (RolloutsReader.export_cache),
and assert the two paths agree exactly — including the two known
representation quirks:

- failed rollouts are results=null in the raw file but may be results={} in
  the cache ("failed" must be `not results` on both paths);
- the raw file carries the full times array while the cache stores only the
  scalar results['simulation_end'].
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'scripts'))

from export.generate_mode_breakout_distributions import (  # noqa: E402
    read_mode_grouped_milestone_times,
    _load_cache_records,
)
from plotting_utils.rollouts_reader import RolloutsReader  # noqa: E402


def _make_record(sample_id, mode, ac_time=None, failed=False, sim_end=2050.0):
    if failed:
        return {
            "sample_id": sample_id,
            "parameters": {"ac_anchor_mode": mode},
            "time_series_parameters": {},
            "results": None,
            "error": "synthetic failure",
        }
    milestones = {}
    if ac_time is not None:
        milestones["AC"] = {"time": ac_time}
    return {
        "sample_id": sample_id,
        "parameters": {"ac_anchor_mode": mode},
        "time_series_parameters": {},
        "results": {
            "times": [2020.0, 2030.0, sim_end],
            "progress": [0.0, 5.0, 10.0],   # trajectory array: must be stripped by the cache
            "aa_time": ac_time,
            "milestones": milestones,
        },
    }


@pytest.fixture
def synthetic_run(tmp_path):
    """A small mixture run: 3 modes, achieved/not-achieved/failed rollouts."""
    records = [
        _make_record(0, "uplift-solved", ac_time=2028.5),
        _make_record(1, "uplift-solved", ac_time=2031.0),
        _make_record(2, "uplift-solved", ac_time=None),          # never achieves
        _make_record(3, "uplift-solved", failed=True),            # censored
        _make_record(4, "time-horizon", ac_time=2029.25),
        _make_record(5, "time-horizon", ac_time=2033.75),
        _make_record(6, "revenue", ac_time=2027.9),
        _make_record(7, "revenue", failed=True),
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    with rollouts.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    # Build the cache exactly the way production does
    reader = RolloutsReader(rollouts, use_cache=False)
    reader.export_cache()
    return rollouts


def test_cache_exists_and_strips_trajectories(synthetic_run):
    records = _load_cache_records(synthetic_run)
    assert records is not None and len(records) == 8
    ok = [r for r in records if r.get("results")]
    assert ok, "cache lost the successful rollouts"
    assert all("progress" not in (r.get("results") or {}) for r in records), (
        "trajectory arrays leaked into the cache")
    assert all(isinstance(r.get("parameters"), dict) for r in records), (
        "cache records lack sampled parameters")


def test_cache_and_raw_paths_agree_exactly(synthetic_run):
    cached = read_mode_grouped_milestone_times(synthetic_run, use_cache=True)
    raw = read_mode_grouped_milestone_times(synthetic_run, use_cache=False)

    times_c, not_ach_c, ok_c, fail_c, sim_end_c = cached
    times_r, not_ach_r, ok_r, fail_r, sim_end_r = raw

    assert times_c == times_r
    assert not_ach_c == not_ach_r
    assert ok_c == ok_r
    assert fail_c == fail_r
    assert sim_end_c == sim_end_r


def test_failure_and_denominator_semantics(synthetic_run):
    _, not_ach, ok, fail, sim_end = read_mode_grouped_milestone_times(
        synthetic_run, use_cache=True)
    # Failed rollouts excluded from the per-mode denominator, counted separately
    assert ok == {"uplift-solved": 3, "time-horizon": 2, "revenue": 1}
    assert fail == {"uplift-solved": 1, "revenue": 1}
    # The never-achieving successful rollout counts as not-achieved, not failed
    assert not_ach["uplift-solved"]["AC"] == 1
    assert sim_end == pytest.approx(2050.0)


def test_missing_cache_falls_back_to_raw(synthetic_run):
    (synthetic_run.parent / "rollouts.cache.json").unlink()
    result = read_mode_grouped_milestone_times(synthetic_run, use_cache=True)
    assert result[2] == {"uplift-solved": 3, "time-horizon": 2, "revenue": 1}


def test_parameterless_cache_falls_back_to_raw(synthetic_run):
    cache_path = synthetic_run.parent / "rollouts.cache.json"
    cache = json.loads(cache_path.read_text())
    for rec in cache["rollouts"]:
        rec.pop("parameters", None)
    cache_path.write_text(json.dumps(cache))
    assert _load_cache_records(synthetic_run) is None
    # And the top-level reader still produces the right grouping via raw
    result = read_mode_grouped_milestone_times(synthetic_run, use_cache=True)
    assert result[2] == {"uplift-solved": 3, "time-horizon": 2, "revenue": 1}
