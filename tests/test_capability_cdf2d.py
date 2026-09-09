"""Tests for the capability_cdf2d exceedance field and the optional Bayesian update
re: P(a covert project exists) (plan_a_covert_proj.pipeline.figures).

Uses a tiny synthetic dataset with piecewise-linear tracks whose exceedance values are
hand-computable, and calls the PRODUCTION csv exporter end-to-end (the same field the
dither figure renders), asserting exact values.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from plan_a_covert_proj.pipeline.figures import capability_cdf2d_csv  # noqa: E402

# Record A: covert never caught, pct climbs 0 -> 100 over 2030-2040 (10 pct/yr).
# Record B: covert caught 2 years after the 2030 pause (frozen at pct 20 from 2032);
#           the deal broke down and the leading co climbs 50 -> 100 over 2035-2040.
REC_A = {
    "ac_time": 2030.0, "ted_ai_time": 2032.0, "pause_year": 2030.0, "t_det": None,
    "covert_t": [2030.0, 2040.0], "covert_pct_to_asi": [0.0, 100.0],
}
REC_B = {
    "ac_time": 2030.0, "ted_ai_time": 2032.0, "pause_year": 2030.0, "t_det": 2.0,
    "covert_t": [2030.0, 2040.0], "covert_pct_to_asi": [0.0, 100.0],
    "leading_t": [2035.0, 2040.0], "leading_pct_to_asi": [50.0, 100.0],
}


def _ds(config=None):
    return {"config": config or {}, "records": [json.loads(json.dumps(REC_A)),
                                                json.loads(json.dumps(REC_B))]}


def _field(tmp_path, ds):
    """Run the production CSV exporter and return {year -> {level -> P}}."""
    name = capability_cdf2d_csv(ds, tmp_path)
    lines = (tmp_path / name).read_text().splitlines()
    levels = [float(h.split("p_ge_")[1]) for h in lines[0].split(",")[1:]]
    out = {}
    for ln in lines[1:]:
        cells = ln.split(",")
        out[round(float(cells[0]), 6)] = dict(zip(levels, map(float, cells[1:])))
    return out


class TestUnmixedField:
    def test_hand_computed_exceedance(self, tmp_path):
        f = _field(tmp_path, _ds())
        # Before anything exists: zero everywhere.
        assert f[2028.0][0.0] == 0.0
        # 2033: A at pct 30 (>=20, <50); B frozen at 20 since its 2032 catch.
        assert f[2033.0][20.0] == 1.0
        assert f[2033.0][50.0] == 0.0
        # 2036: A at 60; B's leading co at 60 (covert frozen at 20). Both >= 50.
        assert f[2036.0][50.0] == 1.0
        assert f[2036.0][70.0] == 0.0
        # 2041: both tracks finished at 100 and persist.
        assert f[2041.0][100.0] == 1.0

    def test_detection_freezes_covert(self, tmp_path):
        # Only B (caught at pct 20, leading starts 2035): between the catch and the
        # leading track, nothing exceeds 20.
        ds = {"config": {}, "records": [json.loads(json.dumps(REC_B))]}
        f = _field(tmp_path, ds)
        assert f[2034.0][20.0] == 1.0
        assert f[2034.0][25.0] == 0.0


class TestExistsUpdate:
    def test_posterior_mixture_survivor(self, tmp_path):
        # Vantage pause+1y: B's catch (t_det=2) is AFTER the vantage, so both covert
        # worlds survive the no-detection evidence; posterior stays at the 50% prior.
        # At 2036, level 50: covert-A yes, covert-B yes (leading), no-covert-A no,
        # no-covert-B yes (leading) => (0.5+0.5+0+0.5)/2 = 0.75.
        ds = _ds({"covert_exists_prior": 0.5,
                  "deal_breakdown": {"min_survival_years": 1.0}})
        f = _field(tmp_path, ds)
        assert f[2036.0][50.0] == pytest.approx(0.75)
        # level 20 at 2033: both covert worlds yes, both no-covert worlds no => 0.5.
        assert f[2033.0][20.0] == pytest.approx(0.5)

    def test_posterior_mixture_ruled_out(self, tmp_path):
        # Vantage pause+3y: B's covert was caught at +2y, contradicting "none detected
        # by +3y" — that covert world is ruled out. Posterior = 0.5*0.5/(0.5*0.5+0.5)
        # ... weights: cov-A 0.5, cov-B 0, nocov-A 0.5, nocov-B 0.5 (total 1.5).
        # 2036 @ 50: cov-A yes + nocov-B yes = 1.0/1.5.
        ds = _ds({"covert_exists_prior": 0.5,
                  "deal_breakdown": {"min_survival_years": 3.0}})
        f = _field(tmp_path, ds)
        # (CSV stores 4 decimals, so compare at that precision)
        assert f[2036.0][50.0] == pytest.approx(2.0 / 3.0, abs=5e-5)
        # 2033 @ 20: only cov-A qualifies => 0.5/1.5.
        assert f[2033.0][20.0] == pytest.approx(1.0 / 3.0, abs=5e-5)

    def test_prior_zero_is_leading_only(self, tmp_path):
        ds = _ds({"covert_exists_prior": 0.0})
        f = _field(tmp_path, ds)
        # No covert anywhere: only B's leading co counts.
        assert f[2033.0][20.0] == 0.0
        assert f[2036.0][50.0] == pytest.approx(0.5)

    def test_prior_one_keeps_covert_conditioned(self, tmp_path):
        # prior=1 with vantage +3y: covert certainly exists but B's world (caught by
        # then) is ruled out; only A's covert world remains.
        ds = _ds({"covert_exists_prior": 1.0,
                  "deal_breakdown": {"min_survival_years": 3.0}})
        f = _field(tmp_path, ds)
        assert f[2036.0][50.0] == pytest.approx(1.0)   # A at 60
        assert f[2036.0][70.0] == pytest.approx(0.0)

    def test_explicit_arg_overrides_config(self, tmp_path):
        ds = _ds()   # no config key
        name = capability_cdf2d_csv(ds, tmp_path, exists_prior=0.0)
        lines = (tmp_path / name).read_text().splitlines()
        row_2036 = [ln for ln in lines if ln.startswith("2036.000000")][0]
        p50 = float(row_2036.split(",")[1 + [float(h.split('p_ge_')[1]) for h in lines[0].split(',')[1:]].index(50.0)])
        assert p50 == pytest.approx(0.5)   # leading-only, as prior 0 demands
