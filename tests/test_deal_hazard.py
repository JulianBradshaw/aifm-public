"""Tests for plan_a_covert_proj.deal_hazard — the deal-breakdown survival curve.

The sampler is an exact inverse-CDF of S(t) = exp(-H(t)) with a piecewise-linear
hazard plus an exponentially-decaying tail, so these tests check the production
functions against closed-form values (trapezoid integrals of the example table)
and against each other (round-trips), not against a reimplementation.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from plan_a_covert_proj.deal_hazard import (  # noqa: E402
    parse_hazard_spec,
    cumulative_hazard,
    survival,
    max_cumulative_hazard,
    p_never,
    sample_breakdown_time,
)

# The elicited example curve (the spreadsheet in the spec discussion).
EXAMPLE = {
    "type": "linear_plus_decay",
    "points": [[0, 0.12], [6, 0.06], [11, 0.03], [16, 0.02]],
    "decay": 0.95,
}


@pytest.fixture
def spec():
    return parse_hazard_spec(EXAMPLE)


class TestCumulativeHazard:
    def test_knot_values_match_trapezoid_integrals(self, spec):
        # H at the knots: trapezoid areas of the hazard table.
        assert cumulative_hazard(spec, 0.0) == 0.0
        assert cumulative_hazard(spec, 6.0) == pytest.approx(0.5 * (0.12 + 0.06) * 6)     # 0.54
        assert cumulative_hazard(spec, 11.0) == pytest.approx(0.54 + 0.5 * (0.06 + 0.03) * 5)
        assert cumulative_hazard(spec, 16.0) == pytest.approx(0.765 + 0.5 * (0.03 + 0.02) * 5)

    def test_mid_segment_value(self, spec):
        # First segment: h(t) = 0.12 - 0.01 t, so H(3) = 0.12*3 - 0.005*9 = 0.315.
        assert cumulative_hazard(spec, 3.0) == pytest.approx(0.315)

    def test_tail_integral_is_finite_and_closed_form(self, spec):
        # Tail: H(inf) - H(16) = h_last / (-ln decay) = 0.02 / -ln(0.95).
        expected_tail = 0.02 / (-math.log(0.95))
        assert max_cumulative_hazard(spec) == pytest.approx(0.89 + expected_tail)
        # 10 years into the tail: 0.02 * (0.95**10 - 1)/ln(0.95).
        tail_10 = 0.02 * (0.95 ** 10 - 1.0) / math.log(0.95)
        assert cumulative_hazard(spec, 26.0) == pytest.approx(0.89 + tail_10)

    def test_p_never(self, spec):
        assert p_never(spec) == pytest.approx(math.exp(-max_cumulative_hazard(spec)))
        assert 0.27 < p_never(spec) < 0.29   # ~28% with the example table

    def test_monotone_nondecreasing(self, spec):
        ts = np.linspace(0, 80, 400)
        H = np.array([cumulative_hazard(spec, float(t)) for t in ts])
        assert np.all(np.diff(H) >= -1e-12)
        S = np.array([survival(spec, float(t)) for t in ts])
        assert np.all(np.diff(S) <= 1e-12)


class TestSampling:
    def test_round_trip_survival(self, spec):
        # sample_breakdown_time inverts S: S(t(u)) == u wherever a breakdown happens.
        for u in [0.999, 0.95, 0.8, 0.6, 0.45, 0.35, 0.30, 0.281]:
            t = sample_breakdown_time(spec, u)
            assert t is not None
            assert survival(spec, t) == pytest.approx(u, rel=1e-9)

    def test_never_breaks_below_survival_floor(self, spec):
        floor = p_never(spec)
        assert sample_breakdown_time(spec, floor * 0.99) is None
        assert sample_breakdown_time(spec, 0.01) is None

    def test_monotone_in_u(self, spec):
        # Lower survival draw (rarer world) => later breakdown.
        us = np.linspace(0.999, 0.30, 50)
        ts = [sample_breakdown_time(spec, float(u)) for u in us]
        assert all(t is not None for t in ts)
        assert np.all(np.diff(ts) > 0)

    def test_empirical_distribution_matches_analytic_survival(self, spec):
        # KS-style check of the sampler against the analytic S(t) it claims to invert.
        rng = np.random.default_rng(0)
        n = 20000
        draws = [sample_breakdown_time(spec, float(u)) for u in rng.random(n)]
        never = sum(1 for d in draws if d is None) / n
        assert never == pytest.approx(p_never(spec), abs=0.01)
        finite = np.array([d for d in draws if d is not None])
        for t in [1.0, 3.0, 6.0, 9.0, 16.0, 30.0]:
            emp_surviving = 1.0 - (finite < t).sum() / n     # None draws survive everything
            assert emp_surviving == pytest.approx(survival(spec, t), abs=0.015)

    def test_u_bounds_rejected(self, spec):
        for bad in [0.0, 1.0, -0.1, 1.1]:
            with pytest.raises(ValueError):
                sample_breakdown_time(spec, bad)


class TestConditionalSampling:
    """min_survival_years conditions on the deal surviving at least that long."""

    def test_samples_respect_floor_and_invert_truncated_survival(self, spec):
        t0 = 5.0
        s0 = survival(spec, t0)
        # conditional P(never | survived 5y) ≈ 0.447 with the example table; draws below
        # that are None by design (covered by test_never_fraction_matches_conditional)
        for u in [0.999, 0.9, 0.7, 0.5, 0.45]:
            t = sample_breakdown_time(spec, u, min_years=t0)
            assert t is not None and t >= t0
            # S(t)/S(t0) == u — exact truncated inverse-CDF.
            assert survival(spec, t) / s0 == pytest.approx(u, rel=1e-9)

    def test_p_never_conditional(self, spec):
        # P(never | survived 5y) = S(inf)/S(5) > unconditional P(never).
        assert p_never(spec, 5.0) == pytest.approx(
            p_never(spec) / survival(spec, 5.0), rel=1e-12)
        assert p_never(spec, 5.0) > p_never(spec)
        assert p_never(spec, 0.0) == pytest.approx(p_never(spec))

    def test_never_fraction_matches_conditional(self, spec):
        t0 = 5.0
        rng = np.random.default_rng(1)
        n = 20000
        draws = [sample_breakdown_time(spec, float(u), min_years=t0) for u in rng.random(n)]
        assert all(d is None or d >= t0 for d in draws)
        never = sum(1 for d in draws if d is None) / n
        assert never == pytest.approx(p_never(spec, t0), abs=0.01)

    def test_zero_hazard_window_clamps_to_floor(self):
        # Hazard is 0 until year 4: H is flat there, so an unconditioned inversion of a
        # tiny target would land at t < min_years — the clamp must keep it at the floor.
        spec = parse_hazard_spec({"type": "linear_plus_decay",
                                  "points": [[0, 0.0], [4, 0.0], [5, 0.1]], "decay": 1.0})
        t = sample_breakdown_time(spec, 0.999999, min_years=2.0)
        assert t is not None and t >= 2.0

    def test_negative_min_years_rejected(self, spec):
        with pytest.raises(ValueError):
            sample_breakdown_time(spec, 0.5, min_years=-1.0)


class TestEdgeCases:
    def test_decay_one_constant_tail_always_breaks(self):
        spec = parse_hazard_spec({"type": "linear_plus_decay",
                                  "points": [[0, 0.1], [5, 0.05]], "decay": 1.0})
        assert math.isinf(max_cumulative_hazard(spec))
        assert p_never(spec) == 0.0
        t = sample_breakdown_time(spec, 0.001)   # even extreme draws break eventually
        assert t is not None
        assert survival(spec, t) == pytest.approx(0.001, rel=1e-9)

    def test_zero_terminal_hazard_caps_total(self):
        spec = parse_hazard_spec({"type": "linear_plus_decay",
                                  "points": [[0, 0.1], [5, 0.0]], "decay": 0.9})
        assert max_cumulative_hazard(spec) == pytest.approx(0.25)   # trapezoid only
        assert sample_breakdown_time(spec, math.exp(-0.25) * 0.999) is None

    def test_flat_segment(self):
        # Constant hazard within a segment exercises the b == 0 linear branch.
        spec = parse_hazard_spec({"type": "linear_plus_decay",
                                  "points": [[0, 0.1], [10, 0.1]], "decay": 0.5})
        assert cumulative_hazard(spec, 4.0) == pytest.approx(0.4)
        t = sample_breakdown_time(spec, math.exp(-0.2))
        assert t == pytest.approx(2.0)

    @pytest.mark.parametrize("bad", [
        {"type": "nope", "points": [[0, 0.1]], "decay": 0.9},
        {"type": "linear_plus_decay", "points": [], "decay": 0.9},
        {"type": "linear_plus_decay", "points": [[1, 0.1]], "decay": 0.9},        # t0 != 0
        {"type": "linear_plus_decay", "points": [[0, 0.1], [0, 0.2]], "decay": 0.9},  # not increasing
        {"type": "linear_plus_decay", "points": [[0, -0.1]], "decay": 0.9},       # negative hazard
        {"type": "linear_plus_decay", "points": [[0, 0.1]], "decay": 0.0},        # decay out of range
        {"type": "linear_plus_decay", "points": [[0, 0.1]], "decay": 1.5},
        {"type": "linear_plus_decay", "points": [[0, 0.1]]},                      # missing decay
    ])
    def test_malformed_specs_raise(self, bad):
        with pytest.raises(ValueError):
            parse_hazard_spec(bad)
