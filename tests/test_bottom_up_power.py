"""Tests for the per-source facility-power model of the bottom-up covert sizing
(covert-AI-projects supplement §3.3, "How much datacenter capacity does the diverted
compute require?").

The supplement's parameter table (833 W/H100e at 2026.0, 1.37x/yr efficiency, smuggled
proxy year 2028.6, decommissioned proxy year 2026.5, 3.7x domestic gap, PUE 1.4) implies
that a 1 GW facility supports 1.9M smuggled / ~1M decommissioned / 525k domestic H100e.
These tests pin the production helper to those published numbers.
"""

import pytest

from plan_a_covert_proj.sim import bottom_up_facility_w_per_h100e

# The supplement's parameter estimates, as configured in covert_compute_bottom_up.yaml.
SUPPLEMENT_POWER_PARAMS = {
    "it_w_per_h100e_2026": 833,
    "power_efficiency_per_year": 1.37,
    "smuggled_proxy_offset_years": 0.1,   # window midpoint 2028.5 + 0.1 -> 2028.6
    "decommissioned_age_years": 1.5,      # window start 2028.0 - 1.5 -> 2026.5
    "domestic_efficiency_gap": 3.7,
    "peak_pue": 1.4,
}
# The supplement's fixed diversion window.
WINDOW = (2028.0, 2029.0)


class TestSupplementNumbers:
    def test_per_source_facility_watts(self):
        w = bottom_up_facility_w_per_h100e(SUPPLEMENT_POWER_PARAMS, *WINDOW)
        # 833 / 1.37^(2028.6-2026) * 1.4  and  833 / 1.37^0.5 * 1.4
        assert w["smuggled_w"] == pytest.approx(833 / 1.37 ** 2.6 * 1.4)
        assert w["decommission_w"] == pytest.approx(833 / 1.37 ** 0.5 * 1.4)
        assert w["domestic_w"] == pytest.approx(833 / 1.37 ** 2.6 * 3.7 * 1.4)

    def test_one_gw_capacities_match_published_rounding(self):
        """Supplement: 1 GW supports 1.9M smuggled / ~1M decommissioned / 525k domestic."""
        w = bottom_up_facility_w_per_h100e(SUPPLEMENT_POWER_PARAMS, *WINDOW)
        assert 1e9 / w["smuggled_w"] == pytest.approx(1.9e6, rel=0.03)
        assert 1e9 / w["decommission_w"] == pytest.approx(1.0e6, rel=0.01)
        assert 1e9 / w["domestic_w"] == pytest.approx(525e3, rel=0.001)

    def test_domestic_is_gap_times_smuggled(self):
        w = bottom_up_facility_w_per_h100e(SUPPLEMENT_POWER_PARAMS, *WINDOW)
        assert w["domestic_w"] == pytest.approx(w["smuggled_w"] * 3.7)


class TestWindowTracking:
    def test_later_window_means_less_power_per_h100e(self):
        """A pause landing years later diverts newer, more efficient chips."""
        w_2028 = bottom_up_facility_w_per_h100e(SUPPLEMENT_POWER_PARAMS, 2028.0, 2029.0)
        w_2033 = bottom_up_facility_w_per_h100e(SUPPLEMENT_POWER_PARAMS, 2033.0, 2034.0)
        for k in ("smuggled_w", "decommission_w", "domestic_w"):
            # exactly 5 years of 1.37x/yr efficiency progress
            assert w_2033[k] == pytest.approx(w_2028[k] / 1.37 ** 5)

    def test_zero_length_window(self):
        """window_start == declaration (no flow window): proxies anchor on that instant."""
        w = bottom_up_facility_w_per_h100e(SUPPLEMENT_POWER_PARAMS, 2029.0, 2029.0)
        assert w["smuggled_w"] == pytest.approx(833 / 1.37 ** (2029.1 - 2026.0) * 1.4)
        assert w["decommission_w"] == pytest.approx(833 / 1.37 ** (2027.5 - 2026.0) * 1.4)
