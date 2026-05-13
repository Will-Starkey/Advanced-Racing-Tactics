"""
Tests for polar parsing and interpolation.
"""
import pytest
from polar.parser import PolarParser
from polar.models import PolarData


# ── Fixtures ─────────────────────────────────────────────────────

EXPEDITION_CONTENT = """\
Twa/Tws\t6\t8\t10\t12\t16
42\t5.40\t6.51\t7.04\t7.30\t7.52
52\t5.70\t6.80\t7.28\t7.51\t7.73
60\t6.11\t7.17\t7.58\t7.78\t8.01
90\t7.07\t7.92\t8.52\t9.02\t9.71
120\t6.98\t8.06\t8.71\t9.16\t10.00
150\t5.13\t6.58\t7.62\t8.35\t9.73
"""

CSV_CONTENT = """\
TWA,6,8,10,12,16
42,5.40,6.51,7.04,7.30,7.52
52,5.70,6.80,7.28,7.51,7.73
60,6.11,7.17,7.58,7.78,8.01
90,7.07,7.92,8.52,9.02,9.71
120,6.98,8.06,8.71,9.16,10.00
150,5.13,6.58,7.62,8.35,9.73
"""

JSON_CONTENT = """\
{
  "speeds": [6, 8, 10, 12, 16],
  "angles": [42, 52, 60, 90, 120, 150],
  "values": [
    [5.40, 6.51, 7.04, 7.30, 7.52],
    [5.70, 6.80, 7.28, 7.51, 7.73],
    [6.11, 7.17, 7.58, 7.78, 8.01],
    [7.07, 7.92, 8.52, 9.02, 9.71],
    [6.98, 8.06, 8.71, 9.16, 10.00],
    [5.13, 6.58, 7.62, 8.35, 9.73]
  ]
}
"""


def make_polar(content=CSV_CONTENT, filename="polar.csv", boat="TestBoat"):
    return PolarParser.parse(filename, content, boat)


# ── Parsing ───────────────────────────────────────────────────────

class TestParsing:
    def test_csv_parses_tws(self):
        p = make_polar(CSV_CONTENT, "polar.csv")
        assert p.tws_values == [6, 8, 10, 12, 16]

    def test_csv_parses_twa(self):
        p = make_polar(CSV_CONTENT, "polar.csv")
        assert 42 in p.twa_values
        assert 150 in p.twa_values

    def test_csv_exact_bsp(self):
        p = make_polar(CSV_CONTENT, "polar.csv")
        assert p.bsp_matrix[42][6] == 5.40
        assert p.bsp_matrix[90][10] == 8.52

    def test_expedition_parses(self):
        p = make_polar(EXPEDITION_CONTENT, "polar.pol")
        assert p.tws_values == [6, 8, 10, 12, 16]
        assert p.bsp_matrix[42][6] == 5.40

    def test_json_parses(self):
        p = make_polar(JSON_CONTENT, "polar.json")
        assert p.tws_values == [6, 8, 10, 12, 16]
        assert p.bsp_matrix[42][6] == 5.40

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported polar format"):
            PolarParser.parse("polar.pdf", "garbage", "TestBoat")

    def test_boat_name_stored(self):
        p = make_polar()
        assert p.boat_name == "TestBoat"


# ── Interpolation ─────────────────────────────────────────────────

class TestInterpolation:
    def setup_method(self):
        self.polar = make_polar()

    def test_exact_grid_point(self):
        bsp = self.polar.target_bsp(twa=42, tws=8)
        assert bsp == 6.51

    def test_interpolates_between_twa(self):
        # Between 42 and 52 at tws=8 → between 6.51 and 6.80
        bsp = self.polar.target_bsp(twa=47, tws=8)
        assert 6.51 < bsp < 6.80

    def test_interpolates_between_tws(self):
        # Between tws=6 and tws=8 at twa=60
        bsp = self.polar.target_bsp(twa=60, tws=7)
        assert 6.11 < bsp < 7.17

    def test_bilinear_interpolation(self):
        # Between both TWA and TWS
        bsp = self.polar.target_bsp(twa=47, tws=7)
        assert bsp > 0

    def test_clamps_below_min_twa(self):
        bsp = self.polar.target_bsp(twa=0, tws=8)
        assert bsp == self.polar.target_bsp(twa=42, tws=8)

    def test_clamps_above_max_twa(self):
        bsp = self.polar.target_bsp(twa=180, tws=8)
        assert bsp == self.polar.target_bsp(twa=150, tws=8)

    def test_performance_ratio_at_target(self):
        target_bsp = self.polar.target_bsp(42, 8)
        ratio = self.polar.performance_ratio(target_bsp, 42, 8)
        assert abs(ratio - 1.0) < 0.01

    def test_performance_ratio_below_target(self):
        target_bsp = self.polar.target_bsp(42, 8)
        ratio = self.polar.performance_ratio(target_bsp * 0.9, 42, 8)
        assert abs(ratio - 0.9) < 0.01


# ── VMG angles ────────────────────────────────────────────────────

class TestVMGAngles:
    def setup_method(self):
        self.polar = make_polar()

    def test_upwind_vmg_angle_computed(self):
        assert 6 in self.polar.upwind_vmg_angles
        assert 16 in self.polar.upwind_vmg_angles

    def test_upwind_vmg_angle_in_range(self):
        for tws, angle in self.polar.upwind_vmg_angles.items():
            assert 30 <= angle <= 70, f"Upwind VMG angle {angle}° at {tws}kts seems wrong"

    def test_downwind_vmg_angle_in_range(self):
        for tws, angle in self.polar.downwind_vmg_angles.items():
            assert 100 <= angle <= 180, f"Downwind VMG angle {angle}° at {tws}kts seems wrong"

    def test_optimal_twa_interpolates(self):
        angle = self.polar.optimal_twa(tws=9, upwind=True)
        lo = self.polar.upwind_vmg_angles[8]
        hi = self.polar.upwind_vmg_angles[10]
        assert min(lo, hi) <= angle <= max(lo, hi)

    def test_optimal_twa_exact_tws(self):
        angle = self.polar.optimal_twa(tws=8, upwind=True)
        assert angle == self.polar.upwind_vmg_angles[8]
