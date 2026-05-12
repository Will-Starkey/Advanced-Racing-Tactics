"""
PolarData — stores a boat's polar table and provides interpolation helpers.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PolarData:
    boat_name:   str
    source_file: str

    tws_values: list = field(default_factory=list)   # wind speeds (kts) in polar
    twa_values: list = field(default_factory=list)   # wind angles (°) in polar

    # bsp_matrix[twa_degrees][tws_knots] = target_bsp_knots
    bsp_matrix: dict = field(default_factory=dict)

    # Precomputed optimal VMG angles per TWS
    upwind_vmg_angles:   dict = field(default_factory=dict)   # {tws: optimal_upwind_twa}
    downwind_vmg_angles: dict = field(default_factory=dict)   # {tws: optimal_downwind_twa}

    # ── Queries ───────────────────────────────────────────────────

    def target_bsp(self, twa: float, tws: float) -> float:
        """Bilinear interpolation: target BSP at given TWA and TWS."""
        twa = min(180.0, max(0.0, abs(twa)))
        tws = max(0.0, tws)

        twa_lo, twa_hi = self._bracket(self.twa_values, twa)
        tws_lo, tws_hi = self._bracket(self.tws_values, tws)

        def g(a, w):
            return self.bsp_matrix.get(a, {}).get(w, 0.0)

        twa_frac = (
            (twa - twa_lo) / (twa_hi - twa_lo)
            if twa_hi != twa_lo else 0.0
        )
        tws_frac = (
            (tws - tws_lo) / (tws_hi - tws_lo)
            if tws_hi != tws_lo else 0.0
        )

        bsp = (
            g(twa_lo, tws_lo) * (1 - twa_frac) * (1 - tws_frac)
            + g(twa_hi, tws_lo) * twa_frac       * (1 - tws_frac)
            + g(twa_lo, tws_hi) * (1 - twa_frac) * tws_frac
            + g(twa_hi, tws_hi) * twa_frac        * tws_frac
        )
        return round(bsp, 2)

    def performance_ratio(self, actual_bsp: float, twa: float, tws: float) -> float:
        """actual_bsp / target_bsp  (1.0 = 100%)."""
        target = self.target_bsp(twa, tws)
        return round(actual_bsp / target, 3) if target > 0 else 0.0

    def optimal_twa(self, tws: float, upwind: bool = True) -> float:
        """Interpolated optimal VMG angle at given TWS."""
        angles = self.upwind_vmg_angles if upwind else self.downwind_vmg_angles
        default = 42.0 if upwind else 150.0

        tws_lo, tws_hi = self._bracket(self.tws_values, tws)
        lo = angles.get(tws_lo, default)
        hi = angles.get(tws_hi, default)

        if tws_hi == tws_lo:
            return lo
        frac = (tws - tws_lo) / (tws_hi - tws_lo)
        return round(lo + frac * (hi - lo), 1)

    # ── Build helpers ─────────────────────────────────────────────

    def compute_vmg_angles(self):
        """
        Precompute the TWA that maximises upwind / downwind VMG
        at each TWS in the polar.  Call once after loading the matrix.
        """
        for tws in self.tws_values:
            best_up_vmg,  best_up_twa  = 0.0, 42.0
            best_dn_vmg,  best_dn_twa  = 0.0, 150.0

            for twa in self.twa_values:
                bsp = self.bsp_matrix.get(twa, {}).get(tws, 0.0)
                if bsp <= 0:
                    continue

                if twa <= 90:
                    vmg = bsp * math.cos(math.radians(twa))
                    if vmg > best_up_vmg:
                        best_up_vmg, best_up_twa = vmg, twa
                else:
                    vmg = bsp * math.cos(math.radians(180 - twa))
                    if vmg > best_dn_vmg:
                        best_dn_vmg, best_dn_twa = vmg, twa

            self.upwind_vmg_angles[tws]   = best_up_twa
            self.downwind_vmg_angles[tws] = best_dn_twa

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _bracket(values: list, target: float):
        """Return (lower, upper) neighbours of target in sorted list."""
        if not values:
            return 0.0, 0.0
        lo = max((v for v in values if v <= target), default=values[0])
        hi = min((v for v in values if v >= target), default=values[-1])
        return lo, hi
