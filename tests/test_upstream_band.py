"""Drifting reflected electrons: A/C are rejected by default and accepted with a measured upstream band."""
from __future__ import annotations

import math
import unittest

from sheath_model import FixedEntryParams, SearchFailure, SearchOptions, SheathSolver

QE, ME = 1.602176634e-19, 9.1093837139e-31
# Regolith test conditions: n=5 cm^-3, Te=10 eV, 400 km/s normal wind, 4.5 uA/m^2 photoelectrons at 2.2 eV.
SOURCE_DENSITY = 2.0 * math.sqrt(math.pi) * (4.5e-6 / QE) / math.sqrt(2.0 * QE * 2.2 / ME)


def params(drift: float, photoelectrons: bool = True) -> FixedEntryParams:
    return FixedEntryParams(ion_density_m3=5.0e6, electron_temperature_ev=10.0, electron_drift_mps=drift,
                            ion_entry_speed_mps=4.0e5,
                            photoelectron_density_m3=SOURCE_DENSITY if photoelectrons else 0.0)


class UpstreamBandTests(unittest.TestCase):
    strict = SheathSolver()
    tolerant = SheathSolver(search=SearchOptions(upstream_band_tolerance=0.15))

    def test_drifting_type_a_needs_tolerance(self) -> None:
        with self.assertRaises(SearchFailure):
            self.strict.solve_equilibrium(params(4.0e5), branch="A")
        with self.assertRaises(SearchFailure):
            SheathSolver(search=SearchOptions(upstream_band_tolerance=0.02)).solve_equilibrium(params(4.0e5), branch="A")

        root = self.tolerant.solve_equilibrium(params(4.0e5), branch="A")
        # Algebraic root solved without the semi-infinite upstream check (independent of this acceptance path).
        self.assertAlmostEqual(root.surface_potential_v, 6.0383, delta=1e-3)
        self.assertAlmostEqual(root.minimum_potential_v, -0.7862, delta=1e-3)
        self.assertGreater(root.upstream_negative_band_v, 0.040)
        self.assertLess(root.upstream_negative_band_v, 0.065)
        self.assertEqual(self.tolerant.solve_equilibrium(params(4.0e5)).branch, "A")

        profile = self.tolerant.build_profile(root)
        self.assertLess(float(profile.potential_v[-1]), -root.upstream_negative_band_v)

    def test_drifting_type_c_band_is_shallow(self) -> None:
        with self.assertRaises(SearchFailure):
            self.strict.solve_equilibrium(params(4.0e5, photoelectrons=False), branch="C")
        root = self.tolerant.solve_equilibrium(params(4.0e5, photoelectrons=False), branch="C")
        self.assertAlmostEqual(root.surface_potential_v, -7.3403, delta=1e-3)
        self.assertGreater(root.upstream_negative_band_v, 0.0)
        self.assertLess(root.upstream_negative_band_v, 0.005)

    def test_tolerance_does_not_change_zero_drift_roots(self) -> None:
        reference = self.strict.solve_equilibrium(params(0.0), branch="A")
        root = self.tolerant.solve_equilibrium(params(0.0), branch="A")
        self.assertEqual(root.surface_potential_v, reference.surface_potential_v)
        self.assertEqual(root.upstream_negative_band_v, 0.0)
        with self.assertRaises(ValueError):
            SearchOptions(upstream_band_tolerance=1.0)


if __name__ == "__main__":
    unittest.main()
