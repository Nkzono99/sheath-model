from __future__ import annotations
from sheath_model import SheathSolver
from sheath_model._equilibrium import EquilibriumProblem

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from sheath_model import ZhaoParams


class ZhaoSolverTests(unittest.TestCase):
    def test_defaults_use_surface_normal_drift(self) -> None:
        params = ZhaoParams(sun_elevation_deg=10.0)

        expected = params.solar_wind_speed_mps * math.sin(math.radians(10.0))
        self.assertEqual(params.electron_drift_mode, "normal")
        self.assertEqual(params.ion_drift_mode, "normal")
        self.assertAlmostEqual(params.electron_drift_mps, expected)
        self.assertAlmostEqual(params.ion_entry_speed_mps, expected)

    def test_swe_current_term_matches_direct_integral(self) -> None:
        params = ZhaoParams(sun_elevation_deg=60.0)
        solver = EquilibriumProblem(params)

        n_swe_inf_m3 = 7.5e6
        a_swe = 0.35

        integral, _ = quad(
            lambda x: (params.v_swe_th_mps * x + params.electron_drift_mps) * math.exp(-(x ** 2)) / math.sqrt(math.pi),
            a_swe,
            math.inf,
        )
        direct = n_swe_inf_m3 * integral / (params.v_phe_th_mps / (2.0 * math.sqrt(math.pi)))

        self.assertAlmostEqual(solver._swe_free_current_term(n_swe_inf_m3, a_swe), direct, places=7)

    def test_type_c_photoelectron_density_keeps_erfc_cutoff(self) -> None:
        params = ZhaoParams(sun_elevation_deg=10.0)
        solver = EquilibriumProblem(params)

        phi0_hat = -2.2
        phi_hat = np.array([phi0_hat, phi0_hat + 0.5, phi0_hat + 2.0])
        dens = solver._densities_hat("C", phi_hat, phi0_hat=phi0_hat, n_swe_inf_hat=0.1, phi_m_hat=phi0_hat)

        expected = 0.5 * math.sin(params.alpha_rad) * np.exp(phi_hat - phi0_hat) * erfc(np.sqrt(phi_hat - phi0_hat))
        np.testing.assert_allclose(dens["n_phe_f_hat"], expected, rtol=1e-12, atol=0.0)

    def test_nondrifting_type_c_reference_cases(self) -> None:
        out_5 = SheathSolver().solve_equilibrium(ZhaoParams(sun_elevation_deg=5.0, electron_drift_mode='zero'), branch='C')
        out_10 = SheathSolver().solve_equilibrium(ZhaoParams(sun_elevation_deg=10.0, electron_drift_mode='zero'), branch='C')

        self.assertLess(out_5.surface_potential_v / out_5.inputs.photoelectron_temperature_ev, -5.0)
        self.assertGreater(out_5.surface_potential_v / out_5.inputs.photoelectron_temperature_ev, -7.0)
        self.assertLess(out_10.surface_potential_v / out_10.inputs.photoelectron_temperature_ev, -1.5)
        self.assertGreater(out_10.surface_potential_v / out_10.inputs.photoelectron_temperature_ev, -3.0)
        self.assertGreater(out_10.ambient_electron_density_m3, 0.0)

    def test_zero_sun_elevation_normal_drift_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "degenerate"):
            EquilibriumProblem(ZhaoParams(sun_elevation_deg=0.0))


if __name__ == "__main__":
    unittest.main()
