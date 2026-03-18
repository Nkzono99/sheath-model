from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from sheath_model import ZhaoParams, ZhaoSheathSolver


class ZhaoSolverTests(unittest.TestCase):
    def test_defaults_use_surface_normal_drift(self) -> None:
        params = ZhaoParams(alpha_deg=10.0)

        expected = params.v_sw_total_mps * math.sin(math.radians(10.0))
        self.assertEqual(params.electron_drift_mode, "normal")
        self.assertEqual(params.ion_drift_mode, "normal")
        self.assertAlmostEqual(params.v_d_electron_mps, expected)
        self.assertAlmostEqual(params.v_d_ion_mps, expected)

    def test_swe_current_term_matches_direct_integral(self) -> None:
        params = ZhaoParams(alpha_deg=60.0)
        solver = ZhaoSheathSolver(params)

        n_swe_inf_m3 = 7.5e6
        a_swe = 0.35

        integral, _ = quad(
            lambda x: (params.v_swe_th_mps * x + params.v_d_electron_mps) * math.exp(-(x ** 2)) / math.sqrt(math.pi),
            a_swe,
            math.inf,
        )
        direct = n_swe_inf_m3 * integral / (params.v_phe_th_mps / (2.0 * math.sqrt(math.pi)))

        self.assertAlmostEqual(solver._swe_free_current_term(n_swe_inf_m3, a_swe), direct, places=7)

    def test_type_c_photoelectron_density_keeps_erfc_cutoff(self) -> None:
        params = ZhaoParams(alpha_deg=10.0)
        solver = ZhaoSheathSolver(params)

        phi0_hat = -2.2
        phi_hat = np.array([phi0_hat, phi0_hat + 0.5, phi0_hat + 2.0])
        dens = solver._densities_hat("C", phi_hat, phi0_hat=phi0_hat, n_swe_inf_hat=0.1, phi_m_hat=phi0_hat)

        expected = 0.5 * math.sin(params.alpha_rad) * np.exp(phi_hat - phi0_hat) * erfc(np.sqrt(phi_hat - phi0_hat))
        np.testing.assert_allclose(dens["n_phe_f_hat"], expected, rtol=1e-12, atol=0.0)

    def test_type_c_reference_cases_follow_paper_trend(self) -> None:
        out_5 = ZhaoSheathSolver(ZhaoParams(alpha_deg=5.0)).solve_unknowns("C")
        out_10 = ZhaoSheathSolver(ZhaoParams(alpha_deg=10.0)).solve_unknowns("C")

        self.assertLess(out_5["phi0_hat"], -5.0)
        self.assertGreater(out_5["phi0_hat"], -7.0)
        self.assertLess(out_10["phi0_hat"], -1.5)
        self.assertGreater(out_10["phi0_hat"], -3.0)
        self.assertLess(out_10["n_swe_inf_m3"], ZhaoParams(alpha_deg=10.0).n_swi_inf_m3)


if __name__ == "__main__":
    unittest.main()
