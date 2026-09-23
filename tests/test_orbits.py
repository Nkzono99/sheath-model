import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx

from sheath_model import ZhaoParams, ZhaoSheathSolver
from sheath_model._orbits import electron_density


class OrbitTests(unittest.TestCase):
    def test_density_and_flux_against_adaptive_velocity_integrals(self):
        # A lower/upper, B, C; arbitrary accessible states, independent of root finding.
        for psi, barrier, reflected in [
            (0.25, -0.08, False),
            (-0.03, -0.08, True),
            (0.25, 0.0, False),
            (-0.3, -0.5, True),
        ]:
            for drift in (0.0, 1e-12, 0.2, -0.2, 1.5):
                cutoff = math.sqrt(psi - barrier)

                def vdf(w):
                    return math.exp(
                        -((math.sqrt(max(0, w * w - psi)) - drift) ** 2)
                    ) / math.sqrt(math.pi)

                direct = quad(vdf, cutoff, math.inf, epsabs=2e-12)[0]
                free, bounced = electron_density(psi, barrier, drift)
                self.assertAlmostEqual(float(free), direct, delta=2e-9)
                if reflected:
                    self.assertAlmostEqual(
                        float(bounced),
                        2 * quad(vdf, 0, cutoff, epsabs=2e-12)[0],
                        delta=2e-9,
                    )
                local_flux = quad(lambda w: w * vdf(w), cutoff, math.inf, epsabs=2e-12)[
                    0
                ]
                upstream_flux = quad(
                    lambda a: a * math.exp(-((a - drift) ** 2)) / math.sqrt(math.pi),
                    math.sqrt(-barrier),
                    math.inf,
                    epsabs=2e-12,
                )[0]
                self.assertAlmostEqual(local_flux, upstream_flux, delta=2e-11)

    def test_zero_drift_positive_potential_cutoff(self):
        for psi in (0.0, 1e-8, 0.25, 3.0, 30.0, 1e4):
            free, bounced = electron_density(psi, 0.0, 0.0)
            self.assertAlmostEqual(float(free), 0.5 * erfcx(math.sqrt(psi)), delta=2e-9)
            self.assertEqual(float(bounced), 0.0)

    def test_a_to_b_density_limit(self):
        solver = ZhaoSheathSolver(ZhaoParams())
        phi = np.array([0.0, 0.2, 1.0])
        b = solver._densities_hat("B", phi, 1.5, 0.12)
        a = solver._densities_hat_type_a_side(phi, 1.5, 0.12, -1e-14, "lower")
        for name in b:
            np.testing.assert_allclose(a[name], b[name], atol=1e-7, rtol=1e-6)

    def test_vdf_support_and_orbit_invariance(self):
        solver = ZhaoSheathSolver(ZhaoParams())
        # Follow one incoming orbit from infinity into positive and negative potentials.
        upstream = 0.8
        amplitudes = []
        for psi in (0.0, -0.1, 0.25):
            w = math.sqrt(upstream**2 + psi)
            cutoff = math.sqrt(psi + 0.2)
            state = dict(
                n_swe_inf_m3=1.0,
                phi_hat=psi * solver.p.tau,
                a_swe=cutoff,
                vcut_swe_mps=cutoff * solver.p.v_swe_th_mps,
                swe_reflected_active=False,
            )
            vdf = solver._swe_vdf_components(
                state, np.array([-w, 0.0]) * solver.p.v_swe_th_mps
            )
            amplitudes.append(vdf["g_free_incoming"][0])
            self.assertEqual(vdf["g_total"][1], 0.0)
        np.testing.assert_allclose(amplitudes, amplitudes[0], rtol=1e-13)

    def test_zero_drift_type_a_integral_is_regular(self):
        # Keep ion drift nonzero while electron normal drift tends to zero.
        values = []
        for alpha in (0.0, 1e-12, 1e-8, 1e-4):
            solver = ZhaoSheathSolver(
                ZhaoParams(alpha_deg=alpha, ion_drift_mode="full")
            )
            value = solver._type_a_e2_sum_at_infinity(3.0, -1.0, 8e6)
            self.assertTrue(math.isfinite(value))
            values.append(value)
        np.testing.assert_allclose(values[:3], values[0], atol=1e-9)

    def test_public_fluxes_are_position_independent(self):
        for branch, alpha in [("A", 19.0), ("B", 20.0), ("C", 10.0)]:
            solver = ZhaoSheathSolver(
                ZhaoParams(alpha_deg=alpha, electron_drift_mode="zero")
            )
            profile = solver.solve_profile(branch)
            currents, passing = [], []
            for z in np.linspace(profile["z_hat"][0], profile["z_hat"][-1], 7):
                flux = solver.fluxes_at_z(profile, float(z))
                currents.append(flux["J_net_Apm2"])
                passing.append(flux["Gamma_swe_free_incoming_m2s"])
            np.testing.assert_allclose(passing, passing[0], rtol=1e-13)
            np.testing.assert_allclose(currents, 0.0, atol=1e-14)

    def test_reject_inadmissible_algebraic_roots(self):
        with self.assertRaisesRegex(RuntimeError, "near neutral infinity"):
            ZhaoSheathSolver(ZhaoParams()).solve_unknowns("A")
        with self.assertRaisesRegex(RuntimeError, "no real connecting"):
            ZhaoSheathSolver(
                ZhaoParams(alpha_deg=1.0, electron_drift_mode="zero")
            ).solve_unknowns("C")
        root = ZhaoSheathSolver(
            ZhaoParams(alpha_deg=19.0, electron_drift_mode="zero")
        ).solve_unknowns("A")
        self.assertLess(root["phi_m_V"], root["phi0_V"])
        self.assertLess(root["phi0_V"], 0.0)

    def test_negative_upstream_field_integral_independent_of_acceptance_guard(self):
        # Fixed algebraic A root at the original 60-degree drift. Integrate its
        # upstream VDF with SciPy quad, without either production quadrature.
        p = ZhaoParams()
        phi0 = 2.6544268139403324 / p.T_phe_eV
        phim = -1.1897238115913789 / p.T_phe_eV
        density = 7923268.55824609 / p.n_phe_ref_m3

        def rho(phi):
            psi = phi / p.tau
            cutoff = math.sqrt((phi - phim) / p.tau)

            def vdf(w):
                return math.exp(-((math.sqrt(w * w - psi) - p.u) ** 2)) / math.sqrt(
                    math.pi
                )

            electrons = density * (
                quad(vdf, cutoff, math.inf, epsabs=1e-12)[0]
                + 2 * quad(vdf, 0, cutoff, epsabs=1e-12)[0]
            )
            ions = (
                p.n_swi_inf_m3
                / p.n_phe_ref_m3
                / math.sqrt(1 - 2 * phi / (p.tau * p.mach**2))
            )
            photo = (
                0.5
                * math.sin(p.alpha_rad)
                * math.exp(phi - phi0)
                * math.erfc(math.sqrt(phi - phim))
            )
            return ions - electrons - photo

        field_squared = 2 * quad(rho, -0.01, 0, epsabs=1e-13)[0]
        self.assertLess(field_squared, -1e-7)

    def test_b_upstream_obstruction_despite_positive_boundary_integral(self):
        p = ZhaoParams(alpha_deg=60.0, electron_drift_mode="zero")
        solver = ZhaoSheathSolver(p)
        phi0 = 10.0 / p.T_phe_eV
        # At infinity B has one incoming half-Maxwellian and escaping PE.
        density = 2 * (p.n_swi_inf_m3 - 0.5 * p.n_phe0_m3 * math.exp(-phi0)) / p.n_phe_ref_m3
        field_squared = 2 * solver._integrate_rho("B", "monotonic", phi0, 0.0, phi0, 0.0, density)
        self.assertGreater(field_squared, 0.0)
        with self.assertRaisesRegex(RuntimeError, "arbitrarily near upstream infinity"):
            solver._validate_profile_root("B", phi0, 0.0, density)


if __name__ == "__main__":
    unittest.main()
