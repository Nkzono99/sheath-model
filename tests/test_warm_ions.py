import math
import unittest

import numpy as np
from scipy.special import lambertw

from sheath_model import ZhaoParams, ZhaoSheathSolver, ion_density_ratio, ion_critical_potential


class WarmIonTests(unittest.TestCase):
    def test_independent_lambert_solution_and_root_selection(self):
        # Independent special-function solution on W_-1; W_0 is a different fluid root.
        for pressure in (0.5, 5.0, 15.0):
            energy = 20.0
            phi = np.array([-20., -2., 0., 0.1, 0.99*ion_critical_potential(energy, pressure)])
            argument = -(2*energy/pressure)*np.exp((2*phi - 2*energy)/pressure)
            reference = np.sqrt(-2*energy/(pressure*lambertw(argument, -1).real))
            actual = ion_density_ratio(phi, energy, pressure)
            np.testing.assert_allclose(actual, reference, rtol=3e-13)
            self.assertEqual(actual[2], 1.0)
            self.assertTrue(np.all(actual < math.sqrt(2*energy/pressure)))

    def test_cold_limit_and_pressure_force(self):
        phi = np.array([-20., -1., 0., 1., 10.])
        cold = (1 - phi/20.)**-0.5
        np.testing.assert_allclose(ion_density_ratio(phi, 20.), cold, rtol=1e-15)
        np.testing.assert_allclose(ion_density_ratio(phi, 20., 1e-8), cold, rtol=1e-9)
        # Independently integrate the differential momentum/continuity equations.
        from scipy.integrate import solve_ivp
        for endpoint in (-20., 1.0):
            solution = solve_ivp(lambda _, n: n**3/(40. - 5.*n**2), (0., endpoint), [1.],
                                 rtol=2e-11, atol=2e-13)
            self.assertTrue(solution.success)
            self.assertAlmostEqual(float(ion_density_ratio(endpoint, 20., 5.)),
                                   solution.y[0, -1], delta=3e-11)

    def test_sonic_limit_and_blocked_states(self):
        critical = ion_critical_potential(20., 5.)
        self.assertEqual(float(ion_density_ratio(critical, 20., 5.)), math.sqrt(8.))
        for energy, pressure, phi in ((20., 5., critical + 1e-8), (20., 0., 20.),
                                     (20., 40., 0.), (-1., 0., 0.), (20., -1., 0.),
                                     (20., 5., math.nan)):
            with self.assertRaises(ValueError):
                ion_density_ratio(phi, energy, pressure)
        gap = 1e-7
        self.assertAlmostEqual(ion_critical_potential(0.5*(1+gap), 1.),
                               0.5*(gap**2/2-gap**3/3), delta=3e-24)

    def test_paper_reference_bohm_critical_potential(self):
        # Eq. (37), T_e=15 eV, T_i/T_e=10: approximately 0.4 V and 0.1 V.
        for factor in (1., 3.):
            pressure = factor*150.
            expected = 7.5*(1-factor*10*math.log1p(1/(factor*10)))
            self.assertAlmostEqual(ion_critical_potential(0.5*(15+pressure), pressure),
                                   expected, delta=2e-13)

    def test_public_solver_density_current_and_local_speed(self):
        params = ZhaoParams(electron_drift_mode="zero", T_swi_eV=12., ion_pressure_factor=3.,
                            n_type_a_grid=800)
        solver = ZhaoSheathSolver(params)
        profile = solver.solve_profile("A")
        heights = profile["z_m_array_m"]
        for height in (0., float(heights[len(heights)//2])):
            state = solver.sample_at_z(profile, height, unit="m")
            self.assertAlmostEqual(state["n_swi_m3"]*state["v_i_mps"]/(params.ion_density_m3*params.ion_entry_speed_mps),
                                   1., delta=3e-15)
            self.assertAlmostEqual(solver.fluxes_at_z(profile, height, unit="m")["J_net_Apm2"],
                                   0., delta=1e-12)
        cold = ZhaoSheathSolver(ZhaoParams(electron_drift_mode="zero")).solve_unknowns("A")
        self.assertGreater(abs(float(profile["phi_m_V"])-float(cold["phi_m_V"])), 1e-6)
        with self.assertRaisesRegex(ValueError, "does not define an ion VDF"):
            solver.vdf_1d_at_z(profile, 0., species="swi", unit="m")

    def test_invalid_temperature_and_entry_speed(self):
        for extra in ({"T_swi_eV": -1.}, {"T_swi_eV": math.nan}, {"ion_pressure_factor": 0.},
                      {"alpha_deg": 1., "T_swi_eV": 100.}):
            with self.assertRaises(ValueError):
                ZhaoSheathSolver(ZhaoParams(**extra))

    def test_warm_monotonic_profiles(self):
        for branch, elevation in (("B", 20.), ("C", 10.)):
            params = ZhaoParams(alpha_deg=elevation, electron_drift_mode="zero", T_swi_eV=1.,
                                ion_pressure_factor=3.)
            solver = ZhaoSheathSolver(params)
            profile = solver.solve_profile(branch)
            for height in (0., float(profile["z_m_array_m"][-1])):
                flux = solver.fluxes_at_z(profile, height, unit="m")
                self.assertAlmostEqual(flux["J_net_Apm2"], 0., delta=1e-12)
                self.assertAlmostEqual(flux["Gamma_swi_signed_m2s"]/(params.ion_density_m3*params.ion_entry_speed_mps),
                                       -1., delta=3e-15)
