from sheath_model import SheathSolver
from sheath_model._equilibrium import EquilibriumProblem
from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.special import lambertw

from sheath_model import FixedEntryParams, ZhaoParams
from sheath_model._constants import QE


class FixedEntryTests(unittest.TestCase):
    def test_fixed_entrance_matches_projected_physical_conditions(self):
        for branch, elevation, temperature in (("A", 60., 12.), ("B", 20., 1.), ("C", 10., 1.)):
            for ion_temperature in (0., temperature):
                with self.subTest(branch=branch, ion_temperature=ion_temperature):
                    solar = ZhaoParams(sun_elevation_deg=elevation, electron_drift_mode="zero",
                                       ion_temperature_ev=ion_temperature, ion_pressure_factor=3.)
                    fixed = FixedEntryParams(
                        ion_density_m3=solar.ion_density_m3,
                        ion_entry_speed_mps=solar.ion_entry_speed_mps,
                        ion_temperature_ev=solar.ion_temperature_ev,
                        ion_pressure_factor=solar.ion_pressure_factor,
                        electron_temperature_ev=solar.electron_temperature_ev,
                        electron_drift_mps=solar.electron_drift_mps,
                        photoelectron_density_m3=solar.photoelectron_density_m3,
                        photoelectron_temperature_ev=solar.photoelectron_temperature_ev,
                        ion_mass_kg=solar.ion_mass_kg,
                    )
                    solver = EquilibriumProblem(fixed)
                    reference = SheathSolver().solve_equilibrium(solar, branch=branch)
                    actual = SheathSolver(search=solver.search).solve_equilibrium(solver.p, branch=branch)
                    for name in ("surface_potential_v", "minimum_potential_v", "ambient_electron_density_m3"):
                        self.assertAlmostEqual(getattr(actual, name), getattr(reference, name), delta=2e-07 * max(1.0, abs(getattr(reference, name))))
                    profile = SheathSolver(search=solver.search).solve_profile(solver.p, branch=branch)
                    for height in (0., float(profile.z_m[-1])):
                        state = profile.sample(height, unit='m')
                        energy = fixed.ion_mass_kg*fixed.ion_entry_speed_mps**2/(2*QE)
                        phi = state["phi_hat"]*fixed.photoelectron_temperature_ev
                        if ion_temperature:
                            pressure = fixed.ion_pressure_factor*ion_temperature
                            arg = -(2*energy/pressure)*math.exp((2*phi-2*energy)/pressure)
                            expected = math.sqrt(-2*energy/(pressure*lambertw(arg, -1).real))
                        else:
                            expected = (1-phi/energy)**-0.5
                        self.assertAlmostEqual(state["n_swi_m3"]/fixed.ion_density_m3,
                                               expected, delta=3e-13)
                        flux = profile.fluxes(height, unit='m')
                        self.assertAlmostEqual(flux["Gamma_swi_signed_m2s"],
                                               -fixed.ion_density_m3*fixed.ion_entry_speed_mps,
                                               delta=2.)
                        self.assertAlmostEqual(flux["J_net_Apm2"], 0., delta=1e-12)
                    self.assertEqual(actual.inputs.ion_entry_speed_mps, fixed.ion_entry_speed_mps)

    def test_source_and_entry_speed_are_independent(self):
        base = FixedEntryParams(ion_temperature_ev=12., ion_pressure_factor=3.,
                                photoelectron_density_m3=64e6*math.sin(math.pi/3))
        reference = SheathSolver().solve_equilibrium(base, branch='A')
        changed = replace(base, photoelectron_density_m3=base.photoelectron_density_m3*1.1)
        actual = SheathSolver().solve_equilibrium(changed, branch='A')
        self.assertEqual(actual.inputs.ion_entry_speed_mps, reference.inputs.ion_entry_speed_mps)
        self.assertGreater(abs(actual.surface_potential_v - reference.surface_potential_v), 0.001)

    def test_zero_emission_and_auto_profile(self):
        params = FixedEntryParams(photoelectron_density_m3=0., ion_temperature_ev=12.)
        solver = EquilibriumProblem(params)
        profile = SheathSolver(search=solver.search).solve_profile(solver.p, branch='C')
        self.assertLess(profile.equilibrium.surface_potential_v, 0.0)
        np.testing.assert_array_equal(profile.density.photoelectron_free_m3 / profile.equilibrium.inputs.density_scale_m3, 0.0)
        self.assertAlmostEqual(profile.fluxes(0.0, unit='hat')['J_net_Apm2'], 0.0, delta=1e-12)
        automatic = SheathSolver().solve_profile(FixedEntryParams(), branch='auto')
        self.assertEqual(automatic.equilibrium.branch, 'C')
        self.assertEqual(automatic.equilibrium.diagnostics.searched, [True, True, True])

    def test_invalid_inputs_and_no_speed_correction(self):
        for extra in ({"ion_entry_speed_mps": 0.}, {"ion_entry_speed_mps": math.nan},
                      {"ion_entry_speed_mps": 1., "ion_temperature_ev": 12.},
                      {"ion_temperature_ev": -1.}, {"ion_pressure_factor": 0.},
                      {"ion_density_m3": -1.}, {"electron_temperature_ev": 0.},
                      {"electron_drift_mps": math.inf}, {"photoelectron_density_m3": -1.}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                EquilibriumProblem(FixedEntryParams(**extra))


        # This speed passes the ion thermal condition but cannot connect the proposed A sheath.
        solver = EquilibriumProblem(FixedEntryParams(ion_entry_speed_mps=10000.0, photoelectron_density_m3=64000000.0 * math.sin(math.pi / 3)))
        with self.assertRaises(RuntimeError):
            SheathSolver(search=solver.search).solve_equilibrium(solver.p, branch='A')
        self.assertEqual(solver.p.ion_entry_speed_mps, 10e3)
