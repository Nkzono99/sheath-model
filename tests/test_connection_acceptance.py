import math
import unittest

import numpy as np

from sheath_model import FixedEntryParams, ZhaoParams, SheathSolver
from sheath_model._constants import QE
from sheath_model._physics import SheathPhysics
from sheath_model._orbits import electron_density


class ConnectionAcceptanceTests(unittest.TestCase):
    def test_connection_normalization_does_not_depend_on_input_convention(self):
        solar = ZhaoParams(electron_drift_mode='zero')
        fixed = FixedEntryParams(ion_density_m3=solar.ion_density_m3,
                                ion_entry_speed_mps=solar.ion_entry_speed_mps,
                                electron_temperature_ev=solar.electron_temperature_ev,
                                photoelectron_temperature_ev=solar.photoelectron_temperature_ev,
                                photoelectron_density_m3=solar.photoelectron_density_m3)
        root = SheathSolver().solve_equilibrium(solar, branch='A')
        values = [SheathPhysics(p)._type_a_connection_residual(root.surface_potential_v/2.2,
                  1.1*root.minimum_potential_v/2.2, root.ambient_electron_density_m3/p.density_scale_m3)
                  for p in (solar, fixed)]
        self.assertAlmostEqual(*values, delta=1e-13)
        self.assertGreater(abs(values[0]), 1e-3)

    def test_collapsed_maxwellian_minimum_is_not_a_connected_profile(self):
        gamma = 1989307674603.747
        me = 9.1093837015e-31
        source_density = gamma*2*math.sqrt(math.pi)/math.sqrt(2*QE*2/me)
        params = FixedEntryParams(ion_density_m3=5e6, electron_temperature_ev=10.,
                                 ion_entry_speed_mps=309496.90072670573,
                                 photoelectron_temperature_ev=2., photoelectron_density_m3=source_density)
        physics = SheathPhysics(params)
        surface, minimum = 0.22886728419952757/2, -8e-15/2
        free, reflected = electron_density(0., minimum/params.tau, 0.)
        pe = 0.5*source_density*math.exp(-surface)*(1-math.erf(math.sqrt(-minimum)))
        density_m3 = (params.ion_density_m3-pe)/(float(free)+float(reflected))
        density = density_m3/params.density_scale_m3
        normalized = physics._type_a_connection_residual(surface, minimum, density)
        # Independent leading-depth expansion, rather than an absolute E^2 test.
        leading = 2/(3*math.sqrt(math.pi))*(source_density*math.exp(-surface)
                   - density_m3/math.sqrt(params.tau))/params.ion_density_m3
        self.assertAlmostEqual(normalized, leading, delta=2e-7)
        self.assertGreater(abs(normalized), 0.4)
        self.assertLess(abs(normalized*(-minimum)**1.5), 1e-20)
        with self.assertRaisesRegex(RuntimeError, "does not connect"):
            physics._validate_profile_root('A', surface, minimum, density)
        # At this depth direct subtraction rounds all orbit densities to their
        # upstream values; the factored limit must still reject the same state.
        minimum = -1e-100/2
        pe = .5*source_density*math.exp(-surface)
        density_m3 = 2*(params.ion_density_m3-pe)
        normalized = physics._type_a_connection_residual(surface, minimum, density_m3/params.density_scale_m3)
        leading = 2/(3*math.sqrt(math.pi))*(source_density*math.exp(-surface)
                   - density_m3/math.sqrt(params.tau))/params.ion_density_m3
        self.assertAlmostEqual(normalized, leading, delta=1e-13)
        with self.assertRaisesRegex(RuntimeError, "does not connect"):
            physics._validate_profile_root('A', surface, minimum, density_m3/params.density_scale_m3)

    def test_accepted_equilibrium_has_a_non_degenerate_upper_connection(self):
        params = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6)
        root = SheathSolver().solve_equilibrium(params, branch='A')
        physics = SheathPhysics(params)
        residual = physics._type_a_connection_residual(root.surface_potential_v/2.2,
                    root.minimum_potential_v/2.2, root.ambient_electron_density_m3/params.density_scale_m3)
        self.assertLess(abs(residual), 1e-9)
        profile = SheathSolver().build_profile(root)
        self.assertTrue(np.all(np.isfinite(profile.electric_field_v_m)))
