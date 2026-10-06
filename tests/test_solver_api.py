from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from sheath_model import (SheathSolver, ZhaoParams, FixedEntryParams, ProfileOptions,
                          SearchOptions, SearchFailure, EquilibriumAtlas)


class SolverApiTests(unittest.TestCase):
    def test_reuse_across_physical_inputs_and_call_order(self):
        solver = SheathSolver()
        first = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6)
        second = ZhaoParams(sun_elevation_deg=10., electron_drift_mode='zero')
        a = solver.solve_equilibrium(first, branch='A')
        c = solver.solve_equilibrium(second, branch='C')
        repeated = solver.solve_equilibrium(first, branch='A')
        self.assertIs(a.inputs, first)
        self.assertIs(c.inputs, second)
        self.assertEqual(a.surface_potential_v, repeated.surface_potential_v)
        self.assertEqual(a.diagnostics, repeated.diagnostics)
        self.assertFalse(c.diagnostics.searched[0])

    def test_profile_retains_its_conditions_and_needs_no_search(self):
        params = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6)
        root = SheathSolver().solve_equilibrium(params, branch='A')
        disabled = SheathSolver(search=SearchOptions(max_iterations=0, use_default_guesses=False))
        with self.assertRaises(SearchFailure):
            disabled.solve_equilibrium(params, branch='A')
        profile = disabled.build_profile(root)
        self.assertIs(profile.equilibrium, root)
        z = float(profile.z_m[len(profile.z_m)//2])
        state = profile.sample(z)
        normalized = profile.sample(z/params.length_scale_m, unit='hat')
        self.assertAlmostEqual(state['phi_V'], normalized['phi_V'])
        local = root.density(state['phi_V'], side=state['side'])
        self.assertAlmostEqual(float(local.ion_m3), state['n_swi_m3'], delta=1e-7)
        self.assertAlmostEqual(profile.fluxes(z)['J_net_Apm2'], 0., delta=1e-12)
        with self.assertRaises(FrozenInstanceError):
            profile.equilibrium = root
        with self.assertRaises(ValueError):
            profile.potential_v[0] = 0.
        # Changing only the solved potential must fail the original equations.
        with self.assertRaises(ValueError):
            disabled.build_profile(replace(root, surface_potential_v=root.surface_potential_v+1.))
        with self.assertRaises(ValueError):
            profile.sample(float(profile.z_m[-1])+1.)

    def test_map_queries_and_failed_queries_do_not_register_roots(self):
        params = ZhaoParams(sun_elevation_deg=20., electron_drift_mode='zero')
        atlas = SheathSolver().build_equilibrium_atlas([params], branches=('B',))
        before = atlas.points
        worker = SheathSolver(search=SearchOptions(method='newton', use_default_guesses=False),
                              equilibrium_atlas=atlas)
        found = worker.solve_equilibrium(params, branch='B')
        self.assertEqual(found.diagnostics.atlas_hits[1], 1)
        self.assertEqual(atlas.points, before)
        with self.assertRaises(SearchFailure):
            worker.solve_equilibrium(ZhaoParams(), branch='A')
        self.assertEqual(atlas.points, before)
        self.assertEqual(worker.solve_equilibrium(params, branch='B').surface_potential_v,
                         found.surface_potential_v)

    def test_public_input_and_option_validation(self):
        with self.assertRaises(TypeError):
            SheathSolver().solve_equilibrium(object())
        with self.assertRaises(TypeError):
            SheathSolver(search={})
        with self.assertRaises(TypeError):
            SheathSolver(equilibrium_atlas=[])
        for extra in ({'zmax_hat': 0.}, {'n_profile_grid': True}, {'profile_phi_tol_hat': np.nan}):
            with self.assertRaises(ValueError):
                ProfileOptions(**extra)
        params = FixedEntryParams()
        with self.assertRaises(ValueError):
            SheathSolver().solve_equilibrium(params, branch='D')
        with self.assertRaises(TypeError):
            SheathSolver().solve_equilibrium(params, initial_guess=(1., 2.))
        # Independent starts are disabled; an empty map is not a physical exclusion.
        with self.assertRaises(SearchFailure) as caught:
            SheathSolver(search=SearchOptions(method='newton', use_default_guesses=False),
                          equilibrium_atlas=EquilibriumAtlas()).solve_equilibrium(params, branch='C')
        self.assertFalse(caught.exception.diagnostics.excluded[2])

    def test_mass_and_speed_scaling_preserves_the_physical_map_key(self):
        params = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6)
        atlas = SheathSolver().build_equilibrium_atlas([params], branches=('A',))
        root = SheathSolver().solve_equilibrium(params, branch='A')
        scaled = replace(params, ion_mass_kg=2*params.ion_mass_kg,
                         electron_mass_kg=2*params.electron_mass_kg,
                         ion_entry_speed_mps=params.ion_entry_speed_mps/np.sqrt(2))
        mapped = SheathSolver(search=SearchOptions(method='newton', max_iterations=0,
                                                   use_default_guesses=False),
                               equilibrium_atlas=atlas).solve_equilibrium(scaled, branch='A')
        self.assertAlmostEqual(mapped.surface_potential_v, root.surface_potential_v, delta=2e-8)
        self.assertEqual(mapped.diagnostics.atlas_hits[0], 1)
        first = SheathSolver().build_profile(root).fluxes(0.)
        second = SheathSolver().build_profile(mapped).fluxes(0.)
        self.assertAlmostEqual(second['Gamma_swi_signed_m2s']/first['Gamma_swi_signed_m2s'],
                               1/np.sqrt(2), delta=1e-14)
        self.assertAlmostEqual(second['J_net_Apm2'], 0., delta=1e-12)
