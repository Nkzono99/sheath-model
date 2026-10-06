from sheath_model import SheathSolver
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from sheath_model import EquilibriumAtlas, AtlasOptions, ContinuationOptions, FixedEntryParams, ZhaoParams, SearchOptions, SearchFailure
from sheath_model.search import SearchDiagnostics
from sheath_model.continuation import continue_guarded_system, find_guarded_roots


class AtlasTests(unittest.TestCase):
    def test_branch_maps_correct_the_query_and_roundtrip(self):
        for branch, angle in (("A", 60.), ("B", 20.), ("C", 10.)):
            base = ZhaoParams(sun_elevation_deg=angle, electron_drift_mode="zero", ion_temperature_ev=1.)
            inputs = [replace(base, photoelectron_reference_density_m3=(value)*1e6) for value in (63., 65.)]
            atlas = SheathSolver().build_equilibrium_atlas(inputs, branches=(branch,))
            self.assertEqual(len(atlas.points), 2)
            query = replace(base, photoelectron_reference_density_m3=(64.)*1e6)
            options = SearchOptions(method="newton", use_default_guesses=False)
            root = SheathSolver(search=options, equilibrium_atlas=atlas).solve_equilibrium(query, branch=branch)
            independent = SheathSolver().solve_equilibrium(query, branch=branch)
            self.assertAlmostEqual(root.surface_potential_v, independent.surface_potential_v, delta=2e-08)
            k = "ABC".index(branch)
            self.assertEqual(root.diagnostics.atlas_hits[k], 1)
            self.assertEqual(root.diagnostics.starts[k], 1)
            self.assertFalse(atlas.neighbors(query, "C" if branch != "C" else "A"))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"atlas.txt"
                atlas.save(path)
                loaded = EquilibriumAtlas.load(path)
                self.assertEqual(loaded.points, atlas.points)
                again = SheathSolver(search=options, equilibrium_atlas=loaded).solve_equilibrium(query, branch=branch)
                self.assertAlmostEqual(again.surface_potential_v, root.surface_potential_v, delta=1e-12)
                candidates = SheathSolver(search=replace(options, max_starts=1), equilibrium_atlas=loaded).solve_equilibrium_candidates(query, branch=branch)
                self.assertEqual(len(candidates.candidates), 1)
                self.assertEqual(candidates.diagnostics.atlas_starts[k], 1)
                self.assertEqual(candidates.diagnostics.atlas_hits[k], 1)

    def test_dimensionless_map_reuse_and_empty_map_failure(self):
        p = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6, ion_temperature_ev=12.,
                             ion_pressure_factor=3.)
        atlas = SheathSolver().build_equilibrium_atlas([p], branches=('A',))
        reference = SheathSolver().solve_equilibrium(p, branch='A')
        query = replace(p, ion_density_m3=p.ion_density_m3*1e9,
                        photoelectron_density_m3=p.photoelectron_density_m3*1e9,
                        electron_temperature_ev=p.electron_temperature_ev*10,
                        photoelectron_temperature_ev=p.photoelectron_temperature_ev*10,
                        ion_temperature_ev=p.ion_temperature_ev*10,
                        ion_entry_speed_mps=p.ion_entry_speed_mps*math.sqrt(10))
        search = SearchOptions(method="newton", use_default_guesses=False, max_iterations=0)
        result = SheathSolver(search=search, equilibrium_atlas=atlas).solve_equilibrium(query, branch='A')
        self.assertAlmostEqual(result.surface_potential_v / 10, reference.surface_potential_v, delta=2e-08)
        with self.assertRaises(SearchFailure):
            SheathSolver(search=search, equilibrium_atlas=EquilibriumAtlas()).solve_equilibrium(p, branch='A')
        bad = replace(reference, surface_potential_v=reference.surface_potential_v + 1.0)
        with self.assertRaises(ValueError):
            atlas.add(bad)

    def test_pseudo_arclength_passes_two_folds(self):
        def residual(y, t):
            return np.array([y[0]**3-y[0]+t]) if np.all(np.isfinite(y)) and math.isfinite(t) else None
        search = SearchOptions(method="newton")
        parameter_diag = SearchDiagnostics()
        _, success = continue_guarded_system(residual, [1.], search,
                    ContinuationOptions(initial_step=.15, max_step=.25), parameter_diag, 0)
        self.assertFalse(success)
        self.assertGreater(parameter_diag.continuation_retries[0], 0)
        arc_diag = SearchDiagnostics()
        result, success = continue_guarded_system(residual, [1.], search,
                    ContinuationOptions(method="arclength", initial_step=.15, max_step=.25), arc_diag, 0)
        self.assertTrue(success)
        self.assertAlmostEqual(result[0], -1.324717957244746, delta=2e-9)
        self.assertLess(abs(residual(result, 1.)[0]), 1e-10)
        self.assertGreater(arc_diag.continuation_steps[0], 10)

    def test_adaptive_continuation_recovers_a_sheath_with_small_corrector_budget(self):
        base = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6,
                                ion_temperature_ev=12., ion_pressure_factor=3.)
        original = SheathSolver().build_equilibrium_atlas([base], branches=('A',))
        query = replace(base, photoelectron_density_m3=120e6)
        reference = SheathSolver().solve_equilibrium(query, branch='A')
        limited = SearchOptions(method="newton", max_iterations=2, max_starts=1)
        with self.assertRaises(SearchFailure):
            SheathSolver(search=limited).solve_equilibrium(query, branch='A')
        for method in ("parameter", "arclength"):
            atlas = EquilibriumAtlas(original.points)
            result = SheathSolver(search=limited, equilibrium_atlas=atlas, continuation=ContinuationOptions(method=method)).solve_equilibrium(query, branch='A')
            self.assertAlmostEqual(result.surface_potential_v, reference.surface_potential_v, delta=2e-08)
            self.assertAlmostEqual(result.minimum_potential_v, reference.minimum_potential_v, delta=2e-08)
            diag = result.diagnostics
            self.assertGreater(diag.continuation_steps[0], 0)
            self.assertGreater(diag.continuation_retries[0], 0)
            self.assertEqual(diag.atlas_hits[0], 1)

    def test_deflation_finds_distinct_roots_and_checks_original_residual(self):
        diag = SearchDiagnostics()
        residual = lambda y: np.array([y[0]**3-y[0]])
        roots = find_guarded_roots(residual, [[.9], [.1], [-.9]], SearchOptions(method="auto"), diag, 0)
        self.assertEqual(len(roots), 3)
        self.assertTrue(np.allclose(sorted(root[0] for root in roots), [-1., 0., 1.], atol=1e-8))
        self.assertGreater(diag.deflations[0], 0)
        self.assertTrue(all(np.max(np.abs(residual(root))) <= 1e-10 for root in roots))
        result = SheathSolver().solve_equilibrium_candidates(ZhaoParams(sun_elevation_deg=60.0, electron_drift_mode='zero'), branch='A')
        self.assertEqual(len(result.candidates), 1)
        self.assertGreater(result.diagnostics.deflations[0], 0)
        p = ZhaoParams(sun_elevation_deg=60., electron_drift_mode="zero")
        atlas = SheathSolver().build_equilibrium_atlas([p], branches=('A',), deflation=True)
        self.assertEqual(len(atlas.points), 1)

    def test_invalid_controls_and_malformed_table(self):
        for extra in ({"neighbors": 0}, {"max_distance": math.nan}, {"interpolate": 1}):
            with self.assertRaises(ValueError):
                AtlasOptions(**extra)
        for extra in ({"method": "bad"}, {"max_steps": 0}, {"min_step": 1.}, {"max_root_distance": -1.}):
            with self.assertRaises(ValueError):
                ContinuationOptions(**extra)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.txt"
            path.write_text("SHEATH_EQUILIBRIUM_ATLAS 1 2\n")
            with self.assertRaises(ValueError):
                EquilibriumAtlas.load(path)
