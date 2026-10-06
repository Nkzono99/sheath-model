from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from sheath_model import (EquilibriumAtlas, AtlasOptions, ContinuationOptions, FixedEntryParams,
                          FixedEntrySheathSolver, ZhaoParams, ZhaoSheathSolver, SearchOptions, SearchFailure)
from sheath_model.search import SearchDiagnostics
from sheath_model.continuation import continue_guarded_system, find_guarded_roots


class AtlasTests(unittest.TestCase):
    def test_branch_maps_correct_the_query_and_roundtrip(self):
        for branch, angle in (("A", 60.), ("B", 20.), ("C", 10.)):
            base = ZhaoParams(alpha_deg=angle, electron_drift_mode="zero", T_swi_eV=1.)
            inputs = [replace(base, n_phe_ref_cm3=value) for value in (63., 65.)]
            atlas = EquilibriumAtlas.build(inputs, branches=(branch,))
            self.assertEqual(len(atlas.points), 2)
            query = replace(base, n_phe_ref_cm3=64.)
            options = SearchOptions(method="newton", use_default_guesses=False)
            root = ZhaoSheathSolver(query, search=options).solve_unknowns(branch, atlas=atlas)
            independent = ZhaoSheathSolver(query).solve_unknowns(branch)
            self.assertAlmostEqual(root["phi0_V"], independent["phi0_V"], delta=2e-8)
            k = "ABC".index(branch)
            self.assertEqual(root["search_diagnostics"].atlas_hits[k], 1)
            self.assertEqual(root["search_diagnostics"].starts[k], 1)
            self.assertFalse(atlas.neighbors(query, "C" if branch != "C" else "A"))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"atlas.txt"
                atlas.save(path)
                loaded = EquilibriumAtlas.load(path)
                self.assertEqual(loaded.points, atlas.points)
                again = ZhaoSheathSolver(query, search=options).solve_unknowns(branch, atlas=loaded)
                self.assertAlmostEqual(again["phi0_V"], root["phi0_V"], delta=1e-12)
                candidates = ZhaoSheathSolver(query, search=replace(options, max_starts=1)).solve_candidates(
                    branch, atlas=loaded)
                self.assertEqual(len(candidates["candidates"]), 1)
                self.assertEqual(candidates["search_diagnostics"].atlas_starts[k], 1)
                self.assertEqual(candidates["search_diagnostics"].atlas_hits[k], 1)

    def test_dimensionless_map_reuse_and_empty_map_failure(self):
        p = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6, ion_temperature_ev=12.,
                             ion_pressure_factor=3.)
        atlas = EquilibriumAtlas.build([p], branches=("A",))
        reference = FixedEntrySheathSolver(p).solve_unknowns("A")
        query = replace(p, ion_density_m3=p.ion_density_m3*1e9,
                        photoelectron_density_m3=p.photoelectron_density_m3*1e9,
                        electron_temperature_ev=p.electron_temperature_ev*10,
                        photoelectron_temperature_ev=p.photoelectron_temperature_ev*10,
                        ion_temperature_ev=p.ion_temperature_ev*10,
                        ion_entry_speed_mps=p.ion_entry_speed_mps*math.sqrt(10))
        search = SearchOptions(method="newton", use_default_guesses=False, max_iterations=0)
        result = FixedEntrySheathSolver(query, search=search).solve_unknowns("A", atlas=atlas)
        self.assertAlmostEqual(result["phi0_V"]/10, reference["phi0_V"], delta=2e-8)
        with self.assertRaises(SearchFailure):
            FixedEntrySheathSolver(p, search=search).solve_unknowns("A", atlas=EquilibriumAtlas())
        bad = dict(reference, phi0_V=reference["phi0_V"]+1.)
        with self.assertRaises(ValueError):
            atlas.add(p, bad)

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
        original = EquilibriumAtlas.build([base], branches=("A",))
        query = replace(base, photoelectron_density_m3=120e6)
        reference = FixedEntrySheathSolver(query).solve_unknowns("A")
        limited = SearchOptions(method="newton", max_iterations=2, max_starts=1)
        with self.assertRaises(SearchFailure):
            FixedEntrySheathSolver(query, search=limited).solve_unknowns("A")
        for method in ("parameter", "arclength"):
            atlas = EquilibriumAtlas(original.points, continuation=ContinuationOptions(method=method))
            result = FixedEntrySheathSolver(query, search=limited).solve_unknowns("A", atlas=atlas)
            self.assertAlmostEqual(result["phi0_V"], reference["phi0_V"], delta=2e-8)
            self.assertAlmostEqual(result["phi_m_V"], reference["phi_m_V"], delta=2e-8)
            diag = result["search_diagnostics"]
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
        result = ZhaoSheathSolver(ZhaoParams(alpha_deg=60., electron_drift_mode="zero")).solve_candidates("A")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertGreater(result["search_diagnostics"].deflations[0], 0)
        p = ZhaoParams(alpha_deg=60., electron_drift_mode="zero")
        atlas = EquilibriumAtlas.build([p], branches=("A",), deflation=True)
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
