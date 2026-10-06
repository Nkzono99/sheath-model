from dataclasses import replace
import math
import unittest
import numpy as np

from sheath_model import (FixedEntryParams, FixedEntrySheathSolver, ZhaoParams, ZhaoSheathSolver,
                          SearchOptions, SearchFailure)
from sheath_model._constants import ME, QE
from sheath_model.search import SearchDiagnostics, solve_guarded_system


class SearchTests(unittest.TestCase):
    def test_algorithms_and_analytic_monotonic_root(self):
        for branch, angle in (("A", 60.), ("B", 20.), ("C", 10.)):
            params = ZhaoParams(alpha_deg=angle, electron_drift_mode="zero", T_swi_eV=1.)
            baseline = ZhaoSheathSolver(params).solve_unknowns(branch)
            for method in (("newton", "lm", "bracket") if branch != "A" else ("newton", "lm")):
                with self.subTest(branch=branch, method=method):
                    result = ZhaoSheathSolver(params, search=SearchOptions(method=method)).solve_unknowns(branch)
                    for key in ("phi0_V", "phi_m_V", "n_swe_inf_m3"):
                        self.assertAlmostEqual(result[key], baseline[key], delta=2e-7*max(1., abs(baseline[key])))
                    diag = result["search_diagnostics"]
                    self.assertGreater(diag.evaluations["ABC".index(branch)], 0)
                    self.assertLessEqual(diag.best_residual["ABC".index(branch)], 1e-10)
                    if method == "lm":
                        self.assertGreater(diag.lm_steps["ABC".index(branch)], 0)
            if branch == "B":
                # Eliminate density analytically for nondrifting Maxwellian electrons.
                ratio = math.sqrt(params.electron_temperature_ev/params.photoelectron_temperature_ev)
                ion = params.ion_density_m3*params.ion_entry_speed_mps*2*math.sqrt(math.pi)/params.v_phe_th_mps
                escaping = (2*params.ion_density_m3*ratio-ion)/(1+ratio)
                expected = params.photoelectron_temperature_ev*math.log(params.photoelectron_density_m3/escaping)
                self.assertAlmostEqual(baseline["phi0_V"], expected, delta=3e-9)

    def test_density_scaling_and_continuation(self):
        base = FixedEntryParams(photoelectron_density_m3=64e6*math.sin(math.pi/3),
                                ion_temperature_ev=12., ion_pressure_factor=3.)
        reference = FixedEntrySheathSolver(base).solve_unknowns("A")
        for scale in (1e-9, 1e9):
            params = replace(base, ion_density_m3=base.ion_density_m3*scale,
                             photoelectron_density_m3=base.photoelectron_density_m3*scale)
            for method in ("auto", "newton", "lm"):
                with self.subTest(scale=scale, method=method):
                    result = FixedEntrySheathSolver(params, search=SearchOptions(method=method)).solve_unknowns("A")
                    self.assertAlmostEqual(result["phi0_V"], reference["phi0_V"], delta=2e-7)
                    self.assertAlmostEqual(result["n_swe_inf_m3"]/scale, reference["n_swe_inf_m3"], delta=.1)
        changed = replace(base, photoelectron_density_m3=base.photoelectron_density_m3*1.01)
        solver = FixedEntrySheathSolver(changed)
        seed = (reference["phi0_V"], reference["phi_m_V"], reference["n_swe_inf_m3"])
        result = solver.solve_unknowns("A", seed, search=SearchOptions(method="newton", use_default_guesses=False))
        independently = solver.solve_unknowns("A")
        self.assertAlmostEqual(result["phi0_V"], independently["phi0_V"], delta=2e-8)
        self.assertEqual(result["search_diagnostics"].starts[0], 1)
        self.assertEqual(result["ion_entry_speed_mps"], base.ion_entry_speed_mps)

    def test_budget_exclusion_and_invalid_options(self):
        params = FixedEntryParams(photoelectron_density_m3=55e6)
        with self.assertRaises(SearchFailure) as caught:
            FixedEntrySheathSolver(params, search=SearchOptions(method="newton", max_iterations=0)).solve_unknowns("A")
        diag = caught.exception.diagnostics
        self.assertGreater(diag.unconverged[0], 0)
        self.assertFalse(diag.excluded[0])
        self.assertEqual(diag.roots_found[0], 0)
        with self.assertRaises(SearchFailure) as caught:
            ZhaoSheathSolver(ZhaoParams()).solve_unknowns("A")
        self.assertTrue(caught.exception.diagnostics.excluded[0])
        self.assertEqual(caught.exception.diagnostics.starts[0], 0)
        with self.assertRaises(ValueError):
            FixedEntrySheathSolver(params, search=SearchOptions(method="bracket")).solve_unknowns("A")
        with self.assertRaises(ValueError):
            FixedEntrySheathSolver(params, search=SearchOptions(method="bracket")).solve_auto()
        for extra in ({"method":"bad"}, {"residual_tolerance":math.nan}, {"max_iterations":-1},
                      {"max_starts":0}, {"potential_extent":0.}, {"bracket_points":1}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                SearchOptions(**extra)

    def test_kernel_domain_guards_and_nonzero_stationary_residual(self):
        # The unconstrained full Newton step from x=10 crosses the log domain.
        for method in ("newton", "lm", "auto"):
            diag = SearchDiagnostics()
            def residual(x):
                return np.log(x) if np.all(x > 0) else None
            x, norm, success = solve_guarded_system(residual, [10.], SearchOptions(method=method), diag, 0)
            self.assertTrue(success)
            self.assertAlmostEqual(x[0], 1., delta=1e-8)
            self.assertLessEqual(norm, 1e-10)
        diag = SearchDiagnostics()
        _, norm, success = solve_guarded_system(lambda x: np.array([1.+x[0]**2]), [0.],
                                                SearchOptions(method="lm"), diag, 0)
        self.assertFalse(success)  # least-squares minimum is not an equation root
        self.assertEqual(norm, 1.)
