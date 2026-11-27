"""Fixed-reservoir branch models against independent integrals and paper cases."""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from sheath_model import (
    BinnedEmission,
    FixedReservoir,
    KappaElectrons,
    MaxwellianElectrons,
    MaxwellianEmission,
    OuterWellSeed,
    classify_branch,
    replacement_targets,
    solve_finite_column,
    solve_presheath,
    turning_field_difference,
)
from sheath_model._constants import EPS0, QE


def wind(temperature):
    return FixedReservoir((MaxwellianElectrons(20e6, temperature, 400e3),), 20e6, 400e3, 10.0)


class BranchModelsTest(unittest.TestCase):
    def test_neutral_valley_boundary_and_flux(self):
        emission = MaxwellianEmission(36e-6/QE, 2.2)
        self.assertEqual(classify_branch(wind(4), emission).outcome, "B")
        self.assertEqual(classify_branch(wind(5), emission).outcome, "A")
        threshold = brentq(lambda te: classify_branch(wind(te), emission).ratio-1, 4, 5)
        self.assertAlmostEqual(threshold, 4.6426455, places=5)
        reservoir = wind(threshold)
        marginal = classify_branch(reservoir, emission)
        self.assertEqual(marginal.outcome, "marginal")
        self.assertAlmostEqual(marginal.effective_emission_temperature_ev, 2.2, places=12)
        depth = marginal.neutral_depth_v
        incoming = quad(lambda v: v*reservoir.phase(v), np.sqrt(depth), 40)[0]*reservoir.speed_unit
        self.assertAlmostEqual((incoming-reservoir.ion_flux_m2_s)/emission.flux_above(marginal.emission_barrier_ev),
                               1.0, places=12)
        # The local boundary cancels the emission amplitude for a Maxwellian.
        reduced = classify_branch(reservoir, MaxwellianEmission(4.5e-6/QE, 2.2))
        self.assertAlmostEqual(reduced.ratio, marginal.ratio, places=12)

    def test_non_maxwellian_inputs_and_field_identity(self):
        population = KappaElectrons(20e6, 8.0, 3.0)
        reservoir = FixedReservoir((population,), 20e6, 400e3)
        density = quad(lambda v: reservoir.phase(np.sqrt(0.2+v*v)), 0, np.inf)[0]
        self.assertAlmostEqual(reservoir.passing_density(0.2)/density, 1.0, places=12)
        emission = BinnedEmission((0.0, 2.0, 8.0, 10.0), (1e13, 2e13, 0))
        self.assertEqual(emission.flux_above(5), 1e13)
        edges = np.arange(31, dtype=float)
        tabulated = BinnedEmission(edges, -36e-6/QE*np.diff(np.exp(-edges)))
        shape = classify_branch(reservoir, tabulated)
        self.assertEqual(shape.outcome, "A")
        self.assertAlmostEqual(shape.emission_barrier_ev, 4.068797017, places=7)
        # At 4 < K < 5, the tail includes the rest of its bin and all subsequent bins.
        tail = np.exp(-5) - np.exp(-30) + (5-shape.emission_barrier_ev)*(np.exp(-4)-np.exp(-5))
        self.assertAlmostEqual(shape.effective_emission_temperature_ev, tail/(np.exp(-4)-np.exp(-5)), places=12)
        offset, depth, barrier = 1.0, 1.2, 2.3

        def integrand(t):
            photo = 2*emission.differential_flux(barrier-t)/reservoir.speed_unit
            return np.sqrt(offset-t)*(photo-reservoir.phase(np.sqrt(depth-t)))

        expected = 4*QE/EPS0 * quad(integrand, 0, offset, points=[barrier-2])[0]
        self.assertAlmostEqual(turning_field_difference(reservoir, emission, depth, barrier, offset),
                               expected, places=11)

    def test_finite_column_against_short_and_tall_reference_cases(self):
        hot = MaxwellianEmission(36e-6/QE, 2.2)
        short = solve_finite_column(wind(4), hot, 10.5, points=513)
        self.assertEqual(short.branch, "B")
        self.assertAlmostEqual(short.monotone_max_height_m, 14.555, places=2)
        self.assertTrue(np.all(short.field_v_m > 0))
        tall = solve_finite_column(wind(4), hot, 56.5, points=1025)
        self.assertEqual(tall.branch, "A")
        self.assertAlmostEqual(tall.minimum_height_m, 44.865, places=2)
        self.assertAlmostEqual(np.min(tall.potential_v), -0.149, places=3)
        self.assertAlmostEqual(tall.z_m[-1], 56.5, places=8)
        self.assertAlmostEqual(tall.potential_v[-1], 0.0, places=12)
        self.assertLess(tall.field_v_m[-1], 0)
        mono = solve_finite_column(wind(2.75), hot, 56.5, points=513)
        self.assertEqual(mono.branch, "B")
        self.assertAlmostEqual(np.interp(0.65*56.5, mono.z_m, mono.potential_v), 0.534, delta=0.005)

    def test_well_outer_matching_and_linear_replacement(self):
        reservoir = wind(4)
        emission = MaxwellianEmission(36e-6/QE, 2.2)
        energies = np.array([-1.0, 0.0, 0.25, 0.5, 1.0])
        outward, inward = replacement_targets(reservoir, energies, 0.5)
        pure_out, pure_in = replacement_targets(reservoir, energies)
        np.testing.assert_allclose(inward-outward, (pure_in-pure_out)*[0, 0, 0.5, 1, 1])
        expected = np.array([0.14867, 0.00536, 0.03918, 3.2016e12])
        seed = OuterWellSeed(*expected)
        outer = solve_presheath(reservoir, emission, "B_out", seed=seed, nodes=33, order=48, far_tau=10)
        self.assertAlmostEqual(outer.band_potential_v, -0.1433144, places=6)
        self.assertAlmostEqual(outer.minimum_potential_v, -0.1486718, places=6)
        self.assertLess(abs(outer.current_residual_a_m2), 1e-12)
        self.assertLess(outer.outer_neutrality_residual_m3, 20.0)
        self.assertLess(abs(outer.outer_potential_v[-1]), 0.002)
        self.assertTrue(np.all(np.diff(outer.sheath_z_m) > 0))
        self.assertAlmostEqual(outer.sheath_potential_v[0], outer.wall_potential_v)
        crossing = solve_presheath(wind(3.5), emission, "B_out", nodes=65, order=96, far_tau=8)
        self.assertAlmostEqual(crossing.band_potential_v, 0.0099832, places=5)
        self.assertAlmostEqual(crossing.minimum_potential_v, -0.00773512, places=6)
        self.assertAlmostEqual(crossing.outer_well_seed.minimum_tau, 0.78372, places=4)

    def test_inner_well_and_negative_wall_outer_matching(self):
        inner = solve_presheath(wind(5), MaxwellianEmission(36e-6/QE, 2.2), "A", order=48, far_tau=10)
        self.assertAlmostEqual(inner.minimum_potential_v, -0.4277, places=4)
        self.assertAlmostEqual(inner.wall_potential_v, 8.71428, places=4)
        self.assertLess(inner.minimum_potential_v, inner.band_potential_v)
        self.assertTrue(np.all(np.diff(inner.outer_potential_v) > 0))
        weak = solve_presheath(wind(4), MaxwellianEmission(0.3e-6/QE, 2.2), "C", order=48, far_tau=10)
        self.assertEqual(weak.wall_potential_v, weak.minimum_potential_v)
        self.assertTrue(np.all(weak.sheath_field_v_m < 0))
        self.assertLess(abs(weak.current_residual_a_m2), 1e-12)

    def test_no_well_outer_matching(self):
        solution = solve_presheath(wind(2.75), MaxwellianEmission(36e-6/QE, 2.2), "B_nw",
                                  order=48, far_tau=12)
        self.assertIsNone(solution.minimum_potential_v)
        self.assertAlmostEqual(solution.band_potential_v, 0.33038, places=4)
        self.assertTrue(np.all(np.diff(solution.outer_potential_v) < 0))
        self.assertLess(abs(solution.current_residual_a_m2), 1e-12)


if __name__ == "__main__":
    unittest.main()
