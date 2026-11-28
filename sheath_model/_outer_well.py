"""Quasineutral collocation between an outer valley and the sheath edge."""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root


@dataclass(frozen=True)
class OuterWellSeed:
    """Continuation seed for the surface side of an outer minimum."""

    minimum_depth_v: float
    band_above_minimum_v: float
    minimum_tau: float
    escaping_flux_m2_s: float
    turning_path: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class TurningRegion:
    seed: OuterWellSeed
    amplitude: float
    potential_v: np.ndarray
    tau: np.ndarray
    current: float
    neutrality_error: float
    path: Callable

    def returning_phase(self, orbits, energy):
        depth = self.seed.minimum_depth_v
        gap = self.seed.band_above_minimum_v
        turn = self.path(np.sqrt(np.clip((depth-energy)/gap, 0, 1)))
        outward, inward = _targets(orbits, energy, depth)
        return returning(orbits.photo(energy, depth, self.amplitude), outward, inward, turn, 0.0)


def replacement_targets(reservoir, energy_ev, outer_depth_v=0.0):
    """Outward and inward replacement targets for surface-side turning orbits.

    Outputs are reduced distributions per velocity measured in sqrt(eV).
    Negative energy uses the far-wind zero-speed value.
    """
    if not math.isfinite(outer_depth_v) or outer_depth_v < 0:
        raise ValueError("outer minimum depth must be finite and nonnegative")
    energy = np.maximum(np.asarray(energy_ev), 0.0)
    speed = np.sqrt(energy)
    outward = reservoir.phase(-speed)
    inward = reservoir.phase(speed)
    if outer_depth_v > 0:
        exchange = 0.5 * np.clip(1 - energy / outer_depth_v, 0, 1) * (inward - outward)
        return outward + exchange, inward - exchange
    return outward, inward


def relax(initial, target, distance):
    return initial + (target - initial) * -np.expm1(-distance)


def returning(initial, out_target, in_target, turning_tau, tau):
    at_turn = relax(initial, out_target, turning_tau)
    return relax(at_turn, in_target, turning_tau - tau)


def _targets(orbits, energy, depth):
    outward, inward = replacement_targets(orbits.reservoir, energy, depth)
    return outward / orbits.density, inward / orbits.density


def _cosine_mesh(nodes):
    return (1 - np.cos(np.linspace(0, np.pi, nodes))) / 2


class _TurningGrid:
    """Cluster potential nodes at the endpoints and split at 0 V when crossed."""

    def __init__(self, nodes, split):
        self.split = split
        split_at = 2*nodes//3
        self.mesh = _cosine_mesh(nodes)
        self.left = _cosine_mesh(split_at+1)
        self.right = _cosine_mesh(nodes-split_at)

    def points(self, depth, gap):
        if not self.split:
            return self.mesh
        cross = math.sqrt(depth / gap)
        return np.r_[cross*self.left, cross + (1-cross)*self.right[1:]**2]

    def spline(self, depth, gap, tau, inspect=False):
        if not self.split:
            path = CubicSpline(self.mesh, tau)
            extrema = path.derivative().roots(extrapolate=False) if inspect else ()
            return path, path(1.0, 1), extrema
        cross = math.sqrt(depth / gap)
        left = CubicSpline(self.left, tau[:len(self.left)])
        right = CubicSpline(self.right, tau[len(self.left)-1:])

        def path(s):
            s = np.asarray(s)
            left_s = np.minimum(s/cross, 1)
            right_s = np.sqrt(np.maximum((s-cross)/(1-cross), 0))
            return np.where(s <= cross, left(left_s), right(right_s))

        extrema = ()
        if inspect:
            left_roots = left.derivative().roots(extrapolate=False)
            right_roots = right.derivative().roots(extrapolate=False)
            extrema = np.r_[cross*left_roots, cross, cross + (1-cross)*right_roots**2]
        return path, right(1.0, 1)/(2*(1-cross)), extrema


def _moments(orbits, path, s, depth, gap, amplitude):
    """Net electron excess and normalized wall current for a trial turning path."""
    tau, psi = path(s), depth - gap*s*s
    cutoff_speed = math.sqrt(gap)*s
    photo_pass = orbits.photo_density(-psi, amplitude, depth)
    passing = orbits.passing(psi, depth, 1) + relax(photo_pass, orbits.passing(psi, depth, -1), tau)
    # Split the energy integral at the wind potential, where the target changes law.
    cut = np.arcsin(np.clip(np.sqrt(np.maximum(-psi, 0)) / np.maximum(cutoff_speed, 1e-100), 0, 1))
    turning_density = np.zeros_like(s)
    back_flux = np.zeros_like(s)
    for lo, hi in ((np.zeros_like(s), cut), (cut, np.full_like(s, np.pi/2))):
        angle = lo[:, None] + (hi-lo)[:, None]*orbits.x
        v = cutoff_speed[:, None]*np.sin(angle)
        energy = psi[:, None] + v*v
        weight = cutoff_speed[:, None]*np.cos(angle)*(hi-lo)[:, None]*orbits.w
        turn = path(s[:, None]*np.cos(angle))
        out_target, in_target = _targets(orbits, energy, depth)
        photo = orbits.photo(energy, depth, amplitude)
        outward = relax(photo, out_target, tau[:, None])
        incoming = returning(photo, out_target, in_target, turn, tau[:, None])
        turning_density += np.sum(weight*(outward + incoming), axis=-1)
        back_flux += np.sum(weight*v*incoming, axis=-1)
    t = orbits.temperature
    current = orbits.influx(depth) + back_flux[-1] - amplitude*t/2*np.exp(gap/t) - orbits.ion_flux
    return passing + turning_density - orbits.ions(-psi), current


def solve_turning_region(orbits, seed, nodes):
    """Solve neutrality, the edge fold and wall current for the turning path."""
    t = orbits.temperature
    d0, gap0 = seed.minimum_depth_v, seed.band_above_minimum_v
    grid = _TurningGrid(nodes, gap0 > d0)

    def residual(pars):
        if (not np.all(np.isfinite(pars)) or np.max(np.abs(pars[:-3])) > 40
                or np.any(np.abs(pars[-3:]) > 27)):
            return np.full(pars.shape, 1e3)
        depth, gap, amplitude = np.exp(pars[-3:])
        if gap-depth >= orbits.ion_limit or gap / t > 80 or (grid.split and gap <= depth):
            return np.full(pars.shape, 1e3)
        path, fold, _ = grid.spline(depth, gap, np.r_[pars[:-3], 0.0])
        excess, current = _moments(orbits, path, grid.points(depth, gap), depth, gap, amplitude)
        return np.r_[excess, fold, current]

    def jacobian(pars):
        base = residual(pars)
        increments = 1e-7 * np.maximum(1.0, np.abs(pars))
        return np.column_stack([(residual(pars + step) - base) / dv
                                for step, dv in zip(np.diag(increments), increments)])

    p0 = seed.escaping_flux_m2_s / (orbits.speed*orbits.density) * 2/t
    s0 = grid.points(d0, gap0)
    tau0 = seed.minimum_tau*(1-s0[:-1])**2
    if seed.turning_path:
        path_seed = np.asarray(seed.turning_path)
        tau0 = np.interp(s0[:-1], path_seed[:, 0], path_seed[:, 1])
    start = np.r_[tau0, np.log([d0, gap0, p0])]
    sol = root(residual, start, jac=jacobian, options={"xtol": 3e-10})
    if np.max(np.abs(residual(sol.x))) > 3e-9:
        raise ArithmeticError(f"outer minimum collocation did not converge: {sol.message}")
    depth, gap, amplitude = np.exp(sol.x[-3:])
    path, _, critical = grid.spline(depth, gap, np.r_[sol.x[:-3], 0.0], inspect=True)
    sample = np.unique(np.r_[np.linspace(0, 1, 4*nodes), grid.points(depth, gap), critical[np.isfinite(critical)]])
    tau = path(sample)
    new_seed = OuterWellSeed(depth, gap, tau[0], amplitude*t/2*orbits.speed*orbits.density, tuple(zip(sample, tau)))
    if not grid.split and gap > depth:
        return solve_turning_region(orbits, new_seed, nodes)
    reversal = np.max(np.maximum.accumulate(tau[::-1])[::-1] - tau)
    if tau[0] <= 0 or reversal > 1e-8:
        raise ArithmeticError(f"outer minimum path reverses by {reversal:g} in tau")
    excess, current = _moments(orbits, path, sample, depth, gap, amplitude)
    return TurningRegion(new_seed, amplitude, gap*sample**2-depth, tau, current, np.max(np.abs(excess)), path)
