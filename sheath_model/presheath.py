"""Fixed far-wind sheath matched to an electron-replacement outer region."""

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, root
from scipy.special import erf, erfcx

from ._constants import EPS0, QE
from ._ions import ion_critical_potential
from .branch import FixedReservoir, MaxwellianEmission, _unit_gauss, classify_branch


@dataclass(frozen=True)
class OuterWellSeed:
    """Continuation seed for the surface side of an outer minimum."""

    minimum_depth_v: float
    band_above_minimum_v: float
    minimum_tau: float
    escaping_flux_m2_s: float
    turning_path: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class PresheathSolution:
    """Thin collisionless sheath in m and outer region in optical distance tau."""

    branch: Literal["A", "B_out", "B_nw", "C"]
    wall_potential_v: float
    band_potential_v: float
    minimum_potential_v: float | None
    outer_tau: np.ndarray
    outer_potential_v: np.ndarray
    sheath_z_m: np.ndarray
    sheath_potential_v: np.ndarray
    sheath_field_v_m: np.ndarray
    current_residual_a_m2: float
    outer_neutrality_residual_m3: float
    outer_well_seed: OuterWellSeed | None = None


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


def _relax(initial, target, distance):
    return initial + (target - initial) * -np.expm1(-distance)


def _returning(initial, out_target, in_target, turning_tau, tau):
    at_turn = _relax(initial, out_target, turning_tau)
    return _relax(at_turn, in_target, turning_tau - tau)


class _FixedReservoirOrbits:
    def __init__(self, reservoir, emission, order):
        self.reservoir = reservoir
        self.temperature = emission.temperature_ev
        self.speed = reservoir.speed_unit
        self.density = reservoir.ion_density_m3
        self.source = emission.flux_m2_s / (self.speed * self.density)
        self.ion_flux = reservoir.ion_flux_m2_s / (self.speed * self.density)
        self.ion_limit = ion_critical_potential(reservoir.ion_energy_ev, reservoir.ion_pressure_ev)
        self.x, self.w = _unit_gauss(order)
        q, weights = _unit_gauss(max(160, 2 * order))
        scale = math.sqrt(max(component.temperature_ev if hasattr(component, "temperature_ev")
                              else component.core_temperature_ev for component in reservoir.electrons))
        self.q, self.weights = scale*q/(1-q), scale*weights/(1-q)**2

    def wind(self, energy, direction):
        return self.reservoir.phase(direction * np.sqrt(np.maximum(energy, 0.0))) / self.density

    def ions(self, potential):
        return self.reservoir.ion_density(potential) / self.density

    def influx(self, depth):
        return self.reservoir.inward_flux(depth) / (self.speed * self.density)

    def passing(self, depth, cutoff, direction):
        low = np.sqrt(np.maximum(cutoff - depth, 0.0))[..., None]
        energy = depth[..., None] + low**2 + self.q*(2*low + self.q)
        return self.wind(energy, direction) @ self.weights

    def panel(self, function, low, high):
        # The sine map removes the square-root cusp at the cutoff.
        a = np.pi * self.x / 2
        widths = (np.asarray(high) - low)[..., None]
        points = np.asarray(low)[..., None] + widths * np.sin(a)**2
        return np.sum(function(points) * widths * np.pi * np.sin(a) * np.cos(a) * self.w, axis=-1)

    def photo(self, energy, depth, amplitude):
        return amplitude * np.exp((depth - energy) / self.temperature)

    def photo_density(self, potential, amplitude, cutoff):
        return amplitude * math.sqrt(math.pi * self.temperature) / 2 * erfcx(
            np.sqrt(np.maximum(cutoff + potential, 0.0) / self.temperature)
        )

    def upstream_edge_excess(self, h, depth, amplitude):
        psi = depth - h
        reflected = self.panel(lambda v: self.wind(psi[..., None] + v*v, 1), 0.0, np.sqrt(h))
        return self.passing(psi, 0.0, 1) + reflected + self.photo_density(-psi, amplitude, depth) - self.ions(-psi)

    def surface_excess(self, h, depth, amplitude):
        t = self.temperature
        return (self.passing(depth-h, depth, 1)
                + amplitude * math.sqrt(math.pi*t)/2 * np.exp(h/t) * (1 + erf(np.sqrt(h/t)))
                - self.ions(h-depth))


class _Presheath(_FixedReservoirOrbits):
    def upward(self, depth, amplitude, tau_start, psi_start, extent):
        """March from the band or valley toward the fixed far reservoir."""
        taus, depths = [tau_start], [psi_start]
        psi = np.asarray(psi_start)
        photo = self.photo_density(-psi, amplitude, depth)
        replacement = self.passing(psi, depth, -1)
        reflected = self.panel(lambda v: self.wind(psi+v*v, 1)-self.wind(psi+v*v, -1),
                               0.0, math.sqrt(depth-psi))
        incoming = self.passing(psi, 0.0, 1)
        initial = incoming + self.passing(psi, 0.0, -1) + reflected + np.exp(-tau_start)*(photo-replacement)
        step, max_error = 0.002, abs(initial-self.ions(-psi))
        while taus[-1] < extent and depths[-1] > 1e-8:
            tau = min(taus[-1] + step, extent)
            old_psi = np.array(depths[::-1])
            old_tau = np.array(taus[::-1])

            def excess(psi):
                history = np.r_[psi, old_psi]
                times = np.r_[tau, old_tau]
                cuts = np.unique(np.sqrt(np.maximum(np.r_[history, depth] - psi, 0.0)))
                speeds = cuts[:-1, None] + np.diff(cuts)[:, None]*self.x
                energy = psi + speeds**2
                turn = np.interp(energy, history, times)
                reflected = (self.wind(energy, 1) - self.wind(energy, -1))*np.exp(turn-tau)
                excess_reflected = np.sum(reflected * np.diff(cuts)[:, None] * self.w)
                passing_delta = self.photo_density(-psi, amplitude, depth) - self.passing(np.asarray(psi), depth, -1)
                return (self.passing(np.asarray(psi), 0, 1) + self.passing(np.asarray(psi), 0, -1)
                        + excess_reflected + np.exp(-tau)*passing_delta - self.ions(-psi))

            psi = _first_root_below(excess, depths[-1], 0.0, max(depths[-1]*1e-4, 1e-12))
            if psi is None:
                raise ArithmeticError(f"lost the upward neutral branch at tau={tau:g}")
            max_error = max(max_error, abs(excess(psi)))
            taus.append(tau)
            depths.append(psi)
            step = min(0.05, step * 1.3)
        return np.array(taus), -np.array(depths), max_error

    def inside_well(self, depth_hint, extent):
        t = self.temperature

        def amplitude(depth):
            return 2 * (self.influx(depth) - self.ion_flux) / t

        def candidate(depth):
            pd = amplitude(depth)
            h = depth * np.linspace(0, 1, 750)**2
            excess = self.upstream_edge_excess(h, depth, pd)
            indices = np.flatnonzero((excess[:-1] < 0) & (excess[1:] >= 0))
            if not indices.size:
                return math.nan, math.nan
            i = indices[0]
            edge = brentq(lambda h: self.upstream_edge_excess(np.asarray(h), depth, pd),
                          h[i], h[i+1], xtol=1e-15)
            y = math.sqrt(edge)*self.x
            integral = math.sqrt(edge) * np.dot(2*y*self.w, self.upstream_edge_excess(y*y, depth, pd))
            return edge, integral / edge

        scan = depth_hint * (1 - np.geomspace(0.6, 1e-10, 90))
        values = np.array([candidate(d)[1] for d in scan])
        k = np.flatnonzero(np.isfinite(values[:-1] + values[1:]) & (np.signbit(values[:-1]) != np.signbit(values[1:])))
        if not k.size:
            raise ArithmeticError("no neutral, field-free edge found above the inner valley")
        i = k[0]
        d = brentq(lambda d: candidate(d)[1], scan[i], scan[i+1], xtol=1e-15)
        pd = amplitude(d)
        h, _ = candidate(d)
        band = h - d
        wall = t * math.log(self.source / (pd*t/2)) - d
        self.check_wall(wall, -d)
        tau, outer_phi, err = self.upward(d, pd, 0, -band, extent)

        left = self.sheath_segment(lambda h: self.surface_excess(h, d, pd), -d, wall, 1)
        right = self.sheath_segment(lambda h: self.upstream_edge_excess(h, d, pd), -d, band, 1,
                                    asymptotic=True)
        z = np.r_[left[0][-1] - left[0][::-1], left[0][-1] + right[0][1:]]
        phi = np.r_[left[1][::-1], right[1][1:]]
        field = np.r_[left[2][::-1], -right[2][1:]]
        return self.result("A", wall, band, -d, tau, outer_phi, (z, phi, field), 0, err)

    def outer_well(self, seed, nodes, extent):
        t = self.temperature
        mesh = (1 - np.cos(np.linspace(0, np.pi, nodes))) / 2
        left_intervals = 2*nodes//3
        left_mesh = (1 - np.cos(np.linspace(0, np.pi, left_intervals+1))) / 2
        right_mesh = (1 - np.cos(np.linspace(0, np.pi, nodes-left_intervals))) / 2
        split = seed.band_above_minimum_v > seed.minimum_depth_v

        def make_grid(depth, gap):
            if not split:
                return mesh
            cross = math.sqrt(depth / gap)
            return np.r_[cross*left_mesh, cross + (1-cross)*right_mesh[1:]**2]

        def make_path(depth, gap, tau, inspect):
            if not split:
                path = CubicSpline(mesh, tau)
                extrema = path.derivative().roots(extrapolate=False) if inspect else ()
                return path, path(1.0, 1), extrema
            cross = math.sqrt(depth / gap)
            left = CubicSpline(left_mesh, tau[:left_intervals+1])
            right = CubicSpline(right_mesh, tau[left_intervals:])

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

        def targets(energy, depth):
            out, inward = replacement_targets(self.reservoir, energy, depth)
            return out / self.density, inward / self.density

        def evaluate(pars, points=None, inspect=False):
            d, gap, pd = np.exp(pars[-3:])
            path, fold, extrema = make_path(d, gap, np.r_[pars[:-3], 0.0], inspect)
            if inspect:
                return path, extrema
            s = make_grid(d, gap) if points is None else points
            tau, psi = path(s), d - gap*s*s
            a = math.sqrt(gap)*s
            photo_pass = self.photo_density(-psi, pd, d)
            passing = self.passing(psi, d, 1) + _relax(photo_pass, self.passing(psi, d, -1), tau)
            cut = np.arcsin(np.clip(np.sqrt(np.maximum(-psi, 0)) / np.maximum(a, 1e-100), 0, 1))
            turning_density = np.zeros_like(s)
            back_flux = np.zeros_like(s)
            for lo, hi in ((np.zeros_like(s), cut), (cut, np.full_like(s, np.pi/2))):
                angle = lo[:, None] + (hi-lo)[:, None]*self.x
                v = a[:, None]*np.sin(angle)
                energy = psi[:, None] + v*v
                weight = a[:, None]*np.cos(angle)*(hi-lo)[:, None]*self.w
                turn = path(s[:, None]*np.cos(angle))
                go, gi = targets(energy, d)
                photo = self.photo(energy, d, pd)
                outward = _relax(photo, go, tau[:, None])
                incoming = _returning(photo, go, gi, turn, tau[:, None])
                turning_density += np.sum(weight*(outward + incoming), axis=-1)
                back_flux += np.sum(weight*v*incoming, axis=-1)
            current = self.influx(d) + back_flux[-1] - pd*t/2*np.exp(gap/t) - self.ion_flux
            return passing + turning_density - self.ions(-psi), fold, current

        def residual(pars):
            if (not np.all(np.isfinite(pars)) or np.max(np.abs(pars[:-3])) > 40
                    or np.any(np.abs(pars[-3:]) > 27)):
                return np.full(pars.shape, 1e3)
            d, gap, _ = np.exp(pars[-3:])
            if gap-d >= self.ion_limit or gap / t > 80 or (split and gap <= d):
                return np.full(pars.shape, 1e3)
            excess, fold, current = evaluate(pars)
            return np.r_[excess, fold, current]

        d0, gap0 = seed.minimum_depth_v, seed.band_above_minimum_v
        p0 = seed.escaping_flux_m2_s / (self.speed*self.density) * 2/t
        s0 = make_grid(d0, gap0)
        tau0 = seed.minimum_tau*(1-s0[:-1])**2
        if seed.turning_path:
            path_seed = np.asarray(seed.turning_path)
            tau0 = np.interp(s0[:-1], path_seed[:, 0], path_seed[:, 1])
        start = np.r_[tau0, np.log([d0, gap0, p0])]

        def jacobian(pars):
            base = residual(pars)
            increments = 1e-7 * np.maximum(1.0, np.abs(pars))
            return np.column_stack([(residual(pars + step) - base) / dv
                                    for step, dv in zip(np.diag(increments), increments)])

        sol = root(residual, start, jac=jacobian, options={"xtol": 3e-10})
        if np.max(np.abs(residual(sol.x))) > 3e-9:
            raise ArithmeticError(f"outer minimum collocation did not converge: {sol.message}")
        d, gap, pd = np.exp(sol.x[-3:])
        _, _, current = evaluate(sol.x)
        path, critical = evaluate(sol.x, inspect=True)
        sample = np.unique(np.r_[np.linspace(0, 1, 4*nodes), make_grid(d, gap), critical[np.isfinite(critical)]])
        tau = path(sample)
        new_seed = OuterWellSeed(d, gap, tau[0], pd*t/2*self.speed*self.density, tuple(zip(sample, tau)))
        if not split and gap > d:
            return self.outer_well(new_seed, nodes, extent)
        max_reversal = np.max(np.maximum.accumulate(tau[::-1])[::-1] - tau)
        if tau[0] <= 0 or max_reversal > 1e-8:
            raise ArithmeticError(f"outer minimum path reverses by {max_reversal:g} in tau")
        band = gap - d
        wall = t * math.log(self.source / (pd*t/2)) - d
        self.check_wall(wall, band)
        far_tau, far_phi, err = self.upward(d, pd, tau[0], d, tau[0]+extent)
        excess, _, _ = evaluate(sol.x, sample)
        err = max(err, np.max(np.abs(excess)))

        def back(energy):
            turn = path(np.sqrt(np.clip((d-energy)/gap, 0, 1)))
            go, gi = targets(energy, d)
            return _returning(self.photo(energy, d, pd), go, gi, turn, 0.0)

        inner = self.monotone_wall(band, wall, d, pd, back)
        return self.result("B_out", wall, band, -d, np.r_[tau[::-1], far_tau[1:]],
                           np.r_[gap*sample[::-1]**2-d, far_phi[1:]], inner, current, err, new_seed)

    def no_well(self, extent):
        t = self.temperature
        g0 = self.wind(0.0, 1)
        far_limit = self.passing(np.asarray(0.0), 0, -1) / (math.sqrt(math.pi*t)/2)

        def march(pd):
            taus, depths = [extent + 6, extent], [0.0, 0.0]
            err = 0.0
            while taus[-1] > 0:
                tau = max(0.0, taus[-1] - (0.1 if taus[-1] > 1.0 else 0.02))
                past_psi = np.array(depths[::-1])
                past_tau = np.array(taus[::-1])

                def back(energy, psi):
                    turn = np.interp(energy, np.r_[psi, past_psi], np.r_[tau, past_tau])
                    return _returning(self.photo(energy, 0, pd), g0, g0, turn, tau)

                def excess(psi):
                    cuts = np.sqrt(np.maximum(np.r_[psi, past_psi] - psi, 0))
                    below = np.sum(self.panel(lambda v: back(psi+v*v, psi), cuts[:-1], cuts[1:]))
                    wind = self.passing(np.asarray(psi), 0, 1) + (1-np.exp(-tau))*(
                        self.passing(np.asarray(psi), 0, -1) + g0*math.sqrt(-psi))
                    return (wind + np.exp(-tau)*self.photo(psi, 0, pd)*math.sqrt(math.pi*t)/2
                            + below - self.ions(-psi))

                floor = max(depths[-1] - 4, -0.999999*self.ion_limit)
                psi = _first_root_below(excess, depths[-1], floor, max(1e-14, abs(depths[-1])*1e-3))
                if psi is None:
                    return None
                err = max(err, abs(excess(psi)))
                taus.append(tau)
                depths.append(psi)
            psi = depths[-1]
            cuts = np.sqrt(np.maximum(np.asarray(depths[::-1])-psi, 0))
            flux = np.sum(self.panel(lambda v: v*back(psi+v*v, psi), cuts[:-1], cuts[1:]))
            current = self.influx(0) + flux - self.photo(psi, 0, pd)*t/2 - self.ion_flux
            return current, np.array(taus[::-1]), np.array(depths[::-1]), err

        def current(pd):
            profile = march(pd)
            return profile[0] if profile is not None else math.nan

        grid = far_limit*np.r_[np.linspace(0.02, 0.98, 18), np.arange(0.981, 0.996, 0.001)]
        previous = None
        for pd in reversed(grid):
            value = current(pd)
            if math.isfinite(value):
                if previous is not None and (value == 0 or np.signbit(value) != np.signbit(previous[1])):
                    pd = brentq(current, pd, previous[0], xtol=1e-12)
                    break
                previous = pd, value
            else:
                previous = None
        else:
            raise ArithmeticError("no current-balanced monotone outer solution found")
        residual, tau, psi, err = march(pd)
        band = -psi[0]
        wall = t * math.log(self.source / (pd*t/2))
        self.check_wall(wall, band)

        def back(energy):
            turn = np.interp(energy, psi, tau)
            return _returning(self.photo(energy, 0, pd), g0, g0, turn, 0)

        inner = self.monotone_wall(band, wall, 0, pd, back)
        return self.result("B_nw", wall, band, None, tau[:-1], -psi[:-1], inner, residual, err)

    def weak_emission(self, extent):
        def balance(depth):
            return self.influx(depth) - self.ion_flux - self.source

        top = 1.0
        while balance(top) > 0 and top < 1e12:
            top *= 2
        if balance(0) <= 0 or balance(top) >= 0:
            raise ArithmeticError("no negative wall potential balances the fixed reservoir and source")
        d = brentq(balance, 0, top, xtol=1e-13)
        pd = self.source * 2 / self.temperature
        edge = _first_root_below(lambda psi: self.upstream_edge_excess(np.asarray(d-psi), d, pd),
                                 d, 0.0, max(d*1e-5, 1e-12))
        if edge is None or edge >= d:
            raise ArithmeticError("no negative-potential neutral edge found above the wall")
        tau, outer_phi, err = self.upward(d, pd, 0, edge, extent)
        inner = self.sheath_segment(
            lambda x: -self.upstream_edge_excess(d-edge-x, d, pd), -edge, -d, -1,
            asymptotic_start=True
        )
        z, phi, field = inner
        return self.result("C", -d, -edge, -d, tau, outer_phi,
                           (z[-1]-z[::-1], phi[::-1], -field[::-1]), balance(d), err)

    def check_wall(self, wall, lower):
        if not lower < wall < self.ion_limit:
            raise ArithmeticError("wall is outside the unblocked positive-drop ion fluid branch")

    def monotone_wall(self, band, wall, depth, pd, back):
        t = self.temperature

        def excess(delta):
            phi = band + delta
            zero = np.sqrt(np.maximum(phi, delta))
            low, top = np.sqrt(delta), np.sqrt(phi + depth)
            middle = np.clip(zero, low, top)
            returning = self.panel(lambda v: back(v*v-phi[:, None]), low, middle)
            returning += self.panel(lambda v: back(v*v-phi[:, None]), middle, top)
            photo = self.photo(-phi, depth, pd) * math.sqrt(math.pi*t)/2 * (1+erf(np.sqrt(delta/t)))
            return self.passing(-phi, depth, 1) + photo + returning - self.ions(phi)

        z, phi, field = self.sheath_segment(excess, band, wall, 1, asymptotic_start=True)
        return z[-1]-z[::-1], phi[::-1], field[::-1]

    def sheath_segment(self, excess, start, end, direction, *, asymptotic=False, asymptotic_start=False):
        span = abs(end-start)
        y = np.linspace(0, math.sqrt(span), 1800)
        h = y*y
        density = excess(h)
        field_squared = 2*QE*self.density/EPS0 * cumulative_trapezoid(2*y*density, y, initial=0)
        floor = max(2e-9, np.max(np.abs(field_squared))*1e-8)
        if np.min(field_squared) < -floor:
            raise ArithmeticError("Poisson first integral is negative on the collisionless segment")
        stop = -4 if asymptotic else None
        begin = 8 if asymptotic_start else 0
        e = np.sqrt(np.maximum(field_squared[begin:stop], 0))
        y = y[begin:stop]
        if np.any(e[1:] <= 0) or (asymptotic_start and e[0] <= 0):
            raise ArithmeticError("collisionless segment has an unresolved zero field")
        integrand = np.empty_like(e)
        integrand[1:] = 2*y[1:]/e[1:]
        # At a simple minimum E is proportional to sqrt(phi-phi_min).
        integrand[0] = 2*y[0]/e[0] if y[0] else integrand[1]
        z = cumulative_trapezoid(integrand, y, initial=0)
        return z, start + direction*y*y, e

    def result(self, branch, wall, band, minimum, taus, outer, inner, current, error, seed=None):
        return PresheathSolution(branch, wall, band, minimum, taus, outer, *inner,
                                 current*QE*self.speed*self.density, error*self.density, seed)


def _first_root_below(function, upper, floor, step):
    right, f_right = upper, function(upper)
    while right > floor:
        left = max(right - step, floor)
        if left == right:
            return None
        f_left = function(left)
        if f_left == 0 or np.signbit(f_left) != np.signbit(f_right):
            return brentq(function, left, right, xtol=1e-15)
        right, f_right = left, f_left
        step *= 1.5
    return None


def solve_presheath(reservoir: FixedReservoir, emission: MaxwellianEmission,
                    branch: Literal["A", "B_out", "B_nw", "C"], *,
                    seed: OuterWellSeed | None = None, nodes: int = 65, order: int = 96,
                    far_tau: float = 16.0) -> PresheathSolution:
    """Solve one branch of the collisionless sheath plus quasi-neutral outer region.

    The far reservoir is at 0 V with equal total electron and ion densities.
    The Maxwellian emission is stationary. The outer coordinate is measured
    from the neutral edge, while the sheath profile starts at the wall and
    stops short of its asymptotic neutral edge.
    A numerical failure raises ArithmeticError. Pass a nearby B_out solution's
    outer_well_seed for continuation, especially for shallow outer minima.
    """
    if not isinstance(emission, MaxwellianEmission) or emission.flux_m2_s <= 0:
        raise ValueError("the presheath solver requires positive half-Maxwellian emission")
    if not math.isclose(sum(c.density_m3 for c in reservoir.electrons), reservoir.ion_density_m3, rel_tol=1e-10):
        raise ValueError("the prescribed far reservoir must be neutral")
    if nodes < 9 or nodes % 2 != 1 or order < 24 or not math.isfinite(far_tau) or not 1 <= far_tau <= 50:
        raise ValueError("use an odd node count >= 9, quadrature order >= 24 and far_tau in [1, 50]")
    if seed is not None and branch != "B_out":
        raise ValueError("only B_out uses an outer-well seed")
    model = _Presheath(reservoir, emission, order)
    if branch == "B_nw":
        return model.no_well(far_tau)
    if branch == "C":
        return model.weak_emission(far_tau)
    if branch not in ("A", "B_out"):
        raise ValueError("branch must be A, B_out, B_nw or C")
    if seed is None:
        marginal = classify_branch(reservoir, emission)
        if marginal.neutral_depth_v is None:
            raise ArithmeticError("could not seed the branch from the hypothetical neutral minimum")
        depth = marginal.neutral_depth_v
        if branch == "A":
            return model.inside_well(depth, far_tau)
        flux = reservoir.inward_flux(depth)-reservoir.ion_flux_m2_s
        seed = OuterWellSeed(depth, 0.035*depth, 0.04, flux)
        try:
            return model.outer_well(seed, nodes, far_tau)
        except ArithmeticError:
            seed = OuterWellSeed(0.4*depth, 0.9*depth, 0.8, flux)
    if not all(math.isfinite(v) and v > 0 for v in (
        seed.minimum_depth_v, seed.band_above_minimum_v, seed.minimum_tau, seed.escaping_flux_m2_s
    )):
        raise ValueError("all outer-well seed parameters must be positive and finite")
    if seed.turning_path:
        path = np.asarray(seed.turning_path)
        if (path.ndim != 2 or path.shape[1] != 2 or not np.all(np.isfinite(path))
                or path[0, 0] != 0 or path[-1, 0] != 1 or np.any(np.diff(path[:, 0]) <= 0)):
            raise ValueError("the outer-well seed path must have increasing coordinates from 0 to 1")
    return model.outer_well(seed, nodes, far_tau)
