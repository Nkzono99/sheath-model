"""Collisionless floating sheath with a fixed inlet in a finite 0 V column."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import brentq

from ._constants import EPS0, QE
from .branch import FixedReservoir, MaxwellianEmission
from ._fixed_orbits import FixedReservoirOrbits


@dataclass(frozen=True)
class FiniteColumnSolution:
    branch: str
    z_m: np.ndarray
    potential_v: np.ndarray
    field_v_m: np.ndarray
    escaping_flux_m2_s: float
    minimum_height_m: float
    monotone_max_height_m: float


def _monotone(excess, phi_min, phi_max, height, factor, points):
    y = np.linspace(0, math.sqrt(phi_max-phi_min), points)
    grad = factor * excess(y*y)
    gradient = CubicSpline(y, grad)
    zeros = np.zeros(len(y)-1)
    # Multiply each cubic in (y-y_i) by 2*y before integrating.
    product = 2*(np.vstack([gradient.c, zeros]) + np.vstack([zeros, gradient.c*y[:-1]]))
    integral = PPoly(product, y).antiderivative()
    roots = gradient.roots(extrapolate=False)
    ends = np.r_[0, roots[np.isfinite(roots)], y[-1]]
    bottom = ends[np.argmin(integral(ends))]
    floor = integral(bottom)
    curvature = float(gradient.derivative()(bottom) / (4*bottom)) if 0 < bottom < y[-1] else 0.0
    pivot = bottom*bottom
    coefficients = ([integral(bottom, n)/math.factorial(n) for n in range(5, 1, -1)]
                    if curvature > 0 else [])

    def field_squared(offset, delta=None):
        value = np.maximum(integral(np.sqrt(np.maximum(offset, 0))) - floor, 0)
        if curvature > 0:
            dy = (offset-pivot if delta is None else delta)/(np.sqrt(offset)+bottom)
            return np.where(np.abs(dy) < 1e-5, np.polyval(coefficients, dy)*dy*dy, value)
        return value

    def parameterize(constant):
        if curvature > 0:
            scale = math.sqrt(constant / curvature)
            a, b = np.arcsinh((np.array([0, y[-1]**2]) - pivot)/scale)

            def offset(t):
                return np.clip(pivot + scale*np.sinh(t), 0, y[-1]**2)

            def derivative(t):
                delta = scale*np.sinh(t)
                return scale*np.cosh(t) / np.sqrt(constant + field_squared(offset(t), delta))
        else:
            a, b = 0, y[-1]
            span = y[-1]**2
            reverse = bottom == y[-1]

            def offset(t):
                return span - t*t if reverse else t*t

            def derivative(t):
                return 2*t / np.sqrt(constant + field_squared(offset(t)))
        return offset, derivative, a, b

    def length(constant):
        _, derivative, a, b = parameterize(constant)
        cuts = None
        if curvature > 0:
            scale = math.sqrt(constant/curvature)
            edge = np.array([bottom-1e-5, bottom, bottom+1e-5])**2-pivot
            cuts = [p for p in np.arcsinh(edge/scale) if a < p < b]
        return quad(derivative, a, b, points=cuts, epsabs=3e-7, limit=120)[0]

    limit = math.inf if curvature > 0 else length(0)
    if height > limit:
        return None, limit
    hi = 1.0
    while length(hi) > height:
        hi *= 4
    lo = 0.0
    if curvature > 0:
        lo = 1e-8
        while length(lo) < height and lo > 1e-23:
            lo *= 0.01
        if lo <= 1e-23 and length(lo) < height:
            raise ArithmeticError("requested height is too close to the interior zero-field limit")
    if height == limit:
        constant = 0.0
    elif curvature > 0:
        constant = math.exp(brentq(lambda log_c: length(math.exp(log_c))-height, math.log(lo), math.log(hi)))
    else:
        constant = brentq(lambda c: length(c)-height, lo, hi, xtol=1e-15)
    offset, derivative, a, b = parameterize(constant)
    coordinate = np.linspace(a, b, points)
    distances = derivative(coordinate[1:-1])
    field = np.sqrt(constant + field_squared(offset(coordinate)))
    interior = cumulative_trapezoid(np.r_[distances[0], distances, distances[-1]], coordinate, initial=0)
    # Use the adaptive integral for the total length; the sampled profile supplies its shape.
    z = height * interior / interior[-1]
    phi = phi_min + offset(coordinate)
    if phi[-1] > phi[0]:
        return (height-z[::-1], phi[::-1], field[::-1]), limit
    return (z, phi, field), limit


def _turning_column(model, height_m, points, lo, upper, monotone_limit):
    """Match a two-sided turning orbit to the specified height."""
    t = model.temperature
    factor = 2*QE*model.density/EPS0
    scale = model.speed * model.density

    def segment(excess, depth, span):
        if span == 0:
            return np.array([0.0]), np.array([-depth]), np.array([0.0])
        y = np.linspace(0, math.sqrt(span), points)
        gradient = factor * excess(y*y)
        field2 = cumulative_trapezoid(2*y*gradient, y, initial=0)
        if gradient[0] <= 0 or np.any(field2[1:] <= 0):
            raise ArithmeticError("no real field along the turning orbit")
        field = np.sqrt(field2)
        distance = cumulative_trapezoid(np.r_[2/math.sqrt(gradient[0]), 2*y[1:]/field[1:]], y, initial=0)
        return distance, y*y-depth, field

    def orbit(depth):
        flux = model.escape_for_zero_current(depth)
        if not 0 < flux <= model.source:
            raise ArithmeticError("turning orbit exceeds the available emission")
        wall_gap = t * math.log(model.source / flux)
        if wall_gap-depth >= model.ion_limit:
            raise ArithmeticError("turning orbit blocks incoming ions")
        pd = flux * 2/t
        inner = segment(lambda h: model.surface_excess(h, depth, pd), depth, wall_gap)
        outer = segment(lambda h: model.upstream_edge_excess(h, depth, pd), depth, depth)
        return inner, outer, flux

    def span(depth):
        inner, outer, _ = orbit(depth)
        return inner[0][-1] + outer[0][-1] - height_m

    previous = None
    bracket = None
    for depth in lo + (upper-lo)*np.r_[0, np.geomspace(1e-7, 0.999999, 130)]:
        try:
            difference = span(depth)
        except ArithmeticError:
            if previous is not None and previous[1] < 0:
                left, right = previous[0], depth
                for _ in range(50):
                    mid = (left + right)/2
                    if mid == left or mid == right:
                        break
                    try:
                        difference = span(mid)
                    except ArithmeticError:
                        right = mid
                        continue
                    if difference >= 0:
                        bracket = left, mid
                        break
                    left = mid
            previous = None
        else:
            if previous is not None and np.signbit(difference) != np.signbit(previous[1]):
                bracket = previous[0], depth
            previous = depth, difference
        if bracket is not None:
            break
    if bracket is None:
        raise ArithmeticError("no real turning orbit was bracketed at the requested height")
    depth = brentq(span, *bracket, xtol=2e-14)
    inner, outer, flux = orbit(depth)
    z_min = inner[0][-1]
    z = np.r_[z_min-inner[0][::-1], z_min+outer[0][1:]]
    phi = np.r_[inner[1][::-1], outer[1][1:]]
    field = np.r_[inner[2][::-1], -outer[2][1:]]
    return FiniteColumnSolution("A", z, phi, field, flux*scale, z_min, monotone_limit)


def solve_finite_column(reservoir: FixedReservoir, emission: MaxwellianEmission, height_m: float,
                        *, branch: str | None = None, points: int = 2049, order: int = 80) -> FiniteColumnSolution:
    """Solve B, A, or C at fixed inlet VDF and upper potential 0 V.

    The upper field is free and the wall current is zero. Specify A or B to
    search that branch independently. With no branch specified, search the
    monotone branch followed by the turning branch when the monotone length
    cannot reach the given height. A finite search gives no uniqueness claim.
    """
    if branch not in (None, "A", "B", "C") or not math.isfinite(height_m) or height_m <= 0:
        raise ValueError("use a positive finite height and branch A, B, C, or None")
    if points < 129 or order < 24 or not isinstance(emission, MaxwellianEmission) or emission.flux_m2_s <= 0:
        raise ValueError("use positive half-Maxwellian emission, at least 129 points and quadrature order >= 24")
    model = FixedReservoirOrbits(reservoir, emission, order)
    t = model.temperature
    factor = 2*QE*model.density/EPS0
    scale = model.speed * model.density

    available = model.escape_for_zero_current(0)
    if available <= 0:
        raise ArithmeticError("electron influx at 0 V does not exceed the ion flux")
    if available == model.source and branch in (None, "B"):
        z = np.linspace(0, height_m, points)
        zeros = np.zeros_like(z)
        if abs(model.surface_excess(np.array(0.0), 0, available*2/t)) < 1e-12:
            return FiniteColumnSolution("B", z, zeros, zeros.copy(), available*scale, height_m, math.inf)
    high = 1.0

    def cutoff(target):
        nonlocal high
        while model.escape_for_zero_current(high) > target and high < 1e12:
            high *= 2
        return brentq(lambda d: model.escape_for_zero_current(d)-target, 0, high, xtol=1e-14)

    if branch in (None, "C") and available > model.source:
        depth = cutoff(model.source)
        pd = model.source * 2/t
        profile, limit = _monotone(lambda h: model.upstream_edge_excess(h, depth, pd), -depth, 0,
                                   height_m, factor, points)
        if profile is None:
            raise ArithmeticError("the monotone negative-wall branch cannot span the requested height")
        z, phi, field = profile
        return FiniteColumnSolution("C", height_m-z[::-1], phi[::-1], -field[::-1], emission.flux_m2_s, 0.0, limit)
    if branch == "C":
        raise ArithmeticError("zero current with all emitted electrons escaping requires a negative wall")

    limit = 0.0
    if available < model.source:
        wall = t * math.log(model.source / available)
        if wall < model.ion_limit:
            profile, limit = _monotone(lambda h: model.surface_excess(h, 0, available*2/t),
                                       0, wall, height_m, factor, points)
            if profile is not None and branch != "A":
                return FiniteColumnSolution("B", *profile, available*scale, height_m, limit)
    if branch == "B":
        raise ArithmeticError("the monotone positive-wall branch cannot span the requested height")

    lo = cutoff(model.source) if available > model.source else 0.0
    upper = cutoff(0.0)
    return _turning_column(model, height_m, points, lo, upper, limit)
