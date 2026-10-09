"""Zhao zero-current roots with an orbit-consistent electron model whose slow incoming part is smoothed.

Incoming upstream electrons have speed distribution (per unit normalization N_e, a = |v|/v_th)
    g(a) = exp(-a^2-u^2)/sqrt(pi) * [cosh(2ua) + w(a) sinh(2ua)],   w(a) = 1 - exp(-(a/a_s)^2).
a_s = 0 is the drifting Maxwellian exp(-(a-u)^2)/sqrt(pi); a_s > 0 removes the drift of electrons slower than
about a_s*v_th, so g'(0) = 0 and the h*log(1/h) term next to upstream disappears. Reflected electrons mirror the
incoming ones; densities and fluxes follow energy conservation (w^2 = a^2 + psi, psi = e*phi/T_e).

Independent of sheath-model: cold ions, Maxwellian photoelectrons emitted from the wall, Poisson first integral.
Usage: python smooth_slow_electrons.py
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy import integrate, optimize, special

QE, ME, MP, EPS0 = 1.602176634e-19, 9.1093837139e-31, 1.67262192369e-27, 8.8541878128e-12
N_I, V_I, T_E, T_PE, J_EMIT = 5.0e6, 4.0e5, 10.0, 2.2, 4.5e-6
VTH_E, VTH_PE = math.sqrt(2 * QE * T_E / ME), math.sqrt(2 * QE * T_PE / ME)
E_ION_V = 0.5 * MP * V_I**2 / QE
GAMMA_I = N_I * V_I
NPE0 = 2 * math.sqrt(math.pi) * (J_EMIT / QE) / VTH_PE  # normalization of the emitted Maxwellian
QUAD = dict(limit=400, epsabs=1e-14, epsrel=1e-12)


class Electrons:
    def __init__(self, u: float, a_s: float):
        self.u, self.a_s = u, a_s

    def g(self, a):
        a = np.asarray(a, dtype=float)
        u = self.u
        if u == 0.0:
            return np.exp(-a * a) / math.sqrt(math.pi)
        if self.a_s <= 0.0:
            return np.exp(-(a - u) ** 2) / math.sqrt(math.pi)
        w = -np.expm1(-(a / self.a_s) ** 2)
        return np.exp(-a * a - u * u) * (np.cosh(2 * u * a) + w * np.sinh(2 * u * a)) / math.sqrt(math.pi)

    def _points(self, lo: float, hi: float, psi: float) -> list[float]:
        # Resolve the smoothing scale: w where sqrt(w^2 - psi) = k*a_s.
        pts = []
        if self.a_s > 0:
            for k in (0.5, 1.0, 2.0, 4.0):
                a = k * self.a_s
                if a * a + psi > 0:
                    w = math.sqrt(a * a + psi)
                    if lo < w < hi:
                        pts.append(w)
        return sorted(pts)

    def _w_integral(self, lo: float, hi: float, psi: float) -> float:
        f = lambda w: float(self.g(math.sqrt(w * w - psi)))
        if hi == math.inf:
            mid = lo + 12.0
            pts = self._points(lo, mid, psi)
            return integrate.quad(f, lo, mid, points=pts or None, **QUAD)[0] + integrate.quad(f, mid, math.inf, **QUAD)[0]
        pts = self._points(lo, hi, psi)
        return integrate.quad(f, lo, hi, points=pts or None, **QUAD)[0]

    def density(self, psi: float, barrier2: float, reflected: bool) -> float:
        """Density per unit N_e at psi; barrier2 = -psi_barrier >= 0. reflected: include the mirrored slow part."""
        wb = math.sqrt(max(barrier2 + psi, 0.0))
        passing = self._w_integral(wb, math.inf, psi)
        return passing + (2.0 * self._w_integral(0.0, wb, psi) if reflected and wb > 0 else 0.0)

    def flux(self, barrier2: float) -> float:
        """Passing flux per unit N_e, in units of v_th."""
        b = math.sqrt(max(barrier2, 0.0))
        pts = [k * self.a_s for k in (0.5, 1, 2, 4) if self.a_s > 0 and k * self.a_s > b] or None
        return integrate.quad(lambda a: a * float(self.g(a)), b, b + 12.0, points=pts, **QUAD)[0]


def ion_density(phi: float) -> float:
    return N_I / math.sqrt(1.0 - phi / E_ION_V)


def pe_density(phi: float, phi0: float, phim: float, side: str) -> float:
    if NPE0 == 0.0:
        return 0.0
    s = math.sqrt(max(phi - phim, 0.0) / T_PE)
    base = NPE0 * math.exp(-(phi0 - phi) / T_PE)
    if side == "lower":
        return base * (0.5 * special.erfc(s) + special.erf(s))
    return base * 0.5 * special.erfc(s)


def rho(phi, phi0, phim, ne, el, side):
    """Charge density / e [m^-3]."""
    psi = phi / T_E
    barrier2 = -phim / T_E
    n_e = ne * el.density(psi, barrier2, reflected=(side != "lower"))
    return ion_density(phi) - n_e - pe_density(phi, phi0, phim, side)


def residual_a(x, el):
    phi0, phim, ne = x
    if not (phim < 0 < phi0) or ne <= 0:
        return [1e3, 1e3, 1e3]
    r1 = (ion_density(0.0) - ne * el.density(0.0, -phim / T_E, True) - pe_density(0.0, phi0, phim, "upper")) / N_I
    escape = NPE0 * VTH_PE / (2 * math.sqrt(math.pi)) * math.exp(-(phi0 - phim) / T_PE)
    r2 = (GAMMA_I + escape - ne * VTH_E * el.flux(-phim / T_E)) / GAMMA_I
    r3 = integrate.quad(lambda p: rho(p, phi0, phim, ne, el, "upper"), phim, 0.0, limit=400, epsabs=1e-6, epsrel=1e-12)[0]
    return [r1, r2, r3 / (N_I * -phim)]


def solve_c(el):
    """Type C without photoelectrons: current fixes N_e for a given phi0, neutrality fixes phi0."""
    def ne_of(phi0):
        return GAMMA_I / (VTH_E * el.flux(-phi0 / T_E))

    def neutral(phi0):
        return (ion_density(0.0) - ne_of(phi0) * el.density(0.0, -phi0 / T_E, True)) / N_I

    phi0 = optimize.brentq(neutral, -30.0, -0.5, xtol=1e-13)
    return phi0, phi0, ne_of(phi0)


def e2_profile(phi0, phim, ne, el, branch):
    """E^2 [V^2/m^2] next to upstream (from the exact upstream end) and over the rest of the profile."""
    scale = 2 * QE / EPS0
    depth_ref = -phim if branch == "A" else -phi0
    side = "upper" if branch == "A" else "upper"
    depths = np.unique(np.concatenate([np.geomspace(1e-9, depth_ref, 160), np.linspace(0, depth_ref, 81)[1:]]))
    upstream = []
    for d in depths:
        val = integrate.quad(lambda p: rho(p, phi0, phim, ne, el, side), -d, 0.0, limit=400, epsabs=1e-9, epsrel=1e-12)[0]
        upstream.append((float(d), scale * val))
    lower = []
    if branch == "A":
        for p in np.linspace(phim, phi0, 81)[1:]:
            val = -integrate.quad(lambda q: rho(q, phi0, phim, ne, el, "lower"), phim, p, limit=400, epsabs=1e-9, epsrel=1e-12)[0]
            lower.append((float(p), scale * val))
    # Ignore round-off: the A minimum (E=0 by construction) and values far below the upstream-side field scale.
    inner = [(d, e2) for d, e2 in upstream if d < 0.999 * depth_ref]
    upstream_max = max(e2 for _, e2 in inner)
    floor = -1e-12 * upstream_max
    neg = [d for d, e2 in inner if e2 < floor]
    band = max(neg) if neg else 0.0
    if band > 0:  # refine the outer edge by bisection
        outside = min([d for d, _ in inner if d > band], default=depth_ref)
        for _ in range(40):
            mid = 0.5 * (band + outside)
            v = scale * integrate.quad(lambda p: rho(p, phi0, phim, ne, el, side), -mid, 0.0, limit=400, epsabs=1e-9, epsrel=1e-12)[0]
            band, outside = (mid, outside) if v < floor else (band, mid)
    return {"band_V": band, "upstream_min_E2": min(e2 for _, e2 in inner), "upstream_max_E2": upstream_max,
            "lower_min_E2": min([e2 for _, e2 in lower], default=float("nan")),
            "wall_E2": lower[-1][1] if lower else upstream[-1][1]}


def report(label, branch, el, root):
    phi0, phim, ne = root
    barrier2 = -phim / T_E
    flux_e = ne * VTH_E * el.flux(barrier2)
    esc = 0.0 if branch == "C" else NPE0 * VTH_PE / (2 * math.sqrt(math.pi)) * math.exp(-(phi0 - phim) / T_PE)
    prof = e2_profile(phi0, phim, ne, el, branch)
    drift_el = Electrons(el.u, 0.0)
    total = integrate.quad(lambda a: a * float(drift_el.g(a)), 0, 14, **QUAD)[0]
    changed = integrate.quad(lambda a: a * abs(float(el.g(a)) - float(drift_el.g(a))), 0, 14, points=[el.a_s] if el.a_s > 0 else None, **QUAD)[0]
    row = dict(label=label, branch=branch, u=el.u, a_s=el.a_s, phi0_V=phi0, phim_V=phim, ne_m3=ne,
               electron_flux=flux_e, pe_escape_flux=esc, modified_incoming_flux_fraction=changed / total, **prof)
    print(json.dumps(row), flush=True)
    return row


def main():
    u = V_I / VTH_E
    rows = []
    # 1. Reproduce sheath-model: zero drift and drifting Maxwellian (no smoothing).
    guesses = {(0.0, 0.0): (7.0478, -0.1317, 5.888e6), (u, 0.0): (6.0383, -0.7862, 4.585e6)}
    for (uu, a_s), g0 in guesses.items():
        el = Electrons(uu, a_s)
        sol = optimize.fsolve(residual_a, g0, args=(el,), xtol=1e-13, full_output=True)
        rows.append(report("A sheath-model check", "A", el, sol[0]))
    global NPE0
    npe_saved = NPE0
    NPE0 = 0.0
    for uu in (0.0, u):
        rows.append(report("C sheath-model check", "C", Electrons(uu, 0.0), solve_c(Electrons(uu, 0.0))))
    # 2. Smoothed slow electrons, Type C (no photoelectrons).
    for a_s in (0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01):
        el = Electrons(u, a_s)
        rows.append(report("C smoothed", "C", el, solve_c(el)))
    NPE0 = npe_saved
    # 3. Smoothed slow electrons, Type A, continued from the unsmoothed drifting root.
    guess = guesses[(u, 0.0)]
    for a_s in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3):
        el = Electrons(u, a_s)
        sol, info, ier, msg = optimize.fsolve(residual_a, guess, args=(el,), xtol=1e-13, full_output=True)
        if ier != 1:
            print(json.dumps({"label": "A smoothed", "a_s": a_s, "failed": msg}), flush=True)
            continue
        guess = tuple(sol)
        rows.append(report("A smoothed", "A", el, sol))
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "smooth_slow_electrons.json", "w"), indent=1)


if __name__ == "__main__":
    main()
