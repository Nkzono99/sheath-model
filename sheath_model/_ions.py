"""Fixed-entry ion fluid transport, including Sana & Mishra (2026)'s pressure term.

The coefficient is pressure_factor * T_i, held constant as in their Eq. (17).
This is a fluid density closure; it does not specify an ion velocity distribution.
"""

import math

import numpy as np


def ion_critical_potential(entry_energy_ev: float, pressure_energy_ev: float = 0.0) -> float:
    """Largest potential [V] on the supersonic root connected to upstream.

    entry_energy_ev = m_i*u_0**2/(2*e); pressure_energy_ev = pressure_factor*T_i.
    The upstream speed must exceed the ion thermal sound speed.
    """
    k, a = float(entry_energy_ev), float(pressure_energy_ev)
    if not math.isfinite(k) or not math.isfinite(a) or k <= 0 or a < 0 or 2*k <= a:
        raise ValueError("ion entry energy must be positive and exceed half the pressure energy")
    if a == 0:
        return k
    gap = (2*k - a)/a
    # Avoid cancellation when the entry speed approaches the thermal sound speed.
    if gap < 1e-3:
        difference = sum((-1)**n * gap**n/n for n in range(2, 9))
    else:
        difference = gap - math.log1p(gap)
    return 0.5*a*difference


def ion_density_ratio(potential_v, entry_energy_ev: float, pressure_energy_ev: float = 0.0):
    """Return n_i/n_i,infinity, choosing the root continuous from n_i(0)=n_i,infinity.

    In log-density x, energy conservation is
    K*expm1(-2*x) + a*x + phi = 0. The other warm-fluid root is excluded.
    Potentials above the sonic turning point raise ValueError.
    """
    k, a = float(entry_energy_ev), float(pressure_energy_ev)
    critical = ion_critical_potential(k, a)
    phi = np.asarray(potential_v, dtype=float)
    if np.any(~np.isfinite(phi)) or np.any(phi > critical) or (a == 0 and np.any(phi >= critical)):
        raise ValueError("potential blocks the upstream-connected ion flow")
    if a == 0:
        return (1.0 - phi/k)**-0.5
    upper = np.where(phi >= 0, 0.5*math.log(2*k/a), 0.0)
    lower = np.where(phi >= 0, 0.0, -0.5*np.log1p(2*np.abs(phi)/k) - 1.0)
    for _ in range(64):
        midpoint = 0.5*(lower + upper)
        residual = k*np.expm1(-2*midpoint) + a*midpoint + phi
        lower = np.where(residual > 0, midpoint, lower)
        upper = np.where(residual > 0, upper, midpoint)
    ratio = np.exp(0.5*(lower + upper))
    ratio = np.where(phi == 0, 1.0, ratio)
    return np.where(phi == critical, math.sqrt(2*k/a), ratio)
