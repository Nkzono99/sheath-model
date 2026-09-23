# SPDX-License-Identifier: MIT AND Apache-2.0
# Zero-drift evaluation adapted from BEACH; see NOTICE.
"""Velocity moments of the upstream electron distribution transported along orbits."""

import numpy as np
from scipy.special import erf, erfcx

_x, _w = np.polynomial.legendre.leggauss(16)
VELOCITY_NODES = ((np.arange(8)[:, None] + (_x + 1) / 2) / 8).ravel()
VELOCITY_WEIGHTS = np.tile(_w / 16, 8)
POTENTIAL_NODES = ((np.arange(4)[:, None] + (_x + 1) / 2) / 4).ravel()
POTENTIAL_WEIGHTS = np.tile(_w / 8, 4)


def electron_density(psi, barrier, drift):
    """Return passing and reflected densities per unit upstream normalization.

    psi=phi/T_e, barrier=min(phi)/T_e <= 0; reflected includes both directions.
    The caller suppresses reflected particles below an internal minimum.
    """
    psi = np.asarray(psi, dtype=float)
    cutoff = np.sqrt(np.maximum(0.0, psi - barrier))
    if drift == 0:
        # BEACH's exact zero-drift evaluation avoids exp(psi) overflow and endpoint quadrature error.
        free = 0.5 * np.exp(barrier) * erfcx(cutoff)
        reflected = np.where(
            psi <= 0,
            np.exp(np.minimum(psi, 0)) * erf(cutoff),
            np.maximum(0.0, erfcx(np.sqrt(np.maximum(psi, 0))) - 2 * free),
        )
        return free, reflected
    amin = max(0.0, drift - 10.0)
    amax = max(np.sqrt(max(0.0, -barrier)), drift, 0.0) + 10.0
    upper = np.sqrt(np.maximum(0.0, amax * amax + psi))

    def moment(lo, hi):
        width = np.maximum(0.0, hi - lo)[..., None]
        velocity = np.asarray(lo)[..., None] + width * VELOCITY_NODES**2
        upstream = np.sqrt(np.maximum(0.0, velocity**2 - psi[..., None]))
        return np.sum(
            VELOCITY_WEIGHTS
            * 2
            * width
            * VELOCITY_NODES
            * np.exp(-((upstream - drift) ** 2)),
            axis=-1,
        ) / np.sqrt(np.pi)

    return (
        moment(
            np.sqrt(np.maximum(0.0, np.maximum(psi - barrier, amin**2 + psi))), upper
        ),
        2 * moment(np.sqrt(np.maximum(0.0, amin**2 + psi)), cutoff),
    )
