"""Poisson first-integral reconstruction of a Zhao sheath profile."""

import math
from typing import Dict

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ._physics import TypeASide


def _from_minimum(
    physics,
    phi_nodes_asc: np.ndarray,
    phi0_hat: float,
    n_swe_inf_hat: float,
    phi_m_hat: float,
    side: TypeASide,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build one Type-A branch starting just above the potential minimum."""
    phi_nodes_asc = np.asarray(phi_nodes_asc, dtype=float)
    if phi_nodes_asc.ndim != 1 or len(phi_nodes_asc) < 2:
        raise ValueError("phi_nodes_asc must be a 1D array with at least 2 points")
    if not np.all(np.diff(phi_nodes_asc) > 0.0):
        raise ValueError("phi_nodes_asc must be strictly increasing")
    if phi_nodes_asc[0] <= phi_m_hat:
        raise ValueError("phi_nodes_asc must start above phi_m_hat")

    dens_nodes = physics._densities_hat_type_a_side(
        phi_nodes_asc, phi0_hat, n_swe_inf_hat, phi_m_hat, side=side
    )
    rho_nodes = physics._rho_hat_from_densities(dens_nodes)

    dens_m = physics._densities_hat_type_a_side(
        np.array([phi_m_hat], dtype=float),
        phi0_hat,
        n_swe_inf_hat,
        phi_m_hat,
        side=side,
    )
    rho_m = float(physics._rho_hat_from_densities(dens_m)[0])
    if rho_m >= 0.:
        raise RuntimeError("charge density does not support an internal minimum")
    rho_m_neg = -rho_m

    # E^2(phi) = -2 ∫_{phi_m}^{phi} rho(psi) dpsi
    dphi0 = float(phi_nodes_asc[0] - phi_m_hat)
    int0 = 0.5 * (rho_m + rho_nodes[0]) * dphi0
    int_from_first = cumulative_trapezoid(rho_nodes, phi_nodes_asc, initial=0.0)
    integral_nodes = int0 + int_from_first
    e2_nodes = -2.0 * integral_nodes
    if np.any(e2_nodes <= 0.) or not np.all(np.isfinite(e2_nodes)):
        raise RuntimeError("Type A profile integral is not positive; refine the potential grid")

    # Start with the local quadratic minimum asymptotic instead of sampling
    # 1/|E| at the singular endpoint.
    s_nodes = np.empty_like(phi_nodes_asc)
    s_nodes[0] = math.sqrt(max(0.0, 2.0 * dphi0 / rho_m_neg))
    for i in range(1, len(phi_nodes_asc)):
        dphi = float(phi_nodes_asc[i] - phi_nodes_asc[i - 1])
        e2_mid = max(0.5 * (e2_nodes[i - 1] + e2_nodes[i]), 1.0e-14)
        s_nodes[i] = s_nodes[i - 1] + dphi / math.sqrt(e2_mid)

    return s_nodes, e2_nodes, dens_nodes


def _type_a_profile(physics, uk, band_hat, options):
    phi0_hat = float(uk["phi0_hat"])
    phi_m_hat = float(uk["phi_m_hat"])
    n_swe_inf_hat = float(uk["n_swe_inf_hat"])

    phi_m_eps = min(options.type_a_phi_m_eps_hat, 0.05 * max(1e-8, abs(phi_m_hat)))
    phi_m_eps = max(phi_m_eps, 1.0e-8)
    phi_end_hat = -abs(options.profile_phi_tol_hat)
    if phi_end_hat <= phi_m_hat:
        phi_end_hat = 0.5 * phi_m_hat
    if band_hat > 0:
        phi_end_hat = min(phi_end_hat, -(band_hat + options.upstream_band_margin * (abs(phi_m_hat) - band_hat)))

    ngrid = max(2000, options.n_type_a_grid)
    ngrid_upper = ngrid
    ngrid_lower = max(ngrid // 2, 1500)

    # Lower branch: surface -> z_m.
    x_lower = np.linspace(0.0, 1.0, ngrid_lower)
    phi_lower_asc = (phi_m_hat + phi_m_eps) + (
        phi0_hat - (phi_m_hat + phi_m_eps)
    ) * x_lower**2
    s_lower_asc, e2_lower_asc, dens_lower_asc = (
        _from_minimum(
            physics, phi_lower_asc, phi0_hat, n_swe_inf_hat, phi_m_hat, side="lower"
        )
    )
    z_m_hat = float(s_lower_asc[-1])

    phi_lower_desc = phi_lower_asc[::-1]
    z_lower_desc = z_m_hat - s_lower_asc[::-1]
    ehat_lower_desc = -np.sqrt(e2_lower_asc[::-1])
    dens_lower_desc = {k: v[::-1] for k, v in dens_lower_asc.items()}

    # Upper branch: z_m -> asymptotic region.
    x_upper = np.linspace(0.0, 1.0, ngrid_upper)
    phi_upper_asc = (phi_m_hat + phi_m_eps) + (
        phi_end_hat - (phi_m_hat + phi_m_eps)
    ) * (2.0 * x_upper - x_upper**2)
    s_upper_asc, e2_upper_asc, dens_upper_asc = (
        _from_minimum(
            physics, phi_upper_asc, phi0_hat, n_swe_inf_hat, phi_m_hat, side="upper"
        )
    )
    z_upper_asc = z_m_hat + s_upper_asc
    ehat_upper_asc = np.sqrt(e2_upper_asc)

    # Concatenate at the barrier without duplicating the first upper point.
    z_hat = np.concatenate([z_lower_desc, z_upper_asc])
    phi_hat = np.concatenate([phi_lower_desc, phi_upper_asc])
    ehat = np.concatenate([ehat_lower_desc, ehat_upper_asc])
    dens = {
        k: np.concatenate([dens_lower_desc[k], dens_upper_asc[k]])
        for k in dens_lower_desc
    }

    return z_hat, phi_hat, ehat, dens, z_m_hat


def _monotonic_profile(physics, branch, uk, band_hat, options):
    phi0_hat = float(uk["phi0_hat"])
    phi_m_hat = uk["phi_m_hat"]
    n_swe_inf_hat = float(uk["n_swe_inf_hat"])

    # Integrate from the exact neutral upstream endpoint, then omit infinity.
    t = np.linspace(0., 1., max(600, options.n_profile_grid))
    phi_hat = phi0_hat * (1.-t)**2
    dens = physics._densities_hat(branch, phi_hat, phi0_hat, n_swe_inf_hat, phi_m_hat)
    rho = physics._rho_hat_from_densities(dens)
    e2 = -2*cumulative_trapezoid(rho[::-1], phi_hat[::-1], initial=0.)[::-1]
    cutoff = min(abs(options.profile_phi_tol_hat), .5*abs(phi0_hat))
    if band_hat > 0:
        cutoff = max(cutoff, band_hat + options.upstream_band_margin*(abs(phi0_hat) - band_hat))
    keep = np.abs(phi_hat) >= cutoff
    keep[-1] = False
    phi_hat, e2 = phi_hat[keep], e2[keep]
    if len(phi_hat) < 2 or np.any(e2 <= 0.) or not np.all(np.isfinite(e2)):
        raise RuntimeError("monotonic profile integral is not positive; refine the potential grid")
    z_hat = np.concatenate(([0.], np.cumsum(.5*np.abs(np.diff(phi_hat)) *
                                (1/np.sqrt(e2[:-1]) + 1/np.sqrt(e2[1:])))))

    gradient_hat = -math.copysign(1., phi0_hat)*np.sqrt(e2)
    return z_hat, phi_hat, gradient_hat, {name: values[keep] for name, values in dens.items()}, None


def reconstruct_profile(physics, root, options):
    """Sample the first integral and map normalized arrays to physical units."""
    p = physics.p
    state = root._kernel_data()
    band_hat = root.upstream_negative_band_v / p.emission_temperature_ev
    if root.branch == "A":
        z_hat, phi_hat, gradient, densities, turning_height = _type_a_profile(physics, state, band_hat, options)
    else:
        z_hat, phi_hat, gradient, densities, turning_height = _monotonic_profile(
            physics, root.branch, state, band_hat, options
        )

    keep = z_hat <= options.zmax_hat
    z_hat, phi_hat, gradient = z_hat[keep], phi_hat[keep], gradient[keep]
    if len(z_hat) < 2:
        raise ValueError("zmax_hat contains fewer than two profile points")
    if turning_height is None:
        turning_height = float(z_hat[np.argmin(phi_hat)])
    return {
        **state,
        "z_hat": z_hat,
        "z_m_hat": turning_height,
        "z_m_m": turning_height * p.length_scale_m,
        "z_m_array_m": z_hat * p.length_scale_m,
        "phi_hat": phi_hat,
        "phi_V": phi_hat * p.emission_temperature_ev,
        "dphi_dzhat": gradient,
        "E_Vpm": -(p.emission_temperature_ev / p.length_scale_m) * gradient,
        **{name: values[keep] for name, values in densities.items()},
    }
