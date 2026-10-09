"""Orbit moments, ion transport, J=0 equations and physical profile conditions."""

from __future__ import annotations

import math
from typing import Dict, Literal

import numpy as np
from scipy.special import erf, erfc
from ._orbits import electron_density, POTENTIAL_NODES, POTENTIAL_WEIGHTS
from ._ions import ion_density_ratio, ion_critical_potential

from .params import FixedEntryParams, ZhaoParams

Branch = Literal["A", "B", "C"]
TypeASide = Literal["lower", "upper"]


class SheathPhysics:
    def __init__(self, params: FixedEntryParams | ZhaoParams):
        self.p = params
        if not isinstance(params, (FixedEntryParams, ZhaoParams)):
            raise TypeError("inputs must be FixedEntryParams or ZhaoParams")
        if isinstance(params, ZhaoParams):
            if not math.isfinite(params.sun_elevation_deg) or not 0 <= params.sun_elevation_deg <= 90:
                raise ValueError("sun_elevation_deg must be finite and in [0, 90]")
            if params.electron_drift_mode not in {"zero", "normal", "full"}:
                raise ValueError("electron_drift_mode must be zero, normal, or full")
            if params.ion_drift_mode not in {"normal", "full"}:
                raise ValueError("ion_drift_mode must be normal or full")
        for name in ("ion_density_m3", "density_scale_m3", "electron_temperature_ev",
                     "photoelectron_temperature_ev", "ion_mass_kg", "electron_mass_kg", "ion_pressure_factor"):
            value = getattr(params, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ion_temperature_ev", "photoelectron_density_m3"):
            value = getattr(params, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(params.electron_drift_mps):
            raise ValueError("electron_drift_mps must be finite")
        if not math.isfinite(params.ion_entry_speed_mps) or params.ion_entry_speed_mps <= 0:
            raise ValueError("Zero or negative ion entry speed is degenerate; specify a positive normal speed")
        ion_critical_potential(0.5*params.electron_temperature_ev*params.mach**2,
                               params.ion_pressure_factor*params.ion_temperature_ev)


    def _ion_density_hat(self, phi_hat):
        p = self.p
        return (p.ion_density_m3/p.density_scale_m3)*ion_density_ratio(
            np.asarray(phi_hat)*p.photoelectron_temperature_ev, 0.5*p.electron_temperature_ev*p.mach**2,
            p.ion_pressure_factor*p.ion_temperature_ev)


    def _validate_params_for_branch(self, branch: Branch) -> None:
        if branch not in {"A", "B", "C"}:
            raise ValueError(f"unknown branch: {branch}")


    def _swe_free_current_term(self, n_swe_inf_m3: float, a_swe: float) -> float:
        """Normalized free solar-wind electron current term from Eq. (16)."""
        p = self.p
        return n_swe_inf_m3 * (
            math.sqrt(p.electron_temperature_ev / p.photoelectron_temperature_ev) * math.exp(-(a_swe**2))
            + math.sqrt(math.pi) * (p.electron_drift_mps / p.v_phe_th_mps) * erfc(a_swe)
        )


    def _type_a_e2_sum_at_infinity(
        self, phi0_V: float, phi_m_V: float, n_swe_inf_m3: float
    ) -> float:
        """First integral of the orbit-mapped charge density; regular at u=0."""
        p = self.p
        return -2 * self._integrate_rho("A", "upper", phi_m_V / p.photoelectron_temperature_ev, 0.0,
                                      phi0_V / p.photoelectron_temperature_ev, phi_m_V / p.photoelectron_temperature_ev,
                                      n_swe_inf_m3 / p.density_scale_m3)


    def _integrate_rho(self, branch, side, lo, hi, phi0, phim, density):
        t = POTENTIAL_NODES
        phi = lo + (hi - lo) * np.sin(0.5 * np.pi * t)**2
        if branch == "A":
            dens = self._densities_hat_type_a_side(phi, phi0, density, phim, side)
        else:
            dens = self._densities_hat(branch, phi, phi0, density, phim)
        return float(np.sum(POTENTIAL_WEIGHTS * self._rho_hat_from_densities(dens) *
                            (hi - lo) * 0.5 * np.pi * np.sin(np.pi * t)))

    def _type_a_connection_residual(self, phi0, phim, density):
        if phim >= 0:
            return math.nan
        p = self.p
        depth = -phim
        energy = .5*p.electron_temperature_ev*p.mach**2
        pressure = p.ion_pressure_factor*p.ion_temperature_ev
        sound = 2*energy-pressure
        if p.u == 0 and depth*p.photoelectron_temperature_ev < 1e-8*min(
                p.electron_temperature_ev, p.photoelectron_temperature_ev, sound*(sound/(6*energy+pressure))):
            # Integrate density differences from neutral infinity. Factoring
            # sqrt(depth) retains the residual when individual densities round
            # to their upstream values.
            f = np.cos(.5*np.pi*POTENTIAL_NODES)**2
            root_depth = math.sqrt(depth)
            phi = -depth*f*p.photoelectron_temperature_ev
            if pressure == 0:
                ratio = np.sqrt(1-phi/energy)
                ions = -root_depth*f*p.photoelectron_temperature_ev/(energy*ratio*(1+ratio))
            else:
                ions = -root_depth*f*p.photoelectron_temperature_ev*(1/sound+(6*energy-pressure)*phi/(2*sound**3))
            ions *= p.ion_density_m3/p.density_scale_m3

            def scaled_delta(temperature_ratio, sign):
                psi = -depth*f/temperature_ratio
                s0 = math.sqrt(depth/temperature_ratio)
                s = s0*np.sqrt(1-f)
                polynomial = 1-(s*s+s*s0+s0*s0)/3+(s**4+s**3*s0+s*s*s0*s0+s*s0**3+s0**4)/10
                delta_erf = -2*f*polynomial/(np.sqrt(np.pi*temperature_ratio)*(1+np.sqrt(1-f)))
                delta_exp = -root_depth*f/temperature_ratio*(1+psi*(.5+psi*(1/6+psi/24)))
                return delta_exp*(1+sign*erf(s0))+sign*np.exp(psi)*delta_erf

            electrons = .5*density*scaled_delta(p.tau, 1)
            photos = .5*p.photoelectron_density_m3/p.density_scale_m3*math.exp(-phi0)*scaled_delta(1., -1)
            coefficient = .5*(1+erf(math.sqrt(depth/p.tau)))
            pe0 = .5*p.photoelectron_density_m3*math.exp(-phi0)*erfc(math.sqrt(depth))
            neutral = (p.ion_density_m3-pe0)/coefficient/p.density_scale_m3
            rho0 = (neutral-density)*coefficient/root_depth
            integral = float(np.sum(POTENTIAL_WEIGHTS*(ions-electrons-photos)*.5*np.pi*np.sin(np.pi*POTENTIAL_NODES)))
            return -2*(integral+rho0)*p.density_scale_m3/p.ion_density_m3
        return (-2*self._integrate_rho("A", "upper", phim, 0., phi0, phim, density)/(-phim))*\
               (self.p.density_scale_m3/self.p.ion_density_m3)/math.sqrt(-phim)


    def _residuals_type_a(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        phi0_V, phi_m_V, n_swe_inf_m3 = x
        if phi_m_V >= 0.0 or phi_m_V >= phi0_V or n_swe_inf_m3 <= 0.0:
            return np.array([1e6, 1e6, 1e6], dtype=float)
        if phi0_V > ion_critical_potential(0.5*p.electron_temperature_ev*p.mach**2,
                                          p.ion_pressure_factor*p.ion_temperature_ev):
            return np.array([1e6, 1e6, 1e6], dtype=float)

        a_swe = math.sqrt(max(0.0, -phi_m_V / p.electron_temperature_ev)) - p.u
        a_phe = math.sqrt(max(0.0, -phi_m_V / p.photoelectron_temperature_ev))
        ion_term = (
            p.ion_density_m3
            * math.sqrt(2.0 * math.pi * p.electron_temperature_ev / p.photoelectron_temperature_ev * p.electron_mass_kg / p.ion_mass_kg)
            * p.mach
        )

        # Eq. (14) Charge Neutrality at Infinity
        r1 = (
            0.5 * n_swe_inf_m3 * (1.0 + 2.0 * erf(p.u) + erf(a_swe))
            + 0.5 * p.photoelectron_density_m3 * math.exp(-phi0_V / p.photoelectron_temperature_ev) * (1.0 - erf(a_phe))
            - p.ion_density_m3
        )

        # Eq. (16) Zero Net Current Density at Infinity (equivalent: at Z = 0)
        r2 = (
            p.photoelectron_density_m3 * math.exp((phi_m_V - phi0_V) / p.photoelectron_temperature_ev)
            - self._swe_free_current_term(n_swe_inf_m3, a_swe)
            + ion_term
        )

        # Eq. (24) for Type A, evaluated at phi(infty)=0.
        r3 = self._type_a_e2_sum_at_infinity(phi0_V, phi_m_V, n_swe_inf_m3)

        return np.array([r1, r2, r3], dtype=float)


    def _residuals_type_b(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        phi0_V, n_swe_inf_m3 = x
        if phi0_V <= 0.0 or n_swe_inf_m3 <= 0.0:
            return np.array([1e6, 1e6], dtype=float)

        ion_term = (
            p.ion_density_m3
            * math.sqrt(2.0 * math.pi * p.electron_temperature_ev / p.photoelectron_temperature_ev * p.electron_mass_kg / p.ion_mass_kg)
            * p.mach
        )

        # Eq. (14) Charge Neutrality at Infinity
        r1 = (
            0.5 * n_swe_inf_m3 * (1.0 + erf(p.u))
            + 0.5 * p.photoelectron_density_m3 * math.exp(-phi0_V / p.photoelectron_temperature_ev)
            - p.ion_density_m3
        )

        # Eq. (16) Zero Net Current Density at Infinity (equivalent: at Z = 0)
        r2 = (
            p.photoelectron_density_m3 * math.exp(-phi0_V / p.photoelectron_temperature_ev)
            - self._swe_free_current_term(n_swe_inf_m3, -p.u)
            + ion_term
        )
        return np.array([r1, r2], dtype=float)


    def _residuals_type_c(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        phi0_V, n_swe_inf_m3 = x
        if phi0_V >= 0.0 or n_swe_inf_m3 <= 0.0:
            return np.array([1e6, 1e6], dtype=float)

        a_swe = math.sqrt(max(0.0, -phi0_V / p.electron_temperature_ev)) - p.u
        a_phe = math.sqrt(max(0.0, -phi0_V / p.photoelectron_temperature_ev))
        ion_term = (
            p.ion_density_m3
            * math.sqrt(2.0 * math.pi * p.electron_temperature_ev / p.photoelectron_temperature_ev * p.electron_mass_kg / p.ion_mass_kg)
            * p.mach
        )

        # Eq. (14) Charge Neutrality at Infinity
        r1 = (
            0.5 * n_swe_inf_m3 * (1.0 + 2.0 * erf(p.u) + erf(a_swe))
            + 0.5 * p.photoelectron_density_m3 * math.exp(-phi0_V / p.photoelectron_temperature_ev) * erfc(a_phe)
            - p.ion_density_m3
        )

        # Eq. (16) Zero Net Current Density at Infinity (equivalent: at Z = 0)
        r2 = p.photoelectron_density_m3 - self._swe_free_current_term(n_swe_inf_m3, a_swe) + ion_term

        return np.array([r1, r2], dtype=float)


    def _validate_profile_root(self, branch, phi0, phim, density, tolerance=0.):
        """Raise unless the root has a real connecting profile; return the accepted negative-E^2 band.

        With inward electron drift, A/C are accepted only for tolerance>0 when E^2<0 is confined next to
        upstream within tolerance*|phi_m| (A) or tolerance*|phi_0| (C). The band is returned in units of the
        photoelectron temperature and is 0 otherwise.
        """
        if branch == "B" and phi0 > 0:
            ambient_edge = (
                density * self.p.density_scale_m3 * math.exp(-self.p.u**2)
                / math.sqrt(math.pi * self.p.electron_temperature_ev)
            )
            photo_edge = self.p.photoelectron_density_m3 * math.exp(-phi0) / math.sqrt(math.pi * self.p.photoelectron_temperature_ev)
            if ambient_edge - photo_edge > 128 * np.finfo(float).eps * max(abs(ambient_edge), abs(photo_edge)):
                raise RuntimeError("Type B has negative field squared arbitrarily near upstream infinity")
        drifting = branch in ("A", "C") and self.p.u > 0
        if drifting and tolerance <= 0:
            raise RuntimeError("algebraic root has no semi-infinite profile: inward drift with reflected slow "
                               "electrons makes E^2 negative near neutral infinity; specify zero electron drift "
                               "for the nondrifting model or a positive upstream_band_tolerance")
        try:
            self._ion_density_hat(max(phi0, 0))
        except ValueError as exc:
            raise RuntimeError("algebraic root blocks the upstream-connected ion flow") from exc
        if branch == "A":
            connection = self._type_a_connection_residual(phi0, phim, density)
            if not math.isfinite(connection):
                raise FloatingPointError("upper connection residual is non-finite")
            if abs(connection) > 1e-7:
                raise RuntimeError("internal minimum does not connect to zero field at infinity")
        def field_squared(side, phi):
            return (-2*self._integrate_rho(branch, side, phim, phi, phi0, phim, density) if branch == "A" else
                    2*self._integrate_rho(branch, side, phi, 0., phi0, phim, density))

        segments = [("monotonic", phi0, 0.)] if branch != "A" else [("lower", phim, phi0), ("upper", phim, 0.)]
        values, lower, upstream = [], [], []
        for side, lo, hi in segments:
            for phi in np.linspace(lo, hi, 129):
                e2 = field_squared(side, phi)
                values.append(e2)
                (lower if side == "lower" else upstream).append((-phi, e2))
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("profile field integration is non-finite")
        threshold = -1e-8*max(1., max(values))
        if not drifting:
            if min(values) < threshold:
                raise RuntimeError("algebraic root has no real connecting field profile")
            return 0.
        if lower and min(e2 for _, e2 in lower) < threshold:
            raise RuntimeError("algebraic root has no real connecting field profile")
        # The log term makes E^2<0 touch upstream; measure the band by sign (it can be below the threshold for C).
        side = "upper" if branch == "A" else "monotonic"
        reference = -phim if branch == "A" else -phi0
        for j in range(1, 49):
            depth = reference*.5**j
            upstream.append((depth, field_squared(side, -depth)))
        if not all(math.isfinite(e2) for _, e2 in upstream):
            raise FloatingPointError("profile field integration is non-finite")
        band = max((depth for depth, e2 in upstream if e2 < 0), default=0.)
        if band > 0:
            outside = min((depth for depth, _ in upstream if depth > band), default=reference)
            for _ in range(40):
                mid = .5*(band+outside)
                if field_squared(side, -mid) < 0:
                    band = mid
                else:
                    outside = mid
        if band > tolerance*reference:
            raise RuntimeError("negative E^2 next to upstream is wider than upstream_band_tolerance allows")
        return float(band)


    def _densities_hat_type_a_side(
        self,
        phi_hat: np.ndarray,
        phi0_hat: float,
        n_swe_inf_hat: float,
        phi_m_hat: float,
        side: TypeASide,
    ) -> Dict[str, np.ndarray]:
        p = self.p
        phi_hat = np.asarray(phi_hat, dtype=float)
        tau = p.tau
        source_density_hat = p.photoelectron_density_m3 / p.density_scale_m3

        n_swi_hat = self._ion_density_hat(phi_hat)

        free, reflected = electron_density(phi_hat / tau, phi_m_hat / tau, p.u)
        n_swe_f_hat = n_swe_inf_hat * free
        s_phe = np.sqrt(np.maximum(0.0, phi_hat - phi_m_hat))
        n_phe_f_hat = 0.5 * source_density_hat * np.exp(phi_hat - phi0_hat) * (1.0 - erf(s_phe))

        if side == "lower":
            n_swe_r_hat = np.zeros_like(phi_hat)
            n_phe_c_hat = source_density_hat * np.exp(phi_hat - phi0_hat) * erf(s_phe)
        elif side == "upper":
            n_swe_r_hat = n_swe_inf_hat * reflected
            n_phe_c_hat = np.zeros_like(phi_hat)
        else:
            raise ValueError(f"unknown Type-A side: {side}")

        return {
            "n_swi_hat": n_swi_hat,
            "n_swe_f_hat": n_swe_f_hat,
            "n_swe_r_hat": n_swe_r_hat,
            "n_phe_f_hat": n_phe_f_hat,
            "n_phe_c_hat": n_phe_c_hat,
        }


    def _densities_hat(
        self,
        branch: Branch,
        phi_hat: np.ndarray,
        phi0_hat: float,
        n_swe_inf_hat: float,
        phi_m_hat: float | None = None,
    ) -> Dict[str, np.ndarray]:
        p = self.p
        phi_hat = np.asarray(phi_hat, dtype=float)
        tau = p.tau
        source_density_hat = p.photoelectron_density_m3 / p.density_scale_m3
        n_swi_hat = self._ion_density_hat(phi_hat)

        if branch == "A":
            raise ValueError(
                "Type A requires side-resolved densities; use _densities_hat_type_a_side()."
            )
        elif branch == "B":
            s_phe = np.sqrt(np.maximum(0.0, phi_hat))
            free, _ = electron_density(phi_hat / tau, 0.0, p.u)
            n_swe_f_hat = n_swe_inf_hat * free
            n_swe_r_hat = np.zeros_like(phi_hat)
            n_phe_f_hat = (
                0.5 * source_density_hat * np.exp(phi_hat - phi0_hat) * (1.0 - erf(s_phe))
            )
            n_phe_c_hat = source_density_hat * np.exp(phi_hat - phi0_hat) * erf(s_phe)
        elif branch == "C":
            free, reflected = electron_density(phi_hat / tau, phi0_hat / tau, p.u)
            s_phe = np.sqrt(np.maximum(0.0, phi_hat - phi0_hat))
            n_swe_f_hat = n_swe_inf_hat * free
            n_swe_r_hat = n_swe_inf_hat * reflected
            n_phe_f_hat = 0.5 * source_density_hat * np.exp(phi_hat - phi0_hat) * erfc(s_phe)
            n_phe_c_hat = np.zeros_like(phi_hat)
        else:
            raise ValueError(f"unknown branch: {branch}")

        return {
            "n_swi_hat": n_swi_hat,
            "n_swe_f_hat": n_swe_f_hat,
            "n_swe_r_hat": n_swe_r_hat,
            "n_phe_f_hat": n_phe_f_hat,
            "n_phe_c_hat": n_phe_c_hat,
        }


    @staticmethod
    def _rho_hat_from_densities(dens: Dict[str, np.ndarray]) -> np.ndarray:
        return (
            dens["n_swi_hat"]
            - dens["n_swe_f_hat"]
            - dens["n_swe_r_hat"]
            - dens["n_phe_f_hat"]
            - dens["n_phe_c_hat"]
        )
