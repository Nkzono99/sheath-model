"""One-dimensional sheath model with orbit-mapped background electrons.

Zhao A/B/C population topology, cold ions and a Maxwell photoelectron source.
See docs/kinetic-model.md for the reservoir closure and admissibility conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root
from scipy.special import erf, erfc
from ._orbits import electron_density, POTENTIAL_NODES, POTENTIAL_WEIGHTS

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31
MP = 1.67262192369e-27

Branch = Literal["A", "B", "C"]
DriftMode = Literal["full", "normal"]
TypeASide = Literal["lower", "upper"]
Species = Literal["swi", "swe", "phe", "all"]
ZUnit = Literal["hat", "m"]


@dataclass(frozen=True)
class ZhaoParams:
    alpha_deg: float = 60.0

    # Table-I values from Zhao et al.
    n_swi_inf_cm3: float = 8.7
    n_phe_ref_cm3: float = 64.0
    T_swe_eV: float = 12.0
    T_phe_eV: float = 2.2
    v_sw_total_mps: float = 468e3
    m_i_kg: float = MP

    # Choice of which drift component enters the 1-D algebra.
    # The Zhao model is 1-D along the sheath normal, so the projected normal
    # drift is the paper-consistent default.
    # "full"   : use the full solar-wind speed in the 1-D formulas.
    # "normal" : use v_sw * sin(alpha).
    electron_drift_mode: Literal["full", "normal", "zero"] = "normal"
    ion_drift_mode: DriftMode = "normal"

    zmax_hat: float = 80.0
    n_profile_grid: int = 600
    n_type_a_grid: int = 8000
    profile_phi_tol_hat: float = 1.0e-3
    type_a_phi_m_eps_hat: float = 1.0e-5

    @property
    def alpha_rad(self) -> float:
        return math.radians(self.alpha_deg)

    @property
    def n_swi_inf_m3(self) -> float:
        return self.n_swi_inf_cm3 * 1e6

    @property
    def n_phe_ref_m3(self) -> float:
        return self.n_phe_ref_cm3 * 1e6

    @property
    def n_phe0_m3(self) -> float:
        return self.n_phe_ref_m3 * math.sin(self.alpha_rad)

    @property
    def v_swe_th_mps(self) -> float:
        return math.sqrt(2.0 * QE * self.T_swe_eV / ME)

    @property
    def v_phe_th_mps(self) -> float:
        return math.sqrt(2.0 * QE * self.T_phe_eV / ME)

    @property
    def cs_mps(self) -> float:
        return math.sqrt(QE * self.T_swe_eV / self.m_i_kg)

    @property
    def v_sw_normal_mps(self) -> float:
        return self.v_sw_total_mps * math.sin(self.alpha_rad)

    @property
    def v_d_electron_mps(self) -> float:
        if self.electron_drift_mode == "zero":
            return 0.0
        return (
            self.v_sw_total_mps
            if self.electron_drift_mode == "full"
            else self.v_sw_normal_mps
        )

    @property
    def v_d_ion_mps(self) -> float:
        return (
            self.v_sw_total_mps
            if self.ion_drift_mode == "full"
            else self.v_sw_normal_mps
        )

    @property
    def mach(self) -> float:
        return self.v_d_ion_mps / self.cs_mps

    @property
    def u(self) -> float:
        return self.v_d_electron_mps / self.v_swe_th_mps

    @property
    def tau(self) -> float:
        return self.T_swe_eV / self.T_phe_eV

    @property
    def lambda_d_phe_ref_m(self) -> float:
        return math.sqrt(EPS0 * QE * self.T_phe_eV / (self.n_phe_ref_m3 * QE * QE))


class ZhaoSheathSolver:
    def __init__(self, params: ZhaoParams):
        self.p = params

    # ------------------------------------------------------------------
    # Algebraic unknown solver
    # ------------------------------------------------------------------
    def _validate_params_for_branch(self, branch: Branch) -> None:
        p = self.p
        if abs(p.v_d_ion_mps) < 1.0e-12:
            raise ValueError(
                f"branch {branch} is degenerate with ion_drift_mode={p.ion_drift_mode!r} at alpha={p.alpha_deg:g} deg: "
                "the 1-D normal ion drift is zero, so the Zhao ion-density model is undefined. "
                "Use alpha > 0 or switch to the full-drift modes explicitly."
            )

    def _swe_free_current_term(self, n_swe_inf_m3: float, a_swe: float) -> float:
        """Normalized free solar-wind electron current term from Eq. (16)."""
        p = self.p
        return n_swe_inf_m3 * (
            math.sqrt(p.T_swe_eV / p.T_phe_eV) * math.exp(-(a_swe**2))
            + math.sqrt(math.pi) * (p.v_d_electron_mps / p.v_phe_th_mps) * erfc(a_swe)
        )

    def _type_a_e2_sum_at_infinity(
        self, phi0_V: float, phi_m_V: float, n_swe_inf_m3: float
    ) -> float:
        """First integral of the orbit-mapped charge density; regular at u=0."""
        p = self.p
        return -2 * self._integrate_rho("A", "upper", phi_m_V / p.T_phe_eV, 0.0,
                                      phi0_V / p.T_phe_eV, phi_m_V / p.T_phe_eV,
                                      n_swe_inf_m3 / p.n_phe_ref_m3)

    def _integrate_rho(self, branch, side, lo, hi, phi0, phim, density):
        t = POTENTIAL_NODES
        phi = lo + (hi - lo) * np.sin(0.5 * np.pi * t)**2
        if branch == "A":
            dens = self._densities_hat_type_a_side(phi, phi0, density, phim, side)
        else:
            dens = self._densities_hat(branch, phi, phi0, density, phim)
        return float(np.sum(POTENTIAL_WEIGHTS * self._rho_hat_from_densities(dens) *
                            (hi - lo) * 0.5 * np.pi * np.sin(np.pi * t)))

    def _residuals_type_a(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        phi0_V, phi_m_V, n_swe_inf_m3 = x
        if phi_m_V >= 0.0 or phi_m_V >= phi0_V or n_swe_inf_m3 <= 0.0:
            return np.array([1e6, 1e6, 1e6], dtype=float)

        a_swe = math.sqrt(max(0.0, -phi_m_V / p.T_swe_eV)) - p.u
        a_phe = math.sqrt(max(0.0, -phi_m_V / p.T_phe_eV))
        ion_term = (
            p.n_swi_inf_m3
            * math.sqrt(2.0 * math.pi * p.T_swe_eV / p.T_phe_eV * ME / p.m_i_kg)
            * p.mach
        )

        # Eq. (14) Charge Neutrality at Infinity
        r1 = (
            0.5 * n_swe_inf_m3 * (1.0 + 2.0 * erf(p.u) + erf(a_swe))
            + 0.5 * p.n_phe0_m3 * math.exp(-phi0_V / p.T_phe_eV) * (1.0 - erf(a_phe))
            - p.n_swi_inf_m3
        )

        # Eq. (16) Zero Net Current Density at Infinity (equivalent: at Z = 0)
        r2 = (
            p.n_phe0_m3 * math.exp((phi_m_V - phi0_V) / p.T_phe_eV)
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
            p.n_swi_inf_m3
            * math.sqrt(2.0 * math.pi * p.T_swe_eV / p.T_phe_eV * ME / p.m_i_kg)
            * p.mach
        )

        # Eq. (14) Charge Neutrality at Infinity
        r1 = (
            0.5 * n_swe_inf_m3 * (1.0 + erf(p.u))
            + 0.5 * p.n_phe0_m3 * math.exp(-phi0_V / p.T_phe_eV)
            - p.n_swi_inf_m3
        )

        # Eq. (16) Zero Net Current Density at Infinity (equivalent: at Z = 0)
        r2 = (
            p.n_phe0_m3 * math.exp(-phi0_V / p.T_phe_eV)
            - self._swe_free_current_term(n_swe_inf_m3, -p.u)
            + ion_term
        )
        return np.array([r1, r2], dtype=float)

    def _residuals_type_c(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        phi0_V, n_swe_inf_m3 = x
        if phi0_V >= 0.0 or n_swe_inf_m3 <= 0.0:
            return np.array([1e6, 1e6], dtype=float)

        a_swe = math.sqrt(max(0.0, -phi0_V / p.T_swe_eV)) - p.u
        a_phe = math.sqrt(max(0.0, -phi0_V / p.T_phe_eV))
        ion_term = (
            p.n_swi_inf_m3
            * math.sqrt(2.0 * math.pi * p.T_swe_eV / p.T_phe_eV * ME / p.m_i_kg)
            * p.mach
        )

        # Eq. (14) Charge Neutrality at Infinity
        r1 = (
            0.5 * n_swe_inf_m3 * (1.0 + 2.0 * erf(p.u) + erf(a_swe))
            + 0.5 * p.n_phe0_m3 * math.exp(-phi0_V / p.T_phe_eV) * erfc(a_phe)
            - p.n_swi_inf_m3
        )

        # Eq. (16) Zero Net Current Density at Infinity (equivalent: at Z = 0)
        r2 = p.n_phe0_m3 - self._swe_free_current_term(n_swe_inf_m3, a_swe) + ion_term

        return np.array([r1, r2], dtype=float)

    def _try_root_guesses(self, func, guesses: Iterable[np.ndarray]) -> np.ndarray:
        def scaled(x):
            values = func(x).copy()
            values[:2] /= self.p.n_phe_ref_m3
            return values
        best = None
        best_norm = float("inf")
        for guess in guesses:
            sol = root(scaled, np.asarray(guess, dtype=float), method="hybr", options={"xtol": 1e-10})
            fnorm = (
                float(np.linalg.norm(sol.fun)) if sol.fun is not None else float("inf")
            )
            if np.all(np.isfinite(sol.x)) and fnorm < best_norm:
                best = sol
                best_norm = fnorm
            if sol.success and np.all(np.isfinite(sol.x)) and fnorm < 1e-10:
                return np.asarray(sol.x, dtype=float)
        if best is None or best_norm > 1e-10:
            raise RuntimeError(
                f"root solve failed; best residual norm={best_norm:.3e}, "
                f"x={None if best is None else best.x}, fun={None if best is None else best.fun}"
            )
        return np.asarray(best.x, dtype=float)

    def solve_unknowns(
        self, branch: Branch, guess: tuple[float, ...] | None = None
    ) -> Dict[str, float | str]:
        p = self.p
        self._validate_params_for_branch(branch)
        if branch == "A":
            guesses = (
                [np.array(guess, dtype=float)]
                if guess is not None
                else [
                    np.array([3.6, -0.5, 8.2e6]),
                    np.array([2.8, -0.3, 8.0e6]),
                    np.array([4.5, -0.8, 8.4e6]),
                    np.array([-0.4, -1.7, 8.0e6]),
                    np.array([-2.2, -4.4, 8.0e6]),
                ]
            )
            phi0_V, phi_m_V, n_swe_inf_m3 = self._try_root_guesses(
                self._residuals_type_a, guesses
            )
        elif branch == "B":
            guesses = (
                [np.array(guess, dtype=float)]
                if guess is not None
                else [
                    np.array([1.3, 7.0e6]),
                    np.array([0.8, 6.5e6]),
                    np.array([2.0, 7.8e6]),
                ]
            )
            phi0_V, n_swe_inf_m3 = self._try_root_guesses(
                self._residuals_type_b, guesses
            )
            phi_m_V = math.nan
        elif branch == "C":
            guesses = (
                [np.array(guess, dtype=float)]
                if guess is not None
                else [
                    np.array([-0.5, 6.0e6]),
                    np.array([-2.0, 7.0e6]),
                    np.array([-5.0, 8.0e6]),
                    np.array([-10.0, 8.2e6]),
                    np.array([-15.0, 8.5e6]),
                ]
            )
            phi0_V, n_swe_inf_m3 = self._try_root_guesses(
                self._residuals_type_c, guesses
            )
            phi_m_V = phi0_V
        else:
            raise ValueError(f"unknown branch: {branch}")

        self._validate_profile_root(branch, phi0_V / p.T_phe_eV,
                                    0.0 if branch == "B" else phi_m_V / p.T_phe_eV,
                                    n_swe_inf_m3 / p.n_phe_ref_m3)
        return {
            "branch": branch,
            "phi0_V": float(phi0_V),
            "phi_m_V": float(phi_m_V),
            "n_swe_inf_m3": float(n_swe_inf_m3),
            "phi0_hat": float(phi0_V / p.T_phe_eV),
            "phi_m_hat": (
                float(phi_m_V / p.T_phe_eV) if math.isfinite(phi_m_V) else math.nan
            ),
            "n_swe_inf_hat": float(n_swe_inf_m3 / p.n_phe_ref_m3),
            "electron_drift_mode": p.electron_drift_mode,
            "ion_drift_mode": p.ion_drift_mode,
            "v_d_electron_mps": p.v_d_electron_mps,
            "v_d_ion_mps": p.v_d_ion_mps,
        }

    def _validate_profile_root(self, branch, phi0, phim, density):
        if branch in ("A", "C") and self.p.u > 0:
            raise RuntimeError("algebraic root has no semi-infinite profile: inward drift with reflected slow "
                               "electrons makes E^2 negative near neutral infinity; use electron_drift_mode='zero' "
                               "for the nondrifting model")
        if 1 - 2 * max(phi0, 0) / (self.p.tau * self.p.mach**2) <= 0:
            raise RuntimeError("algebraic root blocks cold ions")
        segments = [("monotonic", phi0, 0.)] if branch != "A" else [("lower", phim, phi0), ("upper", phim, 0.)]
        values = []
        for side, lo, hi in segments:
            for phi in np.linspace(lo, hi, 129):
                e2 = (-2*self._integrate_rho(branch, side, phim, phi, phi0, phim, density) if branch == "A" else
                      2*self._integrate_rho(branch, side, phi, 0., phi0, phim, density))
                values.append(e2)
        if not np.all(np.isfinite(values)) or min(values) < -1e-8*max(1., max(values)):
            raise RuntimeError("algebraic root has no real connecting field profile")

    # ------------------------------------------------------------------
    # Density helpers
    # ------------------------------------------------------------------
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
        sin_alpha = math.sin(p.alpha_rad)

        arg_ion = 1.0 - 2.0 * phi_hat / (tau * p.mach * p.mach)
        if np.any(arg_ion <= 0.0):
            raise ValueError("ion density argument became non-positive")
        n_swi_hat = (p.n_swi_inf_m3 / p.n_phe_ref_m3) * arg_ion ** (-0.5)

        free, reflected = electron_density(phi_hat / tau, phi_m_hat / tau, p.u)
        n_swe_f_hat = n_swe_inf_hat * free
        s_phe = np.sqrt(np.maximum(0.0, phi_hat - phi_m_hat))
        n_phe_f_hat = 0.5 * sin_alpha * np.exp(phi_hat - phi0_hat) * (1.0 - erf(s_phe))

        if side == "lower":
            n_swe_r_hat = np.zeros_like(phi_hat)
            n_phe_c_hat = sin_alpha * np.exp(phi_hat - phi0_hat) * erf(s_phe)
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
        sin_alpha = math.sin(p.alpha_rad)
        n_swi_hat = (p.n_swi_inf_m3 / p.n_phe_ref_m3) * (
            1.0 - 2.0 * phi_hat / (tau * p.mach * p.mach)
        ) ** (-0.5)

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
                0.5 * sin_alpha * np.exp(phi_hat - phi0_hat) * (1.0 - erf(s_phe))
            )
            n_phe_c_hat = sin_alpha * np.exp(phi_hat - phi0_hat) * erf(s_phe)
        elif branch == "C":
            free, reflected = electron_density(phi_hat / tau, phi0_hat / tau, p.u)
            s_phe = np.sqrt(np.maximum(0.0, phi_hat - phi0_hat))
            n_swe_f_hat = n_swe_inf_hat * free
            n_swe_r_hat = n_swe_inf_hat * reflected
            n_phe_f_hat = 0.5 * sin_alpha * np.exp(phi_hat - phi0_hat) * erfc(s_phe)
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

    # ------------------------------------------------------------------
    # Type-A profile: piecewise first-integral reconstruction
    # ------------------------------------------------------------------
    def _type_a_branch_from_minimum(
        self,
        phi_nodes_asc: np.ndarray,
        phi0_hat: float,
        n_swe_inf_hat: float,
        phi_m_hat: float,
        side: TypeASide,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Build one Type-A branch starting just above the potential minimum."""
        phi_nodes_asc = np.asarray(phi_nodes_asc, dtype=float)
        if phi_nodes_asc.ndim != 1 or len(phi_nodes_asc) < 2:
            raise ValueError("phi_nodes_asc must be a 1D array with at least 2 points")
        if not np.all(np.diff(phi_nodes_asc) > 0.0):
            raise ValueError("phi_nodes_asc must be strictly increasing")
        if phi_nodes_asc[0] <= phi_m_hat:
            raise ValueError("phi_nodes_asc must start above phi_m_hat")

        dens_nodes = self._densities_hat_type_a_side(
            phi_nodes_asc, phi0_hat, n_swe_inf_hat, phi_m_hat, side=side
        )
        rho_nodes = self._rho_hat_from_densities(dens_nodes)

        dens_m = self._densities_hat_type_a_side(
            np.array([phi_m_hat], dtype=float),
            phi0_hat,
            n_swe_inf_hat,
            phi_m_hat,
            side=side,
        )
        rho_m = float(self._rho_hat_from_densities(dens_m)[0])
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

        return s_nodes, e2_nodes, dens_nodes, np.asarray(rho_nodes, dtype=float)

    def _build_type_a_profile(
        self, uk: Dict[str, float | str]
    ) -> Dict[str, np.ndarray | float | str]:
        p = self.p
        phi0_hat = float(uk["phi0_hat"])
        phi_m_hat = float(uk["phi_m_hat"])
        n_swe_inf_hat = float(uk["n_swe_inf_hat"])

        phi_m_eps = min(p.type_a_phi_m_eps_hat, 0.05 * max(1e-8, abs(phi_m_hat)))
        phi_m_eps = max(phi_m_eps, 1.0e-8)
        phi_end_hat = -abs(p.profile_phi_tol_hat)
        if phi_end_hat <= phi_m_hat:
            phi_end_hat = 0.5 * phi_m_hat

        ngrid = max(2000, p.n_type_a_grid)
        ngrid_upper = ngrid
        ngrid_lower = max(ngrid // 2, 1500)

        # Lower branch: surface -> z_m.
        x_lower = np.linspace(0.0, 1.0, ngrid_lower)
        phi_lower_asc = (phi_m_hat + phi_m_eps) + (
            phi0_hat - (phi_m_hat + phi_m_eps)
        ) * x_lower**2
        s_lower_asc, e2_lower_asc, dens_lower_asc, _rho_lower = (
            self._type_a_branch_from_minimum(
                phi_lower_asc, phi0_hat, n_swe_inf_hat, phi_m_hat, side="lower"
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
        s_upper_asc, e2_upper_asc, dens_upper_asc, _rho_upper = (
            self._type_a_branch_from_minimum(
                phi_upper_asc, phi0_hat, n_swe_inf_hat, phi_m_hat, side="upper"
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

        # Do not append an artificial phi=0 tail. If the asymptotic branch reaches
        # beyond the requested zmax_hat, clip it; otherwise keep the physically
        # reconstructed interval only.
        keep = z_hat <= p.zmax_hat
        z_hat = z_hat[keep]
        phi_hat = phi_hat[keep]
        ehat = ehat[keep]
        dens = {k: v[keep] for k, v in dens.items()}

        out: Dict[str, np.ndarray | float | str] = {
            **uk,
            "z_hat": z_hat,
            "z_m_hat": z_m_hat,
            "z_m_m": z_m_hat * p.lambda_d_phe_ref_m,
            "z_m_array_m": z_hat * p.lambda_d_phe_ref_m,
            "phi_hat": phi_hat,
            "phi_V": phi_hat * p.T_phe_eV,
            "dphi_dzhat": ehat,
            "E_Vpm": -(p.T_phe_eV / p.lambda_d_phe_ref_m) * ehat,
            "lambda_d_phe_ref_m": p.lambda_d_phe_ref_m,
            **dens,
        }
        out["n_total_hat"] = (
            out["n_swe_f_hat"]
            + out["n_swe_r_hat"]
            + out["n_phe_f_hat"]
            + out["n_phe_c_hat"]
        )
        out["rho_hat"] = out["n_swi_hat"] - out["n_total_hat"]
        return out

    # ------------------------------------------------------------------
    # Public profile API
    # ------------------------------------------------------------------
    def solve_profile(
        self, branch: Branch, guess_unknowns: tuple[float, ...] | None = None
    ) -> Dict[str, np.ndarray | float | str]:
        p = self.p
        uk = self.solve_unknowns(branch, guess_unknowns)

        if branch == "A":
            return self._build_type_a_profile(uk)

        phi0_hat = float(uk["phi0_hat"])
        phi_m_hat = uk["phi_m_hat"]
        n_swe_inf_hat = float(uk["n_swe_inf_hat"])

        # Integrate from the exact neutral upstream endpoint, then omit infinity.
        t = np.linspace(0., 1., max(600, p.n_profile_grid))
        phi_hat = phi0_hat * (1.-t)**2
        dens = self._densities_hat(branch, phi_hat, phi0_hat, n_swe_inf_hat, phi_m_hat)
        rho = self._rho_hat_from_densities(dens)
        e2 = -2*cumulative_trapezoid(rho[::-1], phi_hat[::-1], initial=0.)[::-1]
        cutoff = min(abs(p.profile_phi_tol_hat), .5*abs(phi0_hat))
        keep = np.abs(phi_hat) >= cutoff
        keep[-1] = False
        phi_hat, e2 = phi_hat[keep], e2[keep]
        if len(phi_hat) < 2 or np.any(e2 <= 0.) or not np.all(np.isfinite(e2)):
            raise RuntimeError("monotonic profile integral is not positive; refine the potential grid")
        z_hat = np.concatenate(([0.], np.cumsum(.5*np.abs(np.diff(phi_hat)) *
                                    (1/np.sqrt(e2[:-1]) + 1/np.sqrt(e2[1:])))))
        keep = z_hat <= p.zmax_hat
        phi_hat, e2, z_hat = phi_hat[keep], e2[keep], z_hat[keep]
        if len(z_hat) < 2:
            raise ValueError("zmax_hat contains fewer than two profile points")
        e_hat = -math.copysign(1., phi0_hat)*np.sqrt(e2)
        dens = self._densities_hat(branch, phi_hat, phi0_hat, n_swe_inf_hat, phi_m_hat)

        out: Dict[str, np.ndarray | float | str] = {
            **uk,
            "z_hat": z_hat,
            "z_m_hat": float(z_hat[np.argmin(phi_hat)]),
            "z_m_m": float(z_hat[np.argmin(phi_hat)] * p.lambda_d_phe_ref_m),
            "z_m_array_m": z_hat * p.lambda_d_phe_ref_m,
            "phi_hat": phi_hat,
            "phi_V": phi_hat * p.T_phe_eV,
            "dphi_dzhat": e_hat,
            "E_Vpm": -(p.T_phe_eV / p.lambda_d_phe_ref_m) * e_hat,
            "lambda_d_phe_ref_m": p.lambda_d_phe_ref_m,
            **dens,
        }
        out["n_total_hat"] = (
            out["n_swe_f_hat"]
            + out["n_swe_r_hat"]
            + out["n_phe_f_hat"]
            + out["n_phe_c_hat"]
        )
        out["rho_hat"] = out["n_swi_hat"] - out["n_total_hat"]
        return out

    def solve_auto(self) -> Dict[str, np.ndarray | float | str]:
        if self.p.alpha_deg < 20.0:
            order: list[Branch] = ["C", "A", "B"]
        else:
            order = ["A", "B", "C"]
        errs = []
        for br in order:
            try:
                return self.solve_profile(br)
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{br}: {exc}")
        raise RuntimeError("auto branch selection failed: " + " | ".join(errs))

    # ------------------------------------------------------------------
    # Local diagnostics: densities, fluxes, reduced 1D VDFs
    # ------------------------------------------------------------------
    def _to_z_hat(self, z: float, unit: ZUnit) -> float:
        if unit == "hat":
            return float(z)
        if unit == "m":
            return float(z) / self.p.lambda_d_phe_ref_m
        raise ValueError(f"unknown z unit: {unit}")

    @staticmethod
    def _interp_on_profile(profile: Dict[str, np.ndarray | float | str], key: str, z_hat: float) -> float:
        z_arr = np.asarray(profile["z_hat"], dtype=float)
        y_arr = np.asarray(profile[key], dtype=float)
        if z_hat < float(z_arr[0]) or z_hat > float(z_arr[-1]):
            raise ValueError(
                f"requested z_hat={z_hat:.6g} is outside the solved interval "
                f"[{float(z_arr[0]):.6g}, {float(z_arr[-1]):.6g}]"
            )
        return float(np.interp(z_hat, z_arr, y_arr))

    def sample_at_z(
        self,
        profile: Dict[str, np.ndarray | float | str],
        z: float,
        unit: ZUnit = "hat",
    ) -> Dict[str, float | str]:
        """Return a branch-consistent local state at a single position.

        Densities are recomputed from the local potential and branch formulas,
        rather than directly interpolated from the stored profile arrays, so that
        they stay exactly consistent with the local VDF and flux reconstruction.
        """
        p = self.p
        z_hat = self._to_z_hat(z, unit)
        branch = str(profile["branch"])
        if branch not in {"A", "B", "C"}:
            raise ValueError("profile does not contain a valid branch label")

        phi_hat = self._interp_on_profile(profile, "phi_hat", z_hat)
        dphi_dzhat = self._interp_on_profile(profile, "dphi_dzhat", z_hat)
        E_Vpm = self._interp_on_profile(profile, "E_Vpm", z_hat)
        phi0_hat = float(profile["phi0_hat"])
        phi_m_hat = float(profile["phi_m_hat"])
        n_swe_inf_hat = float(profile["n_swe_inf_hat"])
        z_m_hat = float(profile["z_m_hat"])

        if branch == "A":
            side: str = "lower" if z_hat <= z_m_hat else "upper"
            dens_hat = self._densities_hat_type_a_side(
                np.array([phi_hat], dtype=float),
                phi0_hat,
                n_swe_inf_hat,
                phi_m_hat,
                side=side,  # type: ignore[arg-type]
            )
        else:
            side = "monotonic"
            dens_hat = self._densities_hat(
                branch,  # type: ignore[arg-type]
                np.array([phi_hat], dtype=float),
                phi0_hat,
                n_swe_inf_hat,
                phi_m_hat,
            )

        dens_hat_scalar = {k: float(v[0]) for k, v in dens_hat.items()}
        dens_m3 = {k.replace("_hat", "_m3"): v * p.n_phe_ref_m3 for k, v in dens_hat_scalar.items()}
        n_total_hat = (
            dens_hat_scalar["n_swe_f_hat"]
            + dens_hat_scalar["n_swe_r_hat"]
            + dens_hat_scalar["n_phe_f_hat"]
            + dens_hat_scalar["n_phe_c_hat"]
        )
        rho_hat = dens_hat_scalar["n_swi_hat"] - n_total_hat

        arg_ion = 1.0 - 2.0 * phi_hat / (p.tau * p.mach * p.mach)
        if arg_ion <= 0.0:
            raise ValueError("local ion energy argument became non-positive")
        v_i_local = p.v_d_ion_mps * math.sqrt(arg_ion)

        if branch == "B":
            a_swe = math.sqrt(max(0.0, phi_hat / p.tau))
            a_phe = math.sqrt(max(0.0, phi_hat))
            swe_reflected_active = False
            phe_captured_active = True
        elif branch == "C":
            a_swe = math.sqrt(max(0.0, (phi_hat - phi0_hat) / p.tau))
            a_phe = math.sqrt(max(0.0, phi_hat - phi0_hat))
            swe_reflected_active = True
            phe_captured_active = False
        else:  # branch == "A"
            a_swe = math.sqrt(max(0.0, (phi_hat - phi_m_hat) / p.tau))
            a_phe = math.sqrt(max(0.0, phi_hat - phi_m_hat))
            swe_reflected_active = side == "upper"
            phe_captured_active = side == "lower"

        return {
            "branch": branch,
            "side": side,
            "z_hat": z_hat,
            "z_m": z_hat * p.lambda_d_phe_ref_m,
            "phi_hat": phi_hat,
            "phi_V": phi_hat * p.T_phe_eV,
            "phi0_hat": phi0_hat,
            "phi0_V": phi0_hat * p.T_phe_eV,
            "phi_m_hat": phi_m_hat,
            "phi_m_V": phi_m_hat * p.T_phe_eV if math.isfinite(phi_m_hat) else math.nan,
            "z_m_hat": z_m_hat,
            "dphi_dzhat": dphi_dzhat,
            "E_Vpm": E_Vpm,
            "n_swe_inf_hat": n_swe_inf_hat,
            "n_swe_inf_m3": n_swe_inf_hat * p.n_phe_ref_m3,
            "a_swe": a_swe,
            "a_phe": a_phe,
            "vcut_swe_mps": a_swe * p.v_swe_th_mps,
            "vcut_phe_mps": a_phe * p.v_phe_th_mps,
            "v_i_mps": v_i_local,
            "swe_reflected_active": swe_reflected_active,
            "phe_captured_active": phe_captured_active,
            "n_swi_hat": dens_hat_scalar["n_swi_hat"],
            "n_swe_f_hat": dens_hat_scalar["n_swe_f_hat"],
            "n_swe_r_hat": dens_hat_scalar["n_swe_r_hat"],
            "n_phe_f_hat": dens_hat_scalar["n_phe_f_hat"],
            "n_phe_c_hat": dens_hat_scalar["n_phe_c_hat"],
            "n_total_hat": n_total_hat,
            "rho_hat": rho_hat,
            **dens_m3,
            "n_total_m3": n_total_hat * p.n_phe_ref_m3,
            "rho_m3": rho_hat * p.n_phe_ref_m3,
        }

    def _velocity_grid_for_species(
        self,
        state: Dict[str, float | str],
        species: Species,
        n_v: int,
        n_sigma: float,
    ) -> np.ndarray:
        p = self.p
        if n_v < 201:
            raise ValueError("n_v must be >= 201")
        if n_v % 2 == 0:
            n_v += 1

        if species == "swe":
            vmax = max(
                state["vcut_swe_mps"],
                p.v_swe_th_mps * math.sqrt(max(0.0, (abs(p.u) + n_sigma)**2 + float(state["phi_hat"])/p.tau)),
            )
            vmax = float(vmax)
            return np.linspace(-vmax, vmax, n_v)
        if species == "phe":
            vmax = max(
                state["vcut_phe_mps"],
                p.v_phe_th_mps * math.sqrt(n_sigma**2 + max(0.0, float(state["phi_hat"])-float(state["phi0_hat"]))),
            )
            vmax = float(vmax)
            return np.linspace(-vmax, vmax, n_v)
        if species == "swi":
            v_i = abs(float(state["v_i_mps"]))
            vmax = max(v_i * 1.4, v_i + 6.0 * self.p.cs_mps, 2.0 * self.p.cs_mps)
            return np.linspace(-vmax, vmax, n_v)
        raise ValueError(f"unknown species for grid construction: {species}")

    def _swe_vdf_components(self, state: Dict[str, float | str], vz_mps: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.p
        vz = np.asarray(vz_mps, dtype=float)
        amp = float(state["n_swe_inf_m3"]) / (math.sqrt(math.pi) * p.v_swe_th_mps)
        a = float(state["a_swe"])
        vcut = float(state["vcut_swe_mps"])
        w = vz / p.v_swe_th_mps
        upstream_squared = w**2 - float(state["phi_hat"]) / p.tau
        orbit = amp * np.exp(-(np.sqrt(np.maximum(upstream_squared, 0.0)) - p.u)**2)
        orbit = np.where(upstream_squared >= 0.0, orbit, 0.0)
        free_in = orbit * (vz <= -vcut)
        if bool(state["swe_reflected_active"]):
            reflected_in = orbit * ((vz > -vcut) & (vz <= 0.0))
            reflected_out = orbit * ((vz > 0.0) & (vz < vcut))
        else:
            reflected_in = np.zeros_like(vz)
            reflected_out = np.zeros_like(vz)

        total = free_in + reflected_in + reflected_out
        return {
            "vz_mps": vz,
            "g_free_incoming": free_in,
            "g_reflected_incoming": reflected_in,
            "g_reflected_outgoing": reflected_out,
            "g_total": total,
            "support_cutoff_mps": np.full_like(vz, vcut, dtype=float),
            "a_swe": np.full_like(vz, a, dtype=float),
        }

    def _phe_vdf_components(self, state: Dict[str, float | str], vz_mps: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.p
        vz = np.asarray(vz_mps, dtype=float)
        amp = p.n_phe0_m3 * math.exp(float(state["phi_hat"]) - float(state["phi0_hat"])) / (
            math.sqrt(math.pi) * p.v_phe_th_mps
        )
        vcut = float(state["vcut_phe_mps"])
        w = vz / p.v_phe_th_mps
        free_out = amp * np.exp(-(w**2)) * (vz >= vcut)

        if bool(state["phe_captured_active"]):
            captured_out = amp * np.exp(-(w**2)) * ((vz >= 0.0) & (vz < vcut))
            captured_ret = amp * np.exp(-(w**2)) * ((vz > -vcut) & (vz < 0.0))
        else:
            captured_out = np.zeros_like(vz)
            captured_ret = np.zeros_like(vz)

        total = free_out + captured_out + captured_ret
        return {
            "vz_mps": vz,
            "g_free_outgoing": free_out,
            "g_captured_outgoing": captured_out,
            "g_captured_returning": captured_ret,
            "g_total": total,
            "support_cutoff_mps": np.full_like(vz, vcut, dtype=float),
        }

    def _swi_vdf_components(
        self,
        state: Dict[str, float | str],
        vz_mps: np.ndarray,
        ion_sigma_frac: float,
    ) -> Dict[str, np.ndarray | float | str]:
        p = self.p
        vz = np.asarray(vz_mps, dtype=float)
        n_i = float(state["n_swi_m3"])
        v_peak = -float(state["v_i_mps"])
        sigma = max(ion_sigma_frac * abs(v_peak), 0.02 * p.cs_mps, 1.0)
        g_total = n_i * np.exp(-0.5 * ((vz - v_peak) / sigma) ** 2) / (
            math.sqrt(2.0 * math.pi) * sigma
        )
        return {
            "vz_mps": vz,
            "g_total": g_total,
            "distribution_kind": "cold-delta-regularized",
            "v_peak_mps": v_peak,
            "sigma_mps": sigma,
        }

    def vdf_1d_at_z(
        self,
        profile: Dict[str, np.ndarray | float | str],
        z: float,
        species: Species = "all",
        unit: ZUnit = "hat",
        n_v: int = 4001,
        n_sigma: float = 6.0,
        ion_sigma_frac: float = 0.03,
    ) -> Dict[str, object]:
        """Return reduced 1D velocity distributions at one location.

        The returned arrays satisfy approximately
            ∫ g(v_z) dv_z = n(z)
        with the exact branch-wise accessibility rules used in the solver.
        """
        state = self.sample_at_z(profile, z, unit=unit)
        out: Dict[str, object] = {"state": state, "species": species}

        if species in {"swe", "all"}:
            vz_swe = self._velocity_grid_for_species(state, "swe", n_v, n_sigma)
            out["swe"] = self._swe_vdf_components(state, vz_swe)
        if species in {"phe", "all"}:
            vz_phe = self._velocity_grid_for_species(state, "phe", n_v, n_sigma)
            out["phe"] = self._phe_vdf_components(state, vz_phe)
        if species in {"swi", "all"}:
            vz_swi = self._velocity_grid_for_species(state, "swi", n_v, n_sigma)
            out["swi"] = self._swi_vdf_components(state, vz_swi, ion_sigma_frac=ion_sigma_frac)
        return out

    def fluxes_at_z(
        self,
        profile: Dict[str, np.ndarray | float | str],
        z: float,
        unit: ZUnit = "hat",
    ) -> Dict[str, float | str]:
        """Exact orbit fluxes, independent of plotting resolution."""
        state = self.sample_at_z(profile, z, unit=unit)
        p = self.p
        psi = float(state["phi_hat"]) / p.tau
        barrier = (0.0 if state["branch"] == "B" else float(state["phi_m_hat"]) / p.tau)
        cutoff = math.sqrt(max(0.0, -barrier))
        coefficient = p.v_phe_th_mps / (2 * math.sqrt(math.pi))
        free_e = coefficient * self._swe_free_current_term(float(state["n_swe_inf_m3"]), cutoff - p.u)
        reflected_e = 0.0
        if state["swe_reflected_active"]:
            accessible = coefficient * self._swe_free_current_term(float(state["n_swe_inf_m3"]),
                                                                   math.sqrt(max(0., -psi)) - p.u)
            reflected_e = max(0., accessible - free_e)
        free_pe = coefficient*p.n_phe0_m3*math.exp(barrier*p.tau - float(state["phi0_hat"]))
        captured_pe = 0.0
        if state["phe_captured_active"]:
            captured_pe = max(0., coefficient*p.n_phe0_m3*math.exp(float(state["phi_hat"])-float(state["phi0_hat"]))-free_pe)
        ion_flux = p.n_swi_inf_m3*p.v_d_ion_mps

        def moment(density, plus, minus, charge):
            signed = plus - minus
            return dict(Gamma_signed_m2s=signed, J_signed_Apm2=charge*signed,
                        mean_v_mps=signed/density if density > 0 else math.nan)
        swe_free = moment(float(state['n_swe_f_m3']), 0., free_e, -QE)
        swe_ref_in = moment(float(state['n_swe_r_m3'])/2, 0., reflected_e, -QE)
        swe_ref_out = moment(float(state['n_swe_r_m3'])/2, reflected_e, 0., -QE)
        swe_total = moment(float(state['n_swe_f_m3'])+float(state['n_swe_r_m3']), reflected_e, free_e+reflected_e, -QE)
        phe_free = moment(float(state['n_phe_f_m3']), free_pe, 0., -QE)
        phe_cap_out = moment(float(state['n_phe_c_m3'])/2, captured_pe, 0., -QE)
        phe_cap_ret = moment(float(state['n_phe_c_m3'])/2, 0., captured_pe, -QE)
        phe_total = moment(float(state['n_phe_f_m3'])+float(state['n_phe_c_m3']), free_pe+captured_pe, captured_pe, -QE)
        swi_total = moment(float(state['n_swi_m3']), 0., ion_flux, QE)

        net_gamma = (
            swe_total["Gamma_signed_m2s"]
            + phe_total["Gamma_signed_m2s"]
            + swi_total["Gamma_signed_m2s"]
        )
        net_current = (
            swe_total["J_signed_Apm2"]
            + phe_total["J_signed_Apm2"]
            + swi_total["J_signed_Apm2"]
        )

        out: Dict[str, float | str] = {
            **state,
            "Gamma_net_m2s": float(net_gamma),
            "J_net_Apm2": float(net_current),
            "Gamma_swi_signed_m2s": float(swi_total["Gamma_signed_m2s"]),
            "J_swi_signed_Apm2": float(swi_total["J_signed_Apm2"]),
            "mean_v_swi_mps": float(swi_total["mean_v_mps"]),
            "Gamma_swe_signed_m2s": float(swe_total["Gamma_signed_m2s"]),
            "J_swe_signed_Apm2": float(swe_total["J_signed_Apm2"]),
            "mean_v_swe_mps": float(swe_total["mean_v_mps"]),
            "Gamma_phe_signed_m2s": float(phe_total["Gamma_signed_m2s"]),
            "J_phe_signed_Apm2": float(phe_total["J_signed_Apm2"]),
            "mean_v_phe_mps": float(phe_total["mean_v_mps"]),
            "Gamma_swe_free_incoming_m2s": float(swe_free["Gamma_signed_m2s"]),
            "Gamma_swe_reflected_incoming_m2s": float(swe_ref_in["Gamma_signed_m2s"]),
            "Gamma_swe_reflected_outgoing_m2s": float(swe_ref_out["Gamma_signed_m2s"]),
            "Gamma_phe_free_outgoing_m2s": float(phe_free["Gamma_signed_m2s"]),
            "Gamma_phe_captured_outgoing_m2s": float(phe_cap_out["Gamma_signed_m2s"]),
            "Gamma_phe_captured_returning_m2s": float(phe_cap_ret["Gamma_signed_m2s"]),
            "J_swe_free_incoming_Apm2": float(swe_free["J_signed_Apm2"]),
            "J_swe_reflected_incoming_Apm2": float(swe_ref_in["J_signed_Apm2"]),
            "J_swe_reflected_outgoing_Apm2": float(swe_ref_out["J_signed_Apm2"]),
            "J_phe_free_outgoing_Apm2": float(phe_free["J_signed_Apm2"]),
            "J_phe_captured_outgoing_Apm2": float(phe_cap_out["J_signed_Apm2"]),
            "J_phe_captured_returning_Apm2": float(phe_cap_ret["J_signed_Apm2"]),
        }
        return out
