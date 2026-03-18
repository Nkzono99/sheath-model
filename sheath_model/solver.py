from __future__ import annotations

"""
Complete Maxwellian semi-analytic sheath solver for Zhao et al. (2021).

Revision notes
--------------
This revision fixes the Type-A profile construction.
The previous implementation solved Type A on a finite interval with a hard
Dirichlet condition phi(zmax)=0. That can generate an artificial boundary layer
near zmax and an unrealistically flat potential in the recovery region.

Here, Type A is reconstructed from the first integral of Poisson's equation in
piecewise form around the potential minimum z_m:

    - lower branch (surface -> z_m):
        free solar-wind electrons + free/captured photoelectrons + ions
        reflected solar-wind electrons are absent below the barrier.

    - upper branch (z_m -> infinity):
        free/reflected solar-wind electrons + free photoelectrons + ions
        captured photoelectrons are absent above the barrier.

This matches the population topology shown in Zhao's Fig. 1 and removes the
spurious recovery-to-Dirichlet artifact.
"""

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.optimize import root
from scipy.special import erf, erfc

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31
MP = 1.67262192369e-27

Branch = Literal["A", "B", "C"]
DriftMode = Literal["full", "normal"]
TypeASide = Literal["lower", "upper"]


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
    electron_drift_mode: DriftMode = "normal"
    ion_drift_mode: DriftMode = "normal"

    zmax_hat: float = 80.0
    n_bvp_grid: int = 600
    n_type_a_grid: int = 8000
    type_a_phi_tol_hat: float = 1.0e-3
    type_a_phi_m_eps_hat: float = 1.0e-5
    # Experimental convergence fallback for legacy "full" ion-drift runs.
    # Disabled by default because it changes the branch equations.
    allow_type_c_normal_ion_fallback: bool = False

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
                "Use alpha > 0 or switch to the legacy full-drift modes explicitly."
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
        """Eq. (24) for Type A, evaluated at phi(infty)=0.

        Captured photoelectrons are absent at infinity, so they are not included
        here. This helper is only for solving the Type-A algebraic unknowns.
        """
        p = self.p
        phi = 0.0

        s_swe = math.sqrt(max(0.0, (phi - phi_m_V) / p.T_swe_eV))
        s_phe = math.sqrt(max(0.0, (phi - phi_m_V) / p.T_phe_eV))

        e2_swe_f = (
            (p.T_swe_eV / p.T_phe_eV)
            * (n_swe_inf_m3 / p.n_phe_ref_m3)
            * (
                math.exp(phi / p.T_swe_eV) * (1.0 - math.erf(s_swe - p.u))
                - math.exp(phi_m_V / p.T_swe_eV) * (1.0 - math.erf(-p.u))
                + (1.0 / (math.sqrt(math.pi) * p.u))
                * math.exp(phi_m_V / p.T_swe_eV - p.u * p.u)
                * (math.exp(2.0 * p.u * s_swe) - 1.0)
            )
        )

        e2_swe_r = (
            2.0
            * (p.T_swe_eV / p.T_phe_eV)
            * (n_swe_inf_m3 / p.n_phe_ref_m3)
            * (
                math.exp(phi / p.T_swe_eV) * (math.erf(s_swe - p.u) + math.erf(p.u))
                - (1.0 / (math.sqrt(math.pi) * p.u))
                * math.exp(phi_m_V / p.T_swe_eV - p.u * p.u)
                * (math.exp(2.0 * p.u * s_swe) - 1.0)
            )
        )

        e2_phe_f = (p.n_phe0_m3 / p.n_phe_ref_m3) * (
            math.exp((phi - phi0_V) / p.T_phe_eV) * (1.0 - math.erf(s_phe))
            - math.exp((phi_m_V - phi0_V) / p.T_phe_eV)
            * (1.0 - 2.0 / math.sqrt(math.pi) * s_phe)
        )

        arg_phi = 1.0 - 2.0 * phi / (p.T_swe_eV * p.mach * p.mach)
        arg_m = 1.0 - 2.0 * phi_m_V / (p.T_swe_eV * p.mach * p.mach)
        if arg_phi <= 0.0 or arg_m <= 0.0:
            return 1e30

        e2_swi = (
            2.0
            * (p.T_swe_eV / p.T_phe_eV)
            * (p.n_swi_inf_m3 / p.n_phe_ref_m3)
            * p.mach
            * p.mach
            * (math.sqrt(arg_phi) - math.sqrt(arg_m))
        )
        return e2_swe_f + e2_swe_r + e2_phe_f + e2_swi

    def _residuals_type_a(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        phi0_V, phi_m_V, n_swe_inf_m3 = x
        if phi0_V <= 0.0 or phi_m_V >= 0.0 or phi_m_V >= phi0_V or n_swe_inf_m3 <= 0.0:
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
        best = None
        best_norm = float("inf")
        for guess in guesses:
            sol = root(func, np.asarray(guess, dtype=float), method="hybr")
            fnorm = (
                float(np.linalg.norm(sol.fun)) if sol.fun is not None else float("inf")
            )
            if np.all(np.isfinite(sol.x)) and fnorm < best_norm:
                best = sol
                best_norm = fnorm
            if sol.success and np.all(np.isfinite(sol.x)) and fnorm < 1e-5:
                return np.asarray(sol.x, dtype=float)
        if best is None or best_norm > 1e-5:
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
        note = ""
        if branch == "A":
            guesses = (
                [np.array(guess, dtype=float)]
                if guess is not None
                else [
                    np.array([3.6, -0.5, 8.2e6]),
                    np.array([2.8, -0.3, 8.0e6]),
                    np.array([4.5, -0.8, 8.4e6]),
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
            try:
                phi0_V, n_swe_inf_m3 = self._try_root_guesses(
                    self._residuals_type_c, guesses
                )
            except RuntimeError:
                if (
                    not p.allow_type_c_normal_ion_fallback
                    or p.ion_drift_mode == "normal"
                ):
                    raise
                p2 = replace(
                    p, ion_drift_mode="normal", allow_type_c_normal_ion_fallback=False
                )
                alt = ZhaoSheathSolver(p2).solve_unknowns("C", guess)
                alt["note"] = (
                    "Type C root did not converge with ion_drift_mode='full'; retried with ion_drift_mode='normal' "
                    "(v_sw*sin(alpha)) for the ion term."
                )
                return alt
            phi_m_V = phi0_V
        else:
            raise ValueError(f"unknown branch: {branch}")

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
            "note": note,
            "electron_drift_mode": p.electron_drift_mode,
            "ion_drift_mode": p.ion_drift_mode,
            "v_d_electron_mps": p.v_d_electron_mps,
            "v_d_ion_mps": p.v_d_ion_mps,
        }

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

        s_swe = np.sqrt(np.maximum(0.0, (phi_hat - phi_m_hat) / tau))
        s_phe = np.sqrt(np.maximum(0.0, phi_hat - phi_m_hat))

        n_swe_f_hat = (
            0.5 * n_swe_inf_hat * np.exp(phi_hat / tau) * (1.0 - erf(s_swe - p.u))
        )
        n_phe_f_hat = 0.5 * sin_alpha * np.exp(phi_hat - phi0_hat) * (1.0 - erf(s_phe))

        if side == "lower":
            n_swe_r_hat = np.zeros_like(phi_hat)
            n_phe_c_hat = sin_alpha * np.exp(phi_hat - phi0_hat) * erf(s_phe)
        elif side == "upper":
            n_swe_r_hat = (
                n_swe_inf_hat * np.exp(phi_hat / tau) * (erf(s_swe - p.u) + erf(p.u))
            )
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
            n_swe_f_hat = 0.5 * n_swe_inf_hat * np.exp(phi_hat / tau) * (1.0 + erf(p.u))
            n_swe_r_hat = np.zeros_like(phi_hat)
            n_phe_f_hat = (
                0.5 * sin_alpha * np.exp(phi_hat - phi0_hat) * (1.0 - erf(s_phe))
            )
            n_phe_c_hat = sin_alpha * np.exp(phi_hat - phi0_hat) * erf(s_phe)
        elif branch == "C":
            s_swe = np.sqrt(np.maximum(0.0, (phi_hat - phi0_hat) / tau))
            s_phe = np.sqrt(np.maximum(0.0, phi_hat - phi0_hat))
            n_swe_f_hat = (
                0.5 * n_swe_inf_hat * np.exp(phi_hat / tau) * (1.0 - erf(s_swe - p.u))
            )
            n_swe_r_hat = (
                n_swe_inf_hat * np.exp(phi_hat / tau) * (erf(s_swe - p.u) + erf(p.u))
            )
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
        """Build one Type-A branch starting just above the potential minimum.

        The first version integrated z(phi)=∫dphi/|E| with a node value E(phi_1)=0
        imposed by cumulative_trapezoid(..., initial=0). That makes the first
        1/sqrt(E^2) sample artificially huge and creates a fake plateau just above
        z_m. Here we instead:

        1) include the small interval [phi_m, phi_1] explicitly in E^2, and
        2) start z-z_m with the local asymptotic form near the minimum,
           z-z_m ~ sqrt(2 (phi-phi_m) / (-rho(phi_m))).

        The remaining cells are advanced with midpoint integration in phi, which
        avoids evaluating 1/|E| at the singular endpoint.
        """
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
        rho_m_neg = max(-rho_m, 1.0e-14)

        # E^2(phi) = -2 ∫_{phi_m}^{phi} rho(psi) dpsi
        dphi0 = float(phi_nodes_asc[0] - phi_m_hat)
        int0 = 0.5 * (rho_m + rho_nodes[0]) * dphi0
        int_from_first = cumulative_trapezoid(rho_nodes, phi_nodes_asc, initial=0.0)
        integral_nodes = int0 + int_from_first
        e2_nodes = np.maximum(-2.0 * integral_nodes, 0.0)

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
        phi_end_hat = -abs(p.type_a_phi_tol_hat)
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

        # If solve_unknowns returned a fallback solution from a modified parameter set,
        # continue with that same parameterization for the profile construction.
        if (
            uk.get("ion_drift_mode") != p.ion_drift_mode
            or uk.get("electron_drift_mode") != p.electron_drift_mode
        ):
            p2 = replace(
                p,
                ion_drift_mode=str(uk["ion_drift_mode"]),
                electron_drift_mode=str(uk["electron_drift_mode"]),
                allow_type_c_normal_ion_fallback=False,
            )
            return ZhaoSheathSolver(p2).solve_profile(branch, guess_unknowns)

        if branch == "A":
            return self._build_type_a_profile(uk)

        phi0_hat = float(uk["phi0_hat"])
        phi_m_hat = uk["phi_m_hat"]
        n_swe_inf_hat = float(uk["n_swe_inf_hat"])

        def rhs(z_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
            phi_hat = y[0]
            e_hat = y[1]
            dens = self._densities_hat(
                branch, phi_hat, phi0_hat, n_swe_inf_hat, phi_m_hat
            )
            rho_hat = self._rho_hat_from_densities(dens)
            return np.vstack([e_hat, -rho_hat])

        def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            return np.array([ya[0] - phi0_hat, yb[0]])

        z_hat = np.linspace(0.0, p.zmax_hat, p.n_bvp_grid)
        phi_guess = phi0_hat * np.exp(-z_hat / max(4.0, 0.12 * p.zmax_hat))
        phi_guess[-1] = 0.0
        e_guess = np.gradient(phi_guess, z_hat)

        sol = solve_bvp(
            rhs, bc, z_hat, np.vstack([phi_guess, e_guess]), tol=1e-4, max_nodes=50000
        )
        if not sol.success:
            raise RuntimeError(f"BVP solve failed for branch {branch}: {sol.message}")

        phi_hat = sol.y[0]
        e_hat = sol.y[1]
        z_hat = sol.x
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

    def solve_auto(
        self, prefer_stable: bool = True
    ) -> Dict[str, np.ndarray | float | str]:
        if self.p.alpha_deg < 20.0:
            order: list[Branch] = ["C", "A", "B"]
        else:
            order = ["A", "B", "C"] if prefer_stable else ["B", "A", "C"]
        errs = []
        for br in order:
            try:
                return self.solve_profile(br)
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{br}: {exc}")
        raise RuntimeError("auto branch selection failed: " + " | ".join(errs))
