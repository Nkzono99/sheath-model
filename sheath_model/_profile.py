"""Profile reconstruction and local density, flux and velocity diagnostics."""

from __future__ import annotations

import math
from typing import Dict, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ._constants import QE

TypeASide = Literal["lower", "upper"]
Species = Literal["swi", "swe", "phe", "all"]
ZUnit = Literal["hat", "m"]


from ._physics import SheathPhysics


class ProfilePhysics(SheathPhysics):
    """Profile reconstruction and local orbit diagnostics, independent of search."""
    def __init__(self, params, options):
        super().__init__(params)
        self.options = options

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

        phi_m_eps = min(self.options.type_a_phi_m_eps_hat, 0.05 * max(1e-8, abs(phi_m_hat)))
        phi_m_eps = max(phi_m_eps, 1.0e-8)
        phi_end_hat = -abs(self.options.profile_phi_tol_hat)
        if phi_end_hat <= phi_m_hat:
            phi_end_hat = 0.5 * phi_m_hat

        ngrid = max(2000, self.options.n_type_a_grid)
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
        keep = z_hat <= self.options.zmax_hat
        z_hat = z_hat[keep]
        if len(z_hat) < 2:
            raise ValueError('zmax_hat contains fewer than two profile points')
        phi_hat = phi_hat[keep]
        ehat = ehat[keep]
        dens = {k: v[keep] for k, v in dens.items()}

        out: Dict[str, np.ndarray | float | str] = {
            **uk,
            "z_hat": z_hat,
            "z_m_hat": z_m_hat,
            "z_m_m": z_m_hat * p.length_scale_m,
            "z_m_array_m": z_hat * p.length_scale_m,
            "phi_hat": phi_hat,
            "phi_V": phi_hat * p.photoelectron_temperature_ev,
            "dphi_dzhat": ehat,
            "E_Vpm": -(p.photoelectron_temperature_ev / p.length_scale_m) * ehat,
            "length_scale_m": p.length_scale_m,
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


    def build_profile(
        self, root
    ):
        p = self.p
        uk = root._kernel_data()
        branch = root.branch

        if branch == "A":
            return self._build_type_a_profile(uk)

        phi0_hat = float(uk["phi0_hat"])
        phi_m_hat = uk["phi_m_hat"]
        n_swe_inf_hat = float(uk["n_swe_inf_hat"])

        # Integrate from the exact neutral upstream endpoint, then omit infinity.
        t = np.linspace(0., 1., max(600, self.options.n_profile_grid))
        phi_hat = phi0_hat * (1.-t)**2
        dens = self._densities_hat(branch, phi_hat, phi0_hat, n_swe_inf_hat, phi_m_hat)
        rho = self._rho_hat_from_densities(dens)
        e2 = -2*cumulative_trapezoid(rho[::-1], phi_hat[::-1], initial=0.)[::-1]
        cutoff = min(abs(self.options.profile_phi_tol_hat), .5*abs(phi0_hat))
        keep = np.abs(phi_hat) >= cutoff
        keep[-1] = False
        phi_hat, e2 = phi_hat[keep], e2[keep]
        if len(phi_hat) < 2 or np.any(e2 <= 0.) or not np.all(np.isfinite(e2)):
            raise RuntimeError("monotonic profile integral is not positive; refine the potential grid")
        z_hat = np.concatenate(([0.], np.cumsum(.5*np.abs(np.diff(phi_hat)) *
                                    (1/np.sqrt(e2[:-1]) + 1/np.sqrt(e2[1:])))))
        keep = z_hat <= self.options.zmax_hat
        phi_hat, e2, z_hat = phi_hat[keep], e2[keep], z_hat[keep]
        if len(z_hat) < 2:
            raise ValueError("zmax_hat contains fewer than two profile points")
        e_hat = -math.copysign(1., phi0_hat)*np.sqrt(e2)
        dens = self._densities_hat(branch, phi_hat, phi0_hat, n_swe_inf_hat, phi_m_hat)

        out: Dict[str, np.ndarray | float | str] = {
            **uk,
            "z_hat": z_hat,
            "z_m_hat": float(z_hat[np.argmin(phi_hat)]),
            "z_m_m": float(z_hat[np.argmin(phi_hat)] * p.length_scale_m),
            "z_m_array_m": z_hat * p.length_scale_m,
            "phi_hat": phi_hat,
            "phi_V": phi_hat * p.photoelectron_temperature_ev,
            "dphi_dzhat": e_hat,
            "E_Vpm": -(p.photoelectron_temperature_ev / p.length_scale_m) * e_hat,
            "length_scale_m": p.length_scale_m,
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


    def _to_z_hat(self, z: float, unit: ZUnit) -> float:
        if unit == "hat":
            return float(z)
        if unit == "m":
            return float(z) / self.p.length_scale_m
        raise ValueError(f"unknown z unit: {unit}")


    @staticmethod
    def _interp_on_profile(profile: Dict[str, np.ndarray | float | str], key: str, z_hat: float) -> float:
        z_arr = np.asarray(profile["z_hat"], dtype=float)
        y_arr = np.asarray(profile[key], dtype=float)
        # SI -> normalized round trips can put an endpoint one ulp outside.
        allowance = 4*np.spacing(max(1., abs(z_arr[0]), abs(z_arr[-1])))
        if not math.isfinite(z_hat) or z_hat < float(z_arr[0])-allowance or z_hat > float(z_arr[-1])+allowance:
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
        dens_m3 = {k.replace("_hat", "_m3"): v * p.density_scale_m3 for k, v in dens_hat_scalar.items()}
        n_total_hat = (
            dens_hat_scalar["n_swe_f_hat"]
            + dens_hat_scalar["n_swe_r_hat"]
            + dens_hat_scalar["n_phe_f_hat"]
            + dens_hat_scalar["n_phe_c_hat"]
        )
        rho_hat = dens_hat_scalar["n_swi_hat"] - n_total_hat

        v_i_local = p.ion_entry_speed_mps*p.ion_density_m3/dens_m3["n_swi_m3"]

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
            "z_m": z_hat * p.length_scale_m,
            "phi_hat": phi_hat,
            "phi_V": phi_hat * p.photoelectron_temperature_ev,
            "phi0_hat": phi0_hat,
            "phi0_V": phi0_hat * p.photoelectron_temperature_ev,
            "phi_m_hat": phi_m_hat,
            "phi_m_V": phi_m_hat * p.photoelectron_temperature_ev if math.isfinite(phi_m_hat) else math.nan,
            "z_m_hat": z_m_hat,
            "dphi_dzhat": dphi_dzhat,
            "E_Vpm": E_Vpm,
            "n_swe_inf_hat": n_swe_inf_hat,
            "n_swe_inf_m3": n_swe_inf_hat * p.density_scale_m3,
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
            "n_total_m3": n_total_hat * p.density_scale_m3,
            "rho_m3": rho_hat * p.density_scale_m3,
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
        amp = p.photoelectron_density_m3 * math.exp(float(state["phi_hat"]) - float(state["phi0_hat"])) / (
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
        if p.ion_temperature_ev > 0:
            raise ValueError("warm-ion fluid pressure does not define an ion VDF; request electron species")
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
        free_pe = coefficient*p.photoelectron_density_m3*math.exp(barrier*p.tau - float(state["phi0_hat"]))
        captured_pe = 0.0
        if state["phe_captured_active"]:
            captured_pe = max(0., coefficient*p.photoelectron_density_m3*math.exp(float(state["phi_hat"])-float(state["phi0_hat"]))-free_pe)
        ion_flux = p.ion_density_m3*p.ion_entry_speed_mps

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
