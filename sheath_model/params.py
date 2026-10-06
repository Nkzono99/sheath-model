"""Physical inputs for fixed-entry and solar-illumination sheath problems."""

from dataclasses import dataclass
import math
from typing import Literal

from ._constants import EPS0, QE, ME, MP


class _PlasmaScales:
    @property
    def v_swe_th_mps(self) -> float:
        return math.sqrt(2.0 * QE * self.electron_temperature_ev / ME)

    @property
    def v_phe_th_mps(self) -> float:
        return math.sqrt(2.0 * QE * self.photoelectron_temperature_ev / ME)

    @property
    def cs_mps(self) -> float:
        """Electron-temperature reference speed for normalization, without a Bohm correction."""
        return math.sqrt(QE * self.electron_temperature_ev / self.ion_mass_kg)

    @property
    def mach(self) -> float:
        return self.ion_entry_speed_mps / self.cs_mps

    @property
    def u(self) -> float:
        return self.electron_drift_mps / self.v_swe_th_mps

    @property
    def tau(self) -> float:
        return self.electron_temperature_ev / self.photoelectron_temperature_ev

    @property
    def length_scale_m(self) -> float:
        return math.sqrt(EPS0 * self.photoelectron_temperature_ev / (self.density_scale_m3 * QE))


@dataclass(frozen=True)
class FixedEntryParams(_PlasmaScales):
    """Fixed normal entrance state, with SI units except temperatures [eV].

    Speeds are positive inward. Photoelectron density is the normalization of
    the emitted Maxwellian; its outward half has half this density.
    Electron normalization is solved from neutrality, rather than prescribed.
    """

    ion_density_m3: float = 8.7e6
    ion_entry_speed_mps: float = 405299.88897111727
    ion_temperature_ev: float = 0.0
    ion_pressure_factor: float = 1.0
    electron_temperature_ev: float = 12.0
    electron_drift_mps: float = 0.0
    photoelectron_density_m3: float = 0.0
    photoelectron_temperature_ev: float = 2.2
    ion_mass_kg: float = MP

    zmax_hat: float = 80.0
    n_profile_grid: int = 600
    n_type_a_grid: int = 8000
    profile_phi_tol_hat: float = 1.0e-3
    type_a_phi_m_eps_hat: float = 1.0e-5

    @property
    def density_scale_m3(self) -> float:
        return self.ion_density_m3


@dataclass(frozen=True)
class ZhaoParams(_PlasmaScales):
    """Solar wind and illumination inputs, projected onto the sheath normal."""

    alpha_deg: float = 60.0
    n_swi_inf_cm3: float = 8.7
    n_phe_ref_cm3: float = 64.0
    T_swe_eV: float = 12.0
    T_swi_eV: float = 0.0
    ion_pressure_factor: float = 1.0
    T_phe_eV: float = 2.2
    v_sw_total_mps: float = 468e3
    m_i_kg: float = MP
    electron_drift_mode: Literal["full", "normal", "zero"] = "normal"
    ion_drift_mode: Literal["full", "normal"] = "normal"

    zmax_hat: float = 80.0
    n_profile_grid: int = 600
    n_type_a_grid: int = 8000
    profile_phi_tol_hat: float = 1.0e-3
    type_a_phi_m_eps_hat: float = 1.0e-5

    @property
    def alpha_rad(self) -> float:
        return math.radians(self.alpha_deg)

    @property
    def ion_density_m3(self) -> float:
        return self.n_swi_inf_cm3 * 1e6

    @property
    def density_scale_m3(self) -> float:
        return self.n_phe_ref_cm3 * 1e6

    @property
    def photoelectron_density_m3(self) -> float:
        return self.density_scale_m3 * math.sin(self.alpha_rad)

    @property
    def electron_temperature_ev(self) -> float:
        return self.T_swe_eV

    @property
    def ion_temperature_ev(self) -> float:
        return self.T_swi_eV

    @property
    def photoelectron_temperature_ev(self) -> float:
        return self.T_phe_eV

    @property
    def ion_mass_kg(self) -> float:
        return self.m_i_kg

    @property
    def v_sw_normal_mps(self) -> float:
        return self.v_sw_total_mps * math.sin(self.alpha_rad)

    @property
    def electron_drift_mps(self) -> float:
        if self.electron_drift_mode == "zero":
            return 0.0
        return self.v_sw_total_mps if self.electron_drift_mode == "full" else self.v_sw_normal_mps

    @property
    def ion_entry_speed_mps(self) -> float:
        return self.v_sw_total_mps if self.ion_drift_mode == "full" else self.v_sw_normal_mps
