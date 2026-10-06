"""Physical inputs for fixed-entry and solar-illumination sheath problems."""

from dataclasses import dataclass
import math
from typing import Literal

from ._constants import EPS0, QE, ME, MP


class _PlasmaScales:
    @property
    def v_swe_th_mps(self) -> float:
        return math.sqrt(2.0 * QE * self.electron_temperature_ev / self.electron_mass_kg)

    @property
    def v_phe_th_mps(self) -> float:
        return math.sqrt(2.0 * QE * self.photoelectron_temperature_ev / self.electron_mass_kg)

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
    electron_mass_kg: float = ME

    @property
    def density_scale_m3(self) -> float:
        return self.ion_density_m3


@dataclass(frozen=True)
class ZhaoParams(_PlasmaScales):
    """Solar wind and illumination inputs in SI, except eV and degrees."""

    sun_elevation_deg: float = 60.0
    ion_density_m3: float = 8.7e6
    photoelectron_reference_density_m3: float = 64e6
    electron_temperature_ev: float = 12.0
    ion_temperature_ev: float = 0.0
    ion_pressure_factor: float = 1.0
    photoelectron_temperature_ev: float = 2.2
    solar_wind_speed_mps: float = 468e3
    ion_mass_kg: float = MP
    electron_mass_kg: float = ME
    electron_drift_mode: Literal["full", "normal", "zero"] = "normal"
    ion_drift_mode: Literal["full", "normal"] = "normal"

    @property
    def alpha_rad(self) -> float:
        return math.radians(self.sun_elevation_deg)

    @property
    def density_scale_m3(self) -> float:
        return self.photoelectron_reference_density_m3

    @property
    def photoelectron_density_m3(self) -> float:
        return self.density_scale_m3 * math.sin(self.alpha_rad)

    @property
    def normal_wind_speed_mps(self) -> float:
        return self.solar_wind_speed_mps * math.sin(self.alpha_rad)

    @property
    def electron_drift_mps(self) -> float:
        if self.electron_drift_mode == "zero":
            return 0.0
        return self.solar_wind_speed_mps if self.electron_drift_mode == "full" else self.normal_wind_speed_mps

    @property
    def ion_entry_speed_mps(self) -> float:
        return self.solar_wind_speed_mps if self.ion_drift_mode == "full" else self.normal_wind_speed_mps
