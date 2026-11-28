"""Local branch criterion with a prescribed far-upstream electron reservoir."""

from dataclasses import dataclass
import math
from typing import Literal, Protocol

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import beta, erfc

from ._constants import EPS0, ME, MP, QE
from ._ions import ion_critical_potential, ion_density_ratio
from ._orbits import integrate_panels


def _positive(*values):
    return all(math.isfinite(value) and value > 0 for value in values)


@dataclass(frozen=True)
class MaxwellianElectrons:
    """One reduced upstream Maxwellian. Drift is positive toward the wall."""

    density_m3: float
    temperature_ev: float
    drift_mps: float = 0.0

    def __post_init__(self):
        if not _positive(self.density_m3, self.temperature_ev) or not math.isfinite(self.drift_mps):
            raise ValueError("invalid Maxwellian density, temperature or drift")

    def phase(self, speed, speed_unit):
        """Reduced distribution per velocity measured in sqrt(eV)."""
        return (self.density_m3 / math.sqrt(math.pi * self.temperature_ev)
                * np.exp(-(speed - self.drift_mps / speed_unit)**2 / self.temperature_ev))

    def inward_flux(self, depth, speed_unit):
        drift = self.drift_mps / speed_unit
        thermal = math.sqrt(self.temperature_ev)
        cutoff = (math.sqrt(depth) - drift) / thermal
        return (self.density_m3 * speed_unit / 2
                * (thermal / math.sqrt(math.pi) * math.exp(-cutoff**2) + drift * erfc(cutoff)))

    def passing_density(self, depth, speed_unit):
        thermal = math.sqrt(self.temperature_ev)
        drift = self.drift_mps / speed_unit
        if drift == 0:
            return 0.5 * self.density_m3 * math.exp(-depth / self.temperature_ev)
        speeds = np.maximum(0.0, drift + thermal * np.arange(-12, 13, 2))
        cuts = np.sqrt(np.maximum(speeds**2 - depth, 0.0))
        if cuts[-1] == 0:
            return 0.0
        return float(integrate_panels(
            lambda v: self.phase(np.sqrt(v*v + depth), speed_unit), cuts
        ))


@dataclass(frozen=True)
class KappaElectrons:
    """Centered reduced kappa distribution with the specified core width."""

    density_m3: float
    core_temperature_ev: float
    kappa: float

    def __post_init__(self):
        if not _positive(self.density_m3, self.core_temperature_ev, self.kappa) or self.kappa <= 1:
            raise ValueError("a kappa population requires positive density and core temperature, and kappa > 1")

    @property
    def scale(self):
        return self.kappa * self.core_temperature_ev

    def phase(self, speed, speed_unit):
        return (self.density_m3 / (math.sqrt(self.scale) * beta(0.5, self.kappa - 0.5))
                * np.exp(-self.kappa * np.log1p(speed**2 / self.scale)))

    def inward_flux(self, depth, speed_unit):
        return self.phase(0.0, speed_unit) * speed_unit * self.scale / (2 * (self.kappa - 1)) * math.exp(
            (1 - self.kappa) * math.log1p(depth / self.scale)
        )

    def passing_density(self, depth, speed_unit):
        return 0.5 * self.density_m3 * math.exp((0.5 - self.kappa) * math.log1p(depth / self.scale))


@dataclass(frozen=True)
class FixedReservoir:
    """Far-wind input at 0 V. Electron component densities are full-space densities."""

    electrons: tuple[MaxwellianElectrons | KappaElectrons, ...]
    ion_density_m3: float
    ion_speed_mps: float
    ion_temperature_ev: float = 0.0
    ion_pressure_factor: float = 1.0
    ion_mass_kg: float = MP
    electron_mass_kg: float = ME

    def __post_init__(self):
        object.__setattr__(self, "electrons", tuple(self.electrons))
        if not self.electrons or not _positive(
            self.ion_density_m3, self.ion_speed_mps, self.ion_pressure_factor, self.ion_mass_kg, self.electron_mass_kg
        ):
            raise ValueError("the reservoir requires electrons and positive ion inputs and masses")
        ion_critical_potential(self.ion_energy_ev, self.ion_pressure_ev)

    @property
    def speed_unit(self):
        return math.sqrt(2 * QE / self.electron_mass_kg)

    @property
    def ion_energy_ev(self):
        return self.ion_mass_kg * self.ion_speed_mps**2 / (2 * QE)

    @property
    def ion_pressure_ev(self):
        return self.ion_pressure_factor * self.ion_temperature_ev

    @property
    def ion_flux_m2_s(self):
        return self.ion_density_m3 * self.ion_speed_mps

    def ion_density(self, potential_v):
        return self.ion_density_m3 * ion_density_ratio(potential_v, self.ion_energy_ev, self.ion_pressure_ev)

    def phase(self, speed):
        speed = np.asarray(speed, dtype=float)
        return sum(component.phase(speed, self.speed_unit) for component in self.electrons)

    def inward_flux(self, depth):
        return sum(component.inward_flux(depth, self.speed_unit) for component in self.electrons)

    def passing_density(self, depth):
        return sum(component.passing_density(depth, self.speed_unit) for component in self.electrons)


class EmissionSpectrum(Protocol):
    """Outward number flux as a function of normal energy K, in eV."""

    def flux_above(self, barrier_ev: float) -> float:
        """Integral above the barrier [m^-2 s^-1]."""

    def differential_flux(self, energy_ev: float) -> float:
        """Number flux density [m^-2 s^-1 eV^-1], for scalar or array energy."""


@dataclass(frozen=True)
class MaxwellianEmission:
    """Outgoing half Maxwellian specified by total number flux and temperature."""

    flux_m2_s: float
    temperature_ev: float

    def __post_init__(self):
        if not math.isfinite(self.flux_m2_s) or self.flux_m2_s < 0 or not _positive(self.temperature_ev):
            raise ValueError("invalid Maxwellian emission flux or temperature")

    def flux_above(self, barrier_ev):
        return self.flux_m2_s * math.exp(-barrier_ev / self.temperature_ev)

    def differential_flux(self, energy_ev):
        return self.flux_m2_s / self.temperature_ev * np.exp(-energy_ev / self.temperature_ev)


@dataclass(frozen=True)
class BinnedEmission:
    """Number flux per energy bin, with constant differential flux within each bin."""

    energy_edges_ev: tuple[float, ...]
    bin_flux_m2_s: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "energy_edges_ev", tuple(self.energy_edges_ev))
        object.__setattr__(self, "bin_flux_m2_s", tuple(self.bin_flux_m2_s))
        if (len(self.energy_edges_ev) != len(self.bin_flux_m2_s) + 1 or not self.bin_flux_m2_s
                or not np.all(np.isfinite(self.energy_edges_ev)) or self.energy_edges_ev[0] < 0
                or np.any(np.diff(self.energy_edges_ev) <= 0)
                or not np.all(np.isfinite(self.bin_flux_m2_s)) or min(self.bin_flux_m2_s) < 0):
            raise ValueError("invalid emission energy edges or bin fluxes")

    def flux_above(self, barrier_ev):
        edges = np.asarray(self.energy_edges_ev)
        fractions = np.clip((edges[1:] - barrier_ev) / np.diff(edges), 0, 1)
        return float(fractions @ self.bin_flux_m2_s)

    def differential_flux(self, energy_ev):
        edges = self.energy_edges_ev
        index = np.searchsorted(edges, energy_ev, side="right") - 1
        values = np.r_[0.0, np.asarray(self.bin_flux_m2_s) / np.diff(edges), 0.0]
        return values[index + 1]


@dataclass(frozen=True)
class BranchCriterion:
    """The local sign test at a hypothetical neutral, zero-current minimum."""

    outcome: Literal["A", "B", "marginal", "no_current", "no_minimum", "unresolved"]
    neutral_depth_v: float | None = None
    emission_barrier_ev: float | None = None
    ratio: float | None = None
    effective_emission_temperature_ev: float | None = None


def _emission_at_minimum(emission, flux, total_flux, speed_unit):
    if isinstance(emission, MaxwellianEmission):
        temperature = emission.temperature_ev
        barrier = temperature * math.log(total_flux / flux)
        return barrier, flux / speed_unit * math.sqrt(math.pi / temperature), flux / temperature
    if isinstance(emission, BinnedEmission):
        edges = np.asarray(emission.energy_edges_ev)
        bins = np.asarray(emission.bin_flux_m2_s)
        tails = np.r_[np.cumsum(bins[::-1])[::-1], 0.0]
        k = int(np.searchsorted(-tails, -flux, side="right") - 1)
        barrier = edges[k + 1] - (flux - tails[k + 1]) / bins[k] * (edges[k + 1] - edges[k])
        widths = np.sqrt(np.maximum(edges - barrier, 0))
        density = 2 / speed_unit * (np.diff(widths) @ (bins / np.diff(edges)))
        return barrier, density, bins[k] / (edges[k + 1] - edges[k])
    high = 1.0
    while emission.flux_above(high) > flux and high < 1e12:
        high *= 2
    barrier = brentq(lambda v: emission.flux_above(v) / total_flux - flux / total_flux, 0, high)
    density = (2 * total_flux / speed_unit
               * quad(lambda x: emission.differential_flux(barrier + x*x) / total_flux, 0, np.inf)[0])
    return barrier, density, emission.differential_flux(barrier)


def classify_branch(reservoir: FixedReservoir, emission: EmissionSpectrum, *,
                    ratio_tolerance: float = 1e-7, scan_points: int = 96) -> BranchCriterion:
    """Evaluate D_m=0 and R near the A/B boundary using fixed far-upstream inputs.

    A and B refer to the side of the quasi-neutral region on which the well
    lies. A finite depth scan can miss roots. The test alone gives neither
    a global profile nor a stability or uniqueness result.
    """
    if not _positive(ratio_tolerance) or scan_points < 8:
        raise ValueError("use a positive ratio tolerance and at least eight scan points")
    total = emission.flux_above(0.0)
    if total == 0.0:
        return BranchCriterion("no_current")
    if not _positive(total):
        raise ValueError("emission flux must be finite and nonnegative")

    def balance(depth):
        return reservoir.inward_flux(depth) - reservoir.ion_flux_m2_s

    if balance(0.0) <= 0:
        return BranchCriterion("no_current")
    high = 1.0
    while balance(high) > 0 and high < 1e12:
        high *= 2
    if balance(high) > 0:
        return BranchCriterion("unresolved")
    hi = brentq(balance, 0, high)
    lo = brentq(lambda d: balance(d) - total, 0, hi) if balance(0.0) > total else 0.0

    def evaluate(depth):
        flux = balance(depth)
        wind_excess = reservoir.passing_density(depth) - reservoir.ion_density(-depth)
        if flux <= 0:
            return wind_excess, None, None, None
        barrier, density, differential = _emission_at_minimum(emission, min(flux, total), total, reservoir.speed_unit)
        return wind_excess + density, barrier, differential, flux

    # Exclude the current endpoints, and sample closely against both edges.
    f = np.unique(np.r_[np.linspace(0, 1, scan_points), np.geomspace(1e-10, 0.1, 25),
                        1 - np.geomspace(1e-10, 0.1, 25)])
    depths = lo + (hi - lo) * np.clip(f, 1e-12, 1 - 1e-12)
    values = np.array([evaluate(depth)[0] for depth in depths])
    nonzero = np.flatnonzero(values)
    changes = nonzero[:-1][np.diff(np.sign(values[nonzero])) != 0]
    if changes.size != 1:
        return BranchCriterion("no_minimum" if np.all(values < 0) else "unresolved")
    index = int(changes[0])
    if values[index] < 0:
        return BranchCriterion("unresolved")
    right = nonzero[np.searchsorted(nonzero, index) + 1]
    depth = brentq(lambda d: evaluate(d)[0], depths[index], depths[right], xtol=1e-14)
    _, barrier, differential, flux = evaluate(depth)
    wind_edge = float(reservoir.phase(math.sqrt(depth)))
    photo_edge = 2 * differential / reservoir.speed_unit
    ratio = photo_edge / wind_edge if wind_edge else math.inf
    temperature = flux / differential if differential else math.inf
    outcome = "marginal" if abs(ratio - 1) <= ratio_tolerance else "A" if ratio > 1 else "B"
    return BranchCriterion(outcome, depth, barrier, ratio, temperature)


def turning_field_difference(reservoir: FixedReservoir, emission: EmissionSpectrum,
                             depth_v: float, barrier_ev: float, offset_v: float) -> float:
    """Return E_surface² - E_upstream² [V²/m²] at equal offsets above a single valley.

    Both sides must attain the requested offset. The depth is measured from
    the far reservoir and the barrier from the emitting surface to the valley.
    """
    if not all(math.isfinite(v) for v in (offset_v, barrier_ev, depth_v)) or not 0 <= offset_v <= min(barrier_ev, depth_v):
        raise ValueError("offset must be within both surface and reservoir barriers")
    if offset_v == 0:
        return 0.0
    extra = ()
    if isinstance(emission, BinnedEmission):
        extra = tuple(math.sqrt(edge - (barrier_ev - offset_v)) for edge in emission.energy_edges_ev
                      if barrier_ev - offset_v < edge < barrier_ev)
    edges = np.r_[0, extra, math.sqrt(offset_v)]

    def integrand(x):
        energy_deficit = np.maximum(offset_v - x*x, 0.0)
        photo = 2 * emission.differential_flux(barrier_ev - energy_deficit) / reservoir.speed_unit
        wind = reservoir.phase(np.sqrt(depth_v - energy_deficit))
        return 2 * x*x * (photo - wind)

    return float(4 * QE / EPS0 * integrate_panels(integrand, edges, order=64))
