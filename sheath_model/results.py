"""Physical solutions retain their immutable physical conditions."""
from dataclasses import dataclass, field
import math
import numpy as np
from ._constants import QE
from .params import FixedEntryParams, ZhaoParams
from .search import SearchDiagnostics


@dataclass(frozen=True)
class ProfileOptions:
    """Dimensionless sampling controls, using inputs.length_scale_m."""
    zmax_hat: float = 80.
    n_profile_grid: int = 600
    n_type_a_grid: int = 8000
    profile_phi_tol_hat: float = 1e-3
    type_a_phi_m_eps_hat: float = 1e-5

    def __post_init__(self):
        for name in ('zmax_hat', 'profile_phi_tol_hat', 'type_a_phi_m_eps_hat'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('n_profile_grid', 'n_type_a_grid'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 32:
                raise ValueError(f'{name} must be an integer >= 32')


@dataclass(frozen=True)
class DensityResult:
    ion_m3: object
    electron_free_m3: object
    electron_reflected_m3: object
    photoelectron_free_m3: object
    photoelectron_captured_m3: object

    @property
    def charge_c_m3(self):
        return QE*(self.ion_m3-self.electron_free_m3-self.electron_reflected_m3
                   -self.photoelectron_free_m3-self.photoelectron_captured_m3)

    @classmethod
    def _from_normalized(cls, data, scale):
        return cls(*(np.asarray(data[key])*scale for key in
                     ('n_swi_hat', 'n_swe_f_hat', 'n_swe_r_hat', 'n_phe_f_hat', 'n_phe_c_hat')))


@dataclass(frozen=True)
class EquilibriumResult:
    """Admissible J=0 root; electron density is Maxwellian normalization.

    Diagnostics describe a finite search, not stability or completeness.
    """
    inputs: FixedEntryParams | ZhaoParams
    branch: str
    surface_potential_v: float
    minimum_potential_v: float
    ambient_electron_density_m3: float
    residual_norm: float
    diagnostics: SearchDiagnostics = field(compare=False)

    def __post_init__(self):
        if self.branch not in {'A', 'B', 'C'}:
            raise ValueError('branch must be A, B, or C')
        if not isinstance(self.inputs, (FixedEntryParams, ZhaoParams)):
            raise TypeError('inputs must be FixedEntryParams or ZhaoParams')
        if not all(math.isfinite(value) for value in
                   (self.surface_potential_v, self.minimum_potential_v,
                    self.ambient_electron_density_m3, self.residual_norm)):
            raise ValueError('solution quantities must be finite')
        if self.ambient_electron_density_m3 <= 0 or self.residual_norm < 0:
            raise ValueError('density must be positive and residual nonnegative')
        if (self.branch == 'A' and self.minimum_potential_v >= min(self.surface_potential_v, 0.)) or (
                self.branch == 'B' and (self.surface_potential_v <= 0 or self.minimum_potential_v != 0.)) or (
                self.branch == 'C' and (self.surface_potential_v >= 0 or self.minimum_potential_v != self.surface_potential_v)):
            raise ValueError('potentials are inconsistent with the branch')

    def _kernel_data(self):
        p = self.inputs
        return dict(branch=self.branch, phi0_V=self.surface_potential_v,
                    phi_m_V=self.minimum_potential_v, n_swe_inf_m3=self.ambient_electron_density_m3,
                    phi0_hat=self.surface_potential_v/p.photoelectron_temperature_ev,
                    phi_m_hat=self.minimum_potential_v/p.photoelectron_temperature_ev,
                    n_swe_inf_hat=self.ambient_electron_density_m3/p.density_scale_m3)

    def density(self, potential_v, *, side='upper'):
        """Local densities [m^-3]; side selects lower/upper for A."""
        from ._physics import SheathPhysics
        if self.branch == 'A' and side not in {'lower', 'upper'}:
            raise ValueError('A requires side=lower or upper')
        p = self.inputs
        phi = np.asarray(potential_v)/p.photoelectron_temperature_ev
        lo = self.minimum_potential_v if self.branch == 'A' else min(self.surface_potential_v, 0.)
        hi = self.surface_potential_v if self.branch == 'A' and side == 'lower' else (0. if self.branch == 'A' else max(self.surface_potential_v, 0.))
        if not np.all(np.isfinite(phi)) or np.any(np.asarray(potential_v) < lo) or np.any(np.asarray(potential_v) > hi):
            raise ValueError('potential is outside the selected profile segment')
        physics = SheathPhysics(p)
        args = (phi, self.surface_potential_v/p.photoelectron_temperature_ev,
                self.ambient_electron_density_m3/p.density_scale_m3,
                self.minimum_potential_v/p.photoelectron_temperature_ev)
        data = (physics._densities_hat_type_a_side(*args, side=side) if self.branch == 'A'
                else physics._densities_hat(self.branch, *args))
        return DensityResult._from_normalized(data, p.density_scale_m3)


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[EquilibriumResult, ...]
    diagnostics: SearchDiagnostics


@dataclass(frozen=True)
class SheathProfile:
    """Profile arrays and local orbit diagnostics; positions default to metres."""
    equilibrium: EquilibriumResult
    _data: dict = field(repr=False, compare=False)
    _physics: object = field(repr=False, compare=False)
    z_m: np.ndarray = field(init=False, repr=False, compare=False)
    potential_v: np.ndarray = field(init=False, repr=False, compare=False)
    electric_field_v_m: np.ndarray = field(init=False, repr=False, compare=False)
    turning_height_m: float = field(init=False)
    density: DensityResult = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        data, equilibrium = self._data, self.equilibrium
        for value in data.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        object.__setattr__(self, 'z_m', data['z_m_array_m'])
        object.__setattr__(self, 'potential_v', data['phi_V'])
        object.__setattr__(self, 'electric_field_v_m', data['E_Vpm'])
        object.__setattr__(self, 'turning_height_m', data['z_m_m'] if equilibrium.branch == 'A' else -1.)
        object.__setattr__(self, 'density', DensityResult._from_normalized(data, equilibrium.inputs.density_scale_m3))
        for value in vars(self.density).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    @property
    def z_hat(self):
        return self._data['z_hat']

    def sample(self, z, *, unit='m'):
        return self._physics.sample_at_z(self._data, z, unit=unit)

    def fluxes(self, z, *, unit='m'):
        return self._physics.fluxes_at_z(self._data, z, unit=unit)

    def vdf(self, z, *, species='all', unit='m', n_v=4001, n_sigma=6., ion_sigma_frac=.03):
        if species not in {'swi', 'swe', 'phe', 'all'}:
            raise ValueError('species must be swi, swe, phe, or all')
        if not isinstance(n_v, int) or isinstance(n_v, bool) or n_v < 3:
            raise ValueError('n_v must be an integer >= 3')
        if not math.isfinite(n_sigma) or n_sigma <= 0:
            raise ValueError('n_sigma must be finite and positive')
        return self._physics.vdf_1d_at_z(self._data, z, species=species, unit=unit,
                                       n_v=n_v, n_sigma=n_sigma, ion_sigma_frac=ion_sigma_frac)
