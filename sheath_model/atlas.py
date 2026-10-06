"""Branch-separated equilibrium tables used only to initialize exact solves."""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np

from .params import FixedEntryParams


@dataclass(frozen=True)
class AtlasOptions:
    neighbors: int = 8
    max_distance: float = 1.
    interpolate: bool = True
    component_distance: float = .75

    def __post_init__(self):
        if not isinstance(self.neighbors, int) or isinstance(self.neighbors, bool) or self.neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        if not math.isfinite(self.max_distance) or self.max_distance <= 0:
            raise ValueError("max_distance must be finite and positive")
        if not math.isfinite(self.component_distance) or self.component_distance <= 0:
            raise ValueError("component_distance must be finite and positive")
        if not isinstance(self.interpolate, bool):
            raise ValueError("interpolate must be a bool")


def parameter_key(p):
    """Maxwell-source physics in six dimensionless coordinates.

    log(Te/Tpe), asinh(ue/vthe), log(ui/vthpe), log(mi/me),
    log1p(pressure_factor*Ti/Tpe), log1p(npe/ni).
    """
    return np.array([math.log(p.tau), math.asinh(p.u), math.log(p.ion_entry_speed_mps/p.v_phe_th_mps),
                     math.log(p.ion_mass_kg/p.electron_mass_kg), math.log1p(p.ion_pressure_factor*p.ion_temperature_ev/
                     p.photoelectron_temperature_ev), math.log1p(p.photoelectron_density_m3/p.ion_density_m3)])


def params_from_key(key, reference):
    """Reconstruct equivalent fixed-entry inputs using the query's SI scales."""
    if not np.all(np.isfinite(key)) or np.max(np.abs(key)) > 100 or min(key[4:]) < 0:
        raise ValueError("invalid dimensionless plasma key")
    temperature = reference.photoelectron_temperature_ev
    electron_temperature = temperature*math.exp(key[0])
    return FixedEntryParams(
        ion_density_m3=reference.ion_density_m3,
        ion_entry_speed_mps=reference.v_phe_th_mps*math.exp(key[2]),
        ion_mass_kg=reference.electron_mass_kg*math.exp(key[3]),
        electron_mass_kg=reference.electron_mass_kg,
        ion_temperature_ev=temperature*math.expm1(key[4]), ion_pressure_factor=1.,
        electron_temperature_ev=electron_temperature,
        electron_drift_mps=math.sinh(key[1])*reference.v_phe_th_mps*math.sqrt(math.exp(key[0])),
        photoelectron_density_m3=reference.ion_density_m3*math.expm1(key[5]),
        photoelectron_temperature_ev=temperature,
    )


@dataclass(frozen=True)
class AtlasPoint:
    branch: str
    component: int
    key: tuple[float, ...]
    coordinates: tuple[float, ...]
    # Empty for Maxwellians. Fortran stores normalized bin edges and fractions.
    spectrum_shape: tuple[float, ...] = ()

    def __post_init__(self):
        if (self.branch not in {"A", "B", "C"} or not isinstance(self.component, int) or
                isinstance(self.component, bool) or self.component < 1):
            raise ValueError("invalid atlas branch/component")
        if len(self.key) != 6 or len(self.coordinates) != 3:
            raise ValueError("invalid atlas coordinate dimensions")
        if not all(math.isfinite(v) for v in (*self.key, *self.coordinates, *self.spectrum_shape)):
            raise ValueError("atlas coordinates must be finite")
        if min(self.key[4:]) < 0:
            raise ValueError("negative pressure/source coordinate")
        if self.branch != "A" and self.coordinates[2] != 0:
            raise ValueError("B/C atlas coordinates must have zero padding")


class EquilibriumAtlas:
    """Caller-owned table; an entry is a seed, never an accepted query solution.

    component labels nearby solution families inside an A/B/C type; automatic
    labels use proximity and are not a proof of connectivity. Local linear
    predictions never mix component labels. Blank regions mean that no
    root was stored, not that a root does not exist.
    """
    def __init__(self, points=(), *, options=None):
        self.options = options if options is not None else AtlasOptions()
        if not isinstance(self.options, AtlasOptions):
            raise TypeError("invalid atlas options")
        self._points = list(points)
        if not all(isinstance(point, AtlasPoint) for point in self._points):
            raise TypeError("points must be AtlasPoint objects")
        self.attempts = []

    @property
    def points(self):
        return tuple(self._points)

    def neighbors(self, params, branch):
        key = parameter_key(params)
        nearby = [(np.linalg.norm(key-point.key), point) for point in self._points
                  if point.branch == branch and not point.spectrum_shape]
        return [point for distance, point in sorted(nearby, key=lambda item: item[0])
                if distance <= self.options.max_distance][:self.options.neighbors]

    def predictions(self, params, branch):
        """Return encoded predictions and raw neighbors, preserving components."""
        key = parameter_key(params)
        nearby = self.neighbors(params, branch)
        predictions = []
        for component in dict.fromkeys(point.component for point in nearby):
            group = [point for point in nearby if point.component == component]
            anchor = group[0]
            y0 = np.array(anchor.coordinates)
            if self.options.interpolate and len(group) > 1:
                offsets = np.array([np.array(point.key)-anchor.key for point in group[1:]])
                values = np.array([np.array(point.coordinates)-y0 for point in group[1:]])
                # Ridge regularization handles axes held fixed by a map/sweep.
                normal = offsets.T@offsets+1e-10*np.eye(6)
                slopes = np.linalg.solve(normal, offsets.T@values)
                prediction = y0+(key-anchor.key)@slopes
                if np.all(np.isfinite(prediction)):
                    predictions.append(prediction)
            predictions.extend(np.array(point.coordinates) for point in group)
        unique = []
        for prediction in predictions:
            if not any(np.linalg.norm(prediction-other) < 1e-10 for other in unique):
                unique.append(prediction)
        return unique

    def add(self, result, *, component=None, search=None):
        """Store a root after checking its original equations and profile."""
        from ._equilibrium import EquilibriumProblem
        from .results import EquilibriumResult
        if not isinstance(result, EquilibriumResult):
            raise TypeError("result must be EquilibriumResult")
        params = result.inputs
        solver = EquilibriumProblem(params, search=search)
        branch = result.branch
        solver._validate_params_for_branch(branch)
        physical = np.array([result.surface_potential_v, result.minimum_potential_v, result.ambient_electron_density_m3])
        if (branch == "B" and physical[1] != 0.) or (branch == "C" and physical[1] != physical[0]):
            raise ValueError("minimum potential is inconsistent with the branch")
        encoded = solver._encode_unknowns(branch, physical)
        if encoded is None or solver._decode_unknowns(branch, encoded, solver.search) is None:
            raise ValueError("invalid atlas root")
        raw = solver._encoded_residual(branch, encoded, solver.search)
        if raw is None or np.max(np.abs(raw)) > solver.search.residual_tolerance:
            raise ValueError("atlas root does not satisfy the original equations")
        solver._validate_profile_root(branch, physical[0]/params.photoelectron_temperature_ev,
                                      physical[1]/params.photoelectron_temperature_ev,
                                      physical[2]/params.density_scale_m3)
        coordinates = np.zeros(3)
        coordinates[:len(encoded)] = encoded
        key = parameter_key(params)
        if any(p.branch == branch and np.linalg.norm(key-p.key) < 1e-12 and
               np.linalg.norm(coordinates-p.coordinates) < 1e-6 for p in self._points):
            return
        nearby = self.neighbors(params, branch)
        if component is None:
            matched = [point for point in nearby if np.linalg.norm(coordinates-point.coordinates) <=
                       self.options.component_distance and not any(
                       other.component == point.component and other.branch == branch and
                       np.linalg.norm(key-other.key) < 1e-12 for other in self._points)]
            component = (min(matched, key=lambda p: np.linalg.norm(coordinates-p.coordinates)).component
                         if matched else 1+max((p.component for p in self._points), default=0))
        point = AtlasPoint(branch, component, tuple(key), tuple(coordinates))
        if not any(p.branch == branch and p.component == component and
                   np.linalg.norm(key-p.key) < 1e-12 and np.linalg.norm(coordinates-p.coordinates) < 1e-6
                   for p in self._points):
            self._points.append(point)

    def save(self, path):
        """Write the same versioned text format as the Fortran atlas."""
        lines = [f"SHEATH_EQUILIBRIUM_ATLAS 1 {len(self._points)}"]
        for point in self._points:
            values = (*point.key, *point.coordinates)
            lines.append(f"{point.branch} {point.component} {len(point.spectrum_shape)} "+
                         " ".join(format(value, ".17e") for value in values))
            if point.spectrum_shape:
                lines.append(" ".join(format(value, ".17e") for value in point.spectrum_shape))
        Path(path).write_text("\n".join(lines)+"\n", encoding="ascii")

    @classmethod
    def load(cls, path, *, options=None):
        lines = Path(path).read_text(encoding="ascii").splitlines()
        if not lines:
            raise ValueError("empty atlas")
        header = lines[0].split()
        if len(header) != 3 or header[:2] != ["SHEATH_EQUILIBRIUM_ATLAS", "1"]:
            raise ValueError("unsupported atlas format")
        count = int(header[2])
        if count < 0:
            raise ValueError("invalid atlas point count")
        points = []
        position = 1
        for _ in range(count):
            if position >= len(lines):
                raise ValueError("invalid atlas point count")
            fields = lines[position].split()
            position += 1
            if len(fields) != 12:
                raise ValueError("incomplete atlas point")
            shape_count = int(fields[2])
            if shape_count < 0:
                raise ValueError("invalid spectrum coordinate count")
            values = tuple(float(value) for value in fields[3:])
            shape = ()
            if shape_count:
                if position >= len(lines):
                    raise ValueError("missing spectrum shape")
                shape = tuple(float(value) for value in lines[position].split())
                position += 1
                if len(shape) != shape_count:
                    raise ValueError("invalid spectrum coordinate count")
            points.append(AtlasPoint(fields[0], int(fields[1]), values[:6], values[6:9], shape))
        if position != len(lines):
            raise ValueError("unexpected atlas records")
        return cls(points, options=options)
