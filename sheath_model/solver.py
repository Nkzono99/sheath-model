"""One-dimensional sheath model with orbit-mapped background electrons.

Zhao A/B/C population topology, ion fluid transport and a Maxwell photoelectron source.
See docs/kinetic-model.md for the reservoir closure and admissibility conditions.
"""

from __future__ import annotations

import math
from typing import Dict, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf, erfc
from ._orbits import electron_density, POTENTIAL_NODES, POTENTIAL_WEIGHTS
from ._ions import ion_density_ratio, ion_critical_potential

from ._constants import QE, ME
from .params import FixedEntryParams, ZhaoParams
from .search import SearchOptions, SearchDiagnostics, SearchFailure, solve_guarded_system
from .continuation import continue_guarded_system, find_guarded_roots

Branch = Literal["A", "B", "C"]
TypeASide = Literal["lower", "upper"]
Species = Literal["swi", "swe", "phe", "all"]
ZUnit = Literal["hat", "m"]


class _SheathSolver:
    def __init__(self, params: FixedEntryParams | ZhaoParams, *, search: SearchOptions | None = None):
        self.p = params
        self.search = search if search is not None else SearchOptions()
        if not isinstance(self.search, SearchOptions):
            raise TypeError("search must be SearchOptions")
        if isinstance(params, ZhaoParams):
            if not math.isfinite(params.alpha_deg) or not 0 <= params.alpha_deg <= 90:
                raise ValueError("alpha_deg must be finite and in [0, 90]")
            if params.electron_drift_mode not in {"zero", "normal", "full"}:
                raise ValueError("electron_drift_mode must be zero, normal, or full")
            if params.ion_drift_mode not in {"normal", "full"}:
                raise ValueError("ion_drift_mode must be normal or full")
        for name in ("ion_density_m3", "density_scale_m3", "electron_temperature_ev",
                     "photoelectron_temperature_ev", "ion_mass_kg", "ion_pressure_factor"):
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

    # ------------------------------------------------------------------
    # Algebraic unknown solver
    # ------------------------------------------------------------------
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
            * math.sqrt(2.0 * math.pi * p.electron_temperature_ev / p.photoelectron_temperature_ev * ME / p.ion_mass_kg)
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
            * math.sqrt(2.0 * math.pi * p.electron_temperature_ev / p.photoelectron_temperature_ev * ME / p.ion_mass_kg)
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
            * math.sqrt(2.0 * math.pi * p.electron_temperature_ev / p.photoelectron_temperature_ev * ME / p.ion_mass_kg)
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

    def _encode_unknowns(self, branch, physical):
        phi0, phim, density = physical
        scale = self.p.photoelectron_temperature_ev
        if not np.all(np.isfinite(physical)) or density <= 0:
            return None
        if branch == "A":
            if phim >= min(phi0, 0.):
                return None
            return np.log([(phi0-phim)/scale, -phim/scale, density/self.p.ion_density_m3])
        if (branch == "B" and phi0 <= 0) or (branch == "C" and phi0 >= 0):
            return None
        return np.log([abs(phi0)/scale, density/self.p.ion_density_m3])

    def _decode_unknowns(self, branch, y, options):
        if not np.all(np.isfinite(y)) or min(y) < -50 or max(y) > 700:
            return None
        if y[-1] < -30 or y[-1] > math.log(1e6) or y[0] > math.log(options.potential_extent):
            return None
        p = self.p
        if branch == "A":
            if y[1] > math.log(options.potential_extent):
                return None
            phim = -p.photoelectron_temperature_ev*math.exp(y[1])
            phi0 = phim + p.photoelectron_temperature_ev*math.exp(y[0])
        else:
            phi0 = (1 if branch == "B" else -1)*p.photoelectron_temperature_ev*math.exp(y[0])
            phim = 0. if branch == "B" else phi0
        density = p.ion_density_m3*math.exp(y[-1])
        if not math.isfinite(density) or density <= 0:
            return None
        try:
            self._ion_density_hat(max(phi0, 0)/p.photoelectron_temperature_ev)
        except ValueError:
            return None
        return np.array([phi0, phim, density])

    def _default_guesses(self, branch, options):
        p = self.p
        source_shift = math.log(max(1., .5*p.photoelectron_density_m3/p.ion_density_m3))
        limit = ion_critical_potential(.5*p.electron_temperature_ev*p.mach**2,
                                      p.ion_pressure_factor*p.ion_temperature_ev)/p.photoelectron_temperature_ev
        potentials = []
        if branch == "A":
            for gap, depth in zip((2., 1., .5, 3., 1., 1., 1., .02), (.2, .05, .5, .8, 1., 2., 4., .5)):
                potentials.extend(((gap-depth, -depth), (gap+source_shift-depth, -depth)))
        else:
            for voltage in (.002, .02, .2, .6, 1.5, 4., 12., 50.):
                potentials.extend(((voltage, 0.), (source_shift+voltage*1e-10, 0.)) if branch == "B" else
                                  ((-voltage, -voltage), (-min(180., voltage*max(1e-10, source_shift)),)*2))
        guesses = []
        for surface, minimum in potentials:
            surface = min(surface, .8*limit, .9*options.potential_extent)
            if branch == "A":
                minimum = max(-.9*options.potential_extent, min(minimum, surface-1e-6))
            elif branch == "C":
                surface = max(-.9*options.potential_extent, surface)
                minimum = surface
            else:
                minimum = 0.
            phi0, phim = np.array([surface, minimum])*p.photoelectron_temperature_ev
            # The neutrality equation is affine in electron normalization.
            func = getattr(self, "_residuals_type_"+branch.lower())
            x0 = [phi0, phim, p.ion_density_m3] if branch == "A" else [phi0, p.ion_density_m3]
            x1 = [phi0, phim, 2*p.ion_density_m3] if branch == "A" else [phi0, 2*p.ion_density_m3]
            f0, f1 = func(np.array(x0))[0], func(np.array(x1))[0]
            if not math.isfinite(f1-f0) or f1 == f0:
                continue
            density = p.ion_density_m3*(1-f0/(f1-f0))
            density = max(.1*p.ion_density_m3, min(1e5*p.ion_density_m3, density))
            encoded = self._encode_unknowns(branch, [phi0, phim, density])
            if encoded is not None and not any(np.max(np.abs(encoded-other)) < 1e-10 for other in guesses):
                guesses.append(encoded)
        return guesses

    def _encoded_residual(self, branch, y, options):
        physical = self._decode_unknowns(branch, y, options)
        if physical is None:
            return None
        raw = getattr(self, "_residuals_type_"+branch.lower())(physical if branch == "A" else physical[[0, 2]])
        raw[:2] /= self.p.ion_density_m3
        if branch == "A":
            raw[2] *= self.p.density_scale_m3/self.p.ion_density_m3/(-physical[1]/
                       self.p.photoelectron_temperature_ev)**1.5
        return raw if np.all(np.isfinite(raw)) else None

    def _unknown_result(self, branch, physical, diagnostics):
        p = self.p
        phi0_V, phi_m_V, n_swe_inf_m3 = physical
        return {"branch": branch, "phi0_V": float(phi0_V), "phi_m_V": float(phi_m_V),
                "n_swe_inf_m3": float(n_swe_inf_m3),
                "phi0_hat": float(phi0_V/p.photoelectron_temperature_ev),
                "phi_m_hat": float(phi_m_V/p.photoelectron_temperature_ev),
                "n_swe_inf_hat": float(n_swe_inf_m3/p.density_scale_m3),
                "electron_drift_mps": p.electron_drift_mps,
                "ion_entry_speed_mps": p.ion_entry_speed_mps,
                "search_diagnostics": diagnostics}

    def _continue_from_atlas(self, branch, atlas, options, diagnostics):
        from .atlas import parameter_key, params_from_key
        target_key = parameter_key(self.p)
        n = 3 if branch == "A" else 2
        k = "ABC".index(branch)
        for point in atlas.neighbors(self.p, branch):
            start_key = np.array(point.key)

            def path_solver(t):
                if t == 1.:
                    return self
                return _SheathSolver(params_from_key(start_key+t*(target_key-start_key), self.p), search=options)

            def residual(y, t):
                try:
                    return path_solver(t)._encoded_residual(branch, y, options)
                except (ValueError, OverflowError):
                    return None

            def accept(y, t):
                try:
                    solver = path_solver(t)
                    physical = solver._decode_unknowns(branch, y, options)
                    if physical is None:
                        return False
                    solver._validate_profile_root(branch, physical[0]/solver.p.photoelectron_temperature_ev,
                                                  physical[1]/solver.p.photoelectron_temperature_ev,
                                                  physical[2]/solver.p.density_scale_m3)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                    return False
                return True

            y, success = continue_guarded_system(residual, point.coordinates[:n], options, atlas.continuation,
                                                  diagnostics, k, accept=accept)
            if success:
                return self._decode_unknowns(branch, y, options)
        return None

    def solve_unknowns(self, branch: Branch, guess: tuple[float, ...] | None = None,
                       *, search: SearchOptions | None = None, atlas=None) -> dict[str, float | str | SearchDiagnostics]:
        """Find the first physically connecting root; attach search_diagnostics.

        guess is (surface V, minimum V, density m^-3) for A, or (surface V,
        density m^-3) for B/C. It supplements independent starts. To run only
        a continuation seed, set use_default_guesses=False and method=newton/lm.
        """
        p = self.p
        self._validate_params_for_branch(branch)
        options = self.search if search is None else search
        if not isinstance(options, SearchOptions):
            raise TypeError("search must be SearchOptions")
        if options.method == "bracket" and branch == "A":
            raise ValueError("bracket supports only B/C J=0 branches")
        diagnostics = SearchDiagnostics()
        k = "ABC".index(branch)
        diagnostics.searched[k] = True
        if branch in ("A", "C") and p.u > 0:
            diagnostics.excluded[k] = True
            raise SearchFailure("reflected drifting electrons have no neutral semi-infinite profile", diagnostics)
        if atlas is not None:
            from .atlas import EquilibriumAtlas
            if not isinstance(atlas, EquilibriumAtlas):
                raise TypeError("atlas must be EquilibriumAtlas")

        def residual(y):
            return self._encoded_residual(branch, y, options)

        def accept(physical):
            try:
                self._validate_profile_root(branch, physical[0]/p.photoelectron_temperature_ev,
                                            physical[1]/p.photoelectron_temperature_ev,
                                            physical[2]/p.density_scale_m3)
            except FloatingPointError:
                diagnostics.profile_failures[k] += 1
                return False
            except RuntimeError:
                diagnostics.rejected[k] += 1
                return False
            diagnostics.roots_found[k] += 1
            return True

        def multivariate(starts, from_atlas=False):
            for encoded in starts:
                if diagnostics.starts[k] >= options.max_starts:
                    break
                diagnostics.starts[k] += 1
                if from_atlas:
                    diagnostics.atlas_starts[k] += 1
                y, norm, success = solve_guarded_system(residual, encoded, options, diagnostics, k)
                if not success:
                    diagnostics.unconverged[k] += 1
                    continue
                physical = self._decode_unknowns(branch, y, options)
                if physical is not None and accept(physical):
                    if from_atlas:
                        diagnostics.atlas_hits[k] += 1
                    return physical
            return None

        seeds = []
        if guess is not None:
            if len(guess) != (3 if branch == "A" else 2):
                raise ValueError("guess has wrong number of unknowns")
            physical = guess if branch == "A" else (guess[0], 0. if branch == "B" else guess[0], guess[1])
            encoded = self._encode_unknowns(branch, physical)
            if encoded is None:
                raise ValueError("guess must be finite and satisfy branch signs and positive density")
            seeds.append(encoded)
        found = multivariate(seeds) if options.method != "bracket" else None
        if found is None and atlas is not None and options.method != "bracket":
            n = 3 if branch == "A" else 2
            found = multivariate([prediction[:n] for prediction in atlas.predictions(p, branch)], from_atlas=True)
        if found is None and branch != "A" and options.method in ("auto", "bracket"):
            found = self._scalar_search(branch, options, diagnostics, accept)
        if found is None and options.method != "bracket" and options.use_default_guesses:
            found = multivariate(self._default_guesses(branch, options))
        if found is None and atlas is not None and options.method != "bracket":
            trial = self._continue_from_atlas(branch, atlas, options, diagnostics)
            if trial is not None and accept(trial):
                diagnostics.atlas_hits[k] += 1
                found = trial
        if found is None:
            raise SearchFailure(f"finite {options.method} search found no admissible {branch} root; "
                                f"best residual={diagnostics.best_residual[k]:.3e}, "
                                f"unconverged={diagnostics.unconverged[k]}, rejected={diagnostics.rejected[k]}", diagnostics)
        return self._unknown_result(branch, found, diagnostics)

    def solve_candidates(self, branch: Branch, *, atlas=None, deflation=True, max_roots=16):
        """Locate distinct admissible roots; finite search does not imply completeness."""
        self._validate_params_for_branch(branch)
        options = self.search
        diagnostics = SearchDiagnostics()
        k = "ABC".index(branch)
        diagnostics.searched[k] = True
        if not isinstance(max_roots, int) or isinstance(max_roots, bool) or max_roots < 1:
            raise ValueError("max_roots must be a positive integer")
        if not isinstance(deflation, bool):
            raise ValueError("deflation must be a bool")
        if atlas is not None:
            from .atlas import EquilibriumAtlas
            if not isinstance(atlas, EquilibriumAtlas):
                raise TypeError("atlas must be EquilibriumAtlas")
        if branch in {"A", "C"} and self.p.u > 0:
            diagnostics.excluded[k] = True
            return {"candidates": [], "search_diagnostics": diagnostics}
        roots = []
        n = 3 if branch == "A" else 2

        def accept(physical):
            encoded = self._encode_unknowns(branch, physical)
            raw = self._encoded_residual(branch, encoded, options)
            if raw is None or np.max(np.abs(raw)) > options.residual_tolerance:
                return False
            if any(np.linalg.norm(encoded-root) < 1e-6 for root in roots):
                return False
            try:
                self._validate_profile_root(branch, physical[0]/self.p.photoelectron_temperature_ev,
                                            physical[1]/self.p.photoelectron_temperature_ev,
                                            physical[2]/self.p.density_scale_m3)
            except FloatingPointError:
                diagnostics.profile_failures[k] += 1
                return False
            except RuntimeError:
                diagnostics.rejected[k] += 1
                return False
            roots.append(encoded)
            diagnostics.roots_found[k] += 1
            return len(roots) >= max_roots

        if branch != "A" and options.method in {"auto", "bracket"}:
            self._scalar_search(branch, options, diagnostics, accept)
        elif branch == "A" and options.method == "bracket":
            raise ValueError("bracket supports only B/C J=0 branches")
        if options.method != "bracket" and len(roots) < max_roots:
            starts = [] if atlas is None else [y[:n] for y in atlas.predictions(self.p, branch)]
            atlas_flags = [True]*len(starts)
            if options.use_default_guesses:
                defaults = self._default_guesses(branch, options)
                starts.extend(defaults)
                atlas_flags.extend([False]*len(defaults))
            origins = []
            algebraic = find_guarded_roots(lambda y: self._encoded_residual(branch, y, options), starts,
                                           options, diagnostics, k, max_roots=max_roots,
                                           deflation=deflation, known_roots=roots,
                                           atlas_flags=atlas_flags, origins=origins)
            for y, origin in zip(algebraic, origins):
                physical = self._decode_unknowns(branch, y, options)
                if physical is not None:
                    count_before = len(roots)
                    accept(physical)
                    if len(roots) > count_before and atlas_flags[origin]:
                        diagnostics.atlas_hits[k] += 1
            if not roots and atlas is not None:
                physical = self._continue_from_atlas(branch, atlas, options, diagnostics)
                if physical is not None:
                    accept(physical)
                    if roots:
                        diagnostics.atlas_hits[k] += 1
        candidates = [self._unknown_result(branch, self._decode_unknowns(branch, y, options), diagnostics) for y in roots]
        return {"candidates": candidates, "search_diagnostics": diagnostics}

    def _scalar_search(self, branch, options, diagnostics, accept):
        p = self.p
        k = "ABC".index(branch)
        limit = options.potential_extent*p.photoelectron_temperature_ev
        if branch == "B":
            limit = min(limit, np.nextafter(ion_critical_potential(.5*p.electron_temperature_ev*p.mach**2,
                               p.ion_pressure_factor*p.ion_temperature_ev), -math.inf))
        grid = np.unique(np.concatenate((limit*np.exp(np.linspace(-28., 0., options.bracket_points)),
                                        limit*np.arange(1, options.bracket_points+1)/options.bracket_points)))
        if branch == "C":
            grid = -grid
        func = self._residuals_type_b if branch == "B" else self._residuals_type_c
        ion_term = p.ion_density_m3*math.sqrt(2*math.pi*p.electron_temperature_ev/
                   p.photoelectron_temperature_ev*ME/p.ion_mass_kg)*p.mach

        def evaluate(phi):
            diagnostics.evaluations[k] += 1
            cutoff = -p.u if branch == "B" else math.sqrt(-phi/p.electron_temperature_ev)-p.u
            coefficient = self._swe_free_current_term(1., cutoff)
            source = p.photoelectron_density_m3*(math.exp(-phi/p.photoelectron_temperature_ev) if branch == "B" else 1.)
            if coefficient <= 0 or not math.isfinite(coefficient):
                return None
            density = (source+ion_term)/coefficient
            raw = func(np.array([phi, density]))/p.ion_density_m3
            if not np.all(np.isfinite(raw)) or density <= 0 or abs(raw[1]) > options.residual_tolerance:
                return None
            diagnostics.best_residual[k] = min(diagnostics.best_residual[k], float(np.max(np.abs(raw))))
            return float(raw[0]), density

        previous = None
        for phi in grid:
            value = evaluate(phi)
            if value is None:
                previous = None
                continue
            f, density = value
            located = abs(f) <= options.residual_tolerance
            trial_phi = phi
            if not located and previous is not None and np.signbit(f) != np.signbit(previous[1]):
                diagnostics.brackets[k] += 1
                lo, flo, hi = previous[0], previous[1], phi
                for _ in range(options.max_iterations):
                    mid = .5*lo+.5*hi
                    if mid == lo or mid == hi:
                        break
                    diagnostics.iterations[k] += 1
                    middle = evaluate(mid)
                    if middle is None:
                        break
                    fm, density = middle
                    if abs(fm) <= options.residual_tolerance:
                        trial_phi = mid
                        located = True
                        break
                    if np.signbit(flo) != np.signbit(fm):
                        hi = mid
                    else:
                        lo, flo = mid, fm
                if not located:
                    diagnostics.unconverged[k] += 1
            if located:
                physical = np.array([trial_phi, 0. if branch == "B" else trial_phi, density])
                if accept(physical):
                    return physical
            previous = (phi, f)
        return None

    def _validate_profile_root(self, branch, phi0, phim, density):
        if branch == "B" and phi0 > 0:
            ambient_edge = (
                density * self.p.density_scale_m3 * math.exp(-self.p.u**2)
                / math.sqrt(math.pi * self.p.electron_temperature_ev)
            )
            photo_edge = self.p.photoelectron_density_m3 * math.exp(-phi0) / math.sqrt(math.pi * self.p.photoelectron_temperature_ev)
            if ambient_edge - photo_edge > 128 * np.finfo(float).eps * max(abs(ambient_edge), abs(photo_edge)):
                raise RuntimeError("Type B has negative field squared arbitrarily near upstream infinity")
        if branch in ("A", "C") and self.p.u > 0:
            raise RuntimeError("algebraic root has no semi-infinite profile: inward drift with reflected slow "
                               "electrons makes E^2 negative near neutral infinity; specify zero electron drift "
                               "for the nondrifting model")
        try:
            self._ion_density_hat(max(phi0, 0))
        except ValueError as exc:
            raise RuntimeError("algebraic root blocks the upstream-connected ion flow") from exc
        segments = [("monotonic", phi0, 0.)] if branch != "A" else [("lower", phim, phi0), ("upper", phim, 0.)]
        values = []
        for side, lo, hi in segments:
            for phi in np.linspace(lo, hi, 129):
                e2 = (-2*self._integrate_rho(branch, side, phim, phi, phi0, phim, density) if branch == "A" else
                      2*self._integrate_rho(branch, side, phi, 0., phi0, phim, density))
                values.append(e2)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("profile field integration is non-finite")
        if min(values) < -1e-8*max(1., max(values)):
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
    ) -> Dict[str, np.ndarray | float | str | SearchDiagnostics]:
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

        out: Dict[str, np.ndarray | float | str | SearchDiagnostics] = {
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

    # ------------------------------------------------------------------
    # Public profile API
    # ------------------------------------------------------------------
    def solve_profile(
        self, branch: Branch, guess_unknowns: tuple[float, ...] | None = None, *, atlas=None
    ) -> Dict[str, np.ndarray | float | str | SearchDiagnostics]:
        p = self.p
        uk = self.solve_unknowns(branch, guess_unknowns, atlas=atlas)

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

        out: Dict[str, np.ndarray | float | str | SearchDiagnostics] = {
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

    def solve_auto(self, *, atlas=None) -> Dict[str, np.ndarray | float | str | SearchDiagnostics]:
        if self.search.method == "bracket":
            raise ValueError("bracket requires an explicit B/C branch")
        if isinstance(self.p, ZhaoParams) and self.p.alpha_deg < 20.0:
            order: list[Branch] = ["C", "A", "B"]
        else:
            order = ["A", "B", "C"]
        errs = []
        diagnostics = SearchDiagnostics()
        for br in order:
            try:
                profile = self.solve_profile(br, atlas=atlas)
                diagnostics.include(profile["search_diagnostics"])
                profile["search_diagnostics"] = diagnostics
                return profile
            except SearchFailure as exc:
                diagnostics.include(exc.diagnostics)
                errs.append(f"{br}: {exc}")
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                diagnostics.searched["ABC".index(br)] = True
                diagnostics.profile_failures["ABC".index(br)] += 1
                errs.append(f"{br}: {exc}")
        raise SearchFailure("auto branch selection failed: " + " | ".join(errs), diagnostics)

    # ------------------------------------------------------------------
    # Local diagnostics: densities, fluxes, reduced 1D VDFs
    # ------------------------------------------------------------------
    def _to_z_hat(self, z: float, unit: ZUnit) -> float:
        if unit == "hat":
            return float(z)
        if unit == "m":
            return float(z) / self.p.length_scale_m
        raise ValueError(f"unknown z unit: {unit}")

    @staticmethod
    def _interp_on_profile(profile: Dict[str, np.ndarray | float | str | SearchDiagnostics], key: str, z_hat: float) -> float:
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
        profile: Dict[str, np.ndarray | float | str | SearchDiagnostics],
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
    ) -> Dict[str, np.ndarray | float | str | SearchDiagnostics]:
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
        profile: Dict[str, np.ndarray | float | str | SearchDiagnostics],
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
        profile: Dict[str, np.ndarray | float | str | SearchDiagnostics],
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


class FixedEntrySheathSolver(_SheathSolver):
    """J=0 sheath, densities, profiles and local fluxes at a fixed entrance state.

    The incoming ion speed is used unchanged. Warm-ion pressure specifies
    fluid transport, without defining an ion velocity distribution.
    """

    def __init__(self, params: FixedEntryParams, *, search: SearchOptions | None = None):
        if not isinstance(params, FixedEntryParams):
            raise TypeError("FixedEntrySheathSolver requires FixedEntryParams")
        super().__init__(params, search=search)


class ZhaoSheathSolver(_SheathSolver):
    """Sheath with entrance state and source projected from solar illumination."""

    def __init__(self, params: ZhaoParams, *, search: SearchOptions | None = None):
        if not isinstance(params, ZhaoParams):
            raise TypeError("ZhaoSheathSolver requires ZhaoParams")
        super().__init__(params, search=search)
