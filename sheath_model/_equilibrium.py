"""Per-call J=0 search, scalar reduction, map seeds and continuation."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from ._ions import ion_critical_potential

from .params import FixedEntryParams, ZhaoParams
from .search import SearchOptions, SearchDiagnostics, SearchFailure, solve_guarded_system
from .continuation import continue_guarded_system, find_guarded_roots

Branch = Literal["A", "B", "C"]


from ._physics import SheathPhysics
from .continuation import ContinuationOptions
from .results import CandidateSet, EquilibriumResult


class EquilibriumProblem(SheathPhysics):
    """A per-call equation/search context, never retained by the public solver."""
    def __init__(self, params, *, search=None, continuation=None):
        super().__init__(params)
        self.search = search if search is not None else SearchOptions()
        self.continuation = continuation if continuation is not None else ContinuationOptions()
        if not isinstance(self.search, SearchOptions) or not isinstance(self.continuation, ContinuationOptions):
            raise TypeError("invalid search/continuation options")

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
        y = self._encode_unknowns(branch, physical)
        norm = float(np.max(np.abs(self._encoded_residual(branch, y, self.search))))
        return EquilibriumResult(self.p, branch, *map(float, physical), norm, diagnostics)

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
                return EquilibriumProblem(params_from_key(start_key+t*(target_key-start_key), self.p), search=options, continuation=self.continuation)

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

            y, success = continue_guarded_system(residual, point.coordinates[:n], options, self.continuation,
                                                  diagnostics, k, accept=accept)
            if success:
                return self._decode_unknowns(branch, y, options)
        return None


    def solve_branch(self, branch: Branch, guess: tuple[float, ...] | None = None,
                     *, atlas=None) -> EquilibriumResult:
        """Find the first physically connecting root; attach search_diagnostics.

        guess is (surface V, minimum V, density m^-3) for A, or (surface V,
        density m^-3) for B/C. It supplements independent starts. To run only
        a continuation seed, set use_default_guesses=False and method=newton/lm.
        """
        p = self.p
        self._validate_params_for_branch(branch)
        options = self.search
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


    def find_branch_candidates(self, branch: Branch, *, atlas=None, deflation=True, max_roots=16) -> CandidateSet:
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
            return CandidateSet((), diagnostics)
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
        return CandidateSet(tuple(candidates), diagnostics)


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
                   p.photoelectron_temperature_ev*p.electron_mass_kg/p.ion_mass_kg)*p.mach

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
