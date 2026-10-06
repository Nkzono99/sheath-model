"""Numerical orchestration for independent fixed-entry and Zhao sheath inputs."""
from dataclasses import dataclass, replace
import numpy as np
from .params import FixedEntryParams, ZhaoParams
from .search import SearchOptions, SearchDiagnostics, SearchFailure
from .continuation import ContinuationOptions
from .atlas import EquilibriumAtlas, AtlasOptions
from .results import EquilibriumResult, CandidateSet, ProfileOptions, SheathProfile
from ._equilibrium import EquilibriumProblem
from ._profile import ProfilePhysics


@dataclass(frozen=True)
class SheathSolver:
    """Reusable configuration, with physical conditions passed to each call.

    Queries read the atlas without changing it or retaining the last root.
    Python supports Maxwell-source J=0; Fortran also supports prescribed fields.
    """
    search: SearchOptions = SearchOptions()
    profile: ProfileOptions = ProfileOptions()
    continuation: ContinuationOptions = ContinuationOptions()
    equilibrium_atlas: EquilibriumAtlas | None = None

    def __post_init__(self):
        for name, kind in (('search', SearchOptions), ('profile', ProfileOptions),
                           ('continuation', ContinuationOptions), ('equilibrium_atlas', EquilibriumAtlas)):
            value = getattr(self, name)
            if name == 'equilibrium_atlas' and value is None:
                continue
            if not isinstance(value, kind):
                raise TypeError(f'{name} must be {kind.__name__}')

    def _problem(self, inputs):
        return EquilibriumProblem(inputs, search=self.search, continuation=self.continuation)

    def _branches(self, inputs, branch):
        if branch in {'A', 'B', 'C'}:
            return (branch,)
        if branch != 'auto':
            raise ValueError('branch must be A, B, C, or auto')
        if self.search.method == 'bracket':
            raise ValueError('bracket requires an explicit B/C branch')
        return ('C', 'A', 'B') if isinstance(inputs, ZhaoParams) and inputs.sun_elevation_deg < 20. else ('A', 'B', 'C')

    def solve_equilibrium(self, inputs: FixedEntryParams | ZhaoParams, *, branch='auto',
                          initial_guess: EquilibriumResult | None = None) -> EquilibriumResult:
        """First connecting root; an explicit previous result supplements starts."""
        problem = self._problem(inputs)
        diagnostics = SearchDiagnostics()
        errors = []
        if initial_guess is not None and not isinstance(initial_guess, EquilibriumResult):
            raise TypeError('initial_guess must be EquilibriumResult')
        for kind in self._branches(inputs, branch):
            guess = None
            if initial_guess is not None and initial_guess.branch == kind:
                guess = ((initial_guess.surface_potential_v, initial_guess.minimum_potential_v,
                          initial_guess.ambient_electron_density_m3) if kind == 'A' else
                         (initial_guess.surface_potential_v, initial_guess.ambient_electron_density_m3))
            try:
                result = problem.solve_branch(kind, guess, atlas=self.equilibrium_atlas)
                diagnostics.include(result.diagnostics)
                return replace(result, diagnostics=diagnostics)
            except SearchFailure as exc:
                diagnostics.include(exc.diagnostics)
                errors.append(f'{kind}: {exc}')
        raise SearchFailure('finite search found no admissible root: '+' | '.join(errors), diagnostics)

    def solve_equilibrium_candidates(self, inputs: FixedEntryParams | ZhaoParams, *, branch='auto',
                                     deflation=True, max_roots=16) -> CandidateSet:
        """Located admissible roots; finite per-branch budgets do not prove completeness."""
        problem = self._problem(inputs)
        diagnostics = SearchDiagnostics()
        candidates = []
        for kind in self._branches(inputs, branch):
            found = problem.find_branch_candidates(kind, atlas=self.equilibrium_atlas,
                                             deflation=deflation, max_roots=max_roots)
            diagnostics.include(found.diagnostics)
            candidates.extend(found.candidates)
        return CandidateSet(tuple(replace(root, diagnostics=diagnostics) for root in candidates), diagnostics)

    def build_profile(self, equilibrium: EquilibriumResult) -> SheathProfile:
        """Reconstruct an existing root without running another root search."""
        if not isinstance(equilibrium, EquilibriumResult):
            raise TypeError('equilibrium must be EquilibriumResult')
        problem = self._problem(equilibrium.inputs)
        data = equilibrium._kernel_data()
        y = problem._encode_unknowns(equilibrium.branch,
            [equilibrium.surface_potential_v, equilibrium.minimum_potential_v, equilibrium.ambient_electron_density_m3])
        raw = None if y is None else problem._encoded_residual(equilibrium.branch, y, self.search)
        if raw is None or np.max(np.abs(raw)) > self.search.residual_tolerance:
            raise ValueError('result does not satisfy the original equations')
        physics = ProfilePhysics(equilibrium.inputs, self.profile)
        physics._validate_profile_root(equilibrium.branch, data['phi0_hat'], data['phi_m_hat'], data['n_swe_inf_hat'])
        return SheathProfile(equilibrium, physics.build_profile(equilibrium), physics)

    def solve_profile(self, inputs: FixedEntryParams | ZhaoParams, *, branch='auto',
                      initial_guess: EquilibriumResult | None = None) -> SheathProfile:
        return self.build_profile(self.solve_equilibrium(inputs, branch=branch, initial_guess=initial_guess))

    def build_equilibrium_atlas(self, parameters, *, branches=('A', 'B', 'C'), options=None,
                                deflation=False) -> EquilibriumAtlas:
        """Build a new map, retrying holes using later neighbors; attached map is unchanged."""
        atlas = EquilibriumAtlas(options=options if options is not None else AtlasOptions())
        worker = replace(self, equilibrium_atlas=atlas)
        if not isinstance(deflation, bool):
            raise ValueError('deflation must be a bool')
        branches = tuple(branches)
        if not branches or any(branch not in {'A', 'B', 'C'} for branch in branches):
            raise ValueError('branches must contain A/B/C')
        inputs = tuple(parameters)
        pending = [(index, branch) for index in range(len(inputs)) for branch in branches]
        records = {}
        for _ in range(len(pending)+1):
            size_before = len(atlas.points)
            failed = []
            for index, branch in pending:
                try:
                    if deflation:
                        found = worker.solve_equilibrium_candidates(inputs[index], branch=branch)
                        if not found.candidates:
                            raise SearchFailure('finite search found no admissible root', found.diagnostics)
                        results, diagnostics = found.candidates, found.diagnostics
                    else:
                        result = worker.solve_equilibrium(inputs[index], branch=branch)
                        results, diagnostics = (result,), result.diagnostics
                    for result in results:
                        atlas.add(result, search=self.search)
                    status = 'accepted'
                except SearchFailure as exc:
                    diagnostics = exc.diagnostics
                    status = 'excluded' if diagnostics.excluded['ABC'.index(branch)] else 'unresolved'
                    if status == 'unresolved':
                        failed.append((index, branch))
                if (index, branch) in records:
                    diagnostics.include(records[index, branch]['diagnostics'])
                records[index, branch] = dict(index=index, branch=branch, status=status, diagnostics=diagnostics)
            if not failed or len(atlas.points) == size_before:
                break
            pending = failed
        atlas.attempts = [records[index, branch] for index in range(len(inputs)) for branch in branches]
        return atlas
