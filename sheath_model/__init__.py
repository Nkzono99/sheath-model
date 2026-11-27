"""Fixed-entry and solar-illumination sheath models above an electron-emitting boundary."""

from .params import FixedEntryParams, ZhaoParams
from .solver import SheathSolver
from .results import EquilibriumResult, CandidateSet, ProfileOptions, SheathProfile, DensityResult
from .search import SearchOptions, SearchDiagnostics, SearchFailure
from ._ions import ion_density_ratio, ion_critical_potential
from .atlas import EquilibriumAtlas, AtlasOptions, AtlasPoint
from .continuation import ContinuationOptions
from .branch import (
    BinnedEmission, BranchCriterion, EmissionSpectrum, FixedReservoir, KappaElectrons,
    MaxwellianElectrons, MaxwellianEmission, classify_branch, turning_field_difference,
)
from .finite_column import FiniteColumnSolution, solve_finite_column
from .presheath import OuterWellSeed, PresheathSolution, replacement_targets, solve_presheath

__all__ = ["FixedEntryParams", "ZhaoParams", "SheathSolver", "EquilibriumResult", "CandidateSet",
           "ProfileOptions", "SheathProfile", "DensityResult",
           "SearchOptions", "SearchDiagnostics", "SearchFailure",
           "ion_density_ratio", "ion_critical_potential", "EquilibriumAtlas", "AtlasOptions", "AtlasPoint",
           "ContinuationOptions", "FixedReservoir", "MaxwellianElectrons", "KappaElectrons",
           "EmissionSpectrum", "MaxwellianEmission", "BinnedEmission", "BranchCriterion", "classify_branch",
           "turning_field_difference", "FiniteColumnSolution", "solve_finite_column",
           "OuterWellSeed", "PresheathSolution", "replacement_targets", "solve_presheath"]
