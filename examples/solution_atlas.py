"""Build a reusable root table, then correct a query from its nearby solutions."""
from dataclasses import replace
from pathlib import Path
import sys

from sheath_model import (FixedEntryParams, SheathSolver, EquilibriumAtlas,
                          SearchOptions, ContinuationOptions)


def main():
    base = FixedEntryParams(photoelectron_density_m3=55.42562584220407e6,
                            ion_temperature_ev=12., ion_pressure_factor=3.)
    inputs = [replace(base, photoelectron_density_m3=density) for density in (54e6, 57e6)]
    # Use method="arclength" to follow simple folds along the input path.
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is not None and path.exists():
        atlas = EquilibriumAtlas.load(path)
    else:
        atlas = SheathSolver().build_equilibrium_atlas(inputs, branches=("A",))
    if path is not None and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(path)
    solver = SheathSolver(search=SearchOptions(method="newton", max_starts=1),
                          continuation=ContinuationOptions(method="parameter"), equilibrium_atlas=atlas)
    root = solver.solve_equilibrium(base, branch="A")
    print(f"Stored roots: {len(atlas.points)}")
    print(f"Surface / minimum potential [V]: {root.surface_potential_v:.8f} / {root.minimum_potential_v:.8f}")
    print(f"Starts using the atlas: {root.diagnostics.atlas_starts[0]}")
    # Enumerate additional located roots with deflation; this does not prove completeness.
    found = solver.solve_equilibrium_candidates(base, branch="A", deflation=True)
    print(f"Located admissible candidates: {len(found.candidates)}")


if __name__ == "__main__":
    main()
