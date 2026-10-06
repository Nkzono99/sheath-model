from __future__ import annotations

import argparse

from .params import ZhaoParams
from .solver import SheathSolver
from .results import ProfileOptions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve Zhao et al. lunar photoelectron sheath model (Type A/B/C)."
    )
    parser.add_argument("--branch", choices=["A", "B", "C", "auto"], default="auto")
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--zmax-hat", type=float, default=80.0)
    parser.add_argument("--electron-drift-mode", choices=["full", "normal", "zero"], default="full")
    parser.add_argument("--ion-drift-mode", choices=["full", "normal"], default="full")
    args = parser.parse_args()

    prm = ZhaoParams(
        sun_elevation_deg=args.alpha,
        electron_drift_mode=args.electron_drift_mode,
        ion_drift_mode=args.ion_drift_mode,
    )
    solver = SheathSolver(profile=ProfileOptions(zmax_hat=args.zmax_hat))
    profile = solver.solve_profile(prm, branch=args.branch)
    out = profile.equilibrium

    print(f"=== solved branch {out.branch} ===")
    print(f"alpha                = {prm.sun_elevation_deg:.1f} deg")
    print(f"phi0                 = {out.surface_potential_v:.6f} V")
    print(f"phi_m                = {out.minimum_potential_v:.6f} V")
    print(f"n_swe_inf            = {out.ambient_electron_density_m3:.6e} m^-3")
    print(f"turning height       = {profile.turning_height_m:.6f} m")
    print(f"electron drift mode  = {prm.electron_drift_mode}")
    print(f"ion drift mode       = {prm.ion_drift_mode}")


if __name__ == "__main__":
    main()
