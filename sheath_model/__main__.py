from __future__ import annotations

import argparse

from .solver import ZhaoParams, ZhaoSheathSolver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve Zhao et al. lunar photoelectron sheath model (Type A/B/C)."
    )
    parser.add_argument("--branch", choices=["A", "B", "C", "auto"], default="auto")
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--zmax-hat", type=float, default=80.0)
    parser.add_argument("--electron-drift-mode", choices=["full", "normal"], default="full")
    parser.add_argument("--ion-drift-mode", choices=["full", "normal"], default="full")
    args = parser.parse_args()

    prm = ZhaoParams(
        alpha_deg=args.alpha,
        zmax_hat=args.zmax_hat,
        electron_drift_mode=args.electron_drift_mode,
        ion_drift_mode=args.ion_drift_mode,
    )
    solver = ZhaoSheathSolver(prm)
    out = solver.solve_auto() if args.branch == "auto" else solver.solve_profile(args.branch)

    print(f"=== solved branch {out['branch']} ===")
    print(f"alpha                = {prm.alpha_deg:.1f} deg")
    print(f"phi0                 = {out['phi0_V']:.6f} V")
    print(f"phi_m                = {out['phi_m_V']:.6f} V")
    print(f"n_swe_inf            = {out['n_swe_inf_m3']:.6e} m^-3")
    print(f"z_m_hat              = {out['z_m_hat']:.6f}")
    print(f"electron drift mode  = {out['electron_drift_mode']}")
    print(f"ion drift mode       = {out['ion_drift_mode']}")
    if out.get("note"):
        print(f"note                 = {out['note']}")


if __name__ == "__main__":
    main()
