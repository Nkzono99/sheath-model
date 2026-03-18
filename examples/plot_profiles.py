from __future__ import annotations

import matplotlib.pyplot as plt

from sheath_model import ZhaoParams, ZhaoSheathSolver


def main() -> None:
    # Dayside
    params = ZhaoParams(
        n_swi_inf_cm3=5,
        T_swe_eV=10,
        T_phe_eV=2.2,
        v_sw_total_mps=400e3,
        alpha_deg=90.0,
        zmax_hat=120.0)
    solver = ZhaoSheathSolver(params)

    branches = ["A", "B"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for branch in branches:
        out = solver.solve_profile(branch)
        ax.plot(out["phi_V"], out["z_hat"], label=f"Type {branch}")

    # Low Sun elevation / Type C
    params_c = ZhaoParams(
        n_swi_inf_cm3=5,
        T_swe_eV=10,
        T_phe_eV=2.2,
        v_sw_total_mps=400e3,
        alpha_deg=5.0,
        zmax_hat=120.0)
    out_c = ZhaoSheathSolver(params_c).solve_profile("C")
    ax.plot(out_c["phi_V"], out_c["z_hat"], label="Type C (alpha=5 deg)")

    ax.set_xlabel(r"$\phi$ [V]")
    ax.set_ylabel(r"$\hat z = z/\lambda_D$")
    ax.legend()
    ax.set_title("Zhao sheath profiles")
    fig.tight_layout()
    
    fig.savefig('profiles.png')
    # plt.show()

if __name__ == "__main__":
    main()
