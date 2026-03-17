from __future__ import annotations

import matplotlib.pyplot as plt

from zhao_sheath import ZhaoParams, ZhaoSheathSolver


def main() -> None:
    params = ZhaoParams(alpha_deg=60.0, zmax_hat=120.0)
    solver = ZhaoSheathSolver(params)

    branches = ["A", "B"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for branch in branches:
        out = solver.solve_profile(branch)
        ax.plot(out["phi_V"], out["z_hat"], label=f"Type {branch}")

    params_c = ZhaoParams(alpha_deg=10.0, zmax_hat=120.0)
    out_c = ZhaoSheathSolver(params_c).solve_profile("C")
    ax.plot(out_c["phi_V"], out_c["z_hat"], label="Type C")

    ax.set_xlabel(r"$\phi$ [V]")
    ax.set_ylabel(r"$\hat z = z/\lambda_D$")
    ax.legend()
    ax.set_title("Zhao sheath profiles")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
