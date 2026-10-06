from __future__ import annotations

import matplotlib.pyplot as plt

from sheath_model import ZhaoParams, SheathSolver, ProfileOptions


def main() -> None:
    # Dayside
    params = ZhaoParams(
        electron_drift_mode="zero",
        ion_density_m3=(5)*1e6,
        electron_temperature_ev=10,
        photoelectron_temperature_ev=2.2,
        solar_wind_speed_mps=400e3,
        sun_elevation_deg=90.0)
    solver = SheathSolver(profile=ProfileOptions(zmax_hat=120.))

    branches = ["A", "B"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for branch in branches:
        out = solver.solve_profile(params, branch=branch)
        ax.plot(out.potential_v, out.z_hat, label=f"Type {branch}")

    # Low Sun elevation / Type C
    params_c = ZhaoParams(
        electron_drift_mode="zero",
        ion_density_m3=(5)*1e6,
        electron_temperature_ev=10,
        photoelectron_temperature_ev=2.2,
        solar_wind_speed_mps=400e3,
        sun_elevation_deg=5.0)
    out_c = solver.solve_profile(params_c, branch="C")
    ax.plot(out_c.potential_v, out_c.z_hat, label="Type C (alpha=5 deg)")

    ax.set_xlabel(r"$\phi$ [V]")
    ax.set_ylabel(r"$\hat z = z/\lambda_D$")
    ax.legend()
    ax.set_title("Zhao sheath profiles")
    fig.tight_layout()

    fig.savefig('profiles.png')
    # plt.show()

if __name__ == "__main__":
    main()
