"""Evaluate a warm-ion sheath with the entrance speed and emitted source fixed."""

from sheath_model import FixedEntryParams, SheathSolver, SearchOptions


def main():
    params = FixedEntryParams(
        ion_density_m3=8.7e6, ion_entry_speed_mps=405299.88897111727,
        ion_temperature_ev=12., ion_pressure_factor=3.,
        electron_temperature_ev=12., electron_drift_mps=0.,
        photoelectron_density_m3=55.42562584220407e6, photoelectron_temperature_ev=2.2,
    )
    # Select auto, newton or lm here; bracket is available for B/C.
    solver = SheathSolver(search=SearchOptions(method="newton"))
    profile = solver.solve_profile(params, branch="A")
    fluxes = profile.fluxes(0.)
    print(f"Surface potential [V]: {profile.equilibrium.surface_potential_v:.8f}")
    print(f"Minimum potential [V]: {profile.equilibrium.minimum_potential_v:.8f}")
    print(f"Surface field [V/m]: {profile.electric_field_v_m[0]:.6e}")
    print(f"Inward ion flux [m^-2 s^-1]: {-fluxes['Gamma_swi_signed_m2s']:.6e}")


if __name__ == "__main__":
    main()
